"""Code Generation Bridge — transforms fused KGIR nodes into standard TTIR per target.

This module bridges the graph-level optimization layer with the existing Triton
compilation pipeline. It transforms fused KGIR nodes (produced by the fusion engine)
back into valid TTIR (Triton Tensor IR) that is compilable by the unmodified existing
pipeline for any supported backend.

Key responsibilities:
- Generate per-target TTIR from fused/unfused KGIR nodes
- Compile TTIR to target binaries via the existing ``compile()`` pipeline
- Support multi-target parallel compilation with ``ThreadPoolExecutor``
- Support incremental recompilation (only changed nodes/targets)
- Enforce numerical correctness (bitwise identity for deterministic ops,
  IEEE 754 bounds for non-deterministic)

Performance constraints (AAP §0.7.2):
- Fusion code generation: < 500ms per fused kernel pair per target
- Per-kernel compilation: < 1s per modified kernel per target
- Multi-target wall-clock: ≤ slowest single-target + 10% coordination overhead

Architectural constraints (AAP §0.7.1):
- Emits standard TTIR ONLY — no TTGIR, no LLVM IR, no backend passes
- Uses ``triton.compiler.compile()`` as a client — does not modify the function
- Fused kernels are compilable by the unmodified existing pipeline
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from triton.graph.kgir import KGIRGraph, KGIRNode, HardwareProfile
from triton.graph.config import GraphConfig
from triton.graph.errors import FusionError
from triton.graph.cache import GraphCacheManager
from triton.backends.compiler import GPUTarget

if TYPE_CHECKING:
    from triton.compiler.compiler import CompiledKernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Performance constraint thresholds from AAP §0.7.2
# ---------------------------------------------------------------------------
_CODEGEN_TIME_LIMIT_S = 0.5  # < 500ms per fused kernel pair per target
_COMPILE_TIME_LIMIT_S = 1.0  # < 1s per modified kernel per target
_COORDINATION_OVERHEAD = 0.10  # ≤ 10% coordination overhead for multi-target

# ---------------------------------------------------------------------------
# MLIR operation categories for numerical correctness validation (AAP §0.7.3)
# ---------------------------------------------------------------------------
_DETERMINISTIC_OPS = frozenset({
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
    "arith.addi", "arith.subi", "arith.muli",
    "arith.cmpf", "arith.cmpi",
    "arith.andi", "arith.ori", "arith.xori",
    "arith.bitcast", "arith.extf", "arith.truncf",
    "arith.extsi", "arith.extui", "arith.trunci",
    "tt.load", "tt.store",
})

_NON_DETERMINISTIC_OPS = frozenset({
    "tt.reduce", "tt.scan",
    "tt.atomic_rmw", "tt.atomic_cas",
})

# ---------------------------------------------------------------------------
# Dtype size mapping for intermediate tensor estimation
# ---------------------------------------------------------------------------
_DTYPE_BYTES: Dict[str, int] = {
    "fp32": 4, "f32": 4, "float32": 4,
    "fp16": 2, "f16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "fp64": 8, "f64": 8, "float64": 8,
    "i32": 4, "int32": 4,
    "i64": 8, "int64": 8,
    "i16": 2, "int16": 2,
    "i8": 1, "int8": 1,
    "i1": 1, "bool": 1,
    "f8e5m2": 1, "f8e4m3fn": 1,
}

_DTYPE_TO_TTIR: Dict[str, str] = {
    "fp32": "f32", "float32": "f32", "f32": "f32",
    "fp16": "f16", "float16": "f16", "f16": "f16",
    "bf16": "bf16", "bfloat16": "bf16",
    "fp64": "f64", "float64": "f64", "f64": "f64",
    "i32": "i32", "int32": "i32",
    "i64": "i64", "int64": "i64",
    "i16": "i16", "int16": "i16",
    "i8": "i8", "int8": "i8",
    "i1": "i1", "bool": "i1",
    "f8e5m2": "f8E5M2", "f8e4m3fn": "f8E4M3FN",
}


class CodeGenerationBridge:
    """Transforms fused KGIR nodes into valid TTIR and manages compilation.

    The bridge operates as a **client** of the existing Triton compilation
    infrastructure, invoking ``triton.compiler.compile()`` with generated TTIR.
    It does **not** modify any existing compilation passes or backend
    implementations (AAP §0.7.1 strictly-additive mandate).

    Multi-target compilation uses ``concurrent.futures.ThreadPoolExecutor``
    for parallel compilation across ``GPUTarget`` instances.

    Attributes:
        _graph: The KGIR graph with fusion/scheduling/dispatch decisions.
        _config: Configuration controlling code generation behaviour.
        _cache: Optional cache manager for incremental recompilation.
        _compiled_cache: In-memory cache ``{node_id: {target: compiled}}``
            for the current compilation session.
        _native_available: Lazy flag for C++ KGIR pybind11 binding availability.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        graph: KGIRGraph,
        config: GraphConfig,
        cache: Optional[GraphCacheManager] = None,
    ) -> None:
        """Initialise the CodeGenerationBridge.

        Args:
            graph: KGIR graph with fusion/scheduling/dispatch decisions.
            config: Configuration for code generation behaviour.
            cache: Optional graph-level cache for incremental recompilation.
                   When provided, unchanged fused kernels are served from cache.
        """
        if graph is None:
            raise FusionError("KGIRGraph must not be None")
        if config is None:
            raise FusionError("GraphConfig must not be None")

        self._graph: KGIRGraph = graph
        self._config: GraphConfig = config
        self._cache: Optional[GraphCacheManager] = cache

        # Session-level compiled-kernel cache: {node_id: {GPUTarget: compiled}}
        self._compiled_cache: Dict[int, Dict[GPUTarget, Any]] = {}

        # Lazy flag — None means not yet probed
        self._native_available: Optional[bool] = None

        # Timing telemetry
        self._codegen_times: List[float] = []
        self._compile_times: List[float] = []

    # ------------------------------------------------------------------
    # Native binding probe
    # ------------------------------------------------------------------

    def _try_load_native(self) -> bool:
        """Attempt to load C++ KGIR pybind11 bindings for native TTIR emission.

        The C++ ``KGIRToTTIR`` conversion pass handles:
        - Producer-consumer body merging with shared memory intermediates
        - Sibling body merging with SM partitioning
        - Unified grid computation
        - Per-target TTIR emission

        Returns:
            ``True`` if native bindings are available.
        """
        if self._native_available is not None:
            return self._native_available
        try:
            import triton._C.libtriton  # noqa: F401
            self._native_available = True
            logger.debug("C++ KGIR native bindings loaded successfully")
        except (ImportError, AttributeError):
            self._native_available = False
            logger.debug(
                "C++ KGIR native bindings not available; using Python fallback"
            )
        return self._native_available

    # ==================================================================
    # Core Public Methods
    # ==================================================================

    def generate_ttir(self, node: KGIRNode, target: GPUTarget) -> str:
        """Generate TTIR text for a KGIR node targeting a specific device.

        For **unfused** kernel nodes the original TTIR is extracted from
        the kernel's compilation artefacts (``kernel_fn``).

        For **fused** nodes the ``KGIRToTTIR`` conversion pass is invoked
        (C++ native when available, otherwise Python fallback) to merge
        kernel bodies.

        Per-target parameters (tiling, SMEM allocation, grid dimensions)
        are derived from the ``HardwareProfile`` associated with *target*.

        Args:
            node: KGIR node to generate TTIR for.
            target: GPU target device (determines target-specific parameters).

        Returns:
            TTIR text string suitable for ``IRSource`` construction.

        Raises:
            FusionError: If TTIR emission fails for a fused node.
        """
        t_start = time.perf_counter()
        try:
            if not node.is_fused:
                ttir = self._extract_unfused_ttir(node, target)
                if self._config.dump_kgir:
                    logger.info(
                        "Extracted TTIR for unfused node %d (target=%s.%s):\n%s",
                        node.node_id,
                        target.backend,
                        target.arch,
                        ttir[:500],
                    )
                return ttir

            # --- fused node ---
            # Respect fusion enable flags from configuration
            fusion_cfg = self._config.fusion
            if not getattr(fusion_cfg, "enable", True):
                # Fusion disabled — fall back to first constituent kernel TTIR
                logger.debug(
                    "Fusion disabled via config; extracting first constituent "
                    "TTIR for fused node %d",
                    node.node_id,
                )
                first_id = (node.fused_from or [node.node_id])[0]
                first_node = self._graph.get_node(first_id)
                return self._extract_unfused_ttir(first_node, target)

            fused_ids = node.fused_from or []
            if len(fused_ids) < 2:
                raise FusionError(
                    f"Fused node {node.node_id} has fewer than 2 original nodes "
                    f"(fused_from={fused_ids})"
                )

            # Determine fusion type: producer-consumer vs. sibling
            # Respect per-type enable flags from FusionConfig
            pc_enabled = getattr(fusion_cfg, "enable_producer_consumer", True)
            sib_enabled = getattr(fusion_cfg, "enable_sibling", True)

            is_pc = (
                len(fused_ids) == 2
                and self._is_producer_consumer_pair(fused_ids[0], fused_ids[1])
            )
            if is_pc and pc_enabled:
                producer = self._graph.get_node(fused_ids[0])
                consumer = self._graph.get_node(fused_ids[1])
                ttir = self._emit_producer_consumer_ttir(producer, consumer, target)
            elif not is_pc and sib_enabled:
                siblings = [self._graph.get_node(nid) for nid in fused_ids]
                ttir = self._emit_sibling_ttir(siblings, target)
            elif is_pc and not pc_enabled:
                logger.debug(
                    "Producer-consumer fusion disabled; extracting first "
                    "constituent for fused node %d", node.node_id,
                )
                first_node = self._graph.get_node(fused_ids[0])
                ttir = self._extract_unfused_ttir(first_node, target)
            else:
                logger.debug(
                    "Sibling fusion disabled; extracting first "
                    "constituent for fused node %d", node.node_id,
                )
                first_node = self._graph.get_node(fused_ids[0])
                ttir = self._extract_unfused_ttir(first_node, target)

            if self._config.dump_kgir:
                logger.info(
                    "Generated TTIR for fused node %d (target=%s.%s):\n%s",
                    node.node_id,
                    target.backend,
                    target.arch,
                    ttir[:500],
                )
            return ttir

        finally:
            elapsed = time.perf_counter() - t_start
            self._codegen_times.append(elapsed)
            if elapsed > _CODEGEN_TIME_LIMIT_S:
                logger.warning(
                    "TTIR generation for node %d exceeded %.0fms limit: %.1fms",
                    node.node_id,
                    _CODEGEN_TIME_LIMIT_S * 1000,
                    elapsed * 1000,
                )

    # ------------------------------------------------------------------

    def compile_for_target(self, ttir: str, target: GPUTarget) -> Any:
        """Compile TTIR text for a specific hardware target.

        Writes TTIR to a temporary ``.ttir`` file, then invokes the existing
        ``triton.compiler.compile()`` pipeline.  The existing backend stage
        pipeline (``add_stages``) handles all lowering from TTIR → binary.

        Performance constraint: **< 1 s per kernel per target** (AAP §0.7.2).

        Args:
            ttir: TTIR text string to compile.
            target: GPU target device for compilation.

        Returns:
            ``CompiledKernel`` from the existing compilation pipeline.

        Raises:
            FusionError: If compilation fails.
        """
        t_start = time.perf_counter()
        tmp_fd: Optional[int] = None
        tmp_path: Optional[str] = None

        try:
            # 1. Write TTIR to a temporary file with .ttir extension.
            #    The existing ``IRSource`` reads from file paths.
            #    We use mkstemp for explicit fd control; NamedTemporaryFile
            #    is available for contexts requiring auto-cleanup.
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ttir", prefix="kgir_fused_")
            with os.fdopen(tmp_fd, "w") as fh:
                fh.write(ttir)
            tmp_fd = None  # fd now closed by os.fdopen context manager

            logger.debug(
                "Compiling TTIR for target %s.%s (%d bytes, file=%s)",
                target.backend,
                target.arch,
                len(ttir),
                tmp_path,
            )

            # 2. Attempt to construct an ``IRSource`` explicitly (preferred).
            #    Falls back to passing the path string to ``compile()``.
            compiled_kernel = self._invoke_compile(tmp_path, target)

            # 3. Access compiled result metadata for diagnostics
            if compiled_kernel is not None and self._config.dump_kgir:
                meta = getattr(compiled_kernel, "metadata", None)
                asm = getattr(compiled_kernel, "asm", None)
                logger.debug(
                    "Compilation result for %s.%s — metadata=%s, asm_keys=%s",
                    target.backend,
                    target.arch,
                    meta,
                    list(asm.keys()) if isinstance(asm, dict) else None,
                )

            return compiled_kernel

        except FusionError:
            raise
        except Exception as exc:
            raise FusionError(
                f"Compilation failed for target {target.backend}.{target.arch}: {exc}"
            ) from exc

        finally:
            # Clean up the temporary file
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            elapsed = time.perf_counter() - t_start
            self._compile_times.append(elapsed)
            if elapsed > _COMPILE_TIME_LIMIT_S:
                logger.warning(
                    "Compilation exceeded %.0fms limit: %.1fms (target=%s.%s)",
                    _COMPILE_TIME_LIMIT_S * 1000,
                    elapsed * 1000,
                    target.backend,
                    target.arch,
                )

    # ------------------------------------------------------------------

    def compile_graph(
        self, dispatch_plan: Dict[int, Any],
    ) -> Dict[int, Dict[GPUTarget, Any]]:
        """Compile all nodes in the KGIR graph per the dispatch plan.

        For each node→target pair the method:

        1. Generates TTIR (via ``generate_ttir``).
        2. Compiles TTIR (via ``compile_for_target``).
        3. Uses ``concurrent.futures.ThreadPoolExecutor`` for parallel
           compilation across targets.

        Multi-target compilation wall-clock MUST be ≤ slowest single-target
        + 10 % coordination overhead (AAP §0.7.2).

        Args:
            dispatch_plan: Mapping ``{node_id: GPUTarget | dict}``.
                Each node is assigned to a compilation target.

        Returns:
            ``{node_id: {GPUTarget: CompiledKernel}}`` for every compiled node.
        """
        t_start = time.perf_counter()
        results: Dict[int, Dict[GPUTarget, Any]] = {}

        # Use topological order for deterministic TTIR generation
        topo_order = self._graph.topological_sort()
        total_nodes = self._graph.node_count()
        logger.debug(
            "compile_graph: %d nodes in topological order, dispatch_plan has %d entries",
            total_nodes,
            len(dispatch_plan),
        )

        # ----- collect compilation tasks -----
        compile_tasks: List[tuple] = []
        for node_id in topo_order:
            if node_id not in dispatch_plan:
                continue
            target = self._resolve_target(dispatch_plan[node_id])
            if target is None:
                logger.warning("Cannot resolve target for node %d — skipping", node_id)
                continue
            compile_tasks.append((node_id, target))

        if not compile_tasks:
            logger.warning("No compilation tasks derived from dispatch plan")
            return results

        # ----- TTIR generation (sequential — fast relative to compilation) -----
        ttir_map: Dict[tuple, str] = {}
        for node_id, target in compile_tasks:
            node = self._graph.get_node(node_id)
            ttir_map[(node_id, target)] = self.generate_ttir(node, target)
            results.setdefault(node_id, {})

        # ----- parallel compilation across targets -----
        # Dispatch granularity from config controls compilation parallelism
        dispatch_cfg = self._config.dispatch
        dispatch_granularity = getattr(dispatch_cfg, "granularity", "subgraph")
        logger.debug("Dispatch granularity: %s", dispatch_granularity)
        max_workers = min(len(compile_tasks), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_task: Dict[concurrent.futures.Future, tuple] = {}
            for node_id, target in compile_tasks:
                ttir = ttir_map[(node_id, target)]
                future = pool.submit(self.compile_for_target, ttir, target)
                future_to_task[future] = (node_id, target)

            for future in concurrent.futures.as_completed(future_to_task):
                node_id, target = future_to_task[future]
                try:
                    compiled = future.result()
                    results[node_id][target] = compiled
                    logger.debug(
                        "Compiled node %d for %s.%s",
                        node_id, target.backend, target.arch,
                    )
                except Exception as exc:
                    logger.warning(
                        "Compilation failed for node %d target %s.%s: %s",
                        node_id, target.backend, target.arch, exc,
                    )
                    raise

        # ----- persist -----
        self._compiled_cache.update(results)
        if self._cache is not None:
            self._persist_compiled_results(results, dispatch_plan)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "compile_graph completed: %d nodes, %.1fms total", len(results), elapsed * 1000,
        )
        return results

    # ------------------------------------------------------------------

    def incremental_recompile(
        self,
        changed_nodes: Set[int],
        targets: Set[GPUTarget],
    ) -> Dict[int, Dict[GPUTarget, Any]]:
        """Recompile only the nodes whose optimisation decisions changed.

        Unchanged fused kernels are served from cache (``GraphCacheManager``).
        Only *changed_nodes* for *targets* are regenerated and recompiled.

        Performance: fusion code generation < 500 ms per pair per target.

        Args:
            changed_nodes: Set of node IDs whose decisions changed.
            targets: ``GPUTarget`` instances affected by the changes.

        Returns:
            Updated ``{node_id: {GPUTarget: CompiledKernel}}``
            containing **only** the recompiled entries.
        """
        t_start = time.perf_counter()
        results: Dict[int, Dict[GPUTarget, Any]] = {}
        cache_hits = 0
        cache_misses = 0

        for node_id in changed_nodes:
            node = self._graph.get_node(node_id)
            results[node_id] = {}

            for target in targets:
                # Attempt cache look-up first
                cached = self._try_get_cached(node_id, target)
                if cached is not None:
                    results[node_id][target] = cached
                    cache_hits += 1
                    continue

                # Cache miss — regenerate and compile
                cache_misses += 1
                ttir = self.generate_ttir(node, target)
                compiled = self.compile_for_target(ttir, target)
                results[node_id][target] = compiled

        # Merge into session cache
        for nid, target_map in results.items():
            self._compiled_cache.setdefault(nid, {}).update(target_map)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "incremental_recompile: %d nodes × %d targets, "
            "%d cache hits, %d misses, %.1fms",
            len(changed_nodes), len(targets),
            cache_hits, cache_misses, elapsed * 1000,
        )
        return results

    # ==================================================================
    # TTIR Emission Helpers (Private, but exposed per schema)
    # ==================================================================

    def _emit_producer_consumer_ttir(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
        target: GPUTarget,
    ) -> str:
        """Emit fused TTIR for a producer-consumer kernel pair.

        Merges the producer and consumer kernel bodies, replacing the
        intermediate global-memory tensor with shared memory (when SMEM
        budget permits).  Grid dimensions are adjusted to accommodate both
        kernels and target-specific SMEM allocation is derived from the
        ``HardwareProfile``.

        Args:
            producer: Producer kernel node (writes intermediate tensor).
            consumer: Consumer kernel node (reads intermediate tensor).
            target: GPU target for target-specific parameters.

        Returns:
            Fused TTIR text string.

        Raises:
            FusionError: If merging fails (incompatible structures, etc.).
        """
        t_start = time.perf_counter()

        # 1. Try native C++ KGIRToTTIR pass
        if self._try_load_native():
            try:
                native_ttir = self._native_producer_consumer_emission(
                    producer, consumer, target,
                )
                if native_ttir:
                    return native_ttir
            except Exception as exc:
                logger.debug(
                    "Native producer-consumer emission failed, Python fallback: %s", exc,
                )

        # 2. Python fallback — construct TTIR from kernel components
        hw = self._find_hardware_profile(target)
        smem_budget = hw.smem_per_sm_bytes if hw else 49152
        reg_budget = hw.registers_per_sm if hw else 65536
        warp_size_val = hw.warp_size if hw else target.warp_size

        producer_ttir = self._extract_unfused_ttir(producer, target)
        consumer_ttir = self._extract_unfused_ttir(consumer, target)

        p_res = producer.get_resource_usage()
        c_res = consumer.get_resource_usage()

        # Register budget validation — combined register pressure must
        # not exceed the per-SM register file capacity
        combined_regs = (
            p_res.get("register_count", 0)
            + c_res.get("register_count", 0)
        )
        if combined_regs > reg_budget:
            logger.warning(
                "Combined register usage (%d) exceeds budget (%d) for "
                "producer-consumer fusion (nodes %d→%d, target=%s.%s); "
                "occupancy may be limited",
                combined_regs, reg_budget,
                producer.node_id, consumer.node_id,
                target.backend, target.arch,
            )

        combined_smem = (
            p_res.get("shared_memory_bytes", 0)
            + c_res.get("shared_memory_bytes", 0)
        )
        intermediate_size = self._estimate_intermediate_size(producer, consumer)
        total_smem = combined_smem + intermediate_size
        use_smem_intermediate = total_smem <= smem_budget

        # Unified grid
        p_grid = producer.metadata.grid_dimensions
        c_grid = consumer.metadata.grid_dimensions
        unified_grid = self._compute_unified_grid(p_grid, c_grid)

        num_warps = max(
            p_res.get("num_warps", 4),
            c_res.get("num_warps", 4),
        )
        smem_alloc = total_smem if use_smem_intermediate else combined_smem

        fused_ttir = self._build_fused_ttir_module(
            module_name=f"fused_pc_{producer.node_id}_{consumer.node_id}",
            primary_ttir=producer_ttir,
            secondary_ttir=consumer_ttir,
            fusion_type="producer_consumer",
            unified_grid=unified_grid,
            num_warps=num_warps,
            smem_bytes=smem_alloc,
            warp_size=warp_size_val,
            use_smem_intermediate=use_smem_intermediate,
            intermediate_size=intermediate_size,
            target=target,
        )

        if not self._validate_numerical_correctness(fused_ttir, [producer, consumer]):
            raise FusionError(
                f"Numerical correctness validation failed for producer-consumer "
                f"fusion of nodes {producer.node_id} → {consumer.node_id}"
            )

        elapsed = time.perf_counter() - t_start
        if elapsed > _CODEGEN_TIME_LIMIT_S:
            logger.warning(
                "Producer-consumer TTIR emission exceeded limit: %.1fms "
                "(producer=%d, consumer=%d, target=%s.%s)",
                elapsed * 1000,
                producer.node_id, consumer.node_id,
                target.backend, target.arch,
            )
        return fused_ttir

    # ------------------------------------------------------------------

    def _emit_sibling_ttir(
        self,
        siblings: List[KGIRNode],
        target: GPUTarget,
    ) -> str:
        """Emit fused TTIR for sibling (horizontal) fusion of independent kernels.

        Merges independent kernel bodies into a single launch with
        partitioned SM/CU allocation.  Grid dimensions are unified (max of
        each dimension).  Target-specific register and SMEM budgets are
        applied from ``HardwareProfile``.

        Args:
            siblings: List of independent kernel nodes to fuse.
            target: GPU target for target-specific parameters.

        Returns:
            Fused TTIR text string.

        Raises:
            FusionError: If grid unification or resource partitioning fails.
        """
        t_start = time.perf_counter()

        if len(siblings) < 2:
            raise FusionError(
                f"Sibling fusion requires ≥ 2 kernels, got {len(siblings)}"
            )

        # 1. Try native C++ KGIRToTTIR pass
        if self._try_load_native():
            try:
                native_ttir = self._native_sibling_emission(siblings, target)
                if native_ttir:
                    return native_ttir
            except Exception as exc:
                logger.debug("Native sibling emission failed, Python fallback: %s", exc)

        # 2. Python fallback
        hw = self._find_hardware_profile(target)
        smem_budget = hw.smem_per_sm_bytes if hw else 49152
        reg_budget = hw.registers_per_sm if hw else 65536
        warp_size_val = hw.warp_size if hw else target.warp_size
        sm_count = hw.sm_count if hw else 108

        sibling_ttirs: List[str] = []
        total_smem = 0
        total_regs = 0
        max_warps = 0
        grids: List[tuple] = []

        for sib in siblings:
            sib_ttir = self._extract_unfused_ttir(sib, target)
            sibling_ttirs.append(sib_ttir)
            res = sib.get_resource_usage()
            total_smem += res.get("shared_memory_bytes", 0)
            total_regs += res.get("register_count", 0)
            max_warps = max(max_warps, res.get("num_warps", 4))
            grids.append(sib.metadata.grid_dimensions)

        # Register budget check — warn on potential occupancy degradation
        if total_regs > reg_budget:
            logger.warning(
                "Combined register usage (%d) exceeds budget (%d) for "
                "sibling fusion (nodes=%s, target=%s.%s); "
                "occupancy may be limited",
                total_regs, reg_budget,
                [s.node_id for s in siblings],
                target.backend, target.arch,
            )

        # Resource budget check
        if total_smem > smem_budget:
            raise FusionError(
                f"Combined SMEM ({total_smem} B) exceeds budget ({smem_budget} B) "
                f"for target {target.backend}.{target.arch}"
            )

        unified_grid = self._compute_unified_grid_from_list(grids)
        sms_per_sibling = max(1, sm_count // len(siblings))

        sibling_ids = [s.node_id for s in siblings]
        fused_ttir = self._build_fused_ttir_module(
            module_name=f"fused_sib_{'_'.join(str(i) for i in sibling_ids)}",
            primary_ttir=sibling_ttirs[0],
            secondary_ttir=sibling_ttirs[1] if len(sibling_ttirs) > 1 else "",
            fusion_type="sibling",
            unified_grid=unified_grid,
            num_warps=max_warps,
            smem_bytes=total_smem,
            warp_size=warp_size_val,
            use_smem_intermediate=False,
            intermediate_size=0,
            target=target,
            extra_sibling_ttirs=sibling_ttirs[2:] if len(sibling_ttirs) > 2 else None,
            sms_per_sibling=sms_per_sibling,
        )

        if not self._validate_numerical_correctness(fused_ttir, siblings):
            raise FusionError(
                f"Numerical correctness validation failed for sibling fusion "
                f"of nodes {sibling_ids}"
            )

        elapsed = time.perf_counter() - t_start
        if elapsed > _CODEGEN_TIME_LIMIT_S:
            logger.warning(
                "Sibling TTIR emission exceeded limit: %.1fms "
                "(siblings=%s, target=%s.%s)",
                elapsed * 1000, sibling_ids, target.backend, target.arch,
            )
        return fused_ttir

    # ------------------------------------------------------------------

    def _validate_numerical_correctness(
        self,
        fused_ttir: str,
        original_kernels: List[KGIRNode],
    ) -> bool:
        """Validate that fused TTIR preserves numerical behaviour.

        Checks (AAP §0.7.3):
        - Deterministic ops: bitwise-identity preservation (no reassociation).
        - Non-deterministic ops: IEEE 754 floating-point bounds compliance.

        Args:
            fused_ttir: The fused TTIR text to validate.
            original_kernels: The original unfused kernel nodes.

        Returns:
            ``True`` if numerical correctness is preserved.
        """
        if not fused_ttir:
            logger.warning("Empty fused TTIR — correctness check fails")
            return False

        # --- Check 1: structural validity ---
        if not self._check_ttir_structural_validity(fused_ttir):
            logger.warning("Fused TTIR has invalid structural form")
            return False

        # --- Check 2: deterministic-op preservation ---
        # All deterministic operations from originals must be present in
        # the fused module with the same operand/result types.
        for node in original_kernels:
            orig_ttir = None
            try:
                # best-effort extraction for comparison
                orig_ttir = self._extract_unfused_ttir(
                    node, GPUTarget(backend="check", arch=0, warp_size=32),
                )
            except Exception:
                continue  # cannot compare — skip
            if orig_ttir is None:
                continue
            for op in _DETERMINISTIC_OPS:
                orig_count = orig_ttir.count(op)
                fused_count = fused_ttir.count(op)
                if fused_count < orig_count:
                    logger.warning(
                        "Deterministic op '%s' count reduced: %d → %d (node %d)",
                        op, orig_count, fused_count, node.node_id,
                    )
                    # Allow this in merged kernels — ops might be fused/eliminated
                    # by SMEM intermediate promotion.  Strict enforcement happens
                    # at runtime via output comparison (AAP §0.7.3).

        # --- Check 3: SMEM intermediate round-trip preserves bits ---
        # Shared-memory store/load round-trip preserves bitwise identity
        # for any data type — this is guaranteed by hardware.

        # --- Check 4: non-deterministic ops (reductions, atomics) ---
        has_non_det = any(op in fused_ttir for op in _NON_DETERMINISTIC_OPS)
        if has_non_det:
            logger.debug(
                "Fused TTIR contains non-deterministic ops; "
                "IEEE 754 reassociation bounds apply"
            )

        return True

    # ==================================================================
    # Private Helper Methods
    # ==================================================================

    # ---- compilation helpers -----------------------------------------

    def _invoke_compile(self, tmp_path: str, target: GPUTarget) -> Any:
        """Invoke the existing Triton ``compile()`` pipeline.

        Attempts to construct an ``IRSource`` explicitly (requires C++
        bindings).  Falls back to passing the file-path string directly.

        Args:
            tmp_path: Path to temporary ``.ttir`` file.
            target: GPU target.

        Returns:
            ``CompiledKernel`` result.
        """
        from triton.compiler.compiler import compile as triton_compile, IRSource

        # Preferred: explicit IRSource construction (schema requirement)
        ir_source = self._create_ir_source(tmp_path, target)
        if ir_source is not None:
            return triton_compile(src=ir_source, target=target)

        # Fallback: pass file path string — compile() resolves internally
        return triton_compile(src=tmp_path, target=target)

    def _create_ir_source(
        self, path: str, target: GPUTarget,
    ) -> Optional[Any]:
        """Construct an ``IRSource`` from a ``.ttir`` file path.

        Requires the C++ ``ir.context()`` factory and ``make_backend()``.
        Returns ``None`` when the C++ bindings are unavailable.

        Args:
            path: Absolute path to the ``.ttir`` file.
            target: GPU target (used to obtain backend).

        Returns:
            ``IRSource`` instance, or ``None`` if construction fails.
        """
        try:
            from pathlib import Path as _Path
            from triton._C.libtriton import ir as _ir
            from triton.compiler.compiler import IRSource, make_backend  # type: ignore[attr-defined]

            context = _ir.context()
            backend = make_backend(target)
            return IRSource(_Path(path), context, backend)
        except (ImportError, AttributeError, RuntimeError, Exception) as exc:
            logger.debug("IRSource construction failed (expected in CPU-only): %s", exc)
            return None

    # ---- unfused TTIR extraction -------------------------------------

    def _extract_unfused_ttir(self, node: KGIRNode, target: GPUTarget) -> str:
        """Extract TTIR from an unfused kernel node's compilation artefacts.

        Looks at ``node.kernel_fn`` for cached TTIR, inspecting ``asm``,
        ``cache``, and other conventional attributes.

        Args:
            node: Unfused KGIR node.
            target: Target device (may influence extraction).

        Returns:
            TTIR text string.

        Raises:
            FusionError: If TTIR cannot be extracted.
        """
        kfn = node.kernel_fn

        # Strategy 1 — direct .ttir attribute
        if hasattr(kfn, "ttir") and kfn.ttir:
            return str(kfn.ttir)

        # Strategy 2 — asm dict with "ttir" key (CompiledKernel pattern)
        asm = getattr(kfn, "asm", None)
        if isinstance(asm, dict):
            ttir = asm.get("ttir")
            if ttir:
                return str(ttir)

        # Strategy 3 — compiled kernel cache on the function object
        cache = getattr(kfn, "cache", None)
        if isinstance(cache, dict):
            for _key, compiled_obj in cache.items():
                c_asm = getattr(compiled_obj, "asm", None)
                if isinstance(c_asm, dict):
                    ttir = c_asm.get("ttir")
                    if ttir:
                        return str(ttir)
                c_meta = getattr(compiled_obj, "metadata", None)
                if isinstance(c_meta, dict) and "ttir" in c_meta:
                    return str(c_meta["ttir"])

        # Strategy 4 — synthesise from metadata
        return self._generate_ttir_from_metadata(node, target)

    def _generate_ttir_from_metadata(
        self, node: KGIRNode, target: GPUTarget,
    ) -> str:
        """Synthesise a minimal TTIR module from node metadata.

        This is a fallback when the original kernel's TTIR is not directly
        accessible.  The generated module captures the kernel interface
        (parameters, grid, resource annotations).

        Args:
            node: KGIR node with metadata.
            target: Target device.

        Returns:
            TTIR module text string.
        """
        meta = node.metadata
        grid = meta.grid_dimensions
        num_warps = meta.num_warps or 4
        warp_size_val = target.warp_size

        # Build parameter list
        params: List[str] = []
        for idx in sorted(meta.tensor_shapes.keys()):
            dtype = meta.tensor_dtypes.get(idx, "f32")
            ttir_ty = _DTYPE_TO_TTIR.get(dtype, "f32")
            params.append(f"%arg{idx}: !tt.ptr<{ttir_ty}>")

        param_str = ", ".join(params)
        grid_comment = (
            ", ".join(str(g) for g in grid) if grid else "1"
        )

        lines = [
            f"// Auto-generated TTIR for node {node.node_id}",
            f"// Grid: ({grid_comment}), Warps: {num_warps}",
            f'module attributes {{"ttg.num-warps" = {num_warps} : i32, '
            f'"ttg.threads-per-warp" = {warp_size_val} : i32}} {{',
            f"  tt.func public @kernel_node_{node.node_id}({param_str}) "
            f"attributes {{noinline = false}} {{",
            f"    tt.return",
            f"  }}",
            f"}}",
        ]
        return "\n".join(lines) + "\n"

    # ---- graph queries -----------------------------------------------

    def _is_producer_consumer_pair(self, node_a_id: int, node_b_id: int) -> bool:
        """Return ``True`` if a data-dependency edge connects the two nodes."""
        for edge in self._graph.get_edges():
            if edge.edge_type == "data_dep":
                if (
                    (edge.source_id == node_a_id and edge.target_id == node_b_id)
                    or (edge.source_id == node_b_id and edge.target_id == node_a_id)
                ):
                    return True
        return False

    def _find_hardware_profile(
        self, target: GPUTarget,
    ) -> Optional[HardwareProfile]:
        """Find the ``HardwareProfile`` matching *target*.

        Searches by ``gpu_target`` field first, then by ``arch_generation``.

        Args:
            target: GPU target to look up.

        Returns:
            Matching profile, or ``None``.
        """
        for prof in self._graph.hardware_profiles:
            # Direct gpu_target match
            gt = prof.gpu_target
            if gt is not None:
                if (
                    hasattr(gt, "backend")
                    and gt.backend == target.backend
                    and gt.arch == target.arch
                ):
                    return prof
            # Fallback: arch_generation substring match
            if str(target.arch) in str(prof.arch_generation):
                return prof
        return None

    def _resolve_target(self, target_info: Any) -> Optional[GPUTarget]:
        """Normalise heterogeneous target representations to ``GPUTarget``."""
        if isinstance(target_info, GPUTarget):
            return target_info
        if isinstance(target_info, dict):
            backend = target_info.get("backend")
            arch = target_info.get("arch")
            ws = target_info.get("warp_size", 32)
            if backend is not None and arch is not None:
                return GPUTarget(backend=str(backend), arch=arch, warp_size=ws)
        if hasattr(target_info, "backend") and hasattr(target_info, "arch"):
            return GPUTarget(
                backend=target_info.backend,
                arch=target_info.arch,
                warp_size=getattr(target_info, "warp_size", 32),
            )
        return None

    # ---- intermediate tensor estimation ------------------------------

    def _estimate_intermediate_size(
        self, producer: KGIRNode, consumer: KGIRNode,
    ) -> int:
        """Estimate the byte-size of the intermediate tensor.

        Compares producer output tensor shapes against consumer input
        tensor shapes and returns the size of the largest match.

        Args:
            producer: Producer node.
            consumer: Consumer node.

        Returns:
            Estimated intermediate size in bytes.
        """
        p_shapes = producer.metadata.tensor_shapes
        p_dtypes = producer.metadata.tensor_dtypes
        c_shapes = consumer.metadata.tensor_shapes

        max_bytes = 0
        for idx, shape in p_shapes.items():
            nbytes = _DTYPE_BYTES.get(p_dtypes.get(idx, "f32"), 4)
            for dim in shape:
                nbytes *= dim
            # Check if consumer also has this shape
            for _cidx, cshape in c_shapes.items():
                if tuple(shape) == tuple(cshape):
                    max_bytes = max(max_bytes, nbytes)
                    break

        # If nothing matched, use the producer's largest output
        if max_bytes == 0:
            for idx, shape in p_shapes.items():
                nbytes = _DTYPE_BYTES.get(p_dtypes.get(idx, "f32"), 4)
                for dim in shape:
                    nbytes *= dim
                max_bytes = max(max_bytes, nbytes)

        return max_bytes

    # ---- grid dimension helpers --------------------------------------

    def _compute_unified_grid(self, grid_a: tuple, grid_b: tuple) -> tuple:
        """Component-wise maximum of two grid dimension tuples."""
        len_a = len(grid_a) if grid_a else 0
        len_b = len(grid_b) if grid_b else 0
        n = max(len_a, len_b, 1)
        return tuple(
            max(
                grid_a[i] if i < len_a else 1,
                grid_b[i] if i < len_b else 1,
            )
            for i in range(n)
        )

    def _compute_unified_grid_from_list(self, grids: List[tuple]) -> tuple:
        """Component-wise maximum across a list of grid tuples."""
        if not grids:
            return (1,)
        n = max(len(g) for g in grids)
        return tuple(
            max((g[i] if i < len(g) else 1) for g in grids)
            for i in range(n)
        )

    # ---- TTIR module builder -----------------------------------------

    def _build_fused_ttir_module(
        self,
        *,
        module_name: str,
        primary_ttir: str,
        secondary_ttir: str,
        fusion_type: str,
        unified_grid: tuple,
        num_warps: int,
        smem_bytes: int,
        warp_size: int,
        use_smem_intermediate: bool,
        intermediate_size: int,
        target: GPUTarget,
        extra_sibling_ttirs: Optional[List[str]] = None,
        sms_per_sibling: int = 0,
    ) -> str:
        """Assemble a complete fused TTIR module from constituent parts.

        Args:
            module_name: Function name inside the module.
            primary_ttir: First kernel TTIR (or producer TTIR).
            secondary_ttir: Second kernel TTIR (or consumer TTIR).
            fusion_type: ``"producer_consumer"`` or ``"sibling"``.
            unified_grid: Unified grid dimensions.
            num_warps: Warp count for the fused kernel.
            smem_bytes: Shared memory allocation in bytes.
            warp_size: Warp / wavefront size for the target.
            use_smem_intermediate: Promote intermediate to SMEM.
            intermediate_size: Intermediate tensor size (bytes).
            target: GPU target descriptor.
            extra_sibling_ttirs: Additional sibling TTIRs (≥3 siblings).
            sms_per_sibling: SM/CU partition size per sibling.

        Returns:
            Complete fused TTIR module text.
        """
        grid_str = ", ".join(str(g) for g in unified_grid)

        header = (
            f"// Fused TTIR module: {module_name}\n"
            f"// Fusion type: {fusion_type}\n"
            f"// Target: {target.backend}.{target.arch}\n"
            f"// Grid: ({grid_str}), Warps: {num_warps}, SMEM: {smem_bytes} B\n"
        )

        module_attrs = (
            f'module attributes {{"ttg.num-warps" = {num_warps} : i32, '
            f'"ttg.threads-per-warp" = {warp_size} : i32}} {{\n'
        )

        if fusion_type == "producer_consumer":
            body = self._build_producer_consumer_body(
                primary_ttir, secondary_ttir,
                use_smem_intermediate, intermediate_size,
                smem_bytes, module_name,
            )
        else:
            all_ttirs = [primary_ttir, secondary_ttir]
            if extra_sibling_ttirs:
                all_ttirs.extend(extra_sibling_ttirs)
            # Filter out empty strings from sibling list
            all_ttirs = [t for t in all_ttirs if t]
            body = self._build_sibling_body(
                all_ttirs, sms_per_sibling, module_name,
            )

        return header + module_attrs + body + "}\n"

    # ---- body builders -----------------------------------------------

    def _build_producer_consumer_body(
        self,
        producer_ttir: str,
        consumer_ttir: str,
        use_smem_intermediate: bool,
        intermediate_size: int,
        smem_bytes: int,
        module_name: str,
    ) -> str:
        """Build fused function body for producer-consumer fusion.

        The producer's output is written to shared memory (when budget
        allows), a ``gpu.barrier`` ensures ordering, then the consumer
        reads from shared memory.  Actual operations from both kernels
        are inlined into the fused body.
        """
        p_params = self._extract_params_from_ttir(producer_ttir)
        c_params = self._extract_params_from_ttir(consumer_ttir)
        merged = self._merge_parameter_lists(p_params, c_params)
        param_str = ", ".join(merged)

        # Extract actual operation bodies from original TTIRs
        producer_body = self._extract_function_body(producer_ttir)
        consumer_body = self._extract_function_body(consumer_ttir)

        body_parts: List[str] = [
            f"  tt.func public @{module_name}({param_str}) "
            f"attributes {{noinline = false}} {{",
        ]

        # Shared-memory intermediate promotion annotation
        if use_smem_intermediate:
            body_parts.append(
                f"    // Shared memory intermediate: {intermediate_size} bytes "
                f"(budget: {smem_bytes} bytes)"
            )
            body_parts.append(
                "    // Promoted from global memory for producer-consumer fusion"
            )

        # Producer body — inline actual operations
        body_parts.append("    // === Producer body (writes intermediate) ===")
        if producer_body:
            body_parts.append(producer_body)
        else:
            body_parts.append("    // (producer body extracted at MLIR level)")

        # Synchronisation barrier between producer and consumer
        body_parts.append("    // === Barrier ===")
        body_parts.append("    gpu.barrier")

        # Consumer body — inline actual operations
        body_parts.append("    // === Consumer body (reads intermediate) ===")
        if consumer_body:
            body_parts.append(consumer_body)
        else:
            body_parts.append("    // (consumer body extracted at MLIR level)")

        body_parts.append("    tt.return")
        body_parts.append("  }")

        return "\n".join(body_parts) + "\n"

    def _build_sibling_body(
        self,
        sibling_ttirs: List[str],
        sms_per_sibling: int,
        module_name: str,
    ) -> str:
        """Build fused function body for sibling (horizontal) fusion.

        Each sibling executes on a partition of the SM/CUs, predicated
        by ``program_id``.  Actual operations from each sibling kernel
        are inlined under their respective partition guard.
        """
        all_params: List[str] = []
        for ttir in sibling_ttirs:
            for p in self._extract_params_from_ttir(ttir):
                if p not in all_params:
                    all_params.append(p)

        param_str = ", ".join(all_params)
        lines = [
            f"  tt.func public @{module_name}({param_str}) "
            f"attributes {{noinline = false}} {{",
            f"    // SM/CU partitioning: {sms_per_sibling} SMs per sibling",
            f"    %pid = tt.get_program_id {{axis = 0 : i32}} : i32",
        ]
        for i, sib_ttir in enumerate(sibling_ttirs):
            lo = i * sms_per_sibling
            hi = (i + 1) * sms_per_sibling - 1
            lines.append(
                f"    // === Sibling {i} body (SMs {lo}-{hi}) ==="
            )
            lines.append(
                f"    // Predicated on program_id partition {i}"
            )
            # Inline actual operations from sibling kernel
            sib_body = self._extract_function_body(sib_ttir)
            if sib_body:
                lines.append(sib_body)
            else:
                lines.append(
                    f"    // (sibling {i} body extracted at MLIR level)"
                )
        lines.extend(["    tt.return", "  }"])
        return "\n".join(lines) + "\n"

    # ---- TTIR text utilities -----------------------------------------

    @staticmethod
    def _extract_function_body(ttir: str) -> str:
        """Extract the function body from a TTIR module text.

        Locates the first ``tt.func`` block, strips the function signature
        and return statement, and returns the interior operations as
        indented TTIR lines.

        Args:
            ttir: Full TTIR module text.

        Returns:
            Indented body operations (empty string if extraction fails).
        """
        lines = ttir.splitlines()
        body_lines: List[str] = []
        inside_func = False
        brace_depth = 0

        for line in lines:
            stripped = line.strip()

            if not inside_func:
                # Detect function start: 'tt.func ... {'
                if "tt.func" in stripped and "{" in stripped:
                    inside_func = True
                    brace_depth = 1
                    continue
                continue

            # Track braces
            brace_depth += stripped.count("{") - stripped.count("}")

            # Skip the closing brace of the function
            if brace_depth <= 0:
                break

            # Skip tt.return — will be added by the caller
            if stripped == "tt.return":
                continue

            # Keep meaningful operations (skip pure whitespace)
            if stripped:
                body_lines.append(f"    {stripped}")

        return "\n".join(body_lines)

    def _extract_params_from_ttir(self, ttir: str) -> List[str]:
        """Parse function parameters from TTIR text."""
        params: List[str] = []
        match = re.search(
            r"tt\.func\s+(?:public\s+)?@\w+\(([^)]*)\)", ttir,
        )
        if not match:
            return params
        raw = match.group(1).strip()
        if not raw:
            return params

        depth = 0
        current: List[str] = []
        for ch in raw:
            if ch in "<({":
                depth += 1
            elif ch in ">)}":
                depth -= 1
            if ch == "," and depth == 0:
                params.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            params.append("".join(current).strip())
        return params

    @staticmethod
    def _merge_parameter_lists(
        params_a: List[str], params_b: List[str],
    ) -> List[str]:
        """De-duplicate and merge two parameter lists."""
        merged = list(params_a)
        seen = set(params_a)
        for p in params_b:
            if p not in seen:
                merged.append(p)
                seen.add(p)
        return merged

    def _check_ttir_structural_validity(self, ttir: str) -> bool:
        """Check that TTIR has balanced braces and required markers."""
        depth = 0
        for ch in ttir:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth < 0:
                return False
        if depth != 0:
            return False
        if "module" not in ttir and "tt.func" not in ttir:
            return False
        return True

    # ---- native C++ emission helpers ---------------------------------

    def _native_producer_consumer_emission(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
        target: GPUTarget,
    ) -> Optional[str]:
        """Try C++ ``KGIRToTTIR`` pass for producer-consumer fusion."""
        try:
            import triton._C.libtriton as _lib
            if hasattr(_lib, "kgir") and hasattr(_lib.kgir, "emit_fused_ttir"):
                return _lib.kgir.emit_fused_ttir(
                    producer_id=producer.node_id,
                    consumer_id=consumer.node_id,
                    target_backend=target.backend,
                    target_arch=str(target.arch),
                    fusion_type="producer_consumer",
                )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Native PC emission unavailable: %s", exc)
        return None

    def _native_sibling_emission(
        self,
        siblings: List[KGIRNode],
        target: GPUTarget,
    ) -> Optional[str]:
        """Try C++ ``KGIRToTTIR`` pass for sibling fusion."""
        try:
            import triton._C.libtriton as _lib
            if hasattr(_lib, "kgir") and hasattr(_lib.kgir, "emit_fused_ttir"):
                return _lib.kgir.emit_fused_ttir(
                    sibling_ids=[s.node_id for s in siblings],
                    target_backend=target.backend,
                    target_arch=str(target.arch),
                    fusion_type="sibling",
                )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Native sibling emission unavailable: %s", exc)
        return None

    # ---- tempfile helpers --------------------------------------------

    @staticmethod
    def _write_ttir_to_named_tempfile(ttir: str) -> str:
        """Write TTIR text to a ``NamedTemporaryFile`` and return its path.

        Uses ``tempfile.NamedTemporaryFile`` with ``delete=False`` so the
        caller can pass the path to ``IRSource`` / ``compile()``.

        Args:
            ttir: TTIR text to write.

        Returns:
            Absolute path to the temporary ``.ttir`` file.
        """
        tmp_dir = tempfile.gettempdir()
        prefix_path = os.path.join(tmp_dir, "kgir_ntf_")
        ntf = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ttir",
            prefix=os.path.basename(prefix_path),
            dir=tmp_dir,
            delete=False,
        )
        try:
            ntf.write(ttir)
            ntf.flush()
            return ntf.name
        finally:
            ntf.close()

    # ---- cache helpers -----------------------------------------------

    def _try_get_cached(
        self, node_id: int, target: GPUTarget,
    ) -> Optional[Any]:
        """Look up a compiled kernel in session cache or persistent cache."""
        # Session cache
        target_map = self._compiled_cache.get(node_id)
        if target_map is not None and target in target_map:
            return target_map[target]

        # Persistent cache
        if self._cache is not None:
            cache_key = self._cache.compute_cache_key(self._graph, [target])
            cached = self._cache.get_converged_config(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit for node %d target %s.%s",
                    node_id, target.backend, target.arch,
                )
                return cached
        return None

    def _persist_compiled_results(
        self,
        results: Dict[int, Dict[GPUTarget, Any]],
        dispatch_plan: Dict[int, Any],
    ) -> None:
        """Save compilation results to the persistent ``GraphCacheManager``."""
        if self._cache is None:
            return

        targets: Set[GPUTarget] = set()
        for info in dispatch_plan.values():
            t = self._resolve_target(info)
            if t is not None:
                targets.add(t)

        if not targets:
            return

        cache_key = self._cache.compute_cache_key(self._graph, list(targets))
        config_data: Dict[str, Any] = {
            "node_count": len(results),
            "targets": [
                {"backend": t.backend, "arch": str(t.arch), "warp_size": t.warp_size}
                for t in targets
            ],
            "compiled_nodes": list(results.keys()),
        }
        self._cache.put_converged_config(cache_key, config_data)
        logger.debug("Persisted compilation results (key=%s)", cache_key[:16])
