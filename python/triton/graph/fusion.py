"""Fusion Analysis Engine for the Triton KGIR graph-level optimization layer.

This module implements producer-consumer fusion and sibling/horizontal fusion
analysis operating on the Kernel Graph IR (KGIR). It includes an adaptive
two-phase cost model that starts with cold-start static heuristics (Phase 1)
and transitions to measured runtime data (Phase 2) when profiler feedback
becomes available.

Fusion decisions are per-target: fusibility varies by shared memory capacity,
register file size, warp/wavefront width, and other hardware characteristics
described in the HardwareProfile.

Key classes:
  - FusionPlan: Immutable result container for all fusion decisions.
  - AdaptiveCostModel: Two-phase cost model (heuristic → measured).
  - ProducerConsumerAnalyzer: Identifies fusible producer-consumer kernel pairs.
  - SiblingFusionAnalyzer: Identifies independent kernels for horizontal fusion.
  - FusionEngine: Orchestrates all analysis and returns a combined FusionPlan.

Environment variables (via triton.knobs.graph):
  - TRITON_FUSION_DISABLE: Skip all fusion analysis.
  - TRITON_FUSION_LOG: Enable detailed fusion decision logging.
  - TRITON_FUSION_THRESHOLD: Minimum benefit score for fusion (default "0.10").
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import FusionConfig
from .errors import FusionError
from .kgir import HardwareProfile, KGIRGraph, KGIRNode, NodeMetadata
from .utils import (
    are_independent,
    compute_tensor_size_bytes,
    compute_unified_grid,
    detect_cycle,
    grids_compatible,
    shapes_compatible,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants for the Phase 1 heuristic cost model
# ---------------------------------------------------------------------------
_DEFAULT_LAUNCH_OVERHEAD_US: float = 5.0
"""Typical GPU kernel launch overhead in microseconds."""

_DEFAULT_RESOURCE_PENALTY: float = 0.5
"""Penalty multiplier applied to the resource pressure ratio."""

_DEFAULT_BANDWIDTH_COST_SCALE: float = 1.0
"""Scaling factor for bandwidth-based benefit of eliminated memory traffic."""

_CALIBRATION_CACHE_DIR: str = os.path.join(
    os.path.expanduser("~"), ".triton", "cache", "graph_calibration"
)
"""Directory where per-target calibration JSON files are stored."""


# ---------------------------------------------------------------------------
# Helper: build adjacency dict from the KGIRGraph public API
# ---------------------------------------------------------------------------
def _build_adjacency(graph: KGIRGraph) -> Dict[int, List[int]]:
    """Construct an adjacency dict {node_id: [successor_ids]} from *graph*.

    Uses ``graph.topological_sort()`` to enumerate all node IDs and
    ``graph.get_successors()`` for per-node outgoing edges.
    """
    if graph.node_count() == 0:
        return {}
    topo_order = graph.topological_sort()
    adjacency: Dict[int, List[int]] = {}
    for nid in topo_order:
        adjacency[nid] = list(graph.get_successors(nid))
    return adjacency


def _get_all_node_ids(graph: KGIRGraph) -> List[int]:
    """Return all node IDs in *graph* in topological order."""
    if graph.node_count() == 0:
        return []
    return list(graph.topological_sort())


def _next_free_node_id(graph: KGIRGraph) -> int:
    """Return a node ID guaranteed not to exist in *graph*."""
    all_ids = _get_all_node_ids(graph)
    return (max(all_ids) + 1) if all_ids else 0


def _would_fusion_create_cycle(
    adjacency: Dict[int, List[int]],
    producer_id: int,
    consumer_id: int,
) -> bool:
    """Return *True* if fusing *producer_id* and *consumer_id* would create a DAG cycle.

    The check works by removing the direct producer→consumer edge from the
    adjacency and performing a BFS from the producer. If the consumer is still
    reachable, a cycle would be created by the fusion.
    """
    # Build temporary adjacency without the direct edge
    temp_succs = [s for s in adjacency.get(producer_id, []) if s != consumer_id]

    visited: set[int] = set()
    queue: list[int] = list(temp_succs)
    while queue:
        node = queue.pop()
        if node == consumer_id:
            return True
        if node in visited:
            continue
        visited.add(node)
        for succ in adjacency.get(node, []):
            if succ not in visited:
                queue.append(succ)
    return False


# ---------------------------------------------------------------------------
# Knobs access helpers (lazy import to avoid circular dependency)
# ---------------------------------------------------------------------------
def _read_knob_fusion_disable() -> bool:
    """Return *True* when ``TRITON_FUSION_DISABLE`` is set."""
    try:
        from triton import knobs
        return bool(knobs.graph.fusion_disable)
    except (ImportError, AttributeError):
        return False


def _read_knob_fusion_log() -> bool:
    """Return *True* when ``TRITON_FUSION_LOG`` is set."""
    try:
        from triton import knobs
        return bool(knobs.graph.fusion_log)
    except (ImportError, AttributeError):
        return False


def _read_knob_fusion_threshold() -> float:
    """Return the fusion threshold from knobs, defaulting to 0.10."""
    try:
        from triton import knobs
        return float(knobs.graph.fusion_threshold)
    except (ImportError, AttributeError, ValueError):
        return 0.10


# ===================================================================
# FusionPlan — Immutable result container
# ===================================================================
@dataclass
class FusionPlan:
    """Represents all fusion decisions for a graph on a specific hardware target.

    Attributes:
        producer_consumer_pairs: Ordered list of ``(producer_id, consumer_id)``
            tuples identifying producer-consumer kernel pairs selected for fusion.
        sibling_groups: List of kernel-ID groups selected for horizontal / sibling
            fusion.  Each inner list contains ≥ 2 independent kernel IDs.
        target: The :class:`HardwareProfile` these decisions were computed for.
            ``None`` when the plan is empty (e.g. fusion disabled).
        estimated_speedup: Aggregate estimated speedup ratio (≥ 1.0 means faster).
        cost_model_phase: 1 for heuristic-based decisions, 2 for measurement-based.
    """
    producer_consumer_pairs: List[Tuple[int, int]] = field(default_factory=list)
    sibling_groups: List[List[int]] = field(default_factory=list)
    target: Optional[HardwareProfile] = None
    estimated_speedup: float = 1.0
    cost_model_phase: int = 1

    @property
    def is_empty(self) -> bool:
        """Return *True* when no fusion decisions are present."""
        return (
            len(self.producer_consumer_pairs) == 0
            and len(self.sibling_groups) == 0
        )

    @property
    def total_fusions(self) -> int:
        """Return the total number of individual fusions in this plan."""
        return len(self.producer_consumer_pairs) + len(self.sibling_groups)


# ===================================================================
# AdaptiveCostModel — Two-phase heuristic → measured cost model
# ===================================================================
class AdaptiveCostModel:
    """Two-phase cost model for fusion benefit estimation.

    **Phase 1 (cold-start heuristics):**
    Uses static formulas based on eliminated memory traffic, kernel launch
    overhead, and resource pressure — calibrated per-target from persisted
    JSON calibration data under ``~/.triton/cache/graph_calibration/``.

    **Phase 2 (measured runtime data):**
    Replaces heuristic estimates with actual profiler measurements fed back
    via :pymethod:`update_with_measurements`.

    Parameters:
        config: A :class:`FusionConfig` providing ``threshold``, ``log``, etc.
    """

    def __init__(self, config: FusionConfig) -> None:
        self._config = config
        self._phase: int = 1
        self._measured_data: Dict[str, Any] = {}
        self._calibration_cache: Dict[str, Dict[str, float]] = {}
        self._log_enabled: bool = config.log or _read_knob_fusion_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate_fusion_benefit(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
        target: HardwareProfile,
    ) -> float:
        """Estimate the benefit score of fusing *producer* with *consumer* on *target*.

        Returns a floating-point score; higher is better. A score above the
        configured ``fusion_threshold`` means fusion is recommended.
        """
        if self._phase == 2:
            return self._estimate_from_measurements(producer, consumer, target)
        return self._estimate_from_heuristics(producer, consumer, target)

    def should_fuse(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
        target: HardwareProfile,
    ) -> bool:
        """Decide whether *producer* and *consumer* should be fused on *target*.

        Combines :pymethod:`estimate_fusion_benefit` with the configured
        fusion threshold.
        """
        benefit = self.estimate_fusion_benefit(producer, consumer, target)
        threshold = self._config.threshold
        decision = benefit > threshold

        if self._log_enabled:
            logger.info(
                "CostModel(phase=%d): should_fuse(p=%d, c=%d, target=%s) "
                "benefit=%.6f threshold=%.4f → %s",
                self._phase,
                producer.node_id,
                consumer.node_id,
                target.arch_generation,
                benefit,
                threshold,
                decision,
            )
        return decision

    def update_with_measurements(self, measured: Dict) -> None:
        """Incorporate profiled measurements and transition to Phase 2.

        *measured* should be a dict mapping metric keys (e.g. per-kernel timing
        data, launch overhead) to their values. Once called with a non-empty
        dict, the model switches to Phase 2 permanently for subsequent calls.
        """
        if not measured:
            return
        self._measured_data.update(measured)
        if self._phase == 1:
            self._phase = 2
            if self._log_enabled:
                logger.info(
                    "AdaptiveCostModel transitioned to Phase 2 (measured) "
                    "with %d entries",
                    len(measured),
                )

    def get_phase(self) -> int:
        """Return 1 (heuristic) or 2 (measured)."""
        return self._phase

    # ------------------------------------------------------------------
    # Phase 1: heuristic estimation
    # ------------------------------------------------------------------
    def _estimate_from_heuristics(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
        target: HardwareProfile,
    ) -> float:
        """Phase 1 heuristic: ``Score = eliminated_bytes * bandwidth_cost
        + launch_overhead_benefit - resource_pressure * penalty``."""
        cal = self._load_calibration(target)

        eliminated_bytes = self._compute_eliminated_bytes(producer, consumer)
        resource_pressure = self._compute_resource_pressure(producer, consumer, target)

        bandwidth_cost = cal.get(
            "bandwidth_cost_per_byte",
            _DEFAULT_BANDWIDTH_COST_SCALE / max(target.memory_bandwidth_gbps, 0.001),
        )
        launch_overhead_us = cal.get("launch_overhead_us", _DEFAULT_LAUNCH_OVERHEAD_US)
        resource_penalty = cal.get("resource_penalty", _DEFAULT_RESOURCE_PENALTY)

        score = (
            eliminated_bytes * bandwidth_cost
            + launch_overhead_us * 1e-3  # convert µs → ms-scale additive benefit
            - resource_pressure * resource_penalty
        )
        return score

    # ------------------------------------------------------------------
    # Phase 2: measurement-based estimation
    # ------------------------------------------------------------------
    def _estimate_from_measurements(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
        target: HardwareProfile,
    ) -> float:
        """Phase 2: use profiled performance annotations."""
        target_key = target.arch_generation

        producer_perf = producer.get_performance_annotation(target_key)
        consumer_perf = consumer.get_performance_annotation(target_key)

        if producer_perf and consumer_perf:
            individual_time = (
                producer_perf.get("wall_clock_ms", 0.0)
                + consumer_perf.get("wall_clock_ms", 0.0)
            )

            # Estimate time saved from eliminated global memory traffic
            eliminated_bytes = self._compute_eliminated_bytes(producer, consumer)
            bandwidth_gbps = max(target.memory_bandwidth_gbps, 0.001)
            memory_save_ms = (eliminated_bytes / (bandwidth_gbps * 1e9)) * 1e3

            # Launch overhead saved (from measurements when available)
            launch_overhead_ms = self._measured_data.get(
                "launch_overhead_ms",
                _DEFAULT_LAUNCH_OVERHEAD_US * 1e-3,
            )

            total_benefit = memory_save_ms + launch_overhead_ms
            # Express as fraction of individual time for comparability
            if individual_time > 0.0:
                return total_benefit / individual_time
            return total_benefit

        # Fallback to heuristic when measurements are incomplete
        return self._estimate_from_heuristics(producer, consumer, target)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_calibration(self, target: HardwareProfile) -> Dict[str, float]:
        """Lazy-load per-target calibration data from the cache directory."""
        target_key = f"{target.vendor}_{target.arch_generation}"
        if target_key in self._calibration_cache:
            return self._calibration_cache[target_key]

        cal_path = os.path.join(_CALIBRATION_CACHE_DIR, f"{target_key}.json")
        if os.path.exists(cal_path):
            try:
                with open(cal_path, "r") as fp:
                    data: Dict[str, float] = json.load(fp)
                self._calibration_cache[target_key] = data
                if self._log_enabled:
                    logger.debug(
                        "Loaded calibration for %s from %s", target_key, cal_path
                    )
                return data
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                logger.warning(
                    "Failed to load calibration for %s: %s", target_key, exc
                )

        # Compute sensible defaults from the hardware profile
        defaults: Dict[str, float] = {
            "launch_overhead_us": _DEFAULT_LAUNCH_OVERHEAD_US,
            "bandwidth_cost_per_byte": (
                _DEFAULT_BANDWIDTH_COST_SCALE
                / max(target.memory_bandwidth_gbps, 0.001)
            ),
            "resource_penalty": _DEFAULT_RESOURCE_PENALTY,
        }
        self._calibration_cache[target_key] = defaults
        return defaults

    def _compute_eliminated_bytes(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
    ) -> int:
        """Compute bytes of intermediate tensors eliminated by fusing the pair.

        When a producer kernel writes an intermediate tensor to global memory
        and the consumer reads it, fusion eliminates both the write and the
        read — saving 2× the tensor size in global memory traffic.
        """
        total_eliminated: int = 0
        producer_meta = producer.metadata
        consumer_meta = consumer.metadata

        # Identify producer output tensors that the consumer also references
        producer_tensor_ids = set(producer_meta.tensor_shapes.keys())
        consumer_tensor_ids = set(consumer_meta.tensor_shapes.keys())
        shared_tensor_ids = producer_tensor_ids & consumer_tensor_ids

        for tid in shared_tensor_ids:
            shape = producer_meta.tensor_shapes[tid]
            dtype = producer_meta.tensor_dtypes.get(tid, "fp32")
            tensor_bytes = compute_tensor_size_bytes(shape, dtype)
            # Fusion eliminates one global write + one global read
            total_eliminated += tensor_bytes * 2

        # When tensor-level shape metadata is missing but shapes exist in the
        # producer only, estimate from all producer outputs.
        if total_eliminated == 0 and producer_tensor_ids:
            for tid in producer_tensor_ids:
                shape = producer_meta.tensor_shapes[tid]
                dtype = producer_meta.tensor_dtypes.get(tid, "fp32")
                total_eliminated += compute_tensor_size_bytes(shape, dtype) * 2

        # Final fallback: use shared-memory size as a rough proxy
        if total_eliminated == 0:
            total_eliminated = max(producer_meta.shared_memory_bytes, 1)

        return total_eliminated

    @staticmethod
    def _compute_resource_pressure(
        producer: KGIRNode,
        consumer: KGIRNode,
        target: HardwareProfile,
    ) -> float:
        """Compute the resource pressure ratio ``max(smem_ratio, reg_ratio)``."""
        p_res = producer.get_resource_usage()
        c_res = consumer.get_resource_usage()

        combined_smem = p_res["shared_memory_bytes"] + c_res["shared_memory_bytes"]
        combined_regs = p_res["register_count"] + c_res["register_count"]

        smem_ratio = combined_smem / max(target.smem_per_sm_bytes, 1)
        reg_ratio = combined_regs / max(target.registers_per_sm, 1)

        return max(smem_ratio, reg_ratio)


# ===================================================================
# ProducerConsumerAnalyzer
# ===================================================================
class ProducerConsumerAnalyzer:
    """Identifies fusible producer-consumer kernel pairs in a KGIR graph.

    Fusion criteria (all must hold):
      1. **Single consumer:** The intermediate tensor has exactly one consumer.
      2. **Compatible tiling:** Producer output shapes/strides match consumer
         input shapes/strides (via :func:`shapes_compatible`).
      3. **Resource budget:** Combined SMEM + registers fit within per-SM/CU
         limits for the target (via :pymethod:`KGIRNode.is_compatible_for_fusion`).
      4. **No cycles:** Fusing the pair does not introduce a DAG cycle.

    Parameters:
        cost_model: The :class:`AdaptiveCostModel` used for benefit scoring.
        config: The :class:`FusionConfig` controlling analyzer behaviour.
        target: The :class:`HardwareProfile` for per-target resource checks.
    """

    def __init__(
        self,
        cost_model: AdaptiveCostModel,
        config: FusionConfig,
        target: HardwareProfile,
    ) -> None:
        self._cost_model = cost_model
        self._config = config
        self._target = target
        self._graph: Optional[KGIRGraph] = None
        self._adjacency: Dict[int, List[int]] = {}
        self._log_enabled: bool = config.log or _read_knob_fusion_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_candidates(self, graph: KGIRGraph) -> List[Tuple[int, int]]:
        """Return ``(producer_id, consumer_id)`` pairs passing all fusion criteria.

        The graph is stored internally for use by the individual check helpers.
        """
        self._graph = graph
        self._adjacency = _build_adjacency(graph)

        if not self._config.enable_producer_consumer:
            if self._log_enabled:
                logger.info("ProducerConsumerAnalyzer: disabled by config")
            return []

        candidates: List[Tuple[int, int]] = []
        already_fused_nodes: set[int] = set()

        # Walk all data-dependency edges
        all_edges = graph.get_edges()
        data_dep_edges = [e for e in all_edges if e.edge_type == "data_dep"]

        for edge in data_dep_edges:
            producer_id = edge.source_id
            consumer_id = edge.target_id
            tensor_id = edge.tensor_id

            # Skip nodes already committed to a fusion in this pass
            if producer_id in already_fused_nodes or consumer_id in already_fused_nodes:
                continue

            # Skip already-fused nodes
            producer = graph.get_node(producer_id)
            consumer = graph.get_node(consumer_id)
            if producer is None or consumer is None:
                continue
            if producer.is_fused or consumer.is_fused:
                continue

            # Criterion 1: single consumer for the tensor
            if tensor_id is not None and not self.check_single_consumer(producer_id, tensor_id):
                if self._log_enabled:
                    logger.debug(
                        "PC reject (%d→%d): tensor %s has multiple consumers",
                        producer_id, consumer_id, tensor_id,
                    )
                continue

            # Criterion 2: tiling compatibility
            if not self.check_tiling_compatibility(producer, consumer):
                if self._log_enabled:
                    logger.debug(
                        "PC reject (%d→%d): incompatible tiling",
                        producer_id, consumer_id,
                    )
                continue

            # Criterion 3: resource budget
            if not self.check_resource_budget(producer, consumer, self._target):
                if self._log_enabled:
                    logger.debug(
                        "PC reject (%d→%d): exceeds resource budget for %s",
                        producer_id, consumer_id, self._target.arch_generation,
                    )
                continue

            # Criterion 4: no cycle creation
            if _would_fusion_create_cycle(self._adjacency, producer_id, consumer_id):
                if self._log_enabled:
                    logger.debug(
                        "PC reject (%d→%d): would create cycle",
                        producer_id, consumer_id,
                    )
                continue

            # Cost model gate
            if not self._cost_model.should_fuse(producer, consumer, self._target):
                if self._log_enabled:
                    logger.debug(
                        "PC reject (%d→%d): below cost threshold",
                        producer_id, consumer_id,
                    )
                continue

            candidates.append((producer_id, consumer_id))
            already_fused_nodes.add(producer_id)
            already_fused_nodes.add(consumer_id)

            if self._log_enabled:
                logger.info(
                    "PC candidate accepted: (%d→%d) on %s",
                    producer_id, consumer_id, self._target.arch_generation,
                )

        # Respect max-fused-kernels budget
        max_fusions = self._config.max_fused_kernels
        if len(candidates) > max_fusions:
            if self._log_enabled:
                logger.info(
                    "PC: clamping %d candidates to max_fused_kernels=%d",
                    len(candidates), max_fusions,
                )
            candidates = candidates[:max_fusions]

        return candidates

    def check_single_consumer(self, producer_id: int, tensor_id: int) -> bool:
        """Return *True* if *tensor_id* produced by *producer_id* has exactly one consumer."""
        if self._graph is None:
            return False
        edges = self._graph.get_edges(producer_id)
        consumers: set[int] = set()
        for edge in edges:
            if (
                edge.source_id == producer_id
                and edge.edge_type == "data_dep"
                and edge.tensor_id == tensor_id
            ):
                consumers.add(edge.target_id)
        return len(consumers) == 1

    def check_tiling_compatibility(
        self, producer: KGIRNode, consumer: KGIRNode
    ) -> bool:
        """Return *True* if *producer* output tiling is compatible with *consumer* input.

        Delegates to :func:`shapes_compatible` from ``utils.py`` for each
        overlapping tensor dimension pair.
        """
        p_shapes = producer.metadata.tensor_shapes
        c_shapes = consumer.metadata.tensor_shapes

        # When shape metadata is missing, assume compatibility (conservative)
        if not p_shapes or not c_shapes:
            return True

        # Check shapes for overlapping tensor indices
        shared_ids = set(p_shapes.keys()) & set(c_shapes.keys())
        if not shared_ids:
            # No overlapping tensors — check by positional matching
            # Use producer outputs vs consumer inputs ordering
            p_shape_list = list(p_shapes.values())
            c_shape_list = list(c_shapes.values())
            if p_shape_list and c_shape_list:
                return shapes_compatible(p_shape_list[0], c_shape_list[0])
            return True

        for tid in shared_ids:
            if not shapes_compatible(p_shapes[tid], c_shapes[tid]):
                return False
        return True

    def check_resource_budget(
        self,
        producer: KGIRNode,
        consumer: KGIRNode,
        target: HardwareProfile,
    ) -> bool:
        """Return *True* if the combined resources of the pair fit within *target* limits.

        Delegates to :pymethod:`KGIRNode.is_compatible_for_fusion` which checks
        shared memory, register, and warp budgets against the target profile.
        """
        return producer.is_compatible_for_fusion(consumer, target)

    def fuse(
        self,
        graph: KGIRGraph,
        pairs: List[Tuple[int, int]],
    ) -> KGIRGraph:
        """Apply producer-consumer fusion decisions to *graph* and return it.

        For each ``(producer_id, consumer_id)`` pair, a new fused
        :class:`KGIRNode` is created with merged metadata and the pair is
        replaced in the graph via :pymethod:`KGIRGraph.replace_nodes_with_fused`.

        After all fusions, the graph is verified to remain acyclic via
        :func:`detect_cycle`.
        """
        if not pairs:
            return graph

        for producer_id, consumer_id in pairs:
            producer = graph.get_node(producer_id)
            consumer = graph.get_node(consumer_id)
            if producer is None or consumer is None:
                if self._log_enabled:
                    logger.warning(
                        "PC fuse skip: node(s) missing for (%d, %d)",
                        producer_id, consumer_id,
                    )
                continue

            fused_node = self._create_fused_node(graph, producer, consumer)
            try:
                graph.replace_nodes_with_fused(
                    [producer_id, consumer_id], fused_node
                )
                if self._log_enabled:
                    logger.info(
                        "PC fused (%d, %d) → node %d",
                        producer_id, consumer_id, fused_node.node_id,
                    )
            except Exception as exc:
                raise FusionError(
                    f"Failed to apply producer-consumer fusion for "
                    f"({producer_id}, {consumer_id}): {exc}"
                ) from exc

        # Post-fusion DAG integrity check
        post_adj = _build_adjacency(graph)
        cycle = detect_cycle(post_adj)
        if cycle is not None:
            raise FusionError(
                f"Producer-consumer fusion introduced a cycle in the KGIR "
                f"DAG involving nodes: {cycle}"
            )

        return graph

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_fused_node(
        graph: KGIRGraph,
        producer: KGIRNode,
        consumer: KGIRNode,
    ) -> KGIRNode:
        """Build a fused :class:`KGIRNode` merging *producer* and *consumer*."""
        fused_id = _next_free_node_id(graph)
        p_meta = producer.metadata
        c_meta = consumer.metadata

        # Merge tensor metadata
        merged_shapes: Dict[int, Tuple[int, ...]] = dict(p_meta.tensor_shapes)
        merged_shapes.update(c_meta.tensor_shapes)
        merged_strides: Dict[int, Tuple[int, ...]] = dict(p_meta.tensor_strides)
        merged_strides.update(c_meta.tensor_strides)
        merged_dtypes: Dict[int, str] = dict(p_meta.tensor_dtypes)
        merged_dtypes.update(c_meta.tensor_dtypes)

        # Merge memory access patterns
        merged_access: Dict[str, Any] = dict(p_meta.memory_access_patterns)
        merged_access.update(c_meta.memory_access_patterns)

        # Merge hardware target annotations
        merged_hw_ann: Dict[str, Any] = dict(p_meta.hardware_target_annotations)
        merged_hw_ann.update(c_meta.hardware_target_annotations)

        # Merge runtime performance annotations
        merged_perf_ann: Dict[str, Any] = dict(p_meta.runtime_performance_annotations)
        merged_perf_ann.update(c_meta.runtime_performance_annotations)

        # Compute unified grid as element-wise max
        unified_grid = compute_unified_grid(
            [p_meta.grid_dimensions, c_meta.grid_dimensions]
        )

        fused_metadata = NodeMetadata(
            memory_access_patterns=merged_access,
            tensor_shapes=merged_shapes,
            tensor_strides=merged_strides,
            tensor_dtypes=merged_dtypes,
            grid_dimensions=unified_grid,
            shared_memory_bytes=(
                p_meta.shared_memory_bytes + c_meta.shared_memory_bytes
            ),
            register_count=p_meta.register_count + c_meta.register_count,
            num_warps=max(p_meta.num_warps, c_meta.num_warps),
            hardware_target_annotations=merged_hw_ann,
            runtime_performance_annotations=merged_perf_ann,
        )

        fused_node = KGIRNode(
            node_id=fused_id,
            kernel_fn=(producer.kernel_fn, consumer.kernel_fn),
            metadata=fused_metadata,
        )
        fused_node.is_fused = True
        fused_node.fused_from = [producer.node_id, consumer.node_id]
        return fused_node


# ===================================================================
# SiblingFusionAnalyzer
# ===================================================================
class SiblingFusionAnalyzer:
    """Identifies independent kernels that can be merged into single launches.

    Sibling (horizontal) fusion criteria (all must hold):
      1. **Independence:** No data dependencies between the kernels (neither
         direct nor transitive).
      2. **Compatible grid geometries:** Grid dimensions can be unified or
         partitioned (via :func:`grids_compatible`).
      3. **Combined resources within limits:** Merged kernel's SMEM + register
         usage fits within per-SM/CU hardware budget.
      4. **Same device target:** All siblings must be targeting the same device.

    Parameters:
        cost_model: The :class:`AdaptiveCostModel` for benefit scoring.
        config: The :class:`FusionConfig` controlling analyzer behaviour.
        target: The :class:`HardwareProfile` for per-target resource checks.
    """

    def __init__(
        self,
        cost_model: AdaptiveCostModel,
        config: FusionConfig,
        target: HardwareProfile,
    ) -> None:
        self._cost_model = cost_model
        self._config = config
        self._target = target
        self._log_enabled: bool = config.log or _read_knob_fusion_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_candidates(self, graph: KGIRGraph) -> List[List[int]]:
        """Return groups of independent kernel IDs eligible for sibling fusion."""
        if not self._config.enable_sibling:
            if self._log_enabled:
                logger.info("SiblingFusionAnalyzer: disabled by config")
            return []

        adjacency = _build_adjacency(graph)
        all_ids = _get_all_node_ids(graph)

        # Filter to non-fused nodes
        eligible_ids: List[int] = []
        for nid in all_ids:
            node = graph.get_node(nid)
            if node is not None and not node.is_fused:
                eligible_ids.append(nid)

        if len(eligible_ids) < 2:
            return []

        # Build independence groups using a greedy approach:
        # Iterate in topological order; for each node, try to add it to an
        # existing group where it is independent of all current members.
        groups: List[List[int]] = []
        assigned: set[int] = set()

        for nid in eligible_ids:
            if nid in assigned:
                continue
            node = graph.get_node(nid)
            if node is None:
                continue

            placed = False
            for group in groups:
                # Check independence with every member of the group
                all_independent = True
                for member_id in group:
                    if not are_independent(adjacency, nid, member_id):
                        all_independent = False
                        break
                if not all_independent:
                    continue

                # Check grid compatibility with existing members
                member_nodes = [graph.get_node(mid) for mid in group]
                member_nodes = [n for n in member_nodes if n is not None]
                candidate_nodes = member_nodes + [node]
                if not self._check_grid_compat_list(candidate_nodes):
                    continue

                # Check combined resource budget
                if not self._check_resource_list(candidate_nodes, self._target):
                    continue

                group.append(nid)
                assigned.add(nid)
                placed = True
                break

            if not placed:
                # Start a new potential group
                groups.append([nid])
                assigned.add(nid)

        # Filter to groups with at least 2 members
        valid_groups = [g for g in groups if len(g) >= 2]

        # Respect max-fused-kernels budget
        max_fusions = self._config.max_fused_kernels
        trimmed: List[List[int]] = []
        total = 0
        for g in valid_groups:
            if total + len(g) > max_fusions:
                remaining = max_fusions - total
                if remaining >= 2:
                    trimmed.append(g[:remaining])
                break
            trimmed.append(g)
            total += len(g)

        if self._log_enabled:
            for g in trimmed:
                logger.info(
                    "Sibling candidate group: %s on %s",
                    g, self._target.arch_generation,
                )

        return trimmed

    def check_independence(
        self, kernel_ids: List[int], graph: KGIRGraph
    ) -> bool:
        """Return *True* if all kernels in *kernel_ids* are pairwise independent."""
        if len(kernel_ids) < 2:
            return True
        adjacency = _build_adjacency(graph)
        for i in range(len(kernel_ids)):
            for j in range(i + 1, len(kernel_ids)):
                if not are_independent(adjacency, kernel_ids[i], kernel_ids[j]):
                    return False
        return True

    def check_grid_compatibility(self, kernels: List[KGIRNode]) -> bool:
        """Return *True* if the grid geometries of *kernels* are compatible."""
        return self._check_grid_compat_list(kernels)

    def check_combined_resources(
        self, kernels: List[KGIRNode], target: HardwareProfile
    ) -> bool:
        """Return *True* if the merged resource usage of *kernels* fits *target*."""
        return self._check_resource_list(kernels, target)

    def fuse(
        self,
        graph: KGIRGraph,
        groups: List[List[int]],
    ) -> KGIRGraph:
        """Apply sibling fusion decisions to *graph* and return it.

        After all fusions, the graph is verified to remain acyclic via
        :func:`detect_cycle`.
        """
        if not groups:
            return graph

        for group in groups:
            if len(group) < 2:
                continue

            nodes = [graph.get_node(nid) for nid in group]
            nodes = [n for n in nodes if n is not None]
            if len(nodes) < 2:
                if self._log_enabled:
                    logger.warning(
                        "Sibling fuse skip: insufficient valid nodes in group %s",
                        group,
                    )
                continue

            fused_node = self._create_sibling_fused_node(graph, nodes)
            valid_ids = [n.node_id for n in nodes]
            try:
                graph.replace_nodes_with_fused(valid_ids, fused_node)
                if self._log_enabled:
                    logger.info(
                        "Sibling fused %s → node %d",
                        valid_ids, fused_node.node_id,
                    )
            except Exception as exc:
                raise FusionError(
                    f"Failed to apply sibling fusion for group {valid_ids}: {exc}"
                ) from exc

        # Post-fusion DAG integrity check
        post_adj = _build_adjacency(graph)
        cycle = detect_cycle(post_adj)
        if cycle is not None:
            raise FusionError(
                f"Sibling fusion introduced a cycle in the KGIR DAG "
                f"involving nodes: {cycle}"
            )

        return graph

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_grid_compat_list(kernels: List[KGIRNode]) -> bool:
        """Check pairwise grid compatibility for a list of kernels."""
        if len(kernels) < 2:
            return True
        grids = [k.metadata.grid_dimensions for k in kernels]
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                if not grids_compatible(grids[i], grids[j]):
                    return False
        return True

    @staticmethod
    def _check_resource_list(
        kernels: List[KGIRNode], target: HardwareProfile
    ) -> bool:
        """Check that combined resources of *kernels* fit within *target* limits."""
        total_smem = 0
        total_regs = 0
        total_warps = 0
        for k in kernels:
            res = k.get_resource_usage()
            total_smem += res["shared_memory_bytes"]
            total_regs += res["register_count"]
            total_warps += res.get("num_warps", 0)

        if total_smem > target.smem_per_sm_bytes:
            return False
        if total_regs > target.registers_per_sm:
            return False
        # Warp limit: total warps must not exceed SM capacity
        # A rough heuristic: registers_per_sm / 32 gives approx. max warps
        max_warps = target.registers_per_sm // max(target.warp_size, 1)
        if total_warps > 0 and max_warps > 0 and total_warps > max_warps:
            return False
        return True

    @staticmethod
    def _create_sibling_fused_node(
        graph: KGIRGraph, nodes: List[KGIRNode]
    ) -> KGIRNode:
        """Build a sibling-fused :class:`KGIRNode` from a list of nodes."""
        fused_id = _next_free_node_id(graph)

        # Merge all metadata
        merged_shapes: Dict[int, Tuple[int, ...]] = {}
        merged_strides: Dict[int, Tuple[int, ...]] = {}
        merged_dtypes: Dict[int, str] = {}
        merged_access: Dict[str, Any] = {}
        merged_hw_ann: Dict[str, Any] = {}
        merged_perf_ann: Dict[str, Any] = {}
        total_smem = 0
        total_regs = 0
        max_warps = 0
        grids: List[Tuple[int, ...]] = []
        kernel_fns: List[Any] = []

        for node in nodes:
            m = node.metadata
            merged_shapes.update(m.tensor_shapes)
            merged_strides.update(m.tensor_strides)
            merged_dtypes.update(m.tensor_dtypes)
            merged_access.update(m.memory_access_patterns)
            merged_hw_ann.update(m.hardware_target_annotations)
            merged_perf_ann.update(m.runtime_performance_annotations)
            total_smem += m.shared_memory_bytes
            total_regs += m.register_count
            max_warps = max(max_warps, m.num_warps)
            grids.append(m.grid_dimensions)
            kernel_fns.append(node.kernel_fn)

        unified_grid = compute_unified_grid(grids) if grids else (1, 1, 1)

        fused_metadata = NodeMetadata(
            memory_access_patterns=merged_access,
            tensor_shapes=merged_shapes,
            tensor_strides=merged_strides,
            tensor_dtypes=merged_dtypes,
            grid_dimensions=unified_grid,
            shared_memory_bytes=total_smem,
            register_count=total_regs,
            num_warps=max_warps,
            hardware_target_annotations=merged_hw_ann,
            runtime_performance_annotations=merged_perf_ann,
        )

        fused_node = KGIRNode(
            node_id=fused_id,
            kernel_fn=tuple(kernel_fns),
            metadata=fused_metadata,
        )
        fused_node.is_fused = True
        fused_node.fused_from = [n.node_id for n in nodes]
        return fused_node


# ===================================================================
# FusionEngine — Orchestrator
# ===================================================================
class FusionEngine:
    """Orchestrates fusion analysis: producer-consumer and sibling fusion
    with an adaptive cost model.

    The engine runs both :class:`ProducerConsumerAnalyzer` and
    :class:`SiblingFusionAnalyzer` on the KGIR graph, generates per-target
    :class:`FusionPlan` instances, and optionally applies the fusion mutations.

    Parameters:
        graph: The :class:`KGIRGraph` to analyze.
        config: The :class:`FusionConfig` controlling all fusion behaviour.
    """

    def __init__(self, graph: KGIRGraph, config: FusionConfig) -> None:
        self._graph = graph
        self._config = config
        self._cost_model = AdaptiveCostModel(config)
        self._log_enabled: bool = config.log or _read_knob_fusion_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self) -> FusionPlan:
        """Run both analyzers across all hardware targets and return a combined plan.

        If multiple hardware profiles are attached to the graph, the primary
        (first) profile is used.  Use :pymethod:`analyze_for_target` for
        explicit per-target analysis.
        """
        # Quick-exit: global disable
        if not self._config.enable or _read_knob_fusion_disable():
            if self._log_enabled:
                logger.info("FusionEngine: fusion disabled — returning empty plan")
            return FusionPlan()

        # Validate the graph before analysis
        try:
            self._graph.validate()
        except Exception as exc:
            raise FusionError(
                f"KGIR graph validation failed before fusion analysis: {exc}"
            ) from exc

        # Determine the primary target
        hw_profiles = self._graph.hardware_profiles
        if not hw_profiles:
            if self._log_enabled:
                logger.warning(
                    "FusionEngine: no hardware profiles on graph — returning empty plan"
                )
            return FusionPlan()

        # Use the first hardware profile as the primary target
        if isinstance(hw_profiles, dict):
            primary_target = next(iter(hw_profiles.values()))
        elif isinstance(hw_profiles, (list, tuple)):
            primary_target = hw_profiles[0]
        else:
            primary_target = hw_profiles

        return self.analyze_for_target(primary_target)

    def analyze_for_target(self, target: HardwareProfile) -> FusionPlan:
        """Run producer-consumer and sibling fusion analysis for a specific *target*.

        Returns a :class:`FusionPlan` with all fusion decisions for the given
        hardware profile.
        """
        # Quick-exit: global disable
        if not self._config.enable or _read_knob_fusion_disable():
            if self._log_enabled:
                logger.info(
                    "FusionEngine: fusion disabled for target %s",
                    target.arch_generation,
                )
            return FusionPlan(target=target)

        if self._log_enabled:
            logger.info(
                "FusionEngine: starting analysis for target %s "
                "(vendor=%s, smem=%d, regs=%d, warp_size=%d)",
                target.arch_generation,
                target.vendor,
                target.smem_per_sm_bytes,
                target.registers_per_sm,
                target.warp_size,
            )

        # --- Phase A: Producer-consumer fusion analysis ---
        pc_analyzer = ProducerConsumerAnalyzer(
            self._cost_model, self._config, target
        )
        pc_pairs = pc_analyzer.find_candidates(self._graph)

        # --- Phase B: Sibling fusion analysis ---
        sibling_analyzer = SiblingFusionAnalyzer(
            self._cost_model, self._config, target
        )
        sibling_groups = sibling_analyzer.find_candidates(self._graph)

        # --- Compute estimated speedup ---
        estimated_speedup = self._estimate_plan_speedup(
            pc_pairs, sibling_groups, target
        )

        plan = FusionPlan(
            producer_consumer_pairs=pc_pairs,
            sibling_groups=sibling_groups,
            target=target,
            estimated_speedup=estimated_speedup,
            cost_model_phase=self._cost_model.get_phase(),
        )

        if self._log_enabled:
            logger.info(
                "FusionEngine: analysis complete for %s — "
                "%d PC pairs, %d sibling groups, estimated speedup %.4f, "
                "cost model phase %d",
                target.arch_generation,
                len(pc_pairs),
                len(sibling_groups),
                estimated_speedup,
                plan.cost_model_phase,
            )

        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _estimate_plan_speedup(
        self,
        pc_pairs: List[Tuple[int, int]],
        sibling_groups: List[List[int]],
        target: HardwareProfile,
    ) -> float:
        """Compute an aggregate speedup estimate for the entire plan."""
        total_benefit = 0.0
        node_count = max(self._graph.node_count(), 1)

        # Benefit from producer-consumer fusions
        for prod_id, cons_id in pc_pairs:
            producer = self._graph.get_node(prod_id)
            consumer = self._graph.get_node(cons_id)
            if producer is not None and consumer is not None:
                benefit = self._cost_model.estimate_fusion_benefit(
                    producer, consumer, target
                )
                total_benefit += max(benefit, 0.0)

        # Benefit from sibling fusions: each group saves (N-1) launch overheads
        for group in sibling_groups:
            if len(group) >= 2:
                launches_saved = len(group) - 1
                cal = self._cost_model._load_calibration(target)
                overhead_us = cal.get("launch_overhead_us", _DEFAULT_LAUNCH_OVERHEAD_US)
                total_benefit += launches_saved * overhead_us * 1e-3

        # Convert benefit to a speedup ratio (1.0 = no change)
        # Rough model: benefit is a fraction of total kernel time
        if total_benefit > 0.0:
            # Assume an average kernel time of 1.0 ms for ratio computation
            baseline_ms = node_count * 1.0
            speedup = (baseline_ms + total_benefit) / baseline_ms
        else:
            speedup = 1.0

        return max(speedup, 1.0)
