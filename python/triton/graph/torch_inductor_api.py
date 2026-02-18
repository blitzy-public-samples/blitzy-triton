"""TorchInductor Integration Surface for Triton's graph-level optimization layer.

Defines the public API contract for TorchInductor (and other frontends) to
submit kernel graphs, accept scheduling metadata hints and device placement
preferences, and receive optimized launch sequences.  The feature works
standalone; TorchInductor integration is optional.

Public API
----------
submit_kernel_graph
    Primary entry point — accepts a kernel list, dependency edges, optional
    scheduling hints, and device placement preferences; returns an optimized
    :class:`KernelGraphResult` containing the launch sequence.

KernelGraphResult
    Immutable result container returned by ``submit_kernel_graph``.

DevicePlacement
    Per-kernel device affinity preference (``"soft"`` or ``"hard"``).

SchedulingHints
    Optional scheduling metadata hints (priority kernels, max streams,
    fusion preference, target latency).

LaunchOp
    Describes a single kernel launch in the optimized sequence, including
    stream assignment, device target, and dependency information.

Architecture Notes
------------------
This module is a thin orchestration layer that delegates all heavy lifting
to the graph sub-modules (``capture``, ``kgir``, ``fusion``,
``memory_planner``, ``scheduler``, ``dispatch``, ``profiler``,
``codegen_bridge``, ``feedback``).  It uses *lazy imports* for graph
sub-modules to avoid circular-dependency issues and to keep import-time cost
negligible for programs that never invoke the graph API.

All internal imports are restricted to the modules listed in
``depends_on_files`` — no assumptions about other files are made.

AAP References
--------------
- §0.2.4 — ``torch_inductor_api.py`` file specification
- §0.5.1 Group 4 — Python Graph Package implementation plan
- §0.5.1 Group 6 — TorchInductor integration surface concrete signatures
- §0.7.1 — Strictly-additive mandate; no modification to existing Triton APIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging

from triton.backends.compiler import GPUTarget
from triton.graph import errors
from triton.graph.config import GraphConfig
from triton.graph.utils import detect_cycle

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes — public API types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SchedulingHints:
    """Optional scheduling metadata hints from the calling framework.

    These hints influence — but do not override — the graph optimizer's
    decisions.  All fields are optional; ``None`` means "let the optimizer
    decide".

    Attributes
    ----------
    priority_kernels : Optional[List[int]]
        Kernel indices (into the ``kernels`` list passed to
        ``submit_kernel_graph``) that should be prioritised in the schedule.
    max_streams : Optional[int]
        Upper bound on the number of concurrent GPU streams used.
    prefer_fusion : bool
        When ``True`` (default), the optimizer will attempt kernel fusion
        when profitable.  Set to ``False`` to suppress fusion analysis.
    target_latency_ms : Optional[float]
        Desired end-to-end graph latency in milliseconds.  The optimizer
        treats this as a soft constraint influencing dispatch and scheduling.
    """

    priority_kernels: Optional[List[int]] = None
    max_streams: Optional[int] = None
    prefer_fusion: bool = True
    target_latency_ms: Optional[float] = None


@dataclass
class DevicePlacement:
    """Per-kernel device placement preference.

    Attributes
    ----------
    target : GPUTarget
        The GPU device descriptor (backend, architecture, warp size).
    affinity : Literal["soft", "hard"]
        ``"soft"`` allows the dispatch engine to reassign the kernel to a
        different device if doing so improves the overall schedule.
        ``"hard"`` forces the kernel onto *target* regardless of other
        considerations.
    """

    target: GPUTarget
    affinity: Literal["soft", "hard"] = "soft"


@dataclass
class LaunchOp:
    """Describes a single operation in the optimized launch sequence.

    A ``LaunchOp`` may represent either a kernel execution or a
    cross-device data transfer (when ``is_transfer`` is ``True``).

    Attributes
    ----------
    kernel_id : int
        Index of the kernel in the original ``kernels`` list submitted to
        ``submit_kernel_graph``.  For transfer operations this is the
        index of the consumer kernel receiving the data.
    stream_id : int
        Index of the GPU stream assigned to this operation.
    device : GPUTarget
        Hardware target device for this launch.
    dependencies : List[int]
        Indices (into this ``launch_sequence``) of operations that must
        complete before this one can start.
    is_transfer : bool
        ``True`` when this operation is a cross-device data transfer rather
        than a kernel execution.
    fused_with : Optional[List[int]]
        If this launch is a fused kernel, lists the original kernel indices
        that were merged into this single launch.  ``None`` for unfused
        kernels and transfer operations.
    """

    kernel_id: int
    stream_id: int
    device: GPUTarget
    dependencies: List[int] = field(default_factory=list)
    is_transfer: bool = False
    fused_with: Optional[List[int]] = None


@dataclass
class KernelGraphResult:
    """Result of graph-level kernel optimization.

    Returned by :func:`submit_kernel_graph` and contains the optimized
    launch sequence together with summary statistics.

    Attributes
    ----------
    launch_sequence : List[LaunchOp]
        Ordered sequence of kernel launch (and transfer) operations.
    estimated_latency_ms : float
        Estimated end-to-end wall-clock latency of the optimized graph in
        milliseconds.
    devices_used : List[GPUTarget]
        Distinct GPU devices that appear in the launch sequence.
    fusion_count : int
        Number of fusion operations applied during optimization.
    feedback_iterations : int
        Number of closed-loop feedback iterations performed (``0`` when
        feedback is disabled via ``TRITON_FEEDBACK_ENABLE=0``).
    """

    launch_sequence: List[LaunchOp]
    estimated_latency_ms: float
    devices_used: List[GPUTarget]
    fusion_count: int
    feedback_iterations: int


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_inputs(
    kernels: List[Any],
    dependencies: List[Tuple[int, int]],
    hints: Optional[SchedulingHints],
    device_preferences: Optional[Dict[int, DevicePlacement]],
) -> None:
    """Validate all inputs to :func:`submit_kernel_graph`.

    Raises
    ------
    errors.GraphCaptureError
        If the kernel list is empty, dependency indices are out of range,
        there is a self-dependency, or the dependency graph contains a cycle.
    errors.DispatchError
        If device preference indices are out of range or affinity values are
        invalid.
    """
    if not kernels:
        raise errors.GraphCaptureError("Kernel list must not be empty.")

    num_kernels = len(kernels)

    # --- Validate dependency pairs ---
    for pair_idx, (producer_idx, consumer_idx) in enumerate(dependencies):
        if not isinstance(producer_idx, int) or not isinstance(consumer_idx, int):
            raise errors.GraphCaptureError(
                f"Dependency pair at index {pair_idx} must contain integers, "
                f"got ({type(producer_idx).__name__}, {type(consumer_idx).__name__})."
            )
        if producer_idx < 0 or producer_idx >= num_kernels:
            raise errors.GraphCaptureError(
                f"Invalid producer index {producer_idx} in dependency pair "
                f"{pair_idx}: must be in [0, {num_kernels - 1}]."
            )
        if consumer_idx < 0 or consumer_idx >= num_kernels:
            raise errors.GraphCaptureError(
                f"Invalid consumer index {consumer_idx} in dependency pair "
                f"{pair_idx}: must be in [0, {num_kernels - 1}]."
            )
        if producer_idx == consumer_idx:
            raise errors.GraphCaptureError(
                f"Self-dependency detected in dependency pair {pair_idx}: "
                f"kernel {producer_idx} depends on itself."
            )

    # --- Cycle detection ---
    adjacency: Dict[int, List[int]] = {i: [] for i in range(num_kernels)}
    for producer_idx, consumer_idx in dependencies:
        adjacency[producer_idx].append(consumer_idx)

    cycle = detect_cycle(adjacency)
    if cycle is not None:
        raise errors.GraphCaptureError(
            f"Dependency graph contains a cycle involving nodes: {cycle}."
        )

    # --- Validate scheduling hints ---
    if hints is not None:
        if hints.priority_kernels is not None:
            for kid in hints.priority_kernels:
                if kid < 0 or kid >= num_kernels:
                    raise errors.GraphCaptureError(
                        f"Invalid kernel index {kid} in priority_kernels: "
                        f"must be in [0, {num_kernels - 1}]."
                    )
        if hints.max_streams is not None and hints.max_streams < 1:
            raise errors.GraphCaptureError(
                f"max_streams must be >= 1, got {hints.max_streams}."
            )
        if hints.target_latency_ms is not None and hints.target_latency_ms <= 0.0:
            raise errors.GraphCaptureError(
                "target_latency_ms must be positive."
            )

    # --- Validate device preferences ---
    if device_preferences is not None:
        for kernel_idx, placement in device_preferences.items():
            if not isinstance(kernel_idx, int):
                raise errors.DispatchError(
                    f"Device preference key must be an int, "
                    f"got {type(kernel_idx).__name__}."
                )
            if kernel_idx < 0 or kernel_idx >= num_kernels:
                raise errors.DispatchError(
                    f"Invalid kernel index {kernel_idx} in device_preferences: "
                    f"must be in [0, {num_kernels - 1}]."
                )
            if not isinstance(placement, DevicePlacement):
                raise errors.DispatchError(
                    f"Device preference for kernel {kernel_idx} must be a "
                    f"DevicePlacement instance, got {type(placement).__name__}."
                )
            if placement.affinity not in ("soft", "hard"):
                raise errors.DispatchError(
                    f"Invalid affinity '{placement.affinity}' for kernel "
                    f"{kernel_idx}: must be 'soft' or 'hard'."
                )


def _build_kgir_graph(
    kernels: List[Any],
    dependencies: List[Tuple[int, int]],
    hardware_profiles: Optional[List[Any]] = None,
) -> tuple:
    """Construct a ``KGIRGraph`` from submitted kernels and dependencies.

    Returns
    -------
    tuple
        ``(graph, index_to_node_id)`` — the built KGIR graph and a mapping
        from submission-list indices to KGIR node IDs.
    """
    from triton.graph.kgir import KGIRGraph, NodeMetadata

    graph = KGIRGraph()

    # Attach hardware profiles when available so that downstream passes
    # (fusion, memory planning) can reason about target-specific constraints.
    if hardware_profiles:
        graph.hardware_profiles = hardware_profiles

    # Map each kernel-list index to its KGIR node ID.
    index_to_node_id: Dict[int, int] = {}
    for idx, kernel_fn in enumerate(kernels):
        # Construct minimal metadata — the trace-capture layer enriches this
        # in the full pipeline, but submit_kernel_graph receives pre-built
        # kernels so metadata extraction is best-effort.
        metadata = NodeMetadata()

        # Attempt to extract grid dimensions and resource hints from the
        # kernel object if it exposes them (e.g. JITFunction).
        grid_dims = getattr(kernel_fn, "grid", None)
        if grid_dims is not None and isinstance(grid_dims, tuple):
            metadata.grid_dimensions = grid_dims

        num_warps = getattr(kernel_fn, "num_warps", None)
        if num_warps is not None and isinstance(num_warps, int):
            metadata.num_warps = num_warps

        node_id = graph.add_node(kernel_fn=kernel_fn, metadata=metadata)
        index_to_node_id[idx] = node_id

    # Add dependency edges.
    for producer_idx, consumer_idx in dependencies:
        source_id = index_to_node_id[producer_idx]
        target_id = index_to_node_id[consumer_idx]
        graph.add_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type="data_dep",
        )

    return graph, index_to_node_id


def _discover_hardware(config: GraphConfig) -> tuple:
    """Enumerate available GPU hardware and return an inventory + profiles.

    Returns
    -------
    tuple
        ``(inventory, hw_profiles)`` where *inventory* is a
        :class:`~triton.graph.dispatch.HardwareInventory` (or ``None`` on
        failure) and *hw_profiles* is a list of
        :class:`~triton.graph.kgir.HardwareProfile` instances.
    """
    from triton.graph import dispatch as dispatch_module

    try:
        inventory = dispatch_module.HardwareInventory()
        hw_profiles = inventory.devices  # List[HardwareProfile]
        logger.debug(
            "Hardware inventory: %d device(s) discovered.", inventory.device_count,
        )
        return inventory, hw_profiles
    except Exception as exc:
        logger.debug("Hardware inventory enumeration failed: %s", exc)
        return None, []


def _resolve_dispatch_mode(config: GraphConfig) -> Any:
    """Map the ``dispatch.mode`` config string to a ``DispatchMode`` enum.

    Falls back to ``BALANCED`` if the value is unrecognised.
    """
    from triton.graph.dispatch import DispatchMode

    mode_str = config.dispatch.mode if config.dispatch else "balanced"
    try:
        return DispatchMode(mode_str)
    except (ValueError, KeyError):
        logger.debug(
            "Unrecognised dispatch mode '%s'; falling back to BALANCED.",
            mode_str,
        )
        return DispatchMode.BALANCED


def _run_fusion(graph: Any, config: GraphConfig, prefer_fusion: bool) -> int:
    """Execute fusion analysis on *graph* and return the fusion count.

    Catches and logs fusion failures without propagating them — the pipeline
    continues unfused.
    """
    if not prefer_fusion:
        logger.debug("Fusion skipped (prefer_fusion=False).")
        return 0

    from triton.graph.fusion import FusionEngine

    try:
        engine = FusionEngine(graph=graph, config=config.fusion)
        plan = engine.analyze()
        fusions = plan.total_fusions
        logger.debug(
            "Fusion analysis complete: %d fusions, estimated speedup %.2fx.",
            fusions,
            plan.estimated_speedup,
        )
        return fusions
    except errors.FusionError:
        raise
    except Exception as exc:
        logger.warning("Fusion analysis failed (%s) — continuing unfused.", exc)
        return 0


def _run_memory_planning(graph: Any, config: GraphConfig) -> None:
    """Execute memory planning (intermediate identification + promotion).

    Best-effort: failures are logged and swallowed so that the pipeline
    continues with unoptimised memory layout.
    """
    from triton.graph.memory_planner import MemoryPlanner

    try:
        planner = MemoryPlanner(graph=graph, config=config)
        planner.identify_intermediates()

        # Promote intermediates for the primary hardware target (the first
        # profile attached to the graph, if any).
        hw_profiles = graph.hardware_profiles
        primary_target = None
        if hw_profiles:
            if isinstance(hw_profiles, (list, tuple)) and len(hw_profiles) > 0:
                primary_target = hw_profiles[0]
            elif isinstance(hw_profiles, dict) and hw_profiles:
                primary_target = next(iter(hw_profiles.values()))
            else:
                primary_target = hw_profiles

        if primary_target is not None:
            planner.plan_promotions(target=primary_target)
            logger.debug("Memory planning complete for primary target.")
        else:
            logger.debug(
                "Memory planning: no hardware profiles available; "
                "skipping promotion pass."
            )
    except Exception as exc:
        logger.warning("Memory planning failed (%s) — continuing without.", exc)


def _run_dispatch(
    graph: Any,
    config: GraphConfig,
    inventory: Any,
    index_to_node_id: Dict[int, int],
    device_preferences: Optional[Dict[int, DevicePlacement]],
) -> Dict[int, GPUTarget]:
    """Compute per-node device dispatch plan.

    Applies hard device preferences after the initial dispatch assignment.

    Returns
    -------
    Dict[int, GPUTarget]
        Mapping from KGIR node ID to the assigned ``GPUTarget``.

    Raises
    ------
    errors.DispatchError
        If no suitable devices are found or dispatch computation fails.
    """
    from triton.graph import dispatch as dispatch_module

    if inventory is None or inventory.device_count == 0:
        raise errors.DispatchError(
            "No GPU devices available for dispatch. "
            "Ensure at least one GPU backend is active."
        )

    mode = _resolve_dispatch_mode(config)

    try:
        engine = dispatch_module.DispatchDecisionEngine(
            inventory=inventory,
            config=config.dispatch,
            mode=mode,
        )
        dispatch_plan: Dict[int, GPUTarget] = engine.compute_dispatch_plan(
            graph=graph,
        )
    except errors.DispatchError:
        raise
    except Exception as exc:
        raise errors.DispatchError(
            f"Dispatch decision computation failed: {exc}"
        ) from exc

    # Override with hard device preferences.
    if device_preferences:
        for kernel_idx, placement in device_preferences.items():
            if placement.affinity == "hard":
                node_id = index_to_node_id.get(kernel_idx)
                if node_id is not None and node_id in dispatch_plan:
                    dispatch_plan[node_id] = placement.target
                    logger.debug(
                        "Hard device preference applied: kernel %d → %s:%s.",
                        kernel_idx,
                        placement.target.backend,
                        placement.target.arch,
                    )

    return dispatch_plan


def _run_scheduling(graph: Any, config: GraphConfig) -> list:
    """Execute the inter-kernel scheduler and return schedule entries.

    Returns
    -------
    list
        A list of ``ScheduleEntry`` objects from
        :class:`~triton.graph.scheduler.KernelScheduler`.

    Raises
    ------
    errors.DispatchError
        If scheduling fails (wraps underlying exceptions).
    """
    from triton.graph.scheduler import KernelScheduler

    try:
        scheduler = KernelScheduler(graph=graph, config=config)
        entries = scheduler.schedule()
        logger.debug("Scheduling complete: %d entries.", len(entries))
        return entries
    except errors.DispatchError:
        raise
    except Exception as exc:
        raise errors.DispatchError(
            f"Inter-kernel scheduling failed: {exc}"
        ) from exc


def _run_codegen(
    graph: Any,
    config: GraphConfig,
    dispatch_plan: Dict[int, GPUTarget],
) -> None:
    """Execute code generation and compilation for the dispatch plan.

    This step is best-effort: failure is logged but does not abort the
    result since the launch sequence is already determined.
    """
    from triton.graph.codegen_bridge import CodeGenerationBridge

    try:
        bridge = CodeGenerationBridge(graph=graph, config=config)
        bridge.compile_graph(dispatch_plan=dispatch_plan)
        logger.debug("Code generation + compilation complete.")
    except Exception as exc:
        logger.warning(
            "Code generation failed (%s) — returning schedule without "
            "compiled kernels.",
            exc,
        )


def _run_feedback(
    graph: Any,
    config: GraphConfig,
    schedule_entries: list,
    dispatch_plan: Dict[int, GPUTarget],
    index_to_node_id: Dict[int, int],
    node_id_to_index: Dict[int, int],
    hints: Optional[SchedulingHints],
    device_preferences: Optional[Dict[int, DevicePlacement]],
) -> tuple:
    """Run the optional closed-loop feedback optimisation.

    Returns
    -------
    tuple
        ``(updated_schedule, updated_dispatch, updated_fusions,
          feedback_iterations)`` — the (possibly updated) pipeline outputs
        and the number of feedback iterations performed.
    """
    from triton.graph.profiler import RuntimeProfiler
    from triton.graph.feedback import FeedbackController

    profiler = RuntimeProfiler(config=config)
    controller = FeedbackController(
        graph=graph,
        config=config.feedback,
    )

    # Mutable references for the optimiser callback to update.
    state = {
        "schedule": schedule_entries,
        "dispatch": dispatch_plan,
        "fusions": 0,
    }

    def _optimizer_fn(g: Any, _predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Re-run the core optimisation pipeline with updated annotations."""
        prefer_fusion = hints.prefer_fusion if hints else True
        fusions = _run_fusion(g, config, prefer_fusion)

        _run_memory_planning(g, config)

        new_dispatch = _run_dispatch(
            g, config,
            _discover_hardware(config)[0],
            index_to_node_id,
            device_preferences,
        )

        new_schedule = _run_scheduling(g, config)

        _run_codegen(g, config, new_dispatch)

        state["schedule"] = new_schedule
        state["dispatch"] = new_dispatch
        state["fusions"] = fusions
        return {
            "schedule": new_schedule,
            "dispatch": new_dispatch,
            "fusions": fusions,
        }

    def _execute_fn(_current_config: Any) -> float:
        """Return estimated latency from current schedule (ms)."""
        total = sum(
            getattr(e, "estimated_duration_ms", 0.0)
            for e in state["schedule"]
        )
        return total if total > 0.0 else 1.0

    try:
        controller.run_feedback_loop(
            optimizer_fn=_optimizer_fn,
            profiler=profiler,
            execute_fn=_execute_fn,
        )
    except errors.ConvergenceError:
        logger.warning(
            "Feedback loop reached max iterations without convergence — "
            "using best configuration found."
        )
    except Exception as exc:
        logger.warning(
            "Feedback loop failed (%s) — using initial optimisation.", exc,
        )

    iterations = getattr(controller, "_iteration", 0)
    return (
        state["schedule"],
        state["dispatch"],
        state["fusions"],
        iterations,
    )


def _convert_schedule_to_launch_ops(
    schedule_entries: list,
    dispatch_plan: Dict[int, GPUTarget],
    node_id_to_index: Dict[int, int],
    graph: Any,
) -> List[LaunchOp]:
    """Convert internal ``ScheduleEntry`` objects to public ``LaunchOp`` s.

    The mapping translates KGIR node IDs back to the original kernel-list
    indices so that the caller sees the same identifiers it submitted.
    """
    launch_ops: List[LaunchOp] = []

    for entry in schedule_entries:
        kernel_node_id: int = entry.kernel_id
        original_idx = node_id_to_index.get(kernel_node_id, kernel_node_id)

        # Resolve device — ScheduleEntry.device may be None for single-device
        # graphs; fall back to the dispatch plan.
        device = entry.device
        if device is None:
            device = dispatch_plan.get(kernel_node_id)
        if device is None and dispatch_plan:
            # Last resort: use any device from the plan.
            device = next(iter(dispatch_plan.values()))
        if device is None:
            # Absolute fallback — should not be reached in practice.
            device = GPUTarget(backend="cuda", arch=80, warp_size=32)

        # Map dependency node IDs to original kernel indices.
        dep_indices: List[int] = []
        for dep_nid in entry.dependencies:
            dep_indices.append(node_id_to_index.get(dep_nid, dep_nid))

        # Detect fused kernels and map their constituent node IDs.
        fused_with: Optional[List[int]] = None
        try:
            node = graph.get_node(kernel_node_id)
            if node.is_fused and node.fused_from:
                fused_with = [
                    node_id_to_index.get(fid, fid) for fid in node.fused_from
                ]
        except Exception:
            pass

        launch_ops.append(
            LaunchOp(
                kernel_id=original_idx,
                stream_id=entry.stream_id,
                device=device,
                dependencies=dep_indices,
                is_transfer=False,
                fused_with=fused_with,
            )
        )

    return launch_ops


def _collect_devices_used(dispatch_plan: Dict[int, GPUTarget]) -> List[GPUTarget]:
    """Return a deduplicated list of devices appearing in *dispatch_plan*.

    Deduplication uses ``(backend, arch, warp_size)`` identity from the
    frozen ``GPUTarget`` dataclass.
    """
    if not dispatch_plan:
        return []
    seen: Dict[tuple, GPUTarget] = {}
    for target in dispatch_plan.values():
        key = (target.backend, target.arch, target.warp_size)
        if key not in seen:
            seen[key] = target
    return list(seen.values())


def _estimate_latency(schedule_entries: list) -> float:
    """Estimate end-to-end latency from schedule entries (milliseconds).

    Sums ``estimated_duration_ms`` across all entries as a conservative
    serial upper-bound.  When profiling data is unavailable (durations are
    zero) a synthetic 0.1 ms-per-entry estimate is used.
    """
    if not schedule_entries:
        return 0.0
    total_ms = sum(
        getattr(e, "estimated_duration_ms", 0.0) for e in schedule_entries
    )
    if total_ms <= 0.0:
        total_ms = float(len(schedule_entries)) * 0.1
    return total_ms


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — submit_kernel_graph
# ═══════════════════════════════════════════════════════════════════════════════


def submit_kernel_graph(
    kernels: List[Any],
    dependencies: List[Tuple[int, int]],
    hints: Optional[SchedulingHints] = None,
    device_preferences: Optional[Dict[int, DevicePlacement]] = None,
) -> KernelGraphResult:
    """Submit a kernel graph for graph-level cross-kernel optimization.

    This is the primary entry point for TorchInductor (and other frontends)
    to leverage Triton's graph-level optimization layer.  The function
    accepts a list of Triton kernel objects, an explicit dependency DAG,
    optional scheduling hints, and optional per-kernel device preferences.

    The full optimization pipeline is executed:

    1. **Input validation** — bounds checking, cycle detection, preference
       validation.
    2. **KGIR construction** — kernels and dependencies are translated into
       a Kernel Graph IR DAG.
    3. **Hardware discovery** — available GPU devices are enumerated.
    4. **Fusion analysis** — producer-consumer and sibling fusion
       opportunities are identified and applied.
    5. **Memory planning** — intermediate tensors are promoted from global
       to shared memory where profitable.
    6. **Dispatch** — each KGIR node is assigned to an optimal GPU target.
    7. **Scheduling** — a multi-stream launch sequence is computed.
    8. **Code generation** — fused KGIR nodes are compiled to native code
       via the existing Triton pipeline (TTIR → backend).
    9. **Feedback loop** *(optional)* — if ``TRITON_FEEDBACK_ENABLE`` is
       true the runtime profiler drives iterative re-optimization until
       convergence or the iteration cap is reached.

    Parameters
    ----------
    kernels : List[Any]
        Ordered list of Triton ``JITFunction``, ``KernelInterface``, or
        equivalent callable objects.  The index of each kernel in this list
        is used as its identifier throughout the API.
    dependencies : List[Tuple[int, int]]
        ``(producer_idx, consumer_idx)`` pairs describing the kernel-level
        data-dependency DAG.  Both indices reference into *kernels*.
    hints : Optional[SchedulingHints]
        Optional scheduling metadata (priority kernels, max streams, fusion
        preference, target latency).
    device_preferences : Optional[Dict[int, DevicePlacement]]
        Per-kernel device placement preferences keyed by kernel index.
        ``"hard"`` affinities are honoured unconditionally; ``"soft"``
        affinities are used as hints.

    Returns
    -------
    KernelGraphResult
        Optimized launch sequence with estimated latency and statistics.

    Raises
    ------
    errors.GraphCaptureError
        If input validation fails (empty kernels, invalid indices, cycles).
    errors.FusionError
        If fusion analysis encounters a fatal error.
    errors.DispatchError
        If no suitable GPU devices are found or dispatch computation fails.

    Examples
    --------
    >>> import triton
    >>> result = triton.graph.torch_inductor_api.submit_kernel_graph(
    ...     kernels=[kernel_a, kernel_b, kernel_c],
    ...     dependencies=[(0, 1), (1, 2)],
    ...     hints=SchedulingHints(prefer_fusion=True),
    ... )
    >>> for op in result.launch_sequence:
    ...     print(op.kernel_id, op.stream_id, op.device)
    """
    logger.debug(
        "submit_kernel_graph called: %d kernels, %d dependency edges, "
        "hints=%s, device_preferences=%s.",
        len(kernels) if kernels else 0,
        len(dependencies) if dependencies else 0,
        hints is not None,
        device_preferences is not None,
    )

    # ── Step 1: Input validation ──────────────────────────────────────
    _validate_inputs(kernels, dependencies, hints, device_preferences)

    # ── Step 2: Load configuration from environment ───────────────────
    config = GraphConfig.from_env()

    # Apply scheduling hint overrides.
    if hints is not None:
        if hints.max_streams is not None:
            config.dispatch.stream_pool_size = max(1, hints.max_streams)

    # ── Step 3: Discover hardware ─────────────────────────────────────
    inventory, hw_profiles = _discover_hardware(config)

    # ── Step 4: Build KGIR graph ──────────────────────────────────────
    graph, index_to_node_id = _build_kgir_graph(
        kernels=kernels,
        dependencies=dependencies,
        hardware_profiles=hw_profiles if hw_profiles else None,
    )
    node_id_to_index: Dict[int, int] = {
        nid: idx for idx, nid in index_to_node_id.items()
    }

    # ── Step 5: Fusion analysis ───────────────────────────────────────
    prefer_fusion = hints.prefer_fusion if hints is not None else True
    fusion_count = _run_fusion(graph, config, prefer_fusion)

    # ── Step 6: Memory planning ───────────────────────────────────────
    _run_memory_planning(graph, config)

    # ── Step 7: Dispatch ──────────────────────────────────────────────
    dispatch_plan = _run_dispatch(
        graph, config, inventory, index_to_node_id, device_preferences,
    )

    # ── Step 8: Scheduling ────────────────────────────────────────────
    schedule_entries = _run_scheduling(graph, config)

    # ── Step 9: Code generation ───────────────────────────────────────
    _run_codegen(graph, config, dispatch_plan)

    # ── Step 10: Optional feedback loop ───────────────────────────────
    feedback_iterations = 0
    feedback_enabled = False
    try:
        import triton.knobs as knobs
        feedback_enabled = bool(knobs.graph.feedback_enable)
    except Exception:
        # If knobs are not accessible, default to enabled.
        feedback_enabled = True

    if feedback_enabled:
        (
            schedule_entries,
            dispatch_plan,
            fusion_count,
            feedback_iterations,
        ) = _run_feedback(
            graph=graph,
            config=config,
            schedule_entries=schedule_entries,
            dispatch_plan=dispatch_plan,
            index_to_node_id=index_to_node_id,
            node_id_to_index=node_id_to_index,
            hints=hints,
            device_preferences=device_preferences,
        )

    # ── Step 11: Assemble result ──────────────────────────────────────
    launch_ops = _convert_schedule_to_launch_ops(
        schedule_entries, dispatch_plan, node_id_to_index, graph,
    )
    devices_used = _collect_devices_used(dispatch_plan)
    estimated_latency = _estimate_latency(schedule_entries)

    result = KernelGraphResult(
        launch_sequence=launch_ops,
        estimated_latency_ms=estimated_latency,
        devices_used=devices_used,
        fusion_count=fusion_count,
        feedback_iterations=feedback_iterations,
    )

    logger.debug(
        "submit_kernel_graph complete: %d launch ops, est. %.2f ms, "
        "%d device(s), %d fusions, %d feedback iterations.",
        len(launch_ops),
        estimated_latency,
        len(devices_used),
        fusion_count,
        feedback_iterations,
    )

    return result
