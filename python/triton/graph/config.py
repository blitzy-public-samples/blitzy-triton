"""Configuration dataclasses for the Triton graph-level optimization layer.

Defines structured configuration for fusion analysis, feedback-loop control,
hardware-aware dispatch, and top-level graph optimization settings.  All
default values are aligned with the specification thresholds documented in
the AAP (§0.5.1, §0.7.2, §0.7.3) so that a bare ``GraphConfig()`` instance
is immediately usable with production-safe behaviour.

A ``GraphConfig.from_env()`` factory method reads overrides from Triton's
``knobs.graph`` environment-variable descriptors, allowing runtime
reconfiguration without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import json


# ---------------------------------------------------------------------------
# FusionConfig — controls the fusion analysis engine
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """Configuration for the fusion analysis engine.

    Attributes:
        enable: Master enable/disable switch for all fusion passes.
            Controlled at runtime by ``TRITON_FUSION_DISABLE`` (inverted
            logic: env var *disables* when truthy).
        threshold: Minimum estimated speedup ratio to accept a fusion
            decision.  Matches the ``TRITON_FUSION_THRESHOLD`` default
            of ``"0.10"`` (AAP §0.5.1).
        max_fused_kernels: Upper bound on the number of kernels that may
            be fused into a single launch.
        enable_producer_consumer: Enable producer-consumer fusion analysis.
        enable_sibling: Enable sibling (horizontal) fusion analysis.
        log: Emit detailed fusion-decision logging.  Controlled at runtime
            by ``TRITON_FUSION_LOG``.
    """

    enable: bool = True
    threshold: float = 0.10
    max_fused_kernels: int = 8
    enable_producer_consumer: bool = True
    enable_sibling: bool = True
    log: bool = False


# ---------------------------------------------------------------------------
# FeedbackConfig — controls the closed-loop feedback controller
# ---------------------------------------------------------------------------

@dataclass
class FeedbackConfig:
    """Configuration for the feedback controller and closed-loop optimisation.

    The feedback loop is **enabled by default** (closed-loop by default,
    per AAP §0.1.2).  Users may disable it via ``TRITON_FEEDBACK_ENABLE=0``
    for single-pass static optimisation.

    Attributes:
        enable: Master enable flag.  When ``False`` the optimiser runs a
            single static pass with no runtime profiling.
        sensitivity: Prediction-error threshold above which re-optimisation
            is triggered.  Default ``0.15`` (AAP §0.5.1).
        max_iterations: Hard cap on feedback iterations.  Default ``20``
            (AAP §0.7.3).  Enforced even on adversarial workloads.
        convergence_threshold: Fraction of decision changes below which the
            loop is considered converged.  Default ``0.02`` (< 2%, AAP §0.7.2).
        exploration_tolerance: Number of consecutive degrading iterations
            tolerated before automatic rollback to the best checkpoint.
        log: Emit per-iteration feedback logging.  Controlled by
            ``TRITON_FEEDBACK_LOG``.
        history_dump_path: Optional filesystem path for persisting
            performance-history JSON.  Controlled by
            ``TRITON_FEEDBACK_HISTORY_DUMP``.
    """

    enable: bool = True
    sensitivity: float = 0.15
    max_iterations: int = 20
    convergence_threshold: float = 0.02
    exploration_tolerance: int = 2
    log: bool = False
    history_dump_path: Optional[str] = None


# ---------------------------------------------------------------------------
# DispatchConfig — controls the hardware-aware dispatch layer
# ---------------------------------------------------------------------------

@dataclass
class DispatchConfig:
    """Configuration for the hardware-aware dispatch layer.

    Attributes:
        mode: One of ``"performance"``, ``"cost"``, or ``"balanced"``
            (default, AAP §0.5.1).  Selects the objective-weighting
            strategy used by the ``DispatchDecisionEngine``.
        max_devices: Optional upper bound on the number of devices used.
            ``None`` means *all available devices*.
        stream_pool_size: Bounded CUDA/HIP stream pool.  Must not exceed
            hardware limits (typically ≤ 128).
        granularity: ``"subgraph"`` (default, AAP §0.5.1) dispatches at
            KGIR subgraph granularity; ``"graph"`` dispatches the entire
            graph to a single target.
        targets: Optional allowlist of hardware-target identifiers.
            ``None`` permits all discovered devices.  Parsed from a
            comma-separated ``TRITON_DISPATCH_TARGETS`` string.
        cost_weights: Optional per-objective weight overrides as a mapping
            from objective name to float weight.  Parsed from the
            ``TRITON_DISPATCH_COST_WEIGHTS`` JSON string.
        latency_constraint_ms: Optional hard latency constraint in
            milliseconds.  Parsed from ``TRITON_DISPATCH_LATENCY_CONSTRAINT``.
        log: Emit dispatch-decision logging.  Controlled by
            ``TRITON_DISPATCH_LOG``.
    """

    mode: str = "balanced"
    max_devices: Optional[int] = None
    stream_pool_size: int = 8
    granularity: str = "subgraph"
    targets: Optional[List[str]] = None
    cost_weights: Optional[Dict[str, float]] = None
    latency_constraint_ms: Optional[float] = None
    log: bool = False


# ---------------------------------------------------------------------------
# GraphConfig — top-level configuration aggregating all sub-configs
# ---------------------------------------------------------------------------

@dataclass
class GraphConfig:
    """Top-level configuration for the graph-level optimisation layer.

    Composes ``FusionConfig``, ``FeedbackConfig`` and ``DispatchConfig``
    with graph-wide settings.  Use the ``from_env()`` class method to
    construct an instance whose values are driven by ``TRITON_*``
    environment variables via the ``triton.knobs.graph`` descriptor
    system.

    Attributes:
        fusion: Fusion-analysis sub-configuration.
        feedback: Feedback-controller sub-configuration.
        dispatch: Hardware-dispatch sub-configuration.
        dump_kgir: When ``True``, dump the KGIR IR for debugging.
            Controlled by ``TRITON_KGIR_DUMP``.
        max_graph_nodes: Maximum number of kernel nodes in a graph before
            the optimiser is bypassed (safety valve for pathological
            graphs).
    """

    fusion: FusionConfig = field(default_factory=FusionConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    dump_kgir: bool = False
    max_graph_nodes: int = 100

    # -----------------------------------------------------------------
    # Factory: construct from environment variables
    # -----------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GraphConfig:
        """Create a ``GraphConfig`` populated from environment variables.

        Values are read through ``triton.knobs.graph`` which maps each
        ``TRITON_KGIR_*``, ``TRITON_FUSION_*``, ``TRITON_FEEDBACK_*``
        and ``TRITON_DISPATCH_*`` environment variable to a typed
        descriptor.

        The import of ``triton.knobs`` is intentionally **lazy** to
        avoid circular-dependency issues at module-import time.

        Returns:
            A fully-populated ``GraphConfig`` instance.
        """

        # Lazy import to break potential circular dependency chains.
        from triton import knobs  # noqa: E402  # type: ignore[import]

        graph_knobs = knobs.graph

        # -- Fusion sub-config -------------------------------------------
        fusion = FusionConfig(
            enable=not graph_knobs.fusion_disable,
            threshold=_parse_float(graph_knobs.fusion_threshold, 0.10),
            log=bool(graph_knobs.fusion_log),
        )

        # -- Feedback sub-config -----------------------------------------
        feedback = FeedbackConfig(
            enable=bool(graph_knobs.feedback_enable),
            sensitivity=_parse_float(graph_knobs.feedback_sensitivity, 0.15),
            max_iterations=int(graph_knobs.feedback_max_iters),
            log=bool(graph_knobs.feedback_log),
            history_dump_path=_opt_str(graph_knobs.feedback_history_dump),
        )

        # -- Dispatch sub-config -----------------------------------------
        dispatch_targets = _parse_comma_separated(
            _opt_str(graph_knobs.dispatch_targets),
        )
        dispatch_cost_weights = _parse_json_dict(
            _opt_str(graph_knobs.dispatch_cost_weights),
        )
        dispatch_latency = _parse_opt_float(
            _opt_str(graph_knobs.dispatch_latency_constraint),
        )
        dispatch = DispatchConfig(
            mode=str(graph_knobs.dispatch_mode),
            log=bool(graph_knobs.dispatch_log),
            targets=dispatch_targets,
            cost_weights=dispatch_cost_weights,
            latency_constraint_ms=dispatch_latency,
            granularity=str(graph_knobs.dispatch_granularity),
        )

        # -- Top-level ---------------------------------------------------
        return cls(
            fusion=fusion,
            feedback=feedback,
            dispatch=dispatch,
            dump_kgir=bool(graph_knobs.kgir_dump),
        )


# ---------------------------------------------------------------------------
# Private parsing helpers (used exclusively by ``GraphConfig.from_env``)
# ---------------------------------------------------------------------------

def _parse_float(value: object, default: float) -> float:
    """Attempt to parse *value* as a ``float``, returning *default* on failure."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_opt_float(value: Optional[str]) -> Optional[float]:
    """Parse an optional string as ``float``, returning ``None`` when empty."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: object) -> Optional[str]:
    """Normalise a knob value to ``Optional[str]``.

    The Triton ``env_opt_str`` descriptor returns ``None`` when unset
    and a plain ``str`` when set.  Some descriptors (``env_str``) always
    return a string, which may be empty.
    """

    if value is None:
        return None
    s = str(value)
    return s if s else None


def _parse_comma_separated(value: Optional[str]) -> Optional[List[str]]:
    """Split a comma-separated string into a stripped list.

    Returns ``None`` when *value* is ``None`` or empty.
    """

    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_json_dict(value: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse a JSON string expected to be a ``{"key": float}`` mapping.

    Returns ``None`` when *value* is ``None``, empty, or not valid JSON.
    """

    if value is None or value.strip() == "":
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
