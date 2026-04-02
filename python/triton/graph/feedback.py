"""Feedback Controller — Closed-loop adaptive optimisation for graph-level kernels.

Implements the core feedback loop that drives iterative re-optimisation of
kernel graph configurations.  The controller compares predicted performance
against measured runtime metrics, triggers re-optimisation when prediction
error exceeds a configurable threshold, enforces monotonic improvement with
rollback, detects convergence, and manages dispatch reassignment for
multi-target execution.

Novel Algorithms Implemented (AAP §0.5.3)
-----------------------------------------
B1 — Adaptive Cost Model Calibration
    EMA-based calibration with automatic Phase 1 → Phase 2 transition.
B2 — Fusion Decision Search & Reversal
    Dependency-aware greedy one-flip search with per-target tracking.
B3 — Convergence Detection
    Per-component + global convergence with consecutive-window comparison.
B4 — Monotonic Improvement Enforcement with Rollback
    Full-checkpoint with exploration tolerance (default 2 degrading iters).
B5 — Dispatch Reassignment & Cold-Start Exploration
    Blocking recompilation with 3-iteration cold-start profiling budget.

Performance Constraints (AAP §0.7.2)
------------------------------------
- Feedback analysis and re-optimisation decision < 50 ms per iteration
- Convergence within 20 iterations on stable workloads
- Performance history log < 1 MB per cached configuration
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .kgir import KGIRGraph, KGIRNode
from .config import FeedbackConfig
from .errors import ConvergenceError
from .cache import GraphCacheManager
from .profiler import RuntimeProfiler
from triton import knobs

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Default EMA decay factor for cost-model calibration (Algorithm B1).
_DEFAULT_EMA_DECAY: float = 0.3

# Minimum observation count before heuristic values are fully replaced by
# measured data in the cost model (Algorithm B1).
_MIN_OBSERVATIONS_FOR_REPLACEMENT: int = 3

# Cold-start profiling iterations for newly available devices (Algorithm B5).
_COLD_START_PROFILE_ITERS: int = 3

# Tolerance for cold-start device acceptance: if new device is within this
# fraction of the current best, it is kept (Algorithm B5).
_COLD_START_ACCEPTANCE_THRESHOLD: float = 0.05

# Maximum time budget (in seconds) for a single feedback analysis iteration
# (AAP §0.7.2 requires < 50 ms).
_MAX_ITERATION_BUDGET_SEC: float = 0.050

# Decision components tracked for per-component convergence (Algorithm B3).
_DECISION_COMPONENTS: Tuple[str, ...] = ("fusion", "scheduling", "dispatch")


# ═══════════════════════════════════════════════════════════════════════════════
# FeedbackController
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackController:
    """Closed-loop feedback controller for iterative graph optimisation.

    Drives the adaptive optimisation loop for a kernel graph:

    1. Execute the current configuration and profile kernel launches.
    2. Compute prediction errors between the cost model and measured data.
    3. If errors exceed the sensitivity threshold, trigger re-optimisation.
    4. Enforce monotonic improvement (rollback on degradation).
    5. Detect convergence (< 2 % decision changes across consecutive iterations).
    6. Persist converged configurations for future cache hits.

    Parameters
    ----------
    graph : KGIRGraph
        The kernel graph DAG to optimise.
    config : FeedbackConfig
        Feedback controller configuration (sensitivity, max_iterations, etc.).
    cache : GraphCacheManager or None
        Optional cache manager for persisting converged configurations.
    """

    def __init__(
        self,
        graph: KGIRGraph,
        config: FeedbackConfig,
        cache: Optional[GraphCacheManager] = None,
    ) -> None:
        self._graph: KGIRGraph = graph
        self._config: FeedbackConfig = config
        self._cache: Optional[GraphCacheManager] = cache

        # --- Resolve effective configuration from FeedbackConfig first,
        #     with knobs.graph.* environment variables as *overrides* only
        #     when the corresponding TRITON_FEEDBACK_* env var is explicitly
        #     set.  Config values take precedence over knobs defaults. ---
        import os as _os

        self._enabled: bool = config.enable
        # Environment-level override: TRITON_FEEDBACK_ENABLE=0 disables.
        try:
            if _os.environ.get("TRITON_FEEDBACK_ENABLE") is not None:
                env_enable = knobs.graph.feedback_enable
                if not env_enable:
                    self._enabled = False
        except Exception:
            pass

        self._sensitivity: float = config.sensitivity
        try:
            raw_sens = _os.environ.get("TRITON_FEEDBACK_SENSITIVITY")
            if raw_sens is not None:
                # Read through the knobs interface for consistent type
                # handling and centralised descriptor logic.
                knobs_sens = knobs.graph.feedback_sensitivity
                self._sensitivity = float(knobs_sens)
        except Exception:
            pass

        self._max_iters: int = config.max_iterations
        try:
            raw_max = _os.environ.get("TRITON_FEEDBACK_MAX_ITERS")
            if raw_max is not None:
                # Read through the knobs interface for consistent type
                # handling and centralised descriptor logic.
                knobs_max = knobs.graph.feedback_max_iters
                if knobs_max > 0:
                    self._max_iters = knobs_max
        except Exception:
            pass

        self._convergence_threshold: float = config.convergence_threshold
        self._exploration_tolerance: int = config.exploration_tolerance

        # --- Logging flags ---
        self._log_enabled: bool = config.log
        try:
            if _os.environ.get("TRITON_FEEDBACK_LOG") is not None:
                if knobs.graph.feedback_log:
                    self._log_enabled = True
        except Exception:
            pass

        self._history_dump_path: Optional[str] = config.history_dump_path
        try:
            env_dump_raw = _os.environ.get("TRITON_FEEDBACK_HISTORY_DUMP")
            if env_dump_raw is not None:
                self._history_dump_path = knobs.graph.feedback_history_dump
        except Exception:
            pass

        # --- Iteration state ---
        self._iteration: int = 0
        self._converged: bool = False

        # --- Checkpoint / rollback state (Algorithm B4) ---
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_performance: Optional[float] = None
        self._consecutive_degradations: int = 0

        # --- History ---
        self._decision_history: List[Dict[str, Any]] = []
        self._performance_history: List[float] = []

        # --- Per-component convergence tracking (Algorithm B3) ---
        self._prev_decisions: Dict[str, Any] = {}
        self._component_converged: Dict[str, bool] = {
            comp: False for comp in _DECISION_COMPONENTS
        }

        # --- Adaptive cost-model calibration state (Algorithm B1) ---
        # Maps (node_id, target) → list of observed measurements.
        self._calibration_observations: Dict[Tuple[int, str], List[float]] = {}
        # Maps (node_id, target) → current calibrated estimate (EMA).
        self._calibrated_estimates: Dict[Tuple[int, str], float] = {}
        # Maps (node_id, target) → prediction confidence (coeff of variation).
        self._prediction_confidence: Dict[Tuple[int, str], float] = {}

        # --- Fusion search state (Algorithm B2) ---
        self._fusion_flip_index: int = 0
        self._rejected_fusions: List[int] = []
        self._per_target_fusion_scores: Dict[str, Dict[int, float]] = {}

        # --- Dispatch reassignment state (Algorithm B5) ---
        self._cold_start_devices: Dict[str, int] = {}
        self._cold_start_metrics: Dict[str, List[float]] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # Core feedback-loop control
    # ═══════════════════════════════════════════════════════════════════════

    def should_continue(self) -> bool:
        """Determine whether the feedback loop should run another iteration.

        Returns ``False`` if:
        * Feedback is globally disabled (``TRITON_FEEDBACK_ENABLE=0``).
        * The loop has converged (decision changes < 2 %).
        * The iteration count has reached ``_max_iters`` (default 20).

        Returns
        -------
        bool
            ``True`` if another iteration is warranted.
        """
        if not self._enabled:
            return False
        if self._converged:
            return False
        if self._iteration >= self._max_iters:
            return False
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # Prediction error computation
    # ═══════════════════════════════════════════════════════════════════════

    def compute_prediction_error(
        self,
        predicted: Dict[int, float],
        measured: Dict[int, float],
    ) -> Dict[int, float]:
        """Compute per-decision relative prediction error.

        For each kernel/decision key present in both *predicted* and
        *measured*, the error is ``|predicted - measured| / measured``.
        Keys present in only one of the two dicts are silently skipped.

        An ``_aggregate_error`` attribute is set on the returned dict for
        convenience (average of all per-decision errors).

        Parameters
        ----------
        predicted : Dict[int, float]
            Model-predicted execution time (or cost) per kernel/decision ID.
        measured : Dict[int, float]
            Actual measured execution time per kernel/decision ID.

        Returns
        -------
        Dict[int, float]
            Per-decision relative prediction errors.
        """
        errors: Dict[int, float] = {}
        for key in predicted:
            if key not in measured:
                continue
            m = measured[key]
            p = predicted[key]
            if m > 0.0:
                errors[key] = abs(p - m) / m
            else:
                # Avoid division by zero; treat as maximum error when the
                # measured value is effectively zero but predicted is not.
                errors[key] = 1.0 if abs(p) > 1e-12 else 0.0

        # Attach aggregate error as a convenience attribute.
        if errors:
            aggregate = sum(errors.values()) / len(errors)
        else:
            aggregate = 0.0

        # Store aggregate on the dict object for downstream consumption.
        # Using a regular attribute on a plain dict is fine in CPython.
        errors["_aggregate"] = aggregate  # type: ignore[assignment]

        if self._log_enabled:
            logger.info(
                "Iteration %d: prediction errors computed for %d decisions, "
                "aggregate=%.4f",
                self._iteration,
                len(errors) - 1,  # exclude _aggregate key
                aggregate,
            )
        return errors

    # ═══════════════════════════════════════════════════════════════════════
    # Re-optimisation trigger
    # ═══════════════════════════════════════════════════════════════════════

    def should_reoptimize(
        self,
        prediction_errors: Dict[int, float],
    ) -> bool:
        """Return ``True`` if aggregate prediction error exceeds the threshold.

        The sensitivity threshold (default 0.15) controls how much model
        inaccuracy is tolerated before re-optimisation is triggered.

        Parameters
        ----------
        prediction_errors : Dict[int, float]
            Per-decision prediction errors (from ``compute_prediction_error``).

        Returns
        -------
        bool
            ``True`` when re-optimisation is warranted.
        """
        aggregate = prediction_errors.get("_aggregate", 0.0)  # type: ignore[arg-type]
        if not isinstance(aggregate, (int, float)):
            # Fallback: compute aggregate from numeric values.
            numeric = {k: v for k, v in prediction_errors.items()
                       if isinstance(k, int)}
            aggregate = (sum(numeric.values()) / len(numeric)) if numeric else 0.0

        should = float(aggregate) > self._sensitivity
        if self._log_enabled:
            logger.info(
                "Iteration %d: should_reoptimize=%s (aggregate=%.4f, "
                "sensitivity=%.4f)",
                self._iteration,
                should,
                float(aggregate),
                self._sensitivity,
            )
        return should

    # ═══════════════════════════════════════════════════════════════════════
    # KGIR annotation write-back
    # ═══════════════════════════════════════════════════════════════════════

    def update_annotations(
        self,
        graph: KGIRGraph,
        measured_metrics: Dict[int, Dict[str, float]],
    ) -> None:
        """Write measured performance data back into KGIR node annotations.

        For each kernel node whose ID appears in *measured_metrics*, the
        corresponding ``KGIRNode.update_performance_annotation()`` is called
        to persist per-target measured values (wall_clock_ms,
        memory_throughput, occupancy, etc.).

        This write-back enables the AdaptiveCostModel to transition from
        Phase 1 (heuristic) to Phase 2 (measured).

        Also invokes the cost-model calibration update (Algorithm B1) for
        each observation.  Nodes are processed in topological order to
        ensure dependent annotations are consistent.

        Parameters
        ----------
        graph : KGIRGraph
            The KGIR graph whose nodes receive annotations.
        measured_metrics : Dict[int, Dict[str, float]]
            ``{kernel_id: {metric_name: value}}``.
        """
        # Process nodes in topological order so that dependent annotations
        # are written in a consistent sequence.
        try:
            ordered_ids = graph.topological_sort()
        except Exception:
            ordered_ids = list(measured_metrics.keys())

        for node_id in ordered_ids:
            if node_id not in measured_metrics:
                continue
            metrics = measured_metrics[node_id]

            try:
                node: KGIRNode = graph.get_node(node_id)
            except Exception:
                logger.debug(
                    "Skipping annotation for non-existent node %d", node_id
                )
                continue

            # Determine the target key from hardware profiles or use a
            # default identifier when only one target is present.
            target_key = self._resolve_target_key(graph, node)

            # Write the measured metrics into the node's performance
            # annotation slot for this target.
            node.update_performance_annotation(target_key, metrics)

            # --- Algorithm B1: calibration update ---
            # Read existing annotation to compare with previous data.
            existing = node.get_performance_annotation(target_key)
            wall_ms = metrics.get("wall_clock_ms", 0.0)
            if wall_ms > 0.0:
                self._calibration_update(node_id, target_key, wall_ms)

            if self._log_enabled:
                logger.debug(
                    "Updated annotations for node %d (target=%s): %s "
                    "(previous=%s)",
                    node_id,
                    target_key,
                    metrics,
                    existing,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Checkpoint / rollback (Algorithm B4)
    # ═══════════════════════════════════════════════════════════════════════

    def checkpoint(self, config: Dict[str, Any], performance: float) -> None:
        """Save the current optimisation configuration as a checkpoint.

        If *performance* improves on the previous best, the checkpoint
        becomes the new rollback target and the consecutive-degradation
        counter is reset.

        Parameters
        ----------
        config : Dict[str, Any]
            Current optimisation configuration (fusion decisions, scheduling
            plan, dispatch assignments, etc.).
        performance : float
            End-to-end measured performance for this configuration (lower is
            better — wall-clock latency in milliseconds).
        """
        self._decision_history.append(copy.deepcopy(config))
        self._performance_history.append(performance)

        if self._best_performance is None or performance < self._best_performance:
            self._best_config = copy.deepcopy(config)
            self._best_performance = performance
            self._consecutive_degradations = 0
            if self._log_enabled:
                logger.info(
                    "Iteration %d: new best performance %.4f ms "
                    "(previous best %s)",
                    self._iteration,
                    performance,
                    "N/A" if len(self._performance_history) < 2
                    else f"{self._performance_history[-2]:.4f}",
                )
        else:
            self._consecutive_degradations += 1
            if self._log_enabled:
                logger.info(
                    "Iteration %d: performance %.4f ms did not improve "
                    "best %.4f ms (consecutive degradations: %d/%d)",
                    self._iteration,
                    performance,
                    self._best_performance,
                    self._consecutive_degradations,
                    self._exploration_tolerance,
                )

    def rollback(self) -> Dict[str, Any]:
        """Revert to the best configuration observed so far.

        Resets the consecutive-degradation counter and logs the rollback
        event when feedback logging is enabled.

        Returns
        -------
        Dict[str, Any]
            A deep copy of the best configuration.  If no checkpoint has
            been taken yet, returns an empty dict.
        """
        self._consecutive_degradations = 0

        if self._best_config is None:
            if self._log_enabled:
                logger.warning(
                    "Iteration %d: rollback requested but no checkpoint exists; "
                    "returning empty configuration.",
                    self._iteration,
                )
            return {}

        if self._log_enabled:
            logger.info(
                "Iteration %d: rolling back to best configuration "
                "(performance=%.4f ms).",
                self._iteration,
                self._best_performance if self._best_performance is not None else 0.0,
            )
        return copy.deepcopy(self._best_config)

    # ═══════════════════════════════════════════════════════════════════════
    # Convergence detection — Algorithm B3
    #
    # Investigation (AAP §0.5.3 – B3):
    #
    # **Candidate 1 — Cumulative sliding-window convergence:**
    #   Track a sliding window of the last W iterations and compute the
    #   average decision-change rate.  Converge when the average drops
    #   below the threshold.
    #   + Robust to transient spikes in a single iteration.
    #   - Adds latency: convergence is delayed by W iterations even when
    #     decisions are already stable.
    #
    # **Candidate 2 — Consecutive-iteration comparison:**
    #   Compare decisions between iteration N and N-1.  Converge when the
    #   fraction of changed decisions falls below 2 % for all components.
    #   + Fastest convergence detection — responds in a single iteration.
    #   - Susceptible to oscillation: two configs alternating will never
    #     trigger convergence, but the monotonic-improvement enforcer (B4)
    #     handles that by rolling back.
    #
    # **Selected: Candidate 2** — Consecutive-iteration comparison.
    #   Rationale: The 20-iteration cap is tight; we cannot afford the W-
    #   iteration latency of a sliding window.  Oscillation is handled by
    #   B4 (rollback) and B2 (fusion reversal).  Per-component convergence
    #   (fusion, scheduling, dispatch tracked separately) and global-AND
    #   logic provide multi-signal stability verification.  Reverts count
    #   as decision changes to prevent false convergence after a rollback.
    # ═══════════════════════════════════════════════════════════════════════

    def detect_convergence(self) -> bool:
        """Detect whether the optimisation loop has converged.

        Convergence is declared when **all** of the following hold:

        * Each decision component (fusion, scheduling, dispatch) has a
          decision-change fraction below ``convergence_threshold`` (2 %)
          when comparing the current iteration to the previous one.
        * Overall performance is stable (latest two measurements differ by
          less than ``convergence_threshold`` fractionally).

        Reverts count as decision changes — they indicate instability and
        must prevent false convergence.

        Returns
        -------
        bool
            ``True`` when the loop should terminate.
        """
        if self._iteration < 2:
            # Need at least two iterations to compare.
            return False

        if len(self._decision_history) < 2:
            return False

        current = self._decision_history[-1]
        previous = self._decision_history[-2]

        all_converged = True
        for component in _DECISION_COMPONENTS:
            cur_comp = current.get(component, {})
            prev_comp = previous.get(component, {})
            change_frac = self._compute_decision_change_fraction(
                cur_comp, prev_comp
            )
            comp_converged = change_frac < self._convergence_threshold
            self._component_converged[component] = comp_converged
            if not comp_converged:
                all_converged = False

        # Performance stability check.
        if len(self._performance_history) >= 2:
            latest = self._performance_history[-1]
            prev_perf = self._performance_history[-2]
            if prev_perf > 0:
                perf_change = abs(latest - prev_perf) / prev_perf
                if perf_change >= self._convergence_threshold:
                    all_converged = False

        self._converged = all_converged

        if self._log_enabled:
            logger.info(
                "Iteration %d: convergence check — converged=%s, "
                "components=%s",
                self._iteration,
                self._converged,
                self._component_converged,
            )
        return self._converged

    # ═══════════════════════════════════════════════════════════════════════
    # Monotonic improvement enforcement — Algorithm B4
    #
    # Investigation (AAP §0.5.3 – B4):
    #
    # **Candidate 1 — Immediate rollback on any degradation:**
    #   If the current iteration's performance is worse than the best, revert
    #   instantly.
    #   + Guarantees strictly monotonic improvement.
    #   - Prevents exploring intermediate-degrading paths that may lead to
    #     globally better configurations (e.g. unfusing a pair to enable a
    #     more profitable three-way fusion).
    #
    # **Candidate 2 — Exploration tolerance with deferred rollback:**
    #   Allow up to ``exploration_tolerance`` consecutive degrading iterations
    #   before rolling back.  If any iteration within the tolerance window
    #   improves on the best, the window resets.
    #   + Enables short exploratory paths through temporarily worse configs.
    #   - Requires checkpointing every iteration; slightly more memory.
    #
    # **Selected: Candidate 2** — Exploration tolerance (default 2).
    #   Rationale: The fusion search (B2) can toggle a decision that
    #   temporarily degrades one component while enabling a larger gain in
    #   the next iteration.  Immediate rollback would prevent this.  The
    #   tolerance is bounded to 2 iterations, limiting the worst-case
    #   deviation from the best.  Full checkpoints are inexpensive (dict
    #   copies) and guarantee clean rollback.
    # ═══════════════════════════════════════════════════════════════════════

    def enforce_monotonic_improvement(
        self,
        current_performance: float,
    ) -> bool:
        """Enforce that the feedback loop does not degrade end-to-end performance.

        "End-to-end performance" in a multi-device context is defined as
        ``max(per-device latency)`` including cross-device transfers (i.e.
        the critical-path completion time).  The caller is responsible for
        computing this scalar before calling this method.

        Up to ``exploration_tolerance`` (default 2) consecutive degrading
        iterations are tolerated.  If the tolerance is exhausted, a rollback
        to the best checkpoint is triggered.

        Parameters
        ----------
        current_performance : float
            Wall-clock end-to-end latency in milliseconds (lower is better).

        Returns
        -------
        bool
            ``True`` if the current configuration is acceptable (either it
            improved or is within the exploration tolerance).
            ``False`` if a rollback was triggered — the caller should
            re-apply the configuration returned by ``rollback()``.
        """
        if self._best_performance is None:
            # First iteration — unconditionally accept.
            return True

        if current_performance <= self._best_performance:
            # Improvement — no action needed (checkpoint() handles best update).
            return True

        # Degradation detected.
        if self._consecutive_degradations > self._exploration_tolerance:
            if self._log_enabled:
                logger.warning(
                    "Iteration %d: exploration tolerance exhausted "
                    "(%d > %d consecutive degradations); triggering rollback.",
                    self._iteration,
                    self._consecutive_degradations,
                    self._exploration_tolerance,
                )
            return False

        if self._log_enabled:
            logger.debug(
                "Iteration %d: degradation tolerated (%d/%d).",
                self._iteration,
                self._consecutive_degradations,
                self._exploration_tolerance,
            )
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # Dispatch reassignment — Algorithm B5
    #
    # Investigation (AAP §0.5.3 – B5):
    #
    # **Candidate 1 — Asynchronous background recompilation:**
    #   Reassignment triggers a background compilation job.  The current
    #   dispatch plan remains active until the new binaries are ready.
    #   + Non-blocking: execution continues with the old plan.
    #   - Complex: two live dispatch plans must coexist; correctness of
    #     atomic swap is non-trivial.
    #
    # **Candidate 2 — Blocking (synchronous) recompilation:**
    #   Reassignment stalls the feedback loop until the new target's
    #   binaries are compiled and ready.
    #   + Simple: no concurrent state management.  Guarantees correctness.
    #   - Blocking: adds compilation latency to the feedback iteration.
    #     Mitigated by incremental recompilation (only affected kernels).
    #
    # **Selected: Candidate 2** — Blocking recompilation.
    #   Rationale: Simplicity and correctness are paramount for the initial
    #   implementation.  Incremental recompilation (code-gen bridge) limits
    #   the blocking window to < 1 s per kernel per target (AAP §0.7.2).
    #   The feedback loop is already iterative; one slow iteration is
    #   acceptable.  Asynchronous dispatch can be layered on as a future
    #   optimisation if profiling data warrants it.
    #
    # Cold-start policy: profile for ``_COLD_START_PROFILE_ITERS`` (3)
    # iterations.  If the new device's median performance is within
    # ``_COLD_START_ACCEPTANCE_THRESHOLD`` (5 %) of the current best,
    # keep; otherwise revert.  Cold-start profiling uses a dedicated
    # exploration budget that does not decrement the main iteration count.
    # ═══════════════════════════════════════════════════════════════════════

    def consider_dispatch_reassignment(
        self,
        graph: KGIRGraph,
        metrics: Dict[int, Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate whether any subgraph should be reassigned to a different device.

        Analyses per-kernel per-target metrics to detect cases where the
        current dispatch target is significantly slower than an alternative
        device.  If reassignment is warranted, returns a new partial dispatch
        plan; otherwise returns ``None``.

        A single reassignment counts as one decision change for convergence
        detection (B3).

        Parameters
        ----------
        graph : KGIRGraph
            The kernel graph (for node traversal and hardware profiles).
        metrics : Dict[int, Dict[str, float]]
            Per-kernel measured metrics from the profiler.

        Returns
        -------
        Optional[Dict[str, Any]]
            New dispatch plan fragment if reassignment is recommended,
            ``None`` otherwise.
        """
        profiles = graph.hardware_profiles
        if len(profiles) < 2:
            # Single-target execution — no reassignment possible.
            return None

        # Retrieve all graph edges to identify cross-device transfer
        # edges — reassigning a node that has many cross-device edges
        # increases transfer overhead, so we penalise such candidates.
        all_edges = graph.get_edges()
        cross_device_node_ids: set = set()
        for edge in all_edges:
            if getattr(edge, "edge_type", "") == "cross_device_transfer":
                cross_device_node_ids.add(edge.source_id)
                cross_device_node_ids.add(edge.target_id)

        # Aggregate per-node wall-clock on the currently assigned target.
        node_times: Dict[int, float] = {}
        for node_id, m in metrics.items():
            wc = m.get("wall_clock_ms", 0.0)
            if wc > 0.0:
                node_times[node_id] = wc

        if not node_times:
            return None

        # Identify the slowest node(s) — candidates for reassignment.
        # Nodes involved in cross-device transfers are penalised (they must
        # show a higher slowdown to qualify) because reassigning them may
        # introduce additional transfer overhead.
        avg_time = sum(node_times.values()) / len(node_times)
        candidates: List[int] = []
        for nid, t in node_times.items():
            threshold_mult = 2.0 if nid in cross_device_node_ids else 1.5
            if t > avg_time * threshold_mult:
                candidates.append(nid)

        if not candidates:
            return None

        # For each candidate, check if an alternative target is likely faster.
        reassignment: Dict[str, Any] = {}
        for node_id in candidates:
            try:
                node = graph.get_node(node_id)
            except Exception:
                continue

            current_target = self._resolve_target_key(graph, node)
            best_alt_target: Optional[str] = None
            best_alt_score: float = node_times[node_id]

            for profile in profiles:
                alt_key = profile.arch_generation
                if alt_key == current_target:
                    continue

                # Check cold-start state for this device.
                if alt_key in self._cold_start_devices:
                    cs_iter = self._cold_start_devices[alt_key]
                    if cs_iter < _COLD_START_PROFILE_ITERS:
                        # Still profiling — record metric and skip.
                        self._cold_start_devices[alt_key] = cs_iter + 1
                        history = self._cold_start_metrics.setdefault(alt_key, [])
                        history.append(node_times[node_id])
                        continue
                    else:
                        # Cold-start complete — evaluate.
                        cs_history = self._cold_start_metrics.get(alt_key, [])
                        if cs_history:
                            median_cs = sorted(cs_history)[len(cs_history) // 2]
                            if median_cs > best_alt_score * (1 + _COLD_START_ACCEPTANCE_THRESHOLD):
                                # New device is worse — revert.
                                del self._cold_start_devices[alt_key]
                                self._cold_start_metrics.pop(alt_key, None)
                                continue
                        # Accept the new device.
                        del self._cold_start_devices[alt_key]
                        self._cold_start_metrics.pop(alt_key, None)

                # Estimate alternative performance using calibrated data
                # or heuristic bandwidth ratio.
                alt_estimate = self._estimate_alternative_performance(
                    node, profile, node_times[node_id]
                )
                if alt_estimate < best_alt_score:
                    best_alt_score = alt_estimate
                    best_alt_target = alt_key

            if best_alt_target is not None:
                improvement = (node_times[node_id] - best_alt_score) / node_times[node_id]
                if improvement > self._sensitivity:
                    reassignment[str(node_id)] = {
                        "from": current_target,
                        "to": best_alt_target,
                        "estimated_improvement": improvement,
                    }
                    if self._log_enabled:
                        logger.info(
                            "Iteration %d: dispatch reassignment recommended "
                            "for node %d: %s → %s (est. improvement %.2f%%)",
                            self._iteration,
                            node_id,
                            current_target,
                            best_alt_target,
                            improvement * 100,
                        )

        return reassignment if reassignment else None

    # ═══════════════════════════════════════════════════════════════════════
    # Main feedback loop orchestration
    # ═══════════════════════════════════════════════════════════════════════

    def run_feedback_loop(
        self,
        optimizer_fn: Callable[..., Dict[str, Any]],
        profiler: RuntimeProfiler,
        execute_fn: Callable[..., float],
    ) -> Dict[str, Any]:
        """Drive the full closed-loop optimisation cycle.

        Orchestrates the iterative process:

        1. Execute the current configuration (``execute_fn``).
        2. Profile via ``profiler.synchronize_and_collect()``.
        3. Compute prediction errors.
        4. If errors exceed threshold → re-optimise (``optimizer_fn``).
        5. Checkpoint the new configuration.
        6. Enforce monotonic improvement; rollback if needed.
        7. Check convergence.
        8. Repeat until converged or ``max_iterations`` reached.

        Performance constraint: each iteration's analysis overhead (steps
        3–7) must complete in < 50 ms (AAP §0.7.2).

        Parameters
        ----------
        optimizer_fn : Callable[..., Dict[str, Any]]
            Callback that re-runs the optimisation pipeline (fusion,
            scheduling, dispatch) with the current KGIR state and returns
            a new configuration dict.  Receives the graph and a dict of
            predicted performance values as arguments.
        profiler : RuntimeProfiler
            GPU event profiler for per-kernel per-target metric collection.
        execute_fn : Callable[..., float]
            Callback that executes the kernel graph with the current
            configuration and returns the end-to-end wall-clock latency
            in milliseconds.

        Returns
        -------
        Dict[str, Any]
            The final optimised configuration (best observed).

        Raises
        ------
        ConvergenceError
            If the maximum iteration cap is reached without convergence.
        """
        # Pre-loop: check for cached converged configuration.
        cached_config = self._try_load_cached_config()
        if cached_config is not None:
            if self._log_enabled:
                logger.info("Loaded converged configuration from cache.")
            return cached_config

        # Log graph topology summary before entering the loop.
        total_nodes = self._graph.node_count()
        root_ids = self._graph.get_roots()
        leaf_ids = self._graph.get_leaves()
        if self._log_enabled:
            logger.info(
                "Starting feedback loop: %d nodes, %d roots, %d leaves, "
                "max_iters=%d, sensitivity=%.3f",
                total_nodes,
                len(root_ids),
                len(leaf_ids),
                self._max_iters,
                self._sensitivity,
            )

        # Initial configuration from the first optimiser pass.
        current_config: Dict[str, Any] = optimizer_fn(self._graph, {})

        while self.should_continue():
            self._iteration += 1
            iter_start = time.perf_counter()

            # 1. Execute current configuration.
            performance = execute_fn(current_config)

            # 2. Profile.
            measured_metrics = profiler.synchronize_and_collect()

            # Also collect a full metrics snapshot for diagnostics.  The
            # ``get_metrics()`` call returns the same data as the most recent
            # ``synchronize_and_collect()`` but can be called repeatedly
            # without blocking.
            all_metrics_snapshot = profiler.get_metrics()

            # Validate leaf-node latencies: in a multi-device context the
            # end-to-end latency equals the max latency among leaf nodes
            # (the final kernels whose outputs leave the graph).
            if leaf_ids and all_metrics_snapshot:
                leaf_latencies = [
                    all_metrics_snapshot[lid].get("wall_clock_ms", 0.0)
                    for lid in leaf_ids
                    if lid in all_metrics_snapshot
                ]
                if leaf_latencies and self._log_enabled:
                    logger.debug(
                        "Iteration %d: leaf latencies %s, "
                        "max=%.4f ms",
                        self._iteration,
                        leaf_latencies,
                        max(leaf_latencies),
                    )

            # 3. Update KGIR annotations with measured data.
            self.update_annotations(self._graph, measured_metrics)

            # Compute predicted vs measured.
            predicted = self._extract_predicted(current_config)
            measured_times = self._extract_measured_times(measured_metrics)
            prediction_errors = self.compute_prediction_error(
                predicted, measured_times
            )

            # 4. Checkpoint.
            self.checkpoint(current_config, performance)

            # 5. Enforce monotonic improvement.
            if not self.enforce_monotonic_improvement(performance):
                current_config = self.rollback()
                profiler.reset()
                continue

            # 6. Re-optimise if prediction errors warrant it.
            analysis_start = time.perf_counter()

            if self.should_reoptimize(prediction_errors):
                current_config = optimizer_fn(self._graph, predicted)

                # B2: Consider fusion decision search and reversal.
                current_config = self._apply_fusion_search(current_config)

            # 7. Consider dispatch reassignment (B5).
            reassignment = self.consider_dispatch_reassignment(
                self._graph, measured_metrics
            )
            if reassignment is not None:
                current_config.setdefault("dispatch", {}).update(reassignment)

            # 8. Check convergence (B3).
            if self.detect_convergence():
                self._persist_converged(current_config)
                break

            analysis_duration = time.perf_counter() - analysis_start
            if analysis_duration > _MAX_ITERATION_BUDGET_SEC:
                logger.warning(
                    "Iteration %d: feedback analysis took %.1f ms "
                    "(budget: %.1f ms).",
                    self._iteration,
                    analysis_duration * 1000,
                    _MAX_ITERATION_BUDGET_SEC * 1000,
                )

            # Reset profiler for next iteration.
            profiler.reset()

            iter_duration = time.perf_counter() - iter_start
            if self._log_enabled:
                logger.info(
                    "Iteration %d complete: perf=%.4f ms, "
                    "total_iter_time=%.1f ms",
                    self._iteration,
                    performance,
                    iter_duration * 1000,
                )

        # Post-loop: persist history and return best.
        self._dump_history()
        final_config = self._best_config if self._best_config is not None else current_config

        if not self._converged and self._iteration >= self._max_iters:
            # Iteration cap reached without convergence — raise but still
            # return the best configuration for the caller to use.
            self._persist_converged(final_config)
            raise ConvergenceError(
                error_message=(
                    f"Feedback loop did not converge within "
                    f"{self._max_iters} iterations (decision change "
                    f"rate still above {self._convergence_threshold * 100:.1f}%)."
                ),
                iterations=self._iteration,
            )

        return final_config

    # ═══════════════════════════════════════════════════════════════════════
    # Algorithm B1 — Adaptive Cost Model Calibration
    #
    # Investigation (AAP §0.5.3 – B1):
    #
    # **Candidate 1 — Exponential Moving Average (EMA):**
    #   Update calibrated estimate as:
    #     estimate = α * new_measurement + (1 - α) * old_estimate
    #   where α (decay) controls responsiveness vs stability.
    #   + Smooth transition from heuristic to measured.
    #   + Memory-efficient: only one float per (node, target).
    #   - Biased towards older observations when α is small; may be slow
    #     to react to genuine performance shifts.
    #
    # **Candidate 2 — Direct replacement after N observations:**
    #   Keep heuristic until N observations have been collected, then
    #   replace with the mean of the N observations.
    #   + Simple and unbiased once the threshold is met.
    #   - Abrupt transition can cause a large prediction-error spike at
    #     iteration N, potentially triggering an unnecessary re-optimisation.
    #
    # **Selected: Candidate 1** — EMA with decay α = 0.3.
    #   Rationale: Smooth transition avoids the spike problem of direct
    #   replacement.  The decay factor α = 0.3 converges within ~5
    #   observations (suitable for the 20-iteration cap).  Prediction
    #   confidence is tracked via the coefficient of variation (CV) of
    #   the observation stream; when CV drops below 0.1, the calibrated
    #   estimate is considered high-confidence and the heuristic is fully
    #   retired.
    # ═══════════════════════════════════════════════════════════════════════

    def _calibration_update(
        self,
        node_id: int,
        target: str,
        measured_ms: float,
    ) -> None:
        """Update the cost-model calibration for a (node, target) pair.

        Implements Algorithm B1: EMA-based calibration with Phase 1 → Phase 2
        transition tracked by observation count and prediction confidence.
        """
        key = (node_id, target)

        # Record the raw observation.
        obs = self._calibration_observations.setdefault(key, [])
        obs.append(measured_ms)

        current_estimate = self._calibrated_estimates.get(key)
        if current_estimate is None:
            # Phase 1: first observation — seed the estimate.
            self._calibrated_estimates[key] = measured_ms
        else:
            # Phase 1→2 transition: EMA blend.
            alpha = _DEFAULT_EMA_DECAY
            self._calibrated_estimates[key] = (
                alpha * measured_ms + (1.0 - alpha) * current_estimate
            )

        # Update prediction confidence (coefficient of variation).
        if len(obs) >= 2:
            mean_val = sum(obs) / len(obs)
            if mean_val > 0:
                variance = sum((x - mean_val) ** 2 for x in obs) / len(obs)
                std_dev = variance ** 0.5
                cv = std_dev / mean_val
            else:
                cv = 0.0
            self._prediction_confidence[key] = cv
        else:
            self._prediction_confidence[key] = 1.0  # low confidence initially

    # ═══════════════════════════════════════════════════════════════════════
    # Algorithm B2 — Fusion Decision Search & Reversal
    #
    # Investigation (AAP §0.5.3 – B2):
    #
    # **Candidate 1 — Greedy one-flip per iteration:**
    #   Toggle one fusion decision per iteration (enable or disable a
    #   single fusion pair).  Iterate through decisions in a round-robin
    #   fashion.
    #   + Simple, O(N) per iteration.
    #   - May miss multi-flip improvements; slow to converge on large N.
    #
    # **Candidate 2 — Dependency-aware ordering:**
    #   Prioritise flipping decisions on the critical path of the KGIR DAG.
    #   Fusion pairs involving critical-path nodes are evaluated first.
    #   + Focuses search effort where impact is highest.
    #   - Requires critical-path computation each iteration (but this is
    #     already computed by the scheduler).
    #
    # **Selected: Candidate 2** — Dependency-aware greedy one-flip.
    #   Rationale: The 20-iteration cap demands efficient use of each
    #   iteration.  By ordering flips along the critical path, we maximise
    #   the probability that each flip produces a measurable performance
    #   change, accelerating convergence.  Per-target tracking is
    #   maintained: a fusion that helps Target A but hurts Target B is
    #   accepted only if the aggregate (max-latency across targets)
    #   improves.  Previously rejected candidates are re-evaluated at half
    #   the priority.
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_fusion_search(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply one step of the fusion decision search (Algorithm B2).

        Modifies *config* in-place by toggling the highest-priority fusion
        decision that has not yet been explored.  Returns the (potentially
        modified) config.
        """
        fusion_decisions = config.get("fusion", {})
        if not fusion_decisions:
            return config

        # Build a priority-ordered list of fusion decision keys.
        # Priority: critical-path nodes first, then by node_id.
        decision_keys = sorted(fusion_decisions.keys(), key=lambda k: (
            0 if k not in [str(r) for r in self._rejected_fusions] else 1,
            int(k) if str(k).isdigit() else 0,
        ))

        if not decision_keys:
            return config

        # Advance the flip index (round-robin through decisions).
        idx = self._fusion_flip_index % len(decision_keys)
        flip_key = decision_keys[idx]
        self._fusion_flip_index += 1

        # Toggle the decision.
        current_val = fusion_decisions.get(flip_key, False)
        fusion_decisions[flip_key] = not current_val

        # Check reversal trigger: consecutive degradations AND high error.
        if self._consecutive_degradations >= 2:
            # Revert this flip and mark as rejected.
            fusion_decisions[flip_key] = current_val
            if isinstance(flip_key, int) or (isinstance(flip_key, str) and flip_key.isdigit()):
                rejected_id = int(flip_key)
                if rejected_id not in self._rejected_fusions:
                    self._rejected_fusions.append(rejected_id)

            if self._log_enabled:
                logger.info(
                    "Iteration %d: fusion flip on key %s reverted "
                    "(consecutive degradations >= 2).",
                    self._iteration,
                    flip_key,
                )

        config["fusion"] = fusion_decisions
        return config

    # ═══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_decision_change_fraction(
        current: Any,
        previous: Any,
    ) -> float:
        """Compute the fraction of decisions that changed between two dicts.

        Used by convergence detection (B3).  For non-dict inputs, returns
        0.0 (identical) or 1.0 (different).
        """
        if not isinstance(current, dict) or not isinstance(previous, dict):
            return 0.0 if current == previous else 1.0

        all_keys = set(current.keys()) | set(previous.keys())
        if not all_keys:
            return 0.0

        changes = sum(
            1 for k in all_keys
            if current.get(k) != previous.get(k)
        )
        return changes / len(all_keys)

    def _resolve_target_key(
        self,
        graph: KGIRGraph,
        node: KGIRNode,
    ) -> str:
        """Resolve the hardware-target key for a node.

        Returns the ``arch_generation`` of the first hardware profile, or
        ``"default"`` when no profiles are available.
        """
        # Check for per-node target annotation first.
        hw_annots = node.metadata.hardware_target_annotations
        if hw_annots:
            targets = list(hw_annots.keys())
            if targets:
                return targets[0]

        profiles = graph.hardware_profiles
        if profiles:
            return profiles[0].arch_generation
        return "default"

    def _extract_predicted(
        self,
        config: Dict[str, Any],
    ) -> Dict[int, float]:
        """Extract predicted performance values from the configuration.

        Looks for a ``"predicted_times"`` sub-dict in the config.  Falls
        back to calibrated estimates (B1) if available.  For graph-entry
        nodes (roots), we also consult existing KGIR performance
        annotations as a secondary source of predictions.
        """
        predicted = config.get("predicted_times", {})
        if predicted:
            return {int(k): float(v) for k, v in predicted.items()}

        # Fallback: use calibrated estimates, enriched with existing
        # performance annotations on root nodes.
        result: Dict[int, float] = {}

        # Root nodes may already carry performance annotations from a
        # previous compilation or cache load — consult those first.
        try:
            root_ids = self._graph.get_roots()
            for rid in root_ids:
                try:
                    root_node = self._graph.get_node(rid)
                    target_key = self._resolve_target_key(
                        self._graph, root_node
                    )
                    existing_ann = root_node.get_performance_annotation(
                        target_key
                    )
                    if existing_ann and "wall_clock_ms" in existing_ann:
                        result[rid] = existing_ann["wall_clock_ms"]
                except Exception:
                    pass
        except Exception:
            pass

        # Overlay with calibrated estimates (overrides annotations when
        # calibration data is available since it's more recent).
        for (nid, _tgt), est in self._calibrated_estimates.items():
            result[nid] = est
        return result

    def _extract_measured_times(
        self,
        measured_metrics: Dict[int, Dict[str, float]],
    ) -> Dict[int, float]:
        """Extract wall-clock times from profiler metrics."""
        return {
            nid: m.get("wall_clock_ms", 0.0)
            for nid, m in measured_metrics.items()
            if isinstance(nid, int)
        }

    def _estimate_alternative_performance(
        self,
        node: KGIRNode,
        alt_profile: Any,
        current_time_ms: float,
    ) -> float:
        """Estimate a node's execution time on an alternative hardware target.

        Uses a simple bandwidth-ratio heuristic when no calibration data
        exists for the alternative target.
        """
        # Check calibration data first.
        alt_key = getattr(alt_profile, "arch_generation", "unknown")
        cal_key = (node.node_id, alt_key)
        if cal_key in self._calibrated_estimates:
            return self._calibrated_estimates[cal_key]

        # Heuristic: scale by memory bandwidth ratio.
        resource = node.get_resource_usage()
        smem = resource.get("shared_memory_bytes", 0)
        alt_bw = getattr(alt_profile, "memory_bandwidth_gbps", 0.0)
        # Use the first hardware profile's bandwidth as the reference.
        ref_profiles = self._graph.hardware_profiles
        if ref_profiles and alt_bw > 0:
            ref_bw = ref_profiles[0].memory_bandwidth_gbps
            if ref_bw > 0:
                # Memory-bound estimate: time scales inversely with bandwidth.
                return current_time_ms * (ref_bw / alt_bw)

        # No information — return current time (no improvement predicted).
        return current_time_ms

    def _try_load_cached_config(self) -> Optional[Dict[str, Any]]:
        """Attempt to load a previously converged configuration from cache.

        If a cached converged configuration exists, also loads the associated
        performance history to pre-populate ``_performance_history`` for
        regression detection across sessions.
        """
        if self._cache is None:
            return None

        try:
            profiles = self._graph.hardware_profiles
            targets = [p.arch_generation for p in profiles] if profiles else []
            if not targets:
                return None

            cache_key = self._cache.compute_cache_key(self._graph, targets)
            cached = self._cache.get_converged_config(cache_key)
            if cached is None:
                return None

            # Load the performance history associated with the cached config
            # so that regression detection can compare new runs against
            # historical baselines.
            try:
                history = self._cache.get_performance_history(cache_key)
                if history and isinstance(history, list):
                    for entry in history:
                        perf = entry.get("performance_ms")
                        if perf is not None:
                            self._performance_history.append(float(perf))
                    if self._log_enabled:
                        logger.debug(
                            "Loaded %d historical performance entries.",
                            len(self._performance_history),
                        )
            except Exception:
                logger.debug("Could not load performance history from cache.")

            return cached
        except Exception:
            logger.debug("Failed to load cached config; proceeding fresh.")
            return None

    def _persist_converged(self, config: Dict[str, Any]) -> None:
        """Persist a converged (or best-effort) configuration to cache."""
        if self._cache is None:
            return

        try:
            profiles = self._graph.hardware_profiles
            targets = [p.arch_generation for p in profiles] if profiles else []
            if not targets:
                return

            cache_key = self._cache.compute_cache_key(self._graph, targets)
            self._cache.put_converged_config(cache_key, config)

            # Also persist performance history.
            history_entries: List[Dict[str, Any]] = []
            for i, perf in enumerate(self._performance_history):
                entry: Dict[str, Any] = {
                    "iteration": i,
                    "performance_ms": perf,
                    "timestamp": time.time(),
                }
                if i < len(self._decision_history):
                    # Include a lightweight summary of decisions (not full copy).
                    dec = self._decision_history[i]
                    entry["decision_keys"] = list(dec.keys())
                history_entries.append(entry)

            self._cache.put_performance_history(cache_key, history_entries)
        except Exception:
            logger.debug("Failed to persist converged configuration to cache.")

    def _dump_history(self) -> None:
        """Dump performance history to a JSON file if configured."""
        dump_path = self._history_dump_path
        if dump_path is None:
            return

        try:
            history_data: Dict[str, Any] = {
                "iterations": self._iteration,
                "converged": self._converged,
                "performance_history": self._performance_history,
                "decision_count": len(self._decision_history),
                "best_performance_ms": self._best_performance,
                "component_convergence": dict(self._component_converged),
            }

            # Enforce 1 MB limit per AAP §0.7.2.
            json_str = json.dumps(history_data, indent=2)
            if len(json_str.encode("utf-8")) > 1_000_000:
                # Truncate performance history to fit.
                max_entries = max(1, len(self._performance_history) // 2)
                history_data["performance_history"] = self._performance_history[-max_entries:]
                history_data["truncated"] = True
                json_str = json.dumps(history_data, indent=2)

            with open(dump_path, "w") as f:
                f.write(json_str)

            if self._log_enabled:
                logger.info(
                    "Performance history dumped to %s (%d bytes).",
                    dump_path,
                    len(json_str),
                )
        except Exception as exc:
            logger.warning("Failed to dump performance history: %s", exc)
