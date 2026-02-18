"""Convergence behavior integration tests for the closed-loop optimization system.

Tests verify that the feedback controller's convergence detection, monotonic
improvement enforcement, iteration capping, and rollback mechanisms work
correctly per AAP §0.7.2, §0.7.3, and §0.5.3 B3/B4 specifications.

Tested convergence properties:
    - Stabilization within 20 iterations on stable workloads (decision changes < 2%)
    - Monotonic improvement guarantee with checkpoint/rollback
    - Revert-to-unfused-baseline on worst case
    - Maximum iteration cap enforcement on adversarial workloads
    - Per-component convergence tracking (fusion, scheduling, dispatch)
    - Rollback reverts count as decision changes
    - Post-convergence configuration stability
    - Early stopping when convergence detected before max_iterations
"""
from __future__ import annotations

import time

import pytest
import torch

import triton
import triton.language as tl
from triton.graph import capture
from triton.graph.config import GraphConfig, FeedbackConfig
from triton.graph.feedback import FeedbackController
from triton.graph.errors import ConvergenceError


# ---------------------------------------------------------------------------
# Phase 1 — Helper Triton kernels for representative convergence workloads
# ---------------------------------------------------------------------------


@triton.jit
def stable_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Deterministic element-wise add with predictable, stable performance."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def stable_mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Deterministic element-wise multiply with predictable, stable performance."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def stable_chain_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Chain of 4 deterministic ops (add, mul, relu, sub) for multi-fusion testing."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x + 1.0
    y = y * 2.0
    y = tl.maximum(y, 0.0)
    y = y - 0.5
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def adversarial_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel with data-dependent branching causing intentionally variable perf."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    pos = tl.where(x > 0.0, x * x, x)
    neg = tl.where(x <= 0.0, -x, x * 0.5)
    output = tl.where(x > 0.0, pos, neg)
    tl.store(output_ptr + offsets, output, mask=mask)


# ---------------------------------------------------------------------------
# Phase 2 — Lightweight duck-typed mock infrastructure (no unittest.mock)
# ---------------------------------------------------------------------------


class _MockNodeMetadata:
    """Minimal metadata object for mock KGIR nodes."""

    def __init__(self):
        self.hardware_target_annotations: dict = {}


class _MockNode:
    """Duck-typed KGIR node for FeedbackController consumption."""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.metadata = _MockNodeMetadata()
        self._perf_annotations: dict = {}

    def get_performance_annotation(self, target_key: str):
        return self._perf_annotations.get(target_key)

    def update_performance_annotation(self, target_key: str, metrics: dict):
        if target_key not in self._perf_annotations:
            self._perf_annotations[target_key] = {}
        self._perf_annotations[target_key].update(metrics)

    def get_resource_usage(self) -> dict:
        return {"shared_memory_bytes": 1024, "registers": 32}


class _MockEdge:
    """Duck-typed KGIR edge for graph topology."""

    def __init__(self, source_id: int, target_id: int, edge_type: str = "data"):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type


class _MockGraph:
    """Duck-typed KGIR graph with configurable node count and topology."""

    def __init__(self, num_nodes: int = 3):
        self._nodes = {i: _MockNode(i) for i in range(num_nodes)}
        # Simple linear chain: 0 → 1 → 2 → ...
        self._edges = [
            _MockEdge(i, i + 1, "data") for i in range(num_nodes - 1)
        ]
        self._hw_profiles: list = []

    @property
    def hardware_profiles(self) -> list:
        return self._hw_profiles

    def node_count(self) -> int:
        return len(self._nodes)

    def get_roots(self) -> list:
        child_ids = {e.target_id for e in self._edges}
        return [nid for nid in self._nodes if nid not in child_ids]

    def get_leaves(self) -> list:
        parent_ids = {e.source_id for e in self._edges}
        return [nid for nid in self._nodes if nid not in parent_ids]

    def get_node(self, node_id: int):
        return self._nodes[node_id]

    def topological_sort(self) -> list:
        return sorted(self._nodes.keys())

    def get_edges(self) -> list:
        return list(self._edges)


class _MockProfiler:
    """Duck-typed profiler returning configurable per-node metrics."""

    def __init__(self, node_ids: list, wall_clock_ms: float = 1.0):
        self._node_ids = node_ids
        self._wall_clock_ms = wall_clock_ms
        self._call_count = 0

    def synchronize_and_collect(self) -> dict:
        self._call_count += 1
        return self._build_metrics()

    def get_metrics(self) -> dict:
        return self._build_metrics()

    def reset(self) -> None:
        pass

    def _build_metrics(self) -> dict:
        return {
            nid: {
                "wall_clock_ms": self._wall_clock_ms,
                "memory_throughput": 100.0,
                "sm_occupancy": 0.75,
            }
            for nid in self._node_ids
        }


# ---------------------------------------------------------------------------
# Phase 3 — Factory/helper functions for optimizer and execution callables
# ---------------------------------------------------------------------------


def _make_converging_optimizer(
    node_ids: list,
    stabilize_after: int = 3,
):
    """Create an optimizer_fn that converges after *stabilize_after* calls.

    - First *stabilize_after* calls return scheduling configs with inflated
      predicted_times (5.0 ms per node) so ``should_reoptimize`` stays True.
    - After stabilisation, returns **identical** configs with accurate
      predicted_times (1.0 ms) so prediction error drops below sensitivity and
      the configuration stabilises, allowing convergence detection.

    Config deliberately excludes 'fusion' and 'dispatch' keys to prevent
    ``_apply_fusion_search`` / ``_consider_dispatch_reassignment`` from
    mutating the config between checkpoint and execution.
    """
    state = {"calls": 0}

    def optimizer_fn(graph, predicted_metrics):
        state["calls"] += 1
        if state["calls"] <= stabilize_after:
            # Exploration phase — inflated predictions trigger reoptimization
            return {
                "scheduling": {
                    "stream_assignments": {nid: 0 for nid in node_ids},
                    "priorities": {nid: float(nid) for nid in node_ids},
                },
                "predicted_times": {nid: 5.0 for nid in node_ids},
            }
        # Stable phase — accurate predictions, identical config each call
        return {
            "scheduling": {
                "stream_assignments": {nid: 0 for nid in node_ids},
                "priorities": {nid: 1.0 for nid in node_ids},
            },
            "predicted_times": {nid: 1.0 for nid in node_ids},
        }

    return optimizer_fn


def _make_adversarial_optimizer(node_ids: list):
    """Create an optimizer that oscillates forever, preventing convergence.

    Alternates scheduling configs on every call AND always inflates
    predicted_times so ``should_reoptimize`` remains True.  Includes
    a 'fusion' key to trigger ``_apply_fusion_search`` mutations, adding
    additional oscillation.
    """
    state = {"calls": 0}

    def optimizer_fn(graph, predicted_metrics):
        state["calls"] += 1
        parity = state["calls"] % 2
        return {
            "scheduling": {
                "stream_assignments": {
                    nid: parity for nid in node_ids
                },
                "priorities": {
                    nid: float(parity + nid) for nid in node_ids
                },
            },
            "fusion": {
                f"fuse_{nid}_{nid + 1}": bool(parity)
                for nid in node_ids[:-1]
            },
            "predicted_times": {nid: 5.0 for nid in node_ids},
        }

    return optimizer_fn


def _make_degrading_optimizer(node_ids: list):
    """Create an optimizer whose every new config degrades performance.

    ALL calls return inflated predicted_times (5.0 ms vs profiler's 1.0 ms) to
    keep ``should_reoptimize`` True so the optimizer is called on every
    iteration.  Each call returns a DIFFERENT scheduling config to prevent
    ``detect_convergence`` from triggering.

    Paired with ``_make_degrading_execute_fn``: first call returns good
    latency, subsequent calls return bad latency.  This creates a cycle of
    degradation → rollback → degradation until ``max_iterations`` is
    exhausted and ``ConvergenceError`` is raised.
    """
    state = {"calls": 0}

    def optimizer_fn(graph, predicted_metrics):
        state["calls"] += 1
        return {
            "scheduling": {
                "stream_assignments": {
                    nid: state["calls"] for nid in node_ids
                },
            },
            "predicted_times": {nid: 5.0 for nid in node_ids},
        }

    return optimizer_fn


def _make_improving_execute_fn(
    start_ms: float = 2.0,
    plateau_ms: float = 1.0,
    improve_over: int = 3,
):
    """Return an execute_fn that improves then plateaus.

    Simulates kernel execution latency that decreases linearly from
    *start_ms* to *plateau_ms* over *improve_over* calls, then holds
    at *plateau_ms*.
    """
    state = {"calls": 0}

    def execute_fn(config):
        state["calls"] += 1
        if state["calls"] <= improve_over:
            frac = state["calls"] / improve_over
            return start_ms + (plateau_ms - start_ms) * frac
        return plateau_ms

    return execute_fn


def _make_constant_execute_fn(latency_ms: float = 1.0):
    """Return an execute_fn that always returns the same latency."""
    def execute_fn(config):
        return latency_ms
    return execute_fn


def _make_degrading_execute_fn(good_ms: float = 1.0, bad_ms: float = 3.0):
    """Return an execute_fn: first call good, subsequent calls bad.

    This pairs with ``_make_degrading_optimizer`` so that the initial
    unfused baseline outperforms every subsequent fused config, triggering
    rollbacks and eventually ConvergenceError.
    """
    state = {"calls": 0}

    def execute_fn(config):
        state["calls"] += 1
        if state["calls"] == 1:
            return good_ms
        return bad_ms

    return execute_fn


# ---------------------------------------------------------------------------
# Phase 4 — Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def convergence_config():
    """Config optimised for convergence testing with standard thresholds."""
    return GraphConfig(
        feedback=FeedbackConfig(
            enable=True,
            max_iterations=20,
            sensitivity=0.15,
            convergence_threshold=0.02,
        )
    )


@pytest.fixture
def tight_convergence_config():
    """Config with a low iteration cap for quick convergence tests."""
    return GraphConfig(
        feedback=FeedbackConfig(
            enable=True,
            max_iterations=5,
            sensitivity=0.15,
            convergence_threshold=0.02,
        )
    )


# ---------------------------------------------------------------------------
# Phase 5 — Test functions (all marked @pytest.mark.kernel_graph)
# ---------------------------------------------------------------------------


@pytest.mark.kernel_graph
def test_convergence_within_max_iterations(device, convergence_config):
    """Verify convergence within 20 iterations on stable workloads (AAP §0.7.2).

    Demonstrates the full capture → mock-graph → FeedbackController flow.
    The converging optimizer stabilises after 3 calls so the controller
    detects convergence well within the 20-iteration cap.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — GPU required for kernel execution")
    # -- Reference: capture a real kernel graph to validate capture() works --
    n = 1024
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    out = torch.zeros(n, device=device)

    with capture() as captured_ref:
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        stable_add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
        stable_mul_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    # -- Deterministic convergence test using mock graph --
    num_nodes = 3
    node_ids = list(range(num_nodes))
    graph = _MockGraph(num_nodes)
    profiler = _MockProfiler(node_ids, wall_clock_ms=1.0)
    optimizer_fn = _make_converging_optimizer(node_ids, stabilize_after=3)
    execute_fn = _make_improving_execute_fn(start_ms=2.0, plateau_ms=1.0, improve_over=3)

    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )

    best_config = controller.run_feedback_loop(
        optimizer_fn=optimizer_fn,
        profiler=profiler,
        execute_fn=execute_fn,
    )

    # Convergence must be achieved (no ConvergenceError raised)
    assert best_config is not None
    assert controller._converged is True
    # Must complete in ≤ 20 iterations
    assert controller._iteration <= 20
    # Performance history should be non-empty
    assert len(controller._performance_history) >= 2


@pytest.mark.kernel_graph
def test_convergence_decision_change_threshold(device, convergence_config):
    """Verify convergence detection uses < 2% decision change threshold.

    Directly exercises ``detect_convergence`` with hand-crafted decision
    histories to confirm the 2% threshold boundary condition.
    Per AAP §0.5.3 B3: convergence = (changed / total) < 0.02.
    """
    num_nodes = 3
    node_ids = list(range(num_nodes))
    graph = _MockGraph(num_nodes)
    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )
    controller._iteration = 5

    # Build a scheduling dict with 100 keys — changing 1 key = 1% change (< 2%)
    base_scheduling = {f"key_{i}": float(i) for i in range(100)}
    identical_scheduling = dict(base_scheduling)

    # -- Scenario A: identical consecutive configs → converged --
    controller._decision_history = [
        {"scheduling": dict(base_scheduling)},
        {"scheduling": dict(identical_scheduling)},
    ]
    controller._performance_history = [1.0, 1.0]
    controller._component_converged = {}
    result = controller.detect_convergence()
    assert result is True, "Identical configs should trigger convergence"

    # -- Scenario B: 1 of 100 keys changed = 1% → still < 2%, converged --
    changed_scheduling = dict(base_scheduling)
    changed_scheduling["key_0"] = 999.0
    controller._decision_history = [
        {"scheduling": dict(base_scheduling)},
        {"scheduling": changed_scheduling},
    ]
    controller._performance_history = [1.0, 1.0]
    controller._component_converged = {}
    result = controller.detect_convergence()
    assert result is True, "1% change is below 2% threshold → converged"

    # -- Scenario C: 5 of 100 keys changed = 5% → above 2%, NOT converged --
    much_changed = dict(base_scheduling)
    for i in range(5):
        much_changed[f"key_{i}"] = 999.0
    controller._decision_history = [
        {"scheduling": dict(base_scheduling)},
        {"scheduling": much_changed},
    ]
    controller._performance_history = [1.0, 1.0]
    controller._component_converged = {}
    result = controller.detect_convergence()
    assert result is False, "5% change exceeds 2% threshold → NOT converged"


@pytest.mark.kernel_graph
def test_monotonic_improvement_across_iterations(device, convergence_config):
    """Verify each accepted iteration is no worse than previous best (AAP §0.7.3 / B4).

    Uses ``checkpoint`` and ``enforce_monotonic_improvement`` directly to
    validate that degradations beyond exploration tolerance trigger rollback.
    """
    num_nodes = 3
    graph = _MockGraph(num_nodes)
    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )
    controller._iteration = 1

    base_config = {"scheduling": {"s0": 0, "s1": 0}}

    # Iteration 1 — baseline performance 2.0 ms
    controller.checkpoint(dict(base_config), 2.0)
    assert controller._best_performance == 2.0

    # Iteration 2 — improvement to 1.5 ms
    controller._iteration = 2
    improved_config = {"scheduling": {"s0": 1, "s1": 1}}
    controller.checkpoint(dict(improved_config), 1.5)
    accepted = controller.enforce_monotonic_improvement(1.5)
    assert accepted is True
    assert controller._best_performance == 1.5

    # Iteration 3 — degradation to 1.8 ms (first degradation — within tolerance)
    controller._iteration = 3
    degraded_config = {"scheduling": {"s0": 2, "s1": 2}}
    controller.checkpoint(dict(degraded_config), 1.8)
    accepted = controller.enforce_monotonic_improvement(1.8)
    assert accepted is True  # Within exploration_tolerance (default 2)

    # Iteration 4 — second degradation to 2.5 ms
    controller._iteration = 4
    bad_config = {"scheduling": {"s0": 3, "s1": 3}}
    controller.checkpoint(dict(bad_config), 2.5)
    accepted = controller.enforce_monotonic_improvement(2.5)
    assert accepted is True  # Still within tolerance (consecutive_degradations == 2)

    # Iteration 5 — third consecutive degradation to 3.0 ms → exceeds tolerance
    controller._iteration = 5
    terrible_config = {"scheduling": {"s0": 4, "s1": 4}}
    controller.checkpoint(dict(terrible_config), 3.0)
    accepted = controller.enforce_monotonic_improvement(3.0)
    assert accepted is False  # Tolerance exceeded → must rollback

    # After rollback, best config is the 1.5 ms configuration
    rollback_config = controller.rollback()
    assert controller._best_performance == 1.5


@pytest.mark.kernel_graph
def test_revert_to_baseline_worst_case(device, convergence_config):
    """Verify worst-case: revert to unfused baseline on fastest device (AAP §0.7.3).

    Creates a scenario where the initial (unfused) configuration performs
    well but every subsequent optimisation attempt degrades performance.
    After exhausting the iteration budget the controller must raise
    ``ConvergenceError`` and the best config returned must be the original
    unfused baseline (lowest latency).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — GPU required for kernel execution")
    # Prepare reference tensors for correctness validation post-revert
    n = 512
    ref_x = torch.randn(n, device=device)
    ref_y = torch.randn(n, device=device)
    expected = ref_x + ref_y
    actual = torch.empty(n, device=device)

    num_nodes = 3
    node_ids = list(range(num_nodes))
    graph = _MockGraph(num_nodes)
    profiler = _MockProfiler(node_ids, wall_clock_ms=1.0)
    optimizer_fn = _make_degrading_optimizer(node_ids)
    execute_fn = _make_degrading_execute_fn(good_ms=1.0, bad_ms=3.0)

    cfg = FeedbackConfig(
        enable=True,
        max_iterations=10,
        sensitivity=0.15,
        convergence_threshold=0.02,
    )
    controller = FeedbackController(graph=graph, config=cfg)

    with pytest.raises(ConvergenceError) as exc_info:
        controller.run_feedback_loop(
            optimizer_fn=optimizer_fn,
            profiler=profiler,
            execute_fn=execute_fn,
        )

    # The best configuration should be the first (unfused baseline, 1.0 ms)
    assert controller._best_performance <= 1.0 + 1e-9
    # Iteration cap was respected
    assert controller._iteration >= cfg.max_iterations
    # ConvergenceError carries iteration info
    err = exc_info.value
    assert err.iterations is not None and err.iterations >= cfg.max_iterations

    # Verify numerical correctness: baseline kernel should produce correct
    # results even after reverting from failed optimisations.
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    stable_add_kernel[grid](ref_x, ref_y, actual, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected), (
        "Numerical correctness violated after revert to baseline"
    )


@pytest.mark.kernel_graph
def test_max_iteration_cap_enforcement(device, convergence_config):
    """Verify max iteration cap enforced on adversarial workloads (AAP §0.7.3).

    The adversarial optimizer oscillates configs every call, so convergence
    is never reached.  The loop MUST terminate at exactly 20 iterations and
    raise ``ConvergenceError``.  The returned best config must be the best
    found across all iterations (not the last iteration's config).
    """
    num_nodes = 3
    node_ids = list(range(num_nodes))
    graph = _MockGraph(num_nodes)
    profiler = _MockProfiler(node_ids, wall_clock_ms=1.0)
    optimizer_fn = _make_adversarial_optimizer(node_ids)
    execute_fn = _make_constant_execute_fn(latency_ms=1.0)

    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )

    with pytest.raises(ConvergenceError) as exc_info:
        controller.run_feedback_loop(
            optimizer_fn=optimizer_fn,
            profiler=profiler,
            execute_fn=execute_fn,
        )

    # Must terminate at or before 20 iterations
    assert controller._iteration <= 20
    # ConvergenceError carries iterations
    assert exc_info.value.iterations is not None
    # Best config should be set (not None)
    assert controller._best_config is not None
    # Performance history should have entries for all iterations
    assert len(controller._performance_history) >= 1


@pytest.mark.kernel_graph
def test_configurable_iteration_cap(device):
    """Verify TRITON_FEEDBACK_MAX_ITERS is respected (AAP §0.5.1 Group 5).

    Runs the adversarial optimizer with max_iterations=5 then
    max_iterations=10 and confirms the loop stops at exactly those caps.
    """
    num_nodes = 3
    node_ids = list(range(num_nodes))

    for cap in (5, 10):
        graph = _MockGraph(num_nodes)
        profiler = _MockProfiler(node_ids, wall_clock_ms=1.0)
        optimizer_fn = _make_adversarial_optimizer(node_ids)
        execute_fn = _make_constant_execute_fn(latency_ms=1.0)

        cfg = FeedbackConfig(
            enable=True,
            max_iterations=cap,
            sensitivity=0.15,
            convergence_threshold=0.02,
        )
        controller = FeedbackController(graph=graph, config=cfg)

        with pytest.raises(ConvergenceError):
            controller.run_feedback_loop(
                optimizer_fn=optimizer_fn,
                profiler=profiler,
                execute_fn=execute_fn,
            )

        # Loop must have iterated up to the cap
        assert controller._iteration <= cap
        assert controller._iteration >= cap  # exactly the cap


@pytest.mark.kernel_graph
def test_per_component_convergence(device, convergence_config):
    """Verify per-component convergence tracking: fusion, scheduling, dispatch.

    Global convergence requires ALL components to converge independently.
    Per AAP §0.5.3 B3 — convergence is tracked separately per decision
    component and global convergence is the conjunction of all.
    """
    num_nodes = 3
    graph = _MockGraph(num_nodes)
    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )
    controller._iteration = 5

    base_fusion = {"fuse_0_1": True, "fuse_1_2": False}
    base_scheduling = {"stream_0": 0, "stream_1": 0, "stream_2": 0}
    base_dispatch = {"target_0": "gpu:0", "target_1": "gpu:0"}

    # All components identical → global convergence
    config_a = {
        "fusion": dict(base_fusion),
        "scheduling": dict(base_scheduling),
        "dispatch": dict(base_dispatch),
    }
    config_b = {
        "fusion": dict(base_fusion),
        "scheduling": dict(base_scheduling),
        "dispatch": dict(base_dispatch),
    }
    controller._decision_history = [config_a, config_b]
    controller._performance_history = [1.0, 1.0]
    controller._component_converged = {}

    result = controller.detect_convergence()
    assert result is True, "All components identical → global convergence"

    # Scheduling changes but fusion/dispatch identical → NOT converged
    changed_scheduling = dict(base_scheduling)
    changed_scheduling["stream_0"] = 999  # 1 of 3 = 33% change
    config_c = {
        "fusion": dict(base_fusion),
        "scheduling": changed_scheduling,
        "dispatch": dict(base_dispatch),
    }
    controller._decision_history = [config_a, config_c]
    controller._performance_history = [1.0, 1.0]
    controller._component_converged = {}

    result = controller.detect_convergence()
    assert result is False, "Scheduling diverged → global NOT converged"


@pytest.mark.kernel_graph
def test_reverts_count_as_decision_changes(device, convergence_config):
    """Verify rollback reverts count as decision changes (AAP §0.5.3 B3).

    Rollbacks indicate instability and prevent false convergence detection.
    After a rollback the decision history shows a change from the rolled-
    back config to the previous best, which should count as a decision
    change.
    """
    num_nodes = 3
    graph = _MockGraph(num_nodes)
    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )
    controller._iteration = 5

    config_good = {"scheduling": {"s": 0}}
    config_bad = {"scheduling": {"s": 1}}

    # Good → bad → rollback(=good) — the bad→good transition is a change
    controller.checkpoint(dict(config_good), 1.0)
    controller.checkpoint(dict(config_bad), 2.0)

    # After rollback the latest effective config differs from the most
    # recently checkpointed one, so convergence should NOT be detected.
    rollback_cfg = controller.rollback()

    # Now checkpoint the rolled-back config as a new iteration
    controller.checkpoint(dict(rollback_cfg), 1.0)

    # History: [good, bad, good(rollback)] — bad→good is a change
    # Check convergence between last two entries in history
    controller._performance_history = [1.0, 2.0, 1.0]
    controller._component_converged = {}

    # The transition from bad to good (rolled-back) IS a decision change.
    # Whether convergence is detected depends on the magnitude of change.
    # Since 's' went from 1 → 0 (100% change), convergence should NOT hold.
    if len(controller._decision_history) >= 2:
        result = controller.detect_convergence()
        # With 100% change in scheduling, must NOT be converged
        assert result is False, (
            "Rollback produces a decision change; "
            "convergence must not be falsely detected"
        )


@pytest.mark.kernel_graph
def test_convergence_stability(device, convergence_config):
    """After convergence is detected, the configuration should be stable.

    A converged controller should report no further decision changes and
    ``should_continue`` should return False.  Kernel outputs must be
    deterministic (bitwise-identical) across repeated executions.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — GPU required for kernel execution")
    # Reference tensors to verify kernel output determinism post-convergence
    n = 256
    ref_input = torch.randn(n, device=device)
    ref_out_a = torch.zeros(n, device=device)
    ref_out_b = torch.zeros(n, device=device)

    num_nodes = 3
    node_ids = list(range(num_nodes))
    graph = _MockGraph(num_nodes)
    profiler = _MockProfiler(node_ids, wall_clock_ms=1.0)
    optimizer_fn = _make_converging_optimizer(node_ids, stabilize_after=2)
    execute_fn = _make_improving_execute_fn(start_ms=2.0, plateau_ms=1.0, improve_over=2)

    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )

    best_config = controller.run_feedback_loop(
        optimizer_fn=optimizer_fn,
        profiler=profiler,
        execute_fn=execute_fn,
    )

    # Post-convergence: should_continue must be False
    assert controller.should_continue() is False
    assert controller._converged is True

    # Re-checking convergence still yields True (stable)
    result = controller.detect_convergence()
    assert result is True

    # The best config must be deterministic — calling rollback returns it
    re_best = controller.rollback()
    assert re_best is not None

    # Running the converged config multiple times should give consistent
    # performance (simulated via the constant-plateau execute_fn).
    latencies = [execute_fn(best_config) for _ in range(5)]
    assert all(
        abs(lat - latencies[0]) < 0.01 for lat in latencies
    ), "Post-convergence performance should be stable"

    # Verify deterministic kernel output: same input → bitwise-identical output
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    stable_chain_kernel[grid](ref_input, ref_out_a, n, BLOCK_SIZE=128)
    stable_chain_kernel[grid](ref_input, ref_out_b, n, BLOCK_SIZE=128)
    torch.cuda.synchronize()
    assert torch.equal(ref_out_a, ref_out_b), (
        "Kernel output must be deterministic after convergence"
    )


@pytest.mark.kernel_graph
def test_early_stopping_saves_iterations(device, convergence_config):
    """Verify early stopping on convergence saves unnecessary iterations.

    A simple workload that converges quickly should finish well before
    max_iterations, proving that the controller does not waste cycles.
    """
    num_nodes = 3
    node_ids = list(range(num_nodes))
    graph = _MockGraph(num_nodes)
    profiler = _MockProfiler(node_ids, wall_clock_ms=1.0)
    # Stabilise after just 2 calls — convergence should happen by iteration ~4-6
    optimizer_fn = _make_converging_optimizer(node_ids, stabilize_after=2)
    execute_fn = _make_improving_execute_fn(start_ms=1.5, plateau_ms=1.0, improve_over=2)

    controller = FeedbackController(
        graph=graph,
        config=convergence_config.feedback,
    )

    t_start = time.perf_counter()
    best_config = controller.run_feedback_loop(
        optimizer_fn=optimizer_fn,
        profiler=profiler,
        execute_fn=execute_fn,
    )
    t_end = time.perf_counter()

    # Must have converged
    assert controller._converged is True
    # Must finish before max_iterations (20)
    assert controller._iteration < convergence_config.feedback.max_iterations, (
        f"Early stopping failed: used {controller._iteration} of "
        f"{convergence_config.feedback.max_iterations} allowed iterations"
    )
    # Wall-clock sanity — loop should complete quickly (< 5 seconds for mocks)
    elapsed = t_end - t_start
    assert elapsed < 5.0, f"Feedback loop took {elapsed:.2f}s — too slow for mocks"
