"""Unit tests for the TorchInductor integration surface API contract.

Validates:
- ``submit_kernel_graph()`` function signature, input validation, and return type
- ``KernelGraphResult`` field structure (launch_sequence, estimated_latency_ms,
  devices_used, fusion_count, feedback_iterations)
- ``DevicePlacement`` soft/hard affinity handling and validation
- ``SchedulingHints`` optional parameter acceptance
- API contract stability via ``inspect`` introspection

All tests are marked ``@pytest.mark.kernel_graph`` per AAP §0.7.5.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys

import pytest
from typing import List, Tuple, Optional, Dict, Literal
from unittest.mock import MagicMock, patch

from triton.graph.torch_inductor_api import (
    submit_kernel_graph,
    KernelGraphResult,
    DevicePlacement,
    SchedulingHints,
    LaunchOp,
)
from triton.graph.kgir import HardwareProfile
from triton.graph.errors import GraphCaptureError

# DispatchError is raised for device-preference validation failures
from triton.graph.errors import DispatchError

# ---------------------------------------------------------------------------
# Module path constant for concise patching
# ---------------------------------------------------------------------------
_MOD = "triton.graph.torch_inductor_api"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _sched_entry(kid: int, sid: int = 0, dev=None,
                 deps: Optional[List[int]] = None, dur: float = 0.1):
    """Create a lightweight mock ``ScheduleEntry``."""
    entry = MagicMock()
    entry.kernel_id = kid
    entry.stream_id = sid
    entry.device = dev
    entry.dependencies = deps if deps is not None else []
    entry.estimated_duration_ms = dur
    return entry


def _gpu(backend: str = "cuda", arch=90, ws: int = 32):
    """Create a mock ``GPUTarget`` with the three required attributes."""
    target = MagicMock()
    target.backend = backend
    target.arch = arch
    target.warp_size = ws
    return target


# ---------------------------------------------------------------------------
# Fixture: full internal-pipeline mock
# ---------------------------------------------------------------------------

@pytest.fixture
def pipe(mock_gpu_target):
    """Replace every internal pipeline helper inside ``submit_kernel_graph``
    with deterministic mocks so the public API can be exercised without GPU
    hardware or fully-built graph sub-modules.

    Yields a dict with ``gpu_target``, ``config``, ``graph``, and
    ``inventory`` keys for fine-grained assertions in individual tests.
    """
    # Shared mutable state so the mock functions stay coordinated.
    state: Dict[str, Dict[int, int]] = {"imap": {}}

    # ── Mock KGIR graph ───────────────────────────────────────────────
    mock_graph = MagicMock()
    mock_graph.hardware_profiles = []
    mock_graph.get_node.return_value = MagicMock(
        is_fused=False, fused_from=None,
    )

    def _build(kernels, dependencies, hardware_profiles=None):
        state["imap"] = {i: i for i in range(len(kernels))}
        return mock_graph, dict(state["imap"])

    def _dispatch(g, c, inv, i2n, dp):
        plan = {nid: mock_gpu_target for nid in i2n.values()}
        if dp:
            for kid, pl in dp.items():
                if pl.affinity == "hard" and kid in i2n:
                    plan[i2n[kid]] = pl.target
        return plan

    def _schedule(g, c):
        return [
            _sched_entry(
                i, sid=0, dev=None,
                deps=([i - 1] if i > 0 else []),
                dur=0.1,
            )
            for i in sorted(state["imap"].keys())
        ]

    def _feedback(graph, config, schedule_entries, dispatch_plan,
                  index_to_node_id, node_id_to_index, hints,
                  device_preferences):
        return (schedule_entries, dispatch_plan, 0, 0)

    # ── Mock config ───────────────────────────────────────────────────
    cfg = MagicMock()
    cfg.dispatch = MagicMock(stream_pool_size=4, mode="balanced")
    cfg.fusion = MagicMock()
    cfg.feedback = MagicMock(enable=False)

    inv = MagicMock(device_count=1, devices=[MagicMock()])

    mock_gc = MagicMock()
    mock_gc.from_env.return_value = cfg

    # ── Start patches ─────────────────────────────────────────────────
    _patches = [
        patch(f"{_MOD}.GraphConfig", mock_gc),
        patch(f"{_MOD}._discover_hardware", return_value=(inv, [])),
        patch(f"{_MOD}._build_kgir_graph", side_effect=_build),
        patch(f"{_MOD}._run_fusion", return_value=0),
        patch(f"{_MOD}._run_memory_planning"),
        patch(f"{_MOD}._run_dispatch", side_effect=_dispatch),
        patch(f"{_MOD}._run_scheduling", side_effect=_schedule),
        patch(f"{_MOD}._run_codegen"),
        patch(f"{_MOD}._run_feedback", side_effect=_feedback),
    ]
    for p in _patches:
        p.start()

    yield {
        "gpu_target": mock_gpu_target,
        "config": cfg,
        "graph": mock_graph,
        "inventory": inv,
    }

    for p in _patches:
        p.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — submit_kernel_graph signature & validation tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestSubmitKernelGraphBasic:
    """Tests for the core ``submit_kernel_graph`` happy-path signatures."""

    def test_minimal_required_args(self, pipe, make_mock_kernel):
        """Call with only *kernels* and *dependencies*; expect a
        ``KernelGraphResult`` back."""
        k0 = make_mock_kernel("add")
        k1 = make_mock_kernel("mul")
        k2 = make_mock_kernel("relu")

        result = submit_kernel_graph(
            kernels=[k0, k1, k2],
            dependencies=[(0, 1), (1, 2)],
        )

        assert isinstance(result, KernelGraphResult)
        assert isinstance(result.launch_sequence, list)
        assert len(result.launch_sequence) == 3

    def test_single_kernel_no_deps(self, pipe, make_mock_kernel):
        """Edge case: single kernel with no dependencies."""
        k = make_mock_kernel("single")
        result = submit_kernel_graph(kernels=[k], dependencies=[])

        assert isinstance(result, KernelGraphResult)
        assert len(result.launch_sequence) == 1

    def test_many_independent_kernels(self, pipe, make_mock_kernel):
        """Ten independent kernels with no dependencies."""
        kernels = [make_mock_kernel(f"k{i}") for i in range(10)]
        result = submit_kernel_graph(kernels=kernels, dependencies=[])

        assert isinstance(result, KernelGraphResult)
        assert len(result.launch_sequence) == 10

    def test_with_scheduling_hints(self, pipe, make_mock_kernel):
        """Optional *SchedulingHints* parameter is accepted."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")

        hints = SchedulingHints(
            priority_kernels=[0],
            max_streams=4,
            prefer_fusion=True,
            target_latency_ms=10.0,
        )

        result = submit_kernel_graph(
            kernels=[k0, k1],
            dependencies=[(0, 1)],
            hints=hints,
        )
        assert isinstance(result, KernelGraphResult)

    def test_with_device_preferences(self, pipe, make_mock_kernel):
        """Optional *device_preferences* parameter is accepted."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")

        placement = DevicePlacement(
            target=pipe["gpu_target"], affinity="soft",
        )
        result = submit_kernel_graph(
            kernels=[k0, k1],
            dependencies=[(0, 1)],
            device_preferences={0: placement},
        )
        assert isinstance(result, KernelGraphResult)

    def test_all_optional_params(self, pipe, make_mock_kernel):
        """Both *hints* and *device_preferences* supplied simultaneously."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")

        hints = SchedulingHints(
            priority_kernels=[0], max_streams=2,
            prefer_fusion=False, target_latency_ms=5.0,
        )
        placement = DevicePlacement(
            target=pipe["gpu_target"], affinity="soft",
        )

        result = submit_kernel_graph(
            kernels=[k0, k1],
            dependencies=[(0, 1)],
            hints=hints,
            device_preferences={1: placement},
        )
        assert isinstance(result, KernelGraphResult)


# ── Input validation tests (no pipeline mock needed) ──────────────────────

@pytest.mark.kernel_graph
class TestSubmitKernelGraphValidation:
    """Validation errors raised by ``submit_kernel_graph`` before the
    pipeline executes."""

    def test_empty_kernels(self, make_mock_kernel):
        """Empty kernel list raises *GraphCaptureError*."""
        with pytest.raises(GraphCaptureError, match="must not be empty"):
            submit_kernel_graph(kernels=[], dependencies=[])

    def test_dependency_out_of_range(self, make_mock_kernel):
        """Dependency index exceeding kernel count raises
        *GraphCaptureError*."""
        k = make_mock_kernel("only")
        with pytest.raises(GraphCaptureError, match="Invalid .* index"):
            submit_kernel_graph(kernels=[k], dependencies=[(0, 5)])

    def test_dependency_negative_index(self, make_mock_kernel):
        """Negative dependency index raises *GraphCaptureError*."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        with pytest.raises(GraphCaptureError, match="Invalid .* index"):
            submit_kernel_graph(
                kernels=[k0, k1], dependencies=[(-1, 0)],
            )

    def test_self_dependency(self, make_mock_kernel):
        """Self-dependency raises *GraphCaptureError*."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        with pytest.raises(GraphCaptureError, match="[Ss]elf.dependency"):
            submit_kernel_graph(
                kernels=[k0, k1], dependencies=[(0, 0)],
            )

    def test_cyclic_dependency(self, make_mock_kernel):
        """Circular dependency graph raises *GraphCaptureError*."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        k2 = make_mock_kernel("c")
        with pytest.raises(GraphCaptureError, match="cycle"):
            submit_kernel_graph(
                kernels=[k0, k1, k2],
                dependencies=[(0, 1), (1, 2), (2, 0)],
            )

    def test_non_integer_dependency(self, make_mock_kernel):
        """Non-integer element in a dependency tuple raises
        *GraphCaptureError*."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        with pytest.raises(GraphCaptureError, match="must contain integers"):
            submit_kernel_graph(
                kernels=[k0, k1], dependencies=[("0", 1)],
            )

    def test_invalid_device_pref_key(self, make_mock_kernel):
        """Non-integer key in *device_preferences* raises
        *DispatchError*."""
        k = make_mock_kernel("a")
        target = _gpu()
        placement = DevicePlacement(target=target, affinity="soft")
        with pytest.raises(DispatchError, match="key"):
            submit_kernel_graph(
                kernels=[k], dependencies=[],
                device_preferences={"zero": placement},
            )

    def test_invalid_device_pref_index(self, make_mock_kernel):
        """Out-of-range kernel index in *device_preferences* raises
        *DispatchError*."""
        k = make_mock_kernel("a")
        target = _gpu()
        placement = DevicePlacement(target=target, affinity="soft")
        with pytest.raises(DispatchError, match="[Ii]ndex"):
            submit_kernel_graph(
                kernels=[k], dependencies=[],
                device_preferences={99: placement},
            )

    def test_invalid_device_pref_value_type(self, make_mock_kernel):
        """Non-DevicePlacement value in *device_preferences* raises
        *DispatchError*."""
        k = make_mock_kernel("a")
        with pytest.raises(DispatchError, match="DevicePlacement"):
            submit_kernel_graph(
                kernels=[k], dependencies=[],
                device_preferences={0: "not a placement"},
            )

    def test_invalid_affinity_in_preferences(self, make_mock_kernel):
        """DevicePlacement with invalid affinity raises *DispatchError*."""
        k = make_mock_kernel("a")
        target = _gpu()
        placement = DevicePlacement(target=target, affinity="invalid")
        with pytest.raises(DispatchError, match="Invalid affinity"):
            submit_kernel_graph(
                kernels=[k], dependencies=[],
                device_preferences={0: placement},
            )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — KernelGraphResult field tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestKernelGraphResult:
    """Verify every field of ``KernelGraphResult`` per AAP §0.5.1."""

    def test_all_fields_present(self, pipe, make_mock_kernel):
        """Result exposes launch_sequence, estimated_latency_ms,
        devices_used, fusion_count, and feedback_iterations."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        result = submit_kernel_graph(
            kernels=[k0, k1], dependencies=[(0, 1)],
        )

        assert hasattr(result, "launch_sequence")
        assert hasattr(result, "estimated_latency_ms")
        assert hasattr(result, "devices_used")
        assert hasattr(result, "fusion_count")
        assert hasattr(result, "feedback_iterations")

    def test_field_types(self, pipe, make_mock_kernel):
        """All fields have the correct Python types."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        result = submit_kernel_graph(
            kernels=[k0, k1], dependencies=[(0, 1)],
        )

        assert isinstance(result.launch_sequence, list)
        assert isinstance(result.estimated_latency_ms, float)
        assert isinstance(result.devices_used, list)
        assert isinstance(result.fusion_count, int)
        assert isinstance(result.feedback_iterations, int)

    def test_launch_sequence_contains_launch_ops(self, pipe, make_mock_kernel):
        """Every element in launch_sequence is a ``LaunchOp``."""
        kernels = [make_mock_kernel(f"k{i}") for i in range(3)]
        result = submit_kernel_graph(
            kernels=kernels,
            dependencies=[(0, 1), (1, 2)],
        )

        assert len(result.launch_sequence) == 3
        for op in result.launch_sequence:
            assert isinstance(op, LaunchOp)

    def test_launch_op_attributes(self, pipe, make_mock_kernel):
        """Each ``LaunchOp`` carries kernel_id, stream_id, device,
        dependencies, is_transfer, and fused_with."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        result = submit_kernel_graph(
            kernels=[k0, k1], dependencies=[(0, 1)],
        )

        for op in result.launch_sequence:
            assert hasattr(op, "kernel_id")
            assert hasattr(op, "stream_id")
            assert hasattr(op, "device")
            assert hasattr(op, "dependencies")
            assert hasattr(op, "is_transfer")
            assert hasattr(op, "fused_with")

    def test_launch_sequence_kernel_ids(self, pipe, make_mock_kernel):
        """Kernel IDs in launch_sequence cover the original kernel indices."""
        kernels = [make_mock_kernel(f"k{i}") for i in range(4)]
        result = submit_kernel_graph(kernels=kernels, dependencies=[])

        ids = {op.kernel_id for op in result.launch_sequence}
        assert ids == {0, 1, 2, 3}

    def test_launch_op_dependencies_reflect_dag(self, pipe, make_mock_kernel):
        """LaunchOp.dependencies reflect the scheduling order."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        k2 = make_mock_kernel("c")
        result = submit_kernel_graph(
            kernels=[k0, k1, k2],
            dependencies=[(0, 1), (1, 2)],
        )

        op_map = {op.kernel_id: op for op in result.launch_sequence}
        # First kernel has no deps
        assert op_map[0].dependencies == []
        # Each subsequent kernel should have list-typed dependencies
        assert isinstance(op_map[1].dependencies, list)
        assert isinstance(op_map[2].dependencies, list)

    def test_estimated_latency_positive_finite(self, pipe, make_mock_kernel):
        """estimated_latency_ms is positive and finite."""
        k = make_mock_kernel("a")
        result = submit_kernel_graph(kernels=[k], dependencies=[])

        assert result.estimated_latency_ms > 0.0
        assert math.isfinite(result.estimated_latency_ms)
        assert not math.isnan(result.estimated_latency_ms)

    def test_devices_used_no_duplicates(self, pipe, make_mock_kernel):
        """devices_used has no duplicate target entries."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        result = submit_kernel_graph(
            kernels=[k0, k1], dependencies=[(0, 1)],
        )

        assert isinstance(result.devices_used, list)
        assert len(result.devices_used) >= 1
        # Deduplicated by (backend, arch, warp_size)
        keys = [
            (d.backend, d.arch, d.warp_size) for d in result.devices_used
        ]
        assert len(keys) == len(set(keys))

    def test_fusion_count_value(self, pipe, make_mock_kernel):
        """fusion_count reflects the number of fusion ops (0 with mock)."""
        k = make_mock_kernel("a")
        result = submit_kernel_graph(kernels=[k], dependencies=[])
        assert result.fusion_count == 0

    def test_feedback_iterations_value(self, pipe, make_mock_kernel):
        """feedback_iterations reports iterations performed (0 with mock)."""
        k = make_mock_kernel("a")
        result = submit_kernel_graph(kernels=[k], dependencies=[])
        assert result.feedback_iterations == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — DevicePlacement tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestDevicePlacement:
    """Tests for ``DevicePlacement`` construction, affinity semantics, and
    validation."""

    def test_construction_defaults(self):
        """Default affinity is ``'soft'``."""
        target = _gpu()
        p = DevicePlacement(target=target)
        assert p.target is target
        assert p.affinity == "soft"

    def test_soft_affinity(self):
        """Soft affinity stores correctly."""
        target = _gpu()
        p = DevicePlacement(target=target, affinity="soft")
        assert p.affinity == "soft"

    def test_hard_affinity(self):
        """Hard affinity stores correctly."""
        target = _gpu()
        p = DevicePlacement(target=target, affinity="hard")
        assert p.affinity == "hard"

    def test_soft_affinity_is_hint(self, pipe, make_mock_kernel):
        """A soft placement is advisory — the optimizer may reassign the
        kernel to a different device.  The mock dispatch ignores soft
        preferences, demonstrating overridability."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")

        amd_target = _gpu(backend="hip", arch="gfx942", ws=64)
        placement = DevicePlacement(target=amd_target, affinity="soft")

        result = submit_kernel_graph(
            kernels=[k0, k1],
            dependencies=[(0, 1)],
            device_preferences={0: placement},
        )

        op0 = next(
            op for op in result.launch_sequence if op.kernel_id == 0
        )
        # Soft preference: the mock dispatch does NOT override, so the
        # kernel stays on the default target (cuda) — proving soft is
        # merely a hint.
        assert op0.device is not None
        assert op0.device.backend == "cuda"

    def test_hard_affinity_honoured(self, pipe, make_mock_kernel):
        """A hard placement MUST be honoured unconditionally — the kernel
        is placed on the specified target."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")

        amd_target = _gpu(backend="hip", arch="gfx942", ws=64)
        placement = DevicePlacement(target=amd_target, affinity="hard")

        result = submit_kernel_graph(
            kernels=[k0, k1],
            dependencies=[(0, 1)],
            device_preferences={0: placement},
        )

        op0 = next(
            op for op in result.launch_sequence if op.kernel_id == 0
        )
        assert op0.device.backend == "hip"
        assert op0.device.arch == "gfx942"
        assert op0.device.warp_size == 64

    def test_hard_affinity_devices_used(self, pipe, make_mock_kernel):
        """When hard affinity places a kernel on a different target, both
        targets appear in ``devices_used``."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")

        amd_target = _gpu(backend="hip", arch="gfx942", ws=64)
        placement = DevicePlacement(target=amd_target, affinity="hard")

        result = submit_kernel_graph(
            kernels=[k0, k1],
            dependencies=[(0, 1)],
            device_preferences={0: placement},
        )

        backends = {d.backend for d in result.devices_used}
        assert "hip" in backends
        assert "cuda" in backends

    def test_invalid_affinity_raises(self, make_mock_kernel):
        """``DevicePlacement`` with invalid affinity raises
        ``DispatchError`` during validation."""
        k = make_mock_kernel("a")
        target = _gpu()
        placement = DevicePlacement(target=target, affinity="invalid")

        with pytest.raises(DispatchError, match="Invalid affinity"):
            submit_kernel_graph(
                kernels=[k], dependencies=[],
                device_preferences={0: placement},
            )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — SchedulingHints tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestSchedulingHints:
    """Tests for ``SchedulingHints`` dataclass and its integration."""

    def test_construction_all_fields(self):
        """All fields can be set at construction."""
        hints = SchedulingHints(
            priority_kernels=[0, 2],
            max_streams=8,
            prefer_fusion=True,
            target_latency_ms=5.0,
        )
        assert hints.priority_kernels == [0, 2]
        assert hints.max_streams == 8
        assert hints.prefer_fusion is True
        assert hints.target_latency_ms == 5.0

    def test_defaults(self):
        """Default values match AAP specification."""
        hints = SchedulingHints()
        assert hints.priority_kernels is None
        assert hints.max_streams is None
        assert hints.prefer_fusion is True
        assert hints.target_latency_ms is None

    def test_none_accepted(self, pipe, make_mock_kernel):
        """``hints=None`` (the default) works without error."""
        k = make_mock_kernel("a")
        result = submit_kernel_graph(
            kernels=[k], dependencies=[], hints=None,
        )
        assert isinstance(result, KernelGraphResult)

    def test_prefer_fusion_false(self, pipe, make_mock_kernel):
        """Setting ``prefer_fusion=False`` is accepted."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")
        hints = SchedulingHints(prefer_fusion=False)
        result = submit_kernel_graph(
            kernels=[k0, k1],
            dependencies=[(0, 1)],
            hints=hints,
        )
        assert isinstance(result, KernelGraphResult)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — API contract stability tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestAPIContractStability:
    """Introspection-based tests ensuring the public API surface matches
    the AAP §0.5.1 specification and is stable across versions."""

    # ── submit_kernel_graph signature ─────────────────────────────────

    def test_submit_kernel_graph_params(self):
        """Parameter names match the AAP specification."""
        sig = inspect.signature(submit_kernel_graph)
        params = list(sig.parameters.keys())

        assert "kernels" in params
        assert "dependencies" in params
        assert "hints" in params
        assert "device_preferences" in params

    def test_submit_kernel_graph_optional_defaults(self):
        """Optional parameters default to ``None``."""
        sig = inspect.signature(submit_kernel_graph)

        p_hints = sig.parameters["hints"]
        assert p_hints.default is None
        # Verify parameter kind via inspect.Parameter constants
        assert p_hints.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )

        p_dp = sig.parameters["device_preferences"]
        assert p_dp.default is None
        assert p_dp.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_submit_kernel_graph_parameter_order(self):
        """Parameters appear in the documented order."""
        sig = inspect.signature(submit_kernel_graph)
        params = list(sig.parameters.keys())
        idx_k = params.index("kernels")
        idx_d = params.index("dependencies")
        idx_h = params.index("hints")
        idx_dp = params.index("device_preferences")

        assert idx_k < idx_d < idx_h < idx_dp

    # ── Dataclass field contracts ─────────────────────────────────────

    def test_kernel_graph_result_is_dataclass(self):
        """``KernelGraphResult`` is a proper dataclass."""
        assert dataclasses.is_dataclass(KernelGraphResult)

    def test_kernel_graph_result_field_names(self):
        """``KernelGraphResult`` exposes the five specified fields."""
        names = {f.name for f in dataclasses.fields(KernelGraphResult)}
        assert "launch_sequence" in names
        assert "estimated_latency_ms" in names
        assert "devices_used" in names
        assert "fusion_count" in names
        assert "feedback_iterations" in names

    def test_launch_op_is_dataclass(self):
        """``LaunchOp`` is a proper dataclass."""
        assert dataclasses.is_dataclass(LaunchOp)

    def test_launch_op_field_names(self):
        """``LaunchOp`` exposes the six specified fields."""
        names = {f.name for f in dataclasses.fields(LaunchOp)}
        assert "kernel_id" in names
        assert "stream_id" in names
        assert "device" in names
        assert "dependencies" in names
        assert "is_transfer" in names
        assert "fused_with" in names

    def test_device_placement_is_dataclass(self):
        """``DevicePlacement`` is a proper dataclass."""
        assert dataclasses.is_dataclass(DevicePlacement)

    def test_device_placement_field_names(self):
        """``DevicePlacement`` exposes target and affinity fields."""
        names = {f.name for f in dataclasses.fields(DevicePlacement)}
        assert "target" in names
        assert "affinity" in names

    def test_scheduling_hints_is_dataclass(self):
        """``SchedulingHints`` is a proper dataclass."""
        assert dataclasses.is_dataclass(SchedulingHints)

    def test_scheduling_hints_field_names(self):
        """``SchedulingHints`` exposes the four specified fields."""
        names = {f.name for f in dataclasses.fields(SchedulingHints)}
        assert "priority_kernels" in names
        assert "max_streams" in names
        assert "prefer_fusion" in names
        assert "target_latency_ms" in names

    # ── Standalone mode ───────────────────────────────────────────────

    def test_standalone_without_torch(self, pipe, make_mock_kernel):
        """Per AAP §0.1.1 the feature works standalone — the API can be
        called without *torch* or *TorchInductor* being importable."""
        k0 = make_mock_kernel("a")
        k1 = make_mock_kernel("b")

        orig_torch = sys.modules.get("torch")
        orig_inductor = sys.modules.get("torch._inductor")
        try:
            sys.modules["torch"] = None  # type: ignore[assignment]
            sys.modules["torch._inductor"] = None  # type: ignore[assignment]

            result = submit_kernel_graph(
                kernels=[k0, k1], dependencies=[(0, 1)],
            )
            assert isinstance(result, KernelGraphResult)
        finally:
            # Restore original module entries
            if orig_torch is not None:
                sys.modules["torch"] = orig_torch
            else:
                sys.modules.pop("torch", None)
            if orig_inductor is not None:
                sys.modules["torch._inductor"] = orig_inductor
            else:
                sys.modules.pop("torch._inductor", None)


# ═══════════════════════════════════════════════════════════════════════════
# Supplementary LaunchOp construction tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestLaunchOpConstruction:
    """Direct construction tests for the ``LaunchOp`` dataclass."""

    def test_is_transfer_default(self):
        """``is_transfer`` defaults to ``False``."""
        target = _gpu()
        op = LaunchOp(kernel_id=0, stream_id=0, device=target)
        assert op.is_transfer is False

    def test_fused_with_default(self):
        """``fused_with`` defaults to ``None``."""
        target = _gpu()
        op = LaunchOp(kernel_id=0, stream_id=0, device=target)
        assert op.fused_with is None

    def test_fused_with_explicit(self):
        """``fused_with`` can hold a list of original kernel indices."""
        target = _gpu()
        op = LaunchOp(
            kernel_id=0, stream_id=0, device=target,
            fused_with=[1, 2, 3],
        )
        assert op.fused_with == [1, 2, 3]

    def test_dependencies_default(self):
        """``dependencies`` defaults to an empty list when constructed
        with the factory default."""
        target = _gpu()
        op = LaunchOp(kernel_id=5, stream_id=2, device=target)
        assert isinstance(op.dependencies, list)
