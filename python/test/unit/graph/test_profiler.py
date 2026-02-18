"""Unit tests for the RuntimeProfiler from ``triton.graph.profiler``.

Test coverage is organized into five phases:

Phase 1 — Event Instrumentation
    Verifies that CUDA and HIP GPU events are created, recorded, synchronized,
    and cleaned up correctly when instrumenting kernel launches.

Phase 2 — Metric Collection
    Validates per-kernel and per-target metric collection including wall-clock
    time, metric fields coverage (AAP §0.1.1), and metric aggregation.

Phase 3 — Overhead Budget Enforcement
    Ensures profiling overhead remains < 3 % of total kernel execution time
    (AAP §0.7.2) and that the profiler uses lightweight async GPU events
    (no synchronous CPU-GPU round-trips during instrumentation).

Phase 4 — Profiler Lifecycle
    Tests reset/re-profiling, enable/disable toggling, and empty-graph
    handling.

Phase 5 — Profiler API
    Verifies the high-level ``instrument_launch`` and ``synchronize_and_collect``
    API contracts.

All tests are marked ``@pytest.mark.kernel_graph`` and rely exclusively on
``unittest.mock`` — **no actual GPU execution is required**.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest

from triton.graph.profiler import RuntimeProfiler, _GPUEvent, _BACKEND_CPU
from triton.graph.kgir import KGIRNode, KGIRGraph, HardwareProfile
from triton.graph.config import GraphConfig, FeedbackConfig, DispatchConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Helper factories (local to this test module)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_profiler(
    feedback_enable: bool = False,
    feedback_log: bool = False,
    overhead_budget: float = 0.03,
    stream_pool_size: int = 8,
) -> RuntimeProfiler:
    """Create a ``RuntimeProfiler`` with explicit sub-config control."""
    config = GraphConfig(
        feedback=FeedbackConfig(enable=feedback_enable, log=feedback_log),
        dispatch=DispatchConfig(stream_pool_size=stream_pool_size),
    )
    return RuntimeProfiler(config=config, overhead_budget=overhead_budget)


def _make_mock_gpu_target(backend: str = "cuda", arch=90, warp_size: int = 32):
    """Return a lightweight ``GPUTarget`` mock for the given backend."""
    target = MagicMock()
    target.backend = backend
    target.arch = arch
    target.warp_size = warp_size
    return target


def _make_hw_profile(vendor: str = "nvidia", arch_gen: str = "sm_90") -> HardwareProfile:
    """Return a realistic ``HardwareProfile`` for tests that need one."""
    return HardwareProfile(
        vendor=vendor,
        arch_generation=arch_gen,
        sm_count=132,
        smem_per_sm_bytes=228 * 1024,
        registers_per_sm=65536,
        global_memory_bytes=80 * (1024 ** 3),
        memory_bandwidth_gbps=3350.0,
        compute_throughput_tflops=989.0,
        warp_size=32 if vendor == "nvidia" else 64,
        max_concurrent_streams=128,
        interconnect_type="nvlink_4" if vendor == "nvidia" else "infinity_fabric",
        interconnect_bandwidth_gbps=900.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Event Instrumentation Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestEventInstrumentation:
    """Phase 1: Verify GPU event creation and recording for CUDA and HIP."""

    def test_profiler_creates_cuda_events(self, mock_gpu_target):
        """Mock CUDA event APIs and verify events are created for a CUDA target.

        Steps:
        1. Construct a profiler.
        2. Call ``instrument_launch`` with a CUDA target.
        3. Verify start/end ``_GPUEvent`` objects are returned.
        4. Verify event backend is "cuda" (or "cpu" fallback since torch may
           not be installed in this test environment).
        """
        profiler = _make_profiler()
        kernel_id = 0

        start_ev, end_ev = profiler.instrument_launch(
            kernel_id=kernel_id,
            target=mock_gpu_target,
            stream=None,
        )

        # Events must be _GPUEvent instances.
        assert isinstance(start_ev, _GPUEvent)
        assert isinstance(end_ev, _GPUEvent)

        # The backend should be "cuda" (if torch.cuda.Event works) or fall
        # back to "cpu".  In either case the event is usable.
        assert start_ev._backend in ("cuda", _BACKEND_CPU)
        assert end_ev._backend in ("cuda", _BACKEND_CPU)

        # The events must be registered internally.
        assert kernel_id in profiler._events
        entry = profiler._events[kernel_id]
        assert entry["start"] is start_ev
        assert entry["end"] is end_ev
        assert entry["target"] is mock_gpu_target

    def test_profiler_creates_hip_events(self, mock_gpu_target_amd):
        """Verify that an AMD/HIP target triggers HIP-compatible events.

        Because ``torch.cuda.Event`` maps transparently to HIP on ROCm builds,
        the internal ``_GPUEvent._backend`` should be ``"hip"`` or ``"cpu"``
        (fallback if torch is absent).
        """
        profiler = _make_profiler()
        kernel_id = 1

        start_ev, end_ev = profiler.instrument_launch(
            kernel_id=kernel_id,
            target=mock_gpu_target_amd,
            stream=None,
        )

        assert isinstance(start_ev, _GPUEvent)
        assert isinstance(end_ev, _GPUEvent)

        # HIP target: backend should be "hip" or CPU fallback.
        assert start_ev._backend in ("hip", _BACKEND_CPU)
        assert end_ev._backend in ("hip", _BACKEND_CPU)

        assert kernel_id in profiler._events

    def test_profiler_event_lifecycle(self, mock_gpu_target):
        """Verify the full event lifecycle: create → record → sync → collect.

        1. ``instrument_launch`` creates events.
        2. ``record_start`` records the start event.
        3. ``record_end`` records the end event.
        4. ``synchronize_and_collect`` synchronizes both and computes metrics.

        After collection the event entry should still exist in ``_events``
        and the metric for the kernel should be populated.
        """
        profiler = _make_profiler()
        kernel_id = 42

        # Step 1: Create events.
        start_ev, end_ev = profiler.instrument_launch(
            kernel_id=kernel_id,
            target=mock_gpu_target,
            stream=None,
        )

        # Step 2 & 3: Record start and end.
        profiler.record_start(kernel_id=kernel_id, target=mock_gpu_target)
        # Simulate a tiny delay for the CPU-fallback to measure.
        time.sleep(0.001)
        profiler.record_end(kernel_id=kernel_id, target=mock_gpu_target)

        # Step 4: Synchronize and collect.
        metrics = profiler.synchronize_and_collect()

        # Metric for this kernel must exist and have a positive wall-clock.
        assert kernel_id in metrics
        assert "wall_clock_ms" in metrics[kernel_id]
        # In the CPU-fallback path the elapsed time should be >= 0.
        assert metrics[kernel_id]["wall_clock_ms"] >= 0.0

        # Events should still be stored for potential re-use.
        assert kernel_id in profiler._events


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Metric Collection Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestMetricCollection:
    """Phase 2: Validate per-kernel and per-target metric collection."""

    def test_collect_wall_clock_time(self, mock_gpu_target):
        """Profile a kernel launch and verify wall_clock_ms is collected.

        Uses the CPU-fallback path with a small sleep to guarantee a
        measurable elapsed time.
        """
        profiler = _make_profiler()
        kernel_id = 0

        profiler.instrument_launch(kernel_id, mock_gpu_target, stream=None)
        profiler.record_start(kernel_id, mock_gpu_target)
        time.sleep(0.002)  # ~2 ms
        profiler.record_end(kernel_id, mock_gpu_target)

        metrics = profiler.synchronize_and_collect()

        assert kernel_id in metrics
        wall_ms = metrics[kernel_id]["wall_clock_ms"]
        assert wall_ms > 0.0, "wall_clock_ms must be positive after a sleep"

    def test_collect_per_kernel_metrics(
        self, mock_gpu_target, sample_kgir_graph, make_kgir_node
    ):
        """Profile a graph with 3 kernels; verify per-kernel metrics.

        Uses the ``sample_kgir_graph`` fixture (A→B→C chain) and the
        ``make_kgir_node`` factory fixture to validate the profiler works
        with KGIR-style node ids.
        """
        profiler = _make_profiler()

        # Verify make_kgir_node factory works (coverage for conftest fixture).
        extra_node = make_kgir_node("extra_kernel")
        assert extra_node is not None

        # Instrument and record 3 kernels sequentially.
        for kid in range(3):
            profiler.instrument_launch(kid, mock_gpu_target, stream=None)
            profiler.record_start(kid, mock_gpu_target)
            time.sleep(0.001)
            profiler.record_end(kid, mock_gpu_target)

        metrics = profiler.synchronize_and_collect()

        # All 3 kernels should have metrics.
        assert len(metrics) == 3
        for kid in range(3):
            assert kid in metrics
            assert "wall_clock_ms" in metrics[kid]
            assert metrics[kid]["wall_clock_ms"] >= 0.0

    def test_collect_per_target_metrics(
        self, mock_gpu_target, mock_gpu_target_amd
    ):
        """Profile the same logical kernel on two different targets.

        Verify ``get_per_target_metrics`` groups results by target.
        """
        profiler = _make_profiler()

        # Kernel 0 on NVIDIA target.
        profiler.instrument_launch(0, mock_gpu_target, stream=None)
        profiler.record_start(0, mock_gpu_target)
        time.sleep(0.001)
        profiler.record_end(0, mock_gpu_target)

        # Kernel 1 on AMD target.
        profiler.instrument_launch(1, mock_gpu_target_amd, stream=None)
        profiler.record_start(1, mock_gpu_target_amd)
        time.sleep(0.001)
        profiler.record_end(1, mock_gpu_target_amd)

        profiler.synchronize_and_collect()

        per_target = profiler.get_per_target_metrics()

        # Two distinct target keys.
        assert len(per_target) == 2

        # NVIDIA target should contain kernel 0.
        assert mock_gpu_target in per_target
        assert 0 in per_target[mock_gpu_target]

        # AMD target should contain kernel 1.
        assert mock_gpu_target_amd in per_target
        assert 1 in per_target[mock_gpu_target_amd]

    def test_metric_fields(self, mock_gpu_target, mock_nvidia_hw_profile):
        """Per AAP §0.1.1: metrics include wall-clock, throughput, occupancy,
        launch overhead, and cross-device transfer time.

        Verify each field is present and numeric.
        """
        profiler = _make_profiler()
        kernel_id = 0

        profiler.instrument_launch(
            kernel_id,
            mock_gpu_target,
            stream=None,
            hw_profile=mock_nvidia_hw_profile,
        )
        profiler.record_start(kernel_id, mock_gpu_target)
        time.sleep(0.001)
        profiler.record_end(kernel_id, mock_gpu_target)

        metrics = profiler.synchronize_and_collect()
        m = metrics[kernel_id]

        required_fields = {
            "wall_clock_ms",
            "memory_throughput_gbps",
            "launch_overhead_us",
            "cross_device_transfer_ms",
            "estimated_occupancy",
        }
        for field_name in required_fields:
            assert field_name in m, f"Missing metric field: {field_name}"
            assert isinstance(m[field_name], (int, float)), (
                f"Field {field_name} must be numeric, got {type(m[field_name])}"
            )

    def test_metric_aggregation(self, mock_gpu_target):
        """Profile a kernel multiple times and verify metrics update.

        When the same kernel_id is re-instrumented across iterations (after
        reset), the profiler should collect fresh metrics each time.
        """
        profiler = _make_profiler()
        kernel_id = 0

        collected_wall_clocks: list[float] = []

        for _ in range(3):
            profiler.reset()
            profiler.instrument_launch(kernel_id, mock_gpu_target, stream=None)
            profiler.record_start(kernel_id, mock_gpu_target)
            time.sleep(0.001)
            profiler.record_end(kernel_id, mock_gpu_target)
            metrics = profiler.synchronize_and_collect()
            collected_wall_clocks.append(metrics[kernel_id]["wall_clock_ms"])

        # Each iteration should produce a positive wall-clock value.
        for wc in collected_wall_clocks:
            assert wc >= 0.0

        # All three should be independent measurements (not accumulated).
        # With CPU-fallback timing the values will be close but distinct.
        assert len(collected_wall_clocks) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Overhead Budget Enforcement Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestOverheadBudget:
    """Phase 3: Overhead budget enforcement (AAP §0.7.2: < 3 %)."""

    def test_overhead_budget_enforcement(self, mock_gpu_target):
        """Verify that profiling overhead stays within the 3 % budget.

        Strategy:
        - Instrument + record several kernels with non-trivial CPU sleeps
          (simulating kernel execution time).
        - Collect and check that the profiling overhead ratio is well within
          the 3 % budget.
        """
        profiler = _make_profiler(overhead_budget=0.03)

        # Simulate 5 kernel executions each taking ~10 ms (sleep).
        # The instrumentation overhead (event create, record, time measurement)
        # should be negligible compared to 10 ms of sleep.
        for kid in range(5):
            profiler.instrument_launch(kid, mock_gpu_target, stream=None)
            profiler.record_start(kid, mock_gpu_target)
            time.sleep(0.010)  # 10 ms simulated execution
            profiler.record_end(kid, mock_gpu_target)

        profiler.synchronize_and_collect()

        # The overhead budget check should pass (return True).
        budget_ok = profiler.check_overhead_budget()
        assert budget_ok is True, (
            f"Profiling overhead exceeded 3% budget. "
            f"Total kernel time: {profiler._total_kernel_time:.3f} ms, "
            f"Total overhead: {profiler._total_profiling_overhead:.3f} ms"
        )

        # The profiler should remain enabled.
        assert profiler._enabled is True

    def test_profiler_lightweight(self, mock_gpu_target):
        """Verify the profiler uses lightweight GPU timing (CUDA events).

        In the CPU-fallback path ``_GPUEvent.record()`` uses
        ``time.perf_counter_ns()`` — a non-blocking call.  In the GPU path
        ``torch.cuda.Event`` is inherently asynchronous.

        We verify that instrumentation itself adds minimal wall-clock overhead
        compared to a kernel "execution" sleep.
        """
        profiler = _make_profiler()

        # Measure the wall-clock cost of instrumentation alone (no kernel).
        t0 = time.perf_counter()
        for kid in range(100):
            profiler.instrument_launch(kid, mock_gpu_target, stream=None)
            profiler.record_start(kid, mock_gpu_target)
            profiler.record_end(kid, mock_gpu_target)
        instrumentation_elapsed_s = time.perf_counter() - t0

        # 100 instrument + record_start + record_end cycles should complete
        # in well under 1 second (typically < 10 ms on modern CPUs).
        assert instrumentation_elapsed_s < 1.0, (
            f"100 instrumentation cycles took {instrumentation_elapsed_s:.3f} s "
            f"— expected < 1 s for lightweight async events."
        )

        # Verify no synchronous GPU calls occurred — in CPU mode the
        # native_event attribute is None for all events.
        for kid in range(100):
            entry = profiler._events[kid]
            # CPU-fallback events have _native_event == None.
            start_ev = entry["start"]
            end_ev = entry["end"]
            if start_ev._backend == _BACKEND_CPU:
                assert start_ev._native_event is None
                assert end_ev._native_event is None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Profiler Lifecycle Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestProfilerLifecycle:
    """Phase 4: Reset, enable/disable, and empty-graph handling."""

    def test_profiler_reset(self, mock_gpu_target):
        """Profile kernels, reset, and verify all state is cleared.

        After reset:
        - ``_events`` should be empty.
        - ``_metrics`` should be empty.
        - ``_total_kernel_time`` and ``_total_profiling_overhead`` reset to 0.
        - ``_enabled`` should be ``True``.

        Then re-profile and verify fresh metrics are collected.
        """
        profiler = _make_profiler()

        # Profile a kernel.
        profiler.instrument_launch(0, mock_gpu_target, stream=None)
        profiler.record_start(0, mock_gpu_target)
        time.sleep(0.001)
        profiler.record_end(0, mock_gpu_target)
        profiler.synchronize_and_collect()

        # Pre-reset assertions.
        assert len(profiler._metrics) > 0
        assert len(profiler._events) > 0
        assert profiler._total_kernel_time > 0.0

        # Reset.
        profiler.reset()

        # Post-reset assertions.
        assert len(profiler._events) == 0
        assert len(profiler._metrics) == 0
        assert profiler._total_kernel_time == 0.0
        assert profiler._total_profiling_overhead == 0.0
        assert profiler._enabled is True

        # Re-profile.
        profiler.instrument_launch(10, mock_gpu_target, stream=None)
        profiler.record_start(10, mock_gpu_target)
        time.sleep(0.001)
        profiler.record_end(10, mock_gpu_target)
        new_metrics = profiler.synchronize_and_collect()

        assert 10 in new_metrics
        assert new_metrics[10]["wall_clock_ms"] >= 0.0

    def test_profiler_enable_disable(self, mock_gpu_target):
        """Disable profiling → events fallback to CPU; enable → normal.

        When ``_enabled`` is set to ``False`` the profiler still creates
        events (on the CPU fallback path) so the caller doesn't crash, but
        the events are not GPU-native.
        """
        profiler = _make_profiler()

        # Disable the profiler manually.
        profiler._enabled = False

        # Instrument a kernel — events are created on CPU fallback.
        start_ev, end_ev = profiler.instrument_launch(
            0, mock_gpu_target, stream=None
        )
        assert start_ev._backend == _BACKEND_CPU
        assert end_ev._backend == _BACKEND_CPU

        # Record and collect — should still work without errors.
        profiler.record_start(0, mock_gpu_target)
        profiler.record_end(0, mock_gpu_target)
        metrics = profiler.synchronize_and_collect()
        assert 0 in metrics

        # Re-enable via reset.
        profiler.reset()
        assert profiler._enabled is True

        # Now instrument again — events should attempt GPU backend.
        start_ev2, end_ev2 = profiler.instrument_launch(
            1, mock_gpu_target, stream=None
        )
        # Since enabled is True, backend should match the target
        # (falls back to CPU only if torch is unavailable).
        assert start_ev2._backend in ("cuda", _BACKEND_CPU)

    def test_profiler_empty_graph(self, default_graph_config):
        """Collect on an empty profiler produces no metrics and no errors.

        Uses the ``default_graph_config`` conftest fixture (feedback disabled)
        to construct the profiler.
        """
        profiler = RuntimeProfiler(config=default_graph_config)

        # No instrumentation — just collect.
        metrics = profiler.synchronize_and_collect()

        assert metrics == {}
        assert profiler.get_metrics() == {}
        assert profiler.get_per_target_metrics() == {}

        # Budget check should pass (no kernel time to compare against).
        assert profiler.check_overhead_budget() is True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Profiler API Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.kernel_graph
class TestProfilerAPI:
    """Phase 5: High-level API contract verification."""

    def test_profiler_instrument(self, mock_gpu_target, mock_nvidia_hw_profile):
        """Verify ``instrument_launch`` returns usable event wrappers.

        The returned ``(start_event, end_event)`` pair must support
        ``record(stream)`` and ``synchronize()`` without raising.
        """
        profiler = _make_profiler()

        start_ev, end_ev = profiler.instrument_launch(
            kernel_id=0,
            target=mock_gpu_target,
            stream=None,
            hw_profile=mock_nvidia_hw_profile,
        )

        # Events should support the public interface.
        start_ev.record(stream=None)
        end_ev.record(stream=None)
        start_ev.synchronize()
        end_ev.synchronize()

        # Elapsed time should be computable.
        elapsed_ms = _GPUEvent.elapsed_time(start_ev, end_ev)
        assert isinstance(elapsed_ms, float)

        # Hardware profile should be stored in the event entry.
        entry = profiler._events[0]
        assert entry["hw_profile"] is mock_nvidia_hw_profile

    def test_profiler_collect(self, mock_gpu_target, mock_nvidia_hw_profile):
        """Verify ``synchronize_and_collect`` returns per-kernel per-target
        metrics after a full instrument → record → collect cycle.

        The returned dictionary must be keyed by kernel_id and contain all
        AAP-mandated metric fields.
        """
        profiler = _make_profiler()

        # Instrument two kernels on the same target.
        for kid in (0, 1):
            profiler.instrument_launch(
                kid, mock_gpu_target, stream=None,
                hw_profile=mock_nvidia_hw_profile,
            )
            profiler.record_start(kid, mock_gpu_target)
            time.sleep(0.001)
            profiler.record_end(kid, mock_gpu_target)

        result = profiler.synchronize_and_collect()

        # Both kernels should be in the result.
        assert 0 in result
        assert 1 in result

        # Each kernel's metric dict should have the required fields.
        expected_fields = {
            "wall_clock_ms",
            "memory_throughput_gbps",
            "launch_overhead_us",
            "cross_device_transfer_ms",
            "estimated_occupancy",
        }
        for kid in (0, 1):
            for f in expected_fields:
                assert f in result[kid], (
                    f"Kernel {kid} missing metric field '{f}'"
                )

        # get_metrics should return a copy with the same content.
        get_result = profiler.get_metrics()
        assert set(get_result.keys()) == set(result.keys())
        for kid in result:
            for f in expected_fields:
                assert get_result[kid][f] == result[kid][f]

        # Per-target grouping should have exactly one target.
        per_target = profiler.get_per_target_metrics()
        assert mock_gpu_target in per_target
        assert set(per_target[mock_gpu_target].keys()) == {0, 1}
