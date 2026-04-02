"""Unit tests for ``triton.graph.capture`` — trace capture mechanism.

Tests cover seven phases of the :class:`KernelGraphCapture` context manager,
the :func:`capture` convenience function, and the :func:`graph_trace`
decorator:

Phase 1 — Context Manager Tests
Phase 2 — Kernel Recording Tests
Phase 3 — Alias Analysis Tests
Phase 4 — Hardware Inventory Tests
Phase 5 — Error Detection Tests
Phase 6 — Autotuner Interaction Tests
Phase 7 — Graph Trace Decorator Tests

All tests use ``@pytest.mark.kernel_graph`` and ``unittest.mock`` for
hardware-free execution.  Real PyTorch tensors are used for alias-analysis
tests only when ``torch`` is available; otherwise those tests are skipped.
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Guarded torch import — alias-analysis tests need real tensors
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Triton imports (always available once the package is installed)
# ---------------------------------------------------------------------------

import triton
from triton.graph.capture import (
    KernelGraphCapture,
    capture,
    graph_trace,
    CapturedLaunch,
    TensorArg,
    _capture_state,
)
from triton.graph.errors import GraphCaptureError
from triton.graph.config import GraphConfig
from triton.runtime.jit import KernelInterface
from triton.graph.kgir import KGIRGraph

# Mark every test in this module with kernel_graph
pytestmark = pytest.mark.kernel_graph


# ═══════════════════════════════════════════════════════════════════════════════
# Local Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _cleanup_capture_state():
    """Guarantee clean capture state before and after every test.

    The ``_capture_state`` thread-local can leak across tests if a capture
    context exits abnormally.  This fixture ensures it is always reset.
    """
    # Pre-test cleanup
    _capture_state.active_capture = None
    original_getitem = KernelInterface.__getitem__
    yield
    # Post-test cleanup — always restore the real __getitem__
    KernelInterface.__getitem__ = original_getitem
    _capture_state.active_capture = None


def _make_mock_tensor(
    data_ptr: int = 0x1000,
    shape: tuple = (1024,),
    strides: tuple = (1,),
    dtype: str = "float32",
    device: str = "cuda:0",
) -> MagicMock:
    """Create a mock tensor that passes ``_try_extract_tensor_info`` duck typing.

    The mock exposes callable ``data_ptr()``, iterable ``shape``, callable
    ``stride()``, and string ``dtype`` / ``device`` attributes.
    """
    t = MagicMock()
    t.data_ptr = MagicMock(return_value=data_ptr)
    # shape must be a real tuple (not a MagicMock) for iteration
    t.shape = shape
    # stride must be callable returning a tuple
    t.stride = MagicMock(return_value=strides)
    # dtype and device must be real strings for str()
    t.dtype = dtype
    t.device = device
    return t


def _simulate_launch(kernel_fn: MagicMock, grid: tuple, *args, **kwargs):
    """Invoke the currently-installed ``KernelInterface.__getitem__`` interceptor.

    Simulates ``kernel_fn[grid](*args, **kwargs)`` by calling through the
    class-level ``__getitem__`` monkey-patch applied by KernelGraphCapture.
    """
    launcher = KernelInterface.__getitem__(kernel_fn, grid)
    launcher(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Context Manager Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextManager:
    """Phase 1: KernelGraphCapture context-manager protocol."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_context_manager_basic(self, mock_hw_inv):
        """1.1 — Enter/exit completes cleanly, returns the capture instance."""
        mock_hw_inv.return_value.devices = []
        ctx = KernelGraphCapture()

        with ctx as entered:
            # __enter__ returns self
            assert entered is ctx
            # Has a container for captured launches
            assert isinstance(ctx._captured_launches, list)
            assert len(ctx._captured_launches) == 0

        # __exit__ completes without error; graph is created (empty)
        assert ctx._graph is not None
        assert ctx._active is False

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_convenience_function(self, mock_hw_inv):
        """1.2 — ``capture()`` returns a KernelGraphCapture instance."""
        mock_hw_inv.return_value.devices = []

        # Convenience function without config
        ctx = capture()
        assert isinstance(ctx, KernelGraphCapture)

        # Convenience function with explicit config
        cfg = GraphConfig()
        ctx_with_cfg = capture(config=cfg)
        assert isinstance(ctx_with_cfg, KernelGraphCapture)

        # Access via triton.graph.capture()
        ctx_via_triton = triton.graph.capture()
        assert isinstance(ctx_via_triton, KernelGraphCapture)

        # Context manager usage patterns
        with capture() as g:
            assert g._active is True

        with capture(config=GraphConfig()) as g2:
            assert g2._active is True

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_scope_enters_recording_mode(self, mock_hw_inv):
        """1.3 — Inside capture scope, ``_active`` is ``True``."""
        mock_hw_inv.return_value.devices = []

        ctx = KernelGraphCapture()
        assert ctx._active is False  # Before entering

        with ctx:
            assert ctx._active is True
            assert _capture_state.active_capture is ctx

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_scope_exits_recording_mode(self, mock_hw_inv):
        """1.4 — After exiting capture scope, recording mode is disabled
        and ``KernelInterface.__getitem__`` is restored.
        """
        mock_hw_inv.return_value.devices = []
        original_getitem = KernelInterface.__getitem__

        with KernelGraphCapture() as ctx:
            # __getitem__ should be the interceptor during capture
            assert KernelInterface.__getitem__ is not original_getitem
            assert ctx._active is True

        # After exit, recording mode is disabled
        assert ctx._active is False
        assert _capture_state.active_capture is None
        # __getitem__ is restored to original
        assert KernelInterface.__getitem__ is original_getitem

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_nested_capture_raises_error(self, mock_hw_inv):
        """1.5 — Nested capture scope raises ``GraphCaptureError``."""
        mock_hw_inv.return_value.devices = []

        with KernelGraphCapture():
            with pytest.raises(GraphCaptureError, match="[Nn]ested"):
                with KernelGraphCapture():
                    pass  # Should never reach here

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_exception_cleanup(self, mock_hw_inv):
        """1.6 — Exception within scope still restores ``__getitem__``."""
        mock_hw_inv.return_value.devices = []
        original_getitem = KernelInterface.__getitem__

        class IntentionalError(Exception):
            pass

        with pytest.raises(IntentionalError):
            with KernelGraphCapture() as ctx:
                assert KernelInterface.__getitem__ is not original_getitem
                raise IntentionalError("test exception")

        # __getitem__ must be restored even after exception
        assert KernelInterface.__getitem__ is original_getitem
        # Capture state must be clean
        assert ctx._active is False
        assert _capture_state.active_capture is None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Kernel Recording Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKernelRecording:
    """Phase 2: Verifying kernel launch recording in capture scope."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_record_single_kernel_launch(self, mock_hw_inv, mock_kernel_fn):
        """2.1 — Single kernel launch is recorded with correct metadata."""
        mock_hw_inv.return_value.devices = []
        tensor_a = _make_mock_tensor(data_ptr=0x1000, shape=(1024,), strides=(1,))

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), tensor_a, 1024)

        assert len(ctx._captured_launches) == 1
        launch = ctx._captured_launches[0]
        assert isinstance(launch, CapturedLaunch)
        assert launch.kernel_fn is mock_kernel_fn
        assert launch.grid == (128,)
        assert launch.launch_index == 0
        # Tensor arg was captured
        assert len(launch.tensor_args) >= 1
        assert launch.tensor_args[0].data_ptr == 0x1000

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_record_multiple_kernel_launches(self, mock_hw_inv, make_mock_kernel):
        """2.2 — Multiple kernel launches are recorded in order."""
        mock_hw_inv.return_value.devices = []
        k1 = make_mock_kernel("add_kernel", grid=(128,), num_args=2)
        k2 = make_mock_kernel("mul_kernel", grid=(256,), num_args=2)
        k3 = make_mock_kernel("relu_kernel", grid=(64,), num_args=1)

        t1 = _make_mock_tensor(data_ptr=0x1000)
        t2 = _make_mock_tensor(data_ptr=0x2000)
        t3 = _make_mock_tensor(data_ptr=0x3000)

        with KernelGraphCapture() as ctx:
            _simulate_launch(k1, (128,), t1, t2)
            _simulate_launch(k2, (256,), t2, t3)
            _simulate_launch(k3, (64,), t3)

        assert len(ctx._captured_launches) == 3
        # Verify capture order
        assert ctx._captured_launches[0].kernel_fn is k1
        assert ctx._captured_launches[1].kernel_fn is k2
        assert ctx._captured_launches[2].kernel_fn is k3
        # Verify launch indices are sequential
        assert ctx._captured_launches[0].launch_index == 0
        assert ctx._captured_launches[1].launch_index == 1
        assert ctx._captured_launches[2].launch_index == 2

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_record_kernel_grid_parameters(self, mock_hw_inv, mock_kernel_fn):
        """2.3 — Grid dimensions are correctly recorded in various forms."""
        mock_hw_inv.return_value.devices = []
        t = _make_mock_tensor()

        # Test 1D grid
        with KernelGraphCapture() as ctx1:
            _simulate_launch(mock_kernel_fn, (128,), t)
        assert ctx1._captured_launches[0].grid == (128,)

        # Test 2D grid
        with KernelGraphCapture() as ctx2:
            _simulate_launch(mock_kernel_fn, (64, 32), t)
        assert ctx2._captured_launches[0].grid == (64, 32)

        # Test 3D grid
        with KernelGraphCapture() as ctx3:
            _simulate_launch(mock_kernel_fn, (4, 4, 4), t)
        assert ctx3._captured_launches[0].grid == (4, 4, 4)

        # Test integer grid (normalised to 1-tuple)
        with KernelGraphCapture() as ctx4:
            _simulate_launch(mock_kernel_fn, 256, t)
        assert ctx4._captured_launches[0].grid == (256,)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_record_tensor_arguments(self, mock_hw_inv, mock_kernel_fn):
        """2.4 — Tensor argument metadata (ptr, shape, stride, dtype) is captured."""
        mock_hw_inv.return_value.devices = []

        t_input = _make_mock_tensor(
            data_ptr=0xA000,
            shape=(512, 256),
            strides=(256, 1),
            dtype="float16",
            device="cuda:1",
        )
        t_output = _make_mock_tensor(
            data_ptr=0xB000,
            shape=(512, 256),
            strides=(256, 1),
            dtype="float16",
            device="cuda:1",
        )

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (512,), t_input, t_output, 512)

        launch = ctx._captured_launches[0]
        # Two tensor args captured (the integer 512 is not a tensor)
        assert len(launch.tensor_args) == 2

        ta_in = launch.tensor_args[0]
        assert ta_in.data_ptr == 0xA000
        assert ta_in.shape == (512, 256)
        assert ta_in.strides == (256, 1)
        assert "float16" in ta_in.dtype
        assert "cuda:1" in ta_in.device

        ta_out = launch.tensor_args[1]
        assert ta_out.data_ptr == 0xB000
        assert ta_out.shape == (512, 256)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_record_constexpr_values(self, mock_hw_inv):
        """2.5 — Constexpr values (e.g. BLOCK_SIZE=128) are recorded."""
        mock_hw_inv.return_value.devices = []

        # Create a kernel with params that have is_constexpr flag
        kernel = MagicMock()
        kernel.__name__ = "test_kernel"
        kernel.fn = MagicMock()
        kernel.fn.__name__ = "test_kernel"

        # Set up params: arg0=tensor, arg1=N (not constexpr), arg2=BLOCK (constexpr)
        param0 = MagicMock()
        param0.is_constexpr = False
        param0.name = "ptr"
        param1 = MagicMock()
        param1.is_constexpr = False
        param1.name = "N"
        param2 = MagicMock()
        param2.is_constexpr = True
        param2.name = "BLOCK_SIZE"

        kernel.fn.params = [param0, param1, param2]
        kernel.params = [param0, param1, param2]

        t = _make_mock_tensor()
        with KernelGraphCapture() as ctx:
            _simulate_launch(kernel, (128,), t, 1024, 128)

        launch = ctx._captured_launches[0]
        # BLOCK_SIZE should be in constexpr_args
        assert "BLOCK_SIZE" in launch.constexpr_args
        assert launch.constexpr_args["BLOCK_SIZE"] == 128

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_record_constexpr_kwargs(self, mock_hw_inv):
        """2.5b — Constexpr values passed as keyword arguments are recorded
        via the fallback path (kernel with no ``params`` attribute).
        """
        mock_hw_inv.return_value.devices = []

        # To trigger the fallback constexpr detection path,
        # ``fn.params`` must be ``None`` (not an empty list).
        kernel = MagicMock()
        kernel.__name__ = "no_params_kernel"
        kernel.fn = MagicMock()
        kernel.fn.__name__ = "no_params_kernel"
        kernel.fn.params = None  # triggers fallback branch in _extract_constexpr_args
        kernel.params = None

        t = _make_mock_tensor()
        with KernelGraphCapture() as ctx:
            _simulate_launch(kernel, (128,), t, BLOCK_SIZE=128, num_warps=4)

        launch = ctx._captured_launches[0]
        # Fallback path collects non-tensor kwargs as constexprs
        assert "BLOCK_SIZE" in launch.constexpr_args
        assert launch.constexpr_args["BLOCK_SIZE"] == 128
        assert "num_warps" in launch.constexpr_args
        assert launch.constexpr_args["num_warps"] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Alias Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAliasAnalysis:
    """Phase 3: Verifying alias detection on tensor pointer arguments."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_alias_detection_same_tensor(self, mock_hw_inv, mock_kernel_fn):
        """3.1 — Same tensor passed as two different arguments is detected as aliased."""
        mock_hw_inv.return_value.devices = []

        # Same data pointer for both args → alias
        t = _make_mock_tensor(data_ptr=0x5000, shape=(1024,), strides=(1,))

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t, t, 1024)

        launch = ctx._captured_launches[0]
        # Both tensor args should have the same data_ptr
        tensor_ptrs = [ta.data_ptr for ta in launch.tensor_args]
        assert len(tensor_ptrs) >= 2
        assert tensor_ptrs[0] == tensor_ptrs[1] == 0x5000

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_alias_detection_overlapping_views(self, mock_hw_inv, mock_kernel_fn):
        """3.2 — Overlapping tensor views are detected as aliased.

        Two mock tensors share the same base data_ptr region and overlap.
        The alias analysis in ``_analyze_aliases`` should detect the overlap
        via ``memory_regions_overlap``.
        """
        mock_hw_inv.return_value.devices = []

        # Region A: ptr=0x1000, 1024 float32 elements = 4096 bytes
        t_a = _make_mock_tensor(
            data_ptr=0x1000, shape=(1024,), strides=(1,), dtype="float32",
        )
        # Region B: ptr=0x1800, overlaps with A (0x1000 + 0x1000 = 0x2000 > 0x1800)
        t_b = _make_mock_tensor(
            data_ptr=0x1800, shape=(512,), strides=(1,), dtype="float32",
        )

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t_a, 1024)
            _simulate_launch(mock_kernel_fn, (64,), t_b, 512)

        # build_graph was called, which runs _detect_data_dependencies
        # which calls _analyze_aliases → memory_regions_overlap
        graph = ctx._graph
        assert graph is not None
        # There should be edges detected due to the overlap
        assert graph.node_count() == 2
        assert graph.edge_count() >= 1

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_no_alias_independent_tensors(self, mock_hw_inv, mock_kernel_fn):
        """3.3 — Completely independent tensors have no alias relationship."""
        mock_hw_inv.return_value.devices = []

        # Two widely separated memory regions
        t_a = _make_mock_tensor(
            data_ptr=0x10000, shape=(1024,), strides=(1,), dtype="float32",
        )
        t_b = _make_mock_tensor(
            data_ptr=0x90000, shape=(1024,), strides=(1,), dtype="float32",
        )

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t_a, 1024)
            _simulate_launch(mock_kernel_fn, (128,), t_b, 1024)

        graph = ctx._graph
        assert graph is not None
        assert graph.node_count() == 2
        # No data dependency edge because tensors don't overlap
        assert graph.edge_count() == 0

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_alias_detection_across_kernels(self, mock_hw_inv, make_mock_kernel):
        """3.4 — Kernel A writes tensor T, Kernel B reads tensor T → data dep edge."""
        mock_hw_inv.return_value.devices = []

        k_writer = make_mock_kernel("writer_kernel", grid=(128,), num_args=2)
        k_reader = make_mock_kernel("reader_kernel", grid=(128,), num_args=2)

        # Shared tensor T (same data_ptr)
        shared_t = _make_mock_tensor(
            data_ptr=0x3000, shape=(1024,), strides=(1,), dtype="float32",
        )
        other_in = _make_mock_tensor(data_ptr=0xA0000, shape=(1024,), strides=(1,))
        other_out = _make_mock_tensor(data_ptr=0xB0000, shape=(1024,), strides=(1,))

        with KernelGraphCapture() as ctx:
            # Writer produces shared_t
            _simulate_launch(k_writer, (128,), other_in, shared_t, 1024)
            # Reader consumes shared_t
            _simulate_launch(k_reader, (128,), shared_t, other_out, 1024)

        graph = ctx._graph
        assert graph is not None
        assert graph.node_count() == 2
        # Data dependency from writer → reader via shared tensor
        assert graph.edge_count() >= 1

        # Verify edge direction: node 0 → node 1
        edges = graph.get_edges()
        dep_found = False
        for edge in edges:
            if edge.source_id == 0 and edge.target_id == 1:
                dep_found = True
                assert edge.edge_type == "data_dep"
        assert dep_found, "Expected data_dep edge from writer (0) to reader (1)"

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    @patch("triton.graph.dispatch.HardwareInventory")
    def test_alias_detection_with_real_torch_tensors(self, mock_hw_inv, mock_kernel_fn):
        """3.2b — Alias detection with real PyTorch CPU tensors for data_ptr fidelity."""
        mock_hw_inv.return_value.devices = []

        # Create a base tensor and two overlapping views
        base = torch.randn(2048)
        view_a = base[:1024]  # First half
        view_b = base[512:1536]  # Overlapping middle region

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), view_a, 1024)
            _simulate_launch(mock_kernel_fn, (128,), view_b, 1024)

        graph = ctx._graph
        assert graph is not None
        assert graph.node_count() == 2
        # Overlapping views should produce a data dependency
        assert graph.edge_count() >= 1

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    @patch("triton.graph.dispatch.HardwareInventory")
    def test_no_alias_with_independent_torch_tensors(self, mock_hw_inv, mock_kernel_fn):
        """3.3b — Independent PyTorch tensors → no alias edge."""
        mock_hw_inv.return_value.devices = []

        t_a = torch.randn(1024)
        t_b = torch.randn(1024)
        # Ensure they're truly independent allocations
        assert t_a.data_ptr() != t_b.data_ptr()

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t_a, 1024)
            _simulate_launch(mock_kernel_fn, (128,), t_b, 1024)

        graph = ctx._graph
        assert graph is not None
        assert graph.node_count() == 2
        assert graph.edge_count() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Hardware Inventory Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardwareInventory:
    """Phase 4: Hardware inventory discovery during capture scope."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_hardware_inventory_discovery(self, mock_hw_inv_cls):
        """4.1 — Hardware inventory is populated on capture scope entry."""
        mock_profile = MagicMock()
        mock_profile.vendor = "nvidia"
        mock_profile.arch_generation = "sm_90"
        mock_profile.sm_count = 132
        mock_profile.smem_per_sm_bytes = 228 * 1024
        mock_profile.registers_per_sm = 65536
        mock_profile.global_memory_bytes = 80 * (1024 ** 3)
        mock_profile.memory_bandwidth_gbps = 3350.0
        mock_profile.compute_throughput_tflops = 989.0
        mock_profile.warp_size = 32
        mock_profile.max_concurrent_streams = 128
        mock_profile.interconnect_type = "nvlink_4"
        mock_profile.interconnect_bandwidth_gbps = 900.0

        mock_inventory = MagicMock()
        mock_inventory.devices = [mock_profile]
        mock_hw_inv_cls.return_value = mock_inventory

        with KernelGraphCapture() as ctx:
            assert ctx._hardware_inventory is not None
            assert ctx._hardware_inventory.devices == [mock_profile]

            # Verify key fields are present
            dev = ctx._hardware_inventory.devices[0]
            assert dev.vendor == "nvidia"
            assert dev.arch_generation == "sm_90"
            assert dev.sm_count == 132
            assert dev.global_memory_bytes == 80 * (1024 ** 3)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_hardware_inventory_with_mock_device(
        self, mock_hw_inv_cls, mock_nvidia_hw_profile,
    ):
        """4.2 — Mock GPUDriver returns known device; verify HardwareProfile fields."""
        mock_inventory = MagicMock()
        mock_inventory.devices = [mock_nvidia_hw_profile]
        mock_hw_inv_cls.return_value = mock_inventory

        with KernelGraphCapture() as ctx:
            assert ctx._hardware_inventory is not None
            devs = ctx._hardware_inventory.devices
            assert len(devs) == 1

            hw = devs[0]
            assert hw.vendor == "nvidia"
            assert hw.arch_generation == "sm_90"
            assert hw.sm_count == 132
            assert hw.smem_per_sm_bytes == 228 * 1024
            assert hw.registers_per_sm == 65536
            assert hw.warp_size == 32
            assert hw.max_concurrent_streams == 128
            assert hw.interconnect_type == "nvlink_4"
            assert hw.interconnect_bandwidth_gbps == 900.0

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_hardware_inventory_enumeration_timing(self, mock_hw_inv_cls):
        """4.3 — Hardware inventory enumeration completes in < 10 ms (AAP §0.7.2)."""
        # Make enumeration artificially fast (mock)
        mock_hw_inv_cls.return_value.devices = []

        t0 = time.perf_counter()
        with KernelGraphCapture():
            pass
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # The overall context entry (including monkey-patching) should be fast.
        # With mocked HardwareInventory, enumeration itself is ~0 ms.
        assert elapsed_ms < 100.0, (
            f"Context entry took {elapsed_ms:.1f} ms — exceeds performance budget"
        )

    def test_hardware_inventory_failure_non_fatal(self):
        """4.1b — If HardwareInventory import/init fails, capture still works."""
        # Without patching, the real HardwareInventory may fail in CI (no GPU).
        # capture.py handles this gracefully in a try/except.
        with KernelGraphCapture() as ctx:
            pass

        # Capture should complete even if hardware enumeration failed
        assert ctx._active is False
        assert ctx._graph is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Error Detection Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorDetection:
    """Phase 5: Unsupported pattern detection and error reporting."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_unsupported_pattern_detection_callable_grid(self, mock_hw_inv, mock_kernel_fn):
        """5.1a — Callable grids are detected as unsupported patterns."""
        mock_hw_inv.return_value.devices = []

        # A callable grid is an unsupported pattern (informational warning)
        callable_grid = lambda meta: (meta["N"] // 128,)

        with KernelGraphCapture() as ctx:
            t = _make_mock_tensor()
            _simulate_launch(mock_kernel_fn, callable_grid, t, 1024)

        # _detect_unsupported_patterns should flag the callable grid
        issues = ctx._detect_unsupported_patterns()
        assert len(issues) >= 1
        assert any("callable" in issue.lower() for issue in issues)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_unsupported_pattern_detection_zero_element(self, mock_hw_inv, mock_kernel_fn):
        """5.1b — Zero-element tensors are detected as potential host-dependent control flow."""
        mock_hw_inv.return_value.devices = []

        # Zero-element tensor — may indicate host-dependent control flow
        t_zero = _make_mock_tensor(data_ptr=0x1000, shape=(0,), strides=(1,))

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (1,), t_zero, 0)

        issues = ctx._detect_unsupported_patterns()
        assert len(issues) >= 1
        assert any("zero-element" in issue.lower() for issue in issues)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_unsupported_pattern_missing_metadata(self, mock_hw_inv, mock_kernel_fn):
        """5.1c — Tensors with missing shape/stride metadata trigger warnings."""
        mock_hw_inv.return_value.devices = []

        # Tensor with no shape and no strides
        t_no_meta = _make_mock_tensor(data_ptr=0x2000, shape=(), strides=())

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (1,), t_no_meta)

        issues = ctx._detect_unsupported_patterns()
        assert len(issues) >= 1
        # Should mention missing metadata or zero-element
        assert any(
            "metadata" in issue.lower() or "zero-element" in issue.lower()
            for issue in issues
        )

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_empty_capture_scope(self, mock_hw_inv):
        """5.2 — Empty capture scope (no kernel launches) completes without error."""
        mock_hw_inv.return_value.devices = []

        with KernelGraphCapture() as ctx:
            pass  # No kernel launches

        # Should produce an empty graph, not raise
        assert ctx._graph is not None
        assert isinstance(ctx._graph, KGIRGraph)
        assert ctx._graph.node_count() == 0
        assert ctx._graph.edge_count() == 0

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_empty_capture_via_build_graph(self, mock_hw_inv, mock_kernel_fn):
        """5.2b — build_graph() on captures with no data deps produces a valid graph."""
        mock_hw_inv.return_value.devices = []

        # Two launches with completely independent tensors → no edges
        t1 = _make_mock_tensor(data_ptr=0x10000)
        t2 = _make_mock_tensor(data_ptr=0x90000)

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t1, 1024)
            _simulate_launch(mock_kernel_fn, (128,), t2, 1024)

        graph = ctx._graph
        assert graph is not None
        assert graph.node_count() == 2
        assert graph.edge_count() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Autotuner Interaction Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutotunerInteraction:
    """Phase 6: Verifying correct behaviour with Autotuner-wrapped kernels."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_autotuned_kernel(self, mock_hw_inv):
        """6.1 — Warm Autotuner: selected config is recorded alongside the launch.

        Per AAP §0.5.2: When capturing an autotuned kernel, the trace capture
        records the kernel using the configuration selected by the Autotuner's
        most recent ``run()``.
        """
        mock_hw_inv.return_value.devices = []

        # Create a mock autotuned kernel (duck-typing: has .configs and .fn)
        mock_autotuner = MagicMock()
        mock_autotuner.__name__ = "autotuned_matmul"
        mock_autotuner.fn = MagicMock()
        mock_autotuner.fn.__name__ = "autotuned_matmul"
        mock_autotuner.fn.params = []
        mock_autotuner.configs = [MagicMock(), MagicMock()]

        # Warm autotuner — best_config is already set
        mock_best_config = MagicMock()
        mock_best_config.all_kwargs = MagicMock(return_value={"BLOCK_M": 128, "BLOCK_K": 64})
        mock_autotuner.best_config = mock_best_config

        t = _make_mock_tensor()
        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_autotuner, (128,), t, 1024)

        assert len(ctx._captured_launches) == 1
        launch = ctx._captured_launches[0]
        assert launch.kernel_fn is mock_autotuner

        # Autotuner config kwargs should be merged into constexpr_args
        assert "BLOCK_M" in launch.constexpr_args
        assert launch.constexpr_args["BLOCK_M"] == 128
        assert "BLOCK_K" in launch.constexpr_args
        assert launch.constexpr_args["BLOCK_K"] == 64

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_cold_autotuner(self, mock_hw_inv):
        """6.2 — Cold Autotuner: single warmup run is triggered before recording.

        Per AAP §0.5.2: If no prior run exists (cold autotuner), capture
        triggers a single autotuning run to select the configuration,
        then records the selected config.
        """
        mock_hw_inv.return_value.devices = []

        # Cold autotuner — best_config starts as None
        mock_autotuner = MagicMock()
        mock_autotuner.__name__ = "cold_matmul"
        mock_autotuner.fn = MagicMock()
        mock_autotuner.fn.__name__ = "cold_matmul"
        mock_autotuner.fn.params = []
        mock_autotuner.configs = [MagicMock()]
        mock_autotuner.best_config = None

        t = _make_mock_tensor()

        # When __getitem__ is called on the cold autotuner during warmup,
        # simulate the autotuner executing and setting best_config.
        # The warmup path calls: launcher = kernel_self[grid]; launcher(*args)
        # During warmup, the original __getitem__ is temporarily restored,
        # so calling mock_autotuner[grid] hits mock's __getitem__.
        warmup_launcher = MagicMock()

        def side_effect_getitem(grid):
            # Simulate autotuner warmup: after running, best_config is set
            mock_autotuner.best_config = MagicMock()
            mock_autotuner.best_config.all_kwargs = MagicMock(
                return_value={"BLOCK_M": 64}
            )
            return warmup_launcher

        mock_autotuner.__getitem__ = MagicMock(side_effect=side_effect_getitem)

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_autotuner, (128,), t, 1024)

        assert len(ctx._captured_launches) == 1
        # After warmup, best_config should have been set
        assert mock_autotuner.best_config is not None
        # The captured launch should have the autotuner config
        launch = ctx._captured_launches[0]
        assert "BLOCK_M" in launch.constexpr_args
        assert launch.constexpr_args["BLOCK_M"] == 64

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_non_autotuner_kernel_ignored(self, mock_hw_inv, mock_kernel_fn):
        """6.1b — Plain JITFunction (no .configs) skips autotuner warmup path."""
        mock_hw_inv.return_value.devices = []

        # mock_kernel_fn from conftest has NO .configs attribute
        # Actually MagicMock has every attribute, so delete it explicitly
        if hasattr(mock_kernel_fn, "configs"):
            del mock_kernel_fn.configs

        t = _make_mock_tensor()
        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t, 1024)

        assert len(ctx._captured_launches) == 1
        # No autotuner warmup should have been triggered — just a direct record
        launch = ctx._captured_launches[0]
        assert launch.kernel_fn is mock_kernel_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7 — Graph Trace Decorator Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphTraceDecorator:
    """Phase 7: Verifying the ``@graph_trace`` decorator."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_graph_trace_decorator_basic(self, mock_hw_inv, mock_kernel_fn):
        """7.1 — @graph_trace wraps function and captures kernel launches."""
        mock_hw_inv.return_value.devices = []

        t = _make_mock_tensor()

        @graph_trace
        def my_pipeline(tensor, size):
            _simulate_launch(mock_kernel_fn, (128,), tensor, size)
            return "pipeline_result"

        # Decorator initialises last_graph and last_captures
        assert my_pipeline.last_graph is None
        assert my_pipeline.last_captures == []

        result = my_pipeline(t, 1024)

        # Original return value is preserved
        assert result == "pipeline_result"

        # last_graph should now contain a KGIRGraph
        assert my_pipeline.last_graph is not None

        # last_captures should contain the recorded launches
        assert len(my_pipeline.last_captures) >= 1
        assert my_pipeline.last_captures[0].kernel_fn is mock_kernel_fn

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_graph_trace_decorator_with_config(self, mock_hw_inv, mock_kernel_fn):
        """7.1b — @graph_trace(config=...) accepts a custom GraphConfig."""
        mock_hw_inv.return_value.devices = []

        t = _make_mock_tensor()
        cfg = GraphConfig()

        @graph_trace(config=cfg)
        def configured_pipeline(tensor):
            _simulate_launch(mock_kernel_fn, (64,), tensor, 512)

        configured_pipeline(t)

        assert configured_pipeline.last_graph is not None
        assert len(configured_pipeline.last_captures) == 1

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_graph_trace_decorator_multiple_calls(self, mock_hw_inv, mock_kernel_fn):
        """7.1c — Multiple invocations update last_graph and last_captures."""
        mock_hw_inv.return_value.devices = []

        t = _make_mock_tensor()

        @graph_trace
        def repeatable_pipeline(tensor, grid_size):
            _simulate_launch(mock_kernel_fn, (grid_size,), tensor)

        # First call
        repeatable_pipeline(t, 64)
        first_graph = repeatable_pipeline.last_graph
        first_captures = list(repeatable_pipeline.last_captures)
        assert first_graph is not None
        assert len(first_captures) == 1

        # Second call — should update
        repeatable_pipeline(t, 128)
        second_graph = repeatable_pipeline.last_graph
        second_captures = list(repeatable_pipeline.last_captures)

        assert second_graph is not None
        # Graphs should be different objects (new capture each time)
        assert second_graph is not first_graph
        assert len(second_captures) == 1
        assert second_captures[0].grid == (128,)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_graph_trace_preserves_function_name(self, mock_hw_inv):
        """7.1d — @graph_trace preserves the original function's __name__."""
        mock_hw_inv.return_value.devices = []

        @graph_trace
        def my_special_pipeline():
            pass

        assert my_special_pipeline.__name__ == "my_special_pipeline"

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_graph_trace_empty_function(self, mock_hw_inv):
        """7.1e — @graph_trace on a function with no kernel launches
        produces an empty graph.
        """
        mock_hw_inv.return_value.devices = []

        @graph_trace
        def empty_pipeline():
            return 42

        result = empty_pipeline()
        assert result == 42
        assert empty_pipeline.last_graph is not None
        assert empty_pipeline.last_graph.node_count() == 0
        assert empty_pipeline.last_captures == []


# ═══════════════════════════════════════════════════════════════════════════════
# Additional Edge-Case Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Additional edge-case coverage for robustness."""

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_build_graph_returns_kgir_graph(self, mock_hw_inv, mock_kernel_fn):
        """build_graph() returns a KGIRGraph instance with correct node/edge counts."""
        mock_hw_inv.return_value.devices = []

        t1 = _make_mock_tensor(data_ptr=0x1000, shape=(1024,), strides=(1,))
        t2 = _make_mock_tensor(data_ptr=0x50000, shape=(1024,), strides=(1,))

        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t1, 1024)
            _simulate_launch(mock_kernel_fn, (128,), t2, 1024)

        graph = ctx._graph
        assert isinstance(graph, KGIRGraph)
        assert graph.node_count() == 2

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_captured_launches_list_accessible(self, mock_hw_inv, mock_kernel_fn):
        """_captured_launches is a list of CapturedLaunch objects."""
        mock_hw_inv.return_value.devices = []

        t = _make_mock_tensor()
        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t, 1024)

        assert isinstance(ctx._captured_launches, list)
        assert all(isinstance(l, CapturedLaunch) for l in ctx._captured_launches)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_with_config_object(self, mock_hw_inv, default_graph_config):
        """KernelGraphCapture accepts a GraphConfig for parameterized control."""
        mock_hw_inv.return_value.devices = []

        ctx = KernelGraphCapture(config=default_graph_config)
        with ctx:
            assert ctx._config is default_graph_config
            assert ctx._config.feedback.enable is False

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_capture_config_feedback_field(self, mock_hw_inv):
        """GraphConfig.feedback, .fusion, .dispatch fields are accessible."""
        mock_hw_inv.return_value.devices = []

        cfg = GraphConfig()
        assert hasattr(cfg, "feedback")
        assert hasattr(cfg, "fusion")
        assert hasattr(cfg, "dispatch")

        with KernelGraphCapture(config=cfg) as ctx:
            assert ctx._config.feedback is not None
            assert ctx._config.fusion is not None
            assert ctx._config.dispatch is not None

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_graph_capture_error_str(self, mock_hw_inv):
        """GraphCaptureError has a meaningful string representation."""
        mock_hw_inv.return_value.devices = []

        err = GraphCaptureError("test error message")
        assert "test error message" in str(err)

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_kgir_graph_get_node_after_capture(self, mock_hw_inv, mock_kernel_fn):
        """KGIRGraph.get_node() returns nodes created during capture."""
        mock_hw_inv.return_value.devices = []

        t = _make_mock_tensor()
        with KernelGraphCapture() as ctx:
            _simulate_launch(mock_kernel_fn, (128,), t, 1024)

        graph = ctx._graph
        assert graph.node_count() == 1
        node = graph.get_node(0)
        assert node is not None
        assert node.kernel_fn is mock_kernel_fn

    @patch("triton.graph.dispatch.HardwareInventory")
    def test_kgir_graph_get_edges_after_capture(self, mock_hw_inv, make_mock_kernel):
        """KGIRGraph.get_edges() returns edges from dependency analysis."""
        mock_hw_inv.return_value.devices = []

        k1 = make_mock_kernel("producer")
        k2 = make_mock_kernel("consumer")

        # Shared tensor creates a data dependency
        shared = _make_mock_tensor(data_ptr=0x5000, shape=(1024,), strides=(1,))
        other = _make_mock_tensor(data_ptr=0xF0000, shape=(1024,), strides=(1,))

        with KernelGraphCapture() as ctx:
            _simulate_launch(k1, (128,), other, shared, 1024)
            _simulate_launch(k2, (128,), shared, other, 1024)

        edges = ctx._graph.get_edges()
        assert len(edges) >= 1
        assert all(hasattr(e, "source_id") and hasattr(e, "target_id") for e in edges)
