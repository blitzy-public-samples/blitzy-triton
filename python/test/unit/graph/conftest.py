"""
Shared pytest fixtures for the ``python/test/unit/graph/`` test package.

Provides mock kernel factories, hardware inventory mocks, sample KGIR graphs,
mock HardwareProfile objects, and mock GPUTarget objects.  These fixtures
enable **hardware-free** testing of the graph-level cross-kernel optimization
layer.

All graph-module imports are **lazy** (performed inside individual fixture
bodies) so that this conftest loads even before the ``triton.graph`` package
is fully built.

The root ``python/test/conftest.py`` already exposes the ``device``,
``fresh_triton_cache``, and ``fresh_knobs`` fixtures — they are inherited
automatically and are **never** redefined here.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Phase 1 — Mock GPUTarget Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gpu_target():
    """Create a mock ``GPUTarget`` representing an NVIDIA H100 (sm_90).

    Attributes configured on the mock:
        backend  – ``"cuda"``
        arch     – ``90``
        warp_size – ``32``
    """
    target = MagicMock()
    target.backend = "cuda"
    target.arch = 90
    target.warp_size = 32
    target.__repr__ = lambda self: "GPUTarget(backend='cuda', arch=90, warp_size=32)"
    return target


@pytest.fixture
def mock_gpu_target_amd():
    """Create a mock ``GPUTarget`` representing an AMD MI300X (gfx942).

    Attributes configured on the mock:
        backend  – ``"hip"``
        arch     – ``"gfx942"``
        warp_size – ``64``
    """
    target = MagicMock()
    target.backend = "hip"
    target.arch = "gfx942"
    target.warp_size = 64
    target.__repr__ = lambda self: "GPUTarget(backend='hip', arch='gfx942', warp_size=64)"
    return target


# ---------------------------------------------------------------------------
# Phase 2 — Mock HardwareProfile Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_nvidia_hw_profile():
    """Create a mock ``HardwareProfile`` for an NVIDIA H100 (sm_90).

    All 12 mandatory fields are populated with realistic H100 values so that
    fusion analysis, memory-planning, and dispatch tests can exercise
    resource-limit logic without a physical GPU.
    """
    try:
        from triton.graph.kgir import HardwareProfile
        return HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=132,
            smem_per_sm_bytes=228 * 1024,           # 228 KB
            registers_per_sm=65536,
            global_memory_bytes=80 * (1024 ** 3),    # 80 GB
            memory_bandwidth_gbps=3350.0,
            compute_throughput_tflops=989.0,
            warp_size=32,
            max_concurrent_streams=128,
            interconnect_type="nvlink_4",
            interconnect_bandwidth_gbps=900.0,
        )
    except Exception:
        # Fallback to MagicMock when the graph module is not yet compiled
        profile = MagicMock()
        profile.vendor = "nvidia"
        profile.arch_generation = "sm_90"
        profile.sm_count = 132
        profile.smem_per_sm_bytes = 228 * 1024
        profile.registers_per_sm = 65536
        profile.global_memory_bytes = 80 * (1024 ** 3)
        profile.memory_bandwidth_gbps = 3350.0
        profile.compute_throughput_tflops = 989.0
        profile.warp_size = 32
        profile.max_concurrent_streams = 128
        profile.interconnect_type = "nvlink_4"
        profile.interconnect_bandwidth_gbps = 900.0
        return profile


@pytest.fixture
def mock_amd_hw_profile():
    """Create a mock ``HardwareProfile`` for an AMD MI300X (gfx942).

    All 12 mandatory fields are populated with realistic MI300X values.
    """
    try:
        from triton.graph.kgir import HardwareProfile
        return HardwareProfile(
            vendor="amd",
            arch_generation="gfx942",
            sm_count=304,                            # CUs
            smem_per_sm_bytes=64 * 1024,             # 64 KB LDS
            registers_per_sm=65536,
            global_memory_bytes=192 * (1024 ** 3),   # 192 GB
            memory_bandwidth_gbps=5300.0,
            compute_throughput_tflops=1307.0,
            warp_size=64,                            # wavefront
            max_concurrent_streams=128,
            interconnect_type="infinity_fabric",
            interconnect_bandwidth_gbps=896.0,
        )
    except Exception:
        profile = MagicMock()
        profile.vendor = "amd"
        profile.arch_generation = "gfx942"
        profile.sm_count = 304
        profile.smem_per_sm_bytes = 64 * 1024
        profile.registers_per_sm = 65536
        profile.global_memory_bytes = 192 * (1024 ** 3)
        profile.memory_bandwidth_gbps = 5300.0
        profile.compute_throughput_tflops = 1307.0
        profile.warp_size = 64
        profile.max_concurrent_streams = 128
        profile.interconnect_type = "infinity_fabric"
        profile.interconnect_bandwidth_gbps = 896.0
        return profile


@pytest.fixture
def mock_small_hw_profile():
    """Create a ``HardwareProfile`` with intentionally small resources.

    This profile has minimal shared memory (16 KB) and few SMs (8), so that
    fusion and memory-planning tests can exercise resource-overflow and
    promotion-fallback paths.
    """
    try:
        from triton.graph.kgir import HardwareProfile
        return HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_70",
            sm_count=8,
            smem_per_sm_bytes=16 * 1024,             # 16 KB
            registers_per_sm=16384,
            global_memory_bytes=4 * (1024 ** 3),     # 4 GB
            memory_bandwidth_gbps=480.0,
            compute_throughput_tflops=7.8,
            warp_size=32,
            max_concurrent_streams=16,
            interconnect_type="pcie_4",
            interconnect_bandwidth_gbps=31.5,
        )
    except Exception:
        profile = MagicMock()
        profile.vendor = "nvidia"
        profile.arch_generation = "sm_70"
        profile.sm_count = 8
        profile.smem_per_sm_bytes = 16 * 1024
        profile.registers_per_sm = 16384
        profile.global_memory_bytes = 4 * (1024 ** 3)
        profile.memory_bandwidth_gbps = 480.0
        profile.compute_throughput_tflops = 7.8
        profile.warp_size = 32
        profile.max_concurrent_streams = 16
        profile.interconnect_type = "pcie_4"
        profile.interconnect_bandwidth_gbps = 31.5
        return profile


# ---------------------------------------------------------------------------
# Phase 3 — Mock Kernel Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kernel_fn():
    """Create a mock Triton ``@triton.jit`` kernel function.

    The mock carries ``__name__``, ``fn`` (source function stub), and
    ``module`` attributes so that it can be passed to any API that
    introspects decorated kernel objects.
    """
    kernel = MagicMock()
    kernel.__name__ = "mock_kernel"
    kernel.fn = MagicMock()
    kernel.fn.__name__ = "mock_kernel"
    kernel.fn.__module__ = "test_module"
    kernel.module = "test_module"
    kernel.params = []
    return kernel


@pytest.fixture
def make_mock_kernel():
    """Factory fixture for creating multiple uniquely-named mock kernels.

    Usage::

        def test_example(make_mock_kernel):
            k1 = make_mock_kernel("matmul", grid=(64, 64), num_args=4)
            k2 = make_mock_kernel("softmax", grid=(256,), num_args=2)

    Parameters
    ----------
    name : str
        Kernel name (default ``"kernel"``).
    grid : tuple[int, ...]
        Launch grid dimensions (default ``(128,)``).
    num_args : int
        Number of positional arguments the kernel accepts (default ``3``).
    """

    def _make(name: str = "kernel", grid: tuple = (128,), num_args: int = 3):
        kernel = MagicMock()
        kernel.__name__ = name
        kernel.fn = MagicMock()
        kernel.fn.__name__ = name
        kernel.fn.__module__ = "test_module"
        kernel.module = "test_module"
        kernel.grid = grid
        kernel.num_args = num_args
        # Create mock positional arguments
        kernel.params = [MagicMock() for _ in range(num_args)]
        return kernel

    return _make


# ---------------------------------------------------------------------------
# Phase 4 — Sample KGIR Graph Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_kgir_node():
    """Factory fixture for creating ``KGIRNode``-compatible objects.

    Returns a callable that produces either real ``KGIRNode`` instances
    (when the graph module is importable) or MagicMock stand-ins with
    identical attribute layouts.

    Parameters
    ----------
    kernel_name : str
        Name assigned to the mock kernel (default ``"kernel_0"``).
    grid : tuple[int, ...]
        Grid dimensions (default ``(128,)``).
    tensor_shapes : dict | None
        Tensor shape mapping (default ``None`` → empty dict).
    smem_bytes : int
        Shared-memory usage in bytes (default ``0``).
    register_count : int
        Per-thread register count (default ``32``).
    execution_time_ms : float
        Estimated execution time in milliseconds (default ``1.0``).
    """

    def _make(
        kernel_name: str = "kernel_0",
        grid: tuple = (128,),
        tensor_shapes: dict | None = None,
        smem_bytes: int = 0,
        register_count: int = 32,
        execution_time_ms: float = 1.0,
    ):
        if tensor_shapes is None:
            tensor_shapes = {}

        # Build a mock kernel function for this node
        kernel_fn = MagicMock()
        kernel_fn.__name__ = kernel_name
        kernel_fn.fn = MagicMock()
        kernel_fn.fn.__name__ = kernel_name

        try:
            from triton.graph.kgir import NodeMetadata

            metadata = NodeMetadata(
                grid_dimensions=grid,
                tensor_shapes=tensor_shapes,
                shared_memory_bytes=smem_bytes,
                register_count=register_count,
            )
            # Attach estimated execution time as a runtime annotation
            metadata.runtime_performance_annotations = {
                "estimated_execution_time_ms": execution_time_ms,
            }
        except Exception:
            metadata = MagicMock()
            metadata.grid_dimensions = grid
            metadata.tensor_shapes = tensor_shapes
            metadata.shared_memory_bytes = smem_bytes
            metadata.register_count = register_count
            metadata.runtime_performance_annotations = {
                "estimated_execution_time_ms": execution_time_ms,
            }
            metadata.memory_access_patterns = {}
            metadata.tensor_strides = {}
            metadata.tensor_dtypes = {}
            metadata.num_warps = 4
            metadata.hardware_target_annotations = {}

        # Try constructing a real KGIRNode; fall back to MagicMock
        try:
            from triton.graph.kgir import KGIRNode

            node = KGIRNode(
                node_id=0,  # caller may reassign when adding to graph
                kernel_fn=kernel_fn,
                metadata=metadata,
            )
        except Exception:
            node = MagicMock()
            node.node_id = 0
            node.kernel_fn = kernel_fn
            node.metadata = metadata
            node.is_fused = False
            node.fused_from = None

        return node

    return _make


@pytest.fixture
def sample_kgir_graph(make_kgir_node):
    """A→B→C linear KGIR graph with two data-dependency edges.

    ::

        A (add) ──T1──▶ B (mul) ──T2──▶ C (relu)

    * Node A: ``add`` kernel, produces intermediate **T1**
    * Node B: ``mul`` kernel, consumes T1, produces **T2**
    * Node C: ``relu`` kernel, consumes T2

    This is the standard "three-kernel chain" used across many graph-module
    unit tests (fusion, scheduling, code-gen bridge, …).
    """
    try:
        from triton.graph.kgir import KGIRGraph, KGIREdge

        graph = KGIRGraph()

        node_a = make_kgir_node(kernel_name="add_kernel", grid=(128,), smem_bytes=1024, register_count=32)
        node_b = make_kgir_node(kernel_name="mul_kernel", grid=(128,), smem_bytes=2048, register_count=48)
        node_c = make_kgir_node(kernel_name="relu_kernel", grid=(128,), smem_bytes=512, register_count=24)

        id_a = graph.add_node(node_a.kernel_fn, node_a.metadata)
        id_b = graph.add_node(node_b.kernel_fn, node_b.metadata)
        id_c = graph.add_node(node_c.kernel_fn, node_c.metadata)

        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id="T1")
        graph.add_edge(id_b, id_c, edge_type="data_dep", tensor_id="T2")

        return graph
    except Exception:
        # MagicMock fallback for environments where kgir is not yet built
        graph = MagicMock()

        node_a = make_kgir_node(kernel_name="add_kernel", grid=(128,), smem_bytes=1024, register_count=32)
        node_b = make_kgir_node(kernel_name="mul_kernel", grid=(128,), smem_bytes=2048, register_count=48)
        node_c = make_kgir_node(kernel_name="relu_kernel", grid=(128,), smem_bytes=512, register_count=24)

        node_a.node_id = 0
        node_b.node_id = 1
        node_c.node_id = 2

        graph.nodes = {0: node_a, 1: node_b, 2: node_c}
        graph.edges = [
            MagicMock(source_id=0, target_id=1, edge_type="data_dep", tensor_id="T1", metadata=None),
            MagicMock(source_id=1, target_id=2, edge_type="data_dep", tensor_id="T2", metadata=None),
        ]
        graph.node_count = 3
        graph.get_node = lambda nid: graph.nodes[nid]
        graph.get_edges = MagicMock(return_value=graph.edges)
        graph.get_successors = MagicMock(side_effect=lambda nid: {
            0: [1], 1: [2], 2: []
        }.get(nid, []))

        return graph


@pytest.fixture
def sample_kgir_graph_with_independent_nodes(make_kgir_node):
    """KGIR graph with two independent (unconnected) nodes.

    ::

        A (scale)       B (scale)
        grid=(128,)     grid=(128,)

    Both nodes have compatible grid geometries and are suitable for
    **sibling (horizontal) fusion** testing.
    """
    try:
        from triton.graph.kgir import KGIRGraph

        graph = KGIRGraph()

        node_a = make_kgir_node(kernel_name="scale_a", grid=(128,), smem_bytes=512, register_count=24)
        node_b = make_kgir_node(kernel_name="scale_b", grid=(128,), smem_bytes=512, register_count=24)

        graph.add_node(node_a.kernel_fn, node_a.metadata)
        graph.add_node(node_b.kernel_fn, node_b.metadata)

        return graph
    except Exception:
        graph = MagicMock()

        node_a = make_kgir_node(kernel_name="scale_a", grid=(128,), smem_bytes=512, register_count=24)
        node_b = make_kgir_node(kernel_name="scale_b", grid=(128,), smem_bytes=512, register_count=24)

        node_a.node_id = 0
        node_b.node_id = 1

        graph.nodes = {0: node_a, 1: node_b}
        graph.edges = []
        graph.node_count = 2
        graph.get_node = lambda nid: graph.nodes[nid]
        graph.get_edges = MagicMock(return_value=[])
        graph.get_successors = MagicMock(return_value=[])

        return graph


@pytest.fixture
def sample_kgir_graph_diamond(make_kgir_node):
    """Diamond-shaped KGIR graph for critical-path and scheduling tests.

    ::

              A (source)
             / \\
            B   C
             \\ /
              D (sink)

    Edges: A→B, A→C, B→D, C→D (all ``data_dep``).

    Execution times are deliberately asymmetric (B slower than C) so that
    critical-path analysis can be verified:

    * A: 1.0 ms
    * B: 3.0 ms  (critical path goes through B)
    * C: 1.5 ms
    * D: 2.0 ms
    """
    try:
        from triton.graph.kgir import KGIRGraph

        graph = KGIRGraph()

        node_a = make_kgir_node(kernel_name="source", grid=(256,), execution_time_ms=1.0)
        node_b = make_kgir_node(kernel_name="heavy_path", grid=(256,), execution_time_ms=3.0, smem_bytes=4096)
        node_c = make_kgir_node(kernel_name="light_path", grid=(256,), execution_time_ms=1.5, smem_bytes=1024)
        node_d = make_kgir_node(kernel_name="sink", grid=(256,), execution_time_ms=2.0)

        id_a = graph.add_node(node_a.kernel_fn, node_a.metadata)
        id_b = graph.add_node(node_b.kernel_fn, node_b.metadata)
        id_c = graph.add_node(node_c.kernel_fn, node_c.metadata)
        id_d = graph.add_node(node_d.kernel_fn, node_d.metadata)

        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id="T_AB")
        graph.add_edge(id_a, id_c, edge_type="data_dep", tensor_id="T_AC")
        graph.add_edge(id_b, id_d, edge_type="data_dep", tensor_id="T_BD")
        graph.add_edge(id_c, id_d, edge_type="data_dep", tensor_id="T_CD")

        return graph
    except Exception:
        graph = MagicMock()

        node_a = make_kgir_node(kernel_name="source", grid=(256,), execution_time_ms=1.0)
        node_b = make_kgir_node(kernel_name="heavy_path", grid=(256,), execution_time_ms=3.0, smem_bytes=4096)
        node_c = make_kgir_node(kernel_name="light_path", grid=(256,), execution_time_ms=1.5, smem_bytes=1024)
        node_d = make_kgir_node(kernel_name="sink", grid=(256,), execution_time_ms=2.0)

        node_a.node_id = 0
        node_b.node_id = 1
        node_c.node_id = 2
        node_d.node_id = 3

        graph.nodes = {0: node_a, 1: node_b, 2: node_c, 3: node_d}
        graph.edges = [
            MagicMock(source_id=0, target_id=1, edge_type="data_dep", tensor_id="T_AB", metadata=None),
            MagicMock(source_id=0, target_id=2, edge_type="data_dep", tensor_id="T_AC", metadata=None),
            MagicMock(source_id=1, target_id=3, edge_type="data_dep", tensor_id="T_BD", metadata=None),
            MagicMock(source_id=2, target_id=3, edge_type="data_dep", tensor_id="T_CD", metadata=None),
        ]
        graph.node_count = 4
        graph.get_node = lambda nid: graph.nodes[nid]
        graph.get_edges = MagicMock(return_value=graph.edges)
        graph.get_successors = MagicMock(side_effect=lambda nid: {
            0: [1, 2], 1: [3], 2: [3], 3: []
        }.get(nid, []))

        return graph


# ---------------------------------------------------------------------------
# Phase 5 — Sample Tensor Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tensors(device):
    """Standard GPU tensors for graph capture and numerical-correctness tests.

    Returns a tuple ``(x, y, output)`` of 1-D float32 tensors of length 1024,
    allocated on the test ``device`` (inherited from root conftest).

    If ``torch`` is not installed or the requested device is unavailable, the
    fixture returns ``None`` so that dependent tests can ``pytest.skip``
    gracefully.

    Parameters
    ----------
    device : str
        Injected by the root conftest ``device`` fixture.
    """
    try:
        import torch

        n = 1024
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)
        output = torch.empty(n, device=device, dtype=torch.float32)
        return x, y, output
    except Exception:
        # torch unavailable or device not present — return None sentinel
        return None


# ---------------------------------------------------------------------------
# Phase 6 — Configuration Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_graph_config():
    """``GraphConfig`` with feedback **disabled** — suitable for unit tests.

    Disabling feedback ensures that unit tests exercise single-pass static
    optimisation without triggering the closed-loop profiler.
    """
    try:
        from triton.graph.config import GraphConfig, FeedbackConfig

        return GraphConfig(feedback=FeedbackConfig(enable=False))
    except Exception:
        config = MagicMock()
        config.feedback = MagicMock()
        config.feedback.enable = False
        config.feedback.max_iterations = 20
        config.feedback.sensitivity = 0.15
        config.feedback.convergence_threshold = 0.02
        config.dump_kgir = False
        return config


@pytest.fixture
def feedback_enabled_config():
    """``GraphConfig`` with feedback **enabled** and a capped iteration budget.

    Uses ``max_iterations=5`` (lower than the production default of 20) to
    keep test durations reasonable while still exercising the convergence path.
    """
    try:
        from triton.graph.config import GraphConfig, FeedbackConfig

        return GraphConfig(
            feedback=FeedbackConfig(
                enable=True,
                max_iterations=5,
                sensitivity=0.15,
            ),
        )
    except Exception:
        config = MagicMock()
        config.feedback = MagicMock()
        config.feedback.enable = True
        config.feedback.max_iterations = 5
        config.feedback.sensitivity = 0.15
        config.feedback.convergence_threshold = 0.02
        config.dump_kgir = False
        return config


# ---------------------------------------------------------------------------
# Phase 7 — Mock Hardware Inventory Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hardware_inventory(mock_nvidia_hw_profile):
    """``HardwareInventory`` mock with a **single NVIDIA** device.

    Useful for single-target dispatch and scheduling tests that do not
    require cross-vendor logic.
    """
    inventory = MagicMock()
    inventory.devices = [mock_nvidia_hw_profile]
    inventory.device_count = 1
    inventory.get_device = MagicMock(return_value=mock_nvidia_hw_profile)
    inventory.__len__ = lambda self: 1
    inventory.__iter__ = lambda self: iter(self.devices)
    return inventory


@pytest.fixture
def mock_multi_device_inventory(mock_nvidia_hw_profile, mock_amd_hw_profile):
    """``HardwareInventory`` mock with **NVIDIA + AMD** devices.

    Enables cross-vendor dispatch, multi-target compilation, and
    heterogeneous scheduling tests.
    """
    inventory = MagicMock()
    inventory.devices = [mock_nvidia_hw_profile, mock_amd_hw_profile]
    inventory.device_count = 2
    inventory.get_device = MagicMock(side_effect=lambda idx: inventory.devices[idx])
    inventory.__len__ = lambda self: 2
    inventory.__iter__ = lambda self: iter(self.devices)
    return inventory
