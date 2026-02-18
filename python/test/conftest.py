import pytest
import tempfile


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "kernel_graph: mark test as requiring the graph-level coordination layer")
    config.addinivalue_line("markers", "multi_device: mark test as requiring multiple GPU devices (skip if <2 GPUs available)")
    config.addinivalue_line("markers",
                           "heterogeneous_hw: mark test as requiring heterogeneous GPU hardware (different vendors or generations)")


def pytest_collection_modifyitems(config, items):
    """Skip graph-level tests based on hardware availability."""
    # Detect GPU count and heterogeneity for gating
    gpu_count = 0
    gpu_targets = set()
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_targets.add((props.name, props.major, props.minor))
    except ImportError:
        pass

    # Gate multi_device tests
    skip_multi_device = pytest.mark.skip(reason="requires 2+ GPU devices")
    for item in items:
        if "multi_device" in item.keywords and gpu_count < 2:
            item.add_marker(skip_multi_device)

    # Gate heterogeneous_hw tests
    skip_heterogeneous = pytest.mark.skip(reason="requires heterogeneous GPU hardware (different vendors/generations)")
    for item in items:
        if "heterogeneous_hw" in item.keywords and len(gpu_targets) < 2:
            item.add_marker(skip_heterogeneous)


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


@pytest.fixture
def fresh_knobs():
    """
    Resets all knobs except ``build``, ``nvidia``, and ``amd`` (preserves
    library paths needed to compile kernels).
    """
    try:
        from triton._internal_testing import _fresh_knobs_impl
    except (RuntimeError, ImportError):
        pytest.skip("fresh_knobs requires an active GPU driver (_internal_testing unavailable)")
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_including_libraries():
    """
    Resets ALL knobs including ``build``, ``nvidia``, and ``amd``.
    Use for tests that verify initial values of these knobs.
    """
    try:
        from triton._internal_testing import _fresh_knobs_impl
    except (RuntimeError, ImportError):
        pytest.skip("fresh_knobs_including_libraries requires an active GPU driver (_internal_testing unavailable)")
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())
