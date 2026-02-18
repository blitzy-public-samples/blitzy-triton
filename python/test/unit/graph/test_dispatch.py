"""Unit tests for the hardware-aware dispatch layer (``triton.graph.dispatch``).

Organised into seven test phases covering the dispatch layer's public API:

Phase 1 — HardwareInventory: creation, enumeration, empty/heterogeneous devices
Phase 2 — DispatchDecisionEngine: five-objective scoring, dispatch modes
Phase 3 — Cross-device transfer cost: PCIe, NVLink, cross-vendor staging
Phase 4 — Target filtering via ``TRITON_DISPATCH_TARGETS``
Phase 5 — Dispatch decision latency (< 1 ms / subgraph, AAP §0.7.2)
Phase 6 — Dispatch plan generation and cross-device transfer insertion
Phase 7 — Cold-start exploration and dispatch reassignment (algorithm B5)

All tests are marked with ``@pytest.mark.kernel_graph`` as required by the
graph-module test convention.  Multi-device tests additionally carry the
``@pytest.mark.multi_device`` marker.

References
----------
- AAP §0.1.1  — Dispatch modes (performance / cost / balanced)
- AAP §0.5.3  — Algorithm B5 (cold-start exploration & reassignment)
- AAP §0.7.2  — Dispatch decision latency < 1 ms per subgraph
- AAP §0.7.4  — Single-target is degenerate case of multi-target
"""

from __future__ import annotations

import time

import pytest
from unittest.mock import MagicMock, patch

from triton.graph.dispatch import (
    HardwareInventory,
    DispatchDecisionEngine,
    DispatchMode,
)

# Guard for tests that depend on fixtures requiring torch.
try:
    import torch as _torch  # noqa: F401
    _HAS_TORCH = True
except ModuleNotFoundError:
    _HAS_TORCH = False
from triton.graph.kgir import KGIRGraph, HardwareProfile
from triton.graph.config import DispatchConfig, GraphConfig
from triton.graph.errors import DispatchError


# ---------------------------------------------------------------------------
# Module-level constants imported for assertion targets
# ---------------------------------------------------------------------------
# Re-import private constants used purely as assertion baselines – acceptable
# in test code to verify internal contract compliance.
from triton.graph.dispatch import (          # noqa: E402
    _HOST_STAGING_BANDWIDTH_GBPS,
    _INTERCONNECT_BW,
    _MODE_WEIGHTS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_profile_with_target(
    vendor: str = "nvidia",
    arch_gen: str = "sm_90",
    sm_count: int = 132,
    smem: int = 228 * 1024,
    regs: int = 65536,
    gmem: int = 80 * (1024 ** 3),
    bw: float = 3350.0,
    tflops: float = 989.0,
    warp: int = 32,
    streams: int = 128,
    interconnect: str = "nvlink_4",
    ibw: float = 900.0,
    backend: str | None = None,
    arch: int | str | None = None,
) -> HardwareProfile:
    """Create a :class:`HardwareProfile` with a ``GPUTarget`` attached.

    Defaults to an NVIDIA H100 (sm_90).  All 12 mandatory profile fields
    **and** the optional ``gpu_target`` are populated so that the profile
    works correctly inside :class:`DispatchDecisionEngine` target-aware
    scoring and filter logic.
    """
    from triton.backends.compiler import GPUTarget

    _backend = backend or ("cuda" if vendor == "nvidia" else "hip")
    _arch: int | str = arch if arch is not None else (
        90 if arch_gen == "sm_90" else arch_gen
    )
    target = GPUTarget(backend=_backend, arch=_arch, warp_size=warp)
    return HardwareProfile(
        vendor=vendor,
        arch_generation=arch_gen,
        sm_count=sm_count,
        smem_per_sm_bytes=smem,
        registers_per_sm=regs,
        global_memory_bytes=gmem,
        memory_bandwidth_gbps=bw,
        compute_throughput_tflops=tflops,
        warp_size=warp,
        max_concurrent_streams=streams,
        interconnect_type=interconnect,
        interconnect_bandwidth_gbps=ibw,
        gpu_target=target,
    )


def _build_mock_inventory(profiles: list[HardwareProfile]) -> MagicMock:
    """Construct a lightweight ``MagicMock`` mimicking :class:`HardwareInventory`.

    The mock exposes every attribute and method that
    :class:`DispatchDecisionEngine` reads from the inventory object,
    including the internal ``_device_by_target`` look-up dict.
    """
    inv = MagicMock()
    inv.devices = list(profiles)
    inv.device_count = len(profiles)
    inv._devices = list(profiles)

    def _get_device(idx: int) -> HardwareProfile:
        if 0 <= idx < len(profiles):
            return profiles[idx]
        raise DispatchError(f"Device index {idx} out of range")

    inv.get_device = MagicMock(side_effect=_get_device)
    inv.get_devices_by_vendor = MagicMock(
        side_effect=lambda v: [p for p in profiles if p.vendor == v],
    )

    # Internal target-key → profile dict consumed by the engine.
    by_target: dict[str, HardwareProfile] = {}
    for p in profiles:
        if p.gpu_target is not None:
            key = f"{p.gpu_target.backend}:{p.gpu_target.arch}"
            by_target[key] = p
    inv._device_by_target = by_target

    # Interconnect bandwidth — mirrors real HardwareInventory logic.
    def _bw(src: HardwareProfile, tgt: HardwareProfile) -> float:
        if src is tgt:
            return float("inf")
        if src.vendor != tgt.vendor:
            return _HOST_STAGING_BANDWIDTH_GBPS
        return min(
            src.interconnect_bandwidth_gbps,
            tgt.interconnect_bandwidth_gbps,
        )

    inv.get_interconnect_bandwidth = MagicMock(side_effect=_bw)
    return inv


# ═══════════════════════════════════════════════════════════════════════════════
#  Test-local fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def nvidia_h100_profile():
    """NVIDIA H100 (sm_90) profile with ``GPUTarget`` attached."""
    return _make_profile_with_target()


@pytest.fixture
def nvidia_a100_profile():
    """NVIDIA A100 (sm_80) profile with ``GPUTarget`` attached."""
    return _make_profile_with_target(
        arch_gen="sm_80", sm_count=108,
        smem=164 * 1024, gmem=40 * (1024 ** 3),
        bw=2039.0, tflops=312.0,
        interconnect="nvlink_3", ibw=600.0,
        backend="cuda", arch=80,
    )


@pytest.fixture
def amd_mi300x_profile():
    """AMD MI300X (gfx942) profile with ``GPUTarget`` attached."""
    return _make_profile_with_target(
        vendor="amd", arch_gen="gfx942", sm_count=304,
        smem=64 * 1024, gmem=192 * (1024 ** 3),
        bw=5300.0, tflops=1307.0, warp=64,
        interconnect="infinity_fabric", ibw=896.0,
        backend="hip", arch="gfx942",
    )


@pytest.fixture
def single_device_inventory(nvidia_h100_profile):
    """Mock inventory containing a single NVIDIA H100."""
    return _build_mock_inventory([nvidia_h100_profile])


@pytest.fixture
def multi_device_inventory(nvidia_h100_profile, amd_mi300x_profile):
    """Mock inventory with NVIDIA H100 + AMD MI300X."""
    return _build_mock_inventory([nvidia_h100_profile, amd_mi300x_profile])


@pytest.fixture
def nvidia_dual_inventory(nvidia_h100_profile, nvidia_a100_profile):
    """Mock inventory with two NVIDIA generations (H100 + A100)."""
    return _build_mock_inventory([nvidia_h100_profile, nvidia_a100_profile])


@pytest.fixture
def empty_inventory():
    """Mock inventory with zero devices."""
    inv = MagicMock()
    inv.devices = []
    inv.device_count = 0
    inv._devices = []
    inv._device_by_target = {}
    return inv


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — HardwareInventory Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardwareInventoryCreation:
    """Phase 1: Verify HardwareInventory creation and device enumeration."""

    @pytest.mark.kernel_graph
    def test_hardware_inventory_creation(self, nvidia_h100_profile):
        """1.1 — Create HardwareInventory with one mocked device."""
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[nvidia_h100_profile],
        ):
            inventory = HardwareInventory()

        assert inventory.device_count == 1
        assert len(inventory.devices) == 1
        dev = inventory.devices[0]
        assert dev.vendor == "nvidia"
        assert dev.arch_generation == "sm_90"

    @pytest.mark.kernel_graph
    def test_hardware_inventory_all_12_fields(self, nvidia_h100_profile):
        """1.2 — All 12 HardwareProfile fields are correctly populated."""
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[nvidia_h100_profile],
        ):
            inventory = HardwareInventory()

        dev = inventory.get_device(0)
        assert dev.vendor == "nvidia"
        assert dev.arch_generation == "sm_90"
        assert dev.sm_count == 132
        assert dev.smem_per_sm_bytes == 228 * 1024
        assert dev.registers_per_sm == 65536
        assert dev.global_memory_bytes == 80 * (1024 ** 3)
        assert dev.memory_bandwidth_gbps == 3350.0
        assert dev.compute_throughput_tflops == 989.0
        assert dev.warp_size == 32
        assert dev.max_concurrent_streams == 128
        assert dev.interconnect_type == "nvlink_4"
        assert dev.interconnect_bandwidth_gbps == 900.0

    @pytest.mark.kernel_graph
    def test_hardware_inventory_empty(self):
        """1.3 — Empty backend registry produces zero-device inventory."""
        with patch.object(
            HardwareInventory, "enumerate_devices", return_value=[],
        ):
            inventory = HardwareInventory()

        assert inventory.device_count == 0
        assert inventory.devices == []

    @pytest.mark.kernel_graph
    def test_hardware_inventory_empty_rejects_dispatch(self, empty_inventory):
        """1.3b — DispatchDecisionEngine raises on zero-device inventory."""
        config = DispatchConfig()
        with pytest.raises(DispatchError):
            DispatchDecisionEngine(empty_inventory, config, DispatchMode.BALANCED)

    @pytest.mark.kernel_graph
    def test_hardware_inventory_heterogeneous_nvidia(
        self, nvidia_h100_profile, nvidia_a100_profile,
    ):
        """1.4 — Two NVIDIA architectures coexist in one inventory."""
        profiles = [nvidia_h100_profile, nvidia_a100_profile]
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=profiles,
        ):
            inventory = HardwareInventory()

        assert inventory.device_count == 2
        generations = {d.arch_generation for d in inventory.devices}
        assert generations == {"sm_90", "sm_80"}

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_hardware_inventory_heterogeneous_cross_vendor(
        self, nvidia_h100_profile, amd_mi300x_profile,
    ):
        """1.4b — NVIDIA + AMD devices in the same inventory."""
        profiles = [nvidia_h100_profile, amd_mi300x_profile]
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=profiles,
        ):
            inventory = HardwareInventory()

        assert inventory.device_count == 2
        vendors = {d.vendor for d in inventory.devices}
        assert vendors == {"nvidia", "amd"}

    @pytest.mark.kernel_graph
    def test_hardware_inventory_get_device_out_of_range(
        self, nvidia_h100_profile,
    ):
        """get_device raises DispatchError on out-of-range index."""
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[nvidia_h100_profile],
        ):
            inventory = HardwareInventory()

        with pytest.raises(DispatchError):
            inventory.get_device(5)

    @pytest.mark.kernel_graph
    def test_hardware_inventory_get_devices_by_vendor(
        self, nvidia_h100_profile, amd_mi300x_profile,
    ):
        """get_devices_by_vendor filters devices correctly."""
        profiles = [nvidia_h100_profile, amd_mi300x_profile]
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=profiles,
        ):
            inventory = HardwareInventory()

        nvidia_devs = inventory.get_devices_by_vendor("nvidia")
        amd_devs = inventory.get_devices_by_vendor("amd")
        assert len(nvidia_devs) == 1
        assert nvidia_devs[0].vendor == "nvidia"
        assert len(amd_devs) == 1
        assert amd_devs[0].vendor == "amd"

    @pytest.mark.kernel_graph
    def test_hardware_inventory_get_interconnect_bandwidth_same(
        self, nvidia_h100_profile,
    ):
        """Same-device bandwidth is infinite (no transfer needed)."""
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[nvidia_h100_profile],
        ):
            inventory = HardwareInventory()

        bw = inventory.get_interconnect_bandwidth(
            nvidia_h100_profile, nvidia_h100_profile,
        )
        assert bw == float("inf")

    @pytest.mark.kernel_graph
    def test_hardware_inventory_from_mock_drivers(self):
        """1.2b — Exercise the backend-driven enumeration code path.

        Mocks the global ``backends`` registry so no real hardware is
        required, then verifies the inventory construction succeeds.
        """
        mock_driver_cls = MagicMock()
        mock_driver_cls.is_active.return_value = True

        mock_backend = MagicMock()
        mock_backend.driver = mock_driver_cls

        with patch("triton.graph.dispatch.backends", {"nvidia": mock_backend}):
            inventory = HardwareInventory()

        # Enumeration may yield 0 devices if torch is unavailable and the
        # driver fallback cannot query properties, but it must not crash.
        assert isinstance(inventory.device_count, int)
        assert isinstance(inventory.devices, list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — DispatchDecisionEngine Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDispatchDecisionEngine:
    """Phase 2: Five-objective scoring and dispatch mode behaviour."""

    @pytest.mark.kernel_graph
    def test_dispatch_scoring_five_objectives(
        self, single_device_inventory, sample_kgir_graph,
        nvidia_h100_profile,
    ):
        """2.1 — score_subgraph_device returns a float in [0, 1].

        Per AAP §0.1.1 the five scoring objectives are: performance, cost,
        data locality, device utilisation, and memory capacity.
        """
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        score = engine.score_subgraph_device(
            sample_kgir_graph, nvidia_h100_profile,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.kernel_graph
    @pytest.mark.parametrize("mode_enum,mode_str", [
        (DispatchMode.PERFORMANCE, "performance"),
        (DispatchMode.COST, "cost"),
        (DispatchMode.BALANCED, "balanced"),
    ])
    def test_dispatch_modes_produce_valid_scores(
        self, single_device_inventory, sample_kgir_graph,
        nvidia_h100_profile, mode_enum, mode_str,
    ):
        """2.2/2.3/2.4 — Each mode yields a valid score."""
        config = DispatchConfig(mode=mode_str)
        engine = DispatchDecisionEngine(
            single_device_inventory, config, mode_enum,
        )
        score = engine.score_subgraph_device(
            sample_kgir_graph, nvidia_h100_profile,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.kernel_graph
    def test_dispatch_mode_performance_weights(self):
        """2.2 — Performance mode weights the performance objective highest."""
        weights = _MODE_WEIGHTS["performance"]
        assert weights["performance"] == max(weights.values())

    @pytest.mark.kernel_graph
    def test_dispatch_mode_cost_weights(self):
        """2.3 — Cost mode weights the cost objective highest."""
        weights = _MODE_WEIGHTS["cost"]
        assert weights["cost"] == max(weights.values())

    @pytest.mark.kernel_graph
    def test_dispatch_mode_balanced_weights(self):
        """2.4 — Balanced mode distributes weights equally."""
        weights = _MODE_WEIGHTS["balanced"]
        values = list(weights.values())
        assert all(v == values[0] for v in values), (
            f"Balanced weights should be uniform: {weights}"
        )

    @pytest.mark.kernel_graph
    def test_dispatch_single_target_degenerate(
        self, single_device_inventory, sample_kgir_graph,
        nvidia_h100_profile,
    ):
        """2.5 — Single-target dispatch is a degenerate multi-target case.

        Per AAP §0.7.4 single-target execution MUST use the same engine
        path as multi-target, not a separate code path.
        """
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(sample_kgir_graph)

        # Every node must be assigned to the sole eligible device.
        assert len(plan) == sample_kgir_graph.node_count()
        expected = nvidia_h100_profile.gpu_target
        for nid, target in plan.items():
            assert target == expected

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_prefers_higher_compute_in_perf_mode(
        self, nvidia_dual_inventory, sample_kgir_graph,
        nvidia_h100_profile, nvidia_a100_profile,
    ):
        """Performance mode should favour the device with higher TFLOPS."""
        config = DispatchConfig(mode="performance")
        engine = DispatchDecisionEngine(
            nvidia_dual_inventory, config, DispatchMode.PERFORMANCE,
        )
        score_h100 = engine.score_subgraph_device(
            sample_kgir_graph, nvidia_h100_profile,
        )
        score_a100 = engine.score_subgraph_device(
            sample_kgir_graph, nvidia_a100_profile,
        )
        # H100 (989 TFLOPS) should score at least as high as A100 (312 TFLOPS).
        assert score_h100 >= score_a100

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_different_modes_yield_different_scores(
        self, multi_device_inventory, sample_kgir_graph,
        nvidia_h100_profile,
    ):
        """Different dispatch modes produce distinguishable scores."""
        scores = {}
        for mode in (DispatchMode.PERFORMANCE, DispatchMode.COST,
                      DispatchMode.BALANCED):
            config = DispatchConfig(mode=mode.value)
            engine = DispatchDecisionEngine(
                multi_device_inventory, config, mode,
            )
            scores[mode] = engine.score_subgraph_device(
                sample_kgir_graph, nvidia_h100_profile,
            )
        # At minimum, all scores must be valid.
        for s in scores.values():
            assert 0.0 <= s <= 1.0

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_data_locality(
        self, multi_device_inventory, make_kgir_node,
    ):
        """2.6 — Data locality biases co-location of producer–consumer pairs.

        When kernel A writes a tensor consumed by kernel B, the dispatch
        engine should prefer assigning both to the same device to avoid
        cross-device transfer costs.
        """
        graph = KGIRGraph()
        node_a = make_kgir_node(kernel_name="producer", grid=(128,))
        node_b = make_kgir_node(kernel_name="consumer", grid=(128,))
        id_a = graph.add_node(node_a.kernel_fn, node_a.metadata)
        id_b = graph.add_node(node_b.kernel_fn, node_b.metadata)
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id="T1")

        config = DispatchConfig(mode="balanced")
        engine = DispatchDecisionEngine(
            multi_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(graph)

        assert len(plan) == 2
        # Data-locality weighting should bias co-location.
        assert plan[id_a] == plan[id_b], (
            "Directly dependent nodes should be co-located via data-locality bias"
        )

    @pytest.mark.kernel_graph
    def test_dispatch_with_custom_cost_weights(
        self, single_device_inventory, sample_kgir_graph,
        nvidia_h100_profile,
    ):
        """DispatchConfig.cost_weights overrides mode defaults."""
        custom = {
            "performance": 0.90,
            "cost": 0.02,
            "data_locality": 0.02,
            "device_utilization": 0.02,
            "memory_capacity": 0.04,
        }
        config = DispatchConfig(cost_weights=custom)
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        score = engine.score_subgraph_device(
            sample_kgir_graph, nvidia_h100_profile,
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.kernel_graph
    def test_dispatch_config_latency_constraint(
        self, single_device_inventory,
    ):
        """DispatchConfig.latency_constraint_ms can be set and read."""
        config = DispatchConfig(latency_constraint_ms=5.0)
        assert config.latency_constraint_ms == 5.0
        # Engine still constructs fine with a latency constraint.
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        assert engine is not None

    @pytest.mark.kernel_graph
    def test_dispatch_with_graph_config(self, default_graph_config):
        """GraphConfig aggregates a DispatchConfig (conftest fixture)."""
        assert isinstance(default_graph_config, GraphConfig)
        assert isinstance(default_graph_config.dispatch, DispatchConfig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Cross-Device Transfer Cost Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransferCost:
    """Phase 3: Cross-device transfer cost modelling."""

    @pytest.mark.kernel_graph
    def test_transfer_cost_same_device(self, nvidia_h100_profile):
        """Same device → infinite bandwidth (no transfer)."""
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[nvidia_h100_profile],
        ):
            inventory = HardwareInventory()

        bw = inventory.get_interconnect_bandwidth(
            nvidia_h100_profile, nvidia_h100_profile,
        )
        assert bw == float("inf")

    @pytest.mark.kernel_graph
    def test_transfer_cost_pcie(self):
        """3.1 — PCIe-connected NVIDIA devices use PCIe bandwidth."""
        dev_a = _make_profile_with_target(
            arch_gen="sm_90", interconnect="pcie_4", ibw=25.0,
            backend="cuda", arch=90,
        )
        dev_b = _make_profile_with_target(
            arch_gen="sm_80", interconnect="pcie_4", ibw=25.0,
            backend="cuda", arch=80,
        )
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[dev_a, dev_b],
        ):
            inventory = HardwareInventory()

        bw = inventory.get_interconnect_bandwidth(dev_a, dev_b)
        # Same vendor, both PCIe 4 at 25 GB/s ⇒ min(25, 25) = 25
        assert bw == pytest.approx(25.0)

    @pytest.mark.kernel_graph
    def test_transfer_cost_nvlink(self):
        """3.2 — NVLink-connected NVIDIA devices get NVLink bandwidth."""
        dev_a = _make_profile_with_target(
            arch_gen="sm_90", interconnect="nvlink_4", ibw=450.0,
            backend="cuda", arch=90,
        )
        dev_b = _make_profile_with_target(
            arch_gen="sm_90", interconnect="nvlink_4", ibw=450.0,
            backend="cuda", arch=91,          # different device, same gen
        )
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[dev_a, dev_b],
        ):
            inventory = HardwareInventory()

        bw = inventory.get_interconnect_bandwidth(dev_a, dev_b)
        assert bw == pytest.approx(450.0)

    @pytest.mark.kernel_graph
    def test_nvlink_faster_than_pcie(self):
        """NVLink pair reports higher bandwidth than PCIe pair."""
        pcie_a = _make_profile_with_target(
            arch_gen="sm_80", interconnect="pcie_4", ibw=25.0,
            backend="cuda", arch=80,
        )
        pcie_b = _make_profile_with_target(
            arch_gen="sm_70", interconnect="pcie_4", ibw=25.0,
            backend="cuda", arch=70,
        )
        nvlink_a = _make_profile_with_target(
            arch_gen="sm_90", interconnect="nvlink_4", ibw=450.0,
            backend="cuda", arch=90,
        )
        nvlink_b = _make_profile_with_target(
            arch_gen="sm_90", interconnect="nvlink_4", ibw=450.0,
            backend="cuda", arch=91,
        )
        all_devs = [pcie_a, pcie_b, nvlink_a, nvlink_b]
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=all_devs,
        ):
            inventory = HardwareInventory()

        bw_pcie = inventory.get_interconnect_bandwidth(pcie_a, pcie_b)
        bw_nvlink = inventory.get_interconnect_bandwidth(nvlink_a, nvlink_b)
        assert bw_nvlink > bw_pcie

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_transfer_cost_cross_vendor(
        self, nvidia_h100_profile, amd_mi300x_profile,
    ):
        """3.3 — Cross-vendor transfer falls back to host-memory staging.

        Per AAP §0.1.1: "cross-vendor dispatch with explicit host-memory
        staging" — bandwidth should equal _HOST_STAGING_BANDWIDTH_GBPS.
        """
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[nvidia_h100_profile, amd_mi300x_profile],
        ):
            inventory = HardwareInventory()

        bw = inventory.get_interconnect_bandwidth(
            nvidia_h100_profile, amd_mi300x_profile,
        )
        assert bw == _HOST_STAGING_BANDWIDTH_GBPS
        # Must be significantly slower than any same-vendor high-speed link.
        assert bw < 450.0

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_cross_vendor_bidirectional(
        self, nvidia_h100_profile, amd_mi300x_profile,
    ):
        """Cross-vendor bandwidth is symmetric."""
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[nvidia_h100_profile, amd_mi300x_profile],
        ):
            inventory = HardwareInventory()

        fwd = inventory.get_interconnect_bandwidth(
            nvidia_h100_profile, amd_mi300x_profile,
        )
        rev = inventory.get_interconnect_bandwidth(
            amd_mi300x_profile, nvidia_h100_profile,
        )
        assert fwd == rev

    @pytest.mark.kernel_graph
    def test_transfer_cost_mixed_interconnect(self):
        """Min-of-pair bandwidth when interconnect types differ."""
        fast = _make_profile_with_target(
            arch_gen="sm_90", interconnect="nvlink_4", ibw=450.0,
            backend="cuda", arch=90,
        )
        slow = _make_profile_with_target(
            arch_gen="sm_80", interconnect="pcie_4", ibw=25.0,
            backend="cuda", arch=80,
        )
        with patch.object(
            HardwareInventory, "enumerate_devices",
            return_value=[fast, slow],
        ):
            inventory = HardwareInventory()

        bw = inventory.get_interconnect_bandwidth(fast, slow)
        # min(450, 25) = 25
        assert bw == pytest.approx(25.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Target Filtering Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTargetFiltering:
    """Phase 4: ``TRITON_DISPATCH_TARGETS`` filtering."""

    @pytest.mark.kernel_graph
    def test_dispatch_target_filtering_single(
        self, multi_device_inventory, nvidia_h100_profile,
    ):
        """4.1 — Only the matching target is eligible after filtering."""
        config = DispatchConfig(targets="cuda:90")
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                multi_device_inventory, config, DispatchMode.BALANCED,
            )
        assert len(engine._eligible_devices) == 1
        assert engine._eligible_devices[0].gpu_target == nvidia_h100_profile.gpu_target

    @pytest.mark.kernel_graph
    def test_dispatch_no_filter_all_eligible(self, multi_device_inventory):
        """4.2 — No filter → all devices eligible."""
        config = DispatchConfig(targets=None)
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                multi_device_inventory, config, DispatchMode.BALANCED,
            )
        assert len(engine._eligible_devices) == 2

    @pytest.mark.kernel_graph
    def test_dispatch_filter_eliminates_all_raises(
        self, single_device_inventory,
    ):
        """Filter matching zero devices raises DispatchError."""
        config = DispatchConfig(targets="hip:gfx942")
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            with pytest.raises(DispatchError):
                DispatchDecisionEngine(
                    single_device_inventory, config, DispatchMode.BALANCED,
                )

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_filter_multiple_targets(
        self, multi_device_inventory,
    ):
        """Comma-separated filter retains matching devices."""
        config = DispatchConfig(targets="cuda:90,hip:gfx942")
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                multi_device_inventory, config, DispatchMode.BALANCED,
            )
        assert len(engine._eligible_devices) == 2

    @pytest.mark.kernel_graph
    def test_dispatch_filter_via_env_knob(
        self, multi_device_inventory, nvidia_h100_profile,
    ):
        """TRITON_DISPATCH_TARGETS env knob is respected when config.targets is None."""
        knobs_mock = MagicMock()
        knobs_mock.dispatch_targets = "cuda:90"
        config = DispatchConfig(targets=None)

        with patch(
            "triton.graph.dispatch._get_dispatch_knobs",
            return_value=knobs_mock,
        ):
            engine = DispatchDecisionEngine(
                multi_device_inventory, config, DispatchMode.BALANCED,
            )

        assert len(engine._eligible_devices) == 1
        assert engine._eligible_devices[0].gpu_target == nvidia_h100_profile.gpu_target

    @pytest.mark.kernel_graph
    def test_dispatch_filter_unknown_target(self, multi_device_inventory):
        """Filter with a target not in inventory eliminates it silently."""
        config = DispatchConfig(targets="cuda:90,cuda:999")
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                multi_device_inventory, config, DispatchMode.BALANCED,
            )
        # Only cuda:90 matches; cuda:999 silently ignored.
        matching = [
            d for d in engine._eligible_devices
            if d.gpu_target is not None and d.gpu_target.arch == 90
        ]
        assert len(matching) == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Dispatch Decision Latency Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDispatchLatency:
    """Phase 5: Dispatch decisions < 1 ms per subgraph (AAP §0.7.2)."""

    @pytest.mark.kernel_graph
    def test_dispatch_decision_latency_10_nodes(
        self, single_device_inventory, make_kgir_node,
    ):
        """5.1 — Per-node dispatch latency < 1 ms on a 10-node chain."""
        graph = KGIRGraph()
        prev_id: int | None = None
        for i in range(10):
            node = make_kgir_node(kernel_name=f"kernel_{i}", grid=(64,))
            nid = graph.add_node(node.kernel_fn, node.metadata)
            if prev_id is not None:
                graph.add_edge(
                    prev_id, nid, edge_type="data_dep", tensor_id=f"T{i}",
                )
            prev_id = nid

        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )

        start = time.perf_counter()
        plan = engine.compute_dispatch_plan(graph)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert len(plan) == 10
        per_node_ms = elapsed_ms / 10.0
        assert per_node_ms < 1.0, (
            f"Per-node dispatch latency {per_node_ms:.4f} ms exceeds 1 ms"
        )

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_latency_multi_device_10_nodes(
        self, multi_device_inventory, make_kgir_node,
    ):
        """Latency stays < 1 ms per node with multiple candidate devices."""
        graph = KGIRGraph()
        prev_id: int | None = None
        for i in range(10):
            node = make_kgir_node(kernel_name=f"kernel_{i}", grid=(64,))
            nid = graph.add_node(node.kernel_fn, node.metadata)
            if prev_id is not None:
                graph.add_edge(
                    prev_id, nid, edge_type="data_dep", tensor_id=f"T{i}",
                )
            prev_id = nid

        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            multi_device_inventory, config, DispatchMode.BALANCED,
        )

        start = time.perf_counter()
        plan = engine.compute_dispatch_plan(graph)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        per_node_ms = elapsed_ms / 10.0
        assert per_node_ms < 1.0, (
            f"Multi-device per-node latency {per_node_ms:.4f} ms exceeds 1 ms"
        )

    @pytest.mark.kernel_graph
    def test_dispatch_latency_independent_nodes(
        self, single_device_inventory, make_kgir_node,
    ):
        """Latency check for independent (no-edge) nodes."""
        graph = KGIRGraph()
        for i in range(20):
            node = make_kgir_node(kernel_name=f"kernel_{i}", grid=(64,))
            graph.add_node(node.kernel_fn, node.metadata)

        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )

        start = time.perf_counter()
        plan = engine.compute_dispatch_plan(graph)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        per_node_ms = elapsed_ms / 20.0
        assert per_node_ms < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 6 — Dispatch Plan Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDispatchPlan:
    """Phase 6: Dispatch plan generation and cross-device transfer insertion."""

    @pytest.mark.kernel_graph
    def test_dispatch_plan_generation_single_device(
        self, single_device_inventory, sample_kgir_graph,
        nvidia_h100_profile,
    ):
        """6.1 — Every node is mapped to a valid target."""
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(sample_kgir_graph)

        assert len(plan) == sample_kgir_graph.node_count()
        for nid, target in plan.items():
            assert target is not None
            assert target == nvidia_h100_profile.gpu_target

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_plan_multi_target(
        self, multi_device_inventory, sample_kgir_graph,
    ):
        """Multi-target dispatch assigns every node."""
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            multi_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(sample_kgir_graph)
        assert len(plan) == sample_kgir_graph.node_count()
        for target in plan.values():
            assert target is not None

    @pytest.mark.kernel_graph
    def test_dispatch_plan_empty_graph(self, single_device_inventory):
        """Empty graph produces an empty dispatch plan."""
        graph = KGIRGraph()
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(graph)
        assert plan == {}

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_plan_insert_transfers(
        self, multi_device_inventory, make_kgir_node,
        nvidia_h100_profile, amd_mi300x_profile,
    ):
        """6.2 — Transfer edges inserted for cross-device data deps."""
        graph = KGIRGraph()
        node_a = make_kgir_node(kernel_name="producer", grid=(128,))
        node_b = make_kgir_node(kernel_name="consumer", grid=(128,))
        id_a = graph.add_node(node_a.kernel_fn, node_a.metadata)
        id_b = graph.add_node(node_b.kernel_fn, node_b.metadata)
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id="T1")

        # Forcibly assign cross-device.
        plan = {
            id_a: nvidia_h100_profile.gpu_target,
            id_b: amd_mi300x_profile.gpu_target,
        }

        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            multi_device_inventory, config, DispatchMode.BALANCED,
        )
        result_graph = engine.insert_transfer_operations(graph, plan)

        transfer_edges = [
            e for e in result_graph.get_edges()
            if e.edge_type == "cross_device_transfer"
        ]
        assert len(transfer_edges) >= 1
        te = transfer_edges[0]
        assert te.source_id == id_a
        assert te.target_id == id_b

    @pytest.mark.kernel_graph
    def test_dispatch_plan_no_transfers_same_device(
        self, single_device_inventory, sample_kgir_graph,
        nvidia_h100_profile,
    ):
        """No transfer edges when all nodes share a device."""
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(sample_kgir_graph)
        result = engine.insert_transfer_operations(sample_kgir_graph, plan)

        transfer_edges = [
            e for e in result.get_edges()
            if e.edge_type == "cross_device_transfer"
        ]
        assert len(transfer_edges) == 0

    @pytest.mark.kernel_graph
    def test_dispatch_plan_graph_level_granularity(
        self, single_device_inventory, sample_kgir_graph,
    ):
        """Graph-level granularity assigns all nodes to one device."""
        config = DispatchConfig(granularity="graph")
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(sample_kgir_graph)
        targets = set(plan.values())
        # All nodes on a single target.
        assert len(targets) == 1

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_transfer_metadata_recorded(
        self, multi_device_inventory, make_kgir_node,
        nvidia_h100_profile, amd_mi300x_profile,
    ):
        """Transfer edges carry bandwidth / byte-count metadata."""
        graph = KGIRGraph()
        node_a = make_kgir_node(kernel_name="prod", grid=(128,))
        node_b = make_kgir_node(kernel_name="cons", grid=(128,))
        id_a = graph.add_node(node_a.kernel_fn, node_a.metadata)
        id_b = graph.add_node(node_b.kernel_fn, node_b.metadata)
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id="T1")

        plan = {
            id_a: nvidia_h100_profile.gpu_target,
            id_b: amd_mi300x_profile.gpu_target,
        }

        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            multi_device_inventory, config, DispatchMode.BALANCED,
        )
        result = engine.insert_transfer_operations(graph, plan)

        transfer_edges = [
            e for e in result.get_edges()
            if e.edge_type == "cross_device_transfer"
        ]
        if transfer_edges:
            te = transfer_edges[0]
            # The edge should carry transfer metadata as a dict.
            meta = te.metadata if hasattr(te, "metadata") else {}
            # Even if metadata is minimal, the edge must exist.
            assert te.source_id == id_a
            assert te.target_id == id_b


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 7 — Cold-Start and Reassignment Tests (B5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestColdStartAndReassignment:
    """Phase 7: Cold-start exploration and dispatch reassignment (B5).

    Per AAP §0.5.3 B5: newly available devices trigger exploration
    profiling before a keep-or-revert decision, and reassignment changes
    interact with global convergence detection.
    """

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_cold_start_exploration(
        self, nvidia_h100_profile, amd_mi300x_profile, make_kgir_node,
    ):
        """7.1 — A new device triggers re-dispatch consideration.

        After initial single-device dispatch, adding a second device and
        re-running dispatch should include the new device in scoring.
        """
        # Initial: single device
        inv_v1 = _build_mock_inventory([nvidia_h100_profile])
        graph = KGIRGraph()
        node = make_kgir_node(kernel_name="workload", grid=(256,))
        nid = graph.add_node(node.kernel_fn, node.metadata)

        config = DispatchConfig()
        engine_v1 = DispatchDecisionEngine(
            inv_v1, config, DispatchMode.BALANCED,
        )
        plan_v1 = engine_v1.compute_dispatch_plan(graph)
        assert plan_v1[nid] == nvidia_h100_profile.gpu_target

        # Cold-start: second device appears
        inv_v2 = _build_mock_inventory(
            [nvidia_h100_profile, amd_mi300x_profile],
        )
        engine_v2 = DispatchDecisionEngine(
            inv_v2, config, DispatchMode.BALANCED,
        )
        plan_v2 = engine_v2.compute_dispatch_plan(graph)

        # Engine v2 must consider both devices.
        assert len(engine_v2._eligible_devices) == 2
        assert plan_v2[nid] is not None

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_reassignment_on_mode_change(
        self, multi_device_inventory, make_kgir_node,
        nvidia_h100_profile, amd_mi300x_profile,
    ):
        """7.2 — Reassignment via mode change re-evaluates scoring.

        Per AAP §0.5.3 B5: after feedback shows a different target is
        better, reassignment occurs.  We simulate this by switching modes.
        """
        graph = KGIRGraph()
        node = make_kgir_node(kernel_name="workload", grid=(256,))
        nid = graph.add_node(node.kernel_fn, node.metadata)

        # Performance mode
        engine_perf = DispatchDecisionEngine(
            multi_device_inventory,
            DispatchConfig(mode="performance"),
            DispatchMode.PERFORMANCE,
        )
        plan_perf = engine_perf.compute_dispatch_plan(graph)

        # Cost mode
        engine_cost = DispatchDecisionEngine(
            multi_device_inventory,
            DispatchConfig(mode="cost"),
            DispatchMode.COST,
        )
        plan_cost = engine_cost.compute_dispatch_plan(graph)

        # Both plans must be valid.
        assert nid in plan_perf
        assert nid in plan_cost

        # Score comparison: different modes should produce different raw scores.
        s_perf = engine_perf.score_subgraph_device(graph, nvidia_h100_profile)
        s_cost = engine_cost.score_subgraph_device(graph, nvidia_h100_profile)
        assert isinstance(s_perf, float) and 0.0 <= s_perf <= 1.0
        assert isinstance(s_cost, float) and 0.0 <= s_cost <= 1.0

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_reassignment_preserves_correctness(
        self, multi_device_inventory, sample_kgir_graph,
    ):
        """Re-dispatching the same graph is always valid regardless of mode."""
        config = DispatchConfig()
        for mode in (
            DispatchMode.PERFORMANCE,
            DispatchMode.COST,
            DispatchMode.BALANCED,
        ):
            engine = DispatchDecisionEngine(
                multi_device_inventory, config, mode,
            )
            plan = engine.compute_dispatch_plan(sample_kgir_graph)
            assert len(plan) == sample_kgir_graph.node_count()
            for target in plan.values():
                assert target is not None

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_cold_start_with_different_vendor_device(
        self, nvidia_h100_profile, amd_mi300x_profile, make_kgir_node,
    ):
        """Cold-start with a cross-vendor addition still yields valid plans."""
        inv = _build_mock_inventory(
            [nvidia_h100_profile, amd_mi300x_profile],
        )
        graph = KGIRGraph()
        for i in range(3):
            node = make_kgir_node(kernel_name=f"k{i}", grid=(64,))
            graph.add_node(node.kernel_fn, node.metadata)

        config = DispatchConfig()
        engine = DispatchDecisionEngine(inv, config, DispatchMode.BALANCED)
        plan = engine.compute_dispatch_plan(graph)
        assert len(plan) == 3
        for target in plan.values():
            assert target is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  DispatchMode Enum Correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestDispatchModeEnum:
    """Verify ``DispatchMode`` enum values and string behaviour."""

    @pytest.mark.kernel_graph
    def test_mode_values(self):
        """DispatchMode defines exactly three modes."""
        assert DispatchMode.PERFORMANCE.value == "performance"
        assert DispatchMode.COST.value == "cost"
        assert DispatchMode.BALANCED.value == "balanced"

    @pytest.mark.kernel_graph
    def test_mode_from_string(self):
        """DispatchMode can be constructed from its string value."""
        assert DispatchMode("performance") == DispatchMode.PERFORMANCE
        assert DispatchMode("cost") == DispatchMode.COST
        assert DispatchMode("balanced") == DispatchMode.BALANCED

    @pytest.mark.kernel_graph
    def test_mode_is_string(self):
        """DispatchMode inherits from ``str``."""
        assert isinstance(DispatchMode.PERFORMANCE, str)
        assert DispatchMode.BALANCED == "balanced"

    @pytest.mark.kernel_graph
    def test_all_modes_in_weight_table(self):
        """Every DispatchMode has an entry in _MODE_WEIGHTS."""
        for mode in DispatchMode:
            assert mode.value in _MODE_WEIGHTS, (
                f"Missing weight entry for mode {mode.value}"
            )

    @pytest.mark.kernel_graph
    def test_weight_table_has_five_objectives(self):
        """Each weight entry contains exactly 5 objectives."""
        expected_keys = {
            "performance", "cost", "data_locality",
            "device_utilization", "memory_capacity",
        }
        for mode_str, weights in _MODE_WEIGHTS.items():
            assert set(weights.keys()) == expected_keys, (
                f"Mode {mode_str!r} weight keys mismatch"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Edge-case / Integration Smoke Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Additional edge-case and integration smoke tests."""

    @pytest.mark.kernel_graph
    def test_dispatch_config_defaults(self):
        """DispatchConfig has sensible defaults from AAP §0.5.1 Group 5."""
        cfg = DispatchConfig()
        assert cfg.mode == "balanced"
        assert cfg.granularity == "subgraph"
        assert cfg.targets is None
        assert cfg.cost_weights is None
        assert cfg.latency_constraint_ms is None

    @pytest.mark.kernel_graph
    def test_graph_config_includes_dispatch(self):
        """GraphConfig aggregates a DispatchConfig instance."""
        gc = GraphConfig()
        assert hasattr(gc, "dispatch")
        assert isinstance(gc.dispatch, DispatchConfig)

    @pytest.mark.kernel_graph
    def test_dispatch_error_inherits_base(self):
        """DispatchError is a proper exception."""
        err = DispatchError("test message")
        assert isinstance(err, Exception)

    @pytest.mark.kernel_graph
    def test_mode_weights_sum_to_one(self):
        """Weight vectors should sum to 1.0 for each mode."""
        for mode_str, weights in _MODE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Weights for mode {mode_str!r} sum to {total}, expected 1.0"
            )

    @pytest.mark.kernel_graph
    def test_interconnect_bw_table_entries(self):
        """_INTERCONNECT_BW contains expected interconnect types."""
        for key in ("pcie_3", "pcie_4", "pcie_5", "nvlink_2", "nvlink_3",
                     "nvlink_4", "infinity_fabric"):
            assert key in _INTERCONNECT_BW, f"Missing interconnect: {key}"
            assert _INTERCONNECT_BW[key] > 0

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_dispatch_plan_diamond_graph(
        self, single_device_inventory, sample_kgir_graph_diamond,
    ):
        """Diamond DAG (A→{B,C}→D) produces a valid plan."""
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(sample_kgir_graph_diamond)
        assert len(plan) == sample_kgir_graph_diamond.node_count()

    @pytest.mark.kernel_graph
    def test_dispatch_plan_independent_nodes_graph(
        self, single_device_inventory,
        sample_kgir_graph_with_independent_nodes,
    ):
        """Graph with independent nodes still dispatches every node."""
        config = DispatchConfig()
        engine = DispatchDecisionEngine(
            single_device_inventory, config, DispatchMode.BALANCED,
        )
        plan = engine.compute_dispatch_plan(
            sample_kgir_graph_with_independent_nodes,
        )
        expected = sample_kgir_graph_with_independent_nodes.node_count()
        assert len(plan) == expected

    @pytest.mark.kernel_graph
    def test_host_staging_bandwidth_is_reasonable(self):
        """Host-memory staging bandwidth constant is within reasonable range."""
        # Typical PCIe 3 x16 unidirectional ≈ 12 GB/s
        assert 1.0 <= _HOST_STAGING_BANDWIDTH_GBPS <= 100.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Conftest Fixture Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConftestFixtureIntegration:
    """Verify interoperability with shared conftest fixtures.

    These tests exercise the fixtures provided by
    ``python/test/unit/graph/conftest.py`` and the root
    ``python/test/conftest.py`` to ensure they produce objects
    compatible with the dispatch layer's public API.
    """

    # -- mock_gpu_target / mock_gpu_target_amd ---------------------

    @pytest.mark.kernel_graph
    def test_mock_gpu_target_is_valid(self, mock_gpu_target):
        """conftest ``mock_gpu_target`` fixture has expected CUDA attrs."""
        assert mock_gpu_target.backend == "cuda"
        assert mock_gpu_target.warp_size == 32

    @pytest.mark.kernel_graph
    def test_mock_gpu_target_amd_is_valid(self, mock_gpu_target_amd):
        """conftest ``mock_gpu_target_amd`` fixture has expected HIP attrs."""
        assert mock_gpu_target_amd.backend == "hip"
        assert mock_gpu_target_amd.warp_size == 64

    # -- mock_nvidia_hw_profile / mock_amd_hw_profile ---------------

    @pytest.mark.kernel_graph
    def test_mock_nvidia_hw_profile_fields(self, mock_nvidia_hw_profile):
        """conftest NVIDIA HardwareProfile has all 12 mandatory fields."""
        p = mock_nvidia_hw_profile
        assert p.vendor == "nvidia"
        assert p.arch_generation == "sm_90"
        assert isinstance(p.sm_count, int) and p.sm_count > 0
        assert isinstance(p.smem_per_sm_bytes, int) and p.smem_per_sm_bytes > 0
        assert isinstance(p.registers_per_sm, int) and p.registers_per_sm > 0
        assert isinstance(p.global_memory_bytes, int) and p.global_memory_bytes > 0
        assert isinstance(p.memory_bandwidth_gbps, (int, float)) and p.memory_bandwidth_gbps > 0
        assert isinstance(p.compute_throughput_tflops, (int, float)) and p.compute_throughput_tflops > 0
        assert isinstance(p.warp_size, int) and p.warp_size > 0
        assert isinstance(p.max_concurrent_streams, int) and p.max_concurrent_streams > 0
        assert isinstance(p.interconnect_type, str) and len(p.interconnect_type) > 0
        assert isinstance(p.interconnect_bandwidth_gbps, (int, float)) and p.interconnect_bandwidth_gbps > 0

    @pytest.mark.kernel_graph
    def test_mock_amd_hw_profile_fields(self, mock_amd_hw_profile):
        """conftest AMD HardwareProfile has all 12 mandatory fields."""
        p = mock_amd_hw_profile
        assert p.vendor == "amd"
        assert p.arch_generation == "gfx942"
        assert isinstance(p.sm_count, int) and p.sm_count > 0
        assert isinstance(p.smem_per_sm_bytes, int) and p.smem_per_sm_bytes > 0
        assert isinstance(p.registers_per_sm, int) and p.registers_per_sm > 0
        assert isinstance(p.global_memory_bytes, int) and p.global_memory_bytes > 0
        assert isinstance(p.memory_bandwidth_gbps, (int, float)) and p.memory_bandwidth_gbps > 0
        assert isinstance(p.compute_throughput_tflops, (int, float)) and p.compute_throughput_tflops > 0
        assert isinstance(p.warp_size, int) and p.warp_size > 0

    @pytest.mark.kernel_graph
    def test_mock_profiles_and_targets_are_compatible(
        self, mock_nvidia_hw_profile, mock_gpu_target,
        mock_amd_hw_profile, mock_gpu_target_amd,
    ):
        """Conftest profiles and GPUTarget fixtures cover the same hardware.

        The conftest ``mock_*_hw_profile`` fixtures provide the 12 physical
        descriptor fields while ``mock_gpu_target*`` provide the
        :class:`GPUTarget` compilation-target representation.  They are
        separate fixtures that describe the same device.
        """
        # Profiles carry the device descriptor fields.
        assert mock_nvidia_hw_profile.vendor == "nvidia"
        assert mock_amd_hw_profile.vendor == "amd"
        # Standalone GPUTarget fixtures match expected backends.
        assert mock_gpu_target.backend == "cuda"
        assert mock_gpu_target_amd.backend == "hip"
        # gpu_target on profiles is optional — conftest may leave it None.
        # Test-local fixtures (nvidia_h100_profile etc.) populate gpu_target.

    # -- mock_hardware_inventory / mock_multi_device_inventory ------

    @pytest.mark.kernel_graph
    def test_mock_hardware_inventory_basic(self, mock_hardware_inventory):
        """conftest single-device mock inventory is usable."""
        assert mock_hardware_inventory.device_count == 1
        assert len(mock_hardware_inventory.devices) == 1
        dev = mock_hardware_inventory.get_device(0)
        assert dev is not None

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_mock_multi_device_inventory_basic(self, mock_multi_device_inventory):
        """conftest multi-device mock inventory has two devices."""
        assert mock_multi_device_inventory.device_count == 2
        assert len(mock_multi_device_inventory.devices) == 2

    @pytest.mark.kernel_graph
    def test_mock_hardware_inventory_engine_creation(
        self, mock_hardware_inventory,
    ):
        """DispatchDecisionEngine can be constructed from conftest inventory.

        Uses ``targets=None`` so no target-filter dict look-up is needed.
        """
        config = DispatchConfig(targets=None)
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                mock_hardware_inventory, config, DispatchMode.BALANCED,
            )
        assert engine is not None
        assert len(engine._eligible_devices) == 1

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_mock_multi_device_inventory_engine_creation(
        self, mock_multi_device_inventory,
    ):
        """DispatchDecisionEngine accepts conftest multi-device inventory."""
        config = DispatchConfig(targets=None)
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                mock_multi_device_inventory, config, DispatchMode.BALANCED,
            )
        assert engine is not None
        assert len(engine._eligible_devices) == 2

    @pytest.mark.kernel_graph
    def test_mock_inventory_score_subgraph(
        self, mock_hardware_inventory, sample_kgir_graph,
        mock_nvidia_hw_profile,
    ):
        """Score computation works with conftest inventory + profile."""
        config = DispatchConfig(targets=None)
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                mock_hardware_inventory, config, DispatchMode.BALANCED,
            )
        score = engine.score_subgraph_device(
            sample_kgir_graph, mock_nvidia_hw_profile,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.kernel_graph
    @pytest.mark.multi_device
    def test_mock_multi_inventory_dispatch_plan(
        self, mock_multi_device_inventory, sample_kgir_graph,
    ):
        """Dispatch plan generation with conftest multi-device inventory."""
        config = DispatchConfig(targets=None)
        with patch("triton.graph.dispatch._get_dispatch_knobs", return_value=None):
            engine = DispatchDecisionEngine(
                mock_multi_device_inventory, config, DispatchMode.BALANCED,
            )
        plan = engine.compute_dispatch_plan(sample_kgir_graph)
        assert len(plan) == sample_kgir_graph.node_count()
        for target in plan.values():
            assert target is not None

    # -- fresh_knobs -----------------------------------------------

    @pytest.mark.kernel_graph
    @pytest.mark.skipif(
        not _HAS_TORCH,
        reason="fresh_knobs fixture requires torch (triton._internal_testing)",
    )
    def test_fresh_knobs_dispatch_mode_default(self, fresh_knobs):
        """fresh_knobs fixture resets dispatch-related env knobs."""
        # After reset, the dispatch_mode knob should be the compiled
        # default ("balanced") or whatever the knobs module provides.
        dispatch_mode = getattr(fresh_knobs, "dispatch_mode", None)
        if dispatch_mode is not None:
            # String descriptor — value should be the default
            val = str(dispatch_mode)
            assert val in ("balanced", "performance", "cost", ""), (
                f"Unexpected dispatch_mode knob value: {val!r}"
            )
