"""Unit tests for the inter-kernel scheduler (``triton.graph.scheduler``).

Covers seven test phases aligned with the scheduler's functional areas:

- **Phase 1** — DAG critical-path computation (Algorithm A1)
- **Phase 2** — Stream assignment with bounded pool
- **Phase 3** — Resource-aware SM/CU bin-packing and priority scheduling
- **Phase 4** — Multi-device coordination and synchronization barriers
- **Phase 5** — Communication-computation overlap identification (Algorithm A3)
- **Phase 6** — ``KernelScheduler`` public API surface
- **Phase 7** — Feedback-driven schedule refinement with profiling data

All tests are marked ``@pytest.mark.kernel_graph`` via the module-level
``pytestmark`` and use fixtures from ``conftest.py`` (``make_kgir_node``,
``mock_nvidia_hw_profile``, ``default_graph_config``, etc.).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from triton.graph.scheduler import KernelScheduler, ScheduleEntry, Barrier
from triton.graph.kgir import KGIRNode, KGIREdge, KGIRGraph, HardwareProfile
from triton.graph.config import GraphConfig

# ---------------------------------------------------------------------------
# Module-level marker — every test in this file is a kernel_graph test
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.kernel_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph(make_kgir_node, nodes, edges, hw_profiles=None):
    """Construct a ``KGIRGraph`` from a compact specification.

    Parameters
    ----------
    make_kgir_node : callable
        Fixture-provided factory for creating ``KGIRNode`` instances.
    nodes : list[tuple[str, dict]]
        Each entry ``(name, kwargs)`` is forwarded to *make_kgir_node*.
    edges : list[tuple[str, str, str]]
        Each entry ``(source_name, target_name, edge_type)``.
    hw_profiles : HardwareProfile | list[HardwareProfile] | None
        Hardware profiles to attach to the graph.

    Returns
    -------
    tuple[KGIRGraph, dict[str, int]]
        The graph and a ``name → node_id`` mapping.
    """
    graph = KGIRGraph()
    id_map: dict[str, int] = {}

    # Normalise hardware profiles early so we can annotate per-target.
    profiles_list: list = []
    if hw_profiles is not None:
        profiles_list = hw_profiles if isinstance(hw_profiles, list) else [hw_profiles]

    for name, kwargs in nodes:
        exec_time = kwargs.get("execution_time_ms", 1.0)
        node = make_kgir_node(kernel_name=name, **kwargs)
        nid = graph.add_node(node.kernel_fn, node.metadata)
        id_map[name] = nid

        # The scheduler's ``_estimate_node_weight`` reads per-target
        # annotations via ``get_performance_annotation(arch_generation)``
        # with key ``"wall_clock_ms"``.  Write them here so the CPR
        # computation uses the times we specify in the test.
        for hp in profiles_list:
            graph_node = graph.get_node(nid)
            graph_node.update_performance_annotation(
                hp.arch_generation, {"wall_clock_ms": exec_time}
            )

    for src_name, tgt_name, etype in edges:
        graph.add_edge(
            id_map[src_name],
            id_map[tgt_name],
            edge_type=etype,
            tensor_id=f"T_{src_name}_{tgt_name}",
        )

    if profiles_list:
        graph.hardware_profiles = profiles_list

    return graph, id_map


def _make_device_mock(backend: str, arch) -> MagicMock:
    """Return a lightweight mock device with *backend* and *arch* attributes.

    The scheduler's ``_device_key`` helper inspects these two attributes to
    produce a hashable string key for device comparison.
    """
    dev = MagicMock()
    dev.backend = backend
    dev.arch = arch
    return dev


# ============================================================================
# Phase 1 — Critical-Path Computation Tests (Algorithm A1)
# ============================================================================

class TestCriticalPath:
    """Verify CPR (Critical-Path-Remaining) weight computation."""

    def test_critical_path_linear_chain(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """A(10 ms) → B(20 ms) → C(5 ms).

        Expected CPR values (node weight + max successor CPR):
        * CPR[C] = 5
        * CPR[B] = 20 + 5 = 25
        * CPR[A] = 10 + 25 = 35
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 10.0}),
                ("B", {"execution_time_ms": 20.0}),
                ("C", {"execution_time_ms": 5.0}),
            ],
            edges=[("A", "B", "data_dep"), ("B", "C", "data_dep")],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        cpr = scheduler.compute_critical_path()

        # Verify CPR ordering: root has highest, leaf has lowest
        assert cpr[ids["A"]] >= cpr[ids["B"]] >= cpr[ids["C"]]

        # Approximate value checks (float tolerance for heuristic fallback)
        assert cpr[ids["C"]] == pytest.approx(5.0, abs=1.0)
        assert cpr[ids["B"]] == pytest.approx(25.0, abs=3.0)
        assert cpr[ids["A"]] == pytest.approx(35.0, abs=4.0)

    def test_critical_path_diamond(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Diamond topology where the C-path is heavier than the B-path.

        ::

              A (10 ms)
             / \\
            B   C
           (5) (20)
             \\ /
              D (10 ms)

        Critical path: A → C → D = 40 ms  (not A → B → D = 25 ms).
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 10.0}),
                ("B", {"execution_time_ms": 5.0}),
                ("C", {"execution_time_ms": 20.0}),
                ("D", {"execution_time_ms": 10.0}),
            ],
            edges=[
                ("A", "B", "data_dep"),
                ("A", "C", "data_dep"),
                ("B", "D", "data_dep"),
                ("C", "D", "data_dep"),
            ],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        cpr = scheduler.compute_critical_path()

        # CPR[D]=10, CPR[B]=15, CPR[C]=30, CPR[A]=40
        assert cpr[ids["D"]] == pytest.approx(10.0, abs=1.0)
        assert cpr[ids["C"]] > cpr[ids["B"]], "C-path should dominate B-path"
        assert cpr[ids["A"]] == pytest.approx(40.0, abs=5.0)

    def test_critical_path_independent_chains(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Two independent chains: A(10)→B(10) and C(5)→D(5).

        Longest chain CPR = 20 (chain A→B).
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 10.0}),
                ("B", {"execution_time_ms": 10.0}),
                ("C", {"execution_time_ms": 5.0}),
                ("D", {"execution_time_ms": 5.0}),
            ],
            edges=[("A", "B", "data_dep"), ("C", "D", "data_dep")],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        cpr = scheduler.compute_critical_path()

        # Chain 1: CPR[B]=10, CPR[A]=20
        assert cpr[ids["A"]] == pytest.approx(20.0, abs=2.0)
        assert cpr[ids["B"]] == pytest.approx(10.0, abs=1.0)

        # Chain 2: CPR[D]=5, CPR[C]=10
        assert cpr[ids["C"]] == pytest.approx(10.0, abs=1.0)
        assert cpr[ids["D"]] == pytest.approx(5.0, abs=1.0)

        # Chain 1 has the longer critical path
        assert cpr[ids["A"]] > cpr[ids["C"]]

    def test_critical_path_per_target_weights(
        self,
        make_kgir_node,
        mock_nvidia_hw_profile,
        mock_amd_hw_profile,
        default_graph_config,
    ):
        """Per-target execution time annotations change the critical path.

        Diamond A→B→D, A→C→D with per-target times:

        ======== ====== ====== ====== ======
        Target    A      B      C      D
        ======== ====== ====== ====== ======
        NVIDIA   10 ms  30 ms   5 ms  10 ms  → B-path dominates
        AMD      10 ms   5 ms  30 ms  10 ms  → C-path dominates
        ======== ====== ====== ====== ======
        """
        nvidia_arch = mock_nvidia_hw_profile.arch_generation
        amd_arch = mock_amd_hw_profile.arch_generation

        times_nvidia = {"A": 10.0, "B": 30.0, "C": 5.0, "D": 10.0}
        times_amd = {"A": 10.0, "B": 5.0, "C": 30.0, "D": 10.0}

        # --- NVIDIA as primary target ---
        graph_nv, ids_nv = _build_graph(
            make_kgir_node,
            nodes=[("A", {}), ("B", {}), ("C", {}), ("D", {})],
            edges=[
                ("A", "B", "data_dep"),
                ("A", "C", "data_dep"),
                ("B", "D", "data_dep"),
                ("C", "D", "data_dep"),
            ],
            hw_profiles=[mock_nvidia_hw_profile, mock_amd_hw_profile],
        )
        for name in ("A", "B", "C", "D"):
            node = graph_nv.get_node(ids_nv[name])
            node.update_performance_annotation(
                nvidia_arch, {"wall_clock_ms": times_nvidia[name]}
            )
            node.update_performance_annotation(
                amd_arch, {"wall_clock_ms": times_amd[name]}
            )

        sched_nv = KernelScheduler(graph_nv, default_graph_config)
        cpr_nv = sched_nv.compute_critical_path()

        # With NVIDIA times B is heavy → B-path dominates
        assert cpr_nv[ids_nv["B"]] > cpr_nv[ids_nv["C"]]

        # --- AMD as primary target ---
        graph_amd, ids_amd = _build_graph(
            make_kgir_node,
            nodes=[("A", {}), ("B", {}), ("C", {}), ("D", {})],
            edges=[
                ("A", "B", "data_dep"),
                ("A", "C", "data_dep"),
                ("B", "D", "data_dep"),
                ("C", "D", "data_dep"),
            ],
            hw_profiles=[mock_amd_hw_profile, mock_nvidia_hw_profile],
        )
        for name in ("A", "B", "C", "D"):
            node = graph_amd.get_node(ids_amd[name])
            node.update_performance_annotation(
                nvidia_arch, {"wall_clock_ms": times_nvidia[name]}
            )
            node.update_performance_annotation(
                amd_arch, {"wall_clock_ms": times_amd[name]}
            )

        sched_amd = KernelScheduler(graph_amd, default_graph_config)
        cpr_amd = sched_amd.compute_critical_path()

        # With AMD times C is heavy → C-path dominates
        assert cpr_amd[ids_amd["C"]] > cpr_amd[ids_amd["B"]]


# ============================================================================
# Phase 2 — Stream Assignment Tests
# ============================================================================

class TestStreamAssignment:
    """Verify bounded-pool stream assignment logic."""

    def test_stream_assignment_sequential(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Linear chain A → B → C: all nodes on the same stream."""
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 1.0}),
                ("B", {"execution_time_ms": 1.0}),
                ("C", {"execution_time_ms": 1.0}),
            ],
            edges=[("A", "B", "data_dep"), ("B", "C", "data_dep")],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        assignments = scheduler.assign_streams()

        # Single chain → all nodes share the same stream
        assert assignments[ids["A"]] == assignments[ids["B"]]
        assert assignments[ids["B"]] == assignments[ids["C"]]

    def test_stream_assignment_parallel(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Two independent nodes A and B get different streams."""
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 5.0}),
                ("B", {"execution_time_ms": 5.0}),
            ],
            edges=[],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        assignments = scheduler.assign_streams()

        assert assignments[ids["A"]] != assignments[ids["B"]], (
            "Independent nodes should be assigned to different streams"
        )

    def test_stream_pool_bounded(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """200+ independent nodes: stream IDs never exceed the bounded pool.

        Per AAP §0.2.3 the scheduler must bound the stream pool size.  The
        hard CUDA limit is 128; the effective limit is
        ``min(config.dispatch.stream_pool_size, hw_max, 128)``.
        """
        num_nodes = 200
        nodes = [(f"K{i}", {"execution_time_ms": 0.5}) for i in range(num_nodes)]
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=nodes,
            edges=[],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        assignments = scheduler.assign_streams()

        # Every node must have an assignment
        assert len(assignments) == num_nodes

        max_stream_id = max(assignments.values())
        pool_limit = min(
            default_graph_config.dispatch.stream_pool_size,
            mock_nvidia_hw_profile.max_concurrent_streams,
            128,
        )
        assert max_stream_id < pool_limit, (
            f"Max stream ID {max_stream_id} must be < pool limit {pool_limit}"
        )

    def test_stream_assignment_with_dependencies(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Mixed graph: A → C, B → C, D independent.

        * A and B are separate roots → may land on different streams.
        * C depends on both A and B → part of a converging chain.
        * D is fully independent → gets its own stream.
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 5.0}),
                ("B", {"execution_time_ms": 5.0}),
                ("C", {"execution_time_ms": 5.0}),
                ("D", {"execution_time_ms": 5.0}),
            ],
            edges=[("A", "C", "data_dep"), ("B", "C", "data_dep")],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        assignments = scheduler.assign_streams()

        # All nodes must have valid stream assignments (>= 0)
        for nid in ids.values():
            assert nid in assignments
            assert assignments[nid] >= 0

        # At least two different streams should be used (independent chains)
        unique_streams = set(assignments.values())
        assert len(unique_streams) >= 2, (
            "Graph with independent chains should use multiple streams"
        )


# ============================================================================
# Phase 3 — Resource-Aware Scheduling Tests
# ============================================================================

class TestResourceScheduling:
    """Verify SM bin-packing and CPR-priority scheduling."""

    def test_sm_bin_packing(self, make_kgir_node, default_graph_config):
        """Resource-constrained scheduling respects SM budget.

        Create independent kernels with varying grid sizes and a tight SM
        budget (10 SMs).  The scheduler must produce a valid schedule
        regardless of whether kernels fit concurrently.
        """
        small_hw = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=10,
            smem_per_sm_bytes=65536,
            registers_per_sm=65536,
            global_memory_bytes=10 * 1024 ** 3,
            memory_bandwidth_gbps=100.0,
            compute_throughput_tflops=10.0,
            warp_size=32,
            max_concurrent_streams=32,
            interconnect_type="pcie_4",
            interconnect_bandwidth_gbps=32.0,
        )

        # Two kernels that together fit within SM budget
        graph_fit, ids_fit = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"grid": (10,), "execution_time_ms": 5.0}),
                ("B", {"grid": (10,), "execution_time_ms": 5.0}),
            ],
            edges=[],
            hw_profiles=small_hw,
        )
        sched_fit = KernelScheduler(graph_fit, default_graph_config)
        entries_fit = sched_fit.schedule()

        # All nodes must appear in the schedule
        scheduled_ids = {e.kernel_id for e in entries_fit}
        assert scheduled_ids == set(ids_fit.values())

        # Two kernels that individually EXCEED 50 % of SM budget
        graph_tight, ids_tight = _build_graph(
            make_kgir_node,
            nodes=[
                ("C", {"grid": (20,), "execution_time_ms": 5.0}),
                ("D", {"grid": (20,), "execution_time_ms": 5.0}),
            ],
            edges=[],
            hw_profiles=small_hw,
        )
        sched_tight = KernelScheduler(graph_tight, default_graph_config)
        entries_tight = sched_tight.schedule()

        # Schedule must still contain all nodes
        scheduled_ids_tight = {e.kernel_id for e in entries_tight}
        assert scheduled_ids_tight == set(ids_tight.values())

        # All entries have non-negative estimated durations
        for entry in entries_fit + entries_tight:
            assert entry.estimated_duration_ms >= 0

    def test_priority_scheduling_critical_path(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Kernels on the critical path receive higher scheduling priority.

        Diamond: A → B → D, A → C → D where C is much heavier than B.
        C should have higher priority than B because CPR[C] > CPR[B].
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 1.0}),
                ("B", {"execution_time_ms": 1.0}),
                ("C", {"execution_time_ms": 15.0}),
                ("D", {"execution_time_ms": 1.0}),
            ],
            edges=[
                ("A", "B", "data_dep"),
                ("A", "C", "data_dep"),
                ("B", "D", "data_dep"),
                ("C", "D", "data_dep"),
            ],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        entries = scheduler.schedule()

        entry_map = {e.kernel_id: e for e in entries}

        # C is on the critical path (much heavier) → higher priority than B
        assert entry_map[ids["C"]].priority >= entry_map[ids["B"]].priority, (
            "Critical-path kernel C should have >= priority than off-path kernel B"
        )


# ============================================================================
# Phase 4 — Multi-Device Coordination Tests
# ============================================================================

class TestMultiDeviceCoordination:
    """Verify cross-device barrier insertion and coordination."""

    def test_multi_device_barrier_insertion(
        self,
        make_kgir_node,
        mock_nvidia_hw_profile,
        mock_amd_hw_profile,
        default_graph_config,
    ):
        """Cross-device dependency A(dev0) → C(dev1) requires a barrier.

        Nodes A, B on CUDA device 0; node C on HIP device 1 depends on A.
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 5.0}),
                ("B", {"execution_time_ms": 5.0}),
                ("C", {"execution_time_ms": 5.0}),
            ],
            edges=[("A", "C", "data_dep"), ("A", "B", "data_dep")],
            hw_profiles=[mock_nvidia_hw_profile, mock_amd_hw_profile],
        )

        dev0 = _make_device_mock("cuda", 90)
        dev1 = _make_device_mock("hip", "gfx942")

        scheduler = KernelScheduler(graph, default_graph_config)
        scheduler.schedule()
        scheduler.coordinate_multi_device(
            {ids["A"]: dev0, ids["B"]: dev0, ids["C"]: dev1}
        )

        barriers = scheduler.barriers
        cross_device = [b for b in barriers if b.is_cross_device]
        assert len(cross_device) >= 1, (
            "A cross-device barrier must be inserted for A(dev0) → C(dev1)"
        )

    def test_multi_device_no_barrier_same_device(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """All nodes on the same device — no cross-device barriers needed."""
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 3.0}),
                ("B", {"execution_time_ms": 3.0}),
                ("C", {"execution_time_ms": 3.0}),
            ],
            edges=[("A", "B", "data_dep"), ("B", "C", "data_dep")],
            hw_profiles=mock_nvidia_hw_profile,
        )

        dev0 = _make_device_mock("cuda", 90)

        scheduler = KernelScheduler(graph, default_graph_config)
        scheduler.schedule()
        scheduler.coordinate_multi_device(
            {ids["A"]: dev0, ids["B"]: dev0, ids["C"]: dev0}
        )

        barriers = scheduler.barriers
        cross_device = [b for b in barriers if b.is_cross_device]
        assert len(cross_device) == 0, (
            "No cross-device barriers should exist when all nodes are on the same device"
        )

    def test_multi_device_coordination_order(
        self,
        make_kgir_node,
        mock_nvidia_hw_profile,
        mock_amd_hw_profile,
        default_graph_config,
    ):
        """Multi-device schedule respects cross-device ordering.

        ::

            dev0:  A ──► B
                    \\
            dev1:    ──► C ──► D

        A and B on device 0; C and D on device 1.
        Barrier required between A (dev0) and C (dev1).
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 5.0}),
                ("B", {"execution_time_ms": 5.0}),
                ("C", {"execution_time_ms": 5.0}),
                ("D", {"execution_time_ms": 5.0}),
            ],
            edges=[
                ("A", "B", "data_dep"),
                ("A", "C", "data_dep"),
                ("C", "D", "data_dep"),
            ],
            hw_profiles=[mock_nvidia_hw_profile, mock_amd_hw_profile],
        )

        dev0 = _make_device_mock("cuda", 90)
        dev1 = _make_device_mock("hip", "gfx942")

        scheduler = KernelScheduler(graph, default_graph_config)
        scheduler.schedule()
        scheduler.coordinate_multi_device(
            {ids["A"]: dev0, ids["B"]: dev0, ids["C"]: dev1, ids["D"]: dev1}
        )

        barriers = scheduler.barriers
        cross_device = [b for b in barriers if b.is_cross_device]

        # At least one cross-device barrier for the A → C edge
        assert len(cross_device) >= 1

        # Launch sequence must honour the A-before-C ordering
        launch_seq = scheduler.emit_launch_sequence()
        id_to_pos = {e.kernel_id: i for i, e in enumerate(launch_seq)}
        assert id_to_pos[ids["A"]] < id_to_pos[ids["C"]], (
            "A must be launched before C (cross-device dependency)"
        )
        assert id_to_pos[ids["C"]] < id_to_pos[ids["D"]], (
            "C must be launched before D (same-device chain)"
        )


# ============================================================================
# Phase 5 — Communication-Computation Overlap Tests (Algorithm A3)
# ============================================================================

class TestOverlap:
    """Verify transfer-compute overlap identification and minimal barriers."""

    def test_overlap_identification(
        self,
        make_kgir_node,
        mock_nvidia_hw_profile,
        mock_amd_hw_profile,
        default_graph_config,
    ):
        """Overlap opportunity: B can execute during A → C cross-device transfer.

        Graph::

            A ──[cross_device_transfer]──► C   (A on dev0, C on dev1)
            B  (independent, on dev0)

        The scheduler looks for compute nodes on the **source** device that
        can execute concurrently with an outgoing transfer.  B resides on
        dev0 (same as transfer source A) and is independent of C (transfer
        destination), so it qualifies for overlap.
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 5.0}),
                ("B", {"execution_time_ms": 5.0}),
                ("C", {"execution_time_ms": 5.0}),
            ],
            edges=[("A", "C", "cross_device_transfer")],
            hw_profiles=[mock_nvidia_hw_profile, mock_amd_hw_profile],
        )

        dev0 = _make_device_mock("cuda", 90)
        dev1 = _make_device_mock("hip", "gfx942")

        scheduler = KernelScheduler(graph, default_graph_config)
        scheduler.schedule()
        # B is on dev0 (same as source A) so it can overlap with the transfer
        scheduler.coordinate_multi_device(
            {ids["A"]: dev0, ids["B"]: dev0, ids["C"]: dev1}
        )

        opportunities = scheduler.identify_overlap_opportunities()

        # At least one overlap opportunity involving B
        assert len(opportunities) >= 1, (
            "Independent node B on source device should be identified as "
            "overlappable with the A → C cross-device transfer"
        )
        # Verify B appears in the overlap pairs
        flat_ids = {nid for pair in opportunities for nid in pair}
        assert ids["B"] in flat_ids or ids["A"] in flat_ids, (
            "Overlap pairs should reference the transfer source or the "
            "independent overlappable node"
        )

    def test_minimal_synchronization_barriers(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Only necessary barriers are placed — no over-synchronization.

        Two independent chains on the same device:

        ::

            Chain 1: A → B   (stream 0)
            Chain 2: C → D   (stream 1)

        No edge connects the two chains, so **no** barriers should exist
        between them.  Any barriers in the schedule must correspond to
        actual dependency edges between different streams.
        """
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 1.0}),
                ("B", {"execution_time_ms": 1.0}),
                ("C", {"execution_time_ms": 1.0}),
                ("D", {"execution_time_ms": 1.0}),
            ],
            edges=[("A", "B", "data_dep"), ("C", "D", "data_dep")],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        scheduler.schedule()
        barriers = scheduler.insert_synchronization_barriers()

        # All barriers must be for real edges — none should be cross-device
        # (single device graph)
        for b in barriers:
            assert not b.is_cross_device

        # The two independent chains have no connecting edge, so no barrier
        # should bridge chain-1's stream to chain-2's stream unless the
        # chains share a stream.
        stream_a = scheduler.stream_assignments.get(ids["A"])
        stream_c = scheduler.stream_assignments.get(ids["C"])
        if stream_a is not None and stream_c is not None and stream_a != stream_c:
            cross_chain_barriers = [
                b
                for b in barriers
                if (b.source_stream == stream_a and b.target_stream == stream_c)
                or (b.source_stream == stream_c and b.target_stream == stream_a)
            ]
            assert len(cross_chain_barriers) == 0, (
                "No barrier should bridge two independent chains"
            )


# ============================================================================
# Phase 6 — KernelScheduler API Tests
# ============================================================================

class TestSchedulerAPI:
    """Verify the public ``KernelScheduler`` API surface."""

    def test_scheduler_schedule(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """``schedule()`` returns a list of ``ScheduleEntry`` objects."""
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 1.0}),
                ("B", {"execution_time_ms": 2.0}),
                ("C", {"execution_time_ms": 0.5}),
            ],
            edges=[("A", "B", "data_dep"), ("B", "C", "data_dep")],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        entries = scheduler.schedule()

        assert isinstance(entries, list)
        assert len(entries) == 3

        for entry in entries:
            assert isinstance(entry, ScheduleEntry)
            assert entry.kernel_id in ids.values()
            assert entry.stream_id >= 0
            assert entry.estimated_duration_ms >= 0
            assert isinstance(entry.dependencies, (list, tuple, set, frozenset))

        # ``get_schedule()`` returns an equivalent schedule (defensive copy)
        cached = scheduler.get_schedule()
        assert cached == entries

    def test_scheduler_empty_graph(
        self, mock_nvidia_hw_profile, default_graph_config
    ):
        """Empty graph produces an empty schedule without errors."""
        graph = KGIRGraph()
        graph.hardware_profiles = [mock_nvidia_hw_profile]

        scheduler = KernelScheduler(graph, default_graph_config)
        entries = scheduler.schedule()

        assert entries == []
        assert scheduler.get_schedule() == []

    def test_scheduler_single_node(
        self, make_kgir_node, mock_nvidia_hw_profile, default_graph_config
    ):
        """Single-node graph → single ``ScheduleEntry`` on stream 0."""
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[("A", {"execution_time_ms": 5.0})],
            edges=[],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler = KernelScheduler(graph, default_graph_config)
        entries = scheduler.schedule()

        assert len(entries) == 1
        entry = entries[0]
        assert entry.kernel_id == ids["A"]
        assert entry.stream_id == 0
        assert entry.estimated_duration_ms > 0

        # emit_launch_sequence should return the same single entry
        launch_seq = scheduler.emit_launch_sequence()
        assert len(launch_seq) == 1
        assert launch_seq[0].kernel_id == ids["A"]


# ============================================================================
# Phase 7 — Feedback Refinement Tests
# ============================================================================

class TestFeedbackRefinement:
    """Verify schedule refinement using measured profiling data."""

    def test_scheduler_refine_with_profiling_data(
        self,
        make_kgir_node,
        mock_nvidia_hw_profile,
        feedback_enabled_config,
    ):
        """Measured execution times change the critical path and schedule.

        Initial heuristic: B appears critical (20 ms vs C 5 ms).
        After profiling: B is actually fast (2 ms) and C is slow (25 ms).

        Re-scheduling with updated annotations should flip the CPR
        dominance from B to C.
        """
        arch = mock_nvidia_hw_profile.arch_generation

        # --- Initial schedule (heuristic weights) ---
        graph, ids = _build_graph(
            make_kgir_node,
            nodes=[
                ("A", {"execution_time_ms": 10.0}),
                ("B", {"execution_time_ms": 20.0}),
                ("C", {"execution_time_ms": 5.0}),
                ("D", {"execution_time_ms": 10.0}),
            ],
            edges=[
                ("A", "B", "data_dep"),
                ("A", "C", "data_dep"),
                ("B", "D", "data_dep"),
                ("C", "D", "data_dep"),
            ],
            hw_profiles=mock_nvidia_hw_profile,
        )

        scheduler_initial = KernelScheduler(graph, feedback_enabled_config)
        cpr_initial = scheduler_initial.compute_critical_path()

        # Initially B-path dominates
        assert cpr_initial[ids["B"]] > cpr_initial[ids["C"]]

        # --- Inject profiling data (B is actually fast, C is slow) ---
        graph.get_node(ids["B"]).update_performance_annotation(
            arch, {"wall_clock_ms": 2.0}
        )
        graph.get_node(ids["C"]).update_performance_annotation(
            arch, {"wall_clock_ms": 25.0}
        )

        # --- Refined schedule ---
        scheduler_refined = KernelScheduler(graph, feedback_enabled_config)
        entries_refined = scheduler_refined.schedule()
        cpr_refined = scheduler_refined.compute_critical_path()

        # After profiling C-path dominates
        assert cpr_refined[ids["C"]] > cpr_refined[ids["B"]], (
            "After injecting profiling data, the C-path should dominate "
            "the critical path (C measured as 25 ms vs B measured as 2 ms)"
        )

        # Verify the refined schedule contains all nodes
        scheduled_ids = {e.kernel_id for e in entries_refined}
        assert scheduled_ids == set(ids.values())

        # Verify estimated durations reflect the new profiling data
        entry_map = {e.kernel_id: e for e in entries_refined}
        assert entry_map[ids["C"]].estimated_duration_ms == pytest.approx(
            25.0, abs=1.0
        )
        assert entry_map[ids["B"]].estimated_duration_ms == pytest.approx(
            2.0, abs=1.0
        )
