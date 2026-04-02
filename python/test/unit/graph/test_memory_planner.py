"""Unit tests for the memory planning pass.

Covers five test phases:
  1. Liveness analysis (basic chain, overlapping intervals, dead-after-last-consumer, graph-output exclusion)
  2. Promotion decisions (global→shared, SMEM capacity respect, register file, fallback to global, per-target)
  3. Cross-device transfers (insertion, no-transfer on single device)
  4. Feedback refinement (revert on occupancy degradation, keep beneficial)
  5. MemoryPlanner API (plan, empty graph, no intermediates)

All 16 tests are marked ``@pytest.mark.kernel_graph`` per AAP §0.7.5.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from triton.graph.memory_planner import MemoryPlanner, TransferOp
from triton.graph.kgir import KGIRGraph, KGIRNode, KGIREdge, HardwareProfile, NodeMetadata
from triton.graph.config import GraphConfig, FeedbackConfig, FusionConfig, DispatchConfig


# ---------------------------------------------------------------------------
# Helper: build a graph with controlled tensor metadata
# ---------------------------------------------------------------------------

def _make_metadata(
    tensor_shapes: dict | None = None,
    tensor_dtypes: dict | None = None,
    grid: tuple = (128,),
    smem_bytes: int = 0,
    register_count: int = 32,
) -> NodeMetadata:
    """Create a ``NodeMetadata`` with explicit tensor shapes/dtypes."""
    return NodeMetadata(
        grid_dimensions=grid,
        tensor_shapes=tensor_shapes or {},
        tensor_dtypes=tensor_dtypes or {},
        shared_memory_bytes=smem_bytes,
        register_count=register_count,
    )


def _mock_kernel(name: str = "kernel") -> MagicMock:
    """Create a mock kernel function with standard attributes."""
    k = MagicMock()
    k.__name__ = name
    k.fn = MagicMock()
    k.fn.__name__ = name
    return k


def _default_config(feedback_enable: bool = False) -> GraphConfig:
    """Return a ``GraphConfig`` with feedback toggled on/off."""
    return GraphConfig(
        feedback=FeedbackConfig(enable=feedback_enable),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Liveness Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestLivenessAnalysis:
    """Tests for ``MemoryPlanner.compute_liveness`` and ``identify_intermediates``."""

    def test_liveness_basic_chain(self):
        """A→B→C chain with T1(A→B), T2(B→C).

        Topological order [A, B, C] → positions 0, 1, 2.
        Expected: liveness[1] == (0, 1), liveness[2] == (1, 2).
        Both T1 and T2 are intermediate (produced & consumed inside the graph).
        """
        graph = KGIRGraph()
        # Node A produces tensor 1 (shape 32×32, fp32)
        id_a = graph.add_node(
            _mock_kernel("A"),
            _make_metadata(tensor_shapes={1: (32, 32)}, tensor_dtypes={1: "fp32"}),
        )
        # Node B consumes tensor 1, produces tensor 2
        id_b = graph.add_node(
            _mock_kernel("B"),
            _make_metadata(tensor_shapes={2: (32, 32)}, tensor_dtypes={2: "fp32"}),
        )
        # Node C consumes tensor 2
        id_c = graph.add_node(
            _mock_kernel("C"),
            _make_metadata(),
        )

        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=1)
        graph.add_edge(id_b, id_c, edge_type="data_dep", tensor_id=2)

        planner = MemoryPlanner(graph, _default_config())

        intermediates = planner.identify_intermediates()
        assert 1 in intermediates, "T1 should be intermediate"
        assert 2 in intermediates, "T2 should be intermediate"

        liveness = planner.compute_liveness()
        # Topological order: A(0), B(1), C(2)
        assert liveness[1] == (0, 1), f"T1 liveness mismatch: {liveness[1]}"
        assert liveness[2] == (1, 2), f"T2 liveness mismatch: {liveness[2]}"

    def test_liveness_overlapping_intervals(self):
        """A→C, B→C, C→D with T3(A→C) and T4(B→C).

        Both intermediates T3 and T4 are live simultaneously up to C's position.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("A"),
            _make_metadata(tensor_shapes={3: (16, 16)}, tensor_dtypes={3: "fp32"}),
        )
        id_b = graph.add_node(
            _mock_kernel("B"),
            _make_metadata(tensor_shapes={4: (16, 16)}, tensor_dtypes={4: "fp32"}),
        )
        id_c = graph.add_node(
            _mock_kernel("C"),
            _make_metadata(tensor_shapes={5: (16, 16)}, tensor_dtypes={5: "fp32"}),
        )
        id_d = graph.add_node(_mock_kernel("D"), _make_metadata())

        graph.add_edge(id_a, id_c, edge_type="data_dep", tensor_id=3)
        graph.add_edge(id_b, id_c, edge_type="data_dep", tensor_id=4)
        graph.add_edge(id_c, id_d, edge_type="data_dep", tensor_id=5)

        planner = MemoryPlanner(graph, _default_config())

        intermediates = planner.identify_intermediates()
        assert 3 in intermediates
        assert 4 in intermediates

        liveness = planner.compute_liveness()
        # In topological order A and B precede C which precedes D.
        # T3: birth=pos(A), death=pos(C)
        # T4: birth=pos(B), death=pos(C)
        topo = graph.topological_sort()
        pos = {nid: idx for idx, nid in enumerate(topo)}
        assert liveness[3] == (pos[id_a], pos[id_c])
        assert liveness[4] == (pos[id_b], pos[id_c])

        # Overlapping: both T3 and T4 are live at C's position
        assert liveness[3][1] == liveness[4][1], "T3 and T4 should both die at C"

    def test_liveness_dead_after_last_consumer(self):
        """A→B→C→D chain (T1 A→B, T2 B→C, T3 C→D).

        T1 dies at position of B; it is dead before C and D execute.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("A"),
            _make_metadata(tensor_shapes={1: (8, 8)}, tensor_dtypes={1: "fp32"}),
        )
        id_b = graph.add_node(
            _mock_kernel("B"),
            _make_metadata(tensor_shapes={2: (8, 8)}, tensor_dtypes={2: "fp32"}),
        )
        id_c = graph.add_node(
            _mock_kernel("C"),
            _make_metadata(tensor_shapes={3: (8, 8)}, tensor_dtypes={3: "fp32"}),
        )
        id_d = graph.add_node(_mock_kernel("D"), _make_metadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=1)
        graph.add_edge(id_b, id_c, edge_type="data_dep", tensor_id=2)
        graph.add_edge(id_c, id_d, edge_type="data_dep", tensor_id=3)

        planner = MemoryPlanner(graph, _default_config())
        liveness = planner.compute_liveness()

        topo = graph.topological_sort()
        pos = {nid: idx for idx, nid in enumerate(topo)}

        # T1 dies at B's position, which is before C and D
        assert liveness[1][1] == pos[id_b]
        assert liveness[1][1] < pos[id_c], "T1 must be dead before C"
        assert liveness[1][1] < pos[id_d], "T1 must be dead before D"

    def test_liveness_graph_output_not_promotable(self):
        """Tensors that only appear as producer output (no consuming edge)
        are NOT intermediate and therefore not promotable.

        Graph: A→B with T1(A→B).  Node A also has tensor 99 in its metadata
        but there is no consuming edge for tensor 99 — it is a graph output.
        Tensor 99 must NOT appear in intermediates.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("A"),
            _make_metadata(
                tensor_shapes={1: (32, 32), 99: (64, 64)},
                tensor_dtypes={1: "fp32", 99: "fp32"},
            ),
        )
        id_b = graph.add_node(_mock_kernel("B"), _make_metadata())

        # Only T1 has a consuming edge; T99 has no consuming edge.
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=1)

        planner = MemoryPlanner(graph, _default_config())
        intermediates = planner.identify_intermediates()

        assert 1 in intermediates, "T1 (produced and consumed) should be intermediate"
        assert 99 not in intermediates, "T99 (no consumer edge) should NOT be intermediate"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Promotion Decision Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestPromotionDecisions:
    """Tests for ``MemoryPlanner.plan_promotions``."""

    def test_promote_global_to_shared_basic(self, mock_nvidia_hw_profile):
        """Fused producer-consumer: intermediate tensor (32×32 fp32 = 4 KB)
        fits within NVIDIA SMEM budget (228 KB × 0.5 = 114 KB) and
        producer/consumer are fused → promoted to ``"shared"``.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("producer"),
            _make_metadata(
                tensor_shapes={10: (32, 32)},
                tensor_dtypes={10: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("consumer"),
            _make_metadata(smem_bytes=0),
        )
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=10)

        # Mark producer as fused with consumer via fused_from.
        # _are_producer_consumer_fused checks: pnode.is_fused + cid in pnode.fused_from → True
        node_a = graph.get_node(id_a)
        node_a.is_fused = True
        node_a.fused_from = [id_a, id_b]

        planner = MemoryPlanner(graph, _default_config())
        promotions = planner.plan_promotions(mock_nvidia_hw_profile)

        # 32×32 fp32 = 4096 bytes → fits 114 KB budget, fused → "shared"
        assert promotions[10] == "shared", f"Expected 'shared', got {promotions[10]}"

    def test_promote_respects_smem_capacity(self):
        """When a tensor exceeds the SMEM budget, it stays in global memory.

        Hardware: 64 KB SMEM → budget = 32 KB (50%).
        Small tensor (16 KB): fits → "shared".
        Large tensor (64 KB): exceeds → "global".
        """
        hw = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_80",
            sm_count=108,
            smem_per_sm_bytes=64 * 1024,  # 64 KB → budget = 32 KB
            registers_per_sm=65536,
            global_memory_bytes=40 * (1024 ** 3),
            memory_bandwidth_gbps=2000.0,
            compute_throughput_tflops=312.0,
            warp_size=32,
            max_concurrent_streams=128,
            interconnect_type="nvlink_3",
            interconnect_bandwidth_gbps=600.0,
        )

        graph = KGIRGraph()
        # Node A produces small T10 (64×64 fp32 = 16 KB) and large T11 (128×128 fp32 = 64 KB)
        id_a = graph.add_node(
            _mock_kernel("producer"),
            _make_metadata(
                tensor_shapes={10: (64, 64), 11: (128, 128)},
                tensor_dtypes={10: "fp32", 11: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("consumer_small"),
            _make_metadata(smem_bytes=0),
        )
        id_c = graph.add_node(
            _mock_kernel("consumer_large"),
            _make_metadata(smem_bytes=0),
        )

        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=10)
        graph.add_edge(id_a, id_c, edge_type="data_dep", tensor_id=11)

        # Fuse producer with both consumers for the promotion eligibility check.
        node_a = graph.get_node(id_a)
        node_a.is_fused = True
        node_a.fused_from = [id_a, id_b, id_c]

        planner = MemoryPlanner(graph, _default_config())
        promotions = planner.plan_promotions(hw)

        # 64×64 fp32 = 16384 bytes (16 KB) ≤ 32 KB budget → "shared"
        assert promotions[10] == "shared", f"Small tensor: expected 'shared', got {promotions[10]}"
        # 128×128 fp32 = 65536 bytes (64 KB) > 32 KB budget → "global"
        assert promotions[11] == "global", f"Large tensor: expected 'global', got {promotions[11]}"

    def test_promote_register_file(self):
        """Very small tensor (8×8 fp32 = 64 elements) with ALL nodes fused
        together → promoted to ``"register"``.

        ``_are_all_fused_together`` returns True when both nodes share the
        same ``fused_from`` list (is_fused=False, fused_from=[pid, cid]).
        """
        hw = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=132,
            smem_per_sm_bytes=228 * 1024,
            registers_per_sm=65536,
            global_memory_bytes=80 * (1024 ** 3),
            memory_bandwidth_gbps=3350.0,
            compute_throughput_tflops=989.0,
            warp_size=32,
            max_concurrent_streams=128,
            interconnect_type="nvlink_4",
            interconnect_bandwidth_gbps=900.0,
        )

        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("producer"),
            _make_metadata(
                tensor_shapes={20: (8, 8)},
                tensor_dtypes={20: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("consumer"),
            _make_metadata(smem_bytes=0),
        )
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=20)

        # For _are_all_fused_together: both nodes share same fused_from tuple.
        # Setting is_fused=False + fused_from=[id_a, id_b] on BOTH nodes.
        node_a = graph.get_node(id_a)
        node_a.is_fused = False
        node_a.fused_from = [id_a, id_b]

        node_b = graph.get_node(id_b)
        node_b.is_fused = False
        node_b.fused_from = [id_a, id_b]

        planner = MemoryPlanner(graph, _default_config())
        promotions = planner.plan_promotions(hw)

        # 8×8 fp32 = 64 elements ≤ 64, all fused → "register"
        assert promotions[20] == "register", f"Expected 'register', got {promotions[20]}"

    def test_fallback_to_global_memory(self, mock_nvidia_hw_profile):
        """Unfused nodes: producer-consumer are NOT fused, so the tensor
        stays in global memory even if it fits SMEM budget.

        Both ``_are_all_fused_together`` and ``_are_producer_consumer_fused``
        return False when nodes are not fused.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("producer"),
            _make_metadata(
                tensor_shapes={30: (32, 32)},
                tensor_dtypes={30: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("consumer"),
            _make_metadata(smem_bytes=0),
        )
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=30)

        # Nodes are NOT fused (default: is_fused=False, fused_from=None).
        planner = MemoryPlanner(graph, _default_config())
        promotions = planner.plan_promotions(mock_nvidia_hw_profile)

        assert promotions[30] == "global", f"Unfused → expected 'global', got {promotions[30]}"

    def test_promotion_per_target(
        self,
        mock_nvidia_hw_profile,
        mock_small_hw_profile,
    ):
        """Same fused graph, different hardware targets yield different
        promotion decisions due to varying SMEM capacity.

        NVIDIA (228 KB SMEM → 114 KB budget): 64 KB tensor fits → "shared".
        Small  ( 16 KB SMEM →   8 KB budget): 64 KB tensor exceeds → "global".
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("producer"),
            _make_metadata(
                tensor_shapes={40: (128, 128)},
                tensor_dtypes={40: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("consumer"),
            _make_metadata(smem_bytes=0),
        )
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=40)

        # Fuse producer with consumer.
        node_a = graph.get_node(id_a)
        node_a.is_fused = True
        node_a.fused_from = [id_a, id_b]

        planner = MemoryPlanner(graph, _default_config())

        # NVIDIA: 128×128 fp32 = 65536 bytes (64 KB) ≤ 114 KB budget → "shared"
        nvidia_plan = planner.plan_promotions(mock_nvidia_hw_profile)
        assert nvidia_plan[40] == "shared", f"NVIDIA: expected 'shared', got {nvidia_plan[40]}"

        # Small HW: 64 KB > 8 KB budget → "global"
        # Need a fresh planner to avoid cache
        planner2 = MemoryPlanner(graph, _default_config())
        small_plan = planner2.plan_promotions(mock_small_hw_profile)
        assert small_plan[40] == "global", f"Small HW: expected 'global', got {small_plan[40]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Cross-Device Transfer Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestCrossDeviceTransfers:
    """Tests for ``MemoryPlanner.insert_transfers``."""

    def test_cross_device_transfer_insertion(
        self,
        mock_gpu_target,
        mock_gpu_target_amd,
    ):
        """Graph spanning two devices: transfer operation is inserted at the
        device boundary with correct metadata.

        Node A on NVIDIA (cuda:90), Node B on AMD (hip:gfx942).
        Cross-vendor → transfer_type="host_staged".
        """
        # Build hardware profiles WITH gpu_target set.
        hw_nvidia = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=132,
            smem_per_sm_bytes=228 * 1024,
            registers_per_sm=65536,
            global_memory_bytes=80 * (1024 ** 3),
            memory_bandwidth_gbps=3350.0,
            compute_throughput_tflops=989.0,
            warp_size=32,
            max_concurrent_streams=128,
            interconnect_type="nvlink_4",
            interconnect_bandwidth_gbps=900.0,
            gpu_target=mock_gpu_target,
        )
        hw_amd = HardwareProfile(
            vendor="amd",
            arch_generation="gfx942",
            sm_count=304,
            smem_per_sm_bytes=64 * 1024,
            registers_per_sm=65536,
            global_memory_bytes=192 * (1024 ** 3),
            memory_bandwidth_gbps=5300.0,
            compute_throughput_tflops=1307.0,
            warp_size=64,
            max_concurrent_streams=128,
            interconnect_type="infinity_fabric",
            interconnect_bandwidth_gbps=896.0,
            gpu_target=mock_gpu_target_amd,
        )

        graph = KGIRGraph(hardware_profiles=[hw_nvidia, hw_amd])
        id_a = graph.add_node(
            _mock_kernel("A"),
            _make_metadata(
                tensor_shapes={50: (256, 256)},
                tensor_dtypes={50: "fp32"},
            ),
        )
        id_b = graph.add_node(_mock_kernel("B"), _make_metadata())
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=50)

        # Dispatch plan: A → NVIDIA, B → AMD (different devices)
        dispatch_plan = {id_a: mock_gpu_target, id_b: mock_gpu_target_amd}

        planner = MemoryPlanner(graph, _default_config())
        transfers = planner.insert_transfers(dispatch_plan)

        assert len(transfers) >= 1, "Expected at least one transfer operation"
        t = transfers[0]
        assert isinstance(t, TransferOp)
        assert t.tensor_id == 50
        assert t.source_device.backend == "cuda"
        assert t.target_device.backend == "hip"
        assert t.transfer_type == "host_staged", (
            "Cross-vendor transfer should be host_staged"
        )
        assert t.size_bytes > 0
        assert t.estimated_time_ms > 0

    def test_no_transfer_single_device(self, mock_gpu_target):
        """When all kernels are on the same device, no transfers are inserted."""
        hw_nvidia = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=132,
            smem_per_sm_bytes=228 * 1024,
            registers_per_sm=65536,
            global_memory_bytes=80 * (1024 ** 3),
            memory_bandwidth_gbps=3350.0,
            compute_throughput_tflops=989.0,
            warp_size=32,
            max_concurrent_streams=128,
            interconnect_type="nvlink_4",
            interconnect_bandwidth_gbps=900.0,
            gpu_target=mock_gpu_target,
        )

        graph = KGIRGraph(hardware_profiles=[hw_nvidia])
        id_a = graph.add_node(
            _mock_kernel("A"),
            _make_metadata(
                tensor_shapes={60: (64, 64)},
                tensor_dtypes={60: "fp32"},
            ),
        )
        id_b = graph.add_node(_mock_kernel("B"), _make_metadata())
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=60)

        # Both nodes on the same NVIDIA device
        dispatch_plan = {id_a: mock_gpu_target, id_b: mock_gpu_target}

        planner = MemoryPlanner(graph, _default_config())
        transfers = planner.insert_transfers(dispatch_plan)

        assert len(transfers) == 0, f"Expected no transfers, got {len(transfers)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Feedback Refinement Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestFeedbackRefinement:
    """Tests for ``MemoryPlanner.refine_with_feedback``."""

    def test_revert_promotion_on_occupancy_degradation(self, mock_nvidia_hw_profile):
        """When measured occupancy is significantly below expected occupancy
        (ratio < 0.85), the planner reverts shared promotion to global.

        Scenario: tensor 10 promoted to "shared", then feedback shows
        consumer occupancy = 0.40, expected = 0.80 → ratio 0.50 < 0.85 → revert.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("producer"),
            _make_metadata(
                tensor_shapes={10: (32, 32)},
                tensor_dtypes={10: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("consumer"),
            _make_metadata(smem_bytes=0),
        )
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=10)

        # Fuse so that plan_promotions promotes to "shared".
        node_a = graph.get_node(id_a)
        node_a.is_fused = True
        node_a.fused_from = [id_a, id_b]

        config = GraphConfig(feedback=FeedbackConfig(enable=True, sensitivity=0.15))
        planner = MemoryPlanner(graph, config)

        # Phase 1: initial promotion → "shared"
        initial = planner.plan_promotions(mock_nvidia_hw_profile)
        assert initial[10] == "shared", "Pre-condition: tensor should be promoted to shared"

        # Phase 2: feedback shows occupancy degradation on consumer node.
        # _detect_occupancy_degradation checks consumer metrics: ratio = actual/expected < 0.85.
        measured_metrics = {
            id_b: {
                "occupancy": 0.40,
                "expected_occupancy": 0.80,
                "register_spill_bytes": 0,
                "wall_clock_ms": 2.5,
            },
        }

        refined = planner.refine_with_feedback(measured_metrics)

        # refine_with_feedback returns a flat Dict[int, str] combining all
        # per-target plans.  Tensor 10 should be reverted to "global".
        assert 10 in refined, (
            f"Expected tensor 10 in refined plan, got keys={list(refined.keys())}"
        )
        assert refined[10] == "global", (
            "Expected tensor 10 to be reverted to 'global' after occupancy "
            f"degradation, but got refined[10]={refined[10]}"
        )

    def test_refinement_keeps_beneficial_promotions(self, mock_nvidia_hw_profile):
        """When measured occupancy is close to expected (ratio ≥ 0.85),
        the promotion is kept.

        Scenario: tensor 10 promoted to "shared", feedback shows
        consumer occupancy = 0.75, expected = 0.80 → ratio ≈ 0.94 ≥ 0.85 → kept.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("producer"),
            _make_metadata(
                tensor_shapes={10: (32, 32)},
                tensor_dtypes={10: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("consumer"),
            _make_metadata(smem_bytes=0),
        )
        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=10)

        node_a = graph.get_node(id_a)
        node_a.is_fused = True
        node_a.fused_from = [id_a, id_b]

        config = GraphConfig(feedback=FeedbackConfig(enable=True, sensitivity=0.15))
        planner = MemoryPlanner(graph, config)

        initial = planner.plan_promotions(mock_nvidia_hw_profile)
        assert initial[10] == "shared"

        # Good occupancy: no degradation detected.
        measured_metrics = {
            id_b: {
                "occupancy": 0.75,
                "expected_occupancy": 0.80,
                "register_spill_bytes": 0,
                "wall_clock_ms": 1.0,
            },
        }

        refined = planner.refine_with_feedback(measured_metrics)

        # refine_with_feedback returns a flat Dict[int, str].
        # Tensor 10 should remain "shared" because occupancy was acceptable.
        assert 10 in refined, (
            f"Expected tensor 10 in refined plan, got keys={list(refined.keys())}"
        )
        assert refined[10] == "shared", (
            "Expected tensor 10 to remain 'shared' (occupancy was acceptable), "
            f"but got refined[10]={refined[10]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — MemoryPlanner API Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestMemoryPlannerAPI:
    """Tests for the top-level ``MemoryPlanner`` API surface."""

    def test_memory_planner_plan(self, mock_nvidia_hw_profile):
        """End-to-end: create graph → identify intermediates → compute
        liveness → plan promotions → verify results are consistent.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(
            _mock_kernel("A"),
            _make_metadata(
                tensor_shapes={1: (64, 64)},
                tensor_dtypes={1: "fp32"},
                smem_bytes=0,
            ),
        )
        id_b = graph.add_node(
            _mock_kernel("B"),
            _make_metadata(
                tensor_shapes={2: (64, 64)},
                tensor_dtypes={2: "fp32"},
                smem_bytes=0,
            ),
        )
        id_c = graph.add_node(_mock_kernel("C"), _make_metadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=1)
        graph.add_edge(id_b, id_c, edge_type="data_dep", tensor_id=2)

        config = _default_config()
        planner = MemoryPlanner(graph, config)

        # Step 1: identify intermediates
        intermediates = planner.identify_intermediates()
        assert len(intermediates) == 2
        assert 1 in intermediates and 2 in intermediates

        # Step 2: compute liveness
        liveness = planner.compute_liveness()
        assert len(liveness) == 2
        for tid in intermediates:
            assert tid in liveness
            birth, death = liveness[tid]
            assert birth <= death, f"T{tid}: birth={birth} must be <= death={death}"

        # Step 3: plan promotions
        promotions = planner.plan_promotions(mock_nvidia_hw_profile)
        assert len(promotions) == 2
        for tid in intermediates:
            assert tid in promotions
            assert promotions[tid] in ("shared", "register", "global")

    def test_memory_planner_empty_graph(self):
        """An empty graph yields empty intermediates, liveness, and promotions."""
        graph = KGIRGraph()
        config = _default_config()
        planner = MemoryPlanner(graph, config)

        intermediates = planner.identify_intermediates()
        assert intermediates == [], f"Expected empty, got {intermediates}"

        liveness = planner.compute_liveness()
        assert liveness == {}, f"Expected empty, got {liveness}"

        # plan_promotions with any HW profile should return empty dict.
        hw = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=132,
            smem_per_sm_bytes=228 * 1024,
            registers_per_sm=65536,
            global_memory_bytes=80 * (1024 ** 3),
            memory_bandwidth_gbps=3350.0,
            compute_throughput_tflops=989.0,
            warp_size=32,
            max_concurrent_streams=128,
            interconnect_type="nvlink_4",
            interconnect_bandwidth_gbps=900.0,
        )
        promotions = planner.plan_promotions(hw)
        assert promotions == {}, f"Expected empty, got {promotions}"

    def test_memory_planner_no_intermediates(self):
        """Graph where a tensor has no consuming edge inside the graph.

        Single node with output tensor but no consumer → no intermediates →
        empty promotion plan.
        """
        graph = KGIRGraph()
        # Single node produces tensor 1, but no consumer edge exists.
        graph.add_node(
            _mock_kernel("lone_kernel"),
            _make_metadata(
                tensor_shapes={1: (128, 128)},
                tensor_dtypes={1: "fp32"},
            ),
        )

        config = _default_config()
        planner = MemoryPlanner(graph, config)

        intermediates = planner.identify_intermediates()
        assert intermediates == [], f"Expected no intermediates, got {intermediates}"

        hw = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=132,
            smem_per_sm_bytes=228 * 1024,
            registers_per_sm=65536,
            global_memory_bytes=80 * (1024 ** 3),
            memory_bandwidth_gbps=3350.0,
            compute_throughput_tflops=989.0,
            warp_size=32,
            max_concurrent_streams=128,
            interconnect_type="nvlink_4",
            interconnect_bandwidth_gbps=900.0,
        )
        promotions = planner.plan_promotions(hw)
        assert promotions == {}, f"Expected empty plan, got {promotions}"
