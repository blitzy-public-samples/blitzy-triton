"""Unit tests for KGIR (Kernel Graph Intermediate Representation) construction.

Tests cover:
- KGIRNode creation, metadata access, hardware/performance annotations, and
  target-agnostic construction
- KGIREdge creation for all 4 edge types (data_dep, anti_dep, resource_conflict,
  cross_device_transfer)
- KGIRGraph DAG operations: add_node, add_edge, get_node, topological_sort,
  validate, cycle detection, dependency analysis
- HardwareProfile 12-field construction, NVIDIA/AMD variants, JSON
  serialization round-trip
- Graph mutation: annotation write-back, fusion annotation via
  replace_nodes_with_fused

All tests are marked with ``@pytest.mark.kernel_graph`` as required by the
AAP testing rules (§0.7.5).
"""

from __future__ import annotations

import json
import dataclasses

import pytest
from unittest.mock import MagicMock, patch

from triton.graph.kgir import KGIRNode, KGIREdge, KGIRGraph, HardwareProfile
from triton.graph.kgir import NodeMetadata
from triton.graph.errors import GraphCaptureError


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: KGIRNode Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestKGIRNodeCreation:
    """Tests for KGIRNode instantiation and field accessibility."""

    def test_kgir_node_creation(self):
        """Create a KGIRNode with kernel function ref, grid, shapes, dtype.

        Verify all fields are accessible and correctly stored, and that the
        node has a unique integer identifier.
        """
        kernel_fn = MagicMock()
        kernel_fn.__name__ = "matmul_kernel"

        metadata = NodeMetadata(
            memory_access_patterns={"arg0": "read", "arg1": "read", "arg2": "write"},
            tensor_shapes={0: (1024, 1024), 1: (1024, 1024), 2: (1024, 1024)},
            tensor_strides={0: (1024, 1), 1: (1024, 1), 2: (1024, 1)},
            tensor_dtypes={0: "fp16", 1: "fp16", 2: "fp16"},
            grid_dimensions=(32, 32, 1),
            shared_memory_bytes=16384,
            register_count=64,
            num_warps=4,
        )

        node = KGIRNode(
            node_id=0,
            kernel_fn=kernel_fn,
            metadata=metadata,
        )

        assert node.node_id == 0
        assert isinstance(node.node_id, int)
        assert node.kernel_fn is kernel_fn
        assert node.metadata is metadata
        assert node.metadata.grid_dimensions == (32, 32, 1)
        assert node.metadata.tensor_shapes[0] == (1024, 1024)
        assert node.metadata.tensor_dtypes[0] == "fp16"
        assert node.is_fused is False
        assert node.fused_from is None

    def test_kgir_node_creation_minimal(self):
        """Create a KGIRNode with minimal metadata (all defaults)."""
        kernel_fn = MagicMock()
        metadata = NodeMetadata()

        node = KGIRNode(node_id=42, kernel_fn=kernel_fn, metadata=metadata)

        assert node.node_id == 42
        assert node.metadata.grid_dimensions == (1, 1, 1)
        assert node.metadata.shared_memory_bytes == 0
        assert node.metadata.register_count == 0
        assert node.metadata.num_warps == 4

    def test_kgir_node_unique_ids(self):
        """Verify two separate KGIRNode instances can have distinct IDs."""
        fn1, fn2 = MagicMock(), MagicMock()
        meta = NodeMetadata()
        n1 = KGIRNode(node_id=0, kernel_fn=fn1, metadata=meta)
        n2 = KGIRNode(node_id=1, kernel_fn=fn2, metadata=meta)

        assert n1.node_id != n2.node_id


@pytest.mark.kernel_graph
class TestKGIRNodeMetadata:
    """Tests for KGIRNode metadata reading and default values."""

    def test_kgir_node_metadata(self):
        """Set metadata fields and verify they can be read back via
        ``get_metadata()``.
        """
        metadata = NodeMetadata(
            memory_access_patterns={"inp": "read", "out": "write"},
            tensor_shapes={0: (2048,)},
            grid_dimensions=(256, 1, 1),
            shared_memory_bytes=8192,
            register_count=48,
        )
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=metadata)
        retrieved = node.get_metadata()

        assert retrieved.memory_access_patterns == {"inp": "read", "out": "write"}
        assert retrieved.tensor_shapes == {0: (2048,)}
        assert retrieved.grid_dimensions == (256, 1, 1)
        assert retrieved.shared_memory_bytes == 8192
        assert retrieved.register_count == 48

    def test_kgir_node_metadata_defaults(self):
        """Verify metadata defaults to empty dicts / zero for unset fields."""
        metadata = NodeMetadata()
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=metadata)
        m = node.get_metadata()

        assert m.memory_access_patterns == {}
        assert m.tensor_shapes == {}
        assert m.tensor_strides == {}
        assert m.tensor_dtypes == {}
        assert m.grid_dimensions == (1, 1, 1)
        assert m.shared_memory_bytes == 0
        assert m.register_count == 0
        assert m.num_warps == 4
        assert m.hardware_target_annotations == {}
        assert m.runtime_performance_annotations == {}


@pytest.mark.kernel_graph
class TestKGIRNodeHardwareAnnotations:
    """Tests for per-target hardware annotations on KGIRNode."""

    def test_kgir_node_hardware_annotations(self):
        """Attach hardware target annotations keyed by target identifier."""
        metadata = NodeMetadata()
        metadata.hardware_target_annotations["sm_90"] = {
            "preferred_num_warps": 8,
            "preferred_smem": 32768,
        }
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=metadata)

        ann = node.get_metadata().hardware_target_annotations
        assert "sm_90" in ann
        assert ann["sm_90"]["preferred_num_warps"] == 8

    def test_hardware_annotations_mutable(self):
        """Hardware annotations can be updated (feedback controller mutability)."""
        metadata = NodeMetadata()
        metadata.hardware_target_annotations["sm_90"] = {"optimized": False}
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=metadata)

        # Mutate
        node.metadata.hardware_target_annotations["sm_90"]["optimized"] = True
        assert node.metadata.hardware_target_annotations["sm_90"]["optimized"] is True

    def test_hardware_annotations_per_target(self):
        """Multiple targets can each have independent annotations."""
        metadata = NodeMetadata()
        metadata.hardware_target_annotations["sm_90"] = {"tiling": 128}
        metadata.hardware_target_annotations["gfx942"] = {"tiling": 256}
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=metadata)

        ann = node.metadata.hardware_target_annotations
        assert ann["sm_90"]["tiling"] == 128
        assert ann["gfx942"]["tiling"] == 256


@pytest.mark.kernel_graph
class TestKGIRNodePerformanceAnnotations:
    """Tests for runtime performance annotations (profiler write-back)."""

    def test_kgir_node_performance_annotations(self):
        """Attach runtime metrics and verify write-back works."""
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=NodeMetadata())

        node.update_performance_annotation("sm_90", {
            "wall_clock_ms": 0.42,
            "memory_throughput": 2800.0,
            "sm_occupancy": 0.85,
        })

        ann = node.get_performance_annotation("sm_90")
        assert ann is not None
        assert ann["wall_clock_ms"] == 0.42
        assert ann["memory_throughput"] == 2800.0
        assert ann["sm_occupancy"] == 0.85

    def test_performance_annotations_per_target(self):
        """Performance annotations are stored per-target."""
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=NodeMetadata())

        node.update_performance_annotation("sm_90", {"wall_clock_ms": 0.5})
        node.update_performance_annotation("gfx942", {"wall_clock_ms": 0.8})

        nvidia_ann = node.get_performance_annotation("sm_90")
        amd_ann = node.get_performance_annotation("gfx942")
        assert nvidia_ann["wall_clock_ms"] == 0.5
        assert amd_ann["wall_clock_ms"] == 0.8

    def test_performance_annotations_initially_empty(self):
        """Before any profiling, get_performance_annotation returns None."""
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=NodeMetadata())

        assert node.get_performance_annotation("sm_90") is None
        assert node.get_performance_annotation("nonexistent") is None

    def test_performance_annotation_overwrite(self):
        """Overwriting an existing annotation replaces the old values."""
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=NodeMetadata())

        node.update_performance_annotation("sm_90", {"wall_clock_ms": 1.0})
        node.update_performance_annotation("sm_90", {"wall_clock_ms": 0.5, "occupancy": 0.9})

        ann = node.get_performance_annotation("sm_90")
        assert ann["wall_clock_ms"] == 0.5
        assert ann["occupancy"] == 0.9
        # Old key should be gone since update replaces the dict
        assert "wall_clock_ms" in ann

    def test_resource_usage(self):
        """get_resource_usage returns compact resource summary."""
        metadata = NodeMetadata(
            shared_memory_bytes=4096,
            register_count=32,
            num_warps=8,
        )
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=metadata)
        usage = node.get_resource_usage()

        assert usage["shared_memory_bytes"] == 4096
        assert usage["register_count"] == 32
        assert usage["num_warps"] == 8


@pytest.mark.kernel_graph
class TestKGIRNodeTargetAgnostic:
    """Per AAP §0.1.1: KGIR nodes are target-agnostic at construction."""

    def test_kgir_node_target_agnostic_at_construction(self):
        """A freshly constructed node has no hardware-specific info baked in.

        Hardware-specific info must be encapsulated in HardwareProfile, not
        in the node itself.
        """
        metadata = NodeMetadata(
            grid_dimensions=(128, 1, 1),
            shared_memory_bytes=2048,
            register_count=32,
        )
        node = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=metadata)

        # No hardware target annotations at construction
        assert node.metadata.hardware_target_annotations == {}
        # No runtime performance annotations at construction
        assert node.metadata.runtime_performance_annotations == {}
        # Node itself does not store vendor/arch fields
        assert not hasattr(node, "vendor")
        assert not hasattr(node, "arch_generation")
        assert not hasattr(node, "warp_size")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: KGIREdge Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestKGIREdgeCreation:
    """Tests for KGIREdge creation across all 4 edge types."""

    def test_kgir_edge_data_dependency(self):
        """Create a data dependency edge (producer → consumer)."""
        edge = KGIREdge(
            source_id=0,
            target_id=1,
            edge_type="data_dep",
            tensor_id=0,
        )

        assert edge.edge_type == "data_dep"
        assert edge.source_id == 0
        assert edge.target_id == 1
        assert edge.tensor_id == 0

    def test_kgir_edge_anti_dependency(self):
        """Create an anti-dependency edge (write-after-read)."""
        edge = KGIREdge(
            source_id=0,
            target_id=1,
            edge_type="anti_dep",
        )

        assert edge.edge_type == "anti_dep"
        assert edge.source_id == 0
        assert edge.target_id == 1
        assert edge.tensor_id is None

    def test_kgir_edge_resource_conflict(self):
        """Create a resource conflict edge."""
        edge = KGIREdge(
            source_id=2,
            target_id=3,
            edge_type="resource_conflict",
        )

        assert edge.edge_type == "resource_conflict"
        assert edge.source_id == 2
        assert edge.target_id == 3

    def test_kgir_edge_cross_device_transfer(self):
        """Create a cross-device transfer edge with device metadata."""
        edge = KGIREdge(
            source_id=0,
            target_id=1,
            edge_type="cross_device_transfer",
            tensor_id=0,
            metadata={
                "source_device": "cuda:0",
                "target_device": "cuda:1",
                "transfer_bytes": 1024 * 1024,
            },
        )

        assert edge.edge_type == "cross_device_transfer"
        assert edge.source_id == 0
        assert edge.target_id == 1
        assert edge.tensor_id == 0
        assert edge.metadata["source_device"] == "cuda:0"
        assert edge.metadata["target_device"] == "cuda:1"
        assert edge.metadata["transfer_bytes"] == 1024 * 1024

    def test_kgir_edge_invalid_type_raises(self):
        """Creating an edge with an invalid type raises GraphCaptureError."""
        with pytest.raises(GraphCaptureError):
            KGIREdge(
                source_id=0,
                target_id=1,
                edge_type="invalid_edge_type",
            )

    @pytest.mark.parametrize("edge_type", [
        "data_dep",
        "anti_dep",
        "resource_conflict",
        "cross_device_transfer",
    ])
    def test_kgir_edge_all_valid_types(self, edge_type):
        """All four valid edge types can be created without error."""
        edge = KGIREdge(source_id=0, target_id=1, edge_type=edge_type)
        assert edge.edge_type == edge_type


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: KGIRGraph Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestKGIRGraphCreation:
    """Tests for KGIRGraph creation and node/edge operations."""

    def test_kgir_graph_creation_empty(self):
        """An empty KGIRGraph has 0 nodes, 0 edges, and passes validation."""
        graph = KGIRGraph()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert graph.validate() is True

    def test_kgir_graph_add_node(self):
        """Adding nodes increases node_count and nodes are retrievable."""
        graph = KGIRGraph()
        kernel_fn = MagicMock()
        metadata = NodeMetadata(grid_dimensions=(128,))

        nid = graph.add_node(kernel_fn, metadata)
        assert graph.node_count() == 1

        node = graph.get_node(nid)
        assert node.node_id == nid
        assert node.kernel_fn is kernel_fn
        assert node.metadata.grid_dimensions == (128,)

    def test_kgir_graph_add_multiple_nodes(self):
        """Multiple nodes get unique monotonically increasing IDs."""
        graph = KGIRGraph()
        ids = []
        for i in range(5):
            nid = graph.add_node(MagicMock(), NodeMetadata())
            ids.append(nid)

        assert graph.node_count() == 5
        assert len(set(ids)) == 5  # all unique
        # IDs should be monotonically increasing
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1]

    def test_kgir_graph_get_nonexistent_node_raises(self):
        """Accessing a non-existent node ID raises GraphCaptureError."""
        graph = KGIRGraph()
        with pytest.raises(GraphCaptureError):
            graph.get_node(999)

    def test_kgir_graph_add_edge(self):
        """Adding edges between existing nodes increases edge_count."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep", tensor_id=0)
        assert graph.edge_count() == 1

        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[0].source_id == id_a
        assert edges[0].target_id == id_b
        assert edges[0].edge_type == "data_dep"
        assert edges[0].tensor_id == 0

    def test_kgir_graph_add_edge_invalid_source(self):
        """Adding an edge with a non-existent source raises error."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())

        with pytest.raises(GraphCaptureError, match="does not exist"):
            graph.add_edge(999, id_a, edge_type="data_dep")

    def test_kgir_graph_add_edge_invalid_target(self):
        """Adding an edge with a non-existent target raises error."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())

        with pytest.raises(GraphCaptureError, match="does not exist"):
            graph.add_edge(id_a, 999, edge_type="data_dep")

    def test_kgir_graph_self_loop_raises(self):
        """Adding a self-loop raises GraphCaptureError."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())

        with pytest.raises(GraphCaptureError, match="Self-loop"):
            graph.add_edge(id_a, id_a, edge_type="data_dep")


@pytest.mark.kernel_graph
class TestKGIRGraphTopologicalSort:
    """Tests for topological ordering of the KGIR DAG."""

    def test_kgir_graph_topological_sort(self):
        """Create DAG A → B → C, A → C and verify topological ordering."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        id_c = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep")
        graph.add_edge(id_b, id_c, edge_type="data_dep")
        graph.add_edge(id_a, id_c, edge_type="data_dep")

        order = graph.topological_sort()

        # A must come before B, B before C, A before C
        idx = {nid: i for i, nid in enumerate(order)}
        assert idx[id_a] < idx[id_b]
        assert idx[id_b] < idx[id_c]
        assert idx[id_a] < idx[id_c]
        assert len(order) == 3

    def test_topological_sort_single_node(self):
        """Topological sort of a single-node graph returns that node."""
        graph = KGIRGraph()
        nid = graph.add_node(MagicMock(), NodeMetadata())
        order = graph.topological_sort()
        assert order == [nid]

    def test_topological_sort_empty_graph(self):
        """Topological sort of an empty graph returns empty list."""
        graph = KGIRGraph()
        order = graph.topological_sort()
        assert order == []

    def test_topological_sort_diamond(self):
        """Diamond DAG: A→B, A→C, B→D, C→D."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        id_c = graph.add_node(MagicMock(), NodeMetadata())
        id_d = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep")
        graph.add_edge(id_a, id_c, edge_type="data_dep")
        graph.add_edge(id_b, id_d, edge_type="data_dep")
        graph.add_edge(id_c, id_d, edge_type="data_dep")

        order = graph.topological_sort()
        idx = {nid: i for i, nid in enumerate(order)}

        assert idx[id_a] < idx[id_b]
        assert idx[id_a] < idx[id_c]
        assert idx[id_b] < idx[id_d]
        assert idx[id_c] < idx[id_d]

    def test_topological_sort_independent_nodes(self):
        """Independent (unconnected) nodes all appear in topological order."""
        graph = KGIRGraph()
        ids = [graph.add_node(MagicMock(), NodeMetadata()) for _ in range(4)]

        order = graph.topological_sort()
        assert set(order) == set(ids)
        assert len(order) == 4


@pytest.mark.kernel_graph
class TestKGIRGraphCycleDetection:
    """Tests for cycle detection in the KGIR DAG."""

    def test_kgir_graph_cycle_detection(self):
        """Adding an edge that creates A → B → C → A raises on add_edge."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        id_c = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep")
        graph.add_edge(id_b, id_c, edge_type="data_dep")

        # Closing the cycle C → A should raise
        with pytest.raises(GraphCaptureError, match="cycle"):
            graph.add_edge(id_c, id_a, edge_type="data_dep")

    def test_kgir_graph_validate_acyclicity_valid(self):
        """validate() passes on a valid DAG (no cycles)."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        graph.add_edge(id_a, id_b, edge_type="data_dep")

        assert graph.validate() is True

    def test_kgir_graph_validate_acyclicity_invalid(self):
        """validate() raises on a graph with a cycle injected via internals.

        We bypass the ``add_edge`` cycle check by manipulating internal
        structures to simulate a corrupt graph state.
        """
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        graph.add_edge(id_a, id_b, edge_type="data_dep")

        # Inject a back-edge B → A directly into internal structures
        cycle_edge = KGIREdge(
            source_id=id_b,
            target_id=id_a,
            edge_type="data_dep",
        )
        graph._edges.append(cycle_edge)
        graph._adjacency[id_b].append(id_a)
        graph._reverse_adjacency[id_a].append(id_b)

        with pytest.raises(GraphCaptureError, match="cycle"):
            graph.validate()

    def test_two_node_cycle_rejected(self):
        """A two-node cycle A → B, B → A is rejected on the second add_edge."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep")
        with pytest.raises(GraphCaptureError, match="cycle"):
            graph.add_edge(id_b, id_a, edge_type="data_dep")


@pytest.mark.kernel_graph
class TestKGIRGraphDependencyAnalysis:
    """Tests for dependency analysis and graph structure queries."""

    def test_kgir_graph_dependency_analysis(self, sample_kgir_graph):
        """Verify the sample A→B→C graph has correct dependency structure."""
        graph = sample_kgir_graph

        # Node count
        assert graph.node_count() == 3
        assert graph.edge_count() == 2

        # Successors
        assert 1 in graph.get_successors(0)
        assert 2 in graph.get_successors(1)
        assert graph.get_successors(2) == []

        # Predecessors
        assert graph.get_predecessors(0) == []
        assert 0 in graph.get_predecessors(1)
        assert 1 in graph.get_predecessors(2)

    def test_graph_roots_and_leaves(self):
        """Verify roots (no predecessors) and leaves (no successors)."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        id_c = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep")
        graph.add_edge(id_a, id_c, edge_type="data_dep")

        roots = graph.get_roots()
        leaves = graph.get_leaves()

        assert id_a in roots
        assert id_b in leaves
        assert id_c in leaves
        assert id_a not in leaves

    def test_graph_edges_by_node(self):
        """get_edges(node_id) returns only edges involving that node."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        id_c = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep")
        graph.add_edge(id_b, id_c, edge_type="data_dep")

        edges_b = graph.get_edges(id_b)
        # Node B is source of A→B and target of B→C — both should appear
        assert len(edges_b) == 2

        edges_a = graph.get_edges(id_a)
        assert len(edges_a) == 1
        assert edges_a[0].source_id == id_a


@pytest.mark.kernel_graph
class TestKGIRGraphValidateResourceConstraints:
    """Tests for validate() resource constraint checking."""

    def test_kgir_graph_validate_resource_constraints_valid(self):
        """A graph with valid resources passes validation."""
        graph = KGIRGraph()
        graph.add_node(MagicMock(), NodeMetadata(
            shared_memory_bytes=8192,
            register_count=32,
        ))
        graph.add_node(MagicMock(), NodeMetadata(
            shared_memory_bytes=4096,
            register_count=24,
        ))

        # validate() should pass since there's no hardware profile to compare
        assert graph.validate() is True

    def test_kgir_graph_validate_passes_on_linear_chain(self, sample_kgir_graph):
        """The sample A→B→C linear chain passes validate()."""
        assert sample_kgir_graph.validate() is True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: HardwareProfile Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestHardwareProfileCreation:
    """Tests for HardwareProfile construction with all 12 fields."""

    def test_hardware_profile_creation(self):
        """Create a HardwareProfile with all 12 mandatory fields per AAP §0.5.1."""
        profile = HardwareProfile(
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

        assert profile.vendor == "nvidia"
        assert profile.arch_generation == "sm_90"
        assert profile.sm_count == 132
        assert profile.smem_per_sm_bytes == 228 * 1024
        assert profile.registers_per_sm == 65536
        assert profile.global_memory_bytes == 80 * (1024 ** 3)
        assert profile.memory_bandwidth_gbps == 3350.0
        assert profile.compute_throughput_tflops == 989.0
        assert profile.warp_size == 32
        assert profile.max_concurrent_streams == 128
        assert profile.interconnect_type == "nvlink_4"
        assert profile.interconnect_bandwidth_gbps == 900.0

    def test_hardware_profile_nvidia(self, mock_nvidia_hw_profile):
        """Verify NVIDIA-specific values from the conftest fixture."""
        hp = mock_nvidia_hw_profile
        assert hp.vendor == "nvidia"
        assert hp.arch_generation == "sm_90"
        assert hp.warp_size == 32
        assert hp.sm_count == 132
        assert hp.smem_per_sm_bytes == 228 * 1024
        assert hp.registers_per_sm == 65536
        assert hp.interconnect_type == "nvlink_4"

    def test_hardware_profile_amd(self, mock_amd_hw_profile):
        """Verify AMD-specific values from the conftest fixture."""
        hp = mock_amd_hw_profile
        assert hp.vendor == "amd"
        assert hp.arch_generation == "gfx942"
        assert hp.warp_size == 64
        assert hp.sm_count == 304
        assert hp.smem_per_sm_bytes == 64 * 1024
        assert hp.interconnect_type == "infinity_fabric"

    def test_hardware_profile_serialization(self, mock_nvidia_hw_profile):
        """Round-trip: create → serialize to JSON → deserialize → compare.

        Uses dataclasses.asdict for conversion and json.dumps/loads for
        serialization. Verifies all 12 mandatory fields survive the round-trip.
        """
        hp = mock_nvidia_hw_profile

        # Serialize to dict then JSON
        hp_dict = dataclasses.asdict(hp)
        # Remove optional gpu_target field for JSON (it's None by default)
        hp_dict.pop("gpu_target", None)
        json_str = json.dumps(hp_dict)

        # Deserialize from JSON
        restored_dict = json.loads(json_str)
        restored = HardwareProfile.from_dict(restored_dict)

        assert restored.vendor == hp.vendor
        assert restored.arch_generation == hp.arch_generation
        assert restored.sm_count == hp.sm_count
        assert restored.smem_per_sm_bytes == hp.smem_per_sm_bytes
        assert restored.registers_per_sm == hp.registers_per_sm
        assert restored.global_memory_bytes == hp.global_memory_bytes
        assert restored.memory_bandwidth_gbps == hp.memory_bandwidth_gbps
        assert restored.compute_throughput_tflops == hp.compute_throughput_tflops
        assert restored.warp_size == hp.warp_size
        assert restored.max_concurrent_streams == hp.max_concurrent_streams
        assert restored.interconnect_type == hp.interconnect_type
        assert restored.interconnect_bandwidth_gbps == hp.interconnect_bandwidth_gbps

    def test_hardware_profile_to_dict(self):
        """to_dict() returns all 12 mandatory fields."""
        hp = HardwareProfile(
            vendor="amd",
            arch_generation="gfx942",
            sm_count=304,
            smem_per_sm_bytes=65536,
            registers_per_sm=65536,
            global_memory_bytes=192 * (1024 ** 3),
            memory_bandwidth_gbps=5300.0,
            compute_throughput_tflops=1307.0,
            warp_size=64,
            max_concurrent_streams=128,
            interconnect_type="infinity_fabric",
            interconnect_bandwidth_gbps=896.0,
        )

        d = hp.to_dict()
        assert len(d) == 12
        assert d["vendor"] == "amd"
        assert d["warp_size"] == 64
        assert "gpu_target" not in d  # to_dict excludes optional field

    def test_hardware_profile_from_dict(self):
        """from_dict() reconstructs a HardwareProfile from a plain dict."""
        data = {
            "vendor": "nvidia",
            "arch_generation": "sm_80",
            "sm_count": 108,
            "smem_per_sm_bytes": 163840,
            "registers_per_sm": 65536,
            "global_memory_bytes": 40 * (1024 ** 3),
            "memory_bandwidth_gbps": 1555.0,
            "compute_throughput_tflops": 312.0,
            "warp_size": 32,
            "max_concurrent_streams": 128,
            "interconnect_type": "nvlink_3",
            "interconnect_bandwidth_gbps": 600.0,
        }
        hp = HardwareProfile.from_dict(data)
        assert hp.vendor == "nvidia"
        assert hp.arch_generation == "sm_80"
        assert hp.sm_count == 108

    def test_hardware_profile_convenience_methods(self):
        """Test total_smem_bytes and total_registers convenience helpers."""
        hp = HardwareProfile(
            vendor="nvidia",
            arch_generation="sm_90",
            sm_count=10,
            smem_per_sm_bytes=1024,
            registers_per_sm=256,
            global_memory_bytes=1024,
            memory_bandwidth_gbps=100.0,
            compute_throughput_tflops=10.0,
            warp_size=32,
            max_concurrent_streams=16,
            interconnect_type="pcie_4",
            interconnect_bandwidth_gbps=16.0,
        )

        assert hp.total_smem_bytes() == 10 * 1024
        assert hp.total_registers() == 10 * 256


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Graph Mutation Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestKGIRGraphAnnotationWriteback:
    """Per AAP §0.1.1: KGIR nodes are mutable for runtime annotation write-back."""

    def test_kgir_graph_annotation_writeback(self):
        """Write performance annotations to nodes in a graph and read back."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())

        # Simulate feedback controller writing measurements
        node_a = graph.get_node(id_a)
        node_a.update_performance_annotation("sm_90", {
            "wall_clock_ms": 0.32,
            "memory_throughput": 3000.0,
        })

        node_b = graph.get_node(id_b)
        node_b.update_performance_annotation("sm_90", {
            "wall_clock_ms": 0.18,
            "memory_throughput": 2500.0,
        })

        # Read back from graph
        ann_a = graph.get_node(id_a).get_performance_annotation("sm_90")
        ann_b = graph.get_node(id_b).get_performance_annotation("sm_90")

        assert ann_a["wall_clock_ms"] == 0.32
        assert ann_b["wall_clock_ms"] == 0.18

    def test_annotation_writeback_multiple_iterations(self):
        """Simulate multiple feedback iterations updating the same node."""
        graph = KGIRGraph()
        nid = graph.add_node(MagicMock(), NodeMetadata())
        node = graph.get_node(nid)

        # Iteration 1
        node.update_performance_annotation("sm_90", {"wall_clock_ms": 1.0})
        assert node.get_performance_annotation("sm_90")["wall_clock_ms"] == 1.0

        # Iteration 2 — should overwrite
        node.update_performance_annotation("sm_90", {"wall_clock_ms": 0.8})
        assert node.get_performance_annotation("sm_90")["wall_clock_ms"] == 0.8

        # Iteration 3
        node.update_performance_annotation("sm_90", {"wall_clock_ms": 0.75})
        assert node.get_performance_annotation("sm_90")["wall_clock_ms"] == 0.75


@pytest.mark.kernel_graph
class TestKGIRGraphFusionAnnotation:
    """Tests for fusion annotation via replace_nodes_with_fused."""

    def test_kgir_graph_fusion_annotation(self, sample_kgir_graph):
        """Mark two nodes as fused and verify the graph reflects it.

        We fuse nodes 0 (A) and 1 (B) from the sample A→B→C graph
        into a new fused node, then check that the fused node has correct
        metadata and the graph structure is updated.
        """
        graph = sample_kgir_graph

        # Original state
        assert graph.node_count() == 3
        assert graph.edge_count() == 2

        # Create the fused node
        node_a = graph.get_node(0)
        node_b = graph.get_node(1)
        fused_metadata = NodeMetadata(
            grid_dimensions=node_a.metadata.grid_dimensions,
            shared_memory_bytes=(
                node_a.metadata.shared_memory_bytes
                + node_b.metadata.shared_memory_bytes
            ),
            register_count=max(
                node_a.metadata.register_count,
                node_b.metadata.register_count,
            ),
        )
        fused_node = KGIRNode(
            node_id=100,
            kernel_fn=MagicMock(),
            metadata=fused_metadata,
            is_fused=True,
            fused_from=[0, 1],
        )

        graph.replace_nodes_with_fused([0, 1], fused_node)

        # After fusion: 2 nodes remain (fused + C)
        assert graph.node_count() == 2
        # One edge: fused → C
        assert graph.edge_count() == 1

        # The fused node should be in the graph
        fn = graph.get_node(100)
        assert fn.is_fused is True
        assert fn.fused_from == [0, 1]
        assert fn.metadata.shared_memory_bytes == (
            node_a.metadata.shared_memory_bytes
            + node_b.metadata.shared_memory_bytes
        )

        # Node C (id=2) should still be accessible
        node_c = graph.get_node(2)
        assert node_c is not None

        # The remaining edge should be from fused node to C
        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[0].source_id == 100
        assert edges[0].target_id == 2

    def test_fusion_empty_node_ids_raises(self):
        """Fusing with an empty node_ids list raises GraphCaptureError."""
        graph = KGIRGraph()
        fused = KGIRNode(node_id=0, kernel_fn=MagicMock(), metadata=NodeMetadata(), is_fused=True)
        with pytest.raises(GraphCaptureError, match="non-empty"):
            graph.replace_nodes_with_fused([], fused)

    def test_fusion_nonexistent_node_raises(self):
        """Fusing with non-existent node IDs raises GraphCaptureError."""
        graph = KGIRGraph()
        graph.add_node(MagicMock(), NodeMetadata())
        fused = KGIRNode(
            node_id=10, kernel_fn=MagicMock(), metadata=NodeMetadata(),
            is_fused=True, fused_from=[0, 99],
        )
        with pytest.raises(GraphCaptureError, match="does not exist"):
            graph.replace_nodes_with_fused([0, 99], fused)

    def test_fusion_preserves_dag_validity(self):
        """After fusion the graph still passes validate()."""
        graph = KGIRGraph()
        id_a = graph.add_node(MagicMock(), NodeMetadata())
        id_b = graph.add_node(MagicMock(), NodeMetadata())
        id_c = graph.add_node(MagicMock(), NodeMetadata())

        graph.add_edge(id_a, id_b, edge_type="data_dep")
        graph.add_edge(id_b, id_c, edge_type="data_dep")

        fused_node = KGIRNode(
            node_id=50,
            kernel_fn=MagicMock(),
            metadata=NodeMetadata(),
            is_fused=True,
            fused_from=[id_a, id_b],
        )
        graph.replace_nodes_with_fused([id_a, id_b], fused_node)

        assert graph.validate() is True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Using Fixtures from conftest.py
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestConftestFixtureIntegration:
    """Tests that verify conftest fixtures work correctly with KGIR classes."""

    def test_mock_gpu_target_fixture(self, mock_gpu_target):
        """The mock_gpu_target fixture provides NVIDIA sm_90 GPUTarget."""
        assert mock_gpu_target.backend == "cuda"
        assert mock_gpu_target.arch == 90
        assert mock_gpu_target.warp_size == 32

    def test_make_kgir_node_fixture(self, make_kgir_node):
        """The make_kgir_node factory creates KGIRNode instances."""
        node = make_kgir_node(
            kernel_name="softmax",
            grid=(256, 1),
            smem_bytes=4096,
            register_count=48,
        )

        assert node.kernel_fn.__name__ == "softmax"
        assert node.metadata.grid_dimensions == (256, 1)
        assert node.metadata.shared_memory_bytes == 4096
        assert node.metadata.register_count == 48

    def test_sample_kgir_graph_fixture(self, sample_kgir_graph):
        """The sample_kgir_graph fixture provides a valid A→B→C chain."""
        graph = sample_kgir_graph

        assert graph.node_count() == 3
        assert graph.edge_count() == 2

        # Topological order should respect A→B→C
        order = graph.topological_sort()
        idx = {nid: i for i, nid in enumerate(order)}
        assert idx[0] < idx[1]
        assert idx[1] < idx[2]

    def test_hardware_profiles_attached_to_graph(self, mock_nvidia_hw_profile):
        """HardwareProfile can be attached to a KGIRGraph."""
        graph = KGIRGraph(hardware_profiles=[mock_nvidia_hw_profile])
        profiles = graph.hardware_profiles

        assert len(profiles) == 1
        assert profiles[0].vendor == "nvidia"

    def test_multiple_hardware_profiles(
        self, mock_nvidia_hw_profile, mock_amd_hw_profile
    ):
        """Multiple HardwareProfiles (NVIDIA + AMD) can be attached."""
        graph = KGIRGraph(
            hardware_profiles=[mock_nvidia_hw_profile, mock_amd_hw_profile]
        )

        profiles = graph.hardware_profiles
        assert len(profiles) == 2
        vendors = {p.vendor for p in profiles}
        assert "nvidia" in vendors
        assert "amd" in vendors


# ═══════════════════════════════════════════════════════════════════════════════
# Additional Edge Cases and Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.kernel_graph
class TestKGIRGraphRepr:
    """Tests for __repr__ methods on KGIR classes."""

    def test_node_repr(self):
        """KGIRNode __repr__ includes id and grid."""
        node = KGIRNode(
            node_id=7,
            kernel_fn=MagicMock(),
            metadata=NodeMetadata(grid_dimensions=(64, 32)),
        )
        r = repr(node)
        assert "7" in r
        assert "64" in r

    def test_fused_node_repr(self):
        """Fused node __repr__ includes FUSED tag."""
        node = KGIRNode(
            node_id=10,
            kernel_fn=MagicMock(),
            metadata=NodeMetadata(),
            is_fused=True,
            fused_from=[1, 2],
        )
        r = repr(node)
        assert "FUSED" in r

    def test_graph_repr(self):
        """KGIRGraph __repr__ shows node and edge counts."""
        graph = KGIRGraph()
        graph.add_node(MagicMock(), NodeMetadata())
        graph.add_node(MagicMock(), NodeMetadata())
        graph.add_edge(0, 1, edge_type="data_dep")

        r = repr(graph)
        assert "nodes=2" in r
        assert "edges=1" in r


@pytest.mark.kernel_graph
class TestKGIRGraphMlirRoundTrip:
    """Tests for MLIR text serialization/deserialization round-trip."""

    def test_mlir_round_trip_simple(self):
        """A simple graph can be serialized to MLIR text and parsed back."""
        graph = KGIRGraph()
        graph.add_node(MagicMock(), NodeMetadata(
            grid_dimensions=(128, 1, 1),
            shared_memory_bytes=2048,
            register_count=32,
            num_warps=4,
        ))
        graph.add_node(MagicMock(), NodeMetadata(
            grid_dimensions=(128, 1, 1),
            shared_memory_bytes=1024,
            register_count=16,
            num_warps=4,
        ))
        graph.add_edge(0, 1, edge_type="data_dep", tensor_id=0)

        mlir_text = graph.to_mlir()
        assert "ttkgir" in mlir_text
        assert "kernel_launch" in mlir_text

        # Parse it back
        restored = KGIRGraph.from_mlir(mlir_text)
        assert restored.node_count() == 2
        assert restored.edge_count() == 1

    def test_mlir_round_trip_empty_graph(self):
        """An empty graph can be serialized and parsed back."""
        graph = KGIRGraph()
        mlir_text = graph.to_mlir()
        restored = KGIRGraph.from_mlir(mlir_text)
        assert restored.node_count() == 0
        assert restored.edge_count() == 0
