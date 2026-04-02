"""Unit tests for DAG utilities in triton.graph.utils.

Tests cover topological sort correctness, cycle detection, critical path
computation, shapes_compatible helper, memory_regions_overlap helper, and
dtype_to_bytes helper.  Every test is marked ``@pytest.mark.kernel_graph``
so it can be selected (or skipped) via the graph-level marker.
"""
from __future__ import annotations

import pytest

from triton.graph.utils import (
    topological_sort,
    detect_cycle,
    compute_critical_path,
    shapes_compatible,
    memory_regions_overlap,
    dtype_to_bytes,
)

# ============================================================
# Phase 1 — Topological Sort Tests
# ============================================================


@pytest.mark.kernel_graph
def test_topological_sort_linear():
    """Linear DAG: 0 → 1 → 2.  Only one valid ordering: [0, 1, 2]."""
    adjacency = {0: [1], 1: [2], 2: []}
    result = topological_sort(adjacency)
    assert result == [0, 1, 2]


@pytest.mark.kernel_graph
def test_topological_sort_diamond():
    """Diamond DAG: 0 → 1, 0 → 2, 1 → 3, 2 → 3.

    Valid orderings include [0, 1, 2, 3] and [0, 2, 1, 3].
    The only hard constraints are:
      * 0 before 1, 0 before 2, 1 before 3, 2 before 3.
    """
    adjacency = {0: [1, 2], 1: [3], 2: [3], 3: []}
    result = topological_sort(adjacency)

    assert len(result) == 4
    assert set(result) == {0, 1, 2, 3}
    # Verify ordering constraints
    assert result.index(0) < result.index(1)
    assert result.index(0) < result.index(2)
    assert result.index(1) < result.index(3)
    assert result.index(2) < result.index(3)


@pytest.mark.kernel_graph
def test_topological_sort_multiple_sources():
    """DAG with two source nodes: 0 → 2, 1 → 2, 0 → 3.

    Both 0 and 1 are sources (zero in-degree).  They must appear before
    their dependents.
    """
    adjacency = {0: [2, 3], 1: [2], 2: [], 3: []}
    result = topological_sort(adjacency)

    assert len(result) == 4
    assert set(result) == {0, 1, 2, 3}
    assert result.index(0) < result.index(2)
    assert result.index(0) < result.index(3)
    assert result.index(1) < result.index(2)


@pytest.mark.kernel_graph
def test_topological_sort_single_node():
    """Single-node graph with no edges."""
    adjacency = {0: []}
    result = topological_sort(adjacency)
    assert result == [0]


@pytest.mark.kernel_graph
def test_topological_sort_empty():
    """Empty graph — no nodes, no edges.  Sort must return []."""
    adjacency: dict[int, list[int]] = {}
    result = topological_sort(adjacency)
    assert result == []


@pytest.mark.kernel_graph
def test_topological_sort_disconnected():
    """Two disconnected components: 0 → 1 and 2 → 3.

    All four nodes must appear with valid per-component ordering.
    """
    adjacency = {0: [1], 1: [], 2: [3], 3: []}
    result = topological_sort(adjacency)

    assert len(result) == 4
    assert set(result) == {0, 1, 2, 3}
    assert result.index(0) < result.index(1)
    assert result.index(2) < result.index(3)


@pytest.mark.kernel_graph
def test_topological_sort_cycle_raises():
    """Topological sort on a graph with a cycle must raise ValueError."""
    adjacency = {0: [1], 1: [2], 2: [0]}
    with pytest.raises(ValueError):
        topological_sort(adjacency)


# ============================================================
# Phase 2 — Cycle Detection Tests
# ============================================================


@pytest.mark.kernel_graph
def test_cycle_detection_no_cycle():
    """Acyclic DAG: 0 → 1 → 2.  detect_cycle returns None."""
    adjacency = {0: [1], 1: [2], 2: []}
    result = detect_cycle(adjacency)
    assert result is None


@pytest.mark.kernel_graph
def test_cycle_detection_simple_cycle():
    """Simple 3-node cycle: 0 → 1 → 2 → 0.

    detect_cycle must return a non-None list whose first and last
    elements are identical (the cycle closure).
    """
    adjacency = {0: [1], 1: [2], 2: [0]}
    result = detect_cycle(adjacency)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) >= 2
    assert result[0] == result[-1], "Cycle path must close: first == last"
    # Every node in the cycle path must belong to the graph
    for node in result:
        assert node in adjacency


@pytest.mark.kernel_graph
def test_cycle_detection_self_loop():
    """Self-loop: 0 → 0.  The simplest possible cycle."""
    adjacency = {0: [0]}
    result = detect_cycle(adjacency)

    assert result is not None
    assert isinstance(result, list)
    assert result[0] == result[-1]


@pytest.mark.kernel_graph
def test_cycle_detection_complex():
    """Graph with a cycle nested deeper: 0 → 1 → 2 → 3 → 1.

    Node 0 is NOT on the cycle; the cycle is 1 → 2 → 3 → 1.
    """
    adjacency = {0: [1], 1: [2], 2: [3], 3: [1]}
    result = detect_cycle(adjacency)

    assert result is not None
    assert isinstance(result, list)
    assert result[0] == result[-1]


@pytest.mark.kernel_graph
def test_cycle_detection_empty():
    """Empty graph — no cycle possible."""
    adjacency: dict[int, list[int]] = {}
    result = detect_cycle(adjacency)
    assert result is None


# ============================================================
# Phase 3 — Critical Path Tests
# ============================================================


@pytest.mark.kernel_graph
def test_critical_path_linear():
    """Linear DAG: 0(10) → 1(20) → 2(5).

    Critical path: [0, 1, 2] with total weight 35.
    """
    adjacency = {0: [1], 1: [2], 2: []}
    weights = {0: 10.0, 1: 20.0, 2: 5.0}
    path, total = compute_critical_path(adjacency, weights)

    assert path == [0, 1, 2]
    assert total == pytest.approx(35.0)


@pytest.mark.kernel_graph
def test_critical_path_diamond():
    """Diamond DAG with two paths through it:

        0(10) → 1(5) → 3(10)    total = 25
        0(10) → 2(20) → 3(10)   total = 40  ← critical

    Critical path must traverse the heavier branch through node 2.
    """
    adjacency = {0: [1, 2], 1: [3], 2: [3], 3: []}
    weights = {0: 10.0, 1: 5.0, 2: 20.0, 3: 10.0}
    path, total = compute_critical_path(adjacency, weights)

    assert total == pytest.approx(40.0)
    # The critical path goes through node 2, not node 1
    assert 0 in path
    assert 2 in path
    assert 3 in path
    # Verify ordering within path
    assert path.index(0) < path.index(2)
    assert path.index(2) < path.index(3)


@pytest.mark.kernel_graph
def test_critical_path_parallel():
    """Two independent chains: 0(10) → 1(10) and 2(30).

    Chain 0→1 = 20, standalone 2 = 30.  Critical path: [2], total = 30.
    """
    adjacency = {0: [1], 1: [], 2: []}
    weights = {0: 10.0, 1: 10.0, 2: 30.0}
    path, total = compute_critical_path(adjacency, weights)

    assert total == pytest.approx(30.0)
    assert 2 in path


@pytest.mark.kernel_graph
def test_critical_path_single_node():
    """Single node with weight 15."""
    adjacency = {0: []}
    weights = {0: 15.0}
    path, total = compute_critical_path(adjacency, weights)

    assert path == [0]
    assert total == pytest.approx(15.0)


@pytest.mark.kernel_graph
def test_critical_path_equal_weights():
    """All nodes equal weight — critical path is the longest chain.

        0(1) → 1(1) → 2(1)
        3(1) → 4(1)

    Chain 0→1→2 = 3, chain 3→4 = 2.  Critical path through 0→1→2.
    """
    adjacency = {0: [1], 1: [2], 2: [], 3: [4], 4: []}
    weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    path, total = compute_critical_path(adjacency, weights)

    assert total == pytest.approx(3.0)
    assert path == [0, 1, 2]


@pytest.mark.kernel_graph
def test_critical_path_empty():
    """Empty graph produces empty path with zero total weight."""
    adjacency: dict[int, list[int]] = {}
    weights: dict[int, float] = {}
    path, total = compute_critical_path(adjacency, weights)

    assert path == []
    assert total == pytest.approx(0.0)


# ============================================================
# Phase 4 — Shape Compatibility Tests
# ============================================================


@pytest.mark.kernel_graph
def test_shapes_compatible_identical():
    """Identical shapes are always compatible."""
    assert shapes_compatible((128, 256), (128, 256)) is True


@pytest.mark.kernel_graph
def test_shapes_compatible_different():
    """Shapes with no broadcastable dimension pair are incompatible."""
    assert shapes_compatible((128, 256), (64, 512)) is False


@pytest.mark.kernel_graph
def test_shapes_compatible_broadcastable():
    """Shapes broadcastable via NumPy rules (one dimension is 1)."""
    assert shapes_compatible((128, 1), (128, 256)) is True
    assert shapes_compatible((1, 256), (128, 256)) is True
    assert shapes_compatible((1, 1), (128, 256)) is True


@pytest.mark.kernel_graph
def test_shapes_compatible_different_rank():
    """Different-rank shapes follow left-pad-with-1 broadcasting.

    (128, 256) padded to (1, 128, 256) vs (128, 256, 1):
      dim -1: 256 vs 1 → OK (broadcast)
      dim -2: 128 vs 256 → FAIL (neither 1 nor equal)
    """
    assert shapes_compatible((128, 256), (128, 256, 1)) is False

    # But (256,) vs (128, 256) → padded to (1, 256) vs (128, 256) → OK
    assert shapes_compatible((256,), (128, 256)) is True


@pytest.mark.kernel_graph
def test_shapes_compatible_empty():
    """Two scalar shapes (empty tuples) are always compatible."""
    assert shapes_compatible((), ()) is True


@pytest.mark.kernel_graph
def test_shapes_compatible_scalar_broadcast():
    """A scalar shape broadcasts with any shape."""
    assert shapes_compatible((), (128, 256)) is True
    assert shapes_compatible((128, 256), ()) is True


# ============================================================
# Phase 5 — Memory Regions Overlap Tests
# ============================================================


@pytest.mark.kernel_graph
def test_memory_overlap_same_pointer():
    """Same data_ptr and same size — regions are identical, overlap."""
    assert memory_regions_overlap(1000, 512, 1000, 512) is True


@pytest.mark.kernel_graph
def test_memory_overlap_disjoint():
    """Non-overlapping regions: [1000, 1512) and [2000, 2512)."""
    assert memory_regions_overlap(1000, 512, 2000, 512) is False


@pytest.mark.kernel_graph
def test_memory_overlap_partial():
    """Partially overlapping regions: [1000, 1512) and [1256, 1768)."""
    assert memory_regions_overlap(1000, 512, 1256, 512) is True


@pytest.mark.kernel_graph
def test_memory_overlap_contained():
    """One region fully contained: [1000, 2024) contains [1200, 1456)."""
    assert memory_regions_overlap(1000, 1024, 1200, 256) is True
    # Reverse containment direction
    assert memory_regions_overlap(1200, 256, 1000, 1024) is True


@pytest.mark.kernel_graph
def test_memory_overlap_zero_size():
    """Zero-size regions never overlap — even at the same pointer."""
    assert memory_regions_overlap(1000, 0, 1000, 512) is False
    assert memory_regions_overlap(1000, 512, 1000, 0) is False
    assert memory_regions_overlap(1000, 0, 1000, 0) is False


@pytest.mark.kernel_graph
def test_memory_overlap_adjacent():
    """Adjacent but non-overlapping regions (half-open intervals).

    [1000, 1512) and [1512, 2024) share no elements.
    """
    assert memory_regions_overlap(1000, 512, 1512, 512) is False


# ============================================================
# Phase 6 — dtype_to_bytes Tests
# ============================================================


@pytest.mark.kernel_graph
def test_dtype_to_bytes_float32():
    assert dtype_to_bytes("float32") == 4


@pytest.mark.kernel_graph
def test_dtype_to_bytes_float16():
    assert dtype_to_bytes("float16") == 2


@pytest.mark.kernel_graph
def test_dtype_to_bytes_int8():
    assert dtype_to_bytes("int8") == 1


@pytest.mark.kernel_graph
def test_dtype_to_bytes_float64():
    assert dtype_to_bytes("float64") == 8


@pytest.mark.kernel_graph
def test_dtype_to_bytes_bfloat16():
    """bfloat16 is 2 bytes, same as float16."""
    assert dtype_to_bytes("bfloat16") == 2


@pytest.mark.kernel_graph
def test_dtype_to_bytes_int32():
    assert dtype_to_bytes("int32") == 4


@pytest.mark.kernel_graph
def test_dtype_to_bytes_unknown():
    """Unknown dtype must raise ValueError with a helpful message."""
    with pytest.raises(ValueError, match="[Uu]nsupported|[Uu]nknown"):
        dtype_to_bytes("unknown_type")


@pytest.mark.kernel_graph
def test_dtype_to_bytes_case_insensitive():
    """dtype_to_bytes is case-insensitive."""
    assert dtype_to_bytes("FLOAT32") == 4
    assert dtype_to_bytes("Float16") == 2
    assert dtype_to_bytes("INT8") == 1


@pytest.mark.kernel_graph
def test_dtype_to_bytes_whitespace():
    """Leading/trailing whitespace is stripped before lookup."""
    assert dtype_to_bytes("  float32  ") == 4
    assert dtype_to_bytes("\tfloat16\n") == 2


@pytest.mark.kernel_graph
def test_dtype_to_bytes_fp_aliases():
    """Common fp aliases (fp32, fp16, fp64, bf16) should be recognized."""
    assert dtype_to_bytes("fp32") == 4
    assert dtype_to_bytes("fp16") == 2
    assert dtype_to_bytes("fp64") == 8
    assert dtype_to_bytes("bf16") == 2
