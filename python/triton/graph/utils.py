"""Shared DAG utilities and tensor analysis helpers for the graph-level optimization layer.

This module provides foundational utility functions consumed by multiple graph modules
(kgir.py, fusion.py, scheduler.py, memory_planner.py, capture.py). All functions are
pure utility functions with no class state, no side effects, and no dependencies on
other graph modules.

Functions are organized into four categories:
    1. DAG Algorithms: topological sort, cycle detection, critical path computation
    2. Tensor Shape/Stride Helpers: shape compatibility, memory overlap, size computation
    3. Grid Dimension Helpers: grid compatibility and unification for fusion
    4. Dtype Utilities: dtype-to-byte mapping and dtype compatibility checks
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Dtype Size Mapping (bytes per element)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Comprehensive mapping covering all Triton-supported dtype strings to their
# byte sizes. Includes standard IEEE 754 floating-point, FP8 variants, signed
# and unsigned integers, boolean, and pointer types.  Multiple naming
# conventions are supported for user convenience.

DTYPE_SIZES: Dict[str, int] = {
    # Standard IEEE 754 floating-point types
    "fp64": 8, "f64": 8, "float64": 8,
    "fp32": 4, "f32": 4, "float32": 4,
    "fp16": 2, "f16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    # 8-bit floating-point types (FP8 variants)
    "fp8e5m2": 1, "f8e5m2": 1,
    "fp8e4m3fn": 1, "f8e4m3fn": 1,
    "fp8e4m3": 1, "f8e4m3": 1,
    "fp8e5m2fnuz": 1, "f8e5m2fnuz": 1,
    "fp8e4m3fnuz": 1, "f8e4m3fnuz": 1,
    # Signed integer types
    "i64": 8, "int64": 8,
    "i32": 4, "int32": 4,
    "i16": 2, "int16": 2,
    "i8": 1, "int8": 1,
    "i1": 1, "int1": 1,
    # Unsigned integer types
    "u64": 8, "uint64": 8,
    "u32": 4, "uint32": 4,
    "u16": 2, "uint16": 2,
    "u8": 1, "uint8": 1,
    # Boolean type
    "bool": 1,
    # Pointer types (64-bit addressing)
    "pointer": 8, "ptr": 8,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: DAG Algorithms
# ═══════════════════════════════════════════════════════════════════════════════


def topological_sort(
    adjacency: Dict[int, List[int]],
    nodes: Optional[Set[int]] = None,
) -> List[int]:
    """Compute a topological ordering of a DAG using Kahn's algorithm.

    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Forward adjacency list mapping each node ID to its successor IDs.
        Nodes that appear only as successors (not as keys) are still
        considered part of the graph.
    nodes : Optional[Set[int]]
        Explicit set of all node IDs in the graph.  When *None* (default),
        the node set is derived from all keys and values in *adjacency*.

    Returns
    -------
    List[int]
        Node IDs in a valid topological order.

    Raises
    ------
    ValueError
        If the graph contains a cycle (i.e. is not a DAG).

    Complexity
    ----------
    O(V + E) time and space.
    """
    # Derive node set from adjacency keys and values when not provided.
    if nodes is None:
        all_nodes: Set[int] = set()
        for src, successors in adjacency.items():
            all_nodes.add(src)
            for s in successors:
                all_nodes.add(s)
        nodes = all_nodes

    # Empty graph edge case.
    if not nodes:
        return []

    # Compute in-degree for every node.
    in_degree: Dict[int, int] = {n: 0 for n in nodes}
    for src in nodes:
        for succ in adjacency.get(src, []):
            if succ in in_degree:
                in_degree[succ] += 1

    # Seed the queue with all zero-in-degree nodes.
    # Sorting ensures deterministic output for a given graph.
    queue: deque[int] = deque(sorted(n for n, d in in_degree.items() if d == 0))
    result: List[int] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for succ in adjacency.get(node, []):
            if succ in in_degree:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

    if len(result) != len(nodes):
        raise ValueError(
            f"Graph contains a cycle: topological sort processed {len(result)} "
            f"of {len(nodes)} nodes"
        )

    return result


def detect_cycle(adjacency: Dict[int, List[int]]) -> Optional[List[int]]:
    """Detect a cycle in the directed graph and return the cycle path.

    Uses iterative DFS with three-colour marking (white / gray / black) to
    locate back-edges.  When a back-edge is found the cycle is reconstructed
    via parent pointers.

    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Forward adjacency list.

    Returns
    -------
    Optional[List[int]]
        A list of node IDs forming the cycle (first and last element are
        identical), or *None* if the graph is acyclic.

    Complexity
    ----------
    O(V + E) time and space.
    """
    # Derive full node set.
    all_nodes: Set[int] = set()
    for src, successors in adjacency.items():
        all_nodes.add(src)
        for s in successors:
            all_nodes.add(s)

    if not all_nodes:
        return None

    # Three-colour constants.
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[int, int] = {n: WHITE for n in all_nodes}
    parent: Dict[int, Optional[int]] = {n: None for n in all_nodes}

    # Iterate over nodes in deterministic (sorted) order so that repeated
    # calls on the same graph return the same cycle.
    for start in sorted(all_nodes):
        if color[start] != WHITE:
            continue

        # Iterative DFS using an explicit stack.  Each entry is a tuple of
        # (node_id, iterator_over_children).
        dfs_stack: List[tuple] = [(start, iter(adjacency.get(start, [])))]
        color[start] = GRAY

        while dfs_stack:
            node, children = dfs_stack[-1]
            advanced = False
            for child in children:
                if child not in color:
                    # Node referenced but not in the derived set — skip.
                    continue
                if color[child] == GRAY:
                    # Back-edge detected → reconstruct the cycle path.
                    cycle: List[int] = []
                    current: Optional[int] = node
                    while current is not None and current != child:
                        cycle.append(current)
                        current = parent[current]
                    cycle.append(child)
                    cycle.reverse()
                    cycle.append(child)  # Close the cycle.
                    return cycle
                if color[child] == WHITE:
                    color[child] = GRAY
                    parent[child] = node
                    dfs_stack.append((child, iter(adjacency.get(child, []))))
                    advanced = True
                    break  # Descend into the child immediately.
            if not advanced:
                color[node] = BLACK
                dfs_stack.pop()

    return None


def compute_critical_path(
    adjacency: Dict[int, List[int]],
    node_weights: Dict[int, float],
) -> Tuple[List[int], float]:
    """Compute the longest weighted path (critical path) through the DAG.

    Uses dynamic programming on a topological ordering.  Each node's weight
    represents an execution-time estimate; the critical path is the chain of
    nodes whose cumulative weight is maximised.

    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Forward adjacency list ``{node_id: [successor_ids]}``.
    node_weights : Dict[int, float]
        Execution-time estimate per node ``{node_id: weight}``.
        Nodes absent from this dict are assumed to have weight ``0.0``.

    Returns
    -------
    Tuple[List[int], float]
        ``(path, total_weight)`` where *path* is a list of node IDs on the
        critical path from a root to a leaf, and *total_weight* is the sum
        of their weights.

    Raises
    ------
    ValueError
        If the graph contains a cycle (propagated from *topological_sort*).

    Complexity
    ----------
    O(V + E) time and space.
    """
    # Derive the complete node set.
    all_nodes: Set[int] = set()
    for src, successors in adjacency.items():
        all_nodes.add(src)
        for s in successors:
            all_nodes.add(s)

    if not all_nodes:
        return ([], 0.0)

    # Build a reverse adjacency for predecessor look-up.
    reverse_adj: Dict[int, List[int]] = {n: [] for n in all_nodes}
    for src in all_nodes:
        for succ in adjacency.get(src, []):
            if succ in reverse_adj:
                reverse_adj[succ].append(src)

    topo_order = topological_sort(adjacency, all_nodes)

    # dist[v] = longest path weight ending at v (inclusive of v's weight).
    dist: Dict[int, float] = {}
    prev_node: Dict[int, Optional[int]] = {}

    for node in topo_order:
        weight = node_weights.get(node, 0.0)
        predecessors = reverse_adj.get(node, [])

        if not predecessors:
            dist[node] = weight
            prev_node[node] = None
        else:
            best_dist = -1.0
            best_pred: Optional[int] = None
            for pred in predecessors:
                pred_dist = dist.get(pred, 0.0)
                if pred_dist > best_dist:
                    best_dist = pred_dist
                    best_pred = pred
            dist[node] = best_dist + weight
            prev_node[node] = best_pred

    # Identify the end-node of the critical path.
    end_node = max(dist, key=lambda n: dist[n])
    total_weight = dist[end_node]

    # Reconstruct the path by following predecessor pointers.
    path: List[int] = []
    current: Optional[int] = end_node
    while current is not None:
        path.append(current)
        current = prev_node.get(current)
    path.reverse()

    return (path, total_weight)


def compute_critical_path_remaining(
    adjacency: Dict[int, List[int]],
    node_weights: Dict[int, float],
) -> Dict[int, float]:
    """For each node compute the critical-path weight remaining to any leaf.

    The remaining weight for a node *v* equals its own weight plus the
    maximum remaining weight among its successors.  This is used by the
    scheduler for priority computation (nodes with more remaining work are
    scheduled first).

    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Forward adjacency list.
    node_weights : Dict[int, float]
        Execution-time estimate per node.  Missing nodes default to ``0.0``.

    Returns
    -------
    Dict[int, float]
        ``{node_id: remaining_critical_path_weight}``.

    Raises
    ------
    ValueError
        If the graph contains a cycle.

    Complexity
    ----------
    O(V + E) time and space.
    """
    all_nodes: Set[int] = set()
    for src, successors in adjacency.items():
        all_nodes.add(src)
        for s in successors:
            all_nodes.add(s)

    if not all_nodes:
        return {}

    topo_order = topological_sort(adjacency, all_nodes)
    remaining: Dict[int, float] = {}

    # Process in reverse topological order (leaves first).
    for node in reversed(topo_order):
        weight = node_weights.get(node, 0.0)
        successors = adjacency.get(node, [])

        if not successors:
            remaining[node] = weight
        else:
            max_succ = max(remaining.get(s, 0.0) for s in successors)
            remaining[node] = weight + max_succ

    return remaining


def get_all_predecessors(
    adjacency: Dict[int, List[int]],
    node_id: int,
) -> Set[int]:
    """Return the set of all transitive predecessors of *node_id*.

    A BFS over a reversed adjacency list discovers every node from which
    *node_id* is reachable.  The returned set does **not** include
    *node_id* itself.

    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Forward adjacency list.
    node_id : int
        The target node whose predecessors are requested.

    Returns
    -------
    Set[int]
        All transitive predecessor node IDs.

    Complexity
    ----------
    O(V + E) time and space.
    """
    # Build reverse adjacency on-the-fly.
    reverse_adj: Dict[int, List[int]] = {}
    for src, successors in adjacency.items():
        for succ in successors:
            reverse_adj.setdefault(succ, []).append(src)

    visited: Set[int] = set()
    queue: deque[int] = deque()

    for pred in reverse_adj.get(node_id, []):
        if pred not in visited:
            visited.add(pred)
            queue.append(pred)

    while queue:
        current = queue.popleft()
        for pred in reverse_adj.get(current, []):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)

    return visited


def get_all_successors(
    adjacency: Dict[int, List[int]],
    node_id: int,
) -> Set[int]:
    """Return the set of all transitive successors of *node_id*.

    A BFS over the forward adjacency list discovers every node reachable
    from *node_id*.  The returned set does **not** include *node_id* itself.

    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Forward adjacency list.
    node_id : int
        The source node whose successors are requested.

    Returns
    -------
    Set[int]
        All transitive successor node IDs.

    Complexity
    ----------
    O(V + E) time and space.
    """
    visited: Set[int] = set()
    queue: deque[int] = deque()

    for succ in adjacency.get(node_id, []):
        if succ not in visited:
            visited.add(succ)
            queue.append(succ)

    while queue:
        current = queue.popleft()
        for succ in adjacency.get(current, []):
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    return visited


def are_independent(
    adjacency: Dict[int, List[int]],
    node_a: int,
    node_b: int,
) -> bool:
    """Check whether two nodes are independent (no dependency relationship).

    Two nodes are independent iff neither is a transitive predecessor or
    successor of the other.  Used by ``SiblingFusionAnalyzer`` to identify
    fusible independent kernel pairs.

    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Forward adjacency list.
    node_a : int
        First node ID.
    node_b : int
        Second node ID.

    Returns
    -------
    bool
        *True* if the nodes are independent, *False* otherwise.
    """
    if node_a == node_b:
        return False

    # Check if node_b is reachable from node_a.
    successors_a = get_all_successors(adjacency, node_a)
    if node_b in successors_a:
        return False

    # Check if node_a is reachable from node_b (reverse direction).
    successors_b = get_all_successors(adjacency, node_b)
    if node_a in successors_b:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Tensor Shape and Stride Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def shapes_compatible(
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...],
) -> bool:
    """Check if two tensor shapes are compatible for fusion.

    Compatibility follows NumPy / PyTorch broadcasting rules: shapes are
    aligned from the rightmost dimension and are compatible at each position
    if the dimensions are equal or one of them is ``1``.

    Parameters
    ----------
    shape_a : Tuple[int, ...]
        Shape of the first tensor.
    shape_b : Tuple[int, ...]
        Shape of the second tensor.

    Returns
    -------
    bool
        *True* when the shapes are broadcast-compatible.
    """
    if shape_a == shape_b:
        return True

    # Handle empty shapes (scalar tensors).
    if not shape_a or not shape_b:
        return True  # A scalar broadcasts with anything.

    max_rank = max(len(shape_a), len(shape_b))
    # Left-pad with 1s so both shapes have equal rank.
    padded_a = (1,) * (max_rank - len(shape_a)) + shape_a
    padded_b = (1,) * (max_rank - len(shape_b)) + shape_b

    for dim_a, dim_b in zip(padded_a, padded_b):
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False
    return True


def strides_compatible(
    strides_a: Tuple[int, ...],
    strides_b: Tuple[int, ...],
    shape: Tuple[int, ...],
) -> bool:
    """Check if two stride tuples produce the same memory-access pattern.

    For each dimension, if the shape extent is ``<= 1`` the stride is
    irrelevant (only one element along that axis).  Otherwise the strides
    must match exactly.

    Parameters
    ----------
    strides_a : Tuple[int, ...]
        Strides of the first tensor (in elements, not bytes).
    strides_b : Tuple[int, ...]
        Strides of the second tensor.
    shape : Tuple[int, ...]
        Shared tensor shape used to decide which dimensions matter.

    Returns
    -------
    bool
        *True* when the memory-access patterns are identical.
    """
    if len(strides_a) != len(strides_b) or len(strides_a) != len(shape):
        return False

    for stride_a, stride_b, dim_size in zip(strides_a, strides_b, shape):
        if dim_size <= 1:
            continue  # Stride is irrelevant for size-0 or size-1 dimensions.
        if stride_a != stride_b:
            return False

    return True


def compute_tensor_size_bytes(
    shape: Tuple[int, ...],
    dtype: str,
) -> int:
    """Compute the dense tensor size in bytes.

    ``size = product(shape) * dtype_byte_width``

    Parameters
    ----------
    shape : Tuple[int, ...]
        Tensor shape.
    dtype : str
        Triton dtype string (e.g. ``"fp32"``, ``"bf16"``).

    Returns
    -------
    int
        Total size in bytes.  Returns ``0`` for a zero-element tensor.

    Raises
    ------
    ValueError
        If *dtype* is not recognised.
    """
    if not shape:
        return 0
    element_count = math.prod(shape)
    if element_count <= 0:
        return 0
    element_size = dtype_to_bytes(dtype)
    return element_count * element_size


def memory_regions_overlap(
    ptr_a: int,
    size_a: int,
    ptr_b: int,
    size_b: int,
) -> bool:
    """Check whether two memory regions overlap.

    Region A is ``[ptr_a, ptr_a + size_a)`` and region B is
    ``[ptr_b, ptr_b + size_b)``.  A zero-size region never overlaps.

    Parameters
    ----------
    ptr_a : int
        Start address of region A.
    size_a : int
        Size of region A in bytes.
    ptr_b : int
        Start address of region B.
    size_b : int
        Size of region B in bytes.

    Returns
    -------
    bool
        *True* when the two regions overlap.
    """
    if size_a <= 0 or size_b <= 0:
        return False
    return ptr_a < (ptr_b + size_b) and ptr_b < (ptr_a + size_a)


def compute_memory_footprint(
    shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    dtype: str,
) -> int:
    """Compute the actual memory footprint considering strides.

    For non-contiguous tensors the footprint can exceed the dense element
    count because elements may be spaced further apart in memory.  The
    footprint is computed as::

        span_elements = sum((dim - 1) * abs(stride) for dim, stride in
                            zip(shape, strides)) + 1
        footprint = span_elements * dtype_byte_width

    A tensor with any zero-extent dimension occupies zero bytes.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Tensor shape.
    strides : Tuple[int, ...]
        Strides in elements (not bytes).
    dtype : str
        Triton dtype string.

    Returns
    -------
    int
        Memory footprint in bytes.

    Raises
    ------
    ValueError
        If *shape* and *strides* have mismatched lengths, or *dtype* is
        unknown.
    """
    if not shape or not strides:
        return 0

    if len(shape) != len(strides):
        raise ValueError(
            f"Shape and strides must have the same length: "
            f"len(shape)={len(shape)}, len(strides)={len(strides)}"
        )

    # A zero-element tensor occupies no memory.
    if math.prod(shape) == 0:
        return 0

    element_size = dtype_to_bytes(dtype)

    # The span of addresses touched (in elements).
    max_offset = sum(
        (dim - 1) * abs(stride) for dim, stride in zip(shape, strides)
    )
    return (max_offset + 1) * element_size


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Grid Dimension Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_grid(grid: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Normalise a grid tuple to exactly three dimensions.

    Grids shorter than 3-D are right-padded with ``1``; grids longer than
    3-D have their trailing dimensions collapsed via multiplication.

    This is an internal helper — not part of the public API.
    """
    if len(grid) == 0:
        return (1, 1, 1)
    if len(grid) == 1:
        return (grid[0], 1, 1)
    if len(grid) == 2:
        return (grid[0], grid[1], 1)
    if len(grid) == 3:
        return (grid[0], grid[1], grid[2])
    # > 3-D: collapse extra dims into the third.
    collapsed = math.prod(grid[2:])
    return (grid[0], grid[1], collapsed)


def grids_compatible(
    grid_a: Tuple[int, ...],
    grid_b: Tuple[int, ...],
) -> bool:
    """Check if two launch-grid dimensions can be unified for sibling fusion.

    Two grids are considered compatible when any of the following hold:

    * They are identical (after 3-D normalisation).
    * They have the same total number of work items (thread blocks).
    * They have the same effective rank (number of non-trivial dimensions).

    Parameters
    ----------
    grid_a : Tuple[int, ...]
        Launch grid of the first kernel.
    grid_b : Tuple[int, ...]
        Launch grid of the second kernel.

    Returns
    -------
    bool
        *True* when the grids can be unified.
    """
    norm_a = _normalize_grid(grid_a)
    norm_b = _normalize_grid(grid_b)

    # Identical grids.
    if norm_a == norm_b:
        return True

    # Same total work items.
    total_a = math.prod(norm_a)
    total_b = math.prod(norm_b)
    if total_a > 0 and total_b > 0 and total_a == total_b:
        return True

    # Same effective rank (number of dimensions > 1).
    rank_a = sum(1 for d in norm_a if d > 1)
    rank_b = sum(1 for d in norm_b if d > 1)
    if rank_a == rank_b:
        return True

    return False


def compute_unified_grid(
    grids: List[Tuple[int, ...]],
) -> Tuple[int, ...]:
    """Compute a unified grid that encompasses all input grids.

    For sibling fusion the unified grid takes the element-wise maximum of
    each dimension across all grids.  Shorter grids are right-padded with
    ``1`` before comparison.

    Parameters
    ----------
    grids : List[Tuple[int, ...]]
        List of launch grids to unify.

    Returns
    -------
    Tuple[int, ...]
        The unified grid.  Returns ``(1, 1, 1)`` for an empty list.
    """
    if not grids:
        return (1, 1, 1)

    # Determine the maximum dimensionality among all grids.
    max_dims = max(len(g) for g in grids)
    if max_dims == 0:
        return (1, 1, 1)

    # Element-wise max across all grids (right-pad shorter grids with 1).
    unified: List[int] = [1] * max_dims
    for grid in grids:
        padded = list(grid) + [1] * (max_dims - len(grid))
        for i in range(max_dims):
            if padded[i] > unified[i]:
                unified[i] = padded[i]

    return tuple(unified)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Dtype Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def dtype_to_bytes(dtype: str) -> int:
    """Map a Triton dtype string to its size in bytes.

    The lookup is case-insensitive and strips leading/trailing whitespace.

    Parameters
    ----------
    dtype : str
        Triton dtype string (e.g. ``"fp32"``, ``"bf16"``, ``"i8"``).

    Returns
    -------
    int
        Element size in bytes.

    Raises
    ------
    ValueError
        If *dtype* is not recognised.
    """
    normalised = dtype.lower().strip()
    size = DTYPE_SIZES.get(normalised)
    if size is not None:
        return size
    raise ValueError(
        f"Unknown dtype '{dtype}'. Supported dtypes: "
        f"{sorted(DTYPE_SIZES.keys())}"
    )


def dtypes_compatible(dtype_a: str, dtype_b: str) -> bool:
    """Check if two dtypes have the same byte width.

    This is relevant for pointer-aliasing analysis where two tensors with
    the same element size can share the same memory without misalignment.

    Parameters
    ----------
    dtype_a : str
        First dtype string.
    dtype_b : str
        Second dtype string.

    Returns
    -------
    bool
        *True* when both dtypes occupy the same number of bytes.

    Raises
    ------
    ValueError
        If either dtype is not recognised.
    """
    return dtype_to_bytes(dtype_a) == dtype_to_bytes(dtype_b)
