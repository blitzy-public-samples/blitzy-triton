"""Python-side KGIR (Kernel Graph Intermediate Representation) data structures.

This module defines the primary Python representation of a directed acyclic
graph (DAG) of kernel launches for Triton's graph-level optimization layer.
All downstream analysis passes — fusion, memory planning, scheduling, and
hardware-aware dispatch — operate on the classes defined here.

Classes
-------
HardwareProfile
    Per-device hardware descriptor (SM/CU count, shared memory, registers,
    bandwidth, interconnect, etc.).  Schema matches AAP §0.5.1 exactly.
NodeMetadata
    Per-node metadata for a kernel in the KGIR graph (memory access patterns,
    tensor shapes/strides/dtypes, grid dimensions, resource usage, per-target
    annotations).
KGIRNode
    A single kernel launch node in the graph.  Mutable for runtime annotation
    write-back as required by the closed-loop feedback controller.
KGIREdge
    A dependency edge between kernel nodes (data dependency, anti-dependency,
    resource conflict, or cross-device transfer).
KGIRGraph
    DAG container that maintains adjacency lists, enforces acyclicity, exposes
    traversal helpers, and supports fusion mutations.  Wraps the C++ KGIR MLIR
    dialect via lazy-loaded pybind11 bindings with a pure-Python fallback.

Performance Constraint
----------------------
KGIR memory overhead MUST be < 10 MB for graphs with ≤ 100 kernels
(AAP §0.7.2).  All data structures use compact representations (integer IDs,
dicts-of-dicts) to stay well within this budget.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from .utils import topological_sort, detect_cycle
from .errors import GraphCaptureError

if TYPE_CHECKING:
    from triton.backends.compiler import GPUTarget

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Valid edge types (AAP §0.1.1)
# ═══════════════════════════════════════════════════════════════════════════════

VALID_EDGE_TYPES = frozenset({
    "data_dep",              # Producer writes, consumer reads
    "anti_dep",              # Consumer writes, producer reads
    "resource_conflict",     # Shared resource contention
    "cross_device_transfer", # Cross-device data transfer edge
})


# ═══════════════════════════════════════════════════════════════════════════════
# HardwareProfile — per-device hardware descriptor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareProfile:
    """Per-device hardware descriptor.  Target-agnostic representation of GPU
    capabilities.

    The schema matches AAP §0.5.1 *exactly* (12 mandatory fields plus an
    optional ``gpu_target`` back-reference).  Hardware-specific information is
    encapsulated here rather than baked into KGIR node IR, ensuring that the
    IR remains target-agnostic at construction time.

    Attributes
    ----------
    vendor : str
        Device vendor string, e.g. ``"nvidia"``, ``"amd"``.
    arch_generation : str
        Architecture generation identifier, e.g. ``"sm_90"``, ``"gfx942"``.
    sm_count : int
        Number of Streaming Multiprocessors (NVIDIA) or Compute Units (AMD).
    smem_per_sm_bytes : int
        Shared memory capacity per SM/CU in bytes.
    registers_per_sm : int
        Register file size per SM/CU (total registers).
    global_memory_bytes : int
        Total device global memory in bytes.
    memory_bandwidth_gbps : float
        Peak memory bandwidth in GB/s.
    compute_throughput_tflops : float
        Peak compute throughput in TFLOPS.
    warp_size : int
        Warp size (32 for NVIDIA, 64 for AMD wavefront).
    max_concurrent_streams : int
        Maximum concurrent streams/queues supported.
    interconnect_type : str
        Interconnect type string, e.g. ``"pcie_4"``, ``"nvlink_4"``,
        ``"infinity_fabric"``.
    interconnect_bandwidth_gbps : float
        Interconnect bandwidth in GB/s.
    gpu_target : Optional[Any]
        Optional reference to a ``GPUTarget`` instance from
        ``triton.backends.compiler`` for compilation integration.
    """

    vendor: str
    arch_generation: str
    sm_count: int
    smem_per_sm_bytes: int
    registers_per_sm: int
    global_memory_bytes: int
    memory_bandwidth_gbps: float
    compute_throughput_tflops: float
    warp_size: int
    max_concurrent_streams: int
    interconnect_type: str
    interconnect_bandwidth_gbps: float
    gpu_target: Optional[Any] = None

    # -- Convenience helpers --------------------------------------------------

    def total_smem_bytes(self) -> int:
        """Return the aggregate shared memory across all SMs/CUs."""
        return self.sm_count * self.smem_per_sm_bytes

    def total_registers(self) -> int:
        """Return the aggregate register count across all SMs/CUs."""
        return self.sm_count * self.registers_per_sm

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (for JSON cache persistence)."""
        return {
            "vendor": self.vendor,
            "arch_generation": self.arch_generation,
            "sm_count": self.sm_count,
            "smem_per_sm_bytes": self.smem_per_sm_bytes,
            "registers_per_sm": self.registers_per_sm,
            "global_memory_bytes": self.global_memory_bytes,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "compute_throughput_tflops": self.compute_throughput_tflops,
            "warp_size": self.warp_size,
            "max_concurrent_streams": self.max_concurrent_streams,
            "interconnect_type": self.interconnect_type,
            "interconnect_bandwidth_gbps": self.interconnect_bandwidth_gbps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardwareProfile:
        """Deserialize from a plain dictionary."""
        return cls(
            vendor=data["vendor"],
            arch_generation=data["arch_generation"],
            sm_count=int(data["sm_count"]),
            smem_per_sm_bytes=int(data["smem_per_sm_bytes"]),
            registers_per_sm=int(data["registers_per_sm"]),
            global_memory_bytes=int(data["global_memory_bytes"]),
            memory_bandwidth_gbps=float(data["memory_bandwidth_gbps"]),
            compute_throughput_tflops=float(data["compute_throughput_tflops"]),
            warp_size=int(data["warp_size"]),
            max_concurrent_streams=int(data["max_concurrent_streams"]),
            interconnect_type=data["interconnect_type"],
            interconnect_bandwidth_gbps=float(data["interconnect_bandwidth_gbps"]),
            gpu_target=data.get("gpu_target"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NodeMetadata — per-node kernel metadata
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeMetadata:
    """Per-node metadata for a kernel in the KGIR graph.

    Captures everything the downstream analysis passes need to reason about a
    single kernel launch: memory access patterns, tensor argument details,
    launch grid, resource consumption, and per-target annotations (both
    static/heuristic and runtime-measured).

    Attributes
    ----------
    memory_access_patterns : Dict[str, Any]
        Read/write patterns per tensor argument, keyed by argument name or
        index.  Typical values describe ``"read"``, ``"write"``, or
        ``"read_write"`` access with optional stride/tiling annotations.
    tensor_shapes : Dict[int, Tuple[int, ...]]
        Mapping from argument index to the shape tuple of the corresponding
        tensor.
    tensor_strides : Dict[int, Tuple[int, ...]]
        Mapping from argument index to the stride tuple.
    tensor_dtypes : Dict[int, str]
        Mapping from argument index to a dtype string (e.g. ``"fp16"``).
    grid_dimensions : Tuple[int, ...]
        Launch grid ``(grid_x, grid_y, grid_z)``.
    shared_memory_bytes : int
        Shared memory (SMEM) usage in bytes for this kernel.
    register_count : int
        Estimated register usage per thread.
    num_warps : int
        Number of warps (NVIDIA) or wavefronts (AMD) per block.
    hardware_target_annotations : Dict[str, Any]
        Per-target static/heuristic annotations keyed by target identifier
        string (e.g. ``"sm_90"``).
    runtime_performance_annotations : Dict[str, Dict[str, float]]
        Per-target measured metrics populated by the runtime profiler and
        written back by the feedback controller.  Outer key is the target
        identifier; inner dict maps metric name to value.
    """

    memory_access_patterns: Dict[str, Any] = field(default_factory=dict)
    tensor_shapes: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    tensor_strides: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    tensor_dtypes: Dict[int, str] = field(default_factory=dict)
    grid_dimensions: Tuple[int, ...] = (1, 1, 1)
    shared_memory_bytes: int = 0
    register_count: int = 0
    num_warps: int = 4
    hardware_target_annotations: Dict[str, Any] = field(default_factory=dict)
    runtime_performance_annotations: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )


# ═══════════════════════════════════════════════════════════════════════════════
# KGIRNode — kernel launch node
# ═══════════════════════════════════════════════════════════════════════════════

class KGIRNode:
    """Represents a single kernel launch node in the KGIR graph.

    Nodes are **mutable** so that the runtime profiler / feedback controller
    can write measured performance annotations back into the graph without
    rebuilding it (AAP §0.1.1 requirement).

    Parameters
    ----------
    node_id : int
        Unique identifier within the owning ``KGIRGraph``.
    kernel_fn : Any
        Reference to the Triton ``JITFunction`` (or equivalent callable)
        that this node represents.
    metadata : NodeMetadata
        All per-node metadata (shapes, dtypes, grid, resources, annotations).
    is_fused : bool
        ``True`` if this node is the product of a fusion pass.
    fused_from : Optional[List[int]]
        Original node IDs that were merged to create this fused node.
    """

    __slots__ = ("node_id", "kernel_fn", "metadata", "is_fused", "fused_from")

    def __init__(
        self,
        node_id: int,
        kernel_fn: Any,
        metadata: NodeMetadata,
        is_fused: bool = False,
        fused_from: Optional[List[int]] = None,
    ) -> None:
        self.node_id: int = node_id
        self.kernel_fn: Any = kernel_fn
        self.metadata: NodeMetadata = metadata
        self.is_fused: bool = is_fused
        self.fused_from: Optional[List[int]] = (
            list(fused_from) if fused_from is not None else None
        )

    # -- Public API -----------------------------------------------------------

    def get_metadata(self) -> NodeMetadata:
        """Return the node's full metadata descriptor."""
        return self.metadata

    def update_performance_annotation(
        self,
        target: str,
        metrics: Dict[str, float],
    ) -> None:
        """Write measured runtime metrics for *target* into the node metadata.

        This is the primary mutation path used by the feedback controller to
        propagate profiled data back into the KGIR for re-optimization.

        Parameters
        ----------
        target : str
            Hardware target identifier (e.g. ``"sm_90"``).
        metrics : Dict[str, float]
            Metric name → measured value (e.g. ``{"wall_clock_ms": 0.42}``).
        """
        self.metadata.runtime_performance_annotations[target] = dict(metrics)

    def get_performance_annotation(
        self,
        target: str,
    ) -> Optional[Dict[str, float]]:
        """Return measured metrics for *target*, or ``None`` if not yet profiled.

        Parameters
        ----------
        target : str
            Hardware target identifier.

        Returns
        -------
        Optional[Dict[str, float]]
            Copy of the measured metrics dict, or ``None``.
        """
        ann = self.metadata.runtime_performance_annotations.get(target)
        if ann is not None:
            return dict(ann)
        return None

    def get_resource_usage(self) -> Dict[str, int]:
        """Return a compact summary of the kernel's resource consumption.

        Returns
        -------
        Dict[str, int]
            Keys: ``"shared_memory_bytes"``, ``"register_count"``,
            ``"num_warps"``.
        """
        return {
            "shared_memory_bytes": self.metadata.shared_memory_bytes,
            "register_count": self.metadata.register_count,
            "num_warps": self.metadata.num_warps,
        }

    def is_compatible_for_fusion(
        self,
        other: KGIRNode,
        target: HardwareProfile,
    ) -> bool:
        """Check whether this node can be fused with *other* on *target*.

        Fusion compatibility requires that the combined shared-memory and
        register usage fits within the per-SM/CU limits described by *target*,
        and that the combined warp count does not exceed the maximum warp
        occupancy (``target.sm_count`` × ``target.warp_size`` is the warp
        pool, but per-SM the limit is ``registers_per_sm / register_count``
        — here we use a conservative per-SM resource check).

        Parameters
        ----------
        other : KGIRNode
            The candidate partner node for fusion.
        target : HardwareProfile
            Hardware profile describing per-SM/CU resource limits.

        Returns
        -------
        bool
            ``True`` if the combined resources fit within *target* limits.
        """
        combined_smem = (
            self.metadata.shared_memory_bytes
            + other.metadata.shared_memory_bytes
        )
        if combined_smem > target.smem_per_sm_bytes:
            return False

        combined_regs = (
            self.metadata.register_count + other.metadata.register_count
        )
        if combined_regs > target.registers_per_sm:
            return False

        combined_warps = self.metadata.num_warps + other.metadata.num_warps
        # Conservative limit: each SM can host registers_per_sm / warp_size
        # logical warps at most.  If the kernel's combined register demand
        # exceeds per-SM capacity the fusion is rejected above; warp count is
        # an additional soft check.
        max_warps_per_sm = target.registers_per_sm // max(target.warp_size, 1)
        if combined_warps > max_warps_per_sm and max_warps_per_sm > 0:
            return False

        return True

    def __repr__(self) -> str:
        fused_tag = " [FUSED]" if self.is_fused else ""
        return (
            f"KGIRNode(id={self.node_id}, "
            f"grid={self.metadata.grid_dimensions}{fused_tag})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# KGIREdge — dependency edge
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KGIREdge:
    """Dependency edge between kernel nodes in the KGIR graph.

    Attributes
    ----------
    source_id : int
        Producer / source node ID.
    target_id : int
        Consumer / target node ID.
    edge_type : str
        One of ``"data_dep"``, ``"anti_dep"``, ``"resource_conflict"``, or
        ``"cross_device_transfer"`` (AAP §0.1.1).
    tensor_id : Optional[int]
        Index of the tensor argument involved in this dependency, or ``None``
        for non-tensor edges (e.g. resource conflicts).
    metadata : Optional[Dict[str, Any]]
        Free-form metadata attached to the edge (e.g. transfer cost
        estimates, bandwidth requirements).
    """

    source_id: int
    target_id: int
    edge_type: str
    tensor_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        if self.edge_type not in VALID_EDGE_TYPES:
            raise GraphCaptureError(
                f"Invalid edge type '{self.edge_type}'. "
                f"Must be one of {sorted(VALID_EDGE_TYPES)}."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# KGIRGraph — DAG container
# ═══════════════════════════════════════════════════════════════════════════════

class KGIRGraph:
    """DAG container for the Kernel Graph Intermediate Representation.

    Maintains:
    * A node dictionary (``int → KGIRNode``)
    * An edge list
    * Forward and reverse adjacency lists for efficient traversal
    * An optional list of ``HardwareProfile`` descriptors

    The graph enforces DAG acyclicity on every ``add_edge`` call to prevent
    downstream passes from encountering invalid topologies.

    The class wraps the C++ KGIR MLIR dialect via lazy-loaded pybind11
    bindings.  When the native bindings are unavailable (e.g. in
    pure-Python test environments), all operations fall back to the Python
    implementation with identical semantics.

    Parameters
    ----------
    hardware_profiles : Optional[List[HardwareProfile]]
        Available hardware profiles discovered at trace-capture time.
    """

    def __init__(
        self,
        hardware_profiles: Optional[List[HardwareProfile]] = None,
    ) -> None:
        self._nodes: Dict[int, KGIRNode] = {}
        self._edges: List[KGIREdge] = []
        self._adjacency: Dict[int, List[int]] = {}
        self._reverse_adjacency: Dict[int, List[int]] = {}
        self._hardware_profiles: List[HardwareProfile] = (
            list(hardware_profiles) if hardware_profiles is not None else []
        )
        self._next_node_id: int = 0
        self._native_available: Optional[bool] = None
        self._native_module: Any = None

    # -- Hardware profiles property -------------------------------------------

    @property
    def hardware_profiles(self) -> List[HardwareProfile]:
        """Return the list of hardware profiles associated with this graph."""
        return list(self._hardware_profiles)

    @hardware_profiles.setter
    def hardware_profiles(self, profiles: List[HardwareProfile]) -> None:
        """Replace the hardware profile list."""
        self._hardware_profiles = list(profiles)

    # -- Node operations ------------------------------------------------------

    def add_node(
        self,
        kernel_fn: Any,
        metadata: NodeMetadata,
    ) -> int:
        """Add a kernel launch node to the graph and return its ``node_id``.

        The node is assigned a monotonically increasing ID.  Adjacency lists
        are pre-allocated for the new node.

        Parameters
        ----------
        kernel_fn : Any
            Reference to the kernel function (e.g. ``JITFunction``).
        metadata : NodeMetadata
            Kernel metadata (shapes, grid, resources, etc.).

        Returns
        -------
        int
            The newly assigned node ID.
        """
        node_id = self._next_node_id
        self._next_node_id += 1
        node = KGIRNode(
            node_id=node_id,
            kernel_fn=kernel_fn,
            metadata=metadata,
        )
        self._nodes[node_id] = node
        self._adjacency[node_id] = []
        self._reverse_adjacency[node_id] = []
        return node_id

    def get_node(self, node_id: int) -> KGIRNode:
        """Retrieve a node by ID.

        Parameters
        ----------
        node_id : int
            Target node identifier.

        Returns
        -------
        KGIRNode

        Raises
        ------
        GraphCaptureError
            If the node does not exist in the graph.
        """
        if node_id not in self._nodes:
            raise GraphCaptureError(
                f"Node {node_id} does not exist in the graph."
            )
        return self._nodes[node_id]

    # -- Edge operations ------------------------------------------------------

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        tensor_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a directed dependency edge and verify DAG acyclicity.

        Parameters
        ----------
        source_id : int
            Producer / source node ID.
        target_id : int
            Consumer / target node ID.
        edge_type : str
            Dependency type (see ``VALID_EDGE_TYPES``).
        tensor_id : Optional[int]
            Index of the tensor argument involved, if applicable.
        metadata : Optional[Dict[str, Any]]
            Free-form edge metadata.

        Raises
        ------
        GraphCaptureError
            If either endpoint does not exist, the edge type is invalid, or
            adding the edge would introduce a cycle.
        """
        if source_id not in self._nodes:
            raise GraphCaptureError(
                f"Source node {source_id} does not exist in the graph."
            )
        if target_id not in self._nodes:
            raise GraphCaptureError(
                f"Target node {target_id} does not exist in the graph."
            )
        if source_id == target_id:
            raise GraphCaptureError(
                f"Self-loop detected: source and target are both node {source_id}."
            )

        # Cycle check: temporarily add the edge and run detection.
        # Use a lightweight reachability test: if target can already reach
        # source via existing edges, adding source→target creates a cycle.
        if self._would_create_cycle(source_id, target_id):
            raise GraphCaptureError(
                f"Adding edge {source_id} → {target_id} would create a cycle."
            )

        edge = KGIREdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            tensor_id=tensor_id,
            metadata=metadata,
        )
        self._edges.append(edge)
        self._adjacency[source_id].append(target_id)
        self._reverse_adjacency[target_id].append(source_id)

    def get_edges(
        self,
        node_id: Optional[int] = None,
    ) -> List[KGIREdge]:
        """Return edges, optionally filtered to a specific node.

        Parameters
        ----------
        node_id : Optional[int]
            When provided, return only edges where *node_id* appears as
            source **or** target.  When ``None``, return all edges.

        Returns
        -------
        List[KGIREdge]
        """
        if node_id is None:
            return list(self._edges)
        return [
            e for e in self._edges
            if e.source_id == node_id or e.target_id == node_id
        ]

    # -- Traversal helpers ----------------------------------------------------

    def get_successors(self, node_id: int) -> List[int]:
        """Return the direct successor node IDs of *node_id*.

        Raises
        ------
        GraphCaptureError
            If the node does not exist.
        """
        if node_id not in self._nodes:
            raise GraphCaptureError(
                f"Node {node_id} does not exist in the graph."
            )
        return list(self._adjacency.get(node_id, []))

    def get_predecessors(self, node_id: int) -> List[int]:
        """Return the direct predecessor node IDs of *node_id*.

        Raises
        ------
        GraphCaptureError
            If the node does not exist.
        """
        if node_id not in self._nodes:
            raise GraphCaptureError(
                f"Node {node_id} does not exist in the graph."
            )
        return list(self._reverse_adjacency.get(node_id, []))

    def topological_sort(self) -> List[int]:
        """Return a topological ordering of all nodes in the graph.

        Delegates to the ``topological_sort`` utility from ``.utils``.

        Returns
        -------
        List[int]
            Node IDs in dependency-respecting order.

        Raises
        ------
        GraphCaptureError
            If the graph contains a cycle (should not happen if ``add_edge``
            validation is consistent, but acts as a safety net).
        """
        try:
            return topological_sort(
                self._adjacency,
                set(self._nodes.keys()),
            )
        except ValueError as exc:
            raise GraphCaptureError(str(exc)) from exc

    def get_roots(self) -> List[int]:
        """Return node IDs with no predecessors (graph entry points).

        Returns
        -------
        List[int]
            Sorted list of root node IDs.
        """
        return sorted(
            nid for nid in self._nodes
            if not self._reverse_adjacency.get(nid)
        )

    def get_leaves(self) -> List[int]:
        """Return node IDs with no successors (graph exit points).

        Returns
        -------
        List[int]
            Sorted list of leaf node IDs.
        """
        return sorted(
            nid for nid in self._nodes
            if not self._adjacency.get(nid)
        )

    # -- Size queries ---------------------------------------------------------

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return len(self._edges)

    # -- Validation -----------------------------------------------------------

    def validate(self) -> bool:
        """Verify the structural integrity of the graph.

        Checks performed:
        1. DAG acyclicity (via ``detect_cycle`` from ``.utils``).
        2. Edge endpoint consistency (all referenced node IDs exist).
        3. Adjacency list consistency (forward and reverse match edges).

        Returns
        -------
        bool
            ``True`` if the graph passes all checks.

        Raises
        ------
        GraphCaptureError
            If any integrity check fails.
        """
        # 1. Acyclicity
        cycle = detect_cycle(self._adjacency)
        if cycle is not None:
            raise GraphCaptureError(
                f"Graph contains a cycle: {cycle}"
            )

        # 2. Edge endpoint consistency
        for edge in self._edges:
            if edge.source_id not in self._nodes:
                raise GraphCaptureError(
                    f"Edge references non-existent source node {edge.source_id}."
                )
            if edge.target_id not in self._nodes:
                raise GraphCaptureError(
                    f"Edge references non-existent target node {edge.target_id}."
                )

        # 3. Adjacency list consistency — forward edges must match edge list
        fwd_from_edges: Dict[int, List[int]] = {nid: [] for nid in self._nodes}
        rev_from_edges: Dict[int, List[int]] = {nid: [] for nid in self._nodes}
        for edge in self._edges:
            fwd_from_edges[edge.source_id].append(edge.target_id)
            rev_from_edges[edge.target_id].append(edge.source_id)

        for nid in self._nodes:
            if sorted(self._adjacency.get(nid, [])) != sorted(fwd_from_edges.get(nid, [])):
                raise GraphCaptureError(
                    f"Forward adjacency list for node {nid} is inconsistent "
                    f"with the edge list."
                )
            if sorted(self._reverse_adjacency.get(nid, [])) != sorted(rev_from_edges.get(nid, [])):
                raise GraphCaptureError(
                    f"Reverse adjacency list for node {nid} is inconsistent "
                    f"with the edge list."
                )

        return True

    # -- Fusion mutation ------------------------------------------------------

    def replace_nodes_with_fused(
        self,
        node_ids: List[int],
        fused_node: KGIRNode,
    ) -> None:
        """Replace a set of nodes with a single fused node.

        This is the primary graph-mutation API used by the fusion engine.

        Steps:
        1. Validate that all *node_ids* exist.
        2. Insert *fused_node* into the graph.
        3. Redirect incoming edges from predecessors of the set to the fused
           node and outgoing edges from successors of the set to the fused
           node — but omit any edges that were internal to the fused set.
        4. Remove the original nodes and their internal edges.
        5. Verify DAG acyclicity after mutation.

        Parameters
        ----------
        node_ids : List[int]
            IDs of the nodes being replaced.
        fused_node : KGIRNode
            The new fused node to insert.

        Raises
        ------
        GraphCaptureError
            If any node ID is invalid or the mutation would break DAG
            acyclicity.
        """
        if not node_ids:
            raise GraphCaptureError("node_ids must be non-empty for fusion.")

        fused_set = set(node_ids)

        # Validate existence
        for nid in node_ids:
            if nid not in self._nodes:
                raise GraphCaptureError(
                    f"Cannot fuse: node {nid} does not exist in the graph."
                )

        fused_id = fused_node.node_id

        # Ensure the fused node ID doesn't collide
        if fused_id in self._nodes:
            raise GraphCaptureError(
                f"Fused node ID {fused_id} already exists in the graph."
            )

        # Insert fused node
        self._nodes[fused_id] = fused_node
        self._adjacency[fused_id] = []
        self._reverse_adjacency[fused_id] = []

        # Advance next_node_id if necessary
        if fused_id >= self._next_node_id:
            self._next_node_id = fused_id + 1

        # Collect external edges (those crossing the boundary of fused_set)
        external_incoming_targets: Dict[int, List[KGIREdge]] = {}
        external_outgoing_sources: Dict[int, List[KGIREdge]] = {}
        internal_edges: List[KGIREdge] = []
        surviving_edges: List[KGIREdge] = []

        for edge in self._edges:
            src_in = edge.source_id in fused_set
            tgt_in = edge.target_id in fused_set

            if src_in and tgt_in:
                # Internal edge — will be removed
                internal_edges.append(edge)
            elif src_in and not tgt_in:
                # Outgoing edge from fused set → external consumer
                external_outgoing_sources.setdefault(edge.target_id, []).append(edge)
            elif not src_in and tgt_in:
                # Incoming edge from external producer → fused set
                external_incoming_targets.setdefault(edge.source_id, []).append(edge)
            else:
                # Completely external — keep as-is
                surviving_edges.append(edge)

        # Create redirected incoming edges (external → fused_node)
        seen_incoming: set = set()
        for src_id, edges in external_incoming_targets.items():
            for edge in edges:
                # De-duplicate: one edge per (src, fused_id, type, tensor)
                key = (src_id, fused_id, edge.edge_type, edge.tensor_id)
                if key not in seen_incoming:
                    seen_incoming.add(key)
                    surviving_edges.append(
                        KGIREdge(
                            source_id=src_id,
                            target_id=fused_id,
                            edge_type=edge.edge_type,
                            tensor_id=edge.tensor_id,
                            metadata=edge.metadata,
                        )
                    )

        # Create redirected outgoing edges (fused_node → external)
        seen_outgoing: set = set()
        for tgt_id, edges in external_outgoing_sources.items():
            for edge in edges:
                key = (fused_id, tgt_id, edge.edge_type, edge.tensor_id)
                if key not in seen_outgoing:
                    seen_outgoing.add(key)
                    surviving_edges.append(
                        KGIREdge(
                            source_id=fused_id,
                            target_id=tgt_id,
                            edge_type=edge.edge_type,
                            tensor_id=edge.tensor_id,
                            metadata=edge.metadata,
                        )
                    )

        # Remove old nodes
        for nid in fused_set:
            del self._nodes[nid]
            del self._adjacency[nid]
            del self._reverse_adjacency[nid]

        # Rebuild edge list and adjacency structures
        self._edges = surviving_edges
        # Reset adjacency for all remaining nodes
        for nid in self._nodes:
            self._adjacency[nid] = []
            self._reverse_adjacency[nid] = []
        for edge in self._edges:
            self._adjacency[edge.source_id].append(edge.target_id)
            self._reverse_adjacency[edge.target_id].append(edge.source_id)

        # Final acyclicity check
        cycle = detect_cycle(self._adjacency)
        if cycle is not None:
            raise GraphCaptureError(
                f"Fusion produced a cyclic graph: {cycle}"
            )

    # -- C++ MLIR Dialect Bridge ----------------------------------------------

    def _try_load_native(self) -> bool:
        """Attempt to load the C++ KGIR pybind11 bindings.

        The bindings expose the ``ttkgir`` MLIR dialect for high-performance
        graph manipulation.  If unavailable (build without KGIR, testing
        without compiled C++ extensions), the method returns ``False`` and all
        graph operations use the pure-Python fallback.

        Returns
        -------
        bool
            ``True`` if native bindings were loaded successfully.
        """
        if self._native_available is not None:
            return self._native_available

        try:
            # Lazy import to avoid import-time failures when C++ extensions
            # are not compiled (e.g. CPU-only test environments).
            import triton._C.libtriton as _libtriton  # type: ignore[import]

            if hasattr(_libtriton, "kgir"):
                self._native_module = _libtriton.kgir
                self._native_available = True
                logger.debug("KGIR native bindings loaded successfully.")
            else:
                self._native_available = False
                logger.debug(
                    "KGIR native bindings not found in libtriton; "
                    "using pure-Python fallback."
                )
        except ImportError:
            self._native_available = False
            logger.debug(
                "triton._C.libtriton not importable; "
                "using pure-Python KGIR fallback."
            )
        return self._native_available

    def to_mlir(self) -> str:
        """Serialize the graph to an MLIR text representation.

        When native C++ bindings are available, the serialisation is delegated
        to the ``ttkgir`` dialect printer for full fidelity.  Otherwise, a
        pure-Python generator emits syntactically valid MLIR text that can be
        round-tripped through ``from_mlir``.

        Returns
        -------
        str
            MLIR text representation of the kernel graph.
        """
        if self._try_load_native() and self._native_module is not None:
            try:
                return self._native_module.graph_to_mlir(self)
            except Exception:
                logger.debug(
                    "Native to_mlir failed; falling back to Python generation."
                )

        return self._to_mlir_python()

    @classmethod
    def from_mlir(cls, mlir_text: str) -> KGIRGraph:
        """Parse an MLIR text representation into a ``KGIRGraph``.

        Attempts native C++ parsing first; falls back to a lightweight
        Python regex-based parser for the subset of MLIR produced by
        ``to_mlir``.

        Parameters
        ----------
        mlir_text : str
            MLIR text previously produced by ``to_mlir()``.

        Returns
        -------
        KGIRGraph
            Reconstructed graph instance.
        """
        # Attempt native parsing
        try:
            import triton._C.libtriton as _libtriton  # type: ignore[import]

            if hasattr(_libtriton, "kgir"):
                return _libtriton.kgir.graph_from_mlir(mlir_text)
        except (ImportError, Exception):
            pass

        return cls._from_mlir_python(mlir_text)

    # -- Pure-Python MLIR generation (fallback) -------------------------------

    def _to_mlir_python(self) -> str:
        """Generate a simplified MLIR text representation in pure Python.

        The output uses ``ttkgir`` dialect operations and is designed to be
        parseable by ``_from_mlir_python`` for testing round-trips.
        """
        lines: List[str] = []
        lines.append('module attributes {ttkgir.kernel_graph} {')

        # Emit hardware profiles as module-level attributes
        for idx, hp in enumerate(self._hardware_profiles):
            lines.append(
                f'  // hw_profile[{idx}]: vendor="{hp.vendor}", '
                f'arch="{hp.arch_generation}", '
                f'sm_count={hp.sm_count}, '
                f'smem={hp.smem_per_sm_bytes}, '
                f'regs={hp.registers_per_sm}, '
                f'gmem={hp.global_memory_bytes}, '
                f'bw={hp.memory_bandwidth_gbps}, '
                f'tflops={hp.compute_throughput_tflops}, '
                f'warp_size={hp.warp_size}, '
                f'streams={hp.max_concurrent_streams}, '
                f'interconnect="{hp.interconnect_type}", '
                f'interconnect_bw={hp.interconnect_bandwidth_gbps}'
            )

        # Emit nodes
        topo_order = self.topological_sort() if self._nodes else []
        for nid in topo_order:
            node = self._nodes[nid]
            meta = node.metadata
            fused_tag = ""
            if node.is_fused and node.fused_from:
                fused_tag = f', fused_from=[{",".join(str(x) for x in node.fused_from)}]'
            lines.append(
                f'  ttkgir.kernel_launch @node_{nid} '
                f'{{grid = [{", ".join(str(d) for d in meta.grid_dimensions)}], '
                f'smem = {meta.shared_memory_bytes}, '
                f'regs = {meta.register_count}, '
                f'warps = {meta.num_warps}'
                f'{fused_tag}}}'
            )

        # Emit edges
        for edge in self._edges:
            tensor_attr = ""
            if edge.tensor_id is not None:
                tensor_attr = f", tensor_id = {edge.tensor_id}"
            lines.append(
                f'  ttkgir.{edge.edge_type} '
                f'@node_{edge.source_id} -> @node_{edge.target_id}'
                f'{{{tensor_attr}}}'
            )

        lines.append('}')
        return '\n'.join(lines)

    @classmethod
    def _from_mlir_python(cls, mlir_text: str) -> KGIRGraph:
        """Parse the simplified MLIR text produced by ``_to_mlir_python``.

        This parser handles the subset of MLIR syntax emitted by the Python
        fallback serializer.  It is not a general-purpose MLIR parser.
        """
        graph = cls()

        # Map from MLIR node names (@node_N) to actual node IDs
        node_name_to_id: Dict[str, int] = {}

        # Parse kernel_launch lines
        launch_pattern = re.compile(
            r'ttkgir\.kernel_launch\s+@(\w+)\s*\{'
            r'grid\s*=\s*\[([^\]]*)\],\s*'
            r'smem\s*=\s*(\d+),\s*'
            r'regs\s*=\s*(\d+),\s*'
            r'warps\s*=\s*(\d+)'
            r'(?:,\s*fused_from=\[([^\]]*)\])?'
            r'\}'
        )

        # Parse edge lines
        edge_pattern = re.compile(
            r'ttkgir\.(\w+)\s+@(\w+)\s*->\s*@(\w+)'
            r'(?:\{(?:,?\s*tensor_id\s*=\s*(\d+))?\})?'
        )

        for line in mlir_text.splitlines():
            stripped = line.strip()

            # Kernel launch
            launch_match = launch_pattern.search(stripped)
            if launch_match:
                node_name = launch_match.group(1)
                grid_str = launch_match.group(2)
                smem = int(launch_match.group(3))
                regs = int(launch_match.group(4))
                warps = int(launch_match.group(5))
                fused_from_str = launch_match.group(6)

                grid = tuple(int(x.strip()) for x in grid_str.split(",") if x.strip())
                fused_from = None
                if fused_from_str:
                    fused_from = [int(x.strip()) for x in fused_from_str.split(",") if x.strip()]

                metadata = NodeMetadata(
                    grid_dimensions=grid,
                    shared_memory_bytes=smem,
                    register_count=regs,
                    num_warps=warps,
                )
                nid = graph.add_node(kernel_fn=None, metadata=metadata)
                node_name_to_id[node_name] = nid

                if fused_from:
                    graph._nodes[nid].is_fused = True
                    graph._nodes[nid].fused_from = fused_from
                continue

            # Edge
            edge_match = edge_pattern.search(stripped)
            if edge_match:
                edge_type = edge_match.group(1)
                src_name = edge_match.group(2)
                tgt_name = edge_match.group(3)
                tensor_id_str = edge_match.group(4)

                src_id = node_name_to_id.get(src_name)
                tgt_id = node_name_to_id.get(tgt_name)
                tensor_id = int(tensor_id_str) if tensor_id_str else None

                if src_id is not None and tgt_id is not None:
                    if edge_type in VALID_EDGE_TYPES:
                        graph.add_edge(
                            source_id=src_id,
                            target_id=tgt_id,
                            edge_type=edge_type,
                            tensor_id=tensor_id,
                        )

        return graph

    # -- Internal helpers -----------------------------------------------------

    def _would_create_cycle(self, source_id: int, target_id: int) -> bool:
        """Check if adding ``source_id → target_id`` would create a cycle.

        Performs a BFS from *target_id* along existing forward edges.  If
        *source_id* is reachable from *target_id*, the proposed edge would
        close a cycle.

        Complexity: O(V + E) worst case, but typically much faster for sparse
        graphs because the search terminates early.
        """
        # If target can reach source via existing edges, adding source→target
        # creates a cycle.
        from collections import deque

        visited: set = set()
        queue: deque = deque([target_id])

        while queue:
            current = queue.popleft()
            if current == source_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            for succ in self._adjacency.get(current, []):
                if succ not in visited:
                    queue.append(succ)

        return False

    def __repr__(self) -> str:
        return (
            f"KGIRGraph(nodes={self.node_count()}, "
            f"edges={self.edge_count()}, "
            f"hw_profiles={len(self._hardware_profiles)})"
        )

    def __len__(self) -> int:
        return self.node_count()
