"""Memory planning pass for the Triton graph-level optimization layer.

Analyzes the KGIR to identify intermediate tensors produced and consumed
within the graph, performs liveness analysis, promotes eligible intermediates
from global memory to shared memory or register file across fused kernels
(respecting per-target hardware limits from ``HardwareProfile``), inserts
cross-device transfer operations when dispatch splits graphs across devices,
and refines promotion decisions via runtime feedback.

Classes
-------
TransferOp
    Dataclass describing a cross-device data transfer operation.
MemoryPlanner
    Full memory planning pass operating on a ``KGIRGraph``.

Performance Constraints (AAP §0.7.2)
-------------------------------------
- Producer-consumer fusion MUST eliminate ≥ 80 % of identified redundant
  global-memory round-trips between fusible kernel pairs.
- KGIR memory overhead MUST be < 10 MB for graphs with ≤ 100 kernels.
- Cross-device transfers MUST be correctly synchronised (AAP §0.7.3).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from .kgir import (
    KGIRGraph,
    KGIRNode,
    KGIREdge,
    HardwareProfile,
    NodeMetadata,
)
from .config import GraphConfig
from .utils import topological_sort, compute_tensor_size_bytes
from .errors import TransferError

if TYPE_CHECKING:
    from triton.backends.compiler import GPUTarget

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TransferOp — cross-device data transfer descriptor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferOp:
    """Describes a cross-device data transfer operation inserted by the
    memory planner when a dispatch plan assigns producer and consumer
    kernels to different devices.

    Attributes
    ----------
    tensor_id : int
        Index of the tensor argument being transferred.
    source_device : GPUTarget
        Origin device for the transfer.
    target_device : GPUTarget
        Destination device for the transfer.
    size_bytes : int
        Size of the transfer payload in bytes.
    transfer_type : str
        One of ``"peer_to_peer"`` (direct GPU-to-GPU, e.g. NVLink) or
        ``"host_staged"`` (GPU → host → GPU, required for cross-vendor
        transfers or when peer access is unavailable).
    estimated_time_ms : float
        Estimated transfer wall-clock time in milliseconds, computed from
        ``size_bytes`` and the interconnect bandwidth in the source/target
        ``HardwareProfile``.
    """

    tensor_id: int
    source_device: GPUTarget
    target_device: GPUTarget
    size_bytes: int
    transfer_type: str = "host_staged"
    estimated_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Constants governing promotion heuristics
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum tensor element count considered for register promotion.
# Tensors larger than this are never promoted to registers — they would
# cause catastrophic spilling.
_MAX_REGISTER_ELEMENTS: int = 64

# Fraction of per-SM/CU shared memory that the planner may allocate to
# promoted intermediates.  The remainder is reserved for the kernels'
# own dynamic shared memory.
_SMEM_PROMOTION_BUDGET_RATIO: float = 0.50

# Fraction of per-SM/CU register file that the planner may allocate to
# promoted intermediates.  The remainder is reserved for the kernels'
# own register usage.
_REGISTER_PROMOTION_BUDGET_RATIO: float = 0.30

# Occupancy degradation threshold (fraction below expected).  If actual
# occupancy drops by more than this fraction relative to the expected
# value, the promotion is flagged for revert.
_OCCUPANCY_DEGRADATION_THRESHOLD: float = 0.15

# PCIe Gen 4 bandwidth used as a conservative fallback when the
# interconnect bandwidth is not specified (in GB/s).
_DEFAULT_PCIE_BANDWIDTH_GBPS: float = 25.0

# Host staging overhead multiplier: host-staged transfers traverse the
# link twice (device→host + host→device), so the effective bandwidth is
# halved relative to the one-way interconnect bandwidth.
_HOST_STAGING_OVERHEAD: float = 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryPlanner — full memory planning pass
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryPlanner:
    """Analyzes KGIR for memory optimization: liveness analysis, promotion,
    and transfer insertion.

    The planner operates in six logical phases:

    1. **Intermediate identification** — find tensors produced and consumed
       entirely within the traced graph.
    2. **Liveness analysis** — compute per-tensor birth/death intervals in
       topological order.
    3. **Memory promotion** — decide whether each intermediate can be
       promoted from global memory to shared memory (SMEM) or the register
       file, respecting per-target hardware limits from ``HardwareProfile``.
    4. **Cross-device transfer insertion** — when the dispatch plan assigns
       producer and consumer kernels to different devices, insert
       ``TransferOp`` descriptors into the plan.
    5. **Closed-loop refinement** — after runtime profiling, revert any
       promotion decision that caused occupancy degradation.
    6. **Plan application** — annotate KGIR nodes with the final promotion
       decisions and insert transfer edges.

    Parameters
    ----------
    graph : KGIRGraph
        The KGIR graph with fusion decisions already applied.
    config : GraphConfig
        Top-level configuration for the graph optimisation layer.
    """

    def __init__(self, graph: KGIRGraph, config: GraphConfig) -> None:
        self._graph: KGIRGraph = graph
        self._config: GraphConfig = config

        # Cached analysis results (populated lazily).
        self._intermediates: Optional[List[int]] = None
        self._liveness: Optional[Dict[int, Tuple[int, int]]] = None

        # Per-target promotion plans keyed by target identifier string.
        self._promotion_plans: Dict[str, Dict[int, str]] = {}

        # Accumulated transfer operations.
        self._transfers: List[TransferOp] = []

        logger.debug(
            "MemoryPlanner initialised for graph with %d nodes.",
            graph.node_count(),
        )

    # ------------------------------------------------------------------
    # Phase 1: Intermediate tensor identification
    # ------------------------------------------------------------------

    def identify_intermediates(self) -> List[int]:
        """Find tensor IDs that are produced and consumed entirely within
        the graph — these are the candidates for promotion from global
        memory.

        A tensor is **intermediate** if:
        - It is written (produced) by at least one node inside the graph.
        - It is read (consumed) by at least one other node inside the graph.
        - It is *not* a graph input (not produced by a root with no
          in-graph producer).
        - It is *not* a graph output (not consumed only by a leaf with no
          further in-graph consumer, or by a node whose successor is
          outside the graph).

        Returns
        -------
        List[int]
            Sorted list of intermediate tensor IDs.
        """
        if self._intermediates is not None:
            return list(self._intermediates)

        graph = self._graph
        edges = graph.get_edges()

        # Collect tensor IDs with their producer and consumer node sets.
        # A tensor_id on a data_dep edge means:
        #   source_id *produces* the tensor, target_id *consumes* it.
        producers: Dict[int, set] = {}  # tensor_id -> set of producing node IDs
        consumers: Dict[int, set] = {}  # tensor_id -> set of consuming node IDs

        for edge in edges:
            if edge.tensor_id is None:
                continue
            if edge.edge_type not in ("data_dep", "cross_device_transfer"):
                continue

            tid = edge.tensor_id
            producers.setdefault(tid, set()).add(edge.source_id)
            consumers.setdefault(tid, set()).add(edge.target_id)

        root_set = set(graph.get_roots())
        leaf_set = set(graph.get_leaves())

        intermediates: List[int] = []
        for tid in sorted(producers.keys()):
            # Must have at least one consumer.
            if tid not in consumers or not consumers[tid]:
                continue

            # All producers must be graph-internal (not exclusively graph
            # inputs with no in-graph predecessor producing the tensor).
            prod_nodes = producers[tid]

            # A tensor is a *graph input* if all its producers are roots
            # and the tensor itself has no other in-graph producer edge.
            # We allow intermediates produced by roots as long as the
            # tensor is also consumed internally.
            all_prods_are_roots = all(p in root_set for p in prod_nodes)

            # A tensor is a *graph output* if all its consumers are leaves
            # and nothing else consumes it inside the graph.
            cons_nodes = consumers[tid]
            all_cons_are_leaves = all(c in leaf_set for c in cons_nodes)

            # Use get_successors/get_predecessors to determine if the
            # tensor's producer has in-graph successors beyond just the
            # consumer (confirming it is internal, not a pass-through).
            has_internal_path = False
            for pid in prod_nodes:
                successors = graph.get_successors(pid)
                if successors:
                    has_internal_path = True
                    break
            for cid in cons_nodes:
                predecessors = graph.get_predecessors(cid)
                if predecessors:
                    has_internal_path = True
                    break

            # Exclude pure graph inputs: produced *only* by root(s) with
            # no other in-graph consumer relationship that qualifies it
            # as intermediate.  The heuristic: if the producer set is a
            # strict subset of the roots AND no non-root node also
            # produces the tensor, it's an input.
            # However, if a root produces a tensor consumed by a non-leaf,
            # it qualifies as intermediate.
            if all_prods_are_roots and all_cons_are_leaves and not has_internal_path:
                # Pure pass-through: root → leaf only.  Not intermediate.
                continue

            intermediates.append(tid)

        self._intermediates = intermediates
        logger.debug(
            "Identified %d intermediate tensors: %s",
            len(intermediates),
            intermediates,
        )
        return list(intermediates)

    # ------------------------------------------------------------------
    # Phase 2: Liveness analysis
    # ------------------------------------------------------------------

    def compute_liveness(self) -> Dict[int, Tuple[int, int]]:
        """Compute liveness intervals for all intermediate tensors.

        For each intermediate tensor, the liveness interval is defined as
        ``(birth_step, death_step)`` where:

        - *birth_step* is the topological-order position of the first node
          that produces the tensor.
        - *death_step* is the topological-order position of the last node
          that consumes the tensor.

        Returns
        -------
        Dict[int, Tuple[int, int]]
            ``{tensor_id: (birth_step, death_step)}``.
        """
        if self._liveness is not None:
            return dict(self._liveness)

        graph = self._graph
        intermediates_set = set(self.identify_intermediates())

        # Build topological ordering and position map.
        topo_order = graph.topological_sort()
        position: Dict[int, int] = {nid: idx for idx, nid in enumerate(topo_order)}

        edges = graph.get_edges()

        # For each intermediate tensor, find earliest producer and latest
        # consumer in topological order.
        birth: Dict[int, int] = {}
        death: Dict[int, int] = {}

        for edge in edges:
            if edge.tensor_id is None:
                continue
            tid = edge.tensor_id
            if tid not in intermediates_set:
                continue

            src_pos = position.get(edge.source_id)
            tgt_pos = position.get(edge.target_id)
            if src_pos is None or tgt_pos is None:
                continue

            # Update birth (earliest producer position).
            if tid not in birth or src_pos < birth[tid]:
                birth[tid] = src_pos

            # Update death (latest consumer position).
            if tid not in death or tgt_pos > death[tid]:
                death[tid] = tgt_pos

        # Assemble interval map.
        liveness: Dict[int, Tuple[int, int]] = {}
        for tid in sorted(intermediates_set):
            b = birth.get(tid, 0)
            d = death.get(tid, b)
            liveness[tid] = (b, d)

        self._liveness = liveness
        logger.debug(
            "Liveness intervals computed for %d tensors.",
            len(liveness),
        )
        return dict(liveness)

    # ------------------------------------------------------------------
    # Phase 3: Memory promotion
    # ------------------------------------------------------------------

    def plan_promotions(self, target: HardwareProfile) -> Dict[int, str]:
        """Decide on memory promotion for each intermediate tensor for
        a specific hardware *target*.

        Promotion options:
        - ``"shared"`` — promote from global memory to per-SM/CU shared
          memory.  Requires the tensor to fit within the SMEM budget and
          for producer/consumer to be fused (or schedulable on the same SM).
        - ``"register"`` — promote to the register file.  Only for very
          small tensors with a compatible access pattern.
        - ``"global"`` — keep in global memory (no promotion).

        Parameters
        ----------
        target : HardwareProfile
            Hardware profile describing per-SM/CU resource limits.

        Returns
        -------
        Dict[int, str]
            ``{tensor_id: promotion_type}`` where *promotion_type* is one
            of ``"shared"``, ``"register"``, or ``"global"``.
        """
        graph = self._graph
        intermediates = self.identify_intermediates()
        liveness = self.compute_liveness()
        edges = graph.get_edges()

        # Build a mapping: tensor_id -> (producer_node_ids, consumer_node_ids)
        tensor_producers: Dict[int, set] = {}
        tensor_consumers: Dict[int, set] = {}
        for edge in edges:
            if edge.tensor_id is None:
                continue
            tid = edge.tensor_id
            tensor_producers.setdefault(tid, set()).add(edge.source_id)
            tensor_consumers.setdefault(tid, set()).add(edge.target_id)

        # Access fusion config for promotion threshold: only promote
        # if the estimated benefit exceeds the fusion threshold ratio.
        fusion_threshold = float(self._config.fusion.threshold)

        promotions: Dict[int, str] = {}

        for tid in intermediates:
            # Get tensor metadata from its producer node.
            prod_ids = tensor_producers.get(tid, set())
            cons_ids = tensor_consumers.get(tid, set())

            # Attempt to determine tensor shape and dtype from node metadata.
            shape, dtype = self._get_tensor_shape_dtype(tid, prod_ids)
            if shape is None or dtype is None:
                # Cannot determine tensor dimensions — keep in global.
                promotions[tid] = "global"
                continue

            tensor_size = compute_tensor_size_bytes(shape, dtype)
            if tensor_size <= 0:
                promotions[tid] = "global"
                continue

            # Check register promotion first (small tensors only).
            element_count = math.prod(shape) if shape else 0
            if element_count <= _MAX_REGISTER_ELEMENTS and element_count > 0:
                # Check if all producers and consumers are fused together
                # (register promotion requires same-kernel access).
                if self._are_all_fused_together(prod_ids | cons_ids):
                    promotions[tid] = "register"
                    continue

            # Check shared memory promotion.
            # Requirements:
            # 1. Tensor size fits within the available SMEM budget.
            # 2. Producer and consumer are fused (or at least share a
            #    scheduling group so they execute on the same SM/CU).
            # 3. Estimated benefit ratio exceeds the fusion_threshold.
            smem_budget = int(target.smem_per_sm_bytes * _SMEM_PROMOTION_BUDGET_RATIO)
            if tensor_size <= smem_budget:
                # Check if producer-consumer are fused (strongest
                # criterion for SMEM promotion benefit).
                if self._are_producer_consumer_fused(prod_ids, cons_ids):
                    # Estimate the benefit of promotion: ratio of
                    # eliminated global memory bytes to total tensor
                    # size.  Must exceed the fusion threshold to be
                    # worthwhile.
                    benefit_ratio = 1.0  # Full elimination of global round-trip.
                    if benefit_ratio >= fusion_threshold:
                        promotions[tid] = "shared"
                        continue

            # Default: keep in global memory.
            promotions[tid] = "global"

        # Budget validation: ensure aggregate shared and register
        # promotions fit within per-target limits.
        if not self._check_smem_budget(promotions, target):
            self._downgrade_promotions_to_fit(promotions, target, "shared")

        if not self._check_register_budget(promotions, target):
            self._downgrade_promotions_to_fit(promotions, target, "register")

        target_key = self._target_key(target)
        self._promotion_plans[target_key] = dict(promotions)
        logger.debug(
            "Promotion plan for target '%s': %s",
            target_key,
            {k: v for k, v in promotions.items() if v != "global"},
        )
        return dict(promotions)

    def _check_smem_budget(
        self,
        promotions: Dict[int, str],
        target: HardwareProfile,
    ) -> bool:
        """Verify that the total shared-memory usage from promoted tensors
        does not exceed the per-SM/CU SMEM budget.

        The budget accounts for existing SMEM usage by the kernels
        themselves (from ``NodeMetadata.shared_memory_bytes``).

        Parameters
        ----------
        promotions : Dict[int, str]
            Current promotion decisions.
        target : HardwareProfile
            Hardware profile with SMEM capacity.

        Returns
        -------
        bool
            ``True`` if within budget, ``False`` if over-committed.
        """
        graph = self._graph
        edges = graph.get_edges()

        # Compute per-fused-group SMEM requirements.
        # Group promotions by the set of nodes that produce/consume them.
        smem_total = self._compute_promotion_size(promotions, "shared")

        # Existing kernel SMEM usage across all nodes (using get_resource_usage).
        existing_smem = 0
        topo = graph.topological_sort()
        for nid in topo:
            node = graph.get_node(nid)
            resource_usage = node.get_resource_usage()
            existing_smem = max(existing_smem, resource_usage.get("shared_memory_bytes", 0))

        available = target.smem_per_sm_bytes - existing_smem
        if available < 0:
            available = 0

        if smem_total > available:
            logger.debug(
                "SMEM budget exceeded: promoted=%d, available=%d (capacity=%d, kernel_use=%d)",
                smem_total,
                available,
                target.smem_per_sm_bytes,
                existing_smem,
            )
            return False
        return True

    def _check_register_budget(
        self,
        promotions: Dict[int, str],
        target: HardwareProfile,
    ) -> bool:
        """Verify that the total register usage from promoted tensors does
        not exceed the per-SM/CU register budget.

        Parameters
        ----------
        promotions : Dict[int, str]
            Current promotion decisions.
        target : HardwareProfile
            Hardware profile with register capacity.

        Returns
        -------
        bool
            ``True`` if within budget, ``False`` if over-committed.
        """
        graph = self._graph

        reg_total = self._compute_promotion_size(promotions, "register")

        # Convert byte size to approximate register count.
        # Each register is typically 4 bytes (32-bit).
        reg_count_promoted = (reg_total + 3) // 4  # ceil division

        # Existing register pressure: max across all nodes (using get_resource_usage).
        existing_regs = 0
        topo = graph.topological_sort()
        for nid in topo:
            node = graph.get_node(nid)
            resource_usage = node.get_resource_usage()
            existing_regs = max(existing_regs, resource_usage.get("register_count", 0))

        # Account for warp/wavefront width: registers are allocated per
        # warp/wavefront.  When a tensor is promoted to registers, each
        # thread in the warp holds a portion, so the effective register
        # consumption per SM depends on warp granularity.
        warp_size = target.warp_size
        budget = int(target.registers_per_sm * _REGISTER_PROMOTION_BUDGET_RATIO)
        # Promoted registers per SM = promoted_regs_per_thread * warps_per_sm.
        # The total per-SM budget already factors in all resident warps,
        # so we scale the promoted count by the warp width to convert from
        # aggregate bytes to per-thread register equivalents.
        if warp_size > 0:
            reg_count_promoted = max(1, (reg_count_promoted + warp_size - 1) // warp_size)
        available = budget - existing_regs
        if available < 0:
            available = 0

        if reg_count_promoted > available:
            logger.debug(
                "Register budget exceeded: promoted_regs=%d, available=%d "
                "(budget=%d, kernel_use=%d)",
                reg_count_promoted,
                available,
                budget,
                existing_regs,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Phase 4: Cross-device transfer insertion
    # ------------------------------------------------------------------

    def insert_transfers(
        self,
        dispatch_plan: Dict[int, GPUTarget],
    ) -> List[TransferOp]:
        """Analyze the dispatch plan and insert ``TransferOp`` descriptors
        for every data dependency that crosses a device boundary.

        Transfer type selection:
        - **Intra-vendor** with high-bandwidth interconnect (NVLink,
          Infinity Fabric): ``"peer_to_peer"``.
        - **Cross-vendor** or no high-bandwidth interconnect:
          ``"host_staged"`` (device A → host RAM → device B).

        Parameters
        ----------
        dispatch_plan : Dict[int, GPUTarget]
            Mapping from node ID to the ``GPUTarget`` the node is
            dispatched to.

        Returns
        -------
        List[TransferOp]
            Transfer operations to be added to the execution plan.

        Raises
        ------
        TransferError
            If a required transfer cannot be established (e.g. no viable
            interconnect path).
        """
        graph = self._graph
        edges = graph.get_edges()
        # Build a profile lookup keyed by GPUTarget object identity (id())
        # so that two physical GPUs of the same architecture are not
        # collapsed into a single entry.
        profiles_by_id: Dict[int, Any] = {
            id(hp.gpu_target): hp
            for hp in graph.hardware_profiles
            if hp.gpu_target is not None
        }

        # Dispatch configuration informs transfer strategy logging.
        dispatch_config = self._config.dispatch
        if dispatch_config.log:
            logger.info(
                "Inserting transfers for dispatch plan with %d nodes, "
                "mode=%s, granularity=%s",
                len(dispatch_plan),
                dispatch_config.mode,
                dispatch_config.granularity,
            )

        transfers: List[TransferOp] = []

        for edge in edges:
            if edge.tensor_id is None:
                continue
            if edge.edge_type not in ("data_dep", "cross_device_transfer"):
                continue

            src_target = dispatch_plan.get(edge.source_id)
            tgt_target = dispatch_plan.get(edge.target_id)

            if src_target is None or tgt_target is None:
                continue

            # Same device — no transfer needed.
            if self._same_device(src_target, tgt_target):
                continue

            # Determine tensor size.
            tensor_size = self._estimate_tensor_size(edge.tensor_id, edge.source_id)

            # Determine transfer type — look up by GPUTarget object identity.
            src_profile = profiles_by_id.get(id(src_target))
            tgt_profile = profiles_by_id.get(id(tgt_target))

            transfer_type = self._select_transfer_type(
                src_target, tgt_target, src_profile, tgt_profile,
            )

            # Estimate transfer time.
            estimated_time = self._estimate_transfer_time(
                tensor_size, transfer_type, src_profile, tgt_profile,
            )

            transfer = TransferOp(
                tensor_id=edge.tensor_id,
                source_device=src_target,
                target_device=tgt_target,
                size_bytes=tensor_size,
                transfer_type=transfer_type,
                estimated_time_ms=estimated_time,
            )
            transfers.append(transfer)
            logger.debug(
                "Transfer inserted: tensor=%d, %s -> %s, type=%s, "
                "size=%d bytes, est_time=%.3f ms",
                edge.tensor_id,
                self._gpu_target_key(src_target),
                self._gpu_target_key(tgt_target),
                transfer_type,
                tensor_size,
                estimated_time,
            )

        self._transfers = transfers
        return list(transfers)

    # ------------------------------------------------------------------
    # Phase 5: Closed-loop refinement
    # ------------------------------------------------------------------

    def refine_with_feedback(
        self,
        measured_metrics: Dict,
    ) -> Dict[int, str]:
        """Refine promotion decisions using runtime profiling data.

        After the profiler has collected per-kernel per-target metrics,
        this method checks whether any promotion decision degraded
        performance.  Degrading promotions are reverted to ``"global"``.

        Parameters
        ----------
        measured_metrics : Dict
            Profiling data keyed by node ID or target identifier.
            Expected keys per node: ``"occupancy"``,
            ``"expected_occupancy"``, ``"register_spill_bytes"``,
            ``"wall_clock_ms"``.

        Returns
        -------
        Dict[int, str]
            Updated (potentially downgraded) promotion decisions.
            Combines all per-target plans into a unified view.
        """
        if not self._promotion_plans:
            logger.debug("No promotion plans to refine.")
            return {}

        # Check if feedback refinement is enabled via config.
        if not self._config.feedback.enable:
            logger.debug(
                "Feedback refinement is disabled — returning current plans.",
            )
            all_plans: Dict[int, str] = {}
            for _tk, promo in self._promotion_plans.items():
                all_plans.update(promo)
            return all_plans

        updated_plans: Dict[int, str] = {}

        for target_key, promotions in self._promotion_plans.items():
            degraded_tensors = self._detect_occupancy_degradation(
                measured_metrics, promotions,
            )

            if degraded_tensors:
                logger.debug(
                    "Reverting promotions for tensors %s on target '%s' "
                    "due to occupancy degradation.",
                    degraded_tensors,
                    target_key,
                )
                for tid in degraded_tensors:
                    promotions[tid] = "global"

            # Also check for register spilling.
            spill_reverts = self._detect_register_spill(
                measured_metrics, promotions,
            )
            if spill_reverts:
                logger.debug(
                    "Reverting register promotions for tensors %s on "
                    "target '%s' due to register spilling.",
                    spill_reverts,
                    target_key,
                )
                for tid in spill_reverts:
                    promotions[tid] = "global"

            self._promotion_plans[target_key] = promotions
            updated_plans.update(promotions)

        return updated_plans

    def _detect_occupancy_degradation(
        self,
        metrics: Dict,
        promotions: Dict[int, str],
    ) -> List[int]:
        """Compare expected vs actual occupancy and identify tensor IDs
        whose promotions likely caused degradation.

        The method inspects KGIR node performance annotations (written
        back by the feedback controller) and cross-references with the
        promotion decisions.

        Parameters
        ----------
        metrics : Dict
            Profiling data.  Expected structure is either:
            - ``{node_id: {"occupancy": float, "expected_occupancy": float, ...}}``
            - or a flat dict with ``"occupancy"`` and ``"expected_occupancy"`` keys.
        promotions : Dict[int, str]
            Current promotion decisions.

        Returns
        -------
        List[int]
            Tensor IDs whose SMEM promotions should be reverted.
        """
        graph = self._graph
        edges = graph.get_edges()

        # Build: tensor_id -> set of consuming node IDs.
        tensor_consumers: Dict[int, set] = {}
        for edge in edges:
            if edge.tensor_id is not None:
                tensor_consumers.setdefault(edge.tensor_id, set()).add(
                    edge.target_id,
                )

        degraded: List[int] = []

        for tid, promo in promotions.items():
            if promo != "shared":
                continue

            consumer_nodes = tensor_consumers.get(tid, set())
            for nid in consumer_nodes:
                node_metrics = self._get_node_metrics(nid, metrics)
                if node_metrics is None:
                    continue

                actual_occ = node_metrics.get("occupancy")
                expected_occ = node_metrics.get("expected_occupancy")

                if actual_occ is None or expected_occ is None:
                    continue
                if expected_occ <= 0:
                    continue

                ratio = actual_occ / expected_occ
                if ratio < (1.0 - _OCCUPANCY_DEGRADATION_THRESHOLD):
                    degraded.append(tid)
                    break  # One degraded consumer is enough.

        return sorted(set(degraded))

    # ------------------------------------------------------------------
    # Phase 6: Plan application
    # ------------------------------------------------------------------

    def apply_plan(self, graph: KGIRGraph) -> KGIRGraph:
        """Annotate KGIR nodes with promotion decisions and insert transfer
        operations into the graph.

        This mutates *graph* in-place (KGIR nodes are mutable per
        AAP §0.1.1) and returns the same graph object.

        Parameters
        ----------
        graph : KGIRGraph
            The target graph to annotate.

        Returns
        -------
        KGIRGraph
            The same graph instance with updated metadata.
        """
        # 1. Apply promotion annotations to nodes.
        edges = graph.get_edges()
        tensor_to_nodes: Dict[int, set] = {}
        for edge in edges:
            if edge.tensor_id is not None:
                tensor_to_nodes.setdefault(edge.tensor_id, set()).add(
                    edge.source_id,
                )
                tensor_to_nodes[edge.tensor_id].add(edge.target_id)

        for target_key, promotions in self._promotion_plans.items():
            for tid, promo in promotions.items():
                if promo == "global":
                    continue
                node_ids = tensor_to_nodes.get(tid, set())
                for nid in node_ids:
                    try:
                        node = graph.get_node(nid)
                    except Exception:
                        continue

                    # Update the memory access patterns to reflect the
                    # promotion decision.
                    patterns = node.metadata.memory_access_patterns
                    promo_key = f"tensor_{tid}_promotion"
                    patterns[promo_key] = {
                        "promotion": promo,
                        "target": target_key,
                    }

                    # If promoting to shared memory, update the
                    # shared_memory_bytes estimate.
                    if promo == "shared":
                        shape, dtype = self._get_tensor_shape_dtype(
                            tid, {nid},
                        )
                        if shape is not None and dtype is not None:
                            extra_smem = compute_tensor_size_bytes(
                                shape, dtype,
                            )
                            node.metadata.shared_memory_bytes += extra_smem

                    # Write target-specific annotation.
                    existing_ann = node.get_performance_annotation(target_key)
                    ann = dict(existing_ann) if existing_ann else {}
                    ann[f"promotion_{tid}"] = promo
                    node.update_performance_annotation(target_key, ann)

        # 2. Insert transfer operations as new nodes and edges in the graph.
        # Each TransferOp becomes a lightweight transfer node sitting
        # between the producer and consumer, with explicit
        # cross_device_transfer edges.
        for transfer in self._transfers:
            src_nid = self._find_producer_node(
                transfer.tensor_id, graph,
            )
            tgt_nid = self._find_consumer_node(
                transfer.tensor_id, graph,
            )
            if src_nid is not None and tgt_nid is not None:
                try:
                    # Create a transfer node with metadata describing the
                    # cross-device operation.
                    transfer_metadata = NodeMetadata(
                        memory_access_patterns={
                            "transfer": True,
                            "transfer_type": transfer.transfer_type,
                        },
                        tensor_shapes={},
                        tensor_strides={},
                        tensor_dtypes={},
                        grid_dimensions=(1,),
                        shared_memory_bytes=0,
                        register_count=0,
                        num_warps=1,
                        hardware_target_annotations={
                            "source_device": self._gpu_target_key(
                                transfer.source_device,
                            ),
                            "target_device": self._gpu_target_key(
                                transfer.target_device,
                            ),
                        },
                        runtime_performance_annotations={},
                    )
                    transfer_nid = graph.add_node(
                        kernel_fn=None,
                        metadata=transfer_metadata,
                    )

                    # Edge: producer -> transfer node.
                    graph.add_edge(
                        source_id=src_nid,
                        target_id=transfer_nid,
                        edge_type="cross_device_transfer",
                        tensor_id=transfer.tensor_id,
                        metadata={
                            "transfer_type": transfer.transfer_type,
                            "size_bytes": transfer.size_bytes,
                            "estimated_time_ms": transfer.estimated_time_ms,
                        },
                    )
                    # Edge: transfer node -> consumer.
                    graph.add_edge(
                        source_id=transfer_nid,
                        target_id=tgt_nid,
                        edge_type="cross_device_transfer",
                        tensor_id=transfer.tensor_id,
                        metadata={
                            "transfer_type": transfer.transfer_type,
                            "size_bytes": transfer.size_bytes,
                            "estimated_time_ms": transfer.estimated_time_ms,
                        },
                    )
                except Exception as exc:
                    logger.debug(
                        "Could not add transfer node/edge for tensor %d: %s",
                        transfer.tensor_id,
                        exc,
                    )

        if self._config.dump_kgir:
            logger.info("Memory plan applied to KGIR graph.")

        return graph

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════

    def _get_tensor_shape_dtype(
        self,
        tensor_id: int,
        candidate_node_ids: set,
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
        """Retrieve the shape and dtype of a tensor from its producer
        node metadata.

        Iterates over *candidate_node_ids* and returns the first match
        found in ``NodeMetadata.tensor_shapes`` / ``tensor_dtypes``.

        Returns
        -------
        Tuple[Optional[Tuple[int, ...]], Optional[str]]
            ``(shape, dtype)`` or ``(None, None)`` if not found.
        """
        graph = self._graph
        for nid in sorted(candidate_node_ids):
            try:
                node = graph.get_node(nid)
            except Exception:
                continue
            meta = node.get_metadata()
            shape = meta.tensor_shapes.get(tensor_id)
            dtype = meta.tensor_dtypes.get(tensor_id)
            if shape is not None and dtype is not None:
                return (shape, dtype)
        return (None, None)

    def _are_all_fused_together(self, node_ids: set) -> bool:
        """Check if all given nodes belong to the same fused kernel."""
        if not node_ids:
            return False
        graph = self._graph

        # Collect the fused-group identifier for each node.
        fused_groups: set = set()
        for nid in node_ids:
            try:
                node = graph.get_node(nid)
            except Exception:
                return False
            if node.is_fused:
                fused_groups.add(node.node_id)
            elif node.fused_from:
                # Node is a component of a fused kernel — should have
                # been replaced, but be defensive.
                fused_groups.add(tuple(sorted(node.fused_from)))
            else:
                fused_groups.add(nid)

        return len(fused_groups) == 1

    def _are_producer_consumer_fused(
        self,
        prod_ids: set,
        cons_ids: set,
    ) -> bool:
        """Check if at least one producer-consumer pair is in the same
        fused kernel group.
        """
        graph = self._graph
        for pid in prod_ids:
            try:
                pnode = graph.get_node(pid)
            except Exception:
                continue
            for cid in cons_ids:
                try:
                    cnode = graph.get_node(cid)
                except Exception:
                    continue

                # Both nodes are fused and share fused_from sets.
                if pnode.is_fused and cnode.is_fused:
                    if pnode.node_id == cnode.node_id:
                        return True
                    # Check overlap in fused_from.
                    pf = set(pnode.fused_from) if pnode.fused_from else set()
                    cf = set(cnode.fused_from) if cnode.fused_from else set()
                    if pf and cf and pf & cf:
                        return True

                # One is fused and contains the other.
                if pnode.is_fused and pnode.fused_from:
                    if cid in pnode.fused_from or cnode.node_id in pnode.fused_from:
                        return True
                if cnode.is_fused and cnode.fused_from:
                    if pid in cnode.fused_from or pnode.node_id in cnode.fused_from:
                        return True

                # Same node: trivially fused.
                if pid == cid:
                    return True

        return False

    def _compute_promotion_size(
        self,
        promotions: Dict[int, str],
        promotion_type: str,
    ) -> int:
        """Sum the total byte size of all tensors promoted to the given type.

        Parameters
        ----------
        promotions : Dict[int, str]
            Promotion decisions.
        promotion_type : str
            Filter for ``"shared"`` or ``"register"``.

        Returns
        -------
        int
            Total promoted bytes.
        """
        graph = self._graph
        edges = graph.get_edges()

        # Build tensor -> producer nodes.
        tensor_producers: Dict[int, set] = {}
        for edge in edges:
            if edge.tensor_id is not None:
                tensor_producers.setdefault(edge.tensor_id, set()).add(
                    edge.source_id,
                )

        total = 0
        for tid, promo in promotions.items():
            if promo != promotion_type:
                continue
            prod_ids = tensor_producers.get(tid, set())
            shape, dtype = self._get_tensor_shape_dtype(tid, prod_ids)
            if shape is not None and dtype is not None:
                total += compute_tensor_size_bytes(shape, dtype)
        return total

    def _downgrade_promotions_to_fit(
        self,
        promotions: Dict[int, str],
        target: HardwareProfile,
        promotion_type: str,
    ) -> None:
        """Iteratively downgrade promotions until the budget is satisfied.

        Tensors are downgraded in reverse order of size (largest first) to
        maximise the number of promotions that can survive.

        Parameters
        ----------
        promotions : Dict[int, str]
            Promotion decisions (mutated in-place).
        target : HardwareProfile
            Hardware profile with capacity limits.
        promotion_type : str
            ``"shared"`` or ``"register"``.
        """
        graph = self._graph
        edges = graph.get_edges()

        tensor_producers: Dict[int, set] = {}
        for edge in edges:
            if edge.tensor_id is not None:
                tensor_producers.setdefault(edge.tensor_id, set()).add(
                    edge.source_id,
                )

        # Build a list of (tensor_id, size) for tensors with the given promo.
        candidates: List[Tuple[int, int]] = []
        for tid, promo in promotions.items():
            if promo != promotion_type:
                continue
            prod_ids = tensor_producers.get(tid, set())
            shape, dtype = self._get_tensor_shape_dtype(tid, prod_ids)
            if shape is not None and dtype is not None:
                size = compute_tensor_size_bytes(shape, dtype)
                candidates.append((tid, size))

        # Sort by size descending (downgrade largest first).
        candidates.sort(key=lambda x: x[1], reverse=True)

        for tid, _size in candidates:
            promotions[tid] = "global"
            logger.debug(
                "Downgraded tensor %d from '%s' to 'global' to fit budget.",
                tid,
                promotion_type,
            )
            # Re-check budget.
            if promotion_type == "shared":
                if self._check_smem_budget(promotions, target):
                    break
            elif promotion_type == "register":
                if self._check_register_budget(promotions, target):
                    break

    def _estimate_tensor_size(self, tensor_id: int, producer_nid: int) -> int:
        """Estimate the byte size of a tensor from its producer's metadata."""
        shape, dtype = self._get_tensor_shape_dtype(tensor_id, {producer_nid})
        if shape is not None and dtype is not None:
            return compute_tensor_size_bytes(shape, dtype)
        # Fallback: 0 bytes (unknown tensor).
        return 0

    @staticmethod
    def _same_device(a: GPUTarget, b: GPUTarget) -> bool:
        """Check if two ``GPUTarget`` instances refer to the same physical device.

        Uses object identity (``is``) to distinguish two physical GPUs of
        the same architecture (e.g. 2× A100) whose ``backend`` and ``arch``
        fields are identical but represent distinct hardware.
        """
        return a is b

    @staticmethod
    def _gpu_target_key(target: Optional[GPUTarget]) -> str:
        """Generate a stable string key for a ``GPUTarget``."""
        if target is None:
            return "unknown"
        return f"{target.backend}:{target.arch}"

    @staticmethod
    def _target_key(profile: HardwareProfile) -> str:
        """Generate a stable string key from a ``HardwareProfile``."""
        return f"{profile.vendor}:{profile.arch_generation}"

    @staticmethod
    def _select_transfer_type(
        src_target: GPUTarget,
        tgt_target: GPUTarget,
        src_profile: Optional[HardwareProfile],
        tgt_profile: Optional[HardwareProfile],
    ) -> str:
        """Select the best transfer type between two devices.

        Rules:
        - **Cross-vendor** (different ``backend``) → ``"host_staged"``.
        - **Intra-vendor** with high-bandwidth link (NVLink, Infinity
          Fabric, XGMI) → ``"peer_to_peer"``.
        - **Intra-vendor** with only PCIe → ``"host_staged"``.
        """
        if src_target.backend != tgt_target.backend:
            return "host_staged"

        # Check for high-bandwidth interconnect.
        high_bw_keywords = {"nvlink", "infinity_fabric", "xgmi"}
        for profile in (src_profile, tgt_profile):
            if profile is not None:
                ic_type = profile.interconnect_type.lower()
                for kw in high_bw_keywords:
                    if kw in ic_type:
                        return "peer_to_peer"

        return "host_staged"

    @staticmethod
    def _estimate_transfer_time(
        size_bytes: int,
        transfer_type: str,
        src_profile: Optional[HardwareProfile],
        tgt_profile: Optional[HardwareProfile],
    ) -> float:
        """Estimate transfer wall-clock time in milliseconds.

        For ``"peer_to_peer"`` transfers, the estimate uses the
        interconnect bandwidth of the source device.  For ``"host_staged"``
        transfers, the effective bandwidth is halved (device→host +
        host→device traversal).
        """
        if size_bytes <= 0:
            return 0.0

        # Determine bandwidth (GB/s).
        bw_gbps = _DEFAULT_PCIE_BANDWIDTH_GBPS
        for profile in (src_profile, tgt_profile):
            if profile is not None and profile.interconnect_bandwidth_gbps > 0:
                bw_gbps = profile.interconnect_bandwidth_gbps
                break

        if transfer_type == "host_staged":
            bw_gbps = bw_gbps / _HOST_STAGING_OVERHEAD

        # Convert GB/s to bytes/ms: 1 GB/s = 10^9 bytes / 10^3 ms = 10^6 bytes/ms
        bytes_per_ms = bw_gbps * 1e6
        if bytes_per_ms <= 0:
            return 0.0
        return size_bytes / bytes_per_ms

    def _detect_register_spill(
        self,
        metrics: Dict,
        promotions: Dict[int, str],
    ) -> List[int]:
        """Identify register-promoted tensors whose promotion likely caused
        register spilling.

        Parameters
        ----------
        metrics : Dict
            Profiling data.
        promotions : Dict[int, str]
            Current promotion decisions.

        Returns
        -------
        List[int]
            Tensor IDs whose register promotions should be reverted.
        """
        graph = self._graph
        edges = graph.get_edges()

        tensor_consumers: Dict[int, set] = {}
        for edge in edges:
            if edge.tensor_id is not None:
                tensor_consumers.setdefault(edge.tensor_id, set()).add(
                    edge.target_id,
                )

        spill_reverts: List[int] = []
        for tid, promo in promotions.items():
            if promo != "register":
                continue
            consumer_nodes = tensor_consumers.get(tid, set())
            for nid in consumer_nodes:
                node_metrics = self._get_node_metrics(nid, metrics)
                if node_metrics is None:
                    continue
                spill_bytes = node_metrics.get("register_spill_bytes", 0)
                if spill_bytes > 0:
                    spill_reverts.append(tid)
                    break

        return sorted(set(spill_reverts))

    @staticmethod
    def _get_node_metrics(
        node_id: int,
        metrics: Dict,
    ) -> Optional[Dict]:
        """Safely extract per-node metrics from the profiling data.

        The profiling data may be structured as:
        - ``{node_id: {metric_name: value}}``
        - or a flat dict when there is only one node.

        Returns ``None`` if no metrics are found for the node.
        """
        if not metrics:
            return None

        # Try direct node_id lookup.
        if node_id in metrics:
            val = metrics[node_id]
            if isinstance(val, dict):
                return val

        # Try string key.
        str_key = str(node_id)
        if str_key in metrics:
            val = metrics[str_key]
            if isinstance(val, dict):
                return val

        # Flat structure: if any key is a known metric name, treat the
        # whole dict as the metrics for this node.
        known_keys = {"occupancy", "expected_occupancy", "register_spill_bytes", "wall_clock_ms"}
        if known_keys & set(metrics.keys()):
            return metrics

        return None

    def _find_producer_node(
        self,
        tensor_id: int,
        graph: KGIRGraph,
    ) -> Optional[int]:
        """Find the node ID that produces *tensor_id*."""
        for edge in graph.get_edges():
            if edge.tensor_id == tensor_id and edge.edge_type == "data_dep":
                return edge.source_id
        return None

    def _find_consumer_node(
        self,
        tensor_id: int,
        graph: KGIRGraph,
    ) -> Optional[int]:
        """Find the first node ID that consumes *tensor_id*."""
        for edge in graph.get_edges():
            if edge.tensor_id == tensor_id and edge.edge_type == "data_dep":
                return edge.target_id
        return None
