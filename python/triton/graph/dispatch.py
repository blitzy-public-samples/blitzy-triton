"""Hardware-Aware Dispatch Layer for Triton's graph-level cross-kernel optimization.

This module implements the hardware-aware dispatch layer that enumerates all
available GPU devices, compiles KGIR subgraphs to multiple hardware targets in
parallel, and selects optimal hardware targets per subgraph based on five
objectives. It supports three dispatch modes (performance, cost, balanced),
intra-vendor cross-generation dispatch, and cross-vendor dispatch with explicit
host-memory staging.

Novel Algorithm A2 — Multi-Device Dispatch Assignment
=====================================================

Investigation of two candidate approaches as mandated by AAP §0.7.4:

**Candidate 1: Greedy Topological-Order Assignment**
  Process subgraphs in topological order. For each subgraph, score all eligible
  devices using the five-objective weighted-sum scoring function and assign to
  the highest-scoring device.
  - Complexity: O(V * D) where V=nodes, D=devices
  - Deterministic, reproducible assignments
  - Meets the < 1ms dispatch decision latency bound per subgraph
  - Disadvantage: may miss globally optimal assignments when a locally suboptimal
    choice would yield better downstream scheduling

**Candidate 2: Weighted-Score Ranking with Look-Ahead**
  Score based on five objectives plus a look-ahead step that considers
  downstream subgraph impact before committing each assignment.
  - Complexity: O(V^2 * D) in the worst case
  - Better global decisions for dependency-heavy graphs
  - Disadvantage: higher complexity risks exceeding the 1ms latency bound for
    graphs with > 50 kernels and > 4 devices

**Selected approach: Greedy Topological-Order (Candidate 1)**
  Rationale: Guaranteed to meet the < 1ms latency bound required by AAP §0.7.2.
  For practical Triton workloads (transformer blocks, conv chains, optimizer
  steps), greedy topological-order produces near-optimal results because data
  locality dominates: successor nodes are heavily biased toward the same device
  as their producer, limiting the loss from greedy local decisions. The feedback
  loop (B5) compensates for any suboptimal greedy choices via iterative
  reassignment.

**Fallback:** Single-device dispatch on fastest available device (no multi-device
assignment) when only one device is available or when multi-device dispatch
produces worse estimated latency than single-device.

Five Scoring Objectives
-----------------------
1. Performance — estimated execution time on target device
2. Cost — resource utilization (SM/CU occupancy, memory usage)
3. Data Locality — penalty for data not on target device (transfer cost)
4. Device Utilization — spread work across devices to avoid bottlenecks
5. Memory Capacity — ensure subgraph memory fits on device
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from .kgir import KGIRGraph, KGIRNode, KGIREdge, HardwareProfile
from .config import DispatchConfig
from .utils import topological_sort, compute_tensor_size_bytes
from .errors import DispatchError, TransferError
from triton.backends import backends
from triton.backends.compiler import GPUTarget, BaseBackend
from triton.backends.driver import GPUDriver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default five-objective scoring weights per dispatch mode.
# Each mode emphasises a different primary objective while keeping all five
# active so that degenerate single-objective behaviour is avoided.
_MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "performance": {
        "performance": 0.50,
        "cost": 0.10,
        "data_locality": 0.20,
        "device_utilization": 0.10,
        "memory_capacity": 0.10,
    },
    "cost": {
        "performance": 0.10,
        "cost": 0.50,
        "data_locality": 0.10,
        "device_utilization": 0.20,
        "memory_capacity": 0.10,
    },
    "balanced": {
        "performance": 0.20,
        "cost": 0.20,
        "data_locality": 0.20,
        "device_utilization": 0.20,
        "memory_capacity": 0.20,
    },
}

# Objective names (canonical ordering for deterministic iteration).
_OBJECTIVE_NAMES: Tuple[str, ...] = (
    "performance",
    "cost",
    "data_locality",
    "device_utilization",
    "memory_capacity",
)

# Known interconnect peak bandwidths in GB/s.
_INTERCONNECT_BW: Dict[str, float] = {
    "pcie_3": 12.0,
    "pcie_4": 25.0,
    "pcie_5": 50.0,
    "nvlink_2": 75.0,
    "nvlink_3": 150.0,
    "nvlink_4": 450.0,
    "infinity_fabric": 100.0,
    "xgmi": 100.0,
}

# Worst-case bandwidth for cross-vendor host-memory staging (GB/s).
_HOST_STAGING_BANDWIDTH_GBPS: float = 12.0

# Default device properties used when runtime queries are unavailable.
_DEFAULT_SM_COUNT: int = 80
_DEFAULT_SMEM_PER_SM: int = 48 * 1024  # 48 KiB
_DEFAULT_REGISTERS_PER_SM: int = 65536
_DEFAULT_GLOBAL_MEMORY: int = 16 * (1024 ** 3)  # 16 GiB
_DEFAULT_MEMORY_BW_GBPS: float = 900.0
_DEFAULT_COMPUTE_TFLOPS: float = 19.0
_DEFAULT_MAX_STREAMS: int = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dispatch_knobs() -> Any:
    """Lazily import graph knobs to avoid circular dependency at import time.

    Returns the ``triton.knobs.graph`` singleton which exposes all
    ``TRITON_DISPATCH_*`` environment variable descriptors, or ``None`` when
    the knobs module is not yet available (e.g. during bootstrapping).
    """
    try:
        from triton import knobs  # noqa: E402
        return knobs.graph
    except (ImportError, AttributeError):
        return None


def _parse_cost_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse a comma-separated ``key=value`` string into a weight dict.

    Expected format from ``TRITON_DISPATCH_COST_WEIGHTS``:
    ``"performance=0.3,cost=0.3,data_locality=0.2,device_utilization=0.1,memory_capacity=0.1"``

    Returns ``None`` when *raw* is ``None`` or empty, allowing the caller to
    fall back to the mode-based default weights.
    """
    if not raw:
        return None
    weights: Dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, _, val = pair.partition("=")
        key = key.strip()
        if key in _OBJECTIVE_NAMES:
            try:
                weights[key] = float(val.strip())
            except ValueError:
                continue
    return weights if weights else None


def _parse_target_filter(raw: Optional[str]) -> Optional[Set[str]]:
    """Parse ``TRITON_DISPATCH_TARGETS`` into a set of allowed target strings.

    Expected format: comma-separated list such as ``"cuda:sm_90,hip:gfx942"``.
    Returns ``None`` when *raw* is ``None`` or empty (no filtering).
    """
    if not raw:
        return None
    targets: Set[str] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            targets.add(tok)
    return targets if targets else None


def _target_key(target: GPUTarget) -> str:
    """Produce a stable string key for a GPUTarget for filtering / hashing."""
    return f"{target.backend}:{target.arch}"


# ---------------------------------------------------------------------------
# DispatchMode Enum
# ---------------------------------------------------------------------------

class DispatchMode(str, Enum):
    """Dispatch optimisation mode selecting the primary objective.

    * ``PERFORMANCE`` — minimise end-to-end latency (weight performance highest)
    * ``COST`` — minimise total resource usage / device count
    * ``BALANCED`` — equal weight across all five objectives (default)
    """

    PERFORMANCE = "performance"
    COST = "cost"
    BALANCED = "balanced"


# ---------------------------------------------------------------------------
# HardwareInventory
# ---------------------------------------------------------------------------

class HardwareInventory:
    """Enumerates and catalogs all available GPU devices.

    On construction the inventory queries every registered Triton backend
    (via the ``triton.backends.backends`` registry) to discover active GPUs.
    Each discovered device is represented as a :class:`HardwareProfile`
    containing vendor, architecture generation, SM/CU count, memory capacity,
    bandwidth, interconnect topology and an associated :class:`GPUTarget`.

    Hardware enumeration MUST complete in < 10 ms (AAP §0.7.2).
    """

    def __init__(self) -> None:
        self._devices: List[HardwareProfile] = []
        self._device_by_target: Dict[str, HardwareProfile] = {}
        self._devices = self.enumerate_devices()
        # Build fast lookup index by target key
        for dev in self._devices:
            if dev.gpu_target is not None:
                self._device_by_target[_target_key(dev.gpu_target)] = dev

    # -- Public API ---------------------------------------------------------

    def enumerate_devices(self) -> List[HardwareProfile]:
        """Discover all GPUs via existing Triton backend driver APIs.

        Iterates the ``backends`` registry, checks each driver's
        ``is_active()`` class-method, and — for active backends — queries
        device properties.  Falls back to architecture-based defaults when
        detailed runtime information is unavailable.

        Returns:
            A list of :class:`HardwareProfile` instances, one per physical GPU.
        """
        start = time.perf_counter()
        profiles: List[HardwareProfile] = []

        try:
            for backend_name, backend_info in backends.items():
                try:
                    driver_cls = backend_info.driver
                    if not driver_cls.is_active():
                        continue
                    new_profiles = self._enumerate_backend_devices(
                        backend_name, backend_info, driver_cls,
                    )
                    profiles.extend(new_profiles)
                except Exception as exc:
                    logger.debug(
                        "Skipping backend '%s' during device enumeration: %s",
                        backend_name, exc,
                    )
                    continue
        except Exception as exc:
            logger.debug("Backend registry iteration failed: %s", exc)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > 10.0:
            logger.warning(
                "Hardware inventory enumeration took %.1f ms (> 10 ms limit)",
                elapsed_ms,
            )
        logger.debug(
            "Discovered %d GPU device(s) in %.2f ms",
            len(profiles), elapsed_ms,
        )
        return profiles

    def get_device(self, index: int) -> HardwareProfile:
        """Return the :class:`HardwareProfile` at *index*.

        Raises:
            DispatchError: if *index* is out of range.
        """
        if index < 0 or index >= len(self._devices):
            raise DispatchError(
                f"Device index {index} out of range "
                f"(0..{len(self._devices) - 1})"
            )
        return self._devices[index]

    def get_devices_by_vendor(self, vendor: str) -> List[HardwareProfile]:
        """Return all discovered devices matching *vendor* (e.g. ``"nvidia"``)."""
        return [d for d in self._devices if d.vendor == vendor]

    def get_interconnect_bandwidth(
        self, source: HardwareProfile, target: HardwareProfile,
    ) -> float:
        """Estimate peer-to-peer bandwidth in GB/s between *source* and *target*.

        Returns ``float('inf')`` for same-device transfers.  Cross-vendor
        transfers are penalised to host-staging bandwidth because data must
        transit host memory.
        """
        # Same physical device — no transfer needed.
        if source is target:
            return float("inf")
        if (
            source.gpu_target is not None
            and target.gpu_target is not None
            and source.gpu_target == target.gpu_target
        ):
            return float("inf")

        # Cross-vendor: must route through host memory.
        if source.vendor != target.vendor:
            return _HOST_STAGING_BANDWIDTH_GBPS

        # Same vendor: use the slower of the two interconnects.
        return min(
            source.interconnect_bandwidth_gbps,
            target.interconnect_bandwidth_gbps,
        )

    # -- Properties ---------------------------------------------------------

    @property
    def devices(self) -> List[HardwareProfile]:
        """All discovered hardware profiles (read-only)."""
        return list(self._devices)

    @property
    def device_count(self) -> int:
        """Number of discovered devices."""
        return len(self._devices)

    # -- Private helpers ----------------------------------------------------

    def _enumerate_backend_devices(
        self,
        backend_name: str,
        backend_info: Any,
        driver_cls: Any,
    ) -> List[HardwareProfile]:
        """Enumerate all devices exposed by a single backend driver."""
        profiles: List[HardwareProfile] = []

        # Determine vendor string from backend name.
        vendor = self._vendor_from_backend(backend_name)
        warp_size = 32 if vendor == "nvidia" else 64

        # Determine the device count.
        device_count = self._get_device_count(backend_name, vendor)

        for idx in range(device_count):
            try:
                profile = self._create_profile(
                    backend_name, vendor, driver_cls, idx, warp_size,
                )
                if profile is not None:
                    profiles.append(profile)
            except Exception as exc:
                logger.debug(
                    "Failed to profile device %d on backend '%s': %s",
                    idx, backend_name, exc,
                )
        return profiles

    @staticmethod
    def _vendor_from_backend(backend_name: str) -> str:
        """Map backend registry name to canonical vendor string."""
        mapping: Dict[str, str] = {
            "nvidia": "nvidia",
            "cuda": "nvidia",
            "amd": "amd",
            "hip": "amd",
        }
        return mapping.get(backend_name, backend_name)

    @staticmethod
    def _get_device_count(backend_name: str, vendor: str) -> int:
        """Query the number of devices using torch if available."""
        try:
            import torch  # type: ignore[import-untyped]
            if vendor == "nvidia" and torch.cuda.is_available():
                return max(torch.cuda.device_count(), 1)
            if vendor == "amd":
                if hasattr(torch, "hip") and hasattr(torch.hip, "device_count"):
                    return max(torch.hip.device_count(), 1)
                # Some PyTorch builds expose HIP via the cuda facade.
                if torch.cuda.is_available():
                    return max(torch.cuda.device_count(), 1)
        except (ImportError, RuntimeError, AttributeError):
            pass
        # If torch is unavailable the driver was still active — assume 1 device.
        return 1

    def _create_profile(
        self,
        backend_name: str,
        vendor: str,
        driver_cls: Any,
        device_idx: int,
        warp_size: int,
    ) -> Optional[HardwareProfile]:
        """Create a HardwareProfile for a single physical device."""
        arch: Any = None
        arch_generation: str = "unknown"

        # Try to get architecture from torch device properties first.
        sm_count = _DEFAULT_SM_COUNT
        smem_per_sm = _DEFAULT_SMEM_PER_SM
        registers_per_sm = _DEFAULT_REGISTERS_PER_SM
        global_memory = _DEFAULT_GLOBAL_MEMORY
        memory_bw = _DEFAULT_MEMORY_BW_GBPS
        compute_tflops = _DEFAULT_COMPUTE_TFLOPS

        try:
            import torch  # type: ignore[import-untyped]
            if vendor == "nvidia" and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(device_idx)
                major, minor = props.major, props.minor
                arch = major * 10 + minor
                arch_generation = f"sm_{arch}"
                sm_count = props.multi_processor_count
                global_memory = props.total_mem
                # Shared memory per SM — not always in older torch builds.
                if hasattr(props, "max_shared_memory_per_multiprocessor"):
                    smem_per_sm = props.max_shared_memory_per_multiprocessor
        except (ImportError, RuntimeError, AttributeError):
            pass

        # Fallback: try the driver's get_current_target for arch info.
        if arch is None:
            try:
                driver = driver_cls()
                target = driver.get_current_target()
                arch = target.arch
                warp_size = target.warp_size
                if vendor == "nvidia":
                    arch_generation = f"sm_{arch}"
                else:
                    arch_generation = str(arch)
            except Exception:
                # Cannot determine arch — skip this device.
                return None

        gpu_target = GPUTarget(
            backend=backend_name, arch=arch, warp_size=warp_size,
        )
        interconnect_type = self._detect_interconnect(vendor)
        interconnect_bw = _INTERCONNECT_BW.get(
            interconnect_type, _HOST_STAGING_BANDWIDTH_GBPS,
        )

        return HardwareProfile(
            vendor=vendor,
            arch_generation=arch_generation,
            sm_count=sm_count,
            smem_per_sm_bytes=smem_per_sm,
            registers_per_sm=registers_per_sm,
            global_memory_bytes=global_memory,
            memory_bandwidth_gbps=memory_bw,
            compute_throughput_tflops=compute_tflops,
            warp_size=warp_size,
            max_concurrent_streams=_DEFAULT_MAX_STREAMS,
            interconnect_type=interconnect_type,
            interconnect_bandwidth_gbps=interconnect_bw,
            gpu_target=gpu_target,
        )

    @staticmethod
    def _detect_interconnect(vendor: str) -> str:
        """Best-effort interconnect type detection.

        Without NVML or sysfs queries the only reliable information is the
        vendor.  NVIDIA recent cards typically use PCIe-4 or NVLink; AMD
        uses Infinity Fabric.  We default conservatively to ``pcie_4``.
        """
        if vendor == "amd":
            return "infinity_fabric"
        return "pcie_4"


# ---------------------------------------------------------------------------
# DispatchDecisionEngine
# ---------------------------------------------------------------------------

class DispatchDecisionEngine:
    """Selects optimal hardware targets per subgraph based on multi-objective scoring.

    The engine implements **Novel Algorithm A2 (Greedy Topological-Order
    Assignment)**: subgraphs are visited in dependency order and greedily
    assigned to the highest-scoring eligible device using a five-objective
    weighted-sum scoring function.  The weights are determined by the active
    :class:`DispatchMode` and may be overridden per-session via
    ``TRITON_DISPATCH_COST_WEIGHTS``.

    Parameters:
        inventory: The enumerated hardware inventory.
        config: Dispatch layer configuration (mode, targets, etc.).
        mode: The dispatch optimisation mode.
    """

    def __init__(
        self,
        inventory: HardwareInventory,
        config: DispatchConfig,
        mode: DispatchMode,
    ) -> None:
        if inventory.device_count == 0:
            raise DispatchError("No GPU devices available for dispatch")

        self._inventory: HardwareInventory = inventory
        self._config: DispatchConfig = config
        self._mode: DispatchMode = mode

        # Resolve effective scoring weights.
        self._weights: Dict[str, float] = self._resolve_weights(
            mode, config.cost_weights,
        )

        # Build the set of eligible devices (applying optional target filter).
        self._eligible_devices: List[HardwareProfile] = (
            self._apply_target_filter(inventory.devices, config.targets)
        )
        if not self._eligible_devices:
            raise DispatchError(
                "No eligible devices after applying target filter: "
                f"filter={config.targets}"
            )

        # Per-device cumulative load tracker for utilisation scoring.
        # Maps HardwareProfile id → count of assigned nodes so far.
        self._device_load: Dict[int, int] = {
            id(d): 0 for d in self._eligible_devices
        }

        # Optional compile callback set by the CodeGenerationBridge.
        self._compile_fn: Optional[Any] = None

        # Log configuration if requested.
        knobs = _get_dispatch_knobs()
        log_enabled = config.log or (knobs is not None and knobs.dispatch_log)
        if log_enabled:
            logger.setLevel(logging.DEBUG)
            logger.debug(
                "DispatchDecisionEngine initialised: mode=%s, devices=%d, "
                "weights=%s, granularity=%s",
                mode.value,
                len(self._eligible_devices),
                self._weights,
                config.granularity,
            )

    # -- Public API ---------------------------------------------------------

    def score_subgraph_device(
        self, subgraph: KGIRGraph, device: HardwareProfile,
    ) -> float:
        """Score a *subgraph* against a *device* using the five-objective model.

        Each objective is normalised to [0, 1] (higher is better) and combined
        via weighted sum according to the active dispatch mode weights.

        Args:
            subgraph: A (possibly single-node) :class:`KGIRGraph`.
            device: The candidate :class:`HardwareProfile` to score.

        Returns:
            Aggregate weighted score in [0, 1].
        """
        scores: Dict[str, float] = {}
        for name in _OBJECTIVE_NAMES:
            scores[name] = self._score_objective(name, subgraph, device)

        aggregate = sum(
            self._weights.get(name, 0.0) * scores[name]
            for name in _OBJECTIVE_NAMES
        )
        return aggregate

    def compute_dispatch_plan(
        self, graph: KGIRGraph,
    ) -> Dict[int, GPUTarget]:
        """Compute a dispatch plan mapping every node to a :class:`GPUTarget`.

        Implements Novel Algorithm A2 (Greedy Topological-Order Assignment):
        nodes are visited in topological order and each is assigned to the
        highest-scoring eligible device.

        When ``config.granularity`` is ``"graph"``, the entire graph is scored
        as one unit and all nodes are assigned to the same device.

        Dispatch decision latency MUST be < 1 ms per subgraph (AAP §0.7.2).

        Args:
            graph: The full :class:`KGIRGraph` to schedule.

        Returns:
            A mapping ``{node_id: GPUTarget}`` for every node in the graph.

        Raises:
            DispatchError: when no viable assignment can be found.
        """
        if graph.node_count() == 0:
            return {}

        start = time.perf_counter()

        # Reset device load tracker for a fresh planning pass.
        for key in self._device_load:
            self._device_load[key] = 0

        plan: Dict[int, GPUTarget] = {}

        if self._config.granularity == "graph":
            # Whole-graph dispatch: score once, assign all nodes.
            plan = self._dispatch_graph_level(graph)
        else:
            # Subgraph-level (per-node) dispatch — the default.
            plan = self._dispatch_subgraph_level(graph)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "Dispatch plan computed for %d nodes in %.2f ms",
            graph.node_count(), elapsed_ms,
        )
        return plan

    def insert_transfer_operations(
        self, graph: KGIRGraph, plan: Dict[int, GPUTarget],
    ) -> KGIRGraph:
        """Insert cross-device transfer edges where a data dependency crosses devices.

        For each existing data-dependency edge ``(u -> v)`` where
        ``plan[u] != plan[v]``, a ``cross_device_transfer`` edge is added
        carrying the estimated transfer cost in its metadata.

        Cross-vendor transfers (e.g. NVIDIA → AMD) use explicit host-memory
        staging and incur the highest penalty.

        Args:
            graph: The :class:`KGIRGraph` to augment (mutated in place).
            plan: The dispatch plan from :meth:`compute_dispatch_plan`.

        Returns:
            The same *graph* reference, now containing transfer edges.

        Raises:
            TransferError: when a required transfer cannot be modelled.
        """
        edges = graph.get_edges()
        inserted_count = 0

        for edge in edges:
            src_target = plan.get(edge.source_id)
            tgt_target = plan.get(edge.target_id)

            # Only process data-dependency edges that cross devices.
            if src_target is None or tgt_target is None:
                continue
            if src_target == tgt_target:
                continue
            if edge.edge_type != "data_dep":
                continue

            # Determine transfer bandwidth.
            src_profile = self._profile_for_target(src_target)
            tgt_profile = self._profile_for_target(tgt_target)
            if src_profile is None or tgt_profile is None:
                raise TransferError(
                    error_message=(
                        f"Cannot model transfer for edge "
                        f"({edge.source_id} -> {edge.target_id}): "
                        "missing device profile"
                    ),
                    source_device=str(src_target),
                    target_device=str(tgt_target),
                )

            bandwidth = self._inventory.get_interconnect_bandwidth(
                src_profile, tgt_profile,
            )

            # Estimate transfer size from edge tensor metadata.
            transfer_bytes = self._estimate_transfer_bytes(
                graph, edge,
            )
            transfer_time_ms = (
                (transfer_bytes / (bandwidth * 1e9)) * 1000.0
                if bandwidth > 0.0 and bandwidth != float("inf")
                else 0.0
            )

            is_cross_vendor = (
                src_profile.vendor != tgt_profile.vendor
            )

            transfer_metadata: Dict[str, Any] = {
                "bandwidth_gbps": bandwidth,
                "transfer_bytes": transfer_bytes,
                "estimated_time_ms": transfer_time_ms,
                "cross_vendor": is_cross_vendor,
                "host_staging": is_cross_vendor,
                "source_target": _target_key(src_target),
                "dest_target": _target_key(tgt_target),
            }

            try:
                graph.add_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type="cross_device_transfer",
                    tensor_id=edge.tensor_id,
                    metadata=transfer_metadata,
                )
                inserted_count += 1
            except Exception as exc:
                raise TransferError(
                    error_message=(
                        f"Failed to insert transfer edge "
                        f"({edge.source_id} -> {edge.target_id}): {exc}"
                    ),
                    source_device=str(src_target),
                    target_device=str(tgt_target),
                ) from exc

        logger.debug(
            "Inserted %d cross-device transfer edge(s)", inserted_count,
        )
        return graph

    def compile_for_targets(
        self,
        graph: KGIRGraph,
        targets: List[GPUTarget],
    ) -> Dict[GPUTarget, Any]:
        """Compile *graph* for multiple hardware *targets* in parallel.

        Uses :class:`concurrent.futures.ThreadPoolExecutor` so that
        multi-target compilation wall-clock ≤ slowest single-target + 10 %
        coordination overhead (AAP §0.7.2).

        The actual per-target compile callable is resolved in order:
        1. ``self._compile_fn`` if explicitly set by the caller / bridge.
        2. Lazy import of ``triton.graph.codegen_bridge.compile_for_target``.
        3. Raises :class:`DispatchError` if neither is available.

        Args:
            graph: The :class:`KGIRGraph` to compile.
            targets: Target list (one :class:`GPUTarget` per desired binary).

        Returns:
            ``{GPUTarget: compiled_result}`` — entries are ``None`` for
            targets that failed to compile.
        """
        if not targets:
            raise DispatchError("No targets specified for compilation")

        compile_fn = self._resolve_compile_fn()

        results: Dict[GPUTarget, Any] = {}
        start = time.perf_counter()

        worker_count = min(len(targets), 8)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_target = {
                executor.submit(compile_fn, graph, t): t
                for t in targets
            }
            for future in future_to_target:
                target = future_to_target[future]
                try:
                    results[target] = future.result()
                except Exception as exc:
                    logger.warning(
                        "Compilation failed for target %s: %s", target, exc,
                    )
                    results[target] = None

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "Multi-target compilation for %d target(s) completed in %.1f ms",
            len(targets), elapsed_ms,
        )
        return results

    # -- Private: scoring ---------------------------------------------------

    def _score_objective(
        self,
        objective: str,
        subgraph: KGIRGraph,
        device: HardwareProfile,
    ) -> float:
        """Compute a single normalised objective score in [0, 1]."""
        if objective == "performance":
            return self._score_performance(subgraph, device)
        if objective == "cost":
            return self._score_cost(subgraph, device)
        if objective == "data_locality":
            return self._score_data_locality(subgraph, device)
        if objective == "device_utilization":
            return self._score_device_utilization(device)
        if objective == "memory_capacity":
            return self._score_memory_capacity(subgraph, device)
        return 0.0

    def _score_performance(
        self, subgraph: KGIRGraph, device: HardwareProfile,
    ) -> float:
        """Estimate relative execution speed.

        Uses runtime performance annotations when available (Phase 2 of the
        adaptive cost model) and falls back to a heuristic based on
        compute throughput and memory bandwidth (Phase 1).
        """
        order = subgraph.topological_sort()
        total_estimated_time = 0.0

        for nid in order:
            node = subgraph.get_node(nid)
            if node is None:
                continue

            # Prefer measured performance annotation for this target.
            target_key = (
                _target_key(device.gpu_target)
                if device.gpu_target is not None
                else ""
            )
            perf = node.get_performance_annotation(target_key)
            if perf and isinstance(perf, dict):
                measured_time = perf.get("wall_clock_ms")
                if measured_time is not None:
                    total_estimated_time += float(measured_time)
                    continue

            # Phase 1 heuristic: rough model from resource usage + device specs.
            resources = node.get_resource_usage()
            smem_bytes = resources.get("shared_memory_bytes", 0)
            num_warps = resources.get("num_warps", 1)

            meta = node.get_metadata()
            grid_dims = (
                meta.grid_dimensions
                if meta is not None and meta.grid_dimensions
                else (1, 1, 1)
            )
            total_blocks = 1
            for d in grid_dims:
                total_blocks *= d

            # Estimate: time ∝ blocks / (device_SMs * throughput_factor).
            throughput_factor = max(device.compute_throughput_tflops, 0.1)
            sm_factor = max(device.sm_count, 1)
            heuristic_time = total_blocks / (sm_factor * throughput_factor)
            total_estimated_time += heuristic_time

        # Normalise: lower time → higher score.  Use a soft-max mapping.
        if total_estimated_time <= 0.0:
            return 1.0
        return 1.0 / (1.0 + total_estimated_time)

    def _score_cost(
        self, subgraph: KGIRGraph, device: HardwareProfile,
    ) -> float:
        """Score resource utilisation efficiency (higher = more efficient).

        Penalises over-provisioning: assigning a small kernel to a large GPU
        wastes resources.
        """
        order = subgraph.topological_sort()
        total_sm_demand = 0
        total_smem_demand = 0

        for nid in order:
            node = subgraph.get_node(nid)
            if node is None:
                continue
            resources = node.get_resource_usage()
            num_warps = resources.get("num_warps", 1)
            smem_bytes = resources.get("shared_memory_bytes", 0)

            meta = node.get_metadata()
            grid_dims = (
                meta.grid_dimensions
                if meta is not None and meta.grid_dimensions
                else (1, 1, 1)
            )
            total_blocks = 1
            for d in grid_dims:
                total_blocks *= d

            # Each block requires some SMs.
            total_sm_demand += total_blocks
            total_smem_demand += smem_bytes * total_blocks

        # Utilisation ratio: how well the subgraph fills the device.
        sm_util = min(total_sm_demand / max(device.sm_count, 1), 1.0)
        smem_cap = device.smem_per_sm_bytes * device.sm_count
        smem_util = (
            min(total_smem_demand / max(smem_cap, 1), 1.0) if smem_cap > 0 else 0.0
        )

        # Prefer devices where utilisation is high (less waste).
        return 0.5 * sm_util + 0.5 * smem_util

    def _score_data_locality(
        self, subgraph: KGIRGraph, device: HardwareProfile,
    ) -> float:
        """Score data locality: how much input data already resides on *device*.

        For each predecessor node (outside the subgraph) already assigned to
        *device*, no transfer is needed → high locality.  Cross-device and
        especially cross-vendor predecessors reduce the score.
        """
        if device.gpu_target is None:
            return 0.5  # No target info — neutral score.

        target_key = _target_key(device.gpu_target)
        total_deps = 0
        local_deps = 0

        order = subgraph.topological_sort()
        subgraph_nodes: Set[int] = set(order)

        for nid in order:
            predecessors = subgraph.get_predecessors(nid)
            for pred_id in predecessors:
                if pred_id in subgraph_nodes:
                    continue  # Internal edge — not a cross-device concern.
                total_deps += 1
                # Check assignment (stored in device_load tracking is not
                # sufficient; we compare GPUTarget via the plan being built).
                # During scoring we use a heuristic: if the predecessor's
                # preferred target matches this device, count as local.
                pred_node = subgraph.get_node(pred_id)
                if pred_node is not None:
                    perf = pred_node.get_performance_annotation(target_key)
                    if perf and isinstance(perf, dict):
                        if perf.get("assigned_target") == target_key:
                            local_deps += 1

        if total_deps == 0:
            return 1.0  # No external dependencies — perfectly local.
        return local_deps / total_deps

    def _score_device_utilization(self, device: HardwareProfile) -> float:
        """Prefer under-utilised devices to spread work evenly.

        The score decreases as more nodes are assigned to *device* relative
        to the total tracked load across all eligible devices.
        """
        dev_load = self._device_load.get(id(device), 0)
        total_load = max(sum(self._device_load.values()), 1)
        fair_share = total_load / max(len(self._eligible_devices), 1)

        if fair_share == 0:
            return 1.0  # No load yet — all devices equally good.

        # Score: 1.0 when load ≤ fair share, degrades linearly above.
        ratio = dev_load / fair_share
        return max(1.0 - 0.5 * max(ratio - 1.0, 0.0), 0.0)

    def _score_memory_capacity(
        self, subgraph: KGIRGraph, device: HardwareProfile,
    ) -> float:
        """Ensure the subgraph's memory footprint fits the device.

        Returns 1.0 when the footprint is comfortably within device memory,
        degrades toward 0.0 as it approaches the limit, and returns 0.0 if
        the footprint exceeds device memory.
        """
        order = subgraph.topological_sort()
        total_bytes = 0

        for nid in order:
            node = subgraph.get_node(nid)
            if node is None:
                continue
            meta = node.get_metadata()
            if meta is None:
                continue

            # Sum tensor sizes for all tensors referenced by this node.
            shapes = meta.tensor_shapes if meta.tensor_shapes else []
            dtypes = meta.tensor_dtypes if meta.tensor_dtypes else []
            for shape, dtype in zip(shapes, dtypes):
                if shape and dtype:
                    try:
                        total_bytes += compute_tensor_size_bytes(
                            tuple(shape), dtype,
                        )
                    except (TypeError, ValueError):
                        pass

        device_mem = device.global_memory_bytes
        if device_mem <= 0:
            return 0.0
        if total_bytes > device_mem:
            return 0.0

        usage_ratio = total_bytes / device_mem
        # Comfortable below 80 % → score 1.0; linear degradation 80 %→100 %.
        if usage_ratio <= 0.8:
            return 1.0
        return max(1.0 - (usage_ratio - 0.8) / 0.2, 0.0)

    # -- Private: dispatch plan building ------------------------------------

    def _dispatch_graph_level(
        self, graph: KGIRGraph,
    ) -> Dict[int, GPUTarget]:
        """Assign ALL nodes to the single best-scoring device."""
        best_device: Optional[HardwareProfile] = None
        best_score: float = -1.0

        for device in self._eligible_devices:
            score = self.score_subgraph_device(graph, device)
            if score > best_score:
                best_score = score
                best_device = device

        if best_device is None or best_device.gpu_target is None:
            raise DispatchError("No device could be selected for graph dispatch")

        target = best_device.gpu_target
        order = graph.topological_sort()
        plan: Dict[int, GPUTarget] = {nid: target for nid in order}

        self._device_load[id(best_device)] = len(order)
        return plan

    def _dispatch_subgraph_level(
        self, graph: KGIRGraph,
    ) -> Dict[int, GPUTarget]:
        """Greedy topological-order per-node dispatch (Algorithm A2).

        Each node is scored against every eligible device and assigned to
        the highest-scoring one.  After assignment the device load tracker
        is updated so that subsequent scoring reflects current utilisation.
        """
        adjacency: Dict[int, List[int]] = {}
        all_nodes: Set[int] = set()
        for nid in range(graph.node_count()):
            node = graph.get_node(nid)
            if node is not None:
                all_nodes.add(nid)
                adjacency[nid] = list(graph.get_successors(nid))

        # Also gather nodes that may have non-sequential ids.
        # Walk edges to be safe.
        for edge in graph.get_edges():
            all_nodes.add(edge.source_id)
            all_nodes.add(edge.target_id)
            adjacency.setdefault(edge.source_id, [])
            if edge.target_id not in adjacency.get(edge.source_id, []):
                adjacency[edge.source_id].append(edge.target_id)
            adjacency.setdefault(edge.target_id, [])

        order = topological_sort(adjacency, all_nodes)
        plan: Dict[int, GPUTarget] = {}

        for nid in order:
            start = time.perf_counter()

            # Build a lightweight single-node view.
            node = graph.get_node(nid)
            if node is None:
                continue

            best_device: Optional[HardwareProfile] = None
            best_score: float = -1.0

            for device in self._eligible_devices:
                score = self._score_single_node(node, device, graph, plan)
                if score > best_score:
                    best_score = score
                    best_device = device

            if best_device is None or best_device.gpu_target is None:
                # Fallback: assign to first eligible device.
                best_device = self._eligible_devices[0]

            target = best_device.gpu_target
            plan[nid] = target
            self._device_load[id(best_device)] = (
                self._device_load.get(id(best_device), 0) + 1
            )

            elapsed_us = (time.perf_counter() - start) * 1e6
            if elapsed_us > 1000.0:
                logger.warning(
                    "Dispatch decision for node %d took %.0f µs (> 1 ms limit)",
                    nid, elapsed_us,
                )

        return plan

    def _score_single_node(
        self,
        node: KGIRNode,
        device: HardwareProfile,
        full_graph: KGIRGraph,
        current_plan: Dict[int, GPUTarget],
    ) -> float:
        """Score a single node against a device using the five-objective model.

        This is an optimised path that avoids constructing a temporary
        KGIRGraph wrapper for every node.
        """
        scores: Dict[str, float] = {}

        # 1. Performance objective.
        scores["performance"] = self._node_performance(node, device)

        # 2. Cost objective.
        scores["cost"] = self._node_cost(node, device)

        # 3. Data locality objective.
        scores["data_locality"] = self._node_locality(
            node, device, full_graph, current_plan,
        )

        # 4. Device utilisation objective.
        scores["device_utilization"] = self._score_device_utilization(device)

        # 5. Memory capacity objective.
        scores["memory_capacity"] = self._node_memory(node, device)

        aggregate = sum(
            self._weights.get(name, 0.0) * scores[name]
            for name in _OBJECTIVE_NAMES
        )
        return aggregate

    # -- Node-level objective helpers ---------------------------------------

    def _node_performance(
        self, node: KGIRNode, device: HardwareProfile,
    ) -> float:
        """Heuristic or measured execution time for a single node."""
        target_key = (
            _target_key(device.gpu_target)
            if device.gpu_target is not None
            else ""
        )
        perf = node.get_performance_annotation(target_key)
        if perf and isinstance(perf, dict):
            measured = perf.get("wall_clock_ms")
            if measured is not None:
                t = float(measured)
                return 1.0 / (1.0 + t) if t > 0 else 1.0

        resources = node.get_resource_usage()
        num_warps = resources.get("num_warps", 1)
        meta = node.get_metadata()
        grid_dims = (
            meta.grid_dimensions
            if meta is not None and meta.grid_dimensions
            else (1, 1, 1)
        )
        total_blocks = 1
        for d in grid_dims:
            total_blocks *= d

        throughput = max(device.compute_throughput_tflops, 0.1)
        sm_count = max(device.sm_count, 1)
        heuristic = total_blocks / (sm_count * throughput)
        return 1.0 / (1.0 + heuristic)

    def _node_cost(
        self, node: KGIRNode, device: HardwareProfile,
    ) -> float:
        """Resource efficiency for a single node on *device*."""
        resources = node.get_resource_usage()
        smem_bytes = resources.get("shared_memory_bytes", 0)
        num_warps = resources.get("num_warps", 1)

        meta = node.get_metadata()
        grid_dims = (
            meta.grid_dimensions
            if meta is not None and meta.grid_dimensions
            else (1, 1, 1)
        )
        total_blocks = 1
        for d in grid_dims:
            total_blocks *= d

        sm_util = min(total_blocks / max(device.sm_count, 1), 1.0)
        smem_cap = device.smem_per_sm_bytes * device.sm_count
        smem_util = (
            min(smem_bytes * total_blocks / max(smem_cap, 1), 1.0)
            if smem_cap > 0
            else 0.0
        )
        return 0.5 * sm_util + 0.5 * smem_util

    def _node_locality(
        self,
        node: KGIRNode,
        device: HardwareProfile,
        full_graph: KGIRGraph,
        current_plan: Dict[int, GPUTarget],
    ) -> float:
        """Data locality for a single node given assignments made so far."""
        if device.gpu_target is None:
            return 0.5

        predecessors = full_graph.get_predecessors(node.node_id)
        if not predecessors:
            return 1.0  # Root node — perfectly local.

        local = 0
        total = len(predecessors)
        for pred_id in predecessors:
            pred_target = current_plan.get(pred_id)
            if pred_target is not None and pred_target == device.gpu_target:
                local += 1

        return local / max(total, 1)

    def _node_memory(
        self, node: KGIRNode, device: HardwareProfile,
    ) -> float:
        """Memory capacity check for a single node."""
        meta = node.get_metadata()
        if meta is None:
            return 1.0

        total_bytes = 0
        shapes = meta.tensor_shapes if meta.tensor_shapes else []
        dtypes = meta.tensor_dtypes if meta.tensor_dtypes else []
        for shape, dtype in zip(shapes, dtypes):
            if shape and dtype:
                try:
                    total_bytes += compute_tensor_size_bytes(tuple(shape), dtype)
                except (TypeError, ValueError):
                    pass

        device_mem = device.global_memory_bytes
        if device_mem <= 0:
            return 0.0
        if total_bytes > device_mem:
            return 0.0

        ratio = total_bytes / device_mem
        if ratio <= 0.8:
            return 1.0
        return max(1.0 - (ratio - 0.8) / 0.2, 0.0)

    # -- Private: utilities -------------------------------------------------

    @staticmethod
    def _resolve_weights(
        mode: DispatchMode,
        config_weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Determine effective scoring weights.

        Priority order:
        1. Explicit ``config_weights`` from :class:`DispatchConfig`.
        2. ``TRITON_DISPATCH_COST_WEIGHTS`` environment variable.
        3. Default weights from ``_MODE_WEIGHTS[mode]``.
        """
        if config_weights:
            return dict(config_weights)

        knobs = _get_dispatch_knobs()
        if knobs is not None:
            env_raw = knobs.dispatch_cost_weights
            parsed = _parse_cost_weights(env_raw)
            if parsed:
                return parsed

        return dict(_MODE_WEIGHTS.get(mode.value, _MODE_WEIGHTS["balanced"]))

    @staticmethod
    def _apply_target_filter(
        devices: List[HardwareProfile],
        filter_spec: Optional[str],
    ) -> List[HardwareProfile]:
        """Restrict eligible devices by the ``TRITON_DISPATCH_TARGETS`` filter.

        If *filter_spec* is ``None`` or empty, all devices are eligible.
        """
        if not filter_spec:
            # Also check the environment variable.
            knobs = _get_dispatch_knobs()
            if knobs is not None:
                filter_spec = knobs.dispatch_targets
        if not filter_spec:
            return list(devices)

        allowed = _parse_target_filter(filter_spec)
        if allowed is None:
            return list(devices)

        return [
            d for d in devices
            if d.gpu_target is not None
            and _target_key(d.gpu_target) in allowed
        ]

    def _profile_for_target(
        self, target: GPUTarget,
    ) -> Optional[HardwareProfile]:
        """Look up the :class:`HardwareProfile` associated with *target*."""
        key = _target_key(target)
        prof = self._inventory._device_by_target.get(key)
        if prof is not None:
            return prof
        # Linear scan fallback.
        for dev in self._inventory.devices:
            if dev.gpu_target is not None and dev.gpu_target == target:
                return dev
        return None

    @staticmethod
    def _estimate_transfer_bytes(
        graph: KGIRGraph, edge: KGIREdge,
    ) -> int:
        """Estimate the data volume transferred along *edge*.

        Uses the edge's ``tensor_id`` to look up tensor metadata from the
        source node, then computes size from shape and dtype.  Falls back
        to a conservative 1 MiB estimate when metadata is unavailable.
        """
        fallback = 1 * 1024 * 1024  # 1 MiB default

        src_node = graph.get_node(edge.source_id)
        if src_node is None:
            return fallback

        meta = src_node.get_metadata()
        if meta is None:
            return fallback

        tensor_id = edge.tensor_id
        shapes = meta.tensor_shapes if meta.tensor_shapes else []
        dtypes = meta.tensor_dtypes if meta.tensor_dtypes else []

        if tensor_id is not None and isinstance(tensor_id, int):
            if 0 <= tensor_id < len(shapes) and tensor_id < len(dtypes):
                shape = shapes[tensor_id]
                dtype = dtypes[tensor_id]
                if shape and dtype:
                    try:
                        return compute_tensor_size_bytes(tuple(shape), dtype)
                    except (TypeError, ValueError):
                        pass

        # Sum all output tensors as a conservative estimate.
        total = 0
        for shape, dtype in zip(shapes, dtypes):
            if shape and dtype:
                try:
                    total += compute_tensor_size_bytes(tuple(shape), dtype)
                except (TypeError, ValueError):
                    pass
        return total if total > 0 else fallback

    def _resolve_compile_fn(self) -> Any:
        """Resolve the per-target compile callable.

        Priority:
        1. Explicit ``self._compile_fn`` (set by CodeGenerationBridge).
        2. Lazy import of ``triton.graph.codegen_bridge.compile_for_target``.
        3. Raises :class:`DispatchError`.
        """
        if self._compile_fn is not None:
            return self._compile_fn

        try:
            from triton.graph.codegen_bridge import (  # type: ignore[import]
                compile_for_target,
            )
            return compile_for_target
        except ImportError:
            pass

        raise DispatchError(
            "No compile function available. Set engine._compile_fn or "
            "ensure triton.graph.codegen_bridge is importable."
        )
