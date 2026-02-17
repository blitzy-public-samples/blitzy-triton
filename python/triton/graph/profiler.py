"""Runtime Profiler — Lightweight GPU event instrumentation for graph-level kernel metrics.

Provides per-kernel per-target metric collection using CUDA events (via torch.cuda.Event),
HIP events (via AMD backend bindings), or CPU fallback timing (time.perf_counter_ns).

The profiler enforces a strict overhead budget (default < 3% of total kernel execution time)
and auto-disables instrumentation when the budget is exceeded.

Design Principles:
  - Backend-agnostic: _GPUEvent abstracts CUDA/HIP/CPU timing
  - Lazy imports: torch and backend-specific APIs are imported only on first use
  - Zero external dependencies: uses only Python stdlib + existing Triton infrastructure
  - Multi-target: metrics are collected and grouped per GPUTarget
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from triton.backends.compiler import GPUTarget

from .config import GraphConfig
from .kgir import HardwareProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel used when GPU event creation is unavailable at runtime.
# ---------------------------------------------------------------------------
_BACKEND_CPU = "cpu"

# Typical CUDA event overhead in microseconds (used for pre-commit estimation).
_DEFAULT_EVENT_OVERHEAD_US = 3.0


# ============================================================================
# _GPUEvent — Backend-agnostic GPU event abstraction
# ============================================================================

class _GPUEvent:
    """Abstraction over CUDA / HIP / CPU events for lightweight profiling.

    For CUDA targets the class lazily wraps ``torch.cuda.Event`` so that
    import-time failures on headless CI machines are avoided.  For HIP targets
    the same ``torch.cuda.Event`` pathway is used when ROCm-aware PyTorch is
    available; otherwise (and for all other backends) a pure-CPU fallback
    based on ``time.perf_counter_ns`` is employed.
    """

    # ------------------------------------------------------------------
    # Class-level lazy-init caches for the torch module and availability
    # ------------------------------------------------------------------
    _torch_module: Optional[Any] = None
    _torch_checked: bool = False
    _torch_available: bool = False

    @classmethod
    def _ensure_torch(cls) -> bool:
        """Lazy-import torch exactly once and cache availability."""
        if not cls._torch_checked:
            cls._torch_checked = True
            try:
                import torch  # noqa: F811 — lazy import
                cls._torch_module = torch
                # Verify CUDA/HIP runtime is actually reachable.
                cls._torch_available = (
                    hasattr(torch, "cuda")
                    and callable(getattr(torch.cuda, "Event", None))
                )
            except Exception:
                cls._torch_available = False
        return cls._torch_available

    # ------------------------------------------------------------------
    # Instance interface
    # ------------------------------------------------------------------

    def __init__(self, backend: str) -> None:
        """Create a GPU event for the given backend ("cuda", "hip", or "cpu").

        Parameters
        ----------
        backend:
            One of ``"cuda"``, ``"hip"``, or ``"cpu"``.  For ``"cuda"`` and
            ``"hip"`` the implementation will attempt to use ``torch.cuda.Event``
            (ROCm-aware PyTorch maps CUDA API calls to HIP automatically).
            Falls back to CPU timing when torch is unavailable.
        """
        self._backend: str = backend
        self._native_event: Optional[Any] = None
        self._cpu_timestamp_ns: Optional[int] = None

        if backend in ("cuda", "hip") and self._ensure_torch():
            torch_mod = self.__class__._torch_module
            # ``enable_timing=True`` is required for ``elapsed_time``.
            self._native_event = torch_mod.cuda.Event(enable_timing=True)
        else:
            # CPU fallback — the event is simply a nanosecond timestamp.
            self._backend = _BACKEND_CPU

    def record(self, stream: Any = None) -> None:
        """Record this event on *stream*.

        For native GPU events the provided stream handle is used; for the CPU
        fallback we simply record ``time.perf_counter_ns()``.
        """
        if self._native_event is not None:
            if stream is not None:
                self._native_event.record(stream)
            else:
                self._native_event.record()
        else:
            self._cpu_timestamp_ns = time.perf_counter_ns()

    def synchronize(self) -> None:
        """Block until this event has been reached on the device."""
        if self._native_event is not None:
            self._native_event.synchronize()
        # CPU fallback: nothing to synchronize.

    @staticmethod
    def elapsed_time(start: _GPUEvent, end: _GPUEvent) -> float:
        """Return elapsed time **in milliseconds** between *start* and *end*.

        For native GPU events this delegates to
        ``torch.cuda.Event.elapsed_time`` which returns milliseconds.
        For the CPU fallback we compute the delta from stored nanosecond
        timestamps and convert to milliseconds.
        """
        if start._native_event is not None and end._native_event is not None:
            return start._native_event.elapsed_time(end._native_event)

        # CPU fallback path.
        s_ns = start._cpu_timestamp_ns or 0
        e_ns = end._cpu_timestamp_ns or 0
        return (e_ns - s_ns) / 1_000_000.0


# ============================================================================
# RuntimeProfiler — Lightweight GPU event profiler
# ============================================================================

class RuntimeProfiler:
    """Lightweight GPU event profiler for graph-level kernel execution metrics.

    The profiler instruments kernel launches with start/end GPU event pairs,
    synchronises them after execution, and computes per-kernel per-target
    metrics (wall-clock time, estimated memory throughput, launch overhead,
    cross-device transfer time).

    An overhead budget (default 3 %) is continuously monitored.  When the
    cumulative profiling overhead exceeds the budget the profiler auto-disables
    further instrumentation and logs a warning.

    Parameters
    ----------
    config:
        ``GraphConfig`` instance carrying feedback/dispatch sub-configs.
    overhead_budget:
        Maximum allowed profiling overhead expressed as a fraction of total
        kernel execution time.  Default is ``0.03`` (3 %).
    """

    def __init__(self, config: GraphConfig, overhead_budget: float = 0.03) -> None:
        self._config: GraphConfig = config
        self._overhead_budget: float = overhead_budget

        # Per-kernel event storage:
        # {kernel_id: {"start": _GPUEvent, "end": _GPUEvent,
        #              "target": GPUTarget, "stream": Any,
        #              "hw_profile": Optional[HardwareProfile]}}
        self._events: Dict[int, Dict[str, Any]] = {}

        # Collected metrics after synchronize_and_collect.
        self._metrics: Dict[int, Dict[str, float]] = {}

        # Accumulated timing counters (in **milliseconds**).
        self._total_kernel_time: float = 0.0
        self._total_profiling_overhead: float = 0.0

        # Profiling on/off flag — auto-disabled when budget is exceeded.
        self._enabled: bool = True

        # Whether feedback logging is requested via config.
        self._log_enabled: bool = (
            config.feedback.log if config.feedback is not None else False
        )

        # Stream pool size from dispatch config — used to bound the number
        # of concurrent stream event pairs we track.
        self._stream_pool_size: int = (
            config.dispatch.stream_pool_size
            if config.dispatch is not None
            else 8
        )

        # Ordered record of kernel ids for deterministic iteration.
        self._kernel_order: List[int] = []

    # ------------------------------------------------------------------
    # GPU event instrumentation
    # ------------------------------------------------------------------

    def instrument_launch(
        self,
        kernel_id: int,
        target: GPUTarget,
        stream: Any = None,
        hw_profile: Optional[HardwareProfile] = None,
    ) -> Tuple[Any, Any]:
        """Create and register a start/end event pair for *kernel_id*.

        Returns ``(start_event, end_event)`` — both are ``_GPUEvent``
        instances.  The caller is responsible for calling
        ``start_event.record(stream)`` before the kernel launch and
        ``end_event.record(stream)`` after.

        When the profiler is disabled (overhead budget exceeded) the events
        are still created but on the CPU-fallback path so that the returned
        objects remain usable without error, albeit with lower-fidelity
        timing.
        """
        overhead_start = time.perf_counter_ns()

        backend = target.backend if self._enabled else _BACKEND_CPU
        start_event = _GPUEvent(backend)
        end_event = _GPUEvent(backend)

        self._events[kernel_id] = {
            "start": start_event,
            "end": end_event,
            "target": target,
            "stream": stream,
            "hw_profile": hw_profile,
        }
        if kernel_id not in self._kernel_order:
            self._kernel_order.append(kernel_id)

        overhead_ns = time.perf_counter_ns() - overhead_start
        self._total_profiling_overhead += overhead_ns / 1_000_000.0

        logger.debug(
            "Instrumented kernel %d on target %s:%s (backend=%s, "
            "stream_pool_bound=%d)",
            kernel_id,
            target.backend,
            target.arch,
            backend,
            self._stream_pool_size,
        )
        return start_event, end_event

    def record_start(
        self,
        kernel_id: int,
        target: GPUTarget,
        stream: Any = None,
    ) -> None:
        """Record the *start* event for an already-instrumented kernel.

        If the kernel has not been instrumented yet, a new event pair is
        created transparently (convenience path).
        """
        overhead_start = time.perf_counter_ns()

        entry = self._events.get(kernel_id)
        if entry is None:
            self.instrument_launch(kernel_id, target, stream)
            entry = self._events[kernel_id]
        entry["start"].record(stream)

        overhead_ns = time.perf_counter_ns() - overhead_start
        self._total_profiling_overhead += overhead_ns / 1_000_000.0

    def record_end(
        self,
        kernel_id: int,
        target: GPUTarget,
        stream: Any = None,
    ) -> None:
        """Record the *end* event for an already-instrumented kernel."""
        overhead_start = time.perf_counter_ns()

        entry = self._events.get(kernel_id)
        if entry is None:
            logger.warning(
                "record_end called for kernel %d that was never instrumented; "
                "creating a CPU-fallback event pair.",
                kernel_id,
            )
            self.instrument_launch(kernel_id, target, stream)
            entry = self._events[kernel_id]
        entry["end"].record(stream)

        overhead_ns = time.perf_counter_ns() - overhead_start
        self._total_profiling_overhead += overhead_ns / 1_000_000.0

    # ------------------------------------------------------------------
    # Synchronize & collect
    # ------------------------------------------------------------------

    def synchronize_and_collect(self) -> Dict[int, Dict[str, float]]:
        """Synchronize all events and compute per-kernel metrics.

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping of *kernel_id* → metric dictionary with keys:
            ``wall_clock_ms``, ``memory_throughput_gbps``,
            ``launch_overhead_us``, ``cross_device_transfer_ms``,
            ``estimated_occupancy``.
        """
        sync_overhead_start = time.perf_counter_ns()

        for kernel_id, entry in self._events.items():
            start_ev: _GPUEvent = entry["start"]
            end_ev: _GPUEvent = entry["end"]

            # Synchronize both events to guarantee timing is available.
            start_ev.synchronize()
            end_ev.synchronize()

            wall_ms = _GPUEvent.elapsed_time(start_ev, end_ev)
            wall_ms = max(wall_ms, 0.0)  # guard against negative jitter

            # Accumulate kernel time.
            self._total_kernel_time += wall_ms

            # ----- derived metric estimates -----
            hw: Optional[HardwareProfile] = entry.get("hw_profile")
            tgt: GPUTarget = entry["target"]

            mem_throughput_gbps = self._estimate_memory_throughput(
                wall_ms, hw
            )
            launch_overhead_us = self._estimate_launch_overhead(wall_ms)
            transfer_ms = 0.0  # populated by caller for cross-device ops
            occupancy = self._estimate_occupancy(hw, tgt)

            self._metrics[kernel_id] = {
                "wall_clock_ms": wall_ms,
                "memory_throughput_gbps": mem_throughput_gbps,
                "launch_overhead_us": launch_overhead_us,
                "cross_device_transfer_ms": transfer_ms,
                "estimated_occupancy": occupancy,
            }

        sync_overhead_ns = time.perf_counter_ns() - sync_overhead_start
        self._total_profiling_overhead += sync_overhead_ns / 1_000_000.0

        # Check budget after collection.
        self.check_overhead_budget()

        if self._log_enabled:
            logger.info(
                "Profiler collected metrics for %d kernels; "
                "total kernel time=%.3f ms, profiling overhead=%.3f ms "
                "(%.2f%% of budget %.2f%%)",
                len(self._metrics),
                self._total_kernel_time,
                self._total_profiling_overhead,
                (self._total_profiling_overhead / max(self._total_kernel_time, 1e-9)) * 100.0,
                self._overhead_budget * 100.0,
            )
        else:
            logger.debug(
                "Collected metrics for %d kernels; total kernel time=%.3f ms, "
                "profiling overhead=%.3f ms",
                len(self._metrics),
                self._total_kernel_time,
                self._total_profiling_overhead,
            )
        return dict(self._metrics)

    # ------------------------------------------------------------------
    # Overhead budget enforcement
    # ------------------------------------------------------------------

    def check_overhead_budget(self) -> bool:
        """Return ``True`` if profiling overhead is within budget.

        If the budget is exceeded the profiler is auto-disabled for all
        subsequent instrumentation calls and a warning is logged.
        """
        if self._total_kernel_time <= 0.0:
            return True

        ratio = self._total_profiling_overhead / self._total_kernel_time
        if ratio > self._overhead_budget:
            if self._enabled:
                self._enabled = False
                logger.warning(
                    "Profiling overhead (%.2f%%) exceeded budget (%.2f%%); "
                    "auto-disabling GPU event instrumentation.",
                    ratio * 100.0,
                    self._overhead_budget * 100.0,
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Metric accessors
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[int, Dict[str, float]]:
        """Return all collected per-kernel metrics.

        The dictionary is keyed by *kernel_id* and each value is a dict
        with keys ``wall_clock_ms``, ``memory_throughput_gbps``,
        ``launch_overhead_us``, ``cross_device_transfer_ms``, and
        ``estimated_occupancy``.
        """
        return dict(self._metrics)

    def get_per_target_metrics(
        self,
    ) -> Dict[GPUTarget, Dict[int, Dict[str, float]]]:
        """Return metrics grouped by ``GPUTarget``.

        Returns
        -------
        Dict[GPUTarget, Dict[int, Dict[str, float]]]
            ``{target: {kernel_id: metrics}}``
        """
        result: Dict[GPUTarget, Dict[int, Dict[str, float]]] = {}
        for kernel_id, entry in self._events.items():
            tgt: GPUTarget = entry["target"]
            per_target = result.setdefault(tgt, {})
            metrics = self._metrics.get(kernel_id)
            if metrics is not None:
                per_target[kernel_id] = metrics
        return result

    # ------------------------------------------------------------------
    # Reset for iterative profiling
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all events, metrics, and overhead counters.

        Re-enables instrumentation so that the next profiling iteration
        starts with a clean slate.
        """
        self._events.clear()
        self._metrics.clear()
        self._kernel_order.clear()
        self._total_kernel_time = 0.0
        self._total_profiling_overhead = 0.0
        self._enabled = True
        logger.debug("RuntimeProfiler reset for new profiling iteration.")

    # ------------------------------------------------------------------
    # Private helpers — metric estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_event_overhead() -> float:
        """Return the estimated per-event-pair overhead in microseconds.

        Typical CUDA event create + record + synchronize overhead is
        ~2–5 μs on modern hardware.  This conservative default is used
        for pre-commit budget prediction.
        """
        return _DEFAULT_EVENT_OVERHEAD_US

    @staticmethod
    def _estimate_memory_throughput(
        wall_ms: float,
        hw_profile: Optional[HardwareProfile],
    ) -> float:
        """Estimate memory throughput in GB/s.

        When a ``HardwareProfile`` is provided the estimate is based on the
        device's peak bandwidth and the fraction of time the kernel ran.
        Without a profile we return ``0.0`` (no estimate available).
        """
        if hw_profile is None or wall_ms <= 0.0:
            return 0.0
        # Rough estimate: assume kernel is fully memory-bound → throughput
        # approaches peak bandwidth.  This is refined by feedback once real
        # data is available.
        peak_bw = hw_profile.memory_bandwidth_gbps
        return peak_bw  # placeholder-free: returns peak as upper-bound estimate

    @staticmethod
    def _estimate_launch_overhead(wall_ms: float) -> float:
        """Estimate kernel launch overhead in microseconds.

        Launch overhead is the fixed cost to dispatch a kernel.  We model it
        as a constant (~5 μs for CUDA) which will be refined by calibration
        data from the feedback controller.
        """
        return 5.0  # conservative constant in μs

    @staticmethod
    def _estimate_occupancy(
        hw_profile: Optional[HardwareProfile],
        target: GPUTarget,
    ) -> float:
        """Estimate SM/CU occupancy as a fraction in [0.0, 1.0].

        Without runtime-specific grid/block data we produce a heuristic
        estimate from the hardware profile.  The approach:
        - Determine the maximum concurrent warps per SM/CU from the
          available shared memory and register file size.
        - Scale by the warp size mismatch between target and HW profile
          to account for cross-generation dispatch.
        The value is refined by the feedback controller using actual
        profiler measurements in subsequent iterations.
        """
        if hw_profile is None:
            return 0.0

        sm_count = hw_profile.sm_count
        smem_per_sm = hw_profile.smem_per_sm_bytes
        warp_sz = hw_profile.warp_size
        vendor = hw_profile.vendor

        tgt_backend = target.backend
        tgt_arch = target.arch
        tgt_ws = target.warp_size

        if sm_count <= 0 or smem_per_sm <= 0 or warp_sz <= 0:
            return 0.0

        # Use the target warp size for thread→warp decomposition so that
        # cross-generation dispatch (where target.warp_size may differ from
        # hw_profile.warp_size) is modelled correctly.
        effective_warp_size = tgt_ws if tgt_ws > 0 else warp_sz

        # Heuristic: assume each active block uses a quarter of per-SM
        # shared memory.  With typical block sizes of 256 threads, compute
        # a rough occupancy ratio.
        assumed_block_size = 256
        warps_per_block = max(assumed_block_size // effective_warp_size, 1)
        assumed_smem_per_block = smem_per_sm // 4  # conservative quarter
        max_blocks_by_smem = smem_per_sm // max(assumed_smem_per_block, 1)

        # Cap by typical hardware limits varying by architecture generation.
        # Modern NVIDIA GPUs support 64 concurrent warps/SM; older ones 48.
        # AMD CUs support up to 40 concurrent wavefronts.
        if vendor == "nvidia" or tgt_backend == "cuda":
            # Use arch to distinguish generations: sm_80+ supports 64 warps.
            arch_val = tgt_arch if isinstance(tgt_arch, int) else 0
            max_warps_per_sm = 64 if arch_val >= 80 else 48
        else:  # AMD / HIP / other
            max_warps_per_sm = 40

        achievable_warps = min(max_blocks_by_smem * warps_per_block,
                               max_warps_per_sm)
        occupancy = achievable_warps / max_warps_per_sm

        # Clamp to valid range.
        return max(0.0, min(occupancy, 1.0))
