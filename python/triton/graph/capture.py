"""Trace capture mechanism for Triton kernel graph recording.

This module provides :class:`KernelGraphCapture`, a context manager that
intercepts kernel launches via ``KernelInterface.__getitem__`` monkey-patching
and records them into a Kernel Graph IR (KGIR) instead of executing them.
Alias analysis and data-dependency detection are performed on the captured
launch sequence to construct the KGIR directed acyclic graph (DAG).

Public API
----------
- :func:`capture` — convenience factory for :class:`KernelGraphCapture`.
- :class:`KernelGraphCapture` — context manager for trace recording.
- :func:`graph_trace` — decorator wrapping a function with graph capture.
- :class:`CapturedLaunch` — metadata record for one intercepted launch.
- :class:`TensorArg` — metadata for a single tensor argument.

Usage::

    import triton
    from triton.graph import capture

    with capture() as ctx:
        kernel_a[grid_a](ptr_out, ptr_in, N, BLOCK=1024)
        kernel_b[grid_b](ptr_out2, ptr_out, M, BLOCK=512)

    graph = ctx._graph  # KGIRGraph with nodes and dependency edges
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from functools import wraps
from threading import local
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Thread-local storage ensures concurrent threads maintain independent
# capture contexts without cross-thread interference.
_capture_state = local()

__all__ = [
    "capture",
    "KernelGraphCapture",
    "graph_trace",
    "CapturedLaunch",
    "TensorArg",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TensorArg:
    """Metadata for a single tensor argument extracted during capture.

    Attributes
    ----------
    arg_index : int
        Positional index in the kernel argument list.  ``-1`` for keyword-
        only tensor arguments.
    data_ptr : int
        Raw data pointer (virtual address) of the tensor storage.
    shape : Tuple[int, ...]
        Tensor dimensions.
    strides : Tuple[int, ...]
        Element strides per dimension (in elements, not bytes).
    dtype : str
        Dtype string (e.g. ``"float32"``, ``"int8"``).
    device : str
        Device identifier (e.g. ``"cuda:0"``).
    """

    arg_index: int
    data_ptr: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str
    device: str


@dataclass
class CapturedLaunch:
    """Record of a single kernel launch captured during graph tracing.

    Attributes
    ----------
    kernel_fn : Any
        Reference to the ``JITFunction`` (or ``Autotuner`` wrapper).
    grid : Any
        Grid parameters — a tuple of ``int`` or a callable.
    args : Tuple
        Positional arguments as passed at the call site.
    kwargs : Dict
        Keyword arguments as passed at the call site.
    tensor_args : List[TensorArg]
        Extracted tensor metadata for every tensor-like argument.
    constexpr_args : Dict
        Compile-time constant arguments extracted from the launch.
    launch_index : int
        Zero-based sequential index tracking capture order.
    """

    kernel_fn: Any
    grid: Any
    args: Tuple
    kwargs: Dict
    tensor_args: List[TensorArg] = field(default_factory=list)
    constexpr_args: Dict = field(default_factory=dict)
    launch_index: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Private Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _try_extract_tensor_info(arg: Any) -> Optional[Dict[str, Any]]:
    """Attempt to extract tensor metadata from *arg* via duck typing.

    Detects PyTorch-style tensors (``data_ptr()``, ``shape``, ``stride()``).
    Returns ``None`` when *arg* does not expose the expected tensor protocol.
    """
    data_ptr_fn = getattr(arg, "data_ptr", None)
    if data_ptr_fn is None or not callable(data_ptr_fn):
        return None

    try:
        ptr = int(data_ptr_fn())
    except (TypeError, RuntimeError, ValueError):
        return None

    # --- shape ---
    raw_shape = getattr(arg, "shape", None)
    shape: Tuple[int, ...] = ()
    if raw_shape is not None:
        try:
            shape = tuple(int(s) for s in raw_shape)
        except (TypeError, ValueError):
            shape = ()

    # --- strides ---
    strides_val: Tuple[int, ...] = ()
    stride_attr = getattr(arg, "stride", None)
    if stride_attr is not None:
        if callable(stride_attr):
            try:
                strides_val = tuple(int(s) for s in stride_attr())
            except (TypeError, RuntimeError, ValueError):
                strides_val = ()
        else:
            try:
                strides_val = tuple(int(s) for s in stride_attr)
            except (TypeError, ValueError):
                strides_val = ()

    # --- dtype ---
    dtype_str = "unknown"
    raw_dtype = getattr(arg, "dtype", None)
    if raw_dtype is not None:
        dtype_str = str(raw_dtype).replace("torch.", "")

    # --- device ---
    device_str = "unknown"
    raw_device = getattr(arg, "device", None)
    if raw_device is not None:
        device_str = str(raw_device)

    return {
        "data_ptr": ptr,
        "shape": shape,
        "strides": strides_val,
        "dtype": dtype_str,
        "device": device_str,
    }


def _kernel_name(kernel: Any) -> str:
    """Best-effort extraction of a human-readable kernel name."""
    fn = getattr(kernel, "fn", kernel)
    name = getattr(fn, "__name__", None)
    if name is not None:
        return str(name)
    name = getattr(fn, "name", None)
    if name is not None:
        return str(name)
    return repr(fn)


def _compute_tensor_span(tensor_arg: TensorArg) -> int:
    """Compute the memory span (in bytes) of a :class:`TensorArg`.

    Uses :func:`~triton.graph.utils.compute_memory_footprint` for strided
    tensors.  Falls back to :func:`~triton.graph.utils.dtype_to_bytes` for
    scalar or zero-dim tensors.
    """
    from triton.graph.utils import dtype_to_bytes as _dtype_to_bytes

    if not tensor_arg.shape or not tensor_arg.strides:
        try:
            elem_size = _dtype_to_bytes(tensor_arg.dtype)
        except (ValueError, KeyError):
            elem_size = 4
        return max(elem_size, 1)

    try:
        from triton.graph.utils import compute_memory_footprint
        return compute_memory_footprint(
            tensor_arg.shape, tensor_arg.strides, tensor_arg.dtype,
        )
    except (ValueError, KeyError):
        try:
            elem_size = _dtype_to_bytes(tensor_arg.dtype)
        except (ValueError, KeyError):
            elem_size = 4
        return max(math.prod(tensor_arg.shape) * elem_size, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# KernelGraphCapture
# ═══════════════════════════════════════════════════════════════════════════════


class KernelGraphCapture:
    """Context manager that intercepts kernel launches and records them into
    a KGIR graph.

    Parameters
    ----------
    config : Optional[GraphConfig]
        Optional graph configuration overrides.  When ``None``, defaults are
        loaded from environment variables via ``GraphConfig.from_env()``.

    Attributes
    ----------
    _captured_launches : List[CapturedLaunch]
        Raw captured kernel launch records, in capture order.
    _graph : Optional[KGIRGraph]
        The constructed KGIR graph.  ``None`` until :meth:`build_graph`
        completes (called automatically on clean context-manager exit).

    Examples
    --------
    >>> with KernelGraphCapture() as ctx:
    ...     my_kernel[grid](out, inp, N, BLOCK=1024)
    >>> graph = ctx._graph
    """

    __slots__ = (
        "_config",
        "_captured_launches",
        "_graph",
        "_hardware_inventory",
        "_original_getitem",
        "_interceptor_fn",
        "_active",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[Any] = None) -> None:
        # Lazy import to prevent circular dependency chains.
        from triton.graph.config import GraphConfig

        self._config: Any = config if config is not None else GraphConfig.from_env()
        self._captured_launches: List[CapturedLaunch] = []
        self._graph: Optional[Any] = None
        self._hardware_inventory: Optional[Any] = None
        self._original_getitem: Optional[Callable] = None
        self._interceptor_fn: Optional[Callable] = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Compatibility aliases
    # ------------------------------------------------------------------

    @property
    def _launches(self) -> List[CapturedLaunch]:
        """Alias for ``_captured_launches`` for backwards compatibility."""
        return self._captured_launches

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> KernelGraphCapture:
        """Enter the capture context.

        1. Guard against nested captures on the same thread.
        2. Enumerate hardware inventory (< 10 ms budget).
        3. Save and replace ``KernelInterface.__getitem__`` with a
           recording interceptor.
        """
        if getattr(_capture_state, "active_capture", None) is not None:
            from triton.graph.errors import GraphCaptureError

            raise GraphCaptureError(
                "Nested KernelGraphCapture contexts are not supported. "
                "A capture context is already active on this thread."
            )

        # ---- Hardware inventory (must complete in < 10 ms) ----
        t0 = time.monotonic()
        try:
            from triton.graph.dispatch import HardwareInventory

            self._hardware_inventory = HardwareInventory()
        except Exception:
            # Hardware enumeration failure is non-fatal — capture can still
            # proceed but the resulting KGIR graph will have no hardware
            # profile annotations.
            logger.debug("Hardware inventory enumeration failed", exc_info=True)
            self._hardware_inventory = None
        hw_ms = (time.monotonic() - t0) * 1000.0
        logger.debug("Hardware inventory completed in %.2f ms", hw_ms)

        # ---- Monkey-patch KernelInterface.__getitem__ ----
        from triton.runtime.jit import KernelInterface

        self._original_getitem = KernelInterface.__getitem__
        self._interceptor_fn = self._create_interceptor()
        KernelInterface.__getitem__ = self._interceptor_fn  # type: ignore[assignment]

        self._active = True
        _capture_state.active_capture = self
        logger.debug("Kernel graph capture started")
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the capture context.

        Restores the original ``KernelInterface.__getitem__``.  On a clean
        exit with at least one captured launch, :meth:`build_graph` is
        called automatically.
        """
        try:
            # ---- Restore original __getitem__ ----
            if self._original_getitem is not None:
                from triton.runtime.jit import KernelInterface

                KernelInterface.__getitem__ = self._original_getitem  # type: ignore[assignment]
                self._original_getitem = None

            # ---- Build graph on clean exit ----
            if exc_type is None and self._captured_launches:
                self.build_graph()
            elif exc_type is None and not self._captured_launches:
                # Empty capture — create an empty graph.
                from triton.graph.kgir import KGIRGraph

                hw = []
                if self._hardware_inventory is not None:
                    hw = self._hardware_inventory.devices
                self._graph = KGIRGraph(hardware_profiles=hw if hw else None)
        finally:
            self._active = False
            _capture_state.active_capture = None
            self._interceptor_fn = None
            logger.debug(
                "Kernel graph capture ended — %d launches recorded",
                len(self._captured_launches),
            )

    # ------------------------------------------------------------------
    # Interceptor factory
    # ------------------------------------------------------------------

    def _create_interceptor(self) -> Callable:
        """Build the replacement for ``KernelInterface.__getitem__``.

        The returned callable has the same signature as the original
        ``__getitem__(self, grid)`` and returns a recording callable that
        captures launch arguments instead of dispatching to the GPU.
        """
        capture_ctx = self  # closure reference for the recording callable

        def _recording_getitem(kernel_self: Any, grid: Any) -> Callable:
            """Drop-in replacement for ``KernelInterface.__getitem__``."""

            def _record(*args: Any, **kwargs: Any) -> None:
                capture_ctx._record_launch(kernel_self, grid, args, kwargs)

            return _record

        return _recording_getitem

    # ------------------------------------------------------------------
    # Launch recording
    # ------------------------------------------------------------------

    def _record_launch(
        self,
        kernel_self: Any,
        grid: Any,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Capture a single kernel launch without executing it.

        If *kernel_self* is a cold Autotuner, a single warmup run is
        triggered first so that the best autotuning configuration is
        available for capture.
        """
        # Handle autotuned kernels that have not been benchmarked yet.
        self._handle_autotuner_warmup(kernel_self, grid, args, kwargs)

        tensor_args = self._extract_tensor_args(args, kwargs)
        constexpr_args = self._extract_constexpr_args(kernel_self, args, kwargs)

        # Normalise grid to a tuple when it is a plain integer.
        if isinstance(grid, int):
            normalised_grid: Any = (grid,)
        elif isinstance(grid, tuple):
            normalised_grid = grid
        else:
            # Callable or other exotic grid — store as-is.
            normalised_grid = grid

        launch = CapturedLaunch(
            kernel_fn=kernel_self,
            grid=normalised_grid,
            args=args,
            kwargs=kwargs,
            tensor_args=tensor_args,
            constexpr_args=constexpr_args,
            launch_index=len(self._captured_launches),
        )
        self._captured_launches.append(launch)
        logger.debug(
            "Captured launch #%d: %s",
            launch.launch_index,
            _kernel_name(kernel_self),
        )

    # ------------------------------------------------------------------
    # Autotuner interaction
    # ------------------------------------------------------------------

    def _handle_autotuner_warmup(
        self,
        kernel_self: Any,
        grid: Any,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """If *kernel_self* is a cold Autotuner, trigger a single autotuning
        run so that ``best_config`` is available for capture.

        A warm Autotuner (``best_config`` already set) or a plain
        ``JITFunction`` is silently ignored.

        Thread-safety note: during the warmup window the original
        ``__getitem__`` is temporarily restored.  Launches on other threads
        during this brief window will execute normally rather than being
        captured — an acceptable trade-off given the rarity of cold
        autotuner warmup during graph capture.
        """
        # Duck-type detection: Autotuner has .configs and .fn attributes.
        if not (hasattr(kernel_self, "configs") and hasattr(kernel_self, "fn")):
            return  # Not an autotuner — nothing to do.

        if getattr(kernel_self, "best_config", None) is not None:
            return  # Already warmed up — nothing to do.

        logger.debug(
            "Cold autotuner detected — triggering warmup run for %s",
            _kernel_name(kernel_self),
        )

        from triton.runtime.jit import KernelInterface

        # Temporarily restore the real __getitem__ so the autotuner can
        # perform a genuine benchmark pass on the GPU.
        KernelInterface.__getitem__ = self._original_getitem  # type: ignore[assignment]
        try:
            launcher = kernel_self[grid]
            launcher(*args, **kwargs)
        except Exception:
            logger.warning(
                "Autotuner warmup failed for %s — recording without config",
                _kernel_name(kernel_self),
                exc_info=True,
            )
        finally:
            # Re-install the recording interceptor regardless of outcome.
            KernelInterface.__getitem__ = self._interceptor_fn  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Tensor argument extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tensor_args(
        args: tuple,
        kwargs: dict,
    ) -> List[TensorArg]:
        """Identify tensor-like arguments by duck typing and extract metadata.

        Both positional and keyword arguments are inspected.  Keyword tensor
        arguments receive ``arg_index = -1``.
        """
        tensors: List[TensorArg] = []

        for idx, arg in enumerate(args):
            info = _try_extract_tensor_info(arg)
            if info is not None:
                tensors.append(
                    TensorArg(
                        arg_index=idx,
                        data_ptr=info["data_ptr"],
                        shape=info["shape"],
                        strides=info["strides"],
                        dtype=info["dtype"],
                        device=info["device"],
                    )
                )

        for _key, arg in kwargs.items():
            info = _try_extract_tensor_info(arg)
            if info is not None:
                tensors.append(
                    TensorArg(
                        arg_index=-1,
                        data_ptr=info["data_ptr"],
                        shape=info["shape"],
                        strides=info["strides"],
                        dtype=info["dtype"],
                        device=info["device"],
                    )
                )

        return tensors

    # ------------------------------------------------------------------
    # Constexpr argument extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_constexpr_args(
        kernel_self: Any,
        args: tuple,
        kwargs: dict,
    ) -> Dict[str, Any]:
        """Extract compile-time constant arguments.

        If the kernel exposes a ``params`` attribute (``JITFunction.params``),
        its ``is_constexpr`` flags are used to identify constexpr slots.
        Otherwise all non-tensor, non-callable positional/keyword arguments
        are treated as potential constexprs.

        For Autotuner-wrapped kernels, the ``best_config`` keyword overrides
        are merged into the returned dictionary.
        """
        constexprs: Dict[str, Any] = {}

        # Resolve the underlying JITFunction if wrapped by an Autotuner.
        fn = getattr(kernel_self, "fn", kernel_self)
        params = getattr(fn, "params", None)

        if params is not None:
            for i, param in enumerate(params):
                is_cexpr = getattr(param, "is_constexpr", False)
                if is_cexpr:
                    name = getattr(param, "name", f"arg{i}")
                    if i < len(args):
                        constexprs[name] = args[i]
                    elif name in kwargs:
                        constexprs[name] = kwargs[name]
        else:
            # Fallback: collect all non-tensor scalar arguments.
            for i, arg in enumerate(args):
                if _try_extract_tensor_info(arg) is None and not callable(arg):
                    constexprs[f"arg{i}"] = arg
            for key, val in kwargs.items():
                if _try_extract_tensor_info(val) is None and not callable(val):
                    constexprs[key] = val

        # Merge autotuner best-config kwargs as constexprs.
        best_config = getattr(kernel_self, "best_config", None)
        if best_config is not None:
            all_kwargs_fn = getattr(best_config, "all_kwargs", None)
            if all_kwargs_fn is not None and callable(all_kwargs_fn):
                config_kw = all_kwargs_fn()
                if isinstance(config_kw, dict):
                    constexprs.update(config_kw)

        return constexprs

    # ------------------------------------------------------------------
    # Alias analysis
    # ------------------------------------------------------------------

    def _analyze_aliases(self) -> Dict[int, List[int]]:
        """Map canonical data pointers to launch indices that reference them.

        Two tensor arguments are considered *aliased* when their backing
        memory regions overlap, as determined by
        :func:`~triton.graph.utils.memory_regions_overlap`.

        Returns
        -------
        Dict[int, List[int]]
            ``{canonical_ptr: [launch_index, ...]}`` for every pointer
            that appears in more than one launch (or overlaps with
            another pointer).
        """
        from triton.graph.utils import memory_regions_overlap

        # Collect (data_ptr, span_bytes, launch_index) for every tensor arg.
        regions: List[Tuple[int, int, int]] = []
        for launch in self._captured_launches:
            for t in launch.tensor_args:
                span = _compute_tensor_span(t)
                regions.append((t.data_ptr, span, launch.launch_index))

        # Build alias groups via pairwise overlap check.
        # Complexity: O(R²) where R = total tensor arg count.  R is small
        # for typical graphs (≤50 kernels × ~5 tensors = ≤250).
        alias_map: Dict[int, List[int]] = {}
        n = len(regions)
        for i in range(n):
            ptr_i, size_i, li = regions[i]
            # Ensure the pointer itself is registered.
            if ptr_i not in alias_map:
                alias_map[ptr_i] = []
            if li not in alias_map[ptr_i]:
                alias_map[ptr_i].append(li)

            for j in range(i + 1, n):
                ptr_j, size_j, lj = regions[j]
                if li == lj:
                    continue  # Same launch — no inter-kernel dependency.
                if memory_regions_overlap(ptr_i, size_i, ptr_j, size_j):
                    canonical = min(ptr_i, ptr_j)
                    if canonical not in alias_map:
                        alias_map[canonical] = []
                    if li not in alias_map[canonical]:
                        alias_map[canonical].append(li)
                    if lj not in alias_map[canonical]:
                        alias_map[canonical].append(lj)

        return alias_map

    # ------------------------------------------------------------------
    # Dependency detection
    # ------------------------------------------------------------------

    def _detect_data_dependencies(
        self,
    ) -> List[Tuple[int, int, str, Optional[int]]]:
        """Detect data and anti-dependencies between captured launches.

        For every pair of launches ``(A, B)`` where ``A`` precedes ``B``
        in capture order and both reference an overlapping tensor region,
        a ``"data_dep"`` edge is emitted from ``A`` to ``B``.

        This is conservative: since read/write roles are indistinguishable
        at capture time, *all* potential RAW and WAR hazards are covered.

        Returns
        -------
        List[Tuple[int, int, str, Optional[int]]]
            ``(source_launch_idx, target_launch_idx, edge_type, tensor_ptr)``
        """
        alias_map = self._analyze_aliases()
        edges: List[Tuple[int, int, str, Optional[int]]] = []
        seen: set = set()

        for ptr, launch_indices in alias_map.items():
            if len(launch_indices) < 2:
                continue
            sorted_indices = sorted(launch_indices)
            for i, src in enumerate(sorted_indices):
                for dst in sorted_indices[i + 1 :]:
                    key = (src, dst)
                    if key not in seen:
                        edges.append((src, dst, "data_dep", ptr))
                        seen.add(key)

        return edges

    # ------------------------------------------------------------------
    # Unsupported pattern detection
    # ------------------------------------------------------------------

    def _detect_unsupported_patterns(self) -> List[str]:
        """Detect patterns that the graph optimisation layer cannot handle.

        Currently detected (best-effort, non-blocking):

        1. Callable grids that can only be resolved at execution time.
        2. Zero-element tensors that may indicate scalar results used in
           host-side conditionals.
        3. Tensor arguments with missing shape/stride metadata.

        Returns
        -------
        List[str]
            Descriptive messages for each detected issue.
        """
        issues: List[str] = []

        for launch in self._captured_launches:
            kname = _kernel_name(launch.kernel_fn)

            # Callable grids (informational, not blocking).
            if callable(launch.grid):
                issues.append(
                    f"Launch #{launch.launch_index} ({kname}) has a callable "
                    f"grid that will be resolved at execution time."
                )

            for t in launch.tensor_args:
                # Zero-element tensors — possible host-dependent control flow.
                if t.shape == () or (t.shape and all(s == 0 for s in t.shape)):
                    issues.append(
                        f"Launch #{launch.launch_index} ({kname}) has a "
                        f"zero-element tensor at arg index {t.arg_index} — "
                        f"may indicate host-dependent control flow."
                    )

                # Missing shape/stride metadata.
                if not t.shape and not t.strides:
                    issues.append(
                        f"Launch #{launch.launch_index} ({kname}) tensor at "
                        f"arg index {t.arg_index} has no shape/stride metadata."
                    )

        return issues

    # ------------------------------------------------------------------
    # KGIR graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> Any:
        """Construct a :class:`KGIRGraph` from captured launches and
        detected dependencies.

        Performs alias analysis, inter-kernel data dependency detection,
        unsupported-pattern checking, and assembles a complete KGIR
        directed acyclic graph annotated with hardware profiles.

        The resulting graph is stored in :attr:`_graph` and also returned.

        Returns
        -------
        KGIRGraph
            The fully constructed kernel dependency graph.

        Raises
        ------
        GraphCaptureError
            If a critical, unrecoverable error occurs during graph
            construction (e.g. cyclic dependency detected).
        """
        from triton.graph.kgir import KGIRGraph

        t0 = time.monotonic()

        # ---- Unsupported pattern check (warnings only) ----
        issues = self._detect_unsupported_patterns()
        for msg in issues:
            logger.warning("Unsupported pattern: %s", msg)

        # ---- Hardware profiles ----
        hw_profiles: List[Any] = []
        if self._hardware_inventory is not None:
            hw_profiles = list(self._hardware_inventory.devices)

        # ---- Create graph ----
        graph = KGIRGraph(
            hardware_profiles=hw_profiles if hw_profiles else None,
        )

        # ---- Add nodes (one per captured launch) ----
        node_id_map: Dict[int, int] = {}

        for launch in self._captured_launches:
            metadata = self._build_node_metadata(launch)
            node_id = graph.add_node(
                kernel_fn=launch.kernel_fn,
                metadata=metadata,
            )
            node_id_map[launch.launch_index] = node_id

        # ---- Add edges (from dependency analysis) ----
        dependencies = self._detect_data_dependencies()
        edge_count = 0
        for src_launch, dst_launch, edge_type, tensor_ptr in dependencies:
            src_node = node_id_map.get(src_launch)
            dst_node = node_id_map.get(dst_launch)
            if src_node is not None and dst_node is not None:
                try:
                    graph.add_edge(
                        source_id=src_node,
                        target_id=dst_node,
                        edge_type=edge_type,
                        tensor_id=tensor_ptr,
                    )
                    edge_count += 1
                except ValueError as exc:
                    # Edge would create a cycle — log and skip.
                    logger.warning(
                        "Skipped edge %d→%d (%s): %s",
                        src_launch, dst_launch, edge_type, exc,
                    )

        self._graph = graph

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "KGIR graph built in %.2f ms — %d nodes, %d edges",
            elapsed_ms,
            len(self._captured_launches),
            edge_count,
        )

        return graph

    # ------------------------------------------------------------------
    # Node metadata builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_node_metadata(launch: CapturedLaunch) -> Any:
        """Construct a :class:`NodeMetadata` from a :class:`CapturedLaunch`.

        Tensor shapes, strides, dtypes, and conservative read-write access
        patterns are derived from the captured tensor arguments.  Grid
        dimensions are extracted from the launch grid when it is a concrete
        tuple.
        """
        from triton.graph.kgir import NodeMetadata

        shapes: Dict[int, Tuple[int, ...]] = {}
        strides_map: Dict[int, Tuple[int, ...]] = {}
        dtypes: Dict[int, str] = {}
        access_patterns: Dict[str, str] = {}

        for t in launch.tensor_args:
            idx = t.arg_index
            shapes[idx] = t.shape
            strides_map[idx] = t.strides
            dtypes[idx] = t.dtype
            # Conservative: assume every pointer is both read and written.
            access_patterns[f"arg{idx}"] = "read_write"

        # Grid dimensions — pad to 3D, clip to 3D.
        if isinstance(launch.grid, tuple):
            grid_tuple = launch.grid
            padded = grid_tuple + (1,) * max(0, 3 - len(grid_tuple))
            grid_dims: Tuple[int, ...] = padded[:3]
        else:
            grid_dims = (1, 1, 1)

        num_warps = launch.constexpr_args.get("num_warps", 4)
        if not isinstance(num_warps, int):
            num_warps = 4

        return NodeMetadata(
            memory_access_patterns=access_patterns,
            tensor_shapes=shapes,
            tensor_strides=strides_map,
            tensor_dtypes=dtypes,
            grid_dimensions=grid_dims,
            shared_memory_bytes=0,
            register_count=0,
            num_warps=num_warps,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# graph_trace Decorator
# ═══════════════════════════════════════════════════════════════════════════════


def graph_trace(
    fn: Optional[Callable] = None,
    *,
    config: Optional[Any] = None,
) -> Any:
    """Decorator that wraps a function with kernel graph capture.

    May be applied with or without arguments::

        @graph_trace
        def my_pipeline(x, y):
            kernel_a[grid](...)
            kernel_b[grid](...)

        @graph_trace(config=GraphConfig(...))
        def my_pipeline(x, y):
            ...

    After invocation the wrapper exposes two extra attributes:

    ``wrapper.last_graph``
        The :class:`KGIRGraph` from the most recent call.
    ``wrapper.last_captures``
        The raw :class:`CapturedLaunch` list from the most recent call.

    The original function's return value is preserved unchanged.

    Parameters
    ----------
    fn : Optional[Callable]
        The function to decorate.  When ``None``, returns a partial
        decorator (for the ``@graph_trace(config=...)`` form).
    config : Optional[GraphConfig]
        Optional graph configuration overrides.
    """
    if fn is None:
        # Called as ``@graph_trace(config=...)`` — return partial decorator.
        from functools import partial

        return partial(graph_trace, config=config)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = KernelGraphCapture(config=config)
        with ctx:
            result = fn(*args, **kwargs)
        wrapper.last_graph = ctx._graph  # type: ignore[attr-defined]
        wrapper.last_captures = list(ctx._captured_launches)  # type: ignore[attr-defined]
        return result

    # Initialise decorator attributes.
    wrapper.last_graph = None  # type: ignore[attr-defined]
    wrapper.last_captures = []  # type: ignore[attr-defined]
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Public Convenience Function
# ═══════════════════════════════════════════════════════════════════════════════


def capture(config: Optional[Any] = None) -> KernelGraphCapture:
    """Create a :class:`KernelGraphCapture` context manager.

    This is the primary public entry-point for graph-level trace capture.

    Parameters
    ----------
    config : Optional[GraphConfig]
        Optional configuration overrides.  When ``None``, defaults are
        loaded from environment variables via ``GraphConfig.from_env()``.

    Returns
    -------
    KernelGraphCapture
        A context manager ready for use in a ``with`` statement.

    Examples
    --------
    >>> with capture() as ctx:
    ...     my_kernel[grid](ptr, N, BLOCK=1024)
    >>> graph = ctx._graph
    """
    return KernelGraphCapture(config=config)
