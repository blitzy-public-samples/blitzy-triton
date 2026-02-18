"""
Triton Graph-Level Optimization Package
========================================

This package provides graph-level cross-kernel optimization for Triton,
operating above the existing single-kernel compilation pipeline. It enables
cross-kernel fusion, inter-kernel scheduling, global memory planning,
multi-target hardware-aware dispatch, and a closed-loop runtime feedback
mechanism.

**Opt-in only:** Existing Triton programs behave identically without
modification. Users opt in via the explicit ``triton.graph.capture()``
context manager or the ``@graph_trace`` decorator.

Public API
----------
Trace Capture:
    capture          – Convenience function returning a KernelGraphCapture
                       context manager.
    KernelGraphCapture – Context manager that intercepts kernel launches
                         and records them into a KGIR graph.
    graph_trace      – Decorator wrapping a function with automatic kernel
                       graph capture.

Configuration:
    GraphConfig      – Top-level configuration dataclass.
    DispatchConfig   – Dispatch layer settings.
    FeedbackConfig   – Closed-loop feedback settings.
    FusionConfig     – Fusion engine settings.

Dispatch:
    DispatchMode     – Enum selecting dispatch optimisation objective
                       (PERFORMANCE, COST, BALANCED).

Errors:
    GraphCaptureError  – Trace capture failures.
    FusionError        – Fusion analysis failures.
    DispatchError      – No suitable devices / dispatch failures.
    ConvergenceError   – Feedback loop convergence failures.
    TransferError      – Cross-device transfer failures.

TorchInductor Integration:
    submit_kernel_graph – Submit a kernel graph for optimisation.
    KernelGraphResult   – Result containing optimised launch sequence.
    DevicePlacement     – Device placement preference with affinity.

Example
-------
::

    import triton

    with triton.graph.capture() as ctx:
        kernel_a[grid_a](*args_a)
        kernel_b[grid_b](*args_b)

    graph = ctx.build_graph()
"""

# ---------------------------------------------------------------------------
# Re-exports: Trace Capture
# ---------------------------------------------------------------------------
from .capture import (
    capture,
    KernelGraphCapture,
    graph_trace,
)

# ---------------------------------------------------------------------------
# Re-exports: Configuration Dataclasses
# ---------------------------------------------------------------------------
from .config import (
    GraphConfig,
    DispatchConfig,
    FeedbackConfig,
    FusionConfig,
)

# ---------------------------------------------------------------------------
# Re-exports: Dispatch Mode Enum
# ---------------------------------------------------------------------------
from .dispatch import DispatchMode

# ---------------------------------------------------------------------------
# Re-exports: Error Hierarchy
# ---------------------------------------------------------------------------
from .errors import (
    GraphCaptureError,
    FusionError,
    DispatchError,
    ConvergenceError,
    TransferError,
)

# ---------------------------------------------------------------------------
# Re-exports: TorchInductor Integration Surface
# ---------------------------------------------------------------------------
from .torch_inductor_api import (
    submit_kernel_graph,
    KernelGraphResult,
    DevicePlacement,
)

# ---------------------------------------------------------------------------
# Public API — alphabetically sorted per Triton convention
# ---------------------------------------------------------------------------
__all__ = [
    "capture",
    "ConvergenceError",
    "DevicePlacement",
    "DispatchConfig",
    "DispatchError",
    "DispatchMode",
    "FeedbackConfig",
    "FusionConfig",
    "FusionError",
    "graph_trace",
    "GraphCaptureError",
    "GraphConfig",
    "KernelGraphCapture",
    "KernelGraphResult",
    "submit_kernel_graph",
    "TransferError",
]
