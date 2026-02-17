"""
Graph-specific exception hierarchy for Triton's graph-level optimization layer.

All exception classes inherit from ``triton.errors.TritonError`` to maintain
compatibility with the existing Triton error-handling pattern.  Each class
carries an optional human-readable ``error_message`` and, where appropriate,
additional context parameters (e.g. iteration count, device identifiers).

Classes
-------
GraphCaptureError
    Raised when kernel graph trace capture fails.
FusionError
    Raised when the fusion analysis engine encounters a fatal error.
DispatchError
    Raised when hardware-aware dispatch fails (e.g. no suitable devices).
ConvergenceError
    Raised when the closed-loop feedback controller fails to converge
    within the configured iteration limit.
TransferError
    Raised when a cross-device data transfer fails.
"""

from typing import Optional

from ..errors import TritonError


# ---------------------------------------------------------------------------
# GraphCaptureError
# ---------------------------------------------------------------------------

class GraphCaptureError(TritonError):
    """Raised when kernel graph trace capture fails.

    This error surfaces problems during the ``triton.graph.capture()`` context
    manager scope — for example, unsupported host-side control flow that
    depends on kernel output, or failure to enumerate the hardware inventory.

    Parameters
    ----------
    error_message : str or None
        Optional description of the capture failure.
    """

    def __init__(self, error_message: Optional[str] = None) -> None:
        self.error_message = error_message
        super().__init__(str(self))

    def __str__(self) -> str:
        return self.error_message or "Graph capture error"


# ---------------------------------------------------------------------------
# FusionError
# ---------------------------------------------------------------------------

class FusionError(TritonError):
    """Raised when fusion analysis encounters a fatal error.

    Covers failures in both producer-consumer and sibling/horizontal fusion
    passes, including resource-budget overflows, tiling incompatibility, and
    cost-model evaluation errors.

    Parameters
    ----------
    error_message : str or None
        Optional description of the fusion failure.
    """

    def __init__(self, error_message: Optional[str] = None) -> None:
        self.error_message = error_message
        super().__init__(str(self))

    def __str__(self) -> str:
        return self.error_message or "Fusion analysis error"


# ---------------------------------------------------------------------------
# DispatchError
# ---------------------------------------------------------------------------

class DispatchError(TritonError):
    """Raised when hardware-aware dispatch fails.

    Typical triggers include an empty hardware inventory (no suitable devices
    found), incompatible dispatch-mode constraints, or failure to compile a
    subgraph for any eligible target.

    Parameters
    ----------
    error_message : str or None
        Optional description of the dispatch failure.
    """

    def __init__(self, error_message: Optional[str] = None) -> None:
        self.error_message = error_message
        super().__init__(str(self))

    def __str__(self) -> str:
        return self.error_message or "Hardware dispatch error"


# ---------------------------------------------------------------------------
# ConvergenceError
# ---------------------------------------------------------------------------

class ConvergenceError(TritonError):
    """Raised when the feedback loop fails to converge within the iteration limit.

    The closed-loop feedback controller triggers this error when the maximum
    number of re-optimization iterations (``TRITON_FEEDBACK_MAX_ITERS``,
    default 20) is exhausted without achieving convergence (decision changes
    < 2% across consecutive iterations).

    Parameters
    ----------
    error_message : str or None
        Optional description of the convergence failure.
    iterations : int or None
        Number of iterations completed before the error was raised.
    """

    def __init__(
        self,
        error_message: Optional[str] = None,
        iterations: Optional[int] = None,
    ) -> None:
        self.error_message = error_message
        self.iterations = iterations
        super().__init__(str(self))

    def __str__(self) -> str:
        msg = self.error_message or "Convergence error"
        if self.iterations is not None:
            msg += f" (after {self.iterations} iterations)"
        return msg


# ---------------------------------------------------------------------------
# TransferError
# ---------------------------------------------------------------------------

class TransferError(TritonError):
    """Raised when a cross-device data transfer fails.

    This error is produced by the memory-planning or dispatch layers when an
    inter-device transfer operation cannot be executed — for example, because
    peer-to-peer access is unavailable or the interconnect is saturated.

    Parameters
    ----------
    error_message : str or None
        Optional description of the transfer failure.
    source_device : str or None
        Identifier of the source device (e.g. ``"cuda:0"``).
    target_device : str or None
        Identifier of the target device (e.g. ``"cuda:1"``).
    """

    def __init__(
        self,
        error_message: Optional[str] = None,
        source_device: Optional[str] = None,
        target_device: Optional[str] = None,
    ) -> None:
        self.error_message = error_message
        self.source_device = source_device
        self.target_device = target_device
        super().__init__(str(self))

    def __str__(self) -> str:
        msg = self.error_message or "Cross-device transfer error"
        if self.source_device and self.target_device:
            msg += f" (from {self.source_device} to {self.target_device})"
        return msg
