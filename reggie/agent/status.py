"""Status reporting for agent operations.

This module provides a simple callback-based mechanism for emitting
status updates during agent operations
"""

from typing import Optional, Callable

# Global status callback
_status_callback: Optional[Callable[[str], None]] = None


def set_status_callback(callback: Callable[[str], None]) -> None:
    """Set the callback function for status updates.

    Args:
        callback: Function that takes a status message string
    """
    global _status_callback
    _status_callback = callback


def clear_status_callback() -> None:
    """Clear the status callback."""
    global _status_callback
    _status_callback = None


def emit_status(message: str) -> None:
    """Emit a status update message.

    Args:
        message: Status message to emit
    """
    if _status_callback is not None:
        _status_callback(message)
