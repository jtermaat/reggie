"""Visualization callback system for agent operations.

This module provides a callback-based mechanism for emitting
visualization data during agent operations.
"""

from typing import Optional, Callable, Dict, Any

# Global visualization callback
_visualization_callback: Optional[Callable[[Dict[str, Any]], None]] = None


def set_visualization_callback(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Set the callback function for visualization data.

    Args:
        callback: Function that takes visualization data dict
    """
    global _visualization_callback
    _visualization_callback = callback


def clear_visualization_callback() -> None:
    """Clear the visualization callback."""
    global _visualization_callback
    _visualization_callback = None


def emit_visualization(data: Dict[str, Any]) -> None:
    """Emit visualization data.

    Args:
        data: Visualization data dictionary containing type and relevant data
    """
    if _visualization_callback is not None:
        _visualization_callback(data)
