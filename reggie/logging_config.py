"""Centralized logging configuration for Reggie"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. If None, uses default format.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )

    # Set specific logger levels if needed
    # For example, reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
