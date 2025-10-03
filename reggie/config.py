"""Configuration and environment setup for Reggie"""

import os
from typing import Optional


def setup_langsmith(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
) -> None:
    """Configure LangSmith tracing.

    Args:
        api_key: LangSmith API key. If None, reads from LANGSMITH_API_KEY env var.
        project: LangSmith project name. If None, reads from LANGSMITH_PROJECT env var
                 or defaults to 'reggie'.
    """
    # Set LangSmith API key
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
    elif not os.getenv("LANGSMITH_API_KEY"):
        # LangSmith is optional - just disable tracing if not configured
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    # Set project name
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
    elif not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "reggie"

    # Enable tracing
    os.environ["LANGSMITH_TRACING"] = "true"


# Auto-configure on import
setup_langsmith()
