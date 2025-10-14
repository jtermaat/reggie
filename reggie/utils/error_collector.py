"""Error collection and aggregation for pipeline operations."""

import asyncio
from typing import Dict, List, Optional, Any
from collections import defaultdict


class ErrorCollector:
    """Collects and aggregates errors during pipeline operations.

    This class provides a way to collect errors without immediately logging them,
    allowing for clean progress displays. Errors are aggregated by type and can
    be displayed as a summary at the end of an operation.
    """

    def __init__(self, max_examples_per_type: int = 3):
        """Initialize the error collector.

        Args:
            max_examples_per_type: Maximum number of example errors to keep per type
        """
        self.max_examples_per_type = max_examples_per_type
        self._errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def collect_async(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect an error asynchronously (thread-safe).

        Args:
            error_type: Type/category of the error (e.g., "Categorization Validation")
            message: Error message
            context: Optional context information (e.g., comment_id, comment_preview)
        """
        async with self._lock:
            self._collect_internal(error_type, message, context)

    def collect(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect an error synchronously.

        Args:
            error_type: Type/category of the error (e.g., "Categorization Validation")
            message: Error message
            context: Optional context information (e.g., comment_id, comment_preview)
        """
        self._collect_internal(error_type, message, context)

    def _collect_internal(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Internal method to collect an error.

        Args:
            error_type: Type/category of the error
            message: Error message
            context: Optional context information
        """
        # Only keep up to max_examples_per_type for each error type
        if len(self._errors[error_type]) < self.max_examples_per_type:
            self._errors[error_type].append({
                "message": message,
                "context": context or {}
            })

    def has_errors(self) -> bool:
        """Check if any errors have been collected.

        Returns:
            True if errors exist, False otherwise
        """
        return len(self._errors) > 0

    def get_total_count(self) -> int:
        """Get total number of error types collected.

        Returns:
            Total number of distinct error types
        """
        return len(self._errors)

    def get_summary(self, max_examples: Optional[int] = None) -> str:
        """Generate a formatted summary of collected errors.

        Args:
            max_examples: Override max examples per type (default: use constructor value)

        Returns:
            Rich-formatted string with error summary, or empty string if no errors
        """
        if not self.has_errors():
            return ""

        max_ex = max_examples if max_examples is not None else self.max_examples_per_type

        # Count total errors (considering we only kept examples)
        total_errors = sum(len(errors) for errors in self._errors.values())

        # Build summary
        lines = [f"[yellow]⚠ Encountered errors during processing:[/yellow]\n"]

        for error_type, error_list in sorted(self._errors.items()):
            count = len(error_list)
            lines.append(f"  [bold]{error_type}[/bold] ([yellow]{count}[/yellow] error{'s' if count > 1 else ''}):")

            # Show examples (up to max_ex)
            for i, error_info in enumerate(error_list[:max_ex]):
                message = error_info["message"]
                # Clean up the message - extract the key part from validation errors
                if "validation error" in message.lower():
                    # Extract the actual validation message
                    parts = message.split("\n")
                    # Find lines with actual error details
                    relevant_parts = [p.strip() for p in parts if p.strip() and not p.strip().startswith("1 validation")]
                    if relevant_parts:
                        message = relevant_parts[-1]  # Usually the last line has the key info

                lines.append(f"    [dim]•[/dim] {message}")

            # If there are more errors than examples, indicate that
            if count > max_ex:
                lines.append(f"    [dim]... and {count - max_ex} more[/dim]")

            lines.append("")  # Empty line between error types

        # Add guidance for debugging
        lines.append("  [dim]Run with LOG_LEVEL=DEBUG to see full error details in logs.[/dim]")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all collected errors."""
        self._errors.clear()
