"""Progress display for streaming document operations using Rich.

This module provides a unified progress display for the streaming operation
that combines document loading and comment processing in a single pass,
with real-time cost tracking.
"""

import logging
from typing import Optional, Callable
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from .progress import ProgressDisplay
from ..models.cost import CostReport


class StreamingProgressDisplay(ProgressDisplay):
    """Progress display for streaming document operations.

    Shows progress for:
    1. Metadata fetching (indeterminate)
    2. Streaming comments (determinate with count, chunks, cost, ETA)
    """

    def __init__(self, document_id: str, console: Optional[Console] = None):
        """Initialize the streaming progress display.

        Args:
            document_id: Document ID being streamed
            console: Optional Rich console instance
        """
        super().__init__(console)
        self.document_id = document_id
        self.metadata_task_id: Optional[int] = None
        self.streaming_task_id: Optional[int] = None

    def start(self) -> "StreamingProgressDisplay":
        """Start the progress display and show metadata fetching task.

        Returns:
            Self for chaining
        """
        self.console.print(f"\n[bold]Streaming document:[/bold] {self.document_id}\n")

        # Create progress with custom columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.progress.start()

        # Add metadata task (indeterminate)
        self.metadata_task_id = self.progress.add_task(
            "[cyan]Fetching metadata...",
            total=None
        )

        return self

    def metadata_complete(self):
        """Mark metadata fetching as complete."""
        if self.progress and self.metadata_task_id is not None:
            # Set total and completed to stop spinner and show as finished
            self.progress.update(
                self.metadata_task_id,
                description="[green]✓[/green] Metadata fetched",
                total=1,
                completed=1
            )

    def init_streaming(self, total: int):
        """Initialize the streaming progress bar.

        Args:
            total: Total number of comments to stream
        """
        if self.progress:
            self.streaming_task_id = self.progress.add_task(
                "[cyan]Streaming comments",
                total=total
            )

    def update_streaming(
        self,
        completed: int,
        chunks_created: int = 0,
        cost_report: Optional[CostReport] = None
    ):
        """Update streaming progress.

        Args:
            completed: Number of comments streamed so far
            chunks_created: Number of chunks created
            cost_report: Current cost report for displaying running costs
        """
        if self.progress and self.streaming_task_id is not None:
            # Build description with chunks and cost
            description = "Streaming comments"

            if chunks_created > 0:
                description += f" [dim]({chunks_created} chunks)[/dim]"

            if cost_report and cost_report.total_cost_usd > 0:
                # Format cost compactly for progress line
                cost_str = f"${cost_report.total_cost_usd:.4f}"
                description += f" [dim yellow]• Cost: {cost_str}[/dim yellow]"

            self.progress.update(
                self.streaming_task_id,
                completed=completed,
                description=f"[cyan]{description}"
            )

    def complete(self, total_streamed: int, total_chunks: int, total_skipped: int = 0):
        """Mark streaming as complete.

        Args:
            total_streamed: Total number of comments streamed
            total_chunks: Total number of chunks created
            total_skipped: Total number of comments skipped (already existed)
        """
        if self.progress and self.streaming_task_id is not None:
            description = f"[green]✓[/green] Streamed {total_streamed} comments"

            if total_chunks > 0:
                description += f" [dim]({total_chunks} chunks)[/dim]"

            if total_skipped > 0:
                description += f" [dim]({total_skipped} skipped)[/dim]"

            self.progress.update(
                self.streaming_task_id,
                description=description
            )

    def stop(self):
        """Stop the progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None


def create_streaming_progress_callback(
    display: StreamingProgressDisplay,
) -> Callable:
    """Create a progress callback function for streaming operations.

    Args:
        display: StreamingProgressDisplay instance

    Returns:
        Callback function that can be passed to DocumentStreamer
    """
    def callback(event: str, **kwargs):
        """Progress callback for streaming operations.

        Args:
            event: Event type ('metadata_complete', 'init_streaming', 'update', 'complete')
            **kwargs: Event-specific parameters
        """
        if event == "metadata_complete":
            display.metadata_complete()
        elif event == "init_streaming":
            display.init_streaming(
                total=kwargs.get("total", 0)
            )
        elif event == "update":
            display.update_streaming(
                completed=kwargs.get("completed", 0),
                chunks_created=kwargs.get("chunks_created", 0),
                cost_report=kwargs.get("cost_report")
            )
        elif event == "complete":
            display.complete(
                total_streamed=kwargs.get("total_streamed", 0),
                total_chunks=kwargs.get("total_chunks", 0),
                total_skipped=kwargs.get("total_skipped", 0)
            )

    return callback
