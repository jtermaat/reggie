"""Progress display utilities for CLI operations using Rich.

This module provides progress display classes for long-running operations
like document loading and comment processing. It uses Rich's Progress API
to show detailed, real-time progress information.
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


class ProgressDisplay:
    """Base class for progress displays with common functionality."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress display.

        Args:
            console: Optional Rich console instance
        """
        self.console = console or Console()
        self.progress: Optional[Progress] = None
        self._original_log_level: Optional[int] = None

    def _suppress_logs(self):
        """Temporarily suppress INFO logs during progress display."""
        logger = logging.getLogger("reggie")
        self._original_log_level = logger.level
        # Only show WARNING and above during progress display
        logger.setLevel(logging.WARNING)

    def _restore_logs(self):
        """Restore original logging level."""
        if self._original_log_level is not None:
            logger = logging.getLogger("reggie")
            logger.setLevel(self._original_log_level)

    def __enter__(self):
        """Enter context manager."""
        self._suppress_logs()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._restore_logs()
        if self.progress is not None:
            self.progress.stop()


class LoadingProgressDisplay(ProgressDisplay):
    """Progress display for document loading operations.

    Shows progress for:
    1. Metadata fetching (indeterminate)
    2. Comment loading (determinate with count, rate, ETA)
    """

    def __init__(self, document_id: str, console: Optional[Console] = None):
        """Initialize the loading progress display.

        Args:
            document_id: Document ID being loaded
            console: Optional Rich console instance
        """
        super().__init__(console)
        self.document_id = document_id
        self.metadata_task_id: Optional[int] = None
        self.comments_task_id: Optional[int] = None

    def start(self) -> "LoadingProgressDisplay":
        """Start the progress display and show metadata fetching task.

        Returns:
            Self for chaining
        """
        self.console.print(f"\n[bold]Loading document:[/bold] {self.document_id}\n")

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
            self.progress.update(
                self.metadata_task_id,
                description="[green]✓[/green] Metadata fetched",
                completed=True
            )

    def init_comments(self, total: int, skipped: int = 0):
        """Initialize the comment loading progress bar.

        Args:
            total: Total number of comments to load
            skipped: Number of comments skipped (already exist)
        """
        if self.progress:
            description = "Loading comments"
            if skipped > 0:
                description += f" [dim]({skipped} skipped)[/dim]"

            self.comments_task_id = self.progress.add_task(
                f"[cyan]{description}",
                total=total
            )

    def update_comments(self, completed: int, skipped: int = 0):
        """Update comment loading progress.

        Args:
            completed: Number of comments loaded so far
            skipped: Number of comments skipped (already exist)
        """
        if self.progress and self.comments_task_id is not None:
            description = "Loading comments"
            if skipped > 0:
                description += f" [dim]({skipped} skipped)[/dim]"

            self.progress.update(
                self.comments_task_id,
                completed=completed,
                description=f"[cyan]{description}"
            )

    def complete(self, total_loaded: int, total_skipped: int):
        """Mark loading as complete.

        Args:
            total_loaded: Total number of comments loaded
            total_skipped: Total number of comments skipped
        """
        if self.progress and self.comments_task_id is not None:
            description = f"[green]✓[/green] Loaded {total_loaded} comments"
            if total_skipped > 0:
                description += f" [dim]({total_skipped} skipped)[/dim]"

            self.progress.update(
                self.comments_task_id,
                description=description
            )

    def stop(self):
        """Stop the progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None


class ProcessingProgressDisplay(ProgressDisplay):
    """Progress display for comment processing operations.

    Shows progress for:
    1. Batch processing of comments (categorization + embedding)
    """

    def __init__(self, document_id: str, console: Optional[Console] = None):
        """Initialize the processing progress display.

        Args:
            document_id: Document ID being processed
            console: Optional Rich console instance
        """
        super().__init__(console)
        self.document_id = document_id
        self.processing_task_id: Optional[int] = None

    def start(self, total: int, skip_processed: bool = False) -> "ProcessingProgressDisplay":
        """Start the progress display.

        Args:
            total: Total number of comments to process
            skip_processed: Whether skipping already processed comments

        Returns:
            Self for chaining
        """
        self.console.print(f"\n[bold]Processing comments for:[/bold] {self.document_id}")
        if skip_processed:
            self.console.print("[dim]Skipping already processed comments[/dim]")
        self.console.print()

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

        # Add processing task
        self.processing_task_id = self.progress.add_task(
            "[cyan]Processing comments",
            total=total
        )

        return self

    def update(self, completed: int, chunks_created: int = 0):
        """Update processing progress.

        Args:
            completed: Number of comments processed so far
            chunks_created: Total number of chunks created
        """
        if self.progress and self.processing_task_id is not None:
            description = "Processing comments"
            if chunks_created > 0:
                description += f" [dim]({chunks_created} chunks)[/dim]"

            self.progress.update(
                self.processing_task_id,
                completed=completed,
                description=f"[cyan]{description}"
            )

    def complete(self, total_processed: int, total_chunks: int):
        """Mark processing as complete.

        Args:
            total_processed: Total number of comments processed
            total_chunks: Total number of chunks created
        """
        if self.progress and self.processing_task_id is not None:
            description = f"[green]✓[/green] Processed {total_processed} comments"
            if total_chunks > 0:
                description += f" [dim]({total_chunks} chunks)[/dim]"

            self.progress.update(
                self.processing_task_id,
                description=description
            )

    def stop(self):
        """Stop the progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None


def create_loading_progress_callback(
    display: LoadingProgressDisplay,
) -> Callable:
    """Create a progress callback function for document loading.

    Args:
        display: LoadingProgressDisplay instance

    Returns:
        Callback function that can be passed to DocumentLoader
    """
    def callback(event: str, **kwargs):
        """Progress callback for loading operations.

        Args:
            event: Event type ('metadata_complete', 'init_comments', 'update', 'complete')
            **kwargs: Event-specific parameters
        """
        if event == "metadata_complete":
            display.metadata_complete()
        elif event == "init_comments":
            display.init_comments(
                total=kwargs.get("total", 0),
                skipped=kwargs.get("skipped", 0)
            )
        elif event == "update":
            display.update_comments(
                completed=kwargs.get("completed", 0),
                skipped=kwargs.get("skipped", 0)
            )
        elif event == "complete":
            display.complete(
                total_loaded=kwargs.get("total_loaded", 0),
                total_skipped=kwargs.get("total_skipped", 0)
            )

    return callback


def create_processing_progress_callback(
    display: ProcessingProgressDisplay,
) -> Callable:
    """Create a progress callback function for comment processing.

    Args:
        display: ProcessingProgressDisplay instance

    Returns:
        Callback function that can be passed to PipelineOrchestrator
    """
    def callback(event: str, **kwargs):
        """Progress callback for processing operations.

        Args:
            event: Event type ('init', 'update', 'complete')
            **kwargs: Event-specific parameters
        """
        if event == "init":
            # Init is handled by display.start() in CLI
            pass
        elif event == "update":
            display.update(
                completed=kwargs.get("completed", 0),
                chunks_created=kwargs.get("chunks_created", 0)
            )
        elif event == "complete":
            display.complete(
                total_processed=kwargs.get("total_processed", 0),
                total_chunks=kwargs.get("total_chunks", 0)
            )

    return callback
