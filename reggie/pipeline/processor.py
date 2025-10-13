"""Comment processor for categorization and embedding

This module provides backward compatibility while delegating to the orchestrator.
"""

import logging
from typing import Optional, Callable

from .orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class CommentProcessor:
    """Processes comments: categorizes and embeds them.

    This class now delegates to PipelineOrchestrator for actual processing.
    It's maintained for backward compatibility with existing code.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        connection_string: Optional[str] = None,
    ):
        """Initialize the comment processor.

        Args:
            openai_api_key: OpenAI API key
            connection_string: PostgreSQL connection string
        """
        self.orchestrator = PipelineOrchestrator.create(
            openai_api_key=openai_api_key,
            connection_string=connection_string,
        )

    async def process_comments(
        self,
        document_id: str,
        batch_size: int = 10,
        skip_processed: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Process raw comments: categorize, chunk, and embed.

        This processes comments that have already been loaded into the database.

        Args:
            document_id: Document ID
            batch_size: Number of comments to process in parallel
            skip_processed: If True, only process comments that haven't been processed yet
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the processing
        """
        return await self.orchestrator.process_comments(
            document_id=document_id,
            batch_size=batch_size,
            skip_processed=skip_processed,
            progress_callback=progress_callback,
        )
