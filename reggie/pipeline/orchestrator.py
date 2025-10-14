"""Pipeline orchestrator for coordinating comment processing stages"""

import logging
from typing import List, Optional, Callable
from datetime import datetime

import psycopg

from .stages import BatchCategorizationStage, BatchEmbeddingStage
from .categorizer import CommentCategorizer
from .embedder import CommentEmbedder
from ..db import get_connection, CommentRepository, CommentChunkRepository
from ..models import CommentData
from ..utils import CostTracker
from ..config import get_config

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the comment processing pipeline."""

    def __init__(
        self,
        categorization_stage: BatchCategorizationStage,
        embedding_stage: BatchEmbeddingStage,
        connection_string: str,
    ):
        """Initialize the orchestrator.

        Args:
            categorization_stage: Stage for categorizing comments
            embedding_stage: Stage for embedding comments
            connection_string: Database connection string
        """
        self.categorization_stage = categorization_stage
        self.embedding_stage = embedding_stage
        self.connection_string = connection_string

    @classmethod
    def create(
        cls,
        openai_api_key: Optional[str] = None,
        connection_string: Optional[str] = None,
    ) -> "PipelineOrchestrator":
        """Factory method to create a PipelineOrchestrator with default stages.

        Args:
            openai_api_key: OpenAI API key
            connection_string: PostgreSQL connection string

        Returns:
            Configured PipelineOrchestrator instance
        """
        from ..db import get_connection_string

        categorizer = CommentCategorizer(openai_api_key=openai_api_key)
        embedder = CommentEmbedder(openai_api_key=openai_api_key)

        categorization_stage = BatchCategorizationStage(categorizer)
        embedding_stage = BatchEmbeddingStage(embedder)

        conn_str = connection_string or get_connection_string()

        return cls(categorization_stage, embedding_stage, conn_str)

    async def process_comments(
        self,
        document_id: str,
        batch_size: int = 10,
        skip_processed: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Process comments through the pipeline.

        Args:
            document_id: Document ID
            batch_size: Number of comments to process in parallel
            skip_processed: If True, only process comments that haven't been processed yet
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the processing (includes cost_report)
        """
        logger.info(f"Processing comments for document {document_id}")

        stats = self._initialize_stats(document_id)
        cost_tracker = CostTracker()

        try:
            async with get_connection(self.connection_string) as conn:
                rows = await self._fetch_comments(document_id, conn, skip_processed)

                if not rows:
                    self._log_no_comments(document_id, skip_processed)
                    # Add empty cost report even when no comments
                    stats["cost_report"] = cost_tracker.get_report()
                    return stats

                logger.info(f"Found {len(rows)} comments to process")
                await self._process_batches(
                    rows, batch_size, conn, stats, cost_tracker, progress_callback
                )

        except Exception as e:
            logger.error(f"Error processing comments: {e}")
            stats["errors"] += 1
            raise

        finally:
            # Add cost report to stats
            stats["cost_report"] = cost_tracker.get_report()
            self._finalize_stats(document_id, stats)

            # Notify progress callback that processing is complete
            if progress_callback:
                progress_callback(
                    "complete",
                    total_processed=stats["comments_processed"],
                    total_chunks=stats["chunks_created"]
                )

        return stats

    def _initialize_stats(self, document_id: str) -> dict:
        """Initialize processing statistics.

        Args:
            document_id: Document ID being processed

        Returns:
            Statistics dictionary
        """
        return {
            "document_id": document_id,
            "comments_processed": 0,
            "chunks_created": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

    async def _fetch_comments(
        self, document_id: str, conn, skip_processed: bool
    ) -> List[CommentData]:
        """Fetch comments for processing.

        Args:
            document_id: Document ID
            conn: Database connection
            skip_processed: If True, only fetch unprocessed comments

        Returns:
            List of CommentData objects
        """
        return await CommentRepository.get_comments_for_document(
            document_id, conn, skip_processed=skip_processed
        )

    def _log_no_comments(self, document_id: str, skip_processed: bool) -> None:
        """Log message when no comments are found.

        Args:
            document_id: Document ID
            skip_processed: Whether we were filtering for unprocessed comments
        """
        if skip_processed:
            logger.info(f"No unprocessed comments found for document {document_id}")
        else:
            logger.warning(f"No comments found for document {document_id}")

    async def _process_batches(
        self,
        comments: List[CommentData],
        batch_size: int,
        conn,
        stats: dict,
        cost_tracker: CostTracker,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Process comments in batches.

        Args:
            comments: CommentData objects to process
            batch_size: Number of comments per batch
            conn: Database connection
            stats: Statistics dictionary to update
            cost_tracker: CostTracker instance for tracking API costs
            progress_callback: Optional callback for progress updates
        """
        for i in range(0, len(comments), batch_size):
            batch = comments[i : i + batch_size]

            try:
                await self._process_single_batch(batch, conn, stats, cost_tracker)
                await conn.commit()
                self._log_batch_progress(i, batch_size, len(comments), stats)

                # Update progress callback
                if progress_callback:
                    progress_callback(
                        "update",
                        completed=stats["comments_processed"],
                        chunks_created=stats["chunks_created"]
                    )

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                stats["errors"] += len(batch)

    async def _process_single_batch(
        self, comment_data_list: List[CommentData], conn, stats: dict, cost_tracker: CostTracker
    ) -> None:
        """Process a single batch of comments through the pipeline.

        Args:
            comment_data_list: List of CommentData to process
            conn: Database connection
            stats: Statistics dictionary to update
            cost_tracker: CostTracker instance for tracking API costs
        """
        # Get config for model names
        config = get_config()

        # Track categorization costs (using callback - this works)
        async with cost_tracker.track_operation_async("categorization", config.categorization_model):
            classifications = await self.categorization_stage.process_batch(
                comment_data_list
            )

        # Process embeddings (callback doesn't work for embeddings, so we track manually)
        embeddings = await self.embedding_stage.process_batch(comment_data_list)

        # Manually record embedding costs by summing up tokens from all embeddings
        total_embedding_tokens = sum(emb.get("tokens", 0) for emb in embeddings)
        if total_embedding_tokens > 0:
            cost_tracker.record_embedding_tokens(total_embedding_tokens, config.embedding_model)

        for j, comment_data in enumerate(comment_data_list):
            try:
                await self._store_comment_results(
                    comment_data, classifications[j], embeddings[j], conn
                )
                stats["comments_processed"] += 1
                stats["chunks_created"] += len(embeddings[j]["chunks"])

            except Exception as e:
                logger.error(f"Error processing comment {comment_data.id}: {e}")
                stats["errors"] += 1

    async def _store_comment_results(
        self, comment_data: CommentData, classification: dict, embedding: dict, conn
    ) -> None:
        """Store processing results for a single comment.

        Args:
            comment_data: Comment being processed
            classification: Classification results
            embedding: Embedding results with chunks
            conn: Database connection
        """
        await CommentRepository.update_comment_classification(
            comment_data.id,
            classification["category"],
            classification["sentiment"],
            classification["topics"],
            classification.get("doctor_specialization"),
            classification.get("licensed_professional_type"),
            conn,
        )

        await CommentChunkRepository.store_comment_chunks(
            comment_data.id,
            embedding["chunks"],
            conn,
        )

    def _log_batch_progress(
        self, current_idx: int, batch_size: int, total: int, stats: dict
    ) -> None:
        """Log progress after processing a batch.

        Args:
            current_idx: Current index in the list
            batch_size: Size of batches
            total: Total number of comments
            stats: Statistics dictionary
        """
        logger.info(
            f"Processed {min(current_idx + batch_size, total)}/{total} comments "
            f"({stats['chunks_created']} chunks total)"
        )

    def _finalize_stats(self, document_id: str, stats: dict) -> None:
        """Finalize and log statistics.

        Args:
            document_id: Document ID that was processed
            stats: Statistics dictionary to finalize
        """
        stats["end_time"] = datetime.now()
        stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()

        logger.info(
            f"Completed processing {document_id}: "
            f"{stats['comments_processed']} comments, "
            f"{stats['chunks_created']} chunks, "
            f"{stats['errors']} errors, "
            f"{stats['duration']:.1f}s"
        )
