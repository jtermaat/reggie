"""Pipeline orchestrator for coordinating comment processing stages"""

import logging
from typing import List, Optional
from datetime import datetime

import psycopg

from .stages import BatchCategorizationStage, BatchEmbeddingStage
from .categorizer import CommentCategorizer
from .embedder import CommentEmbedder
from ..db import get_connection, CommentRepository, CommentChunkRepository
from ..models import CommentData

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
    ) -> dict:
        """Process comments through the pipeline.

        Args:
            document_id: Document ID
            batch_size: Number of comments to process in parallel
            skip_processed: If True, only process comments that haven't been processed yet

        Returns:
            Statistics about the processing
        """
        logger.info(f"Processing comments for document {document_id}")

        stats = {
            "document_id": document_id,
            "comments_processed": 0,
            "chunks_created": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

        try:
            async with get_connection(self.connection_string) as conn:
                # Fetch comments from database
                rows = await CommentRepository.get_comments_for_document(
                    document_id, conn, skip_processed=skip_processed
                )

                if not rows:
                    if skip_processed:
                        logger.info(f"No unprocessed comments found for document {document_id}")
                    else:
                        logger.warning(f"No comments found for document {document_id}")
                    return stats

                logger.info(f"Found {len(rows)} comments to process")

                # Process in batches
                for i in range(0, len(rows), batch_size):
                    batch_rows = rows[i : i + batch_size]

                    # Convert to CommentData objects
                    comment_data_list = [
                        CommentData(
                            id=row[0],
                            comment_text=row[1] or "",
                            first_name=row[2],
                            last_name=row[3],
                            organization=row[4],
                        )
                        for row in batch_rows
                    ]

                    # Process batch through stages
                    try:
                        # Stage 1: Categorization
                        classifications = await self.categorization_stage.process_batch(
                            comment_data_list
                        )

                        # Stage 2: Embedding
                        embeddings = await self.embedding_stage.process_batch(
                            comment_data_list
                        )

                        # Store results in database
                        for j, comment_data in enumerate(comment_data_list):
                            try:
                                classification = classifications[j]
                                chunks_with_embeddings = embeddings[j]["chunks"]

                                # Update comment with classification
                                await CommentRepository.update_comment_classification(
                                    comment_data.id,
                                    classification["category"],
                                    classification["sentiment"],
                                    classification["topics"],
                                    conn,
                                )

                                # Store chunks and embeddings
                                await CommentChunkRepository.store_comment_chunks(
                                    comment_data.id,
                                    chunks_with_embeddings,
                                    conn,
                                )

                                stats["comments_processed"] += 1
                                stats["chunks_created"] += len(chunks_with_embeddings)

                            except Exception as e:
                                logger.error(
                                    f"Error processing comment {comment_data.id}: {e}"
                                )
                                stats["errors"] += 1

                        # Commit after each batch
                        await conn.commit()
                        logger.info(
                            f"Processed {min(i + batch_size, len(rows))}/{len(rows)} comments "
                            f"({stats['chunks_created']} chunks total)"
                        )

                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        stats["errors"] += len(batch_rows)
                        continue

        except Exception as e:
            logger.error(f"Error processing comments: {e}")
            stats["errors"] += 1
            raise

        finally:
            stats["end_time"] = datetime.now()
            stats["duration"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()

        logger.info(
            f"Completed processing {document_id}: "
            f"{stats['comments_processed']} comments, "
            f"{stats['chunks_created']} chunks, "
            f"{stats['errors']} errors, "
            f"{stats['duration']:.1f}s"
        )

        return stats
