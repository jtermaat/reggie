"""Comment processor for categorization and embedding"""

import logging
from typing import Optional
from datetime import datetime

import psycopg
from langsmith import traceable

from .categorizer import CommentCategorizer
from .embedder import CommentEmbedder
from ..db import get_connection_string, CommentRepository, CommentChunkRepository

logger = logging.getLogger(__name__)


class CommentProcessor:
    """Processes comments: categorizes and embeds them."""

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
        self.categorizer = CommentCategorizer(openai_api_key=openai_api_key)
        self.embedder = CommentEmbedder(openai_api_key=openai_api_key)
        self.connection_string = connection_string or get_connection_string()


    @traceable(name="process_comments")
    async def process_comments(
        self,
        document_id: str,
        batch_size: int = 10,
    ) -> dict:
        """Process raw comments: categorize, chunk, and embed.

        This processes comments that have already been loaded into the database.

        Args:
            document_id: Document ID
            batch_size: Number of comments to process in parallel

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
            conn = await psycopg.AsyncConnection.connect(self.connection_string)

            try:
                # Fetch unprocessed comments from database
                rows = await CommentRepository.get_comments_for_document(document_id, conn)

                if not rows:
                    logger.warning(f"No comments found for document {document_id}")
                    return stats

                logger.info(f"Found {len(rows)} comments to process")

                # Process in batches
                for i in range(0, len(rows), batch_size):
                    batch_rows = rows[i:i + batch_size]

                    # Prepare comment data
                    comment_data_list = []
                    for row in batch_rows:
                        comment_data_list.append({
                            "id": row[0],
                            "comment_text": row[1] or "",
                            "first_name": row[2],
                            "last_name": row[3],
                            "organization": row[4],
                        })

                    # Categorize the batch
                    classifications = await self.categorizer.categorize_batch(
                        comment_data_list, batch_size=len(comment_data_list)
                    )

                    # Chunk and embed the batch
                    all_chunks = await self.embedder.process_comments_batch(
                        comment_data_list, batch_size=len(comment_data_list)
                    )

                    # Update database with classifications and embeddings
                    for j, comment_data in enumerate(comment_data_list):
                        try:
                            classification = classifications[j]
                            chunks_with_embeddings = all_chunks[j]

                            # Update comment with classification
                            await CommentRepository.update_comment_classification(
                                comment_data["id"],
                                classification.category.value,
                                classification.sentiment.value,
                                [topic.value for topic in classification.topics],
                                conn,
                            )

                            # Store chunks and embeddings
                            await CommentChunkRepository.store_comment_chunks(
                                comment_data["id"],
                                chunks_with_embeddings,
                                conn,
                            )

                            stats["comments_processed"] += 1
                            stats["chunks_created"] += len(chunks_with_embeddings)

                        except Exception as e:
                            logger.error(f"Error processing comment {comment_data['id']}: {e}")
                            stats["errors"] += 1

                    # Commit after each batch
                    await conn.commit()
                    logger.info(
                        f"Processed {min(i + batch_size, len(rows))}/{len(rows)} comments "
                        f"({stats['chunks_created']} chunks total)"
                    )

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Error processing comments: {e}")
            stats["errors"] += 1
            raise

        finally:
            stats["end_time"] = datetime.now()
            stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()

        logger.info(
            f"Completed processing {document_id}: "
            f"{stats['comments_processed']} comments, "
            f"{stats['chunks_created']} chunks, "
            f"{stats['errors']} errors, "
            f"{stats['duration']:.1f}s"
        )

        return stats
