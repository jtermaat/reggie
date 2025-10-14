"""Document streamer for combined loading and processing"""

import asyncio
import logging
import time
from typing import Optional, Callable
from datetime import datetime

import psycopg

from ..api import RegulationsAPIClient
from ..db import get_connection_string, DocumentRepository, CommentRepository, CommentChunkRepository
from ..models import CommentData
from ..utils import CostTracker
from ..config import get_config
from .stages import BatchCategorizationStage, BatchEmbeddingStage
from .categorizer import CommentCategorizer
from .embedder import CommentEmbedder

logger = logging.getLogger(__name__)


class DocumentStreamer:
    """Orchestrates streaming document loading and processing.

    This class combines the loading and processing pipelines into a single
    streaming operation that downloads comments one-by-one, immediately
    processes them (categorization + embedding), and saves results to the
    database. It efficiently uses rate limit waiting time for processing
    and provides real-time cost tracking.
    """

    def __init__(
        self,
        api_client: RegulationsAPIClient,
        categorization_stage: BatchCategorizationStage,
        embedding_stage: BatchEmbeddingStage,
        connection_string: str,
        rate_limit_delay: float = 4.0,
    ):
        """Initialize the document streamer.

        Args:
            api_client: API client for fetching comments
            categorization_stage: Stage for categorizing comments
            embedding_stage: Stage for embedding comments
            connection_string: PostgreSQL connection string
            rate_limit_delay: Minimum seconds between API calls (default: 4.0)
        """
        self.api_client = api_client
        self.categorization_stage = categorization_stage
        self.embedding_stage = embedding_stage
        self.connection_string = connection_string
        self.rate_limit_delay = rate_limit_delay

    @classmethod
    def create(
        cls,
        reg_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        connection_string: Optional[str] = None,
    ) -> "DocumentStreamer":
        """Factory method to create a DocumentStreamer with default configuration.

        Args:
            reg_api_key: Regulations.gov API key
            openai_api_key: OpenAI API key
            connection_string: PostgreSQL connection string

        Returns:
            Configured DocumentStreamer instance
        """
        config = get_config()

        # Create API client WITHOUT automatic rate limiting
        # (we'll manage it manually in the streaming loop)
        api_client = RegulationsAPIClient(api_key=reg_api_key)
        api_client.request_delay = 0  # Disable automatic delay

        # Create processing stages
        categorizer = CommentCategorizer(openai_api_key=openai_api_key)
        embedder = CommentEmbedder(openai_api_key=openai_api_key)
        categorization_stage = BatchCategorizationStage(categorizer)
        embedding_stage = BatchEmbeddingStage(embedder)

        # Get connection string
        conn_str = connection_string or get_connection_string()

        return cls(
            api_client=api_client,
            categorization_stage=categorization_stage,
            embedding_stage=embedding_stage,
            connection_string=conn_str,
            rate_limit_delay=config.reg_api_request_delay,
        )

    async def stream_document(
        self,
        document_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Stream a document: download, process, and store comments one-by-one.

        This method downloads comments sequentially from the API, immediately
        processes each comment (categorization + embedding), and saves results
        to the database. It respects rate limiting while using waiting time
        productively for processing.

        Args:
            document_id: Document ID (e.g., "CMS-2025-0304-0009")
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the streaming process (includes cost_report)
        """
        logger.info(f"Streaming document {document_id}")

        stats = self._initialize_stats(document_id)
        cost_tracker = CostTracker()

        try:
            # 1. Fetch document metadata
            logger.info("Fetching document metadata...")
            document_data = await self.api_client.get_document(document_id)

            if not document_data:
                raise ValueError(f"Document {document_id} not found")

            object_id = document_data.get("attributes", {}).get("objectId")
            if not object_id:
                raise ValueError(f"No objectId found for document {document_id}")

            # Connect to database
            conn = await psycopg.AsyncConnection.connect(self.connection_string)

            try:
                # Store document metadata
                await DocumentRepository.store_document(document_data, conn)
                logger.info(f"Stored document metadata for {document_id}")

                # Notify progress callback that metadata is complete
                if progress_callback:
                    progress_callback("metadata_complete")

                # 2. Get total comment count
                total_comments_expected = await self.api_client.get_comment_count(object_id)
                logger.info(f"Found {total_comments_expected} total comments")

                # Initialize progress tracking
                if progress_callback:
                    progress_callback(
                        "init_streaming",
                        total=total_comments_expected
                    )

                # 3. Stream comments: download → process → save
                last_api_call_time = None
                total_comments_processed = 0
                total_chunks_created = 0
                total_skipped = 0

                config = get_config()

                async for comment in self.api_client.get_all_comments(object_id):
                    try:
                        comment_id = comment.get("id")

                        # Check if comment already exists
                        if await CommentRepository.comment_exists(comment_id, conn):
                            total_skipped += 1
                            if total_skipped % 100 == 0:
                                logger.info(f"Skipped {total_skipped} existing comments")
                            continue

                        # RATE LIMITING: Ensure we respect the delay between API calls
                        if last_api_call_time is not None:
                            elapsed = time.time() - last_api_call_time
                            if elapsed < self.rate_limit_delay:
                                wait_time = self.rate_limit_delay - elapsed
                                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before next API call")
                                await asyncio.sleep(wait_time)

                        # Download comment details from API
                        start_time = time.time()
                        comment_detail = await self.api_client.get_comment_details(comment_id)
                        last_api_call_time = time.time()

                        # Create CommentData object for processing
                        attrs = comment_detail.get("attributes", {})
                        comment_data = CommentData(
                            id=comment_id,
                            comment_text=attrs.get("comment", ""),
                            first_name=attrs.get("firstName"),
                            last_name=attrs.get("lastName"),
                            organization=attrs.get("organization"),
                        )

                        # PROCESS: Categorize the comment (callback tracks this automatically)
                        async with cost_tracker.track_operation_async("categorization", config.categorization_model):
                            _, classification_result = await self.categorization_stage.process(comment_data)

                        # PROCESS: Embed the comment (callback doesn't track embeddings, so we track manually)
                        _, embedding_result = await self.embedding_stage.process(comment_data)

                        # Manually record embedding cost
                        embedding_tokens = embedding_result.get("tokens", 0)
                        if embedding_tokens > 0:
                            cost_tracker.record_embedding_tokens(embedding_tokens, config.embedding_model)

                        # SAVE: Store comment with classification
                        await CommentRepository.store_comment(
                            comment_detail,
                            document_id,
                            category=classification_result["category"],
                            sentiment=classification_result["sentiment"],
                            topics=classification_result["topics"],
                            doctor_specialization=classification_result.get("doctor_specialization"),
                            licensed_professional_type=classification_result.get("licensed_professional_type"),
                            conn=conn,
                        )

                        # SAVE: Store comment chunks with embeddings
                        await CommentChunkRepository.store_comment_chunks(
                            comment_id,
                            embedding_result["chunks"],
                            conn,
                        )

                        # Commit immediately so data is visible
                        await conn.commit()

                        # Update statistics
                        total_comments_processed += 1
                        total_chunks_created += embedding_result["num_chunks"]

                        # Update progress with running cost report
                        if progress_callback:
                            progress_callback(
                                "update",
                                completed=total_comments_processed + total_skipped,
                                chunks_created=total_chunks_created,
                                cost_report=cost_tracker.get_report()
                            )

                        if total_comments_processed % 10 == 0:
                            logger.info(
                                f"Streamed {total_comments_processed} comments "
                                f"({total_chunks_created} chunks, "
                                f"${cost_tracker.get_report().total_cost_usd:.4f} so far)"
                            )

                    except Exception as e:
                        logger.error(f"Error streaming comment {comment_id}: {e}")
                        stats["errors"] += 1

                # Update final statistics
                stats["comments_processed"] = total_comments_processed
                stats["chunks_created"] = total_chunks_created
                stats["skipped"] = total_skipped

                if total_skipped > 0:
                    logger.info(f"Skipped {total_skipped} comments that already existed")

                logger.info("All comments streamed successfully")

                # Notify progress callback that streaming is complete
                if progress_callback:
                    progress_callback(
                        "complete",
                        total_streamed=total_comments_processed,
                        total_chunks=total_chunks_created,
                        total_skipped=total_skipped
                    )

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Error streaming document: {e}")
            stats["errors"] += 1
            raise

        finally:
            await self.api_client.close()
            stats["cost_report"] = cost_tracker.get_report()
            self._finalize_stats(document_id, stats)

        return stats

    def _initialize_stats(self, document_id: str) -> dict:
        """Initialize streaming statistics.

        Args:
            document_id: Document ID being streamed

        Returns:
            Statistics dictionary
        """
        return {
            "document_id": document_id,
            "comments_processed": 0,
            "chunks_created": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

    def _finalize_stats(self, document_id: str, stats: dict) -> None:
        """Finalize and log statistics.

        Args:
            document_id: Document ID that was streamed
            stats: Statistics dictionary to finalize
        """
        stats["end_time"] = datetime.now()
        stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()

        logger.info(
            f"Completed streaming {document_id}: "
            f"{stats['comments_processed']} comments, "
            f"{stats['chunks_created']} chunks, "
            f"{stats['skipped']} skipped, "
            f"{stats['errors']} errors, "
            f"{stats['duration']:.1f}s, "
            f"${stats['cost_report'].total_cost_usd:.4f} total cost"
        )
