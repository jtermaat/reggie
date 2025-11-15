"""Document streamer for combined loading and processing"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict
from datetime import datetime

from ..api import RegulationsAPIClient
from ..db import get_connection, get_db_path, DocumentRepository, CommentRepository, CommentChunkRepository
from ..models import CommentData
from ..utils import CostTracker, ErrorCollector
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
        db_path: str,
        rate_limit_delay: float = 4.0,
        error_collector: Optional[ErrorCollector] = None,
    ):
        """Initialize the document streamer.

        Args:
            api_client: API client for fetching comments
            categorization_stage: Stage for categorizing comments
            embedding_stage: Stage for embedding comments
            db_path: SQLite database path
            rate_limit_delay: Minimum seconds between API calls (default: 4.0)
            error_collector: Optional error collector for aggregating errors
        """
        self.api_client = api_client
        self.categorization_stage = categorization_stage
        self.embedding_stage = embedding_stage
        self.db_path = db_path
        self.rate_limit_delay = rate_limit_delay
        self.error_collector = error_collector

        # Pass error collector to stages
        if error_collector:
            self.categorization_stage.set_error_collector(error_collector)

    @classmethod
    def create(
        cls,
        reg_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        db_path: Optional[str] = None,
        error_collector: Optional[ErrorCollector] = None,
    ) -> "DocumentStreamer":
        """Factory method to create a DocumentStreamer with default configuration.

        Args:
            reg_api_key: Regulations.gov API key
            openai_api_key: OpenAI API key
            db_path: SQLite database path
            error_collector: Optional error collector for aggregating errors

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

        # Get database path
        db_path_to_use = db_path or get_db_path()

        return cls(
            api_client=api_client,
            categorization_stage=categorization_stage,
            embedding_stage=embedding_stage,
            db_path=db_path_to_use,
            rate_limit_delay=config.reg_api_request_delay,
            error_collector=error_collector,
        )

    async def _produce_comments(
        self,
        object_id: str,
        document_id: str,
        conn,
        queue: asyncio.Queue,
        stats: dict,
        last_api_call_time: list,  # Using list to allow mutation in nested scope
    ) -> None:
        """Producer: Load comments one-by-one and push to queue.

        Args:
            object_id: Document object ID for API
            document_id: Document ID for database
            conn: Database connection
            queue: Queue to push comments into
            stats: Statistics dictionary to update
            last_api_call_time: List containing last API call timestamp (mutable)
        """
        try:
            logger.info("Producer: Starting to load comments")
            async for comment in self.api_client.get_all_comments(object_id):
                try:
                    comment_id = comment.get("id")

                    # Check if comment already exists (sync operation)
                    if CommentRepository.comment_exists(comment_id, conn):
                        stats["skipped"] += 1
                        if stats["skipped"] % 100 == 0:
                            logger.info(f"Producer: Skipped {stats['skipped']} existing comments")
                        continue

                    # RATE LIMITING: Ensure we respect the delay between API calls
                    if last_api_call_time[0] is not None:
                        elapsed = time.time() - last_api_call_time[0]
                        if elapsed < self.rate_limit_delay:
                            wait_time = self.rate_limit_delay - elapsed
                            logger.debug(f"Producer: Rate limiting, waiting {wait_time:.2f}s")
                            await asyncio.sleep(wait_time)

                    # Download comment details from API
                    comment_detail = await self.api_client.get_comment_details(comment_id)
                    last_api_call_time[0] = time.time()

                    # Create CommentData object for processing
                    attrs = comment_detail.get("attributes", {})
                    comment_data = CommentData(
                        id=comment_id,
                        comment_text=attrs.get("comment", ""),
                        first_name=attrs.get("firstName"),
                        last_name=attrs.get("lastName"),
                        organization=attrs.get("organization"),
                    )

                    # Push to queue (will block if queue is full - backpressure)
                    await queue.put((comment_data, comment_detail))
                    logger.debug(f"Producer: Queued comment {comment_id}")

                except Exception as e:
                    # Log at DEBUG level for file logs
                    logger.debug(f"Producer: Error loading comment {comment_id}: {e}")

                    # Collect error for summary (if collector available)
                    if self.error_collector:
                        self.error_collector.collect(
                            error_type="Streaming Load Error",
                            message=str(e),
                            context={"comment_id": comment_id}
                        )

                    stats["errors"] += 1

            logger.info("Producer: Finished loading all comments")

        except Exception as e:
            logger.error(f"Producer: Fatal error: {e}")
            stats["errors"] += 1
            raise
        finally:
            # Signal consumer that production is complete
            await queue.put(None)
            logger.info("Producer: Sent completion signal")

    async def _consume_and_process_batches(
        self,
        document_id: str,
        queue: asyncio.Queue,
        conn,
        stats: dict,
        cost_tracker: CostTracker,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Consumer: Accumulate comments into batches and process them.

        Args:
            document_id: Document ID for database
            queue: Queue to pull comments from
            conn: Database connection
            stats: Statistics dictionary to update
            cost_tracker: Cost tracker for API usage
            progress_callback: Optional callback for progress updates
        """
        config = get_config()
        batch_size = 10
        batch_timeout = 2.0  # seconds to wait for batch to fill

        logger.info("Consumer: Starting to process batches")

        current_batch = []
        current_batch_details = []

        while True:
            try:
                # Try to accumulate a batch of comments
                while len(current_batch) < batch_size:
                    try:
                        # Wait for next item with timeout
                        item = await asyncio.wait_for(
                            queue.get(),
                            timeout=batch_timeout if current_batch else None
                        )

                        # Check for completion signal
                        if item is None:
                            logger.info("Consumer: Received completion signal")
                            # Process any remaining comments in batch
                            if current_batch:
                                await self._process_batch(
                                    current_batch,
                                    current_batch_details,
                                    document_id,
                                    conn,
                                    stats,
                                    cost_tracker,
                                    progress_callback,
                                )
                            return

                        # Add to batch
                        comment_data, comment_detail = item
                        current_batch.append(comment_data)
                        current_batch_details.append(comment_detail)

                    except asyncio.TimeoutError:
                        # Timeout waiting for more comments - process partial batch
                        if current_batch:
                            logger.debug(f"Consumer: Timeout reached with {len(current_batch)} comments")
                            break
                        # If batch is empty, continue waiting
                        continue

                # Process the batch
                if current_batch:
                    await self._process_batch(
                        current_batch,
                        current_batch_details,
                        document_id,
                        conn,
                        stats,
                        cost_tracker,
                        progress_callback,
                    )

                    # Clear batch for next iteration
                    current_batch = []
                    current_batch_details = []

            except Exception as e:
                # Log at DEBUG level for file logs
                logger.debug(f"Consumer: Error processing batch: {e}")

                # Collect error for summary (if collector available)
                if self.error_collector:
                    self.error_collector.collect(
                        error_type="Streaming Batch Processing Error",
                        message=str(e)
                    )

                stats["errors"] += len(current_batch)
                # Clear batch and continue
                current_batch = []
                current_batch_details = []

    async def _process_batch(
        self,
        comment_data_list: list,
        comment_details_list: list,
        document_id: str,
        conn,
        stats: dict,
        cost_tracker: CostTracker,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Process a batch of comments.

        Args:
            comment_data_list: List of CommentData objects
            comment_details_list: List of comment detail dicts from API
            document_id: Document ID
            conn: Database connection
            stats: Statistics dictionary to update
            cost_tracker: Cost tracker for API usage
            progress_callback: Optional callback for progress updates
        """
        config = get_config()
        batch_len = len(comment_data_list)

        logger.info(f"Consumer: Processing batch of {batch_len} comments")

        try:
            # PROCESS: Categorize the batch (callback tracks this automatically)
            async with cost_tracker.track_operation_async("categorization", config.categorization_model):
                classifications = await self.categorization_stage.process_batch(comment_data_list)

            # PROCESS: Embed the batch (callback doesn't track embeddings, so we track manually)
            embeddings = await self.embedding_stage.process_batch(comment_data_list)

            # Manually record embedding cost
            total_embedding_tokens = sum(emb.get("tokens", 0) for emb in embeddings)
            if total_embedding_tokens > 0:
                cost_tracker.record_embedding_tokens(total_embedding_tokens, config.embedding_model)

            # SAVE: Store all comments in batch (sync database operations)
            for i, comment_data in enumerate(comment_data_list):
                try:
                    classification = classifications[i]
                    embedding = embeddings[i]

                    # Store comment with classification
                    CommentRepository.store_comment(
                        comment_details_list[i],
                        document_id,
                        category=classification["category"],
                        sentiment=classification["sentiment"],
                        topics=classification["topics"],
                        doctor_specialization=classification.get("doctor_specialization"),
                        licensed_professional_type=classification.get("licensed_professional_type"),
                        conn=conn,
                    )

                    # Store comment chunks with embeddings
                    CommentChunkRepository.store_comment_chunks(
                        comment_data.id,
                        embedding["chunks"],
                        conn,
                    )

                    # Update statistics
                    stats["comments_processed"] += 1
                    stats["chunks_created"] += embedding["num_chunks"]

                except Exception as e:
                    # Log at DEBUG level for file logs
                    logger.debug(f"Consumer: Error storing comment {comment_data.id}: {e}")

                    # Collect error for summary (if collector available)
                    if self.error_collector:
                        self.error_collector.collect(
                            error_type="Streaming Storage Error",
                            message=str(e),
                            context={"comment_id": comment_data.id}
                        )

                    stats["errors"] += 1

            # Commit batch (sync operation)
            conn.commit()

            logger.info(
                f"Consumer: Completed batch - "
                f"{stats['comments_processed']} total processed, "
                f"{stats['chunks_created']} chunks, "
                f"${cost_tracker.get_report().total_cost_usd:.4f} so far"
            )

            # Update progress with running cost report
            if progress_callback:
                progress_callback(
                    "update",
                    completed=stats["comments_processed"] + stats["skipped"],
                    chunks_created=stats["chunks_created"],
                    cost_report=cost_tracker.get_report()
                )

        except Exception as e:
            logger.error(f"Consumer: Error in batch processing: {e}")
            stats["errors"] += batch_len
            raise

    async def stream_document(
        self,
        document_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Stream a document: download and process comments in parallel batches.

        This method uses a producer-consumer pattern to achieve optimal throughput:
        - Producer: Downloads comments one-by-one (respecting rate limits)
        - Consumer: Processes comments in batches of 10 (efficient OpenAI API calls)
        - Both run concurrently for maximum efficiency

        Args:
            document_id: Document ID (e.g., "CMS-2025-0304-0009")
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the streaming process (includes cost_report)
        """
        logger.info(f"Streaming document {document_id} using producer-consumer pattern")

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

            # Connect to database (sync connection)
            with get_connection(self.db_path) as conn:
                # Store document metadata
                DocumentRepository.store_document(document_data, conn)
                conn.commit()
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

                # 3. Create queue for producer-consumer communication
                # Queue size limits buffering to prevent memory issues
                queue = asyncio.Queue(maxsize=50)

                # Track last API call time (mutable list to share between tasks)
                last_api_call_time = [None]

                # 4. Run producer and consumer concurrently
                logger.info("Starting producer and consumer tasks")
                producer_task = asyncio.create_task(
                    self._produce_comments(
                        object_id,
                        document_id,
                        conn,
                        queue,
                        stats,
                        last_api_call_time,
                    )
                )

                consumer_task = asyncio.create_task(
                    self._consume_and_process_batches(
                        document_id,
                        queue,
                        conn,
                        stats,
                        cost_tracker,
                        progress_callback,
                    )
                )

                # Wait for both tasks to complete
                await asyncio.gather(producer_task, consumer_task)

                logger.info("Producer and consumer tasks completed")

                if stats["skipped"] > 0:
                    logger.info(f"Skipped {stats['skipped']} comments that already existed")

                logger.info("All comments streamed successfully")

                # Notify progress callback that streaming is complete
                if progress_callback:
                    progress_callback(
                        "complete",
                        total_streamed=stats["comments_processed"],
                        total_chunks=stats["chunks_created"],
                        total_skipped=stats["skipped"]
                    )

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
