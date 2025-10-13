"""Main document loader that orchestrates the entire loading pipeline"""

import asyncio
import logging
from typing import Optional, Callable
from datetime import datetime

import psycopg

from ..api import RegulationsAPIClient
from ..db import get_connection_string, DocumentRepository, CommentRepository

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Orchestrates loading and processing of documents and comments."""

    def __init__(
        self,
        reg_api_key: Optional[str] = None,
        connection_string: Optional[str] = None,
    ):
        """Initialize the document loader.

        Args:
            reg_api_key: Regulations.gov API key
            connection_string: PostgreSQL connection string
        """
        self.api_client = RegulationsAPIClient(api_key=reg_api_key)
        self.connection_string = connection_string or get_connection_string()


    async def load_document(
        self,
        document_id: str,
        commit_every: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Load a document and all its comments into the database.

        This fetches document metadata and raw comments from the API
        and stores them in the database. Comments are NOT processed
        (categorized/embedded) during this stage - use process_comments()
        for that.

        Args:
            document_id: Document ID (e.g., "CMS-2025-0304-0009")
            commit_every: Commit to database after this many comments (default: 10)
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the loading process
        """
        logger.info(f"Loading document {document_id}")

        stats = {
            "document_id": document_id,
            "comments_processed": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

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
                # Store document
                await DocumentRepository.store_document(document_data, conn)
                logger.info(f"Stored document metadata for {document_id}")

                # Notify progress callback that metadata is complete
                if progress_callback:
                    progress_callback("metadata_complete")

                # 2. Get total comment count and initialize progress
                total_comments_expected = await self.api_client.get_comment_count(object_id)
                logger.info(f"Found {total_comments_expected} total comments")

                # Initialize progress tracking
                total_comments_fetched = 0
                skipped_comments = 0

                if progress_callback:
                    progress_callback(
                        "init_comments",
                        total=total_comments_expected,
                        skipped=0
                    )

                # 3. Fetch and store comments one at a time
                logger.info("Starting sequential comment loading (4 seconds per comment)...")

                async for comment_detail in self.api_client.get_all_comment_details(object_id):
                    try:
                        comment_id = comment_detail.get("id")

                        # Skip if comment already exists
                        if await CommentRepository.comment_exists(comment_id, conn):
                            skipped_comments += 1
                            if skipped_comments % 100 == 0:
                                logger.info(f"Skipped {skipped_comments} existing comments")

                            # Update progress
                            if progress_callback:
                                progress_callback(
                                    "update",
                                    completed=total_comments_fetched + skipped_comments,
                                    skipped=skipped_comments
                                )
                            continue

                        # Store comment immediately
                        await CommentRepository.store_comment(
                            comment_detail,
                            document_id,
                            category=None,
                            sentiment=None,
                            topics=None,
                            conn=conn,
                        )
                        stats["comments_processed"] += 1
                        total_comments_fetched += 1

                        # Update progress
                        if progress_callback:
                            progress_callback(
                                "update",
                                completed=total_comments_fetched + skipped_comments,
                                skipped=skipped_comments
                            )

                        # Commit every N comments so data is visible
                        if total_comments_fetched % commit_every == 0:
                            await conn.commit()
                            logger.info(
                                f"Stored {total_comments_fetched} comments "
                                f"(committed to database)"
                            )

                    except Exception as e:
                        logger.error(f"Error storing comment: {e}")
                        stats["errors"] += 1

                # Final commit for any remaining comments
                if total_comments_fetched % commit_every != 0:
                    await conn.commit()
                    logger.info(
                        f"Stored {total_comments_fetched} comments total "
                        f"(all committed to database)"
                    )

                if skipped_comments > 0:
                    logger.info(f"Skipped {skipped_comments} comments that already existed")

                if total_comments_fetched == 0 and skipped_comments == 0:
                    logger.warning(f"No comments found for document {document_id}")

                logger.info("All comments loaded successfully")

                # Notify progress callback that loading is complete
                if progress_callback:
                    progress_callback(
                        "complete",
                        total_loaded=total_comments_fetched,
                        total_skipped=skipped_comments
                    )

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Error loading document: {e}")
            stats["errors"] += 1
            raise

        finally:
            await self.api_client.close()
            stats["end_time"] = datetime.now()
            stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()

        logger.info(
            f"Completed loading {document_id}: "
            f"{stats['comments_processed']} comments, "
            f"{stats['errors']} errors, "
            f"{stats['duration']:.1f}s"
        )

        return stats

    async def list_documents(self) -> list[dict]:
        """List all documents currently in the database.

        Returns:
            List of document summaries with comment counts
        """
        conn = await psycopg.AsyncConnection.connect(self.connection_string)

        try:
            return await DocumentRepository.list_documents(conn)
        finally:
            await conn.close()
