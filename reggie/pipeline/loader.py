"""Main document loader that orchestrates the entire loading pipeline"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

import psycopg
from psycopg.types.json import Json
from langsmith import traceable

from ..api import RegulationsAPIClient
from ..db import get_connection_string
from ..config import setup_langsmith

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

    async def _store_document(self, document_data: dict, conn) -> None:
        """Store document metadata in database.

        Args:
            document_data: Document data from API
            conn: Database connection
        """
        attrs = document_data.get("attributes", {})

        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO documents (id, title, object_id, docket_id, document_type, posted_date, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    docket_id = EXCLUDED.docket_id,
                    document_type = EXCLUDED.document_type,
                    posted_date = EXCLUDED.posted_date,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                (
                    document_data.get("id"),
                    attrs.get("title"),
                    attrs.get("objectId"),
                    attrs.get("docketId"),
                    attrs.get("documentType"),
                    attrs.get("postedDate"),
                    Json(attrs),
                ),
            )

    async def _store_comment(
        self,
        comment_data: dict,
        document_id: str,
        category: str = None,
        sentiment: str = None,
        conn = None,
    ) -> None:
        """Store comment in database.

        Args:
            comment_data: Comment data from API
            document_id: Parent document ID
            category: Classified category (optional)
            sentiment: Classified sentiment (optional)
            conn: Database connection
        """
        attrs = comment_data.get("attributes", {})

        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO comments (
                    id, document_id, comment_text, category, sentiment,
                    first_name, last_name, organization, posted_date, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    comment_text = EXCLUDED.comment_text,
                    category = COALESCE(EXCLUDED.category, comments.category),
                    sentiment = COALESCE(EXCLUDED.sentiment, comments.sentiment),
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                (
                    comment_data.get("id"),
                    document_id,
                    attrs.get("comment", ""),
                    category,
                    sentiment,
                    attrs.get("firstName"),
                    attrs.get("lastName"),
                    attrs.get("organization"),
                    attrs.get("postedDate"),
                    Json(attrs),
                ),
            )

    @traceable(name="load_document")
    async def load_document(
        self,
        document_id: str,
        commit_every: int = 10,
    ) -> dict:
        """Load a document and all its comments into the database.

        This fetches document metadata and raw comments from the API
        and stores them in the database. Comments are NOT processed
        (categorized/embedded) during this stage - use process_comments()
        for that.

        Args:
            document_id: Document ID (e.g., "CMS-2025-0304-0009")
            commit_every: Commit to database after this many comments (default: 10)

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
                await self._store_document(document_data, conn)
                logger.info(f"Stored document metadata for {document_id}")

                # 2. Fetch and store comments one at a time
                logger.info("Starting sequential comment loading (4 seconds per comment)...")

                total_comments_fetched = 0

                async for comment_detail in self.api_client.get_all_comment_details(object_id):
                    try:
                        # Store comment immediately
                        await self._store_comment(
                            comment_detail,
                            document_id,
                            category=None,
                            sentiment=None,
                            conn=conn,
                        )
                        stats["comments_processed"] += 1
                        total_comments_fetched += 1

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

                if total_comments_fetched == 0:
                    logger.warning(f"No comments found for document {document_id}")

                logger.info("All comments loaded successfully")

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
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        d.id,
                        d.title,
                        d.docket_id,
                        d.posted_date,
                        COUNT(c.id) as comment_count,
                        COUNT(DISTINCT c.category) as unique_categories,
                        d.created_at
                    FROM documents d
                    LEFT JOIN comments c ON d.id = c.document_id
                    GROUP BY d.id, d.title, d.docket_id, d.posted_date, d.created_at
                    ORDER BY d.created_at DESC
                    """
                )

                rows = await cur.fetchall()

                documents = []
                for row in rows:
                    documents.append({
                        "id": row[0],
                        "title": row[1],
                        "docket_id": row[2],
                        "posted_date": row[3],
                        "comment_count": row[4],
                        "unique_categories": row[5],
                        "loaded_at": row[6],
                    })

                return documents

        finally:
            await conn.close()
