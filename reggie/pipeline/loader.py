"""Main document loader that orchestrates the entire loading pipeline"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

import psycopg
from langsmith import traceable

from ..api import RegulationsAPIClient
from .categorizer import CommentCategorizer
from .embedder import CommentEmbedder
from ..db import get_connection_string
from ..config import setup_langsmith

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Orchestrates loading and processing of documents and comments."""

    def __init__(
        self,
        reg_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        connection_string: Optional[str] = None,
    ):
        """Initialize the document loader.

        Args:
            reg_api_key: Regulations.gov API key
            openai_api_key: OpenAI API key
            connection_string: PostgreSQL connection string
        """
        self.api_client = RegulationsAPIClient(api_key=reg_api_key)
        self.categorizer = CommentCategorizer(openai_api_key=openai_api_key)
        self.embedder = CommentEmbedder(openai_api_key=openai_api_key)
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
                    attrs,
                ),
            )

    async def _store_comment(
        self,
        comment_data: dict,
        document_id: str,
        category: str,
        sentiment: str,
        conn,
    ) -> None:
        """Store comment in database.

        Args:
            comment_data: Comment data from API
            document_id: Parent document ID
            category: Classified category
            sentiment: Classified sentiment
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
                    category = EXCLUDED.category,
                    sentiment = EXCLUDED.sentiment,
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
                    attrs,
                ),
            )

    async def _store_comment_chunks(
        self,
        comment_id: str,
        chunks_with_embeddings: list,
        conn,
    ) -> None:
        """Store comment chunks and embeddings in database.

        Args:
            comment_id: Comment ID
            chunks_with_embeddings: List of (chunk_text, embedding) tuples
            conn: Database connection
        """
        if not chunks_with_embeddings:
            return

        async with conn.cursor() as cur:
            # Delete existing chunks for this comment
            await cur.execute(
                "DELETE FROM comment_chunks WHERE comment_id = %s",
                (comment_id,)
            )

            # Insert new chunks
            for idx, (chunk_text, embedding) in enumerate(chunks_with_embeddings):
                await cur.execute(
                    """
                    INSERT INTO comment_chunks (comment_id, chunk_text, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (comment_id, chunk_text, idx, embedding),
                )

    @traceable(name="load_document")
    async def load_document(
        self,
        document_id: str,
        batch_size: int = 10,
    ) -> dict:
        """Load a document and all its comments into the database.

        This orchestrates the entire pipeline:
        1. Fetch document metadata
        2. Fetch all comments
        3. Categorize comments (sentiment + category)
        4. Chunk and embed comments
        5. Store everything in PostgreSQL

        Args:
            document_id: Document ID (e.g., "CMS-2025-0304-0009")
            batch_size: Number of comments to process in parallel

        Returns:
            Statistics about the loading process
        """
        logger.info(f"Loading document {document_id}")

        stats = {
            "document_id": document_id,
            "comments_processed": 0,
            "chunks_created": 0,
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

                # 2. Fetch all comment details
                logger.info("Fetching comments...")
                comments = []
                async for comment_detail in self.api_client.get_all_comment_details(
                    object_id, batch_size=batch_size
                ):
                    comments.append(comment_detail)

                logger.info(f"Fetched {len(comments)} comments")

                if not comments:
                    logger.warning(f"No comments found for document {document_id}")
                    await conn.commit()
                    return stats

                # 3. Prepare comment data for processing
                comment_data_list = []
                for comment in comments:
                    attrs = comment.get("attributes", {})
                    comment_data_list.append({
                        "id": comment.get("id"),
                        "comment_text": attrs.get("comment", ""),
                        "first_name": attrs.get("firstName"),
                        "last_name": attrs.get("lastName"),
                        "organization": attrs.get("organization"),
                    })

                # 4. Categorize comments in batches
                logger.info("Categorizing comments...")
                classifications = await self.categorizer.categorize_batch(
                    comment_data_list, batch_size=batch_size
                )

                # 5. Chunk and embed comments in batches
                logger.info("Chunking and embedding comments...")
                all_chunks = await self.embedder.process_comments_batch(
                    comment_data_list, batch_size=batch_size
                )

                # 6. Store everything in database
                logger.info("Storing data in database...")
                for i, comment in enumerate(comments):
                    try:
                        classification = classifications[i]
                        chunks_with_embeddings = all_chunks[i]

                        # Store comment with classification
                        await self._store_comment(
                            comment,
                            document_id,
                            classification.category.value,
                            classification.sentiment.value,
                            conn,
                        )

                        # Store chunks and embeddings
                        await self._store_comment_chunks(
                            comment.get("id"),
                            chunks_with_embeddings,
                            conn,
                        )

                        stats["comments_processed"] += 1
                        stats["chunks_created"] += len(chunks_with_embeddings)

                        # Periodic progress logging
                        if (i + 1) % 10 == 0:
                            logger.info(
                                f"Stored {i + 1}/{len(comments)} comments "
                                f"({stats['chunks_created']} chunks)"
                            )

                    except Exception as e:
                        logger.error(f"Error storing comment {i}: {e}")
                        stats["errors"] += 1

                # Commit transaction
                await conn.commit()
                logger.info("Database transaction committed")

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
            f"{stats['chunks_created']} chunks, "
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
