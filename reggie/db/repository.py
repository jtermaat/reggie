"""Database repository layer for centralized data access"""

import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import psycopg
from psycopg.types.json import Json

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document-related database operations."""

    @staticmethod
    async def store_document(document_data: dict, conn) -> None:
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

    @staticmethod
    async def list_documents(conn) -> List[dict]:
        """List all documents with comment counts.

        Args:
            conn: Database connection

        Returns:
            List of document summaries with statistics
        """
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


class CommentRepository:
    """Repository for comment-related database operations."""

    @staticmethod
    async def comment_exists(comment_id: str, conn) -> bool:
        """Check if a comment already exists in the database.

        Args:
            comment_id: Comment ID to check
            conn: Database connection

        Returns:
            True if comment exists, False otherwise
        """
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM comments WHERE id = %s LIMIT 1",
                (comment_id,)
            )
            return await cur.fetchone() is not None

    @staticmethod
    async def store_comment(
        comment_data: dict,
        document_id: str,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        topics: Optional[List[str]] = None,
        conn = None,
    ) -> None:
        """Store comment in database.

        Args:
            comment_data: Comment data from API
            document_id: Parent document ID
            category: Classified category (optional)
            sentiment: Classified sentiment (optional)
            topics: Classified topics (optional)
            conn: Database connection
        """
        attrs = comment_data.get("attributes", {})

        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO comments (
                    id, document_id, comment_text, category, sentiment, topics,
                    first_name, last_name, organization, posted_date, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    comment_text = EXCLUDED.comment_text,
                    category = COALESCE(EXCLUDED.category, comments.category),
                    sentiment = COALESCE(EXCLUDED.sentiment, comments.sentiment),
                    topics = COALESCE(EXCLUDED.topics, comments.topics),
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                (
                    comment_data.get("id"),
                    document_id,
                    attrs.get("comment", ""),
                    category,
                    sentiment,
                    topics,
                    attrs.get("firstName"),
                    attrs.get("lastName"),
                    attrs.get("organization"),
                    attrs.get("postedDate"),
                    Json(attrs),
                ),
            )

    @staticmethod
    async def update_comment_classification(
        comment_id: str,
        category: str,
        sentiment: str,
        topics: Optional[List[str]] = None,
        conn = None,
    ) -> None:
        """Update comment with classification results.

        Args:
            comment_id: Comment ID
            category: Classified category
            sentiment: Classified sentiment
            topics: Classified topics (optional)
            conn: Database connection
        """
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE comments
                SET category = %s, sentiment = %s, topics = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (category, sentiment, topics, comment_id)
            )

    @staticmethod
    async def get_comments_for_document(
        document_id: str,
        conn,
    ) -> List[Tuple]:
        """Fetch all comments for a document.

        Args:
            document_id: Document ID
            conn: Database connection

        Returns:
            List of comment rows (id, comment_text, first_name, last_name, organization)
        """
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, comment_text, first_name, last_name, organization
                FROM comments
                WHERE document_id = %s
                ORDER BY created_at
                """,
                (document_id,)
            )
            return await cur.fetchall()


class CommentChunkRepository:
    """Repository for comment chunk and embedding operations."""

    @staticmethod
    async def delete_chunks_for_comment(comment_id: str, conn) -> None:
        """Delete existing chunks for a comment.

        Args:
            comment_id: Comment ID
            conn: Database connection
        """
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM comment_chunks WHERE comment_id = %s",
                (comment_id,)
            )

    @staticmethod
    async def store_comment_chunks(
        comment_id: str,
        chunks_with_embeddings: List[Tuple[str, List[float]]],
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

        # Delete existing chunks
        await CommentChunkRepository.delete_chunks_for_comment(comment_id, conn)

        # Insert new chunks
        async with conn.cursor() as cur:
            for idx, (chunk_text, embedding) in enumerate(chunks_with_embeddings):
                await cur.execute(
                    """
                    INSERT INTO comment_chunks (comment_id, chunk_text, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (comment_id, chunk_text, idx, embedding),
                )
