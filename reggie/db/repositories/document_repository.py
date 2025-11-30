"""Repository for document-related database operations."""

import logging
import psycopg
from psycopg.types.json import Json
from typing import List
from datetime import datetime

from ..exceptions import RepositoryError

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document-related database operations."""

    def __init__(self, connection: psycopg.AsyncConnection):
        """
        Initialize repository with database connection.

        Args:
            connection: PostgreSQL async database connection
        """
        self._conn = connection

    async def store_document(self, document_data: dict) -> None:
        """Store document metadata in database.

        Args:
            document_data: Document data from API

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            attrs = document_data.get("attributes", {})

            # Parse posted_date string to datetime if present
            posted_date = attrs.get("postedDate")
            if posted_date and isinstance(posted_date, str):
                try:
                    posted_date = datetime.fromisoformat(posted_date.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    posted_date = None

            async with self._conn.cursor() as cur:
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
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        document_data.get("id"),
                        attrs.get("title"),
                        attrs.get("objectId"),
                        attrs.get("docketId"),
                        attrs.get("documentType"),
                        posted_date,
                        Json(attrs),  # Wrap dict for JSONB conversion
                    ),
                )
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to store document: {e}") from e

    async def document_exists(self, document_id: str) -> bool:
        """Check if a document exists in the database.

        Args:
            document_id: The document ID to check

        Returns:
            True if document exists, False otherwise

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            async with self._conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM documents WHERE id = %s",
                    (document_id,)
                )
                return await cur.fetchone() is not None
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to check document existence: {e}") from e

    async def delete_document(self, document_id: str) -> dict:
        """Delete a document and all related data (comments, chunks).

        Due to ON DELETE CASCADE constraints, deleting the document
        automatically removes all associated comments and comment_chunks.

        Args:
            document_id: The document ID to delete

        Returns:
            Dictionary with counts of deleted entities

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            # First, count the related entities before deletion
            async with self._conn.cursor() as cur:
                # Count comments
                await cur.execute(
                    "SELECT COUNT(*) as count FROM comments WHERE document_id = %s",
                    (document_id,)
                )
                row = await cur.fetchone()
                comment_count = row["count"] if row else 0

                # Count chunks (through comments)
                await cur.execute(
                    """
                    SELECT COUNT(*) as count FROM comment_chunks cc
                    JOIN comments c ON cc.comment_id = c.id
                    WHERE c.document_id = %s
                    """,
                    (document_id,)
                )
                row = await cur.fetchone()
                chunk_count = row["count"] if row else 0

                # Delete the document (cascades to comments and chunks)
                await cur.execute(
                    "DELETE FROM documents WHERE id = %s",
                    (document_id,)
                )
                deleted_count = cur.rowcount

            return {
                "document_deleted": deleted_count > 0,
                "comments_deleted": comment_count,
                "chunks_deleted": chunk_count,
            }
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to delete document: {e}") from e

    async def list_documents(self) -> List[dict]:
        """List all documents with comment counts.

        Returns:
            List of document summaries with statistics

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            async with self._conn.cursor() as cur:
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
                    "id": row["id"],
                    "title": row["title"],
                    "docket_id": row["docket_id"],
                    "posted_date": row["posted_date"],
                    "comment_count": row["comment_count"],
                    "unique_categories": row["unique_categories"],
                    "loaded_at": row["created_at"],
                })

            return documents
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to list documents: {e}") from e

