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
