"""Repository for document-related database operations."""

import json
import logging
import sqlite3
from typing import List

from ..exceptions import RepositoryError

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document-related database operations."""

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize repository with database connection.

        Args:
            connection: SQLite database connection
        """
        self._conn = connection

    def store_document(self, document_data: dict) -> None:
        """Store document metadata in database.

        Args:
            document_data: Document data from API

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            attrs = document_data.get("attributes", {})

            self._conn.execute(
                """
                INSERT INTO documents (id, title, object_id, docket_id, document_type, posted_date, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    title = excluded.title,
                    docket_id = excluded.docket_id,
                    document_type = excluded.document_type,
                    posted_date = excluded.posted_date,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    document_data.get("id"),
                    attrs.get("title"),
                    attrs.get("objectId"),
                    attrs.get("docketId"),
                    attrs.get("documentType"),
                    attrs.get("postedDate"),
                    json.dumps(attrs),
                ),
            )
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to store document: {e}") from e

    def list_documents(self) -> List[dict]:
        """List all documents with comment counts.

        Returns:
            List of document summaries with statistics

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            cur = self._conn.execute(
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

            rows = cur.fetchall()

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
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to list documents: {e}") from e
