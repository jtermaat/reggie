"""Repository for comment CRUD operations."""

import json
import logging
import sqlite3
from typing import Optional, List

from ...models.comment import CommentData
from ..exceptions import RepositoryError

logger = logging.getLogger(__name__)


class CommentRepository:
    """Repository for comment-related database operations (CRUD only)."""

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize repository with database connection.

        Args:
            connection: SQLite database connection
        """
        self._conn = connection

    def comment_exists(self, comment_id: str) -> bool:
        """Check if a comment already exists in the database.

        Args:
            comment_id: Comment ID to check

        Returns:
            True if comment exists, False otherwise

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            cur = self._conn.execute(
                "SELECT 1 FROM comments WHERE id = ? LIMIT 1",
                (comment_id,)
            )
            return cur.fetchone() is not None
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to check comment existence: {e}") from e

    def store_comment(
        self,
        comment_data: dict,
        document_id: str,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        topics: Optional[List[str]] = None,
        doctor_specialization: Optional[str] = None,
        licensed_professional_type: Optional[str] = None,
    ) -> None:
        """Store comment in database.

        Args:
            comment_data: Comment data from API
            document_id: Parent document ID
            category: Classified category (optional)
            sentiment: Classified sentiment (optional)
            topics: Classified topics (optional)
            doctor_specialization: Doctor specialization (optional)
            licensed_professional_type: Licensed professional type (optional)

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            attrs = comment_data.get("attributes", {})

            # Convert topics list to JSON string
            topics_json = json.dumps(topics) if topics else None

            self._conn.execute(
                """
                INSERT INTO comments (
                    id, document_id, comment_text, category, sentiment, topics,
                    doctor_specialization, licensed_professional_type,
                    first_name, last_name, organization, posted_date, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    comment_text = excluded.comment_text,
                    category = COALESCE(excluded.category, comments.category),
                    sentiment = COALESCE(excluded.sentiment, comments.sentiment),
                    topics = COALESCE(excluded.topics, comments.topics),
                    doctor_specialization = COALESCE(excluded.doctor_specialization, comments.doctor_specialization),
                    licensed_professional_type = COALESCE(excluded.licensed_professional_type, comments.licensed_professional_type),
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    comment_data.get("id"),
                    document_id,
                    attrs.get("comment", ""),
                    category,
                    sentiment,
                    topics_json,
                    doctor_specialization,
                    licensed_professional_type,
                    attrs.get("firstName"),
                    attrs.get("lastName"),
                    attrs.get("organization"),
                    attrs.get("postedDate"),
                    json.dumps(attrs),
                ),
            )
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to store comment: {e}") from e

    def update_comment_classification(
        self,
        comment_id: str,
        category: str,
        sentiment: str,
        topics: Optional[List[str]] = None,
        doctor_specialization: Optional[str] = None,
        licensed_professional_type: Optional[str] = None,
    ) -> None:
        """Update comment with classification results.

        Args:
            comment_id: Comment ID
            category: Classified category
            sentiment: Classified sentiment
            topics: Classified topics (optional)
            doctor_specialization: Doctor specialization (optional)
            licensed_professional_type: Licensed professional type (optional)

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            # Convert topics list to JSON string
            topics_json = json.dumps(topics) if topics else None

            self._conn.execute(
                """
                UPDATE comments
                SET category = ?, sentiment = ?, topics = ?,
                    doctor_specialization = ?, licensed_professional_type = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (category, sentiment, topics_json, doctor_specialization, licensed_professional_type, comment_id)
            )
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to update comment classification: {e}") from e

    def get_comments_for_document(
        self,
        document_id: str,
        skip_processed: bool = False,
    ) -> List[CommentData]:
        """Fetch all comments for a document.

        Args:
            document_id: Document ID
            skip_processed: If True, only fetch comments that haven't been processed yet
                           (i.e., comments without sentiment or category)

        Returns:
            List of CommentData objects

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            if skip_processed:
                cur = self._conn.execute(
                    """
                    SELECT id, comment_text, first_name, last_name, organization
                    FROM comments
                    WHERE document_id = ?
                    AND (sentiment IS NULL OR category IS NULL)
                    ORDER BY created_at
                    """,
                    (document_id,)
                )
            else:
                cur = self._conn.execute(
                    """
                    SELECT id, comment_text, first_name, last_name, organization
                    FROM comments
                    WHERE document_id = ?
                    ORDER BY created_at
                    """,
                    (document_id,)
                )
            rows = cur.fetchall()
            return [CommentData.from_db_row(row) for row in rows]
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get comments for document: {e}") from e

    def get_full_text(self, comment_id: str) -> str:
        """Get the full text of a comment.

        Args:
            comment_id: Comment ID

        Returns:
            The full comment text, or empty string if not found

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            cur = self._conn.execute(
                "SELECT comment_text FROM comments WHERE id = ?",
                (comment_id,)
            )
            row = cur.fetchone()
            return row[0] if row else ""
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get comment full text: {e}") from e
