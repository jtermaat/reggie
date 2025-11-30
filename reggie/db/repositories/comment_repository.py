"""Repository for comment CRUD operations."""

import logging
import psycopg
from psycopg.types.json import Json
from typing import Optional, List
from datetime import datetime

from ...models.comment import CommentData
from ..exceptions import RepositoryError

logger = logging.getLogger(__name__)


class CommentRepository:
    """Repository for comment-related database operations (CRUD only)."""

    def __init__(self, connection: psycopg.AsyncConnection):
        """
        Initialize repository with database connection.

        Args:
            connection: PostgreSQL async database connection
        """
        self._conn = connection

    async def comment_exists(self, comment_id: str) -> bool:
        """Check if a comment already exists in the database.

        Args:
            comment_id: Comment ID to check

        Returns:
            True if comment exists, False otherwise

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            async with self._conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM comments WHERE id = %s LIMIT 1",
                    (comment_id,)
                )
                row = await cur.fetchone()
                return row is not None
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to check comment existence: {e}") from e

    async def store_comment(
        self,
        comment_data: dict,
        document_id: str,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        topics: Optional[List[str]] = None,
        doctor_specialization: Optional[str] = None,
        licensed_professional_type: Optional[str] = None,
        keywords_phrases: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
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
            keywords_phrases: Extracted keywords and phrases (optional)
            entities: Extracted named entities (optional)

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            attrs = comment_data.get("attributes", {})

            # Parse posted_date string to datetime if present
            posted_date = attrs.get("postedDate")
            if posted_date and isinstance(posted_date, str):
                try:
                    posted_date = datetime.fromisoformat(posted_date.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    posted_date = None

            # Build keywords_entities JSONB structure
            keywords_entities = {
                "keywords_phrases": keywords_phrases or [],
                "entities": entities or [],
            }

            async with self._conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO comments (
                        id, document_id, comment_text, category, sentiment, topics,
                        doctor_specialization, licensed_professional_type, keywords_entities,
                        first_name, last_name, organization, posted_date, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        comment_text = EXCLUDED.comment_text,
                        category = COALESCE(EXCLUDED.category, comments.category),
                        sentiment = COALESCE(EXCLUDED.sentiment, comments.sentiment),
                        topics = COALESCE(EXCLUDED.topics, comments.topics),
                        doctor_specialization = COALESCE(EXCLUDED.doctor_specialization, comments.doctor_specialization),
                        licensed_professional_type = COALESCE(EXCLUDED.licensed_professional_type, comments.licensed_professional_type),
                        keywords_entities = COALESCE(EXCLUDED.keywords_entities, comments.keywords_entities),
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        comment_data.get("id"),
                        document_id,
                        attrs.get("comment", ""),
                        category,
                        sentiment,
                        Json(topics) if topics is not None else None,  # Wrap list for JSONB array conversion
                        doctor_specialization,
                        licensed_professional_type,
                        Json(keywords_entities),
                        attrs.get("firstName"),
                        attrs.get("lastName"),
                        attrs.get("organization"),
                        posted_date,
                        Json(attrs),  # Wrap dict for JSONB conversion
                    ),
                )
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to store comment: {e}") from e

    async def update_comment_classification(
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
            async with self._conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE comments
                    SET category = %s, sentiment = %s, topics = %s,
                        doctor_specialization = %s, licensed_professional_type = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (
                        category,
                        sentiment,
                        Json(topics) if topics is not None else None,
                        doctor_specialization,
                        licensed_professional_type,
                        comment_id,
                    )
                )
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to update comment classification: {e}") from e

    async def get_comments_for_document(
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
            async with self._conn.cursor() as cur:
                if skip_processed:
                    await cur.execute(
                        """
                        SELECT id, comment_text, first_name, last_name, organization
                        FROM comments
                        WHERE document_id = %s
                        AND (sentiment IS NULL OR category IS NULL)
                        ORDER BY created_at
                        """,
                        (document_id,)
                    )
                else:
                    await cur.execute(
                        """
                        SELECT id, comment_text, first_name, last_name, organization
                        FROM comments
                        WHERE document_id = %s
                        ORDER BY created_at
                        """,
                        (document_id,)
                    )
                rows = await cur.fetchall()
                return [CommentData.from_db_row(row) for row in rows]
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to get comments for document: {e}") from e

    async def get_full_text(self, comment_id: str) -> str:
        """Get the full text of a comment.

        Args:
            comment_id: Comment ID

        Returns:
            The full comment text, or empty string if not found

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            async with self._conn.cursor() as cur:
                await cur.execute(
                    "SELECT comment_text FROM comments WHERE id = %s",
                    (comment_id,)
                )
                row = await cur.fetchone()
                return row["comment_text"] if row else ""
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to get comment full text: {e}") from e
