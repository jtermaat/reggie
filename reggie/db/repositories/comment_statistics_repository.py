"""Repository for comment statistics and aggregation operations."""

import logging
import sqlite3
from typing import Optional, List, Tuple

from ...models.agent import StatisticsResponse, StatisticsBreakdownItem
from ..exceptions import RepositoryError, InvalidFilterError
from ..utils.filter_builder import build_comment_filter_clause

logger = logging.getLogger(__name__)


class CommentStatisticsRepository:
    """Repository for comment statistics and aggregation operations."""

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize repository with database connection.

        Args:
            connection: SQLite database connection
        """
        self._conn = connection

    def count_comments_for_document(
        self,
        document_id: str,
        skip_processed: bool = False,
    ) -> int:
        """Count comments for a document.

        Args:
            document_id: Document ID
            skip_processed: If True, only count comments that haven't been processed yet

        Returns:
            Number of comments

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            if skip_processed:
                # Count only comments that haven't been categorized yet
                cur = self._conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM comments
                    WHERE document_id = ?
                      AND category IS NULL
                    """,
                    (document_id,)
                )
            else:
                # Count all comments
                cur = self._conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM comments
                    WHERE document_id = ?
                    """,
                    (document_id,)
                )
            row = cur.fetchone()
            return row[0] if row else 0
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to count comments: {e}") from e

    def get_statistics(
        self,
        document_id: str,
        group_by: str,
        sentiment_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        topics_filter: Optional[List[str]] = None,
        topic_filter_mode: str = "any",
        doctor_specialization_filter: Optional[str] = None,
        licensed_professional_type_filter: Optional[str] = None
    ) -> StatisticsResponse:
        """Get statistical breakdown of comments.

        Args:
            document_id: Document ID
            group_by: What to group by - 'sentiment', 'category', 'topic', 'doctor_specialization', or 'licensed_professional_type'
            sentiment_filter: Optional sentiment filter
            category_filter: Optional category filter
            topics_filter: Optional topics filter
            topic_filter_mode: 'any' or 'all' for topic filtering
            doctor_specialization_filter: Optional doctor specialization filter
            licensed_professional_type_filter: Optional licensed professional type filter

        Returns:
            StatisticsResponse with total_comments and breakdown

        Raises:
            InvalidFilterError: If group_by is not valid
            RepositoryError: If database operation fails
        """
        valid_group_by = ["sentiment", "category", "topic", "doctor_specialization", "licensed_professional_type"]
        if group_by not in valid_group_by:
            raise InvalidFilterError(f"group_by must be one of {valid_group_by}")

        try:
            where_clause, filter_params = build_comment_filter_clause(
                document_id, sentiment_filter, category_filter, topics_filter, topic_filter_mode,
                doctor_specialization_filter, licensed_professional_type_filter
            )

            # Get total count
            cur = self._conn.execute(
                f"SELECT COUNT(*) FROM comments c WHERE {where_clause}",
                filter_params
            )
            total = cur.fetchone()[0]

            # Get breakdown - no ORDER BY, will sort in Python
            if group_by == "topic":
                # Unnest topics JSON array for counting
                query = f"""
                    SELECT value as topic, COUNT(*) as count
                    FROM comments c, json_each(c.topics)
                    WHERE {where_clause}
                    GROUP BY value
                """
            else:
                # Simple grouping
                query = f"""
                    SELECT c.{group_by} as value, COUNT(*) as count
                    FROM comments c
                    WHERE {where_clause}
                    GROUP BY c.{group_by}
                """

            cur = self._conn.execute(query, filter_params)
            rows = cur.fetchall()

            breakdown = []
            for row in rows:
                value, count = row
                breakdown.append(StatisticsBreakdownItem(
                    value=value or "unknown",
                    count=count,
                    percentage=round((count / total * 100) if total > 0 else 0, 1)
                ))

            # Apply custom sorting based on group_by dimension
            breakdown = self._sort_breakdown_for_visualization(breakdown, group_by)

            return StatisticsResponse(
                total_comments=total,
                breakdown=breakdown
            )
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to get statistics: {e}") from e

    def _sort_breakdown_for_visualization(
        self,
        breakdown: List[StatisticsBreakdownItem],
        group_by: str
    ) -> List[StatisticsBreakdownItem]:
        """Sort breakdown items for visualization display.

        For sentiment: uses fixed order (for, against, mixed, unclear)
        For category/topic: sorts alphabetically by value

        Args:
            breakdown: List of StatisticsBreakdownItem to sort
            group_by: The dimension being grouped by

        Returns:
            Sorted list of StatisticsBreakdownItem
        """
        if group_by == "sentiment":
            # Fixed order for sentiment
            sentiment_order = {"for": 0, "against": 1, "mixed": 2, "unclear": 3}
            # Sort by the defined order, putting any unknown sentiments at the end
            return sorted(
                breakdown,
                key=lambda item: sentiment_order.get(item.value.lower(), 999)
            )
        else:
            # Alphabetical order for categories and topics
            return sorted(breakdown, key=lambda item: item.value.lower())
