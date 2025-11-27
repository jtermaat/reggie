"""Repository for comment analytics and cross-tabulation operations."""

import logging
import psycopg
from typing import Dict

from ..exceptions import RepositoryError, InvalidFilterError

logger = logging.getLogger(__name__)


class CommentAnalyticsRepository:
    """Repository for comment analytics and cross-tabulation operations."""

    def __init__(self, connection: psycopg.AsyncConnection):
        """
        Initialize repository with database connection.

        Args:
            connection: PostgreSQL async database connection
        """
        self._conn = connection

    async def get_sentiment_by_category(
        self,
        document_id: str,
    ) -> Dict[str, Dict[str, int]]:
        """Get sentiment breakdown for each category.

        This method returns a cross-tabulation of categories and sentiments,
        showing how many comments of each sentiment exist for each category.

        Args:
            document_id: Document ID

        Returns:
            Nested dict: {category: {sentiment: count}}
            Only includes categories with at least 1 comment.
            Results are ordered by total comment count (descending).

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            # Query to get category, sentiment, and count
            # Group by both dimensions and order by total comments per category
            async with self._conn.cursor() as cur:
                await cur.execute(
                    """
                    WITH category_totals AS (
                        SELECT category, COUNT(*) as total
                        FROM comments
                        WHERE document_id = %s AND category IS NOT NULL
                        GROUP BY category
                    )
                    SELECT
                        c.category,
                        c.sentiment,
                        COUNT(*) as count
                    FROM comments c
                    INNER JOIN category_totals ct ON c.category = ct.category
                    WHERE c.document_id = %s AND c.category IS NOT NULL
                    GROUP BY c.category, c.sentiment, ct.total
                    ORDER BY ct.total DESC, c.category, c.sentiment
                    """,
                    (document_id, document_id)
                )

                rows = await cur.fetchall()

            # Build nested dictionary structure
            result: Dict[str, Dict[str, int]] = {}
            for row in rows:
                category, sentiment, count = row["category"], row["sentiment"], row["count"]
                if category not in result:
                    result[category] = {}
                result[category][sentiment or "unclear"] = count

            return result
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to get sentiment by category: {e}") from e

    async def get_sentiment_by_specialization(
        self,
        document_id: str,
        field_name: str,
        category_filter: str,
    ) -> Dict[str, Dict[str, int]]:
        """Get sentiment breakdown for specializations within a category.

        This method returns a cross-tabulation of specializations (either
        doctor_specialization or licensed_professional_type) and sentiments,
        filtered to a specific category.

        Args:
            document_id: Document ID
            field_name: Field to group by ('doctor_specialization' or 'licensed_professional_type')
            category_filter: Category to filter by (e.g., 'Physicians & Surgeons')

        Returns:
            Nested dict: {specialization: {sentiment: count}}
            Only includes specializations with at least 1 comment.
            Results are ordered by total comment count (descending).

        Raises:
            InvalidFilterError: If field_name is not valid
            RepositoryError: If database operation fails
        """
        # Validate field_name for SQL injection protection
        valid_fields = ["doctor_specialization", "licensed_professional_type"]
        if field_name not in valid_fields:
            raise InvalidFilterError(f"field_name must be one of {valid_fields}")

        try:
            # Query to get specialization, sentiment, and count within the filtered category
            # Group by both dimensions and order by total comments per specialization
            query = f"""
                WITH specialization_totals AS (
                    SELECT {field_name}, COUNT(*) as total
                    FROM comments
                    WHERE document_id = %s
                      AND category = %s
                      AND {field_name} IS NOT NULL
                    GROUP BY {field_name}
                )
                SELECT
                    c.{field_name},
                    c.sentiment,
                    COUNT(*) as count
                FROM comments c
                INNER JOIN specialization_totals st ON c.{field_name} = st.{field_name}
                WHERE c.document_id = %s
                  AND c.category = %s
                  AND c.{field_name} IS NOT NULL
                GROUP BY c.{field_name}, c.sentiment, st.total
                ORDER BY st.total DESC, c.{field_name}, c.sentiment
            """

            async with self._conn.cursor() as cur:
                await cur.execute(query, (document_id, category_filter, document_id, category_filter))
                rows = await cur.fetchall()

            # Build nested dictionary structure
            result: Dict[str, Dict[str, int]] = {}
            for row in rows:
                # Access by field name
                specialization = row[field_name]
                sentiment = row["sentiment"]
                count = row["count"]

                if specialization not in result:
                    result[specialization] = {}
                result[specialization][sentiment or "unclear"] = count

            return result
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to get sentiment by specialization: {e}") from e
