"""Database repository layer for centralized data access"""

import json
import logging
import struct
from typing import Optional, List, Dict, Tuple

import sqlite_vec

from ..models.agent import (
    StatisticsResponse,
    StatisticsBreakdownItem,
    CommentChunkSearchResult
)
from ..models.comment import CommentData

logger = logging.getLogger(__name__)


def serialize_vector(vector: List[float]) -> bytes:
    """Serialize a list of floats to bytes for SQLite storage.

    Args:
        vector: List of float values

    Returns:
        Serialized bytes representation
    """
    return sqlite_vec.serialize_float32(vector)


def deserialize_vector(blob: bytes) -> List[float]:
    """Deserialize bytes to a list of floats.

    Args:
        blob: Bytes to deserialize

    Returns:
        List of float values
    """
    # Unpack as float32 (4 bytes per float)
    num_floats = len(blob) // 4
    return list(struct.unpack(f'{num_floats}f', blob))


class DocumentRepository:
    """Repository for document-related database operations."""

    @staticmethod
    def store_document(document_data: dict, conn) -> None:
        """Store document metadata in database.

        Args:
            document_data: Document data from API
            conn: Database connection
        """
        attrs = document_data.get("attributes", {})

        conn.execute(
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

    @staticmethod
    def list_documents(conn) -> List[dict]:
        """List all documents with comment counts.

        Args:
            conn: Database connection

        Returns:
            List of document summaries with statistics
        """
        cur = conn.execute(
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


class CommentRepository:
    """Repository for comment-related database operations."""

    @staticmethod
    def comment_exists(comment_id: str, conn) -> bool:
        """Check if a comment already exists in the database.

        Args:
            comment_id: Comment ID to check
            conn: Database connection

        Returns:
            True if comment exists, False otherwise
        """
        cur = conn.execute(
            "SELECT 1 FROM comments WHERE id = ? LIMIT 1",
            (comment_id,)
        )
        return cur.fetchone() is not None

    @staticmethod
    def count_comments_for_document(
        document_id: str,
        conn,
        skip_processed: bool = False,
    ) -> int:
        """Count comments for a document.

        Args:
            document_id: Document ID
            conn: Database connection
            skip_processed: If True, only count comments that haven't been processed yet

        Returns:
            Number of comments
        """
        if skip_processed:
            # Count only comments that haven't been categorized yet
            cur = conn.execute(
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
            cur = conn.execute(
                """
                SELECT COUNT(*)
                FROM comments
                WHERE document_id = ?
                """,
                (document_id,)
            )
        row = cur.fetchone()
        return row[0] if row else 0

    @staticmethod
    def store_comment(
        comment_data: dict,
        document_id: str,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        topics: Optional[List[str]] = None,
        doctor_specialization: Optional[str] = None,
        licensed_professional_type: Optional[str] = None,
        conn = None,
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
            conn: Database connection
        """
        attrs = comment_data.get("attributes", {})

        # Convert topics list to JSON string
        topics_json = json.dumps(topics) if topics else None

        conn.execute(
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

    @staticmethod
    def update_comment_classification(
        comment_id: str,
        category: str,
        sentiment: str,
        topics: Optional[List[str]] = None,
        doctor_specialization: Optional[str] = None,
        licensed_professional_type: Optional[str] = None,
        conn = None,
    ) -> None:
        """Update comment with classification results.

        Args:
            comment_id: Comment ID
            category: Classified category
            sentiment: Classified sentiment
            topics: Classified topics (optional)
            doctor_specialization: Doctor specialization (optional)
            licensed_professional_type: Licensed professional type (optional)
            conn: Database connection
        """
        # Convert topics list to JSON string
        topics_json = json.dumps(topics) if topics else None

        conn.execute(
            """
            UPDATE comments
            SET category = ?, sentiment = ?, topics = ?,
                doctor_specialization = ?, licensed_professional_type = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (category, sentiment, topics_json, doctor_specialization, licensed_professional_type, comment_id)
        )

    @staticmethod
    def get_comments_for_document(
        document_id: str,
        conn,
        skip_processed: bool = False,
    ) -> List[CommentData]:
        """Fetch all comments for a document.

        Args:
            document_id: Document ID
            conn: Database connection
            skip_processed: If True, only fetch comments that haven't been processed yet
                           (i.e., comments without sentiment or category)

        Returns:
            List of CommentData objects
        """
        if skip_processed:
            cur = conn.execute(
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
            cur = conn.execute(
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

    @staticmethod
    def _build_filter_clause(
        document_id: str,
        sentiment_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        topics_filter: Optional[List[str]] = None,
        topic_filter_mode: str = "any",
        doctor_specialization_filter: Optional[str] = None,
        licensed_professional_type_filter: Optional[str] = None
    ) -> Tuple[str, List]:
        """Build WHERE clause and parameters for filtering comments.

        Args:
            document_id: Document ID (required)
            sentiment_filter: Filter by sentiment
            category_filter: Filter by category
            topics_filter: Filter by topics
            topic_filter_mode: 'any' or 'all' for topic filtering
            doctor_specialization_filter: Filter by doctor specialization
            licensed_professional_type_filter: Filter by licensed professional type

        Returns:
            Tuple of (where_clause, params)
        """
        where_clauses = ["c.document_id = ?"]
        params = [document_id]

        if sentiment_filter:
            where_clauses.append("c.sentiment = ?")
            params.append(sentiment_filter)

        if category_filter:
            where_clauses.append("c.category = ?")
            params.append(category_filter)

        if topics_filter:
            # For SQLite, we need to use JSON functions to check array membership
            if topic_filter_mode == "all":
                # Check that all topics from topics_filter are in the comment's topics
                # This is complex in SQLite - we'll need to use a subquery or JSON functions
                # For simplicity, we'll check each topic individually
                for topic in topics_filter:
                    where_clauses.append("EXISTS (SELECT 1 FROM json_each(c.topics) WHERE value = ?)")
                    params.append(topic)
            else:  # any
                # Check if any topic from topics_filter is in the comment's topics
                placeholders = ','.join('?' * len(topics_filter))
                where_clauses.append(f"EXISTS (SELECT 1 FROM json_each(c.topics) WHERE value IN ({placeholders}))")
                params.extend(topics_filter)

        if doctor_specialization_filter:
            where_clauses.append("c.doctor_specialization = ?")
            params.append(doctor_specialization_filter)

        if licensed_professional_type_filter:
            where_clauses.append("c.licensed_professional_type = ?")
            params.append(licensed_professional_type_filter)

        return " AND ".join(where_clauses), params

    @staticmethod
    def _sort_breakdown_for_visualization(
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

    @staticmethod
    def get_statistics(
        document_id: str,
        group_by: str,
        conn,
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
            conn: Database connection
            sentiment_filter: Optional sentiment filter
            category_filter: Optional category filter
            topics_filter: Optional topics filter
            topic_filter_mode: 'any' or 'all' for topic filtering
            doctor_specialization_filter: Optional doctor specialization filter
            licensed_professional_type_filter: Optional licensed professional type filter

        Returns:
            StatisticsResponse with total_comments and breakdown

        Raises:
            ValueError: If group_by is not valid
        """
        valid_group_by = ["sentiment", "category", "topic", "doctor_specialization", "licensed_professional_type"]
        if group_by not in valid_group_by:
            raise ValueError(f"group_by must be one of {valid_group_by}")

        where_clause, filter_params = CommentRepository._build_filter_clause(
            document_id, sentiment_filter, category_filter, topics_filter, topic_filter_mode,
            doctor_specialization_filter, licensed_professional_type_filter
        )

        # Get total count
        cur = conn.execute(
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

        cur = conn.execute(query, filter_params)
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
        breakdown = CommentRepository._sort_breakdown_for_visualization(breakdown, group_by)

        return StatisticsResponse(
            total_comments=total,
            breakdown=breakdown
        )

    @staticmethod
    def get_full_text(comment_id: str, conn) -> str:
        """Get the full text of a comment.

        Args:
            comment_id: Comment ID
            conn: Database connection

        Returns:
            The full comment text, or empty string if not found
        """
        cur = conn.execute(
            "SELECT comment_text FROM comments WHERE id = ?",
            (comment_id,)
        )
        row = cur.fetchone()
        return row[0] if row else ""

    @staticmethod
    def get_sentiment_by_category(
        document_id: str,
        conn
    ) -> Dict[str, Dict[str, int]]:
        """Get sentiment breakdown for each category.

        This method returns a cross-tabulation of categories and sentiments,
        showing how many comments of each sentiment exist for each category.

        Args:
            document_id: Document ID
            conn: Database connection

        Returns:
            Nested dict: {category: {sentiment: count}}
            Only includes categories with at least 1 comment.
            Results are ordered by total comment count (descending).
        """
        # Query to get category, sentiment, and count
        # Group by both dimensions and order by total comments per category
        cur = conn.execute(
            """
            WITH category_totals AS (
                SELECT category, COUNT(*) as total
                FROM comments
                WHERE document_id = ? AND category IS NOT NULL
                GROUP BY category
            )
            SELECT
                c.category,
                c.sentiment,
                COUNT(*) as count
            FROM comments c
            INNER JOIN category_totals ct ON c.category = ct.category
            WHERE c.document_id = ? AND c.category IS NOT NULL
            GROUP BY c.category, c.sentiment, ct.total
            ORDER BY ct.total DESC, c.category, c.sentiment
            """,
            (document_id, document_id)
        )

        rows = cur.fetchall()

        # Build nested dictionary structure
        result: Dict[str, Dict[str, int]] = {}
        for row in rows:
            category, sentiment, count = row
            if category not in result:
                result[category] = {}
            result[category][sentiment or "unclear"] = count

        return result

    @staticmethod
    def get_sentiment_by_specialization(
        document_id: str,
        field_name: str,
        category_filter: str,
        conn
    ) -> Dict[str, Dict[str, int]]:
        """Get sentiment breakdown for specializations within a category.

        This method returns a cross-tabulation of specializations (either
        doctor_specialization or licensed_professional_type) and sentiments,
        filtered to a specific category.

        Args:
            document_id: Document ID
            field_name: Field to group by ('doctor_specialization' or 'licensed_professional_type')
            category_filter: Category to filter by (e.g., 'Physicians & Surgeons')
            conn: Database connection

        Returns:
            Nested dict: {specialization: {sentiment: count}}
            Only includes specializations with at least 1 comment.
            Results are ordered by total comment count (descending).
        """
        # Validate field_name for SQL injection protection
        valid_fields = ["doctor_specialization", "licensed_professional_type"]
        if field_name not in valid_fields:
            raise ValueError(f"field_name must be one of {valid_fields}")

        # Query to get specialization, sentiment, and count within the filtered category
        # Group by both dimensions and order by total comments per specialization
        query = f"""
            WITH specialization_totals AS (
                SELECT {field_name}, COUNT(*) as total
                FROM comments
                WHERE document_id = ?
                  AND category = ?
                  AND {field_name} IS NOT NULL
                GROUP BY {field_name}
            )
            SELECT
                c.{field_name},
                c.sentiment,
                COUNT(*) as count
            FROM comments c
            INNER JOIN specialization_totals st ON c.{field_name} = st.{field_name}
            WHERE c.document_id = ?
              AND c.category = ?
              AND c.{field_name} IS NOT NULL
            GROUP BY c.{field_name}, c.sentiment, st.total
            ORDER BY st.total DESC, c.{field_name}, c.sentiment
        """

        cur = conn.execute(query, (document_id, category_filter, document_id, category_filter))
        rows = cur.fetchall()

        # Build nested dictionary structure
        result: Dict[str, Dict[str, int]] = {}
        for row in rows:
            specialization, sentiment, count = row
            if specialization not in result:
                result[specialization] = {}
            result[specialization][sentiment or "unclear"] = count

        return result


class CommentChunkRepository:
    """Repository for comment chunk and embedding operations."""

    @staticmethod
    def delete_chunks_for_comment(comment_id: str, conn) -> None:
        """Delete existing chunks for a comment.

        Args:
            comment_id: Comment ID
            conn: Database connection
        """
        conn.execute(
            "DELETE FROM comment_chunks WHERE comment_id = ?",
            (comment_id,)
        )

    @staticmethod
    def store_comment_chunks(
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
        CommentChunkRepository.delete_chunks_for_comment(comment_id, conn)

        # Insert new chunks
        for idx, (chunk_text, embedding) in enumerate(chunks_with_embeddings):
            # Serialize embedding to bytes
            embedding_blob = serialize_vector(embedding)

            conn.execute(
                """
                INSERT INTO comment_chunks (comment_id, chunk_text, chunk_index, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (comment_id, chunk_text, idx, embedding_blob),
            )

    @staticmethod
    def search_by_vector(
        document_id: str,
        query_embedding: List[float],
        conn,
        limit: int = 10,
        sentiment_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        topics_filter: Optional[List[str]] = None,
        topic_filter_mode: str = "any",
        doctor_specialization_filter: Optional[str] = None,
        licensed_professional_type_filter: Optional[str] = None
    ) -> List[CommentChunkSearchResult]:
        """Search comment chunks using vector similarity.

        Args:
            document_id: Document ID to search within
            query_embedding: Embedding vector for the query
            conn: Database connection
            limit: Maximum number of chunks to return
            sentiment_filter: Optional sentiment filter
            category_filter: Optional category filter
            topics_filter: Optional topics filter
            topic_filter_mode: 'any' or 'all' for topic filtering
            doctor_specialization_filter: Optional doctor specialization filter
            licensed_professional_type_filter: Optional licensed professional type filter

        Returns:
            List of CommentChunkSearchResult objects
        """
        # Build filter clause for comments table
        where_clause, filter_params = CommentRepository._build_filter_clause(
            document_id, sentiment_filter, category_filter, topics_filter, topic_filter_mode,
            doctor_specialization_filter, licensed_professional_type_filter
        )

        # Serialize query embedding
        query_blob = serialize_vector(query_embedding)

        # Build the query using sqlite-vec's distance function
        query = f"""
            SELECT
                cc.comment_id,
                cc.chunk_text,
                cc.chunk_index,
                vec_distance_cosine(cc.embedding, ?) as distance,
                c.sentiment,
                c.category,
                c.topics
            FROM comment_chunks cc
            JOIN comments c ON cc.comment_id = c.id
            WHERE {where_clause}
            ORDER BY distance
            LIMIT ?
        """

        # Assemble params in correct order
        params = [query_blob] + filter_params + [limit]

        cur = conn.execute(query, params)
        rows = cur.fetchall()

        results = []
        for row in rows:
            # Parse topics from JSON string
            topics = json.loads(row[6]) if row[6] else []

            results.append(CommentChunkSearchResult(
                comment_id=row[0],
                chunk_text=row[1],
                chunk_index=row[2],
                distance=float(row[3]),
                sentiment=row[4],
                category=row[5],
                topics=topics
            ))

        return results
