"""Repository for comment chunk and embedding operations."""

import json
import logging
import sqlite3
from typing import List, Tuple, Optional

from ...models.agent import CommentChunkSearchResult
from ..utils.vector_utils import serialize_vector
from ..utils.filter_builder import build_comment_filter_clause
from ..exceptions import RepositoryError

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Repository for comment chunk and embedding operations."""

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize repository with database connection.

        Args:
            connection: SQLite database connection
        """
        self._conn = connection

    def delete_chunks_for_comment(self, comment_id: str) -> None:
        """Delete existing chunks for a comment.

        Args:
            comment_id: Comment ID

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            self._conn.execute(
                "DELETE FROM comment_chunks WHERE comment_id = ?",
                (comment_id,)
            )
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to delete chunks for comment: {e}") from e

    def store_comment_chunks(
        self,
        comment_id: str,
        chunks_with_embeddings: List[Tuple[str, List[float]]],
    ) -> None:
        """Store comment chunks and embeddings in database.

        Args:
            comment_id: Comment ID
            chunks_with_embeddings: List of (chunk_text, embedding) tuples

        Raises:
            RepositoryError: If database operation fails
        """
        if not chunks_with_embeddings:
            return

        try:
            # Delete existing chunks
            self.delete_chunks_for_comment(comment_id)

            # Insert new chunks
            for idx, (chunk_text, embedding) in enumerate(chunks_with_embeddings):
                # Serialize embedding to bytes
                embedding_blob = serialize_vector(embedding)

                self._conn.execute(
                    """
                    INSERT INTO comment_chunks (comment_id, chunk_text, chunk_index, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (comment_id, chunk_text, idx, embedding_blob),
                )
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to store comment chunks: {e}") from e

    def search_by_vector(
        self,
        document_id: str,
        query_embedding: List[float],
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
            limit: Maximum number of chunks to return
            sentiment_filter: Optional sentiment filter
            category_filter: Optional category filter
            topics_filter: Optional topics filter
            topic_filter_mode: 'any' or 'all' for topic filtering
            doctor_specialization_filter: Optional doctor specialization filter
            licensed_professional_type_filter: Optional licensed professional type filter

        Returns:
            List of CommentChunkSearchResult objects

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            # Build filter clause for comments table
            where_clause, filter_params = build_comment_filter_clause(
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

            cur = self._conn.execute(query, params)
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
        except sqlite3.Error as e:
            raise RepositoryError(f"Failed to search by vector: {e}") from e
