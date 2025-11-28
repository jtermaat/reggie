"""Repository for comment chunk and embedding operations."""

import logging
import psycopg
from typing import List, Tuple, Optional

from ...models.agent import CommentChunkSearchResult
from ..utils.vector_utils import serialize_vector
from ..utils.filter_builder import build_comment_filter_clause
from ..exceptions import RepositoryError

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Repository for comment chunk and embedding operations."""

    def __init__(self, connection: psycopg.AsyncConnection):
        """
        Initialize repository with database connection.

        Args:
            connection: PostgreSQL async database connection
        """
        self._conn = connection

    async def delete_chunks_for_comment(self, comment_id: str) -> None:
        """Delete existing chunks for a comment.

        Args:
            comment_id: Comment ID

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            async with self._conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM comment_chunks WHERE comment_id = %s",
                    (comment_id,)
                )
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to delete chunks for comment: {e}") from e

    async def store_comment_chunks(
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
            await self.delete_chunks_for_comment(comment_id)

            # Insert new chunks
            async with self._conn.cursor() as cur:
                for idx, (chunk_text, embedding) in enumerate(chunks_with_embeddings):
                    # Serialize embedding (passthrough for pgvector)
                    embedding_vector = serialize_vector(embedding)

                    await cur.execute(
                        """
                        INSERT INTO comment_chunks (comment_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (comment_id, chunk_text, idx, embedding_vector),
                    )
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to store comment chunks: {e}") from e

    async def search_by_vector(
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

            # Prepare query embedding for pgvector
            query_vector = serialize_vector(query_embedding)

            # Build the query using pgvector's <=> operator for cosine distance
            query = f"""
                SELECT
                    cc.comment_id,
                    cc.chunk_text,
                    cc.chunk_index,
                    (cc.embedding <=> %s::vector) as distance,
                    c.sentiment,
                    c.category,
                    c.topics
                FROM comment_chunks cc
                JOIN comments c ON cc.comment_id = c.id
                WHERE {where_clause}
                ORDER BY distance
                LIMIT %s
            """

            # Assemble params in correct order
            params = [query_vector] + filter_params + [limit]

            async with self._conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

            results = []
            for row in rows:
                # Topics is already JSONB, psycopg handles deserialization
                topics = row["topics"] if row["topics"] else []

                results.append(CommentChunkSearchResult(
                    comment_id=row["comment_id"],
                    chunk_text=row["chunk_text"],
                    chunk_index=row["chunk_index"],
                    distance=float(row["distance"]),
                    sentiment=row["sentiment"],
                    category=row["category"],
                    topics=topics
                ))

            return results
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to search by vector: {e}") from e

    async def search_by_fulltext(
        self,
        document_id: str,
        query_text: str,
        limit: int = 10,
        sentiment_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        topics_filter: Optional[List[str]] = None,
        topic_filter_mode: str = "any",
        doctor_specialization_filter: Optional[str] = None,
        licensed_professional_type_filter: Optional[str] = None
    ) -> List[CommentChunkSearchResult]:
        """Search comment chunks using PostgreSQL full-text search.

        Uses ts_rank_cd for BM25-like ranking based on term frequency
        and inverse document frequency.

        Args:
            document_id: Document ID to search within
            query_text: Natural language query to search for
            limit: Maximum number of chunks to return
            sentiment_filter: Optional sentiment filter
            category_filter: Optional category filter
            topics_filter: Optional topics filter
            topic_filter_mode: 'any' or 'all' for topic filtering
            doctor_specialization_filter: Optional doctor specialization filter
            licensed_professional_type_filter: Optional licensed professional type filter

        Returns:
            List of CommentChunkSearchResult objects with FTS scores

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            # Build filter clause for comments table
            where_clause, filter_params = build_comment_filter_clause(
                document_id, sentiment_filter, category_filter, topics_filter, topic_filter_mode,
                doctor_specialization_filter, licensed_professional_type_filter
            )

            # Build the query using PostgreSQL full-text search
            # ts_rank_cd uses cover density ranking (similar to BM25)
            query = f"""
                SELECT
                    cc.comment_id,
                    cc.chunk_text,
                    cc.chunk_index,
                    ts_rank_cd(cc.chunk_text_tsv, plainto_tsquery('english', %s)) as score,
                    c.sentiment,
                    c.category,
                    c.topics
                FROM comment_chunks cc
                JOIN comments c ON cc.comment_id = c.id
                WHERE {where_clause}
                AND cc.chunk_text_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """

            # Assemble params in correct order
            params = [query_text] + filter_params + [query_text, limit]

            async with self._conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

            results = []
            for row in rows:
                topics = row["topics"] if row["topics"] else []
                # Convert FTS score to distance-like metric (lower is better)
                # ts_rank_cd returns 0-1 typically, invert for consistency
                distance = 1.0 - min(float(row["score"]), 1.0)

                results.append(CommentChunkSearchResult(
                    comment_id=row["comment_id"],
                    chunk_text=row["chunk_text"],
                    chunk_index=row["chunk_index"],
                    distance=distance,
                    sentiment=row["sentiment"],
                    category=row["category"],
                    topics=topics
                ))

            return results
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to search by fulltext: {e}") from e

    async def search_hybrid(
        self,
        document_id: str,
        query_embedding: List[float],
        query_text: str,
        limit: int = 10,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
        rrf_k: int = 60,
        sentiment_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        topics_filter: Optional[List[str]] = None,
        topic_filter_mode: str = "any",
        doctor_specialization_filter: Optional[str] = None,
        licensed_professional_type_filter: Optional[str] = None
    ) -> List[CommentChunkSearchResult]:
        """Hybrid search combining vector similarity and full-text search.

        Uses Reciprocal Rank Fusion (RRF) to combine rankings from both
        search methods: RRF_score = sum(weight_i / (k + rank_i))

        Args:
            document_id: Document ID to search within
            query_embedding: Embedding vector for semantic search
            query_text: Natural language query for full-text search
            limit: Maximum number of chunks to return
            vector_weight: Weight for vector search (0-1)
            fts_weight: Weight for full-text search (0-1)
            rrf_k: RRF constant (higher = less aggressive rank decay)
            sentiment_filter: Optional sentiment filter
            category_filter: Optional category filter
            topics_filter: Optional topics filter
            topic_filter_mode: 'any' or 'all' for topic filtering
            doctor_specialization_filter: Optional doctor specialization filter
            licensed_professional_type_filter: Optional licensed professional type filter

        Returns:
            List of CommentChunkSearchResult with combined RRF scores

        Raises:
            RepositoryError: If database operation fails
        """
        try:
            where_clause, filter_params = build_comment_filter_clause(
                document_id, sentiment_filter, category_filter,
                topics_filter, topic_filter_mode,
                doctor_specialization_filter, licensed_professional_type_filter
            )

            # Prepare query embedding for pgvector
            query_vector = serialize_vector(query_embedding)

            # Fetch more candidates from each method for better fusion
            candidate_limit = limit * 3

            # CTE-based query combining both search methods with RRF
            query = f"""
                WITH vector_search AS (
                    SELECT
                        cc.id,
                        cc.comment_id,
                        cc.chunk_text,
                        cc.chunk_index,
                        c.sentiment,
                        c.category,
                        c.topics,
                        ROW_NUMBER() OVER (ORDER BY cc.embedding <=> %s::vector) as rank
                    FROM comment_chunks cc
                    JOIN comments c ON cc.comment_id = c.id
                    WHERE {where_clause}
                    LIMIT %s
                ),
                fts_search AS (
                    SELECT
                        cc.id,
                        cc.comment_id,
                        cc.chunk_text,
                        cc.chunk_index,
                        c.sentiment,
                        c.category,
                        c.topics,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank_cd(cc.chunk_text_tsv, plainto_tsquery('english', %s)) DESC
                        ) as rank
                    FROM comment_chunks cc
                    JOIN comments c ON cc.comment_id = c.id
                    WHERE {where_clause}
                    AND cc.chunk_text_tsv @@ plainto_tsquery('english', %s)
                    LIMIT %s
                ),
                combined AS (
                    SELECT
                        COALESCE(v.id, f.id) as id,
                        COALESCE(v.comment_id, f.comment_id) as comment_id,
                        COALESCE(v.chunk_text, f.chunk_text) as chunk_text,
                        COALESCE(v.chunk_index, f.chunk_index) as chunk_index,
                        COALESCE(v.sentiment, f.sentiment) as sentiment,
                        COALESCE(v.category, f.category) as category,
                        COALESCE(v.topics, f.topics) as topics,
                        -- RRF formula: weighted sum of 1/(k+rank)
                        COALESCE(%s::float / (%s + v.rank), 0) +
                        COALESCE(%s::float / (%s + f.rank), 0) as rrf_score
                    FROM vector_search v
                    FULL OUTER JOIN fts_search f ON v.id = f.id
                )
                SELECT
                    comment_id, chunk_text, chunk_index, sentiment, category, topics,
                    rrf_score,
                    1.0 - rrf_score as distance
                FROM combined
                ORDER BY rrf_score DESC
                LIMIT %s
            """

            # Build params in order they appear in query
            params = (
                [query_vector] + filter_params + [candidate_limit] +  # vector_search
                [query_text] + filter_params + [query_text, candidate_limit] +  # fts_search
                [vector_weight, rrf_k, fts_weight, rrf_k] +  # RRF weights
                [limit]  # final limit
            )

            async with self._conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

            results = []
            for row in rows:
                topics = row["topics"] if row["topics"] else []
                results.append(CommentChunkSearchResult(
                    comment_id=row["comment_id"],
                    chunk_text=row["chunk_text"],
                    chunk_index=row["chunk_index"],
                    distance=float(row["distance"]),
                    sentiment=row["sentiment"],
                    category=row["category"],
                    topics=topics
                ))

            return results
        except psycopg.Error as e:
            raise RepositoryError(f"Failed to perform hybrid search: {e}") from e
