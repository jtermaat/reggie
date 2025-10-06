"""Tools for the discussion agent."""

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ..db.connection import get_connection
from ..db.repository import CommentChunkRepository

logger = logging.getLogger(__name__)


class StatisticalQueryFilters(BaseModel):
    """Filters for statistical queries."""

    sentiment: Optional[str] = Field(None, description="Filter by specific sentiment value")
    category: Optional[str] = Field(None, description="Filter by specific category value")
    topics: Optional[List[str]] = Field(None, description="Filter by topics (any or all)")


class StatisticalQueryInput(BaseModel):
    """Input for statistical query tool."""

    group_by: str = Field(
        ...,
        description="What to group results by: 'sentiment', 'category', or 'topic'"
    )
    filters: Optional[StatisticalQueryFilters] = Field(
        None,
        description="Optional filters to apply before grouping"
    )
    topic_filter_mode: str = Field(
        "any",
        description="When filtering by topics: 'any' (has any topic) or 'all' (has all topics)"
    )


class TextQueryInput(BaseModel):
    """Input for text-based query tool."""

    query: str = Field(..., description="The question or search query about comment content")
    filters: Optional[StatisticalQueryFilters] = Field(
        None,
        description="Optional filters to apply (sentiment, category, topics)"
    )
    topic_filter_mode: str = Field(
        "any",
        description="When filtering by topics: 'any' or 'all'"
    )


async def get_comment_statistics(
    document_id: str,
    group_by: str,
    filters: Optional[Dict[str, Any]] = None,
    topic_filter_mode: str = "any"
) -> Dict[str, Any]:
    """Get statistical breakdown of comments by sentiment, category, or topic.

    Args:
        document_id: The document ID to query
        group_by: What to group by - 'sentiment', 'category', or 'topic'
        filters: Optional filters (sentiment, category, topics)
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        Dictionary with total_comments and breakdown list
    """
    if group_by not in ["sentiment", "category", "topic"]:
        raise ValueError("group_by must be 'sentiment', 'category', or 'topic'")

    filters = filters or {}
    filter_sentiment = filters.get("sentiment")
    filter_category = filters.get("category")
    filter_topics = filters.get("topics", [])

    # Build WHERE clause
    where_clauses = ["document_id = %s"]
    params = [document_id]

    if filter_sentiment:
        where_clauses.append("sentiment = %s")
        params.append(filter_sentiment)

    if filter_category:
        where_clauses.append("category = %s")
        params.append(filter_category)

    if filter_topics:
        if topic_filter_mode == "all":
            where_clauses.append("topics @> %s::text[]")
        else:  # any
            where_clauses.append("topics && %s::text[]")
        params.append(filter_topics)

    where_clause = " AND ".join(where_clauses)

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            # Get total count
            await cur.execute(
                f"SELECT COUNT(*) FROM comments WHERE {where_clause}",
                params
            )
            total = (await cur.fetchone())[0]

            # Get breakdown
            if group_by == "topic":
                # Unnest topics array for counting
                query = f"""
                    SELECT topic as value, COUNT(*) as count
                    FROM comments, UNNEST(topics) as topic
                    WHERE {where_clause}
                    GROUP BY topic
                    ORDER BY count DESC
                """
            else:
                # Simple grouping
                query = f"""
                    SELECT {group_by} as value, COUNT(*) as count
                    FROM comments
                    WHERE {where_clause}
                    GROUP BY {group_by}
                    ORDER BY count DESC
                """

            await cur.execute(query, params)
            rows = await cur.fetchall()

            breakdown = []
            for row in rows:
                value, count = row
                breakdown.append({
                    "value": value or "unknown",
                    "count": count,
                    "percentage": round((count / total * 100) if total > 0 else 0, 1)
                })

            return {
                "total_comments": total,
                "breakdown": breakdown
            }


async def search_comment_chunks(
    document_id: str,
    query_embedding: List[float],
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    topic_filter_mode: str = "any"
) -> List[Dict[str, Any]]:
    """Search comment chunks using vector similarity.

    Args:
        document_id: The document ID to search within
        query_embedding: The embedding vector for the search query
        limit: Maximum number of chunks to return
        filters: Optional filters (sentiment, category, topics)
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        List of chunks with their comment IDs and metadata
    """
    filters = filters or {}
    filter_sentiment = filters.get("sentiment")
    filter_category = filters.get("category")
    filter_topics = filters.get("topics", [])

    # Build WHERE clause for comments table
    comment_filters = ["c.document_id = %s"]
    params = [document_id]

    if filter_sentiment:
        comment_filters.append("c.sentiment = %s")
        params.append(filter_sentiment)

    if filter_category:
        comment_filters.append("c.category = %s")
        params.append(filter_category)

    if filter_topics:
        if topic_filter_mode == "all":
            comment_filters.append("c.topics @> %s::text[]")
        else:
            comment_filters.append("c.topics && %s::text[]")
        params.append(filter_topics)

    # Add embedding and limit
    params.append(query_embedding)
    params.append(limit)

    comment_filter_clause = " AND ".join(comment_filters)

    query = f"""
        SELECT
            cc.comment_id,
            cc.chunk_text,
            cc.chunk_index,
            cc.embedding <=> %s::vector as distance,
            c.sentiment,
            c.category,
            c.topics
        FROM comment_chunks cc
        JOIN comments c ON cc.comment_id = c.id
        WHERE {comment_filter_clause}
        ORDER BY cc.embedding <=> %s::vector
        LIMIT %s
    """

    # Need to add embedding twice (once for distance calc, once for ordering)
    params_with_dup = params[:-2] + [query_embedding, query_embedding] + [params[-1]]

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params_with_dup)
            rows = await cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    "comment_id": row[0],
                    "chunk_text": row[1],
                    "chunk_index": row[2],
                    "distance": float(row[3]),
                    "sentiment": row[4],
                    "category": row[5],
                    "topics": row[6] or []
                })

            return results


async def get_full_comment_text(comment_id: str) -> str:
    """Get the full text of a comment by ID.

    Args:
        comment_id: The comment ID

    Returns:
        The full comment text
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT comment_text FROM comments WHERE id = %s",
                (comment_id,)
            )
            row = await cur.fetchone()
            return row[0] if row else ""
