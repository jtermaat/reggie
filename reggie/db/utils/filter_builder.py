"""SQL filter building utilities for comment queries."""

from typing import Optional, List, Tuple
import json


def build_comment_filter_clause(
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
    where_clauses = ["c.document_id = %s"]
    params = [document_id]

    if sentiment_filter:
        where_clauses.append("c.sentiment = %s")
        params.append(sentiment_filter)

    if category_filter:
        where_clauses.append("c.category = %s")
        params.append(category_filter)

    if topics_filter:
        # For PostgreSQL, we use JSONB array containment operators
        if topic_filter_mode == "all":
            # Check that c.topics contains all topics from topics_filter
            # Use @> operator (contains)
            where_clauses.append("c.topics @> %s::jsonb")
            params.append(json.dumps(topics_filter))
        else:  # any
            # Check if c.topics contains any of the topics from topics_filter
            # Use ?| operator (contains any of the keys/values)
            where_clauses.append("c.topics ?| %s")
            params.append(topics_filter)

    if doctor_specialization_filter:
        where_clauses.append("c.doctor_specialization = %s")
        params.append(doctor_specialization_filter)

    if licensed_professional_type_filter:
        where_clauses.append("c.licensed_professional_type = %s")
        params.append(licensed_professional_type_filter)

    return " AND ".join(where_clauses), params
