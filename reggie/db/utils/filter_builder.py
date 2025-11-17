"""SQL filter building utilities for comment queries."""

from typing import Optional, List, Tuple


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
