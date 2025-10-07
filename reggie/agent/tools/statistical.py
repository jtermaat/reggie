"""Statistical query tool for comment analysis."""

import logging
from typing import Optional
from langchain_core.tools import tool

from ...db.connection import get_connection
from ...db.repository import CommentRepository
from ...models.agent import StatisticsResponse, StatisticsBreakdownItem

logger = logging.getLogger(__name__)


@tool
async def get_statistics(
    document_id: str,
    group_by: str,
    sentiment_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    topics_filter: Optional[list] = None,
    topic_filter_mode: str = "any"
) -> str:
    """Get statistical breakdown of comments.

    Use this tool to get counts and percentages of comments grouped by sentiment, category, or topic.
    You can also filter the results before grouping.

    Args:
        document_id: The document ID to query
        group_by: What to group results by - 'sentiment', 'category', or 'topic'
        sentiment_filter: Optional - filter to specific sentiment (e.g., 'for', 'against', 'mixed', 'unclear')
        category_filter: Optional - filter to specific category (e.g., 'Physicians & Surgeons')
        topics_filter: Optional - filter to comments discussing certain topics (list of topics)
        topic_filter_mode: When filtering by topics, use 'any' (has any topic) or 'all' (has all topics)

    Returns:
        A formatted string with the statistical breakdown

    Examples:
        - "What is the sentiment breakdown among physicians?"
          get_statistics(document_id="...", group_by="sentiment", category_filter="Physicians & Surgeons")

        - "What categories of people wrote about health equity?"
          get_statistics(document_id="...", group_by="category", topics_filter=["health_equity"])

        - "Among people against the regulation, what topics do they discuss?"
          get_statistics(document_id="...", group_by="topic", sentiment_filter="against")
    """
    async with get_connection() as conn:
        result = await CommentRepository.get_statistics(
            document_id=document_id,
            group_by=group_by,
            conn=conn,
            sentiment_filter=sentiment_filter,
            category_filter=category_filter,
            topics_filter=topics_filter,
            topic_filter_mode=topic_filter_mode
        )

    # Format the result nicely
    output = [f"Total comments matching filters: {result['total_comments']}\n"]
    output.append(f"Breakdown by {group_by}:")

    for item in result["breakdown"]:
        output.append(
            f"  • {item['value']}: {item['count']} ({item['percentage']}%)"
        )

    return "\n".join(output)
