"""Agent tools for interacting with comment data."""

import logging
from typing import Optional

from langchain_core.tools import StructuredTool

from ..db.connection import get_connection
from ..db.repository import CommentRepository
from ..exceptions import RAGSearchError
from ..prompts import prompts
from ..models.agent import GetStatisticsInput, SearchCommentsInput
from .rag_graph import run_rag_search
from .status import emit_status

logger = logging.getLogger(__name__)


async def get_statistics(
    document_id: str,
    group_by: str,
    sentiment_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    topics_filter: Optional[list] = None,
    topic_filter_mode: str = "any"
) -> str:
    """Get statistical breakdown of comments.

    Args:
        document_id: The document to query
        group_by: What to group results by ('sentiment', 'category', or 'topic')
        sentiment_filter: Optional sentiment filter
        category_filter: Optional category filter
        topics_filter: Optional list of topics to filter by
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        Formatted string with statistical breakdown
    """
    # Build filter info for status message
    filter_parts = []
    if sentiment_filter:
        filter_parts.append(f"sentiment={sentiment_filter}")
    if category_filter:
        filter_parts.append(f"category={category_filter}")
    if topics_filter:
        filter_parts.append(f"topics={topics_filter}")

    if filter_parts:
        emit_status(f"querying comment statistics (filtered on {', '.join(filter_parts)})")
    else:
        emit_status("querying comment statistics")

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

    # Format the result
    output = [f"Total comments matching filters: {result.total_comments}\n"]
    output.append(f"Breakdown by {group_by}:")

    for item in result.breakdown:
        output.append(
            f"  • {item.value}: {item.count} ({item.percentage}%)"
        )

    return "\n".join(output)


async def search_comments(
    document_id: str,
    query: str,
    sentiment_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    topics_filter: Optional[list] = None,
    topic_filter_mode: str = "any"
) -> str:
    """Search comment text for relevant information.

    Args:
        document_id: The document to search within
        query: The question or topic to search for
        sentiment_filter: Optional sentiment filter
        category_filter: Optional category filter
        topics_filter: Optional list of topics to filter by
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        Formatted text with relevant comment snippets and IDs
    """
    filters = {}
    if sentiment_filter:
        filters["sentiment"] = sentiment_filter
    if category_filter:
        filters["category"] = category_filter
    if topics_filter:
        filters["topics"] = topics_filter

    snippets = await run_rag_search(
        document_id=document_id,
        question=query,
        filters=filters,
        topic_filter_mode=topic_filter_mode
    )

    if not snippets:
        raise RAGSearchError(f"No relevant comments found for query: {query}")

    # Format the results
    output = [f"Found {len(snippets)} relevant comments:\n"]

    for i, snippet in enumerate(snippets, 1):
        output.append(f"{i}. Comment ID: {snippet.comment_id}")
        output.append(f"   {snippet.snippet}\n")

    return "\n".join(output)


def create_discussion_tools(document_id: str) -> list[StructuredTool]:
    """Create tools for the discussion agent with document_id bound.

    Args:
        document_id: The document ID to bind to the tools

    Returns:
        List of StructuredTool instances configured for the document
    """
    # Create partial functions with document_id bound
    async def get_statistics_bound(
        group_by: str,
        sentiment_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        topics_filter: Optional[list] = None,
        topic_filter_mode: str = "any"
    ) -> str:
        return await get_statistics(
            document_id=document_id,
            group_by=group_by,
            sentiment_filter=sentiment_filter,
            category_filter=category_filter,
            topics_filter=topics_filter,
            topic_filter_mode=topic_filter_mode
        )

    async def search_comments_bound(
        query: str,
        sentiment_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        topics_filter: Optional[list] = None,
        topic_filter_mode: str = "any"
    ) -> str:
        return await search_comments(
            document_id=document_id,
            query=query,
            sentiment_filter=sentiment_filter,
            category_filter=category_filter,
            topics_filter=topics_filter,
            topic_filter_mode=topic_filter_mode
        )

    return [
        StructuredTool.from_function(
            func=get_statistics_bound,
            name="get_statistics",
            description=prompts.TOOL_GET_STATISTICS_DESC,
            args_schema=GetStatisticsInput,
            coroutine=get_statistics_bound
        ),
        StructuredTool.from_function(
            func=search_comments_bound,
            name="search_comments",
            description=prompts.TOOL_SEARCH_COMMENTS_DESC,
            args_schema=SearchCommentsInput,
            coroutine=search_comments_bound
        )
    ]
