"""Agent tools for interacting with comment data."""

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from langsmith.run_helpers import get_current_run_tree

from ..db.connection import get_connection
from ..db.repository import CommentRepository
from ..exceptions import RAGSearchError
from ..prompts import prompts
from ..models.agent import GetStatisticsInput
from .rag_graph import run_rag_search
from .status import emit_status
from .visualizations import emit_visualization

logger = logging.getLogger(__name__)


async def get_statistics(
    document_id: str,
    group_by: str,
    sentiment_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    topics_filter: Optional[list] = None,
    topic_filter_mode: str = "any",
    doctor_specialization_filter: Optional[str] = None,
    licensed_professional_type_filter: Optional[str] = None
) -> str:
    """Get statistical breakdown of comments.

    Args:
        document_id: The document to query
        group_by: What to group results by ('sentiment', 'category', 'topic', 'doctor_specialization', or 'licensed_professional_type')
        sentiment_filter: Optional sentiment filter
        category_filter: Optional category filter
        topics_filter: Optional list of topics to filter by
        topic_filter_mode: 'any' or 'all' for topic filtering
        doctor_specialization_filter: Optional doctor specialization filter
        licensed_professional_type_filter: Optional licensed professional type filter

    Returns:
        Formatted string with statistical breakdown
    """
    # Build filter info for status message and metadata
    filter_parts = []
    filters_applied = {}
    if sentiment_filter:
        filter_parts.append(f"sentiment={sentiment_filter}")
        filters_applied["sentiment"] = sentiment_filter
    if category_filter:
        filter_parts.append(f"category={category_filter}")
        filters_applied["category"] = category_filter
    if topics_filter:
        filter_parts.append(f"topics={topics_filter}")
        filters_applied["topics"] = topics_filter
    if doctor_specialization_filter:
        filter_parts.append(f"doctor_specialization={doctor_specialization_filter}")
        filters_applied["doctor_specialization"] = doctor_specialization_filter
    if licensed_professional_type_filter:
        filter_parts.append(f"licensed_professional_type={licensed_professional_type_filter}")
        filters_applied["licensed_professional_type"] = licensed_professional_type_filter

    if filter_parts:
        emit_status(f"querying comment statistics (filtered on {', '.join(filter_parts)})")
    else:
        emit_status("querying comment statistics")

    # Add metadata to LangSmith trace
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_tags(["statistical_query", f"group_by_{group_by}"])
        run_tree.add_metadata({
            "document_id": document_id,
            "query_type": "statistical",
            "group_by": group_by,
            "filters_applied": bool(filters_applied),
            "filters": filters_applied,
            "topic_filter_mode": topic_filter_mode
        })

    async with get_connection() as conn:
        result = await CommentRepository.get_statistics(
            document_id=document_id,
            group_by=group_by,
            conn=conn,
            sentiment_filter=sentiment_filter,
            category_filter=category_filter,
            topics_filter=topics_filter,
            topic_filter_mode=topic_filter_mode,
            doctor_specialization_filter=doctor_specialization_filter,
            licensed_professional_type_filter=licensed_professional_type_filter
        )

    # Emit visualization data
    emit_visualization({
        "type": "statistics",
        "group_by": group_by,
        "total_comments": result.total_comments,
        "breakdown": [
            {
                "value": item.value,
                "count": item.count,
                "percentage": item.percentage
            }
            for item in result.breakdown
        ],
        "filters": filters_applied if filters_applied else None
    })

    # Format the result
    output = [f"Total comments matching filters: {result.total_comments}\n"]
    output.append(f"Breakdown by {group_by}:")

    for item in result.breakdown:
        output.append(
            f"  • {item.value}: {item.count} ({item.percentage}%)"
        )

    # Add guidance to avoid redundancy with the visualization
    output.append("\nNote: The user can also see those numbers. Focus your response on any summary conclusions, and how you can help them next.")

    return "\n".join(output)


async def search_comments(
    document_id: str,
    question: str
) -> str:
    """Search comment text for relevant information.

    The RAG system will autonomously generate optimal queries and filters.

    Args:
        document_id: The document to search within
        question: The question to answer

    Returns:
        Formatted text with complete relevant comments and IDs
    """
    # Add metadata to LangSmith trace
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_tags(["rag_query", "text_search"])
        run_tree.add_metadata({
            "document_id": document_id,
            "query_type": "rag_search",
            "question_length": len(question)
        })

    snippets = await run_rag_search(
        document_id=document_id,
        question=question
    )

    if not snippets:
        raise RAGSearchError(f"No relevant comments found for question: {question}")

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
        topic_filter_mode: str = "any",
        doctor_specialization_filter: Optional[str] = None,
        licensed_professional_type_filter: Optional[str] = None
    ) -> str:
        return await get_statistics(
            document_id=document_id,
            group_by=group_by,
            sentiment_filter=sentiment_filter,
            category_filter=category_filter,
            topics_filter=topics_filter,
            topic_filter_mode=topic_filter_mode,
            doctor_specialization_filter=doctor_specialization_filter,
            licensed_professional_type_filter=licensed_professional_type_filter
        )

    async def search_comments_bound(question: str) -> str:
        """Find relevant comments that answer the given question."""
        return await search_comments(
            document_id=document_id,
            question=question
        )

    # Create simple input schema for search_comments
    from pydantic import BaseModel, Field

    class SearchCommentsInput(BaseModel):
        """Input schema for search_comments tool."""
        question: str = Field(
            description="The question you want to answer using comment data"
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
