"""RAG search tool for finding relevant comment text."""

import logging
from typing import Optional
from langchain_core.tools import tool

from ..rag_graph import run_rag_search
from ...exceptions import RAGSearchError

logger = logging.getLogger(__name__)


@tool
async def search_comments(
    document_id: str,
    query: str,
    sentiment_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    topics_filter: Optional[list] = None,
    topic_filter_mode: str = "any"
) -> str:
    """Search comment text to find relevant information.

    Use this tool to find what commenters said about specific topics or questions.
    The tool will search through comment text and return relevant snippets.

    Args:
        document_id: The document ID to search within
        query: The question or topic to search for (e.g., "what did people say about Medicare requirements?")
        sentiment_filter: Optional - only search comments with specific sentiment
        category_filter: Optional - only search comments from specific category
        topics_filter: Optional - only search comments discussing certain topics
        topic_filter_mode: When filtering by topics, use 'any' or 'all'

    Returns:
        Formatted text with relevant comment snippets and their IDs

    Examples:
        - "What did physicians say about the new requirements?"
          search_comments(document_id="...", query="new requirements", category_filter="Physicians & Surgeons")

        - "What concerns do people have about costs?"
          search_comments(document_id="...", query="concerns about costs")
    """
    filters = {}
    if sentiment_filter:
        filters["sentiment"] = sentiment_filter
    if category_filter:
        filters["category"] = category_filter
    if topics_filter:
        filters["topics"] = topics_filter

    # Run the RAG search
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

    for i, item in enumerate(snippets, 1):
        output.append(f"{i}. Comment ID: {item['comment_id']}")
        output.append(f"   {item['snippet']}\n")

    return "\n".join(output)
