"""Reusable LangChain chains for common operations using LCEL patterns."""

import logging
from typing import List, Dict, Any

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from ..db.connection import get_connection
from ..db.repository import CommentChunkRepository, CommentRepository
from ..models.agent import (
    CommentChunkSearchResult,
    RelevanceAssessment,
    RelevantCommentSelection,
    CommentSnippet
)
from ..models import CommentClassification
from ..config import get_config
from ..prompts import prompts

logger = logging.getLogger(__name__)


def create_embedding_chain() -> Runnable:
    """Create reusable embedding chain using LCEL.

    Returns:
        Runnable that takes text and returns embedding vector
    """
    config = get_config()
    embeddings = OpenAIEmbeddings(model=config.embeddings_model)

    async def embed_text(text: str) -> List[float]:
        """Embed a single text string."""
        return await embeddings.aembed_query(text)

    return RunnableLambda(embed_text).with_config({"run_name": "embed_text"})


def create_categorization_chain() -> Runnable:
    """Create LCEL chain for comment categorization.

    Returns:
        Runnable that takes context string and returns CommentClassification
    """
    config = get_config()
    llm = ChatOpenAI(model=config.categorization_model)

    # LCEL chain: prompt | llm with structured output
    return (
        prompts.CATEGORIZATION
        | llm.with_structured_output(CommentClassification)
    ).with_config({"run_name": "categorize_comment"})


def create_relevance_assessment_chain() -> Runnable:
    """Create LCEL chain for RAG relevance assessment.

    Returns:
        Runnable that takes dict with question, chunk_count, comment_count, chunks_summary
        and returns RelevanceAssessment
    """
    config = get_config()
    llm = ChatOpenAI(model=config.rag_model, temperature=0)

    # LCEL chain: prompt | llm with structured output
    return (
        prompts.RAG_RELEVANCE_ASSESSMENT
        | llm.with_structured_output(RelevanceAssessment)
    ).with_config({"run_name": "assess_relevance"})


def create_comment_selection_chain() -> Runnable:
    """Create LCEL chain for selecting relevant comments.

    Returns:
        Runnable that takes dict with question, comment_summaries
        and returns RelevantCommentSelection
    """
    config = get_config()
    llm = ChatOpenAI(model=config.rag_model, temperature=0)

    # LCEL chain: prompt | llm with structured output
    return (
        prompts.RAG_SELECT_COMMENTS
        | llm.with_structured_output(RelevantCommentSelection)
    ).with_config({"run_name": "select_comments"})


def create_snippet_extraction_chain() -> Runnable:
    """Create LCEL chain for extracting comment snippets.

    Returns:
        Runnable that takes dict with question, full_text
        and returns CommentSnippet
    """
    config = get_config()
    llm = ChatOpenAI(model=config.rag_model, temperature=0)

    # LCEL chain: prompt | llm with structured output
    return (
        prompts.RAG_EXTRACT_SNIPPET
        | llm.with_structured_output(CommentSnippet)
    ).with_config({"run_name": "extract_snippet"})


def create_vector_search_chain(
    document_id: str,
    limit: int = 10,
    sentiment_filter: str = None,
    category_filter: str = None,
    topics_filter: List[str] = None,
    topic_filter_mode: str = "any"
) -> Runnable:
    """Create reusable vector search chain using LCEL.

    Args:
        document_id: Document to search within
        limit: Maximum number of results
        sentiment_filter: Optional sentiment filter
        category_filter: Optional category filter
        topics_filter: Optional topics filter
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        Runnable that takes query text and returns search results
    """
    embedding_chain = create_embedding_chain()

    async def search_impl(query: str) -> List[CommentChunkSearchResult]:
        """Search for similar comments using the query."""
        # Generate embedding for query using LCEL chain
        embedding = await embedding_chain.ainvoke(query)

        # Search using repository
        async with get_connection() as conn:
            results = await CommentChunkRepository.search_by_vector(
                document_id=document_id,
                query_embedding=embedding,
                conn=conn,
                limit=limit,
                sentiment_filter=sentiment_filter,
                category_filter=category_filter,
                topics_filter=topics_filter,
                topic_filter_mode=topic_filter_mode
            )

        return results

    return RunnableLambda(search_impl).with_config(
        {"run_name": f"search_comments_{document_id}"}
    )
