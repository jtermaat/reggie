"""Reusable LangChain chains for common operations using LCEL patterns."""

import logging
from typing import List, Dict, Any

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from ..db.unit_of_work import UnitOfWork
from ..models.agent import (
    CommentChunkSearchResult,
    QueryGeneration,
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


def create_query_generation_chain() -> Runnable:
    """Create LCEL chain for RAG query and filter generation.

    Returns:
        Runnable that takes dict with question, iteration_context
        and returns QueryGeneration
    """
    config = get_config()
    llm = ChatOpenAI(model=config.rag_model)

    # LCEL chain: prompt | llm with structured output
    return (
        prompts.RAG_GENERATE_QUERY
        | llm.with_structured_output(QueryGeneration)
    ).with_config({"run_name": "generate_query"})


def create_relevance_assessment_chain() -> Runnable:
    """Create LCEL chain for RAG relevance assessment.

    Returns:
        Runnable that takes dict with question, chunk_count, comment_count, chunks_summary
        and returns RelevanceAssessment
    """
    config = get_config()
    llm = ChatOpenAI(model=config.rag_model)

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
    llm = ChatOpenAI(model=config.rag_model)

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
    llm = ChatOpenAI(model=config.rag_model)

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

        # Search using repository (async database operation)
        async with UnitOfWork() as uow:
            results = await uow.chunks.search_by_vector(
                document_id=document_id,
                query_embedding=embedding,
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


def create_hybrid_search_chain(
    document_id: str,
    limit: int = 10,
    vector_weight: float = 0.5,
    fts_weight: float = 0.5,
    rrf_k: int = 60,
    sentiment_filter: str = None,
    category_filter: str = None,
    topics_filter: List[str] = None,
    topic_filter_mode: str = "any"
) -> Runnable:
    """Create hybrid search chain combining vector and full-text search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both
    vector similarity search (pgvector) and full-text search (tsvector).

    Args:
        document_id: Document to search within
        limit: Maximum number of results
        vector_weight: Weight for vector search (0-1)
        fts_weight: Weight for full-text search (0-1)
        rrf_k: RRF constant for rank decay
        sentiment_filter: Optional sentiment filter
        category_filter: Optional category filter
        topics_filter: Optional topics filter
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        Runnable that accepts dict with 'semantic_query' and 'keyword_query'
    """
    embedding_chain = create_embedding_chain()

    async def search_impl(
        query_input: Dict[str, str]
    ) -> List[CommentChunkSearchResult]:
        """
        Execute hybrid search with separate queries for each backend.

        Args:
            query_input: Dict with:
                - semantic_query: Query for vector embedding
                - keyword_query: Query for full-text search

        Returns:
            List of search results ranked by RRF fusion
        """
        semantic_query = query_input.get("semantic_query", "")
        keyword_query = query_input.get("keyword_query", semantic_query)

        # Generate embedding from semantic query
        embedding = await embedding_chain.ainvoke(semantic_query)

        # Execute hybrid search with separate queries
        async with UnitOfWork() as uow:
            results = await uow.chunks.search_hybrid(
                document_id=document_id,
                query_embedding=embedding,
                query_text=keyword_query,  # Use keyword query for FTS
                limit=limit,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
                rrf_k=rrf_k,
                sentiment_filter=sentiment_filter,
                category_filter=category_filter,
                topics_filter=topics_filter,
                topic_filter_mode=topic_filter_mode,
            )

        return results

    return RunnableLambda(search_impl).with_config(
        {"run_name": f"hybrid_search_{document_id}"}
    )
