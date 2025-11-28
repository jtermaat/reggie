"""RAG sub-agent graph for retrieving relevant comments."""

import logging
from typing import List, Dict, Any, Literal
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from ..models.agent import (
    RAGState,
    RAGSnippet,
    HasEnoughInformation
)
from ..config import get_config
from ..db.unit_of_work import UnitOfWork
from ..exceptions import RAGSearchError
from .chains import (
    create_vector_search_chain,
    create_hybrid_search_chain,
    create_query_generation_chain,
    create_relevance_assessment_chain,
    create_comment_selection_chain
)
from .status import emit_status

logger = logging.getLogger(__name__)


def format_document_context(aggregated_keywords: dict) -> str:
    """
    Format aggregated keywords for the query generation prompt.

    Args:
        aggregated_keywords: Dict with keywords_phrases and entities lists

    Returns:
        Formatted string for prompt context
    """
    lines = []

    keywords = aggregated_keywords.get("keywords_phrases", [])
    entities = aggregated_keywords.get("entities", [])

    if keywords:
        lines.append("### Keywords/Phrases Available")
        lines.append("Use these exact terms for keyword_query:")
        lines.append(", ".join(keywords[:50]))  # Limit for prompt size
        lines.append("")

    if entities:
        lines.append("### Entities Mentioned")
        lines.append(", ".join(entities[:30]))  # Limit for prompt size

    if not lines:
        return "No keywords available for this document."

    return "\n".join(lines)


@lru_cache(maxsize=1)
def create_rag_graph() -> StateGraph:
    """Create and compile the RAG sub-agent graph (cached).

    This graph iteratively searches for relevant comment chunks, assesses whether
    enough information has been found, and retrieves complete comments.

    Returns:
        Compiled StateGraph
    """
    # Create reusable LCEL chains
    query_generation_chain = create_query_generation_chain()
    relevance_chain = create_relevance_assessment_chain()
    selection_chain = create_comment_selection_chain()

    async def generate_query(state: RAGState) -> Dict[str, Any]:
        """
        Generate dual search queries using document context.

        This node:
        1. Fetches aggregated_keywords from documents table (or uses cached)
        2. Formats keywords as context for the LLM
        3. Calls the query generation chain
        4. Returns both semantic and keyword queries

        Args:
            state: Current RAG state

        Returns:
            Updated state with queries and cached keywords
        """
        document_id = state["document_id"]
        logger.debug(f"Generating queries for document {document_id}")

        # Get the user's question from messages
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not user_messages:
            logger.error("No user question found in messages")
            raise ValueError("No user question found in messages")

        question = user_messages[-1].content

        # Build context about previous iterations if any
        iteration_count = state.get("iteration_count", 0)
        if iteration_count == 0:
            iteration_context = "This is the first search iteration."
        else:
            all_retrieved = state.get("all_retrieved_chunks", {})
            comment_count = len(all_retrieved)
            iteration_context = (
                f"Previous searches found {comment_count} comments. "
                "Generate DIFFERENT queries to find additional relevant content. "
                "Try varying the keyword_query terms or using different filters."
            )

        # Fetch aggregated keywords (use cached if available)
        aggregated_keywords = state.get("aggregated_keywords")
        if not aggregated_keywords:
            async with UnitOfWork() as uow:
                aggregated_keywords = await uow.documents.get_aggregated_keywords(document_id)

            if not aggregated_keywords:
                logger.warning(f"No keywords found for document {document_id}, using empty context")
                document_context = "No keywords available for this document."
            else:
                document_context = format_document_context(aggregated_keywords)
        else:
            document_context = format_document_context(aggregated_keywords)

        # Use LLM to generate dual queries and filters
        query_gen = await query_generation_chain.ainvoke({
            "question": question,
            "iteration_context": iteration_context,
            "document_context": document_context,
        })

        logger.info(f"Generated semantic_query: {query_gen.semantic_query}")
        logger.info(f"Generated keyword_query: {query_gen.keyword_query}")
        logger.debug(f"Query reasoning: {query_gen.reasoning}")

        # Build filters dict from generated values
        filters = {}
        if query_gen.sentiment_filter:
            filters["sentiment"] = query_gen.sentiment_filter.value
        if query_gen.category_filter:
            filters["category"] = query_gen.category_filter.value
        if query_gen.topics_filter:
            filters["topics"] = [t.value for t in query_gen.topics_filter]

        return {
            "aggregated_keywords": aggregated_keywords,  # Cache for subsequent iterations
            "current_semantic_query": query_gen.semantic_query,
            "current_keyword_query": query_gen.keyword_query,
            "filters": filters,
            "topic_filter_mode": query_gen.topic_filter_mode,
            "messages": [
                AIMessage(content=(
                    f"Searching with:\n"
                    f"  semantic: '{query_gen.semantic_query}'\n"
                    f"  keywords: '{query_gen.keyword_query}'\n"
                    f"  reasoning: {query_gen.reasoning}"
                ))
            ],
        }

    async def search_vectors(state: RAGState) -> Dict[str, Any]:
        """
        Search for relevant comment chunks using the generated queries.

        In hybrid mode, passes separate queries:
        - semantic_query → embedded for vector search
        - keyword_query → used for ts_rank_cd full-text search

        Args:
            state: Current RAG state with queries

        Returns:
            Updated state with search results
        """
        document_id = state["document_id"]
        semantic_query = state.get("current_semantic_query", "")
        keyword_query = state.get("current_keyword_query", "")

        logger.debug(f"Searching with semantic='{semantic_query}', keywords='{keyword_query}'")

        # Create search chain with current state parameters
        config = get_config()
        filters = state.get("filters", {})

        # Choose search mode based on configuration
        if config.search_mode == "hybrid":
            search_chain = create_hybrid_search_chain(
                document_id=document_id,
                limit=config.vector_search_limit,
                vector_weight=config.hybrid_vector_weight,
                fts_weight=config.hybrid_fts_weight,
                rrf_k=config.hybrid_rrf_k,
                sentiment_filter=filters.get("sentiment"),
                category_filter=filters.get("category"),
                topics_filter=filters.get("topics"),
                topic_filter_mode=state.get("topic_filter_mode", "any")
            )
            search_mode_label = "hybrid"

            # Pass both queries as dict
            query_input = {
                "semantic_query": semantic_query,
                "keyword_query": keyword_query,
            }
        else:
            # Default to vector-only search
            search_chain = create_vector_search_chain(
                document_id=document_id,
                limit=config.vector_search_limit,
                sentiment_filter=filters.get("sentiment"),
                category_filter=filters.get("category"),
                topics_filter=filters.get("topics"),
                topic_filter_mode=state.get("topic_filter_mode", "any")
            )
            search_mode_label = "vector"
            # Vector-only uses semantic query
            query_input = semantic_query

        # Emit status with filter info if present
        filter_parts = []
        if filters.get("sentiment"):
            filter_parts.append(f"sentiment={filters['sentiment']}")
        if filters.get("category"):
            filter_parts.append(f"category={filters['category']}")
        if filters.get("topics"):
            filter_parts.append(f"topics={filters['topics']}")

        if filter_parts:
            emit_status(f"querying comment text ({search_mode_label}, filtered on {', '.join(filter_parts)})")
        else:
            emit_status(f"querying comment text ({search_mode_label})")

        # Use LCEL chain to search
        results = await search_chain.ainvoke(query_input)

        logger.info(f"Search returned {len(results)} chunks")

        iteration_count = state.get("iteration_count", 0)
        if not results and iteration_count == 0:
            logger.warning(f"No comment chunks found for query")
            raise RAGSearchError(f"No comment chunks found for query")

        # Organize by comment_id
        all_retrieved = state.get("all_retrieved_chunks", {})
        for result in results:
            comment_id = result.comment_id
            if comment_id not in all_retrieved:
                all_retrieved[comment_id] = []
            all_retrieved[comment_id].append(result)

        return {
            "search_results": results,
            "all_retrieved_chunks": all_retrieved,
            "messages": [AIMessage(content=f"Retrieved {len(results)} relevant chunks")]
        }

    async def assess_relevance(state: RAGState) -> Dict[str, Any]:
        """Assess whether we have enough relevant information to answer the question."""
        logger.debug("Assessing relevance of retrieved information")

        emit_status("evaluating result completeness")

        config = get_config()

        # Get user's question
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        question = user_messages[-1].content

        # Format the chunks we've seen so far
        all_retrieved = state.get("all_retrieved_chunks", {})
        chunks_summary = []
        for comment_id, chunks in all_retrieved.items():
            for chunk in chunks:
                chunks_summary.append(
                    f"Comment {comment_id} (chunk {chunk.chunk_index}): {chunk.chunk_text[:config.chunk_preview_chars]}..."
                )

        # Use LCEL chain for assessment
        assessment = await relevance_chain.ainvoke({
            "question": question,
            "chunk_count": len(chunks_summary),
            "comment_count": len(all_retrieved),
            "chunks_summary": chr(10).join(chunks_summary[:config.chunks_summary_display_limit])
        })

        logger.debug(f"Assessment: {assessment.has_enough_information.value}, reasoning: {assessment.reasoning}")

        return {
            "has_enough_information": assessment.has_enough_information,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": [AIMessage(content=f"Assessment: {assessment.reasoning}")]
        }

    async def select_relevant_comments(state: RAGState) -> Dict[str, Any]:
        """Select which comments contain relevant information."""
        logger.debug("Selecting relevant comments")

        emit_status("selecting relevant comments")

        config = get_config()

        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        question = user_messages[-1].content

        # Format all unique comments we've retrieved
        all_retrieved = state.get("all_retrieved_chunks", {})
        comment_summaries = []
        for comment_id, chunks in all_retrieved.items():
            # Combine chunks for this comment
            chunk_texts = [c.chunk_text for c in chunks]
            combined = " ... ".join(chunk_texts)
            comment_summaries.append(f"Comment ID: {comment_id}\n{combined[:config.comment_preview_chars]}...")

        # Use LCEL chain for selection
        selection = await selection_chain.ainvoke({
            "question": question,
            "comment_summaries": chr(10).join(comment_summaries)
        })

        logger.debug(f"Selected {len(selection.relevant_comment_ids)} relevant comments")

        return {
            "messages": [AIMessage(content=f"Selected {len(selection.relevant_comment_ids)} relevant comments")],
            "relevant_comment_ids": selection.relevant_comment_ids
        }

    async def extract_snippets(state: RAGState) -> Dict[str, Any]:
        """Retrieve full text from each selected comment."""
        logger.debug("Retrieving full text from relevant comments")

        emit_status("retrieving selected comments")

        # Get relevant comment IDs from the selection step
        relevant_ids = state.get("relevant_comment_ids", [])
        if not relevant_ids:
            # Fallback to all retrieved chunks if no selection was made
            relevant_ids = list(state.get("all_retrieved_chunks", {}).keys())

        snippets = []

        async with UnitOfWork() as uow:
            for comment_id in relevant_ids:
                # Get full comment text using repository
                full_text = await uow.comments.get_full_text(comment_id)

                if not full_text:
                    logger.warning(f"No text found for comment {comment_id}")
                    continue

                # Return full comment text instead of extracted snippet
                snippets.append(RAGSnippet(
                    comment_id=comment_id,
                    snippet=full_text
                ))

        logger.debug(f"Retrieved {len(snippets)} complete comments")

        if not snippets:
            logger.error("Failed to retrieve any comments")
            raise RAGSearchError("Failed to retrieve any comments")

        return {
            "final_snippets": snippets,
            "messages": [AIMessage(content=f"Retrieved {len(snippets)} complete comments")]
        }

    def should_continue_searching(state: RAGState) -> Literal["search", "select"]:
        """Determine if we should continue searching or move to selection."""
        # Check if we've reached max iterations (safety check)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        if iteration_count >= max_iterations:
            logger.debug("Reached max iterations, moving to selection")
            return "select"

        # Check the binary decision from assess_relevance
        has_enough = state.get("has_enough_information")
        if has_enough == HasEnoughInformation.yes:
            logger.debug("Assessment says we have enough information, moving to selection")
            return "select"
        else:
            logger.debug("Assessment says we need more information, continuing search")
            return "search"

    # Build the graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("search", search_vectors)
    workflow.add_node("assess", assess_relevance)
    workflow.add_node("select", select_relevant_comments)
    workflow.add_node("extract", extract_snippets)

    # Add edges
    workflow.set_entry_point("generate_query")
    workflow.add_edge("generate_query", "search")
    workflow.add_edge("search", "assess")

    # Conditional edge from assess
    workflow.add_conditional_edges(
        "assess",
        should_continue_searching,
        {
            "search": "generate_query",  # Loop back to try a new query
            "select": "select"  # Move to selection
        }
    )

    workflow.add_edge("select", "extract")
    workflow.add_edge("extract", END)

    return workflow.compile()


@traceable(
    name="rag_search",
    run_type="chain"
)
async def run_rag_search(
    document_id: str,
    question: str
) -> List[RAGSnippet]:
    """Run the RAG search graph to find relevant comments.

    The graph will autonomously generate queries and filters based on the question.

    Args:
        document_id: The document to search within
        question: The user's question

    Returns:
        List of RAGSnippet objects containing complete comment text
    """
    # Add metadata to LangSmith trace
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_tags(["rag_search", f"doc-{document_id}"])
        run_tree.add_metadata({
            "document_id": document_id,
            "question": question,
            "question_length": len(question)
        })

    graph = create_rag_graph()
    config = get_config()

    initial_state: RAGState = {
        "document_id": document_id,
        "messages": [HumanMessage(content=question)],
        "aggregated_keywords": None,  # Will be fetched on first generate_query
        "filters": {},
        "topic_filter_mode": "any",
        "current_semantic_query": "",
        "current_keyword_query": "",
        "search_results": [],
        "all_retrieved_chunks": {},
        "final_snippets": [],
        "relevant_comment_ids": [],
        "iteration_count": 0,
        "max_iterations": config.max_rag_iterations
    }

    # Run the graph with proper configuration for tracing
    graph_config = {
        "run_name": "rag_graph_execution",
        "tags": ["rag", "retrieval", f"doc-{document_id}"],
        "metadata": {
            "max_iterations": config.max_rag_iterations
        }
    }

    final_state = await graph.ainvoke(initial_state, config=graph_config)

    # Add result metadata
    if run_tree:
        run_tree.add_metadata({
            "snippets_found": len(final_state["final_snippets"]),
            "iterations_used": final_state.get("iteration_count", 0)
        })

    return final_state["final_snippets"]
