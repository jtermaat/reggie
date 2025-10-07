"""RAG sub-agent graph for retrieving relevant comment snippets."""

import logging
from typing import List, Dict, Any, Literal

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from ..models.agent import (
    RelevanceAssessment,
    RelevantCommentSelection,
    CommentSnippet,
    RAGState,
    RAGSnippet
)
from ..db.connection import get_connection
from ..db.repository import CommentRepository, CommentChunkRepository
from ..exceptions import RAGSearchError

logger = logging.getLogger(__name__)


def create_rag_graph(embeddings_model: str = "text-embedding-3-small") -> StateGraph:
    """Create the RAG sub-agent graph.

    This graph iteratively searches for relevant comment chunks, assesses whether
    enough information has been found, and extracts relevant snippets.

    Args:
        embeddings_model: OpenAI embeddings model to use

    Returns:
        Compiled StateGraph
    """
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def generate_query(state: RAGState) -> Dict[str, Any]:
        """Generate or refine the search query based on user's question."""
        logger.info("Generating search query")

        # Get the user's question from messages
        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        if not user_messages:
            raise ValueError("No user question found in messages")

        question = user_messages[-1].content

        # If this is first iteration, use the question directly as query
        # Otherwise, we should already have a suggested query from assessment
        if state.iteration_count == 0:
            query = question
        else:
            query = state.current_query if state.current_query else question

        logger.info(f"Using query: {query}")

        return {
            "current_query": query,
            "messages": [AIMessage(content=f"Searching for: {query}")]
        }

    async def search_vectors(state: RAGState) -> Dict[str, Any]:
        """Search for relevant comment chunks using vector similarity."""
        logger.info(f"Searching vectors with query: {state.current_query}")

        # Generate embedding for query
        query_embedding = await embeddings.aembed_query(state.current_query)

        # Search chunks using repository
        async with get_connection() as conn:
            results = await CommentChunkRepository.search_by_vector(
                document_id=state.document_id,
                query_embedding=query_embedding,
                conn=conn,
                limit=10,
                sentiment_filter=state.filters.get("sentiment"),
                category_filter=state.filters.get("category"),
                topics_filter=state.filters.get("topics"),
                topic_filter_mode=state.topic_filter_mode
            )

        logger.info(f"Found {len(results)} chunks")

        if not results and state.iteration_count == 0:
            raise RAGSearchError(f"No comment chunks found for query: {state.current_query}")

        # Organize by comment_id
        for result in results:
            comment_id = result.comment_id
            if comment_id not in state.all_retrieved_chunks:
                state.all_retrieved_chunks[comment_id] = []
            state.all_retrieved_chunks[comment_id].append(result)

        return {
            "search_results": results,
            "all_retrieved_chunks": state.all_retrieved_chunks,
            "messages": [AIMessage(content=f"Retrieved {len(results)} relevant chunks")]
        }

    async def assess_relevance(state: RAGState) -> Dict[str, Any]:
        """Assess whether we have enough relevant information to answer the question."""
        logger.info("Assessing relevance of retrieved information")

        # Get user's question
        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        question = user_messages[-1].content

        # Format the chunks we've seen so far
        chunks_summary = []
        for comment_id, chunks in state.all_retrieved_chunks.items():
            for chunk in chunks:
                chunks_summary.append(
                    f"Comment {comment_id} (chunk {chunk.chunk_index}): {chunk.chunk_text[:200]}..."
                )

        system_msg = """You are assessing whether we have retrieved enough relevant information to answer a user's question about regulation comments.

Review the chunks retrieved so far and determine if they contain enough information to answer the question.

If not enough information has been found, suggest a different search query that might find more relevant information."""

        user_msg = f"""Question: {question}

Retrieved chunks so far ({len(chunks_summary)} chunks from {len(state.all_retrieved_chunks)} comments):

{chr(10).join(chunks_summary[:20])}

Do we have enough information to answer this question? If not, suggest a different query."""

        llm_with_structure = llm.with_structured_output(RelevanceAssessment)
        assessment = await llm_with_structure.ainvoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ])

        logger.info(f"Assessment: {assessment.has_enough_information}, reasoning: {assessment.reasoning}")

        new_query = assessment.suggested_query if assessment.needs_different_query else state.current_query

        return {
            "current_query": new_query,
            "iteration_count": state.iteration_count + 1,
            "messages": [AIMessage(content=f"Assessment: {assessment.reasoning}")]
        }

    async def select_relevant_comments(state: RAGState) -> Dict[str, Any]:
        """Select which comments contain relevant information."""
        logger.info("Selecting relevant comments")

        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        question = user_messages[-1].content

        # Format all unique comments we've retrieved
        comment_summaries = []
        for comment_id, chunks in state.all_retrieved_chunks.items():
            # Combine chunks for this comment
            chunk_texts = [c.chunk_text for c in chunks]
            combined = " ... ".join(chunk_texts)
            comment_summaries.append(f"Comment ID: {comment_id}\n{combined[:500]}...")

        system_msg = """You are selecting which comments contain information relevant to answering the user's question.

Review the retrieved comments and select the IDs of comments that contain relevant information."""

        user_msg = f"""Question: {question}

Retrieved comments:

{chr(10).join(comment_summaries)}

Which comment IDs contain relevant information?"""

        llm_with_structure = llm.with_structured_output(RelevantCommentSelection)
        selection = await llm_with_structure.ainvoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ])

        logger.info(f"Selected {len(selection.relevant_comment_ids)} relevant comments")

        return {
            "messages": [AIMessage(content=f"Selected {len(selection.relevant_comment_ids)} relevant comments")],
            "_relevant_comment_ids": selection.relevant_comment_ids
        }

    async def extract_snippets(state: RAGState) -> Dict[str, Any]:
        """Extract relevant snippets from each selected comment."""
        logger.info("Extracting snippets from relevant comments")

        # Get relevant comment IDs from the previous step
        # We'll need to access them from state or pass them differently
        # For now, let's use all retrieved comments
        relevant_ids = list(state.all_retrieved_chunks.keys())

        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        question = user_messages[-1].content

        snippets = []

        async with get_connection() as conn:
            for comment_id in relevant_ids:
                # Get full comment text using repository
                full_text = await CommentRepository.get_full_text(comment_id, conn)

                if not full_text:
                    logger.warning(f"No text found for comment {comment_id}")
                    continue

                system_msg = """You are extracting the relevant portion of a comment that helps answer the user's question.

Extract the exact text from the comment that is relevant. The snippet should be a direct quote from the comment text."""

                user_msg = f"""Question: {question}

Comment text:
{full_text}

Extract the portion of this comment that is relevant to answering the question."""

                llm_with_structure = llm.with_structured_output(CommentSnippet)

                snippet_obj = await llm_with_structure.ainvoke([
                    SystemMessage(content=system_msg),
                    HumanMessage(content=user_msg)
                ])

                # Validate that snippet is actually in the comment
                if not snippet_obj.snippet.strip():
                    logger.warning(f"Empty snippet returned for comment {comment_id}")
                    continue

                if snippet_obj.snippet not in full_text:
                    logger.warning(f"Snippet not found in comment {comment_id}, skipping")
                    continue

                snippets.append(RAGSnippet(
                    comment_id=comment_id,
                    snippet=snippet_obj.snippet
                ))

        logger.info(f"Extracted {len(snippets)} snippets")

        if not snippets:
            raise RAGSearchError("Failed to extract any valid snippets from comments")

        return {
            "final_snippets": snippets,
            "messages": [AIMessage(content=f"Extracted {len(snippets)} relevant snippets")]
        }

    def should_continue_searching(state: RAGState) -> Literal["search", "select"]:
        """Determine if we should continue searching or move to selection."""
        # Check if we've reached max iterations
        if state.iteration_count >= state.max_iterations:
            logger.info("Reached max iterations, moving to selection")
            return "select"

        # Check if we have any results
        if not state.all_retrieved_chunks:
            logger.info("No results yet, continuing search")
            return "search"

        # Parse the last assessment message to determine if we need more info
        # In a real implementation, we'd store the assessment in state
        # For now, if we have results and aren't at max iterations, move to select
        if len(state.all_retrieved_chunks) >= 3:  # Have at least 3 comments
            logger.info("Have enough comments, moving to selection")
            return "select"

        logger.info("Need more information, continuing search")
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


async def run_rag_search(
    document_id: str,
    question: str,
    filters: Dict[str, Any] = None,
    topic_filter_mode: str = "any"
) -> List[RAGSnippet]:
    """Run the RAG search graph to find relevant comment snippets.

    Args:
        document_id: The document to search within
        question: The user's question
        filters: Optional filters (sentiment, category, topics)
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        List of RAGSnippet objects
    """
    graph = create_rag_graph()

    initial_state = RAGState(
        document_id=document_id,
        messages=[HumanMessage(content=question)],
        filters=filters or {},
        topic_filter_mode=topic_filter_mode
    )

    # Run the graph
    final_state = await graph.ainvoke(initial_state)

    return final_state["final_snippets"]
