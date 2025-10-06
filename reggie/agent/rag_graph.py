"""RAG sub-agent graph for retrieving relevant comment snippets."""

import logging
from typing import List, Dict, Any, Annotated, Literal
from operator import add

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .tools import search_comment_chunks, get_full_comment_text

logger = logging.getLogger(__name__)


class RelevanceAssessment(BaseModel):
    """Assessment of whether enough relevant information has been found."""

    has_enough_information: bool = Field(
        description="True if the retrieved chunks contain enough information to answer the question"
    )
    reasoning: str = Field(
        description="Brief explanation of why we do or don't have enough information"
    )
    needs_different_query: bool = Field(
        description="True if we should try a different search query to find more relevant information"
    )
    suggested_query: str = Field(
        default="",
        description="If needs_different_query is True, suggest a new query to try"
    )


class RelevantCommentSelection(BaseModel):
    """Selection of which comments contain relevant information."""

    relevant_comment_ids: List[str] = Field(
        description="List of comment IDs that contain relevant information to answer the question"
    )
    reasoning: str = Field(
        description="Brief explanation of why these comments are relevant"
    )


class CommentSnippet(BaseModel):
    """A snippet extracted from a comment."""

    snippet: str = Field(
        description="The exact text from the comment that is relevant to the question"
    )


class RAGState(BaseModel):
    """State for the RAG graph."""

    document_id: str
    messages: Annotated[List[BaseMessage], add]
    filters: Dict[str, Any] = Field(default_factory=dict)
    topic_filter_mode: str = "any"
    current_query: str = ""
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    all_retrieved_chunks: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)  # comment_id -> chunks
    final_snippets: List[Dict[str, str]] = Field(default_factory=list)  # List of {comment_id, snippet}
    iteration_count: int = 0
    max_iterations: int = 3


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

        # Search chunks
        results = await search_comment_chunks(
            document_id=state.document_id,
            query_embedding=query_embedding,
            limit=10,
            filters=state.filters,
            topic_filter_mode=state.topic_filter_mode
        )

        logger.info(f"Found {len(results)} chunks")

        # Organize by comment_id
        for result in results:
            comment_id = result["comment_id"]
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
                    f"Comment {comment_id} (chunk {chunk['chunk_index']}): {chunk['chunk_text'][:200]}..."
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
            chunk_texts = [c["chunk_text"] for c in chunks]
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

        for comment_id in relevant_ids:
            # Get full comment text
            full_text = await get_full_comment_text(comment_id)

            if not full_text:
                continue

            system_msg = """You are extracting the relevant portion of a comment that helps answer the user's question.

Extract the exact text from the comment that is relevant. The snippet should be a direct quote from the comment text."""

            user_msg = f"""Question: {question}

Comment text:
{full_text}

Extract the portion of this comment that is relevant to answering the question."""

            llm_with_structure = llm.with_structured_output(CommentSnippet)

            try:
                snippet_obj = await llm_with_structure.ainvoke([
                    SystemMessage(content=system_msg),
                    HumanMessage(content=user_msg)
                ])

                # Validate that snippet is actually in the comment
                if snippet_obj.snippet.strip() and snippet_obj.snippet in full_text:
                    snippets.append({
                        "comment_id": comment_id,
                        "snippet": snippet_obj.snippet
                    })
                else:
                    # Retry logic or just use first 500 chars as fallback
                    logger.warning(f"Snippet not found in comment {comment_id}, using excerpt")
                    snippets.append({
                        "comment_id": comment_id,
                        "snippet": full_text[:500] + "..."
                    })

            except Exception as e:
                logger.error(f"Error extracting snippet from comment {comment_id}: {e}")
                # Use excerpt as fallback
                snippets.append({
                    "comment_id": comment_id,
                    "snippet": full_text[:500] + "..."
                })

        logger.info(f"Extracted {len(snippets)} snippets")

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
) -> List[Dict[str, str]]:
    """Run the RAG search graph to find relevant comment snippets.

    Args:
        document_id: The document to search within
        question: The user's question
        filters: Optional filters (sentiment, category, topics)
        topic_filter_mode: 'any' or 'all' for topic filtering

    Returns:
        List of {comment_id, snippet} dictionaries
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
