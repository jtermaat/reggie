"""Models for agent operations."""

from typing import List, Optional, Annotated
from operator import add
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


# Tool input models
class StatisticalQueryFilters(BaseModel):
    """Filters for statistical queries."""

    sentiment: Optional[str] = Field(None, description="Filter by specific sentiment value")
    category: Optional[str] = Field(None, description="Filter by specific category value")
    topics: Optional[List[str]] = Field(None, description="Filter by topics (any or all)")


class StatisticalQueryInput(BaseModel):
    """Input for statistical query tool."""

    group_by: str = Field(
        ...,
        description="What to group results by: 'sentiment', 'category', or 'topic'"
    )
    filters: Optional[StatisticalQueryFilters] = Field(
        None,
        description="Optional filters to apply before grouping"
    )
    topic_filter_mode: str = Field(
        "any",
        description="When filtering by topics: 'any' (has any topic) or 'all' (has all topics)"
    )


class TextQueryInput(BaseModel):
    """Input for text-based query tool."""

    query: str = Field(..., description="The question or search query about comment content")
    filters: Optional[StatisticalQueryFilters] = Field(
        None,
        description="Optional filters to apply (sentiment, category, topics)"
    )
    topic_filter_mode: str = Field(
        "any",
        description="When filtering by topics: 'any' or 'all'"
    )


# Tool output models
class StatisticsBreakdownItem(BaseModel):
    """Single item in a statistics breakdown."""

    value: str = Field(description="The category/sentiment/topic value")
    count: int = Field(description="Number of comments with this value")
    percentage: float = Field(description="Percentage of total comments")


class StatisticsResponse(BaseModel):
    """Response from statistical query."""

    total_comments: int = Field(description="Total number of comments matching filters")
    breakdown: List[StatisticsBreakdownItem] = Field(description="Breakdown by requested dimension")


class RAGSnippet(BaseModel):
    """A snippet from a comment found by RAG search."""

    comment_id: str = Field(description="ID of the comment this snippet is from")
    snippet: str = Field(description="The relevant text snippet")


class SearchResponse(BaseModel):
    """Response from RAG search."""

    snippets: List[RAGSnippet] = Field(description="List of relevant comment snippets")


# RAG graph models
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
    filters: dict = Field(default_factory=dict)
    topic_filter_mode: str = "any"
    current_query: str = ""
    search_results: List[dict] = Field(default_factory=list)
    all_retrieved_chunks: dict = Field(default_factory=dict)  # comment_id -> chunks
    final_snippets: List[dict] = Field(default_factory=list)  # List of {comment_id, snippet}
    iteration_count: int = 0
    max_iterations: int = 3
