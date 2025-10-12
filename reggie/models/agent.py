"""Models for agent operations."""

from typing import List, Annotated, Optional, TypedDict, Sequence, Literal
from operator import add
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from .comment import Category, Sentiment, Topic


# Enums for RAG graph
class HasEnoughInformation(str, Enum):
    """Binary decision for whether enough information has been retrieved."""
    yes = "yes"
    no = "no"


# Repository result models
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


class CommentChunkSearchResult(BaseModel):
    """Result from vector similarity search on comment chunks."""

    comment_id: str = Field(description="ID of the comment this chunk belongs to")
    chunk_text: str = Field(description="The text of the chunk")
    chunk_index: int = Field(description="Index of this chunk within the comment")
    distance: float = Field(description="Vector similarity distance")
    sentiment: Optional[str] = Field(description="Sentiment of the parent comment")
    category: Optional[str] = Field(description="Category of the parent comment")
    topics: List[str] = Field(default_factory=list, description="Topics of the parent comment")


# Tool input models
class GetStatisticsInput(BaseModel):
    """Input schema for get_statistics tool."""

    group_by: Literal["sentiment", "category", "topic"] = Field(
        description="What to group results by: 'sentiment', 'category', or 'topic'"
    )
    sentiment_filter: Optional[Sentiment] = Field(
        default=None,
        description="Optional filter for specific sentiment"
    )
    category_filter: Optional[Category] = Field(
        default=None,
        description="Optional filter for specific category"
    )
    topics_filter: Optional[List[Topic]] = Field(
        default=None,
        description="Optional list of topics to filter by"
    )
    topic_filter_mode: Literal["any", "all"] = Field(
        default="any",
        description="When using topics_filter: 'any' means has any topic, 'all' means has all topics"
    )




# RAG graph models
class QueryGeneration(BaseModel):
    """Query and filter generation for RAG search."""

    query: str = Field(
        description="The search query text to use for vector similarity search"
    )
    sentiment_filter: Optional[Sentiment] = Field(
        default=None,
        description="Optional filter for specific sentiment"
    )
    category_filter: Optional[Category] = Field(
        default=None,
        description="Optional filter for specific category"
    )
    topics_filter: Optional[List[Topic]] = Field(
        default=None,
        description="Optional list of topics to filter by"
    )
    topic_filter_mode: Literal["any", "all"] = Field(
        default="any",
        description="When using topics_filter: 'any' means has any topic, 'all' means has all topics"
    )
    reasoning: str = Field(
        description="Brief explanation of why this query and these filters were chosen"
    )


class RelevanceAssessment(BaseModel):
    """Assessment of whether enough relevant information has been found."""

    has_enough_information: HasEnoughInformation = Field(
        description="Whether the retrieved chunks contain enough information to answer the question"
    )
    reasoning: str = Field(
        description="Brief explanation of why we do or don't have enough information"
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


class RAGState(TypedDict, total=False):
    """State for the RAG graph.

    Uses TypedDict for better LangGraph integration and state handling.
    """

    # Required fields
    document_id: str
    messages: Annotated[Sequence[BaseMessage], add]

    # Optional fields (total=False allows these to be missing)
    filters: dict
    topic_filter_mode: str
    current_query: str
    search_results: List[dict]
    all_retrieved_chunks: dict  # comment_id -> chunks
    final_snippets: List[RAGSnippet]
    relevant_comment_ids: List[str]
    iteration_count: int
    max_iterations: int
    has_enough_information: HasEnoughInformation
