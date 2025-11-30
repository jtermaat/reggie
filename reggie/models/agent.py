"""Models for agent operations."""

from typing import List, Annotated, Optional, TypedDict, Sequence, Literal
from operator import add
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from .comment import Category, Sentiment, Topic, DoctorSpecialization, LicensedProfessionalType


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


class RetrievedComment(BaseModel):
    """A complete comment retrieved by RAG search."""

    comment_id: str = Field(description="ID of the comment")
    text: str = Field(description="The full comment text")


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

    group_by: Literal["sentiment", "category", "topic", "doctor_specialization", "licensed_professional_type"] = Field(
        description="What to group results by: 'sentiment', 'category', 'topic', 'doctor_specialization', or 'licensed_professional_type'"
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
    doctor_specialization_filter: Optional[DoctorSpecialization] = Field(
        default=None,
        description="Optional filter for specific doctor specialization (only applicable when dealing with physicians)"
    )
    licensed_professional_type_filter: Optional[LicensedProfessionalType] = Field(
        default=None,
        description="Optional filter for specific licensed professional type (only applicable when dealing with licensed clinicians)"
    )




# RAG graph models
class QueryGeneration(BaseModel):
    """
    Dual query and filter generation for hybrid RAG search.

    Generates two separate queries optimized for different search backends:
    - semantic_query: Verbose, contextual query for vector/embedding search
    - keyword_query: Concise, precise query for full-text search (ts_rank_cd)
    """

    semantic_query: str = Field(
        description=(
            "Verbose query (8-15 words) for vector/embedding similarity search. "
            "Use complete phrases that capture the full meaning and intent. "
            "Include related concepts, synonyms, and contextual language."
        )
    )

    keyword_query: str = Field(
        description=(
            "Concise query (2-5 terms) for PostgreSQL full-text search. "
            "Use EXACT terms from the document's keyword_phrases when relevant. "
            "These terms are ANDed together, so fewer precise terms = better recall."
        )
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
    doctor_specialization_filter: Optional[DoctorSpecialization] = Field(
        default=None,
        description="Optional filter for specific doctor specialization (only applicable when category is 'physicians_surgeons')"
    )
    licensed_professional_type_filter: Optional[LicensedProfessionalType] = Field(
        default=None,
        description="Optional filter for specific licensed professional type (only applicable when category is 'licensed_clinicians')"
    )
    reasoning: str = Field(
        description="Brief explanation of why these queries and filters were chosen"
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

    # Current search parameters - dual queries for hybrid search
    current_semantic_query: str  # Verbose query for vector search
    current_keyword_query: str   # Concise query for FTS

    search_results: List[dict]
    all_retrieved_chunks: dict  # comment_id -> chunks
    retrieved_comments: List[RetrievedComment]
    relevant_comment_ids: List[str]
    iteration_count: int
    max_iterations: int
    has_enough_information: HasEnoughInformation
