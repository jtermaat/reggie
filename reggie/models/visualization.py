"""Data models for visualizations."""

from typing import Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field


class StatisticsVisualizationData(BaseModel):
    """Data for single-dimension statistics visualization.

    Used when displaying statistics grouped by one dimension
    (sentiment, category, or topic).
    """

    type: Literal["statistics"] = "statistics"
    group_by: Literal["sentiment", "category", "topic"] = Field(
        description="The dimension being grouped by"
    )
    total_comments: int = Field(description="Total number of comments in the result")
    breakdown: List[Dict[str, Union[str, int, float]]] = Field(
        description="List of items with 'value', 'count', and 'percentage'"
    )
    filters: Optional[Dict[str, Union[str, List[str]]]] = Field(
        default=None,
        description="Filters applied to the query"
    )


class OppositionSupportVisualizationData(BaseModel):
    """Data for opposition/support visualization by category.

    Used for the centered horizontal bar chart showing sentiment
    breakdown across categories.
    """

    type: Literal["opposition_support"] = "opposition_support"
    document_id: str = Field(description="Document ID being visualized")
    document_title: Optional[str] = Field(
        default=None,
        description="Title of the document"
    )
    total_comments: int = Field(description="Total number of comments")
    breakdown: Dict[str, Dict[str, int]] = Field(
        description="Nested dict: {category: {sentiment: count}}"
    )


# Discriminated union for all visualization types
VisualizationData = Union[
    StatisticsVisualizationData,
    OppositionSupportVisualizationData
]
