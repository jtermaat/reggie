"""Document data models"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model."""

    id: str
    title: Optional[str] = None
    object_id: str
    docket_id: Optional[str] = None
    document_type: Optional[str] = None
    posted_date: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class DocumentStats(BaseModel):
    """Statistics for a loaded document."""

    id: str
    title: Optional[str]
    docket_id: Optional[str]
    posted_date: Optional[datetime]
    comment_count: int
    unique_categories: int
    loaded_at: Optional[datetime]

    class Config:
        """Pydantic configuration."""
        from_attributes = True
