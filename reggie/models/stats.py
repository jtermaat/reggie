"""Statistics models"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ProcessingStats(BaseModel):
    """Statistics for processing operations."""

    document_id: str
    comments_processed: int = 0
    chunks_created: int = 0
    errors: int = 0
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True
