"""Statistics models"""

from datetime import datetime
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from .cost import CostReport


class ProcessingStats(BaseModel):
    """Statistics for processing operations."""

    document_id: str
    comments_processed: int = 0
    chunks_created: int = 0
    errors: int = 0
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    cost_report: Optional["CostReport"] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True
