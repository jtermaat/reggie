"""Cost tracking models for OpenAI API usage"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class UsageCost(BaseModel):
    """Token usage and cost for a single operation or model."""

    prompt_tokens: int = Field(default=0, description="Number of prompt/input tokens")
    completion_tokens: int = Field(default=0, description="Number of completion/output tokens")
    total_tokens: int = Field(default=0, description="Total tokens used")
    cost_usd: float = Field(default=0.0, description="Total cost in USD")
    model_name: Optional[str] = Field(default=None, description="Model name used")

    def __add__(self, other: "UsageCost") -> "UsageCost":
        """Add two UsageCost objects together."""
        if not isinstance(other, UsageCost):
            return NotImplemented

        return UsageCost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            model_name=self.model_name or other.model_name
        )

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class CostReport(BaseModel):
    """Aggregate cost report for a command execution."""

    categorization_cost: UsageCost = Field(
        default_factory=UsageCost,
        description="Costs for comment categorization"
    )
    embedding_cost: UsageCost = Field(
        default_factory=UsageCost,
        description="Costs for comment embedding"
    )
    agent_cost: UsageCost = Field(
        default_factory=UsageCost,
        description="Costs for agent operations (discuss command)"
    )
    total_cost_usd: float = Field(
        default=0.0,
        description="Total cost across all operations in USD"
    )
    costs_by_model: Dict[str, UsageCost] = Field(
        default_factory=dict,
        description="Breakdown of costs by model name"
    )

    def calculate_total(self) -> None:
        """Calculate total cost from component costs."""
        self.total_cost_usd = (
            self.categorization_cost.cost_usd +
            self.embedding_cost.cost_usd +
            self.agent_cost.cost_usd
        )

    class Config:
        """Pydantic configuration."""
        from_attributes = True
