"""Cost tracking utility for OpenAI API usage using LangChain callbacks."""

import logging
from typing import Dict, Optional, List
from contextlib import asynccontextmanager, contextmanager

from langchain_community.callbacks import get_openai_callback

from ..models.cost import UsageCost, CostReport

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks OpenAI API costs using LangChain's callback system.

    This class wraps LangChain's get_openai_callback to provide a clean
    interface for tracking costs across categorization, embedding, and
    agent operations.
    """

    def __init__(self):
        """Initialize the cost tracker."""
        self.categorization_costs: List[UsageCost] = []
        self.embedding_costs: List[UsageCost] = []
        self.agent_costs: List[UsageCost] = []
        self._costs_by_model: Dict[str, UsageCost] = {}

    @contextmanager
    def track_operation(self, operation_type: str, model_name: Optional[str] = None):
        """Track costs for a single operation using a context manager.

        Args:
            operation_type: Type of operation ('categorization', 'embedding', or 'agent')
            model_name: Optional model name for more detailed tracking

        Yields:
            The OpenAI callback handler

        Example:
            with tracker.track_operation('categorization', 'gpt-5-nano'):
                # Make OpenAI API calls here
                result = llm.invoke(prompt)
        """
        with get_openai_callback() as cb:
            yield cb

            # After the context exits, record the costs
            usage = UsageCost(
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_tokens=cb.total_tokens,
                cost_usd=cb.total_cost,
                model_name=model_name
            )

            # Store by operation type
            if operation_type == "categorization":
                self.categorization_costs.append(usage)
            elif operation_type == "embedding":
                self.embedding_costs.append(usage)
            elif operation_type == "agent":
                self.agent_costs.append(usage)
            else:
                logger.warning(f"Unknown operation type: {operation_type}")

            # Also track by model if model_name is provided
            if model_name:
                if model_name in self._costs_by_model:
                    self._costs_by_model[model_name] = self._costs_by_model[model_name] + usage
                else:
                    self._costs_by_model[model_name] = usage

    @asynccontextmanager
    async def track_operation_async(self, operation_type: str, model_name: Optional[str] = None):
        """Async version of track_operation.

        Args:
            operation_type: Type of operation ('categorization', 'embedding', or 'agent')
            model_name: Optional model name for more detailed tracking

        Yields:
            The OpenAI callback handler

        Example:
            async with tracker.track_operation_async('categorization', 'gpt-5-nano'):
                # Make async OpenAI API calls here
                result = await llm.ainvoke(prompt)
        """
        with get_openai_callback() as cb:
            yield cb

            # After the context exits, record the costs
            usage = UsageCost(
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_tokens=cb.total_tokens,
                cost_usd=cb.total_cost,
                model_name=model_name
            )

            # Store by operation type
            if operation_type == "categorization":
                self.categorization_costs.append(usage)
            elif operation_type == "embedding":
                self.embedding_costs.append(usage)
            elif operation_type == "agent":
                self.agent_costs.append(usage)
            else:
                logger.warning(f"Unknown operation type: {operation_type}")

            # Also track by model if model_name is provided
            if model_name:
                if model_name in self._costs_by_model:
                    self._costs_by_model[model_name] = self._costs_by_model[model_name] + usage
                else:
                    self._costs_by_model[model_name] = usage

    def _aggregate_costs(self, costs: List[UsageCost]) -> UsageCost:
        """Aggregate a list of UsageCost objects into a single one.

        Args:
            costs: List of UsageCost objects

        Returns:
            Aggregated UsageCost object
        """
        if not costs:
            return UsageCost()

        result = costs[0]
        for cost in costs[1:]:
            result = result + cost

        return result

    def get_report(self) -> CostReport:
        """Generate a cost report from tracked operations.

        Returns:
            CostReport with aggregated costs
        """
        report = CostReport(
            categorization_cost=self._aggregate_costs(self.categorization_costs),
            embedding_cost=self._aggregate_costs(self.embedding_costs),
            agent_cost=self._aggregate_costs(self.agent_costs),
            costs_by_model=self._costs_by_model.copy()
        )

        report.calculate_total()
        return report

    def reset(self) -> None:
        """Reset all tracked costs."""
        self.categorization_costs.clear()
        self.embedding_costs.clear()
        self.agent_costs.clear()
        self._costs_by_model.clear()
