"""LangChain pipeline for categorizing comments with structured output"""

import asyncio
import logging
from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from langsmith import traceable

from ..config import setup_langsmith, APIConfig, ProcessingConfig
from ..models import CommentClassification, Category, Sentiment
from ..exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class CommentCategorizer:
    """LangChain-based comment categorizer using structured output."""

    SYSTEM_PROMPT = """You are an expert at analyzing public comments on healthcare regulations.

Your task is to classify each comment by:
1. **Category**: Identify the commenter's role/affiliation
2. **Sentiment**: Determine their position on the regulation
3. **Topics**: Identify all topics discussed in the comment (can be multiple)

Guidelines for Category:
- Look for explicit mentions of profession, organization, or role
- Consider the context and language used
- Default to "Individuals / Private Citizens" for unclear cases
- Use "Anonymous / Not Specified" only when truly no information is available

Guidelines for Sentiment:
- "for": Clearly supports or agrees with the regulation
- "against": Clearly opposes or disagrees with the regulation
- "mixed": Expresses both support and opposition
- "unclear": Cannot determine clear position

Guidelines for Topics (select all that apply):
- "reimbursement_payment": Payment rates, reimbursement methodologies, fee schedules
- "cost_financial": Financial impact, costs, economic burden
- "service_coverage": Coverage policies, benefits, covered services
- "access_to_care": Patient access, availability of care, barriers to care
- "workforce_staffing": Staffing requirements, workforce issues, provider shortages
- "methodology_measurement": Quality metrics, measurement approaches, data collection
- "implementation_feasibility": Operational challenges, timeline concerns, practical implementation
- "administrative_burden": Paperwork, reporting requirements, administrative complexity
- "telehealth_digital": Telemedicine, digital health, remote care
- "health_equity": Health disparities, equity concerns, underserved populations
- "quality_programs": Quality reporting, quality improvement, value-based care
- "legal_clarity": Legal concerns, regulatory clarity, compliance issues
- "unclear": Comment topic cannot be determined

Provide your classification along with brief reasoning."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the categorizer.

        Args:
            openai_api_key: OpenAI API key. If None, reads from config/env.
        """
        api_config = APIConfig()
        processing_config = ProcessingConfig()

        api_key = openai_api_key or api_config.openai_api_key
        if not api_key:
            raise ConfigurationException(
                "OPENAI_API_KEY must be set in environment or .env file"
            )

        # Initialize LangChain model with structured output
        base_model = ChatOpenAI(
            model=processing_config.categorization_model,
            api_key=api_key
        )

        # Bind structured output schema
        self.model = base_model.with_structured_output(CommentClassification)

    def _build_comment_context(
        self,
        comment_text: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """Build context string from comment data.

        Args:
            comment_text: The comment text
            first_name: Commenter's first name
            last_name: Commenter's last name
            organization: Commenter's organization

        Returns:
            Formatted context string
        """
        parts = []

        if organization:
            parts.append(f"Organization: {organization}")

        if first_name or last_name:
            name = f"{first_name or ''} {last_name or ''}".strip()
            if name:
                parts.append(f"Name: {name}")

        parts.append(f"Comment: {comment_text}")

        return "\n".join(parts)

    @traceable(name="categorize_comment")
    async def categorize(
        self,
        comment_text: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> CommentClassification:
        """Categorize a single comment asynchronously.

        Args:
            comment_text: The comment text
            first_name: Commenter's first name
            last_name: Commenter's last name
            organization: Commenter's organization

        Returns:
            CommentClassification with category and sentiment

        Raises:
            Exception: If categorization fails after retries
        """
        context = self._build_comment_context(
            comment_text, first_name, last_name, organization
        )

        try:
            result = await self.model.ainvoke(
                f"{self.SYSTEM_PROMPT}\n\n{context}"
            )
            return result
        except Exception as e:
            logger.error(f"Error categorizing comment: {e}")
            # Return default classification on error
            from ..models import Topic
            return CommentClassification(
                category=Category.ANONYMOUS_NOT_SPECIFIED,
                sentiment=Sentiment.UNCLEAR,
                topics=[Topic.UNCLEAR],
                reasoning=f"Error during classification: {str(e)[:100]}"
            )

    @traceable(name="categorize_batch")
    async def categorize_batch(
        self,
        comments: list[Dict],
        batch_size: int = 10,
    ) -> list[CommentClassification]:
        """Categorize multiple comments in batches.

        Args:
            comments: List of comment dicts with 'comment_text' and optional metadata
            batch_size: Number of comments to process in parallel

        Returns:
            List of CommentClassification objects
        """
        results = []

        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]

            # Process batch in parallel
            tasks = [
                self.categorize(
                    comment_text=c.get("comment_text", ""),
                    first_name=c.get("first_name"),
                    last_name=c.get("last_name"),
                    organization=c.get("organization"),
                )
                for c in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch categorization error: {result}")
                    # Add default classification for failed items
                    from ..models import Topic
                    results.append(
                        CommentClassification(
                            category=Category.ANONYMOUS_NOT_SPECIFIED,
                            sentiment=Sentiment.UNCLEAR,
                            topics=[Topic.UNCLEAR],
                            reasoning=f"Error: {str(result)[:100]}"
                        )
                    )
                else:
                    results.append(result)

            # Rate limiting between batches
            if i + batch_size < len(comments):
                await asyncio.sleep(1.0)

        return results
