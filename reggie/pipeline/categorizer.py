"""LangChain pipeline for categorizing comments with structured output"""

import os
import asyncio
import logging
from typing import Dict, Optional
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith import traceable

from ..config import setup_langsmith

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    """Comment sentiment categories."""
    FOR = "for"
    AGAINST = "against"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class Category(str, Enum):
    """Commenter categories."""
    PHYSICIANS_SURGEONS = "Physicians & Surgeons"
    OTHER_LICENSED_CLINICIANS = "Other Licensed Clinicians"
    HEALTHCARE_PRACTICE_STAFF = "Healthcare Practice Staff"
    PATIENTS_CAREGIVERS = "Patients & Caregivers"
    PATIENT_ADVOCATES = "Patient/Disability Advocates & Advocacy Organizations"
    PROFESSIONAL_ASSOCIATIONS = "Professional Associations"
    HOSPITALS_HEALTH_SYSTEMS = "Hospitals Health Systems & Networks"
    HEALTHCARE_COMPANIES = "Healthcare Companies & Corporations"
    PHARMA_BIOTECH = "Pharmaceutical & Biotech Companies"
    MEDICAL_DEVICE_DIGITAL_HEALTH = "Medical Device & Digital Health Companies"
    GOVERNMENT_PUBLIC_PROGRAMS = "Government & Public Programs"
    ACADEMIC_RESEARCH = "Academic & Research Institutions"
    NONPROFITS_FOUNDATIONS = "Nonprofits & Foundations"
    INDIVIDUALS_PRIVATE_CITIZENS = "Individuals / Private Citizens"
    ANONYMOUS_NOT_SPECIFIED = "Anonymous / Not Specified"


class CommentClassification(BaseModel):
    """Structured output for comment classification."""
    category: Category = Field(
        description="The category of the commenter based on their role and affiliation"
    )
    sentiment: Sentiment = Field(
        description="The overall sentiment of the comment toward the regulation"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification decisions"
    )


class CommentCategorizer:
    """LangChain-based comment categorizer using structured output."""

    SYSTEM_PROMPT = """You are an expert at analyzing public comments on healthcare regulations.

Your task is to classify each comment by:
1. **Category**: Identify the commenter's role/affiliation
2. **Sentiment**: Determine their position on the regulation

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

Provide your classification along with brief reasoning."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the categorizer.

        Args:
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        # Initialize LangChain model with structured output
        base_model = ChatOpenAI(
            model="gpt-5-nano",
            api_key=api_key,
            temperature=0,
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
            return CommentClassification(
                category=Category.ANONYMOUS_NOT_SPECIFIED,
                sentiment=Sentiment.UNCLEAR,
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
                    results.append(
                        CommentClassification(
                            category=Category.ANONYMOUS_NOT_SPECIFIED,
                            sentiment=Sentiment.UNCLEAR,
                            reasoning=f"Error: {str(result)[:100]}"
                        )
                    )
                else:
                    results.append(result)

            # Rate limiting between batches
            if i + batch_size < len(comments):
                await asyncio.sleep(1.0)

        return results
