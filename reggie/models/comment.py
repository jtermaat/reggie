"""Comment data models"""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    """Comment sentiment categories."""
    FOR = "for"
    AGAINST = "against"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class Topic(str, Enum):
    """Comment topic categories."""
    REIMBURSEMENT_PAYMENT = "reimbursement_payment"
    COST_FINANCIAL = "cost_financial"
    SERVICE_COVERAGE = "service_coverage"
    ACCESS_TO_CARE = "access_to_care"
    WORKFORCE_STAFFING = "workforce_staffing"
    METHODOLOGY_MEASUREMENT = "methodology_measurement"
    IMPLEMENTATION_FEASIBILITY = "implementation_feasibility"
    ADMINISTRATIVE_BURDEN = "administrative_burden"
    TELEHEALTH_DIGITAL = "telehealth_digital"
    HEALTH_EQUITY = "health_equity"
    QUALITY_PROGRAMS = "quality_programs"
    LEGAL_CLARITY = "legal_clarity"
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


class Comment(BaseModel):
    """Comment model."""

    id: str
    document_id: str
    comment_text: Optional[str] = None
    category: Optional[str] = None
    sentiment: Optional[str] = None
    topics: Optional[List[str]] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization: Optional[str] = None
    posted_date: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class CommentData(BaseModel):
    """Simple comment data for processing."""

    id: str
    comment_text: str = ""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization: Optional[str] = None

    @classmethod
    def from_db_row(cls, row: tuple) -> "CommentData":
        """Create CommentData from database row.

        Args:
            row: Database row (id, comment_text, first_name, last_name, organization)

        Returns:
            CommentData instance
        """
        return cls(
            id=row[0],
            comment_text=row[1] or "",
            first_name=row[2],
            last_name=row[3],
            organization=row[4],
        )


class CommentClassification(BaseModel):
    """Structured output for comment classification."""

    category: Category = Field(
        description="The category of the commenter based on their role and affiliation"
    )
    sentiment: Sentiment = Field(
        description="The overall sentiment of the comment toward the regulation"
    )
    topics: List[Topic] = Field(
        description="The topics discussed in the comment (can be multiple topics)"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification decisions"
    )
