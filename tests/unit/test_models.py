"""Unit tests for Pydantic models"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from reggie.models import (
    Comment,
    CommentData,
    CommentClassification,
    Document,
    DocumentStats,
    Category,
    Sentiment,
    Topic,
    DoctorSpecialization,
    LicensedProfessionalType
)


@pytest.mark.unit
class TestEnums:
    """Test enum constraints."""

    def test_sentiment_enum_values(self):
        """Verify Sentiment enum has correct values."""
        assert Sentiment.FOR == "for"
        assert Sentiment.AGAINST == "against"
        assert Sentiment.MIXED == "mixed"
        assert Sentiment.UNCLEAR == "unclear"

    def test_category_enum_values(self):
        """Verify Category enum has expected values."""
        assert Category.PHYSICIANS_SURGEONS == "Physicians & Surgeons"
        assert Category.ANONYMOUS_NOT_SPECIFIED == "Anonymous / Not Specified"
        # Test a few more
        assert Category.PATIENTS_CAREGIVERS == "Patients & Caregivers"
        assert Category.PHARMA_BIOTECH == "Pharmaceutical & Biotech Companies"


@pytest.mark.unit
class TestCommentModel:
    """Test Comment model validation."""

    def test_comment_creation_with_required_fields(self):
        """Comment can be created with just required fields."""
        comment = Comment(
            id="TEST-001",
            document_id="DOC-001"
        )
        assert comment.id == "TEST-001"
        assert comment.document_id == "DOC-001"
        assert comment.comment_text is None
        assert comment.category is None
        assert comment.sentiment is None

    def test_comment_creation_with_all_fields(self):
        """Comment can be created with all fields."""
        posted_date = datetime(2024, 1, 15, 10, 30)
        comment = Comment(
            id="TEST-001",
            document_id="DOC-001",
            comment_text="Test comment",
            category="Physicians & Surgeons",
            sentiment="for",
            first_name="John",
            last_name="Doe",
            organization="Test Org",
            posted_date=posted_date,
            metadata={"key": "value"}
        )

        assert comment.id == "TEST-001"
        assert comment.document_id == "DOC-001"
        assert comment.comment_text == "Test comment"
        assert comment.category == "Physicians & Surgeons"
        assert comment.sentiment == "for"
        assert comment.first_name == "John"
        assert comment.last_name == "Doe"
        assert comment.organization == "Test Org"
        assert comment.posted_date == posted_date
        assert comment.metadata == {"key": "value"}

    def test_comment_missing_required_fields(self):
        """Comment validation fails without required fields."""
        with pytest.raises(ValidationError) as exc_info:
            Comment(id="TEST-001")  # Missing document_id

        assert "document_id" in str(exc_info.value)

    def test_comment_default_metadata(self):
        """Comment metadata defaults to empty dict."""
        comment = Comment(id="TEST-001", document_id="DOC-001")
        assert comment.metadata == {}


@pytest.mark.unit
class TestCommentDataModel:
    """Test CommentData model validation."""

    def test_comment_data_creation(self):
        """CommentData can be created with required fields."""
        data = CommentData(
            id="TEST-001",
            comment_text="Test comment"
        )
        assert data.id == "TEST-001"
        assert data.comment_text == "Test comment"
        assert data.first_name is None

    def test_comment_data_empty_text_default(self):
        """CommentData has empty string default for comment_text."""
        data = CommentData(id="TEST-001")
        assert data.comment_text == ""

    def test_comment_data_with_optional_fields(self):
        """CommentData accepts optional fields."""
        data = CommentData(
            id="TEST-001",
            comment_text="Test",
            first_name="Jane",
            last_name="Smith",
            organization="Org"
        )
        assert data.first_name == "Jane"
        assert data.last_name == "Smith"
        assert data.organization == "Org"


@pytest.mark.unit
class TestCommentClassificationModel:
    """Test CommentClassification model validation."""

    def test_classification_creation_with_valid_enums(self):
        """Classification can be created with valid enum values."""
        classification = CommentClassification(
            category=Category.PHYSICIANS_SURGEONS,
            sentiment=Sentiment.FOR,
            topics=[Topic.REIMBURSEMENT_PAYMENT, Topic.COST_FINANCIAL],
            reasoning="Clear support from physician"
        )

        assert classification.category == Category.PHYSICIANS_SURGEONS
        assert classification.sentiment == Sentiment.FOR
        assert len(classification.topics) == 2
        assert Topic.REIMBURSEMENT_PAYMENT in classification.topics
        assert classification.reasoning == "Clear support from physician"

    def test_classification_enum_validation(self):
        """Classification validates enum constraints."""
        # Valid enum values should work
        classification = CommentClassification(
            category=Category.PATIENTS_CAREGIVERS,
            sentiment=Sentiment.AGAINST,
            topics=[Topic.ACCESS_TO_CARE],
            reasoning="Test"
        )
        assert classification.category == Category.PATIENTS_CAREGIVERS
        assert classification.sentiment == Sentiment.AGAINST
        assert classification.topics == [Topic.ACCESS_TO_CARE]

    def test_classification_requires_all_fields(self):
        """Classification requires all fields."""
        with pytest.raises(ValidationError) as exc_info:
            CommentClassification(
                category=Category.PHYSICIANS_SURGEONS,
                sentiment=Sentiment.FOR,
                topics=[Topic.UNCLEAR]
                # Missing reasoning
            )
        assert "reasoning" in str(exc_info.value)

    def test_classification_with_doctor_specialization(self):
        """Classification accepts doctor_specialization when category is PHYSICIANS_SURGEONS."""
        classification = CommentClassification(
            category=Category.PHYSICIANS_SURGEONS,
            sentiment=Sentiment.FOR,
            topics=[Topic.REIMBURSEMENT_PAYMENT],
            doctor_specialization=DoctorSpecialization.CARDIOLOGY,
            reasoning="Cardiologist supports reimbursement changes"
        )
        assert classification.category == Category.PHYSICIANS_SURGEONS
        assert classification.doctor_specialization == DoctorSpecialization.CARDIOLOGY
        assert classification.licensed_professional_type is None

    def test_classification_with_licensed_professional_type(self):
        """Classification accepts licensed_professional_type when category is OTHER_LICENSED_CLINICIANS."""
        classification = CommentClassification(
            category=Category.OTHER_LICENSED_CLINICIANS,
            sentiment=Sentiment.AGAINST,
            topics=[Topic.ADMINISTRATIVE_BURDEN],
            licensed_professional_type=LicensedProfessionalType.NURSE_PRACTITIONER,
            reasoning="Nurse practitioner opposes burden"
        )
        assert classification.category == Category.OTHER_LICENSED_CLINICIANS
        assert classification.licensed_professional_type == LicensedProfessionalType.NURSE_PRACTITIONER
        assert classification.doctor_specialization is None

    def test_classification_rejects_doctor_specialization_for_non_physicians(self):
        """Classification rejects doctor_specialization when category is not PHYSICIANS_SURGEONS."""
        with pytest.raises(ValidationError) as exc_info:
            CommentClassification(
                category=Category.PATIENTS_CAREGIVERS,  # Not a physician
                sentiment=Sentiment.FOR,
                topics=[Topic.ACCESS_TO_CARE],
                doctor_specialization=DoctorSpecialization.CARDIOLOGY,  # Should not be allowed
                reasoning="Test"
            )
        assert "doctor_specialization" in str(exc_info.value)
        assert "Physicians & Surgeons" in str(exc_info.value)

    def test_classification_rejects_licensed_professional_type_for_non_clinicians(self):
        """Classification rejects licensed_professional_type when category is not OTHER_LICENSED_CLINICIANS."""
        with pytest.raises(ValidationError) as exc_info:
            CommentClassification(
                category=Category.HOSPITALS_HEALTH_SYSTEMS,  # Not a licensed clinician
                sentiment=Sentiment.FOR,
                topics=[Topic.QUALITY_PROGRAMS],
                licensed_professional_type=LicensedProfessionalType.NURSE_PRACTITIONER,  # Should not be allowed
                reasoning="Test"
            )
        assert "licensed_professional_type" in str(exc_info.value)
        assert "Other Licensed Clinicians" in str(exc_info.value)

    def test_classification_allows_null_subcategories(self):
        """Classification allows null sub-categories regardless of category."""
        # For physicians - null specialization is OK
        classification1 = CommentClassification(
            category=Category.PHYSICIANS_SURGEONS,
            sentiment=Sentiment.FOR,
            topics=[Topic.UNCLEAR],
            doctor_specialization=None,  # Explicitly None
            reasoning="Physician without clear specialization"
        )
        assert classification1.doctor_specialization is None

        # For licensed clinicians - null type is OK
        classification2 = CommentClassification(
            category=Category.OTHER_LICENSED_CLINICIANS,
            sentiment=Sentiment.AGAINST,
            topics=[Topic.UNCLEAR],
            licensed_professional_type=None,  # Explicitly None
            reasoning="Clinician without clear type"
        )
        assert classification2.licensed_professional_type is None

        # For other categories - both should be None (default)
        classification3 = CommentClassification(
            category=Category.INDIVIDUALS_PRIVATE_CITIZENS,
            sentiment=Sentiment.MIXED,
            topics=[Topic.UNCLEAR],
            reasoning="Individual with no professional classification"
        )
        assert classification3.doctor_specialization is None
        assert classification3.licensed_professional_type is None


@pytest.mark.unit
class TestDocumentModel:
    """Test Document model validation."""

    def test_document_creation_with_required_fields(self):
        """Document can be created with just required fields."""
        doc = Document(
            id="DOC-001",
            object_id="OBJ-001"
        )
        assert doc.id == "DOC-001"
        assert doc.object_id == "OBJ-001"
        assert doc.title is None

    def test_document_creation_with_all_fields(self):
        """Document can be created with all fields."""
        posted_date = datetime(2024, 1, 1)
        doc = Document(
            id="DOC-001",
            title="Test Document",
            object_id="OBJ-001",
            docket_id="DOCKET-001",
            document_type="Rule",
            posted_date=posted_date,
            metadata={"attr": "value"}
        )

        assert doc.id == "DOC-001"
        assert doc.title == "Test Document"
        assert doc.object_id == "OBJ-001"
        assert doc.docket_id == "DOCKET-001"
        assert doc.document_type == "Rule"
        assert doc.posted_date == posted_date
        assert doc.metadata == {"attr": "value"}

    def test_document_missing_required_object_id(self):
        """Document validation fails without object_id."""
        with pytest.raises(ValidationError) as exc_info:
            Document(id="DOC-001")  # Missing object_id
        assert "object_id" in str(exc_info.value)

    def test_document_default_metadata(self):
        """Document metadata defaults to empty dict."""
        doc = Document(id="DOC-001", object_id="OBJ-001")
        assert doc.metadata == {}


@pytest.mark.unit
class TestDocumentStatsModel:
    """Test DocumentStats model validation."""

    def test_document_stats_creation(self):
        """DocumentStats can be created with all fields."""
        posted_date = datetime(2024, 1, 1)
        loaded_at = datetime(2024, 1, 15)

        stats = DocumentStats(
            id="DOC-001",
            title="Test Document",
            docket_id="DOCKET-001",
            posted_date=posted_date,
            comment_count=100,
            unique_categories=5,
            loaded_at=loaded_at
        )

        assert stats.id == "DOC-001"
        assert stats.title == "Test Document"
        assert stats.docket_id == "DOCKET-001"
        assert stats.posted_date == posted_date
        assert stats.comment_count == 100
        assert stats.unique_categories == 5
        assert stats.loaded_at == loaded_at

    def test_document_stats_optional_fields(self):
        """DocumentStats allows None for optional fields."""
        stats = DocumentStats(
            id="DOC-001",
            title=None,
            docket_id=None,
            posted_date=None,
            comment_count=0,
            unique_categories=0,
            loaded_at=None
        )

        assert stats.title is None
        assert stats.docket_id is None
        assert stats.posted_date is None
        assert stats.loaded_at is None

    def test_document_stats_requires_counts(self):
        """DocumentStats requires count fields."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentStats(
                id="DOC-001",
                title="Test"
                # Missing comment_count and unique_categories
            )
        errors = str(exc_info.value)
        assert "comment_count" in errors or "unique_categories" in errors
