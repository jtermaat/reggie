"""Unit tests for CommentCategorizer"""

import pytest
from unittest.mock import AsyncMock

from reggie.pipeline.categorizer import CommentCategorizer
from reggie.models import CommentClassification, Category, Sentiment
from reggie.exceptions import ConfigurationException


@pytest.mark.unit
class TestCategorizerInitialization:
    """Test categorizer initialization."""

    def test_categorizer_requires_api_key(self, mocker):
        """Categorizer raises error without OpenAI API key."""
        # Remove the test API key from environment
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)

        with pytest.raises(ConfigurationException) as exc_info:
            CommentCategorizer()

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_categorizer_initialization_with_api_key(self, mocker):
        """Categorizer initializes successfully with API key."""
        # Mock the LangChain components
        mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")

        # Should not raise
        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        assert categorizer is not None


@pytest.mark.unit
class TestContextBuilding:
    """Test context string building."""

    def test_build_comment_context_with_all_fields(self, mocker):
        """Context includes all provided fields."""
        mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        context = categorizer._build_comment_context(
            comment_text="I support this regulation.",
            first_name="John",
            last_name="Doe",
            organization="Medical Association"
        )

        assert "Organization: Medical Association" in context
        assert "Name: John Doe" in context
        assert "Comment: I support this regulation." in context

    def test_build_comment_context_with_partial_name(self, mocker):
        """Context handles partial names correctly."""
        mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        context = categorizer._build_comment_context(
            comment_text="Test comment",
            first_name="Jane",
            last_name=None
        )

        assert "Name: Jane" in context
        assert "Comment: Test comment" in context

    def test_build_comment_context_with_no_metadata(self, mocker):
        """Context works with just comment text."""
        mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        context = categorizer._build_comment_context(
            comment_text="Just a comment"
        )

        assert "Comment: Just a comment" in context
        assert "Name:" not in context
        assert "Organization:" not in context

    def test_build_comment_context_organization_only(self, mocker):
        """Context includes organization without name."""
        mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        context = categorizer._build_comment_context(
            comment_text="Test",
            organization="Big Pharma Inc"
        )

        assert "Organization: Big Pharma Inc" in context
        assert "Comment: Test" in context


@pytest.mark.unit
class TestCategorization:
    """Test comment categorization logic."""

    async def test_categorize_returns_classification(self, mocker):
        """Categorize returns CommentClassification on success."""
        mock_chat = mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")
        mock_classification = CommentClassification(
            category=Category.PHYSICIANS_SURGEONS,
            sentiment=Sentiment.FOR,
            reasoning="Test reasoning"
        )

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_classification)
        mock_chat.return_value.with_structured_output.return_value = mock_model

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        result = await categorizer.categorize(
            comment_text="I support this as a physician",
            organization="Medical Association"
        )

        assert isinstance(result, CommentClassification)
        assert result.category == Category.PHYSICIANS_SURGEONS
        assert result.sentiment == Sentiment.FOR
        assert result.reasoning == "Test reasoning"

    async def test_categorize_error_returns_default(self, mocker):
        """Categorize returns default classification on error."""
        mock_chat = mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")

        # Mock API failure
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        mock_chat.return_value.with_structured_output.return_value = mock_model

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        result = await categorizer.categorize(
            comment_text="Test comment"
        )

        # Should return default/fallback classification
        assert isinstance(result, CommentClassification)
        assert result.category == Category.ANONYMOUS_NOT_SPECIFIED
        assert result.sentiment == Sentiment.UNCLEAR
        assert "Error" in result.reasoning


@pytest.mark.unit
class TestBatchCategorization:
    """Test batch categorization."""

    async def test_categorize_batch_processes_all_comments(self, mocker):
        """Batch categorization processes all comments."""
        mock_chat = mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")
        mock_classification = CommentClassification(
            category=Category.PHYSICIANS_SURGEONS,
            sentiment=Sentiment.FOR,
            reasoning="Test"
        )

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_classification)
        mock_chat.return_value.with_structured_output.return_value = mock_model

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")

        comments = [
            {"comment_text": "Comment 1", "organization": "Org A"},
            {"comment_text": "Comment 2", "first_name": "John"},
            {"comment_text": "Comment 3"}
        ]

        results = await categorizer.categorize_batch(comments, batch_size=2)

        assert len(results) == 3
        assert all(isinstance(r, CommentClassification) for r in results)

    async def test_categorize_batch_handles_mixed_success_failure(self, mocker):
        """Batch categorization handles mixed success/failure."""
        mock_chat = mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")

        # First call succeeds, second fails, third succeeds
        mock_classification = CommentClassification(
            category=Category.PHYSICIANS_SURGEONS,
            sentiment=Sentiment.FOR,
            reasoning="Success"
        )

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(
            side_effect=[
                mock_classification,
                Exception("API Error"),
                mock_classification
            ]
        )
        mock_chat.return_value.with_structured_output.return_value = mock_model

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")

        comments = [
            {"comment_text": "Comment 1"},
            {"comment_text": "Comment 2"},
            {"comment_text": "Comment 3"}
        ]

        results = await categorizer.categorize_batch(comments, batch_size=10)

        assert len(results) == 3
        # First result should be successful
        assert results[0].reasoning == "Success"
        # Second should be error/default
        assert results[1].category == Category.ANONYMOUS_NOT_SPECIFIED
        assert results[1].sentiment == Sentiment.UNCLEAR
        # Third should be successful
        assert results[2].reasoning == "Success"

    async def test_categorize_batch_empty_list(self, mocker):
        """Batch categorization handles empty list."""
        mock_chat = mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")
        mock_chat.return_value.with_structured_output.return_value = AsyncMock()

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")
        results = await categorizer.categorize_batch([])

        assert results == []

    async def test_categorize_batch_respects_batch_size(self, mocker):
        """Batch categorization respects batch size."""
        mock_chat = mocker.patch("reggie.pipeline.categorizer.ChatOpenAI")
        mock_classification = CommentClassification(
            category=Category.PHYSICIANS_SURGEONS,
            sentiment=Sentiment.FOR,
            reasoning="Test"
        )

        call_count = 0

        async def mock_ainvoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_classification

        mock_model = AsyncMock()
        mock_model.ainvoke = mock_ainvoke
        mock_chat.return_value.with_structured_output.return_value = mock_model

        # Mock asyncio.sleep to speed up test
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)

        categorizer = CommentCategorizer(openai_api_key="sk-test-key")

        # 5 comments with batch_size=2 should process in 3 batches
        comments = [{"comment_text": f"Comment {i}"} for i in range(5)]
        results = await categorizer.categorize_batch(comments, batch_size=2)

        assert len(results) == 5
        assert call_count == 5  # All comments processed
