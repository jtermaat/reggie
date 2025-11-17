"""Shared pytest fixtures for all tests"""

import os
import sqlite3
import tempfile
from typing import Generator, Dict, Any
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Save original env
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-testing"

    yield

    # Restore original env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_db() -> Generator[sqlite3.Connection, None, None]:
    """Provides a clean test database for each test.

    Creates a temporary SQLite database, initializes schema, yields connection,
    then deletes the database after the test.
    """
    from reggie.db.connection import _init_db_schema, load_sqlite_vec_extension

    # Create temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    db_path = temp_db.name

    try:
        # Create connection
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Try to load sqlite-vec extension (may not be available on all platforms)
        try:
            load_sqlite_vec_extension(conn)
        except RuntimeError as e:
            # Extension loading not supported on this platform
            # Tests that don't use vector search will still work
            import warnings
            warnings.warn(f"sqlite-vec extension not loaded: {e}", UserWarning)

        # Initialize schema
        _init_db_schema(conn)

        yield conn
    finally:
        # Clean up
        conn.close()
        # Delete temporary database file
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.fixture
def mock_regulations_api(httpx_mock):
    """Mocks regulations.gov API - ZERO external calls.

    Provides helper functions to configure mock responses.
    """
    class MockAPI:
        def __init__(self, httpx_mock):
            self.httpx_mock = httpx_mock
            self.base_url = "https://api.regulations.gov/v4"

        def mock_document_response(self, doc_id: str, title: str = "Test Document"):
            """Mock document metadata response."""
            self.httpx_mock.add_response(
                url=f"{self.base_url}/documents/{doc_id}",
                json={
                    "data": {
                        "id": doc_id,
                        "attributes": {
                            "objectId": f"{doc_id}-OBJ",
                            "title": title,
                            "docketId": "TEST-DOCKET-001",
                            "documentType": "Rule",
                            "postedDate": "2024-01-01T00:00:00Z",
                        }
                    }
                }
            )

        def mock_comments_page(
            self,
            object_id: str,
            page_number: int = 1,
            comments: list = None,
            has_next: bool = False,
            total_elements: int = None
        ):
            """Mock a page of comments."""
            if comments is None:
                comments = []

            if total_elements is None:
                total_elements = len(comments)

            response_data = {
                "data": comments,
                "meta": {
                    "hasNextPage": has_next,
                    "totalElements": total_elements,
                    "pageNumber": page_number
                }
            }

            # Match the actual API query parameters
            self.httpx_mock.add_response(
                url=f"{self.base_url}/comments",
                json=response_data,
                match_params={
                    "filter[commentOnId]": object_id,
                    "page[size]": "250",
                    "page[number]": str(page_number)
                }
            )

        def mock_comment_details(self, comment_id: str, comment_text: str = "Test comment"):
            """Mock comment details response."""
            self.httpx_mock.add_response(
                url=f"{self.base_url}/comments/{comment_id}",
                json={
                    "data": {
                        "id": comment_id,
                        "attributes": {
                            "comment": comment_text,
                            "firstName": "John",
                            "lastName": "Doe",
                            "organization": "Test Org",
                            "postedDate": "2024-01-01T00:00:00Z",
                            "lastModifiedDate": "2024-01-01T00:00:00Z"
                        }
                    }
                }
            )

    return MockAPI(httpx_mock)


@pytest.fixture
def mock_openai(mocker):
    """Mocks OpenAI API - ZERO external calls.

    Mocks both ChatOpenAI and OpenAIEmbeddings.
    """
    from reggie.models import CommentClassification, Category, Sentiment

    # Mock ChatOpenAI for categorization
    mock_classification = CommentClassification(
        category=Category.PHYSICIANS_SURGEONS,
        sentiment=Sentiment.FOR,
        reasoning="Test classification"
    )

    mock_chat = mocker.patch("langchain_openai.ChatOpenAI")
    mock_chat_instance = mock_chat.return_value

    # Create mock that works with both sync and async
    if hasattr(mocker, 'AsyncMock'):
        mock_chat_instance.with_structured_output.return_value.ainvoke = mocker.AsyncMock(
            return_value=mock_classification
        )
        mock_chat_instance.with_structured_output.return_value.invoke = mocker.Mock(
            return_value=mock_classification
        )
    else:
        # Fallback for older pytest-mock versions
        mock_chat_instance.with_structured_output.return_value.ainvoke = mocker.Mock(
            return_value=mock_classification
        )
        mock_chat_instance.with_structured_output.return_value.invoke = mocker.Mock(
            return_value=mock_classification
        )

    # Mock OpenAIEmbeddings for embeddings
    mock_embeddings = mocker.patch("langchain_openai.OpenAIEmbeddings")
    mock_embeddings_instance = mock_embeddings.return_value

    # Support both sync and async embeddings
    if hasattr(mocker, 'AsyncMock'):
        mock_embeddings_instance.aembed_documents = mocker.AsyncMock(
            return_value=[[0.1] * 1536]  # Mock embedding vector
        )
        mock_embeddings_instance.embed_documents = mocker.Mock(
            return_value=[[0.1] * 1536]
        )
    else:
        mock_embeddings_instance.aembed_documents = mocker.Mock(
            return_value=[[0.1] * 1536]
        )
        mock_embeddings_instance.embed_documents = mocker.Mock(
            return_value=[[0.1] * 1536]
        )

    return {
        "chat": mock_chat_instance,
        "embeddings": mock_embeddings_instance,
        "classification": mock_classification
    }


@pytest.fixture
def sample_comment_data() -> Dict[str, Any]:
    """Realistic comment data for testing."""
    return {
        "id": "TEST-COMMENT-001",
        "attributes": {
            "comment": "I support this regulation as a practicing physician.",
            "firstName": "Jane",
            "lastName": "Smith",
            "organization": "Medical Association",
            "postedDate": "2024-01-15T10:30:00Z",
            "lastModifiedDate": "2024-01-15T10:30:00Z"
        }
    }


@pytest.fixture
def sample_document_data() -> Dict[str, Any]:
    """Realistic document data for testing."""
    return {
        "id": "CMS-2024-0001-0001",
        "attributes": {
            "objectId": "0900006484c8e5f0",
            "title": "Medicare Physician Fee Schedule Test Document",
            "docketId": "CMS-2024-0001",
            "documentType": "Proposed Rule",
            "postedDate": "2024-01-01T00:00:00Z"
        }
    }
