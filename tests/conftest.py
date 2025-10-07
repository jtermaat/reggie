"""Shared pytest fixtures for all tests"""

import os
import asyncio
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

import pytest
import psycopg
from psycopg import AsyncConnection


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Save original env
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-testing"
    os.environ["POSTGRES_DB"] = "reggie_test"

    yield

    # Restore original env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db() -> AsyncGenerator[AsyncConnection, None]:
    """Provides a clean test database for each test.

    Creates a test database, initializes schema, yields connection,
    then drops all tables after the test.
    """
    from reggie.config import get_config
    from reggie.db import init_db

    # Use test database
    config = get_config()
    conn_string = config.connection_string

    # Initialize database schema
    await init_db(conn_string)

    # Create connection for test
    conn = await psycopg.AsyncConnection.connect(conn_string)

    try:
        yield conn
    finally:
        # Clean up: Drop all tables
        async with conn.cursor() as cur:
            await cur.execute("""
                DROP TABLE IF EXISTS comment_chunks CASCADE;
                DROP TABLE IF EXISTS comments CASCADE;
                DROP TABLE IF EXISTS documents CASCADE;
                DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;
            """)
        await conn.commit()
        await conn.close()


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
    mock_chat_instance.with_structured_output.return_value.ainvoke = mocker.AsyncMock(
        return_value=mock_classification
    )

    # Mock OpenAIEmbeddings for embeddings
    mock_embeddings = mocker.patch("langchain_openai.OpenAIEmbeddings")
    mock_embeddings_instance = mock_embeddings.return_value
    mock_embeddings_instance.aembed_documents = mocker.AsyncMock(
        return_value=[[0.1] * 1536]  # Mock embedding vector
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
