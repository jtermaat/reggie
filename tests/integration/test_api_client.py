"""Integration tests for Regulations.gov API client with mocked HTTP responses"""

import pytest
from datetime import datetime

from reggie.api.client import RegulationsAPIClient


@pytest.mark.integration
class TestAPIClientInitialization:
    """Test API client initialization."""

    async def test_client_initialization_with_default_key(self):
        """Client initializes with config/env API key."""
        async with RegulationsAPIClient() as client:
            assert client.api_key is not None
            assert client.base_url == "https://api.regulations.gov/v4"

    async def test_client_initialization_with_custom_key(self):
        """Client initializes with provided API key."""
        async with RegulationsAPIClient(api_key="CUSTOM_KEY") as client:
            assert client.api_key == "CUSTOM_KEY"

    async def test_client_context_manager(self):
        """Client works as async context manager."""
        async with RegulationsAPIClient() as client:
            assert client.client is not None


@pytest.mark.integration
class TestGetDocument:
    """Test fetching document metadata."""

    async def test_get_document_success(self, mock_regulations_api):
        """Fetch document metadata successfully."""
        doc_id = "CMS-2024-0001-0001"
        mock_regulations_api.mock_document_response(doc_id, "Test Document")

        async with RegulationsAPIClient() as client:
            document = await client.get_document(doc_id)

        assert document["id"] == doc_id
        assert document["attributes"]["title"] == "Test Document"
        assert document["attributes"]["objectId"] == f"{doc_id}-OBJ"

    async def test_get_document_extracts_data_field(self, mock_regulations_api):
        """get_document returns the 'data' field from response."""
        doc_id = "TEST-DOC-001"
        mock_regulations_api.mock_document_response(doc_id, "Document Title")

        async with RegulationsAPIClient() as client:
            document = await client.get_document(doc_id)

        # Should return the 'data' object, not the full response
        assert "id" in document
        assert "attributes" in document


@pytest.mark.integration
class TestGetCommentsPage:
    """Test fetching a single page of comments."""

    async def test_get_comments_page_success(self, mock_regulations_api):
        """Fetch single page of comments."""
        object_id = "test-object-id"
        comments_data = [
            {
                "id": "COMMENT-001",
                "attributes": {
                    "comment": "Test comment 1",
                    "firstName": "John",
                    "lastName": "Doe"
                }
            }
        ]

        mock_regulations_api.mock_comments_page(
            object_id,
            page_number=1,
            comments=comments_data,
            has_next=False,
            total_elements=1
        )

        async with RegulationsAPIClient() as client:
            response = await client.get_comments_page(object_id, page_number=1)

        assert "data" in response
        assert "meta" in response
        assert len(response["data"]) == 1
        assert response["data"][0]["id"] == "COMMENT-001"
        assert response["meta"]["hasNextPage"] is False

    async def test_get_comments_page_with_pagination(self, mock_regulations_api):
        """Fetch comments page with pagination parameters."""
        object_id = "test-object-id"

        mock_regulations_api.mock_comments_page(
            object_id,
            page_number=2,
            comments=[{"id": "C2"}],
            has_next=True
        )

        async with RegulationsAPIClient() as client:
            response = await client.get_comments_page(
                object_id,
                page_number=2,
                page_size=250
            )

        assert response["meta"]["hasNextPage"] is True
        assert response["meta"]["pageNumber"] == 2

    async def test_get_comments_page_empty_results(self, mock_regulations_api):
        """Fetch comments page with no results."""
        object_id = "test-object-id"

        mock_regulations_api.mock_comments_page(
            object_id,
            page_number=1,
            comments=[],
            has_next=False,
            total_elements=0
        )

        async with RegulationsAPIClient() as client:
            response = await client.get_comments_page(object_id)

        assert response["data"] == []
        assert response["meta"]["totalElements"] == 0


@pytest.mark.integration
class TestGetAllComments:
    """Test fetching all comments with pagination."""

    async def test_get_all_comments_single_page(self, mock_regulations_api):
        """Fetch all comments when only one page exists."""
        object_id = "test-object-id"
        comments_data = [
            {"id": "C1", "attributes": {"comment": "Comment 1"}},
            {"id": "C2", "attributes": {"comment": "Comment 2"}}
        ]

        mock_regulations_api.mock_comments_page(
            object_id,
            page_number=1,
            comments=comments_data,
            has_next=False
        )

        async with RegulationsAPIClient() as client:
            comments = []
            async for comment in client.get_all_comments(object_id):
                comments.append(comment)

        assert len(comments) == 2
        assert comments[0]["id"] == "C1"
        assert comments[1]["id"] == "C2"

    async def test_get_all_comments_multiple_pages(self, httpx_mock):
        """Fetch all comments across multiple pages."""
        object_id = "test-object-id"
        base_url = "https://api.regulations.gov/v4/comments"

        # Page 1: has next page
        httpx_mock.add_response(
            url=base_url,
            json={
                "data": [{"id": "C1"}],
                "meta": {"hasNextPage": True, "totalElements": 2, "pageNumber": 1}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "1",
                "sort": "lastModifiedDate,documentId"
            }
        )

        # Page 2: no next page
        httpx_mock.add_response(
            url=base_url,
            json={
                "data": [{"id": "C2"}],
                "meta": {"hasNextPage": False, "totalElements": 2, "pageNumber": 2}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "2",
                "sort": "lastModifiedDate,documentId"
            }
        )

        async with RegulationsAPIClient() as client:
            comments = []
            async for comment in client.get_all_comments(object_id):
                comments.append(comment)

        assert len(comments) == 2
        assert comments[0]["id"] == "C1"
        assert comments[1]["id"] == "C2"

    async def test_get_all_comments_empty(self, mock_regulations_api):
        """Fetch all comments when none exist."""
        object_id = "test-object-id"

        mock_regulations_api.mock_comments_page(
            object_id,
            page_number=1,
            comments=[],
            has_next=False
        )

        async with RegulationsAPIClient() as client:
            comments = []
            async for comment in client.get_all_comments(object_id):
                comments.append(comment)

        assert comments == []

    async def test_get_all_comments_handles_page_20_limit(self, httpx_mock):
        """Fetch all comments using date windowing when reaching page 20 limit."""
        object_id = "test-object-id"
        base_url = "https://api.regulations.gov/v4/comments"

        # Page 20: has next page, triggers windowing
        httpx_mock.add_response(
            url=base_url,
            json={
                "data": [{
                    "id": "C20",
                    "attributes": {
                        "lastModifiedDate": "2024-01-15T10:30:00Z"
                    }
                }],
                "meta": {"hasNextPage": True, "totalElements": 6000, "pageNumber": 20}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "20",
                "sort": "lastModifiedDate,documentId"
            }
        )

        # Page 1 of next window with date filter
        httpx_mock.add_response(
            url=base_url,
            json={
                "data": [{"id": "C21"}],
                "meta": {"hasNextPage": False, "totalElements": 1, "pageNumber": 1}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "1",
                "sort": "lastModifiedDate,documentId",
                "filter[lastModifiedDate][ge]": "2024-01-15 10:30:00"
            }
        )

        async with RegulationsAPIClient() as client:
            comments = []
            async for comment in client.get_all_comments(object_id):
                comments.append(comment)

        assert len(comments) == 2
        assert comments[0]["id"] == "C20"
        assert comments[1]["id"] == "C21"


@pytest.mark.integration
class TestGetCommentDetails:
    """Test fetching individual comment details."""

    async def test_get_comment_details_success(self, mock_regulations_api):
        """Fetch comment details successfully."""
        comment_id = "COMMENT-001"
        mock_regulations_api.mock_comment_details(comment_id, "Detailed comment text")

        async with RegulationsAPIClient() as client:
            comment = await client.get_comment_details(comment_id)

        assert comment["id"] == comment_id
        assert comment["attributes"]["comment"] == "Detailed comment text"

    async def test_get_comment_details_extracts_data(self, mock_regulations_api):
        """get_comment_details returns the 'data' field."""
        comment_id = "TEST-001"
        mock_regulations_api.mock_comment_details(comment_id)

        async with RegulationsAPIClient() as client:
            comment = await client.get_comment_details(comment_id)

        # Should return 'data' object, not full response
        assert "id" in comment
        assert "attributes" in comment


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API error handling and retry logic."""

    async def test_api_rate_limiting_delay(self, httpx_mock, mocker):
        """API client respects rate limiting delay."""
        mock_sleep = mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        httpx_mock.add_response(
            url="https://api.regulations.gov/v4/documents/TEST-001",
            json={"data": {"id": "TEST-001", "attributes": {}}}
        )

        async with RegulationsAPIClient() as client:
            await client.get_document("TEST-001")

        # Should sleep for request_delay (4 seconds by default)
        mock_sleep.assert_called()
        call_args = mock_sleep.call_args[0][0]
        assert call_args == 4.0  # Default request delay

    async def test_api_retry_on_http_error(self, httpx_mock, mocker):
        """API client retries on HTTP errors."""
        # Mock sleep to speed up test
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        doc_id = "TEST-001"
        url = f"https://api.regulations.gov/v4/documents/{doc_id}"

        # First call fails with 500, second succeeds
        httpx_mock.add_response(url=url, status_code=500)
        httpx_mock.add_response(
            url=url,
            json={"data": {"id": doc_id, "attributes": {}}}
        )

        async with RegulationsAPIClient() as client:
            # Should retry and succeed
            document = await client.get_document(doc_id)

        assert document["id"] == doc_id

    async def test_api_max_retries_exhausted(self, httpx_mock, mocker):
        """API client raises error after max retries."""
        # Mock sleep to speed up test
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        doc_id = "TEST-001"
        url = f"https://api.regulations.gov/v4/documents/{doc_id}"

        # All retries fail
        for _ in range(10):  # More than max retries (5)
            httpx_mock.add_response(url=url, status_code=500)

        async with RegulationsAPIClient() as client:
            with pytest.raises(Exception):  # Should raise HTTPStatusError
                await client.get_document(doc_id)


@pytest.mark.integration
class TestGetAllCommentDetails:
    """Test fetching details for all comments."""

    async def test_get_all_comment_details_success(self, httpx_mock, mocker):
        """Fetch details for all comments successfully."""
        # Mock sleep to speed up test
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        object_id = "test-object-id"
        base_url = "https://api.regulations.gov/v4"

        # Mock comments list
        httpx_mock.add_response(
            url=f"{base_url}/comments",
            json={
                "data": [
                    {"id": "C1"},
                    {"id": "C2"}
                ],
                "meta": {"hasNextPage": False}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "1"
            }
        )

        # Mock comment details
        httpx_mock.add_response(
            url=f"{base_url}/comments/C1",
            json={"data": {"id": "C1", "attributes": {"comment": "Comment 1"}}}
        )
        httpx_mock.add_response(
            url=f"{base_url}/comments/C2",
            json={"data": {"id": "C2", "attributes": {"comment": "Comment 2"}}}
        )

        async with RegulationsAPIClient() as client:
            details = []
            async for detail in client.get_all_comment_details(object_id):
                details.append(detail)

        assert len(details) == 2
        assert details[0]["id"] == "C1"
        assert details[1]["id"] == "C2"

    async def test_get_all_comment_details_handles_errors(self, httpx_mock, mocker):
        """Fetch details continues on individual comment errors."""
        # Mock sleep to speed up test
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        object_id = "test-object-id"
        base_url = "https://api.regulations.gov/v4"

        # Mock comments list
        httpx_mock.add_response(
            url=f"{base_url}/comments",
            json={
                "data": [
                    {"id": "C1"},
                    {"id": "C2"}
                ],
                "meta": {"hasNextPage": False}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "1"
            }
        )

        # First comment succeeds, second fails
        httpx_mock.add_response(
            url=f"{base_url}/comments/C1",
            json={"data": {"id": "C1", "attributes": {"comment": "Comment 1"}}}
        )
        httpx_mock.add_response(
            url=f"{base_url}/comments/C2",
            status_code=500
        )

        async with RegulationsAPIClient() as client:
            details = []
            async for detail in client.get_all_comment_details(object_id):
                details.append(detail)

        # Should only get C1, C2 error is handled
        assert len(details) == 1
        assert details[0]["id"] == "C1"
