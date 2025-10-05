"""Async client for Regulations.gov API"""

from typing import Dict, List, Optional, AsyncIterator
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx

from ..config import APIConfig


class RegulationsAPIClient:
    """Async client for interacting with Regulations.gov API v4."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the API client.

        Args:
            api_key: API key for regulations.gov. If None, reads from config/env.
        """
        config = APIConfig()
        self.api_key = api_key or config.reg_api_key
        self.base_url = config.reg_api_base_url
        self.request_delay = config.reg_api_request_delay
        self.client = httpx.AsyncClient(
            headers={"X-Api-Key": self.api_key},
            timeout=30.0,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a GET request to the API with retry logic and rate limiting.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            httpx.HTTPStatusError: On HTTP errors after retries
        """
        # Simple rate limiting: wait between requests based on config
        # Default 4s ensures we never exceed 1000 requests/hour (4s = 900 req/hr)
        await asyncio.sleep(self.request_delay)

        url = f"{self.base_url}/{endpoint}"
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def get_document(self, document_id: str) -> Dict:
        """Get document metadata by document ID.

        Args:
            document_id: Document ID (e.g., "CMS-2025-0304-0009")

        Returns:
            Document data
        """
        data = await self._get(f"documents/{document_id}")
        return data.get("data", {})

    async def get_comments_page(
        self,
        object_id: str,
        page_number: int = 1,
        page_size: int = 250,
        sort: str = None,
        last_modified_date_ge: str = None,
    ) -> Dict:
        """Get a single page of comments for a document.

        Args:
            object_id: Document object ID
            page_number: Page number (1-indexed)
            page_size: Number of results per page (max 250)
            sort: Sort order (e.g., "lastModifiedDate,documentId")
            last_modified_date_ge: Filter for lastModifiedDate >= this value (format: "YYYY-MM-DD HH:mm:ss")

        Returns:
            Response with comments data and metadata
        """
        params = {
            "filter[commentOnId]": object_id,
            "page[size]": min(page_size, 250),
            "page[number]": page_number,
        }

        if sort:
            params["sort"] = sort

        if last_modified_date_ge:
            params["filter[lastModifiedDate][ge]"] = last_modified_date_ge

        return await self._get("comments", params=params)

    async def get_all_comments(
        self, object_id: str, page_size: int = 250
    ) -> AsyncIterator[Dict]:
        """Asynchronously iterate over all comments for a document.

        The API limits page numbers to 20 max. To fetch >5000 comments, this method
        uses date-based windowing following the official regulations.gov approach:
        - Sort by lastModifiedDate,documentId
        - When hitting page 20, use the last comment's lastModifiedDate
        - Continue with filter[lastModifiedDate][ge] for next batch
        - Repeat until all comments retrieved

        Args:
            object_id: Document object ID
            page_size: Number of results per page (max 250)

        Yields:
            Individual comment data dicts
        """
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)
        MAX_PAGE_NUMBER = 20  # API limitation

        # Use lastModifiedDate for windowing per regulations.gov official docs
        sort_order = "lastModifiedDate,documentId"
        last_modified_filter = None
        total_fetched = 0

        while True:
            page_number = 1

            # Fetch up to 20 pages in this window
            while page_number <= MAX_PAGE_NUMBER:
                response = await self.get_comments_page(
                    object_id,
                    page_number,
                    page_size,
                    sort=sort_order,
                    last_modified_date_ge=last_modified_filter,
                )

                comments = response.get("data", [])
                if not comments:
                    # No more comments in this window
                    return

                for comment in comments:
                    yield comment
                    total_fetched += 1

                # Check if there are more pages in this window
                meta = response.get("meta", {})
                total_elements = meta.get("totalElements", 0)
                has_next_page = meta.get("hasNextPage", False)

                if not has_next_page:
                    # We've reached the end of all comments
                    logger.info(f"Fetched all {total_fetched} comments for document")
                    return

                # If we're on page 20 and there are more pages, we need to move to next window
                if page_number >= MAX_PAGE_NUMBER:
                    # Get the lastModifiedDate of the last comment on this page
                    last_comment = comments[-1]
                    last_modified_iso = last_comment.get("attributes", {}).get("lastModifiedDate")

                    if not last_modified_iso:
                        logger.warning("Last comment missing lastModifiedDate, cannot continue windowing")
                        return

                    # Convert ISO format to "YYYY-MM-DD HH:mm:ss" (Eastern time as per docs)
                    # The API expects Eastern time format
                    try:
                        dt = datetime.fromisoformat(last_modified_iso.replace("Z", "+00:00"))
                        last_modified_filter = dt.strftime("%Y-%m-%d %H:%M:%S")
                        logger.info(
                            f"Reached page limit. Fetched {total_fetched} comments so far. "
                            f"Continuing with filter[lastModifiedDate][ge]={last_modified_filter}"
                        )
                        # Break inner loop to start a new window
                        break
                    except Exception as e:
                        logger.error(f"Error parsing lastModifiedDate: {e}")
                        return

                page_number += 1

    async def get_comment_details(self, comment_id: str) -> Dict:
        """Get detailed information for a specific comment.

        Args:
            comment_id: Comment ID

        Returns:
            Detailed comment data
        """
        data = await self._get(f"comments/{comment_id}")
        return data.get("data", {})

    async def get_all_comment_details(
        self, object_id: str
    ) -> AsyncIterator[Dict]:
        """Get detailed information for all comments on a document.

        Fetches comments sequentially with rate limiting (4 seconds between requests).

        Args:
            object_id: Document object ID

        Yields:
            Detailed comment data dicts
        """
        async for comment in self.get_all_comments(object_id):
            try:
                comment_id = comment["id"]

                # Fetch details one at a time (rate limited by _get method)
                detail = await self.get_comment_details(comment_id)
                yield detail
            except Exception as e:
                # Log error but continue
                print(f"Error fetching comment detail: {e}")
                continue
