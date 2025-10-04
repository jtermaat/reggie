"""Async client for Regulations.gov API"""

import os
from typing import Dict, List, Optional, AsyncIterator
import asyncio
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx


class RateLimiter:
    """Rate limiter to ensure we don't exceed API limits (1000 requests/hour)."""

    def __init__(self, max_requests: int = 1000, time_window: float = 3600.0):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds (default: 3600 = 1 hour)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []  # List of request timestamps
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made without exceeding rate limit."""
        while True:
            wait_time = None
            async with self._lock:
                now = time.time()

                # Remove timestamps outside the current window
                self.requests = [ts for ts in self.requests if now - ts < self.time_window]

                # If we're at the limit, calculate wait time
                if len(self.requests) >= self.max_requests:
                    oldest = self.requests[0]
                    wait_time = self.time_window - (now - oldest) + 0.1  # Add small buffer
                else:
                    # We can proceed - record this request
                    self.requests.append(time.time())
                    return

            # Wait outside the lock to avoid deadlock
            if wait_time and wait_time > 0:
                await asyncio.sleep(wait_time)


class RegulationsAPIClient:
    """Async client for interacting with Regulations.gov API v4."""

    BASE_URL = "https://api.regulations.gov/v4"

    def __init__(self, api_key: Optional[str] = None, max_requests_per_hour: int = 990):
        """Initialize the API client.

        Args:
            api_key: API key for regulations.gov.
            max_requests_per_hour: Maximum requests per hour (default: 990, under API limit of 1000)
        """
        self.api_key = api_key or os.getenv("REG_API_KEY", "DEMO_KEY")
        self.client = httpx.AsyncClient(
            headers={"X-Api-Key": self.api_key},
            timeout=30.0,
        )
        self.rate_limiter = RateLimiter(max_requests=max_requests_per_hour, time_window=3600.0)

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
        # Wait for rate limiter before making request
        await self.rate_limiter.acquire()

        url = f"{self.BASE_URL}/{endpoint}"
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
        self, object_id: str, page_number: int = 1, page_size: int = 250
    ) -> Dict:
        """Get a single page of comments for a document.

        Args:
            object_id: Document object ID
            page_number: Page number (1-indexed)
            page_size: Number of results per page (max 250)

        Returns:
            Response with comments data and metadata
        """
        params = {
            "filter[commentOnId]": object_id,
            "page[size]": min(page_size, 250),
            "page[number]": page_number,
        }
        return await self._get("comments", params=params)

    async def get_all_comments(
        self, object_id: str, page_size: int = 250
    ) -> AsyncIterator[Dict]:
        """Asynchronously iterate over all comments for a document.

        Args:
            object_id: Document object ID
            page_size: Number of results per page (max 250)

        Yields:
            Individual comment data dicts
        """
        page_number = 1
        while True:
            response = await self.get_comments_page(object_id, page_number, page_size)

            comments = response.get("data", [])
            if not comments:
                break

            for comment in comments:
                yield comment

            # Check if there are more pages
            meta = response.get("meta", {})
            total_elements = meta.get("totalElements", 0)
            total_pages = (total_elements + page_size - 1) // page_size

            if page_number >= total_pages:
                break

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
        self, object_id: str, batch_size: int = 10
    ) -> AsyncIterator[Dict]:
        """Get detailed information for all comments on a document.

        Args:
            object_id: Document object ID
            batch_size: Number of comments to fetch details for in parallel

        Yields:
            Detailed comment data dicts
        """
        batch = []
        async for comment in self.get_all_comments(object_id):
            batch.append(comment)

            if len(batch) >= batch_size:
                # Fetch details for batch in parallel
                tasks = [
                    self.get_comment_details(c["id"])
                    for c in batch
                ]
                details = await asyncio.gather(*tasks, return_exceptions=True)

                for detail in details:
                    if isinstance(detail, Exception):
                        # Log error but continue
                        print(f"Error fetching comment detail: {detail}")
                        continue
                    yield detail

                batch = []

        # Process remaining batch
        if batch:
            tasks = [
                self.get_comment_details(c["id"])
                for c in batch
            ]
            details = await asyncio.gather(*tasks, return_exceptions=True)

            for detail in details:
                if isinstance(detail, Exception):
                    print(f"Error fetching comment detail: {detail}")
                    continue
                yield detail
