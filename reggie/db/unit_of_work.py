"""Unit of Work pattern for managing database transactions and repository instances."""

import psycopg
from typing import Optional

from .connection import get_database_url
from .exceptions import DatabaseConnectionError


class UnitOfWork:
    """
    Unit of Work pattern for managing database transactions.

    Provides automatic transaction management and repository access.
    All repositories share the same connection and transaction scope.

    Usage:
        async with UnitOfWork() as uow:
            await uow.documents.store_document(doc_data)
            await uow.comments.store_comment(comment_data, doc_id)
            # Auto-commit on success, rollback on exception
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize Unit of Work.

        Args:
            database_url: PostgreSQL connection URL. If None, will be read from config.
        """
        self.database_url = database_url or get_database_url()
        self._conn: Optional[psycopg.AsyncConnection] = None
        self._documents = None
        self._comments = None
        self._comment_statistics = None
        self._comment_analytics = None
        self._chunks = None

    async def __aenter__(self):
        """Enter async context manager and initialize connection."""
        try:
            # Connect to database
            self._conn = await psycopg.AsyncConnection.connect(
                self.database_url,
                row_factory=psycopg.rows.dict_row,
                autocommit=False
            )

            return self
        except Exception as e:
            if self._conn:
                await self._conn.close()
            raise DatabaseConnectionError(f"Failed to establish database connection: {e}") from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager and handle transaction commit/rollback."""
        if self._conn:
            try:
                if exc_type is None:
                    # No exception - commit transaction
                    await self._conn.commit()
                else:
                    # Exception occurred - rollback transaction
                    await self._conn.rollback()
            finally:
                await self._conn.close()
                self._conn = None

    @property
    def documents(self):
        """Get DocumentRepository instance (lazy initialization)."""
        if self._documents is None:
            from .repositories.document_repository import DocumentRepository
            self._documents = DocumentRepository(self._conn)
        return self._documents

    @property
    def comments(self):
        """Get CommentRepository instance (lazy initialization)."""
        if self._comments is None:
            from .repositories.comment_repository import CommentRepository
            self._comments = CommentRepository(self._conn)
        return self._comments

    @property
    def comment_statistics(self):
        """Get CommentStatisticsRepository instance (lazy initialization)."""
        if self._comment_statistics is None:
            from .repositories.comment_statistics_repository import CommentStatisticsRepository
            self._comment_statistics = CommentStatisticsRepository(self._conn)
        return self._comment_statistics

    @property
    def comment_analytics(self):
        """Get CommentAnalyticsRepository instance (lazy initialization)."""
        if self._comment_analytics is None:
            from .repositories.comment_analytics_repository import CommentAnalyticsRepository
            self._comment_analytics = CommentAnalyticsRepository(self._conn)
        return self._comment_analytics

    @property
    def chunks(self):
        """Get ChunkRepository instance (lazy initialization)."""
        if self._chunks is None:
            from .repositories.chunk_repository import ChunkRepository
            self._chunks = ChunkRepository(self._conn)
        return self._chunks

    async def commit(self):
        """Manually commit the current transaction."""
        if self._conn:
            await self._conn.commit()

    async def rollback(self):
        """Manually rollback the current transaction."""
        if self._conn:
            await self._conn.rollback()
