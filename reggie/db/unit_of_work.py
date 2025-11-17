"""Unit of Work pattern for managing database transactions and repository instances."""

import sqlite3
from typing import Optional

from .connection import get_db_path, load_sqlite_vec_extension, ensure_db_directory
from .exceptions import DatabaseConnectionError


class UnitOfWork:
    """
    Unit of Work pattern for managing database transactions.

    Provides automatic transaction management and repository access.
    All repositories share the same connection and transaction scope.

    Usage:
        with UnitOfWork() as uow:
            uow.documents.store_document(doc_data)
            uow.comments.store_comment(comment_data, doc_id)
            # Auto-commit on success, rollback on exception
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Unit of Work.

        Args:
            db_path: Path to SQLite database file. If None, will be read from config.
        """
        self.db_path = db_path or get_db_path()
        self._conn: Optional[sqlite3.Connection] = None
        self._documents = None
        self._comments = None
        self._comment_statistics = None
        self._comment_analytics = None
        self._chunks = None

    def __enter__(self):
        """Enter context manager and initialize connection."""
        try:
            # Ensure database directory exists
            ensure_db_directory()

            # Connect to database
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row  # Enable column access by name

            # Load sqlite-vec extension
            load_sqlite_vec_extension(self._conn)

            return self
        except Exception as e:
            if self._conn:
                self._conn.close()
            raise DatabaseConnectionError(f"Failed to establish database connection: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and handle transaction commit/rollback."""
        if self._conn:
            try:
                if exc_type is None:
                    # No exception - commit transaction
                    self._conn.commit()
                else:
                    # Exception occurred - rollback transaction
                    self._conn.rollback()
            finally:
                self._conn.close()
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

    def commit(self):
        """Manually commit the current transaction."""
        if self._conn:
            self._conn.commit()

    def rollback(self):
        """Manually rollback the current transaction."""
        if self._conn:
            self._conn.rollback()
