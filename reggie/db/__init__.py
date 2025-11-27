"""Database utilities for Reggie"""

from .connection import get_connection, init_db, get_database_url
from .unit_of_work import UnitOfWork
from .repositories import (
    DocumentRepository,
    CommentRepository,
    CommentStatisticsRepository,
    CommentAnalyticsRepository,
    ChunkRepository,
)

# For backwards compatibility during migration
CommentChunkRepository = ChunkRepository

__all__ = [
    "get_connection",
    "init_db",
    "get_database_url",
    "UnitOfWork",
    "DocumentRepository",
    "CommentRepository",
    "CommentStatisticsRepository",
    "CommentAnalyticsRepository",
    "ChunkRepository",
    "CommentChunkRepository",  # Backwards compatibility
]
