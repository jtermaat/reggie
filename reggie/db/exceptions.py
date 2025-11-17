"""Custom exceptions for database repository layer."""


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class EntityNotFoundError(RepositoryError):
    """Raised when entity not found in database."""
    pass


class DuplicateEntityError(RepositoryError):
    """Raised when attempting to create duplicate entity."""
    pass


class DatabaseConnectionError(RepositoryError):
    """Raised when database connection fails."""
    pass


class InvalidFilterError(RepositoryError):
    """Raised when invalid filter parameters are provided."""
    pass
