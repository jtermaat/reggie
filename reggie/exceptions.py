"""Custom exceptions for Reggie"""


class ReggieException(Exception):
    """Base exception for all Reggie errors."""
    pass


class APIException(ReggieException):
    """Exception raised for Regulations.gov API errors."""
    pass


class DatabaseException(ReggieException):
    """Exception raised for database operation errors."""
    pass


class ProcessingException(ReggieException):
    """Exception raised for comment processing errors."""
    pass


class ConfigurationException(ReggieException):
    """Exception raised for configuration errors."""
    pass


class ValidationException(ReggieException):
    """Exception raised for data validation errors."""
    pass


class AgentException(ReggieException):
    """Base exception for agent-related errors."""
    pass


class DocumentNotFoundError(AgentException):
    """Exception raised when a document is not found."""
    pass


class NoProcessedCommentsError(AgentException):
    """Exception raised when a document has no processed comments."""
    pass


class RAGSearchError(AgentException):
    """Exception raised when RAG search fails."""
    pass


class AgentInvocationError(AgentException):
    """Exception raised when agent invocation fails."""
    pass
