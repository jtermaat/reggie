"""Configuration and environment setup for Reggie"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """Configuration for external APIs.

    All settings can be overridden via environment variables.
    For example: REG_API_KEY, OPENAI_API_KEY, etc.
    """

    # Regulations.gov API
    reg_api_key: str = "DEMO_KEY"
    reg_api_base_url: str = "https://api.regulations.gov/v4"
    reg_api_request_delay: float = 4.0  # seconds between requests

    # OpenAI API
    openai_api_key: str  # Required - no default

    # LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "reggie"
    langsmith_tracing: bool = False

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow reading from environment variables
        env_prefix = ""
        extra = "ignore"


class DatabaseConfig(BaseSettings):
    """Configuration for database connection."""

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "reggie"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class EmbeddingConfig(BaseSettings):
    """Configuration for embeddings."""

    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


class ProcessingConfig(BaseSettings):
    """Configuration for processing."""

    categorization_model: str = "gpt-5-nano"
    default_batch_size: int = 10
    commit_every: int = 10

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


def setup_langsmith(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
) -> None:
    """Configure LangSmith tracing.

    Args:
        api_key: LangSmith API key. If None, reads from LANGSMITH_API_KEY env var.
        project: LangSmith project name. If None, reads from LANGSMITH_PROJECT env var
                 or defaults to 'reggie'.
    """
    # Set LangSmith API key
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
    elif not os.getenv("LANGSMITH_API_KEY"):
        # LangSmith is optional - just disable tracing if not configured
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    # Set project name
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
    elif not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "reggie"

    # Enable tracing
    os.environ["LANGSMITH_TRACING"] = "true"
