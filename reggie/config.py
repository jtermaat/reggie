"""Configuration and environment setup for Reggie"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class ReggieConfig(BaseSettings):
    """Unified configuration for reggie application."""

    # Database
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="reggie")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="")

    # API - Regulations.gov
    reg_api_key: str = Field(default="DEMO_KEY")
    reg_api_base_url: str = Field(default="https://api.regulations.gov/v4")
    reg_api_request_delay: float = Field(default=4.0)

    # API - OpenAI
    openai_api_key: str = Field()  # Required - no default

    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Processing
    categorization_model: str = Field(default="gpt-5-nano")
    default_batch_size: int = Field(default=10)
    commit_every: int = Field(default=10)

    # Agent
    discussion_model: str = Field(default="gpt-5-mini")
    rag_model: str = Field(default="gpt-5-mini")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4000)
    embeddings_model: str = Field(default="text-embedding-3-small")
    max_rag_iterations: int = Field(default=3)

    # LangSmith
    langsmith_enabled: bool = Field(default=False)
    langsmith_project: str = Field(default="reggie")
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: Optional[str] = Field(default=None)

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

    def apply_langsmith(self, enable_tracing: bool = False) -> None:
        """Apply LangSmith configuration to environment if enabled.

        Args:
            enable_tracing: Whether to enable LangSmith tracing. If False, tracing is disabled.
        """
        if enable_tracing and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        else:
            # Ensure tracing is disabled by unsetting the environment variable
            os.environ.pop("LANGCHAIN_TRACING_V2", None)


# Singleton pattern for configuration
_config: Optional[ReggieConfig] = None


def get_config() -> ReggieConfig:
    """Get application configuration (singleton).

    Returns:
        ReggieConfig instance
    """
    global _config
    if _config is None:
        _config = ReggieConfig()
    return _config
