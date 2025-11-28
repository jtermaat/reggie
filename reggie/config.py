"""Configuration and environment setup for Reggie"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class ReggieConfig(BaseSettings):
    """Unified configuration for reggie application."""

    # Database
    database_url: str = Field(default="postgresql://reggie:reggie@localhost:5432/reggie")

    # API - Regulations.gov
    reg_api_key: str = Field(default="DEMO_KEY")
    reg_api_base_url: str = Field(default="https://api.regulations.gov/v4")
    reg_api_request_delay: float = Field(default=4.0)
    reg_api_timeout: float = Field(default=30.0)
    reg_api_page_size: int = Field(default=250)
    reg_api_retry_attempts: int = Field(default=5)
    reg_api_retry_wait_min: int = Field(default=2)
    reg_api_retry_wait_max: int = Field(default=60)
    reg_api_retry_wait_multiplier: int = Field(default=1)

    # API - OpenAI
    openai_api_key: str = Field()  # Required - no default

    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)
    chunk_size: int = Field(default=256)
    chunk_overlap: int = Field(default=40)

    # Processing
    categorization_model: str = Field(default="gpt-5-nano")
    default_batch_size: int = Field(default=10)
    commit_every: int = Field(default=10)
    embedding_batch_size: int = Field(default=100)
    embedding_rate_limit_sleep: float = Field(default=1.0)
    categorization_rate_limit_sleep: float = Field(default=1.0)

    # Agent
    discussion_model: str = Field(default="gpt-5-mini")
    rag_model: str = Field(default="gpt-5-mini")
    embeddings_model: str = Field(default="text-embedding-3-small")
    max_rag_iterations: int = Field(default=3)
    vector_search_limit: int = Field(default=10)
    min_comments_threshold: int = Field(default=3)
    chunks_summary_display_limit: int = Field(default=20)
    chunk_preview_chars: int = Field(default=200)
    comment_preview_chars: int = Field(default=500)

    # Hybrid Search
    search_mode: str = Field(default="hybrid")  # "vector", "fts", or "hybrid"
    hybrid_vector_weight: float = Field(default=0.7)
    hybrid_fts_weight: float = Field(default=0.3)
    hybrid_rrf_k: int = Field(default=60)

    # LangSmith
    langsmith_enabled: bool = Field(default=False)
    langsmith_project: str = Field(default="reggie-agent-eval")
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: Optional[str] = Field(default=None)

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

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
