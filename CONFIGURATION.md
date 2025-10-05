# Configuration Guide

This document explains how configuration and API keys are loaded in Reggie.

## Configuration Architecture

All configuration is centralized in `reggie/config.py` using Pydantic Settings, which provides:
- Type-safe configuration
- Automatic loading from environment variables and `.env` files
- Clear defaults and validation

## Configuration Classes

### 1. APIConfig
Manages all external API credentials and settings.

**Environment Variables:**
- `REG_API_KEY` - Regulations.gov API key (default: "DEMO_KEY")
- `REG_API_BASE_URL` - Base URL for Regulations.gov API (default: "https://api.regulations.gov/v4")
- `REG_API_REQUEST_DELAY` - Delay between API requests in seconds (default: 4.0)
- `OPENAI_API_KEY` - OpenAI API key (required for categorization and embeddings)
- `LANGSMITH_API_KEY` - LangSmith API key (optional, for tracing)
- `LANGSMITH_PROJECT` - LangSmith project name (default: "reggie")
- `LANGSMITH_TRACING` - Enable LangSmith tracing (default: false)

**Used by:**
- `RegulationsAPIClient` - loads `REG_API_KEY`, `REG_API_BASE_URL`, `REG_API_REQUEST_DELAY`
- `CommentCategorizer` - loads `OPENAI_API_KEY`
- `CommentEmbedder` - loads `OPENAI_API_KEY`

### 2. DatabaseConfig
Manages PostgreSQL connection settings.

**Environment Variables:**
- `POSTGRES_HOST` - Database host (default: "localhost")
- `POSTGRES_PORT` - Database port (default: 5432)
- `POSTGRES_DB` - Database name (default: "reggie")
- `POSTGRES_USER` - Database user (default: "postgres")
- `POSTGRES_PASSWORD` - Database password (default: "")

**Used by:**
- `get_connection_string()` in `reggie/db/connection.py`

### 3. EmbeddingConfig
Manages embedding and chunking settings.

**Environment Variables:**
- `EMBEDDING_MODEL` - OpenAI embedding model (default: "text-embedding-3-small")
- `EMBEDDING_DIMENSION` - Vector dimension (default: 1536)
- `CHUNK_SIZE` - Text chunk size in tokens (default: 1000)
- `CHUNK_OVERLAP` - Token overlap between chunks (default: 200)

**Used by:**
- `CommentEmbedder` - configures chunking and embedding behavior

### 4. ProcessingConfig
Manages processing pipeline settings.

**Environment Variables:**
- `CATEGORIZATION_MODEL` - OpenAI model for categorization (default: "gpt-5-nano")
- `DEFAULT_BATCH_SIZE` - Number of comments to process in parallel (default: 10)
- `COMMIT_EVERY` - Database commit frequency (default: 10)

**Used by:**
- `CommentCategorizer` - selects the model for classification
- `CommentProcessor` - sets batch sizes

## Loading Order

Configuration is loaded in the following order (later values override earlier ones):

1. **Defaults** - Hardcoded defaults in config classes
2. **.env file** - Values from `.env` file in project root
3. **Environment variables** - System environment variables
4. **Constructor arguments** - Explicit values passed to class constructors

## Example .env File

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here

# Optional - Regulations.gov
REG_API_KEY=your-regulations-gov-key-here

# Optional - Database (if not using defaults)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=reggie
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password

# Optional - LangSmith (for tracing)
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=reggie
LANGSMITH_TRACING=true

# Optional - Logging
LOG_LEVEL=INFO
```

## Key Points

1. **OPENAI_API_KEY is required** - The application will raise a `ConfigurationException` if this is not set
2. **REG_API_KEY is optional** - Defaults to "DEMO_KEY" which has rate limits
3. **All config classes use Pydantic** - Provides validation and type safety
4. **No hardcoded secrets** - All sensitive values come from environment
5. **Clear error messages** - Missing required config raises helpful exceptions

## Usage in Code

```python
from reggie.config import APIConfig, DatabaseConfig

# Config objects auto-load from environment
api_config = APIConfig()
db_config = DatabaseConfig()

# Access configuration values
print(api_config.openai_api_key)
print(db_config.connection_string)
```

## Migration from Old Code

Previously, API keys were loaded ad-hoc with `os.getenv()` scattered throughout:
- `api/client.py` line 30: `os.getenv("REG_API_KEY", "DEMO_KEY")`
- `pipeline/categorizer.py` line 46: `os.getenv("OPENAI_API_KEY")`
- `pipeline/embedder.py` line 37: `os.getenv("OPENAI_API_KEY")`

Now all configuration flows through centralized config classes, making it:
- Easier to test (can inject config objects)
- Easier to validate (Pydantic handles validation)
- Easier to document (all settings in one place)
- Easier to override (consistent pattern everywhere)
