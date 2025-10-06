Never read .env

# Database Access
All database operations MUST use repository classes in reggie/db/repository.py.
Database connection setup uses reggie/db/connection.py.
Use DocumentRepository for document operations, CommentRepository for comments, CommentChunkRepository for embeddings.
Database schema is in reggie/db/schema.sql (documents, comments, comment_chunks tables with pgvector).

# Configuration & Environment
All env vars and config MUST use config.py classes (APIConfig, DatabaseConfig, EmbeddingConfig, ProcessingConfig).
Never directly access os.environ; use Pydantic config classes.
Database connection string via DatabaseConfig().connection_string.
OpenAI API key required (OPENAI_API_KEY); regulation.gov key optional (REG_API_KEY).

# Models
Use Pydantic models from reggie/models/ for all data structures, rather than dicts.

# Logging
Use logging_config.py setup_logging() and get_logger() for all logging.
Never use print() except in CLI display; use logger instead.

# Code Organization
Reuse existing code when possible
Separation of concerns is paramount.  We want a clean, modular architecture where each file is concerned with one thing, and we use classes as appropriate to keep our code clean and easy to understand.

# Comments
We prefer using intelligent and descriptive variable names to overreliance on comments.  But comments can be included for sophisticated code blocks that might be hard to understand at a glance.

# Libraries
Don't reinvent the wheel.  Use the most intelligent and appropriate library for every task.