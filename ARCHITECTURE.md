# Reggie Architecture

This document provides a technical overview of Reggie's architecture and design decisions.

## High-Level Architecture

Reggie follows a pipeline architecture where data flows through several stages:

```
Regulations.gov API
        ↓
    API Client (async)
        ↓
    Document Loader (orchestrator)
        ├→ Categorizer (LangChain + GPT-5-nano)
        └→ Embedder (LangChain + OpenAI embeddings)
        ↓
    PostgreSQL + pgvector
```

## Core Components

### 1. API Client (`reggie/api/client.py`)

**Purpose**: Async interface to Regulations.gov API v4

**Key Features**:
- Async HTTP client using `httpx`
- Automatic retry with exponential backoff (via `tenacity`)
- Rate limiting between requests
- Batch fetching of comment details

**Design Decisions**:
- Uses async generators for memory-efficient streaming of large comment sets
- Implements context manager protocol for proper resource cleanup
- Separates listing comments from fetching full comment details

### 2. Database Layer (`reggie/db/`)

**Purpose**: PostgreSQL schema and connection management

**Schema Design**:
```sql
documents (metadata)
    ↓ 1:N
comments (full text + classification)
    ↓ 1:N
comment_chunks (text chunks + embeddings)
```

**Key Features**:
- pgvector extension for vector similarity search
- JSONB for flexible metadata storage
- Indexes on category, sentiment, and embeddings
- Automatic timestamp tracking

**Design Decisions**:
- Normalized schema to avoid data duplication
- Chunks reference parent comments for context retrieval
- IVFFlat index on embeddings for fast similarity search
- Upsert logic (ON CONFLICT) for idempotent loading

### 3. Categorizer Pipeline (`reggie/pipeline/categorizer.py`)

**Purpose**: Classify comments by category and sentiment using LangChain

**Key Features**:
- Pydantic models for structured output
- LangChain's `with_structured_output()` for type-safe responses
- Async batch processing with configurable concurrency
- Graceful error handling with default classifications

**Design Decisions**:
- Uses Enum types for categories and sentiments (type safety)
- Includes reasoning field in output for transparency
- Falls back to "unclear" classifications on errors
- Batches requests to respect rate limits

**LangChain Integration**:
```python
base_model = ChatOpenAI(model="gpt-5-nano")
model = base_model.with_structured_output(CommentClassification)
result = await model.ainvoke(prompt)
```

### 4. Embedder Pipeline (`reggie/pipeline/embedder.py`)

**Purpose**: Chunk text and generate embeddings using LangChain

**Key Features**:
- Token-aware text splitting (RecursiveCharacterTextSplitter)
- Tiktoken for accurate token counting
- Async batch embedding with OpenAI
- 1000-token chunks with 200-token overlap

**Design Decisions**:
- Chunk size chosen to balance context vs. specificity
- Overlap preserves context across chunk boundaries
- Uses LangChain's `aembed_documents()` for async processing
- Batches embeddings to respect rate limits

**Chunking Strategy**:
```
Comment: [---1000 tokens---][---1000 tokens---]
Chunks:  [----chunk 1----]
             [----chunk 2----]
         (200-token overlap)
```

### 5. Document Loader (`reggie/pipeline/loader.py`)

**Purpose**: Orchestrate the entire loading pipeline

**Process Flow**:
1. Fetch document metadata from API
2. Stream all comment details from API
3. Categorize comments in batches
4. Chunk and embed comments in batches
5. Store everything in PostgreSQL transactionally

**Key Features**:
- Single async transaction for data consistency
- Parallel processing where possible
- Progress logging every 10 comments
- Comprehensive error tracking

**Design Decisions**:
- Orchestrates API, categorization, and embedding in sequence
- Uses database transaction for atomicity
- Returns detailed statistics for monitoring
- Closes resources properly (API client, DB connection)

### 6. CLI Interface (`reggie/cli/main.py`)

**Purpose**: User-friendly command-line interface

**Key Features**:
- Click for command parsing
- Rich for beautiful output (tables, progress, colors)
- Environment variable validation
- Helpful error messages

**Commands**:
- `init`: Initialize database
- `load`: Load a document
- `list`: Display loaded documents
- `discuss`: Interactive mode (coming soon)

## Async Architecture

Reggie is async-first for performance:

### Why Async?

1. **I/O Bound**: Most time spent waiting for API/database responses
2. **Parallelism**: Process multiple comments simultaneously
3. **Efficiency**: Single thread can handle many concurrent operations

### Async Patterns Used

**API Fetching**:
```python
async for comment in api_client.get_all_comment_details(object_id):
    # Process comment
```

**Batch Processing**:
```python
tasks = [categorize(comment) for comment in batch]
results = await asyncio.gather(*tasks)
```

**Database Operations**:
```python
async with await psycopg.AsyncConnection.connect(conn_str) as conn:
    async with conn.cursor() as cur:
        await cur.execute(sql, params)
```

## Error Handling

### Retry Strategy

- **API Calls**: Exponential backoff with tenacity
- **Rate Limiting**: Automatic retry up to 5 attempts
- **Backoff Formula**: wait = 2^attempt (max 60s)

### Graceful Degradation

- **Categorization Failures**: Default to "unclear" sentiment / "anonymous" category
- **Embedding Failures**: Zero vector (logged as error)
- **Comment Processing Errors**: Skip comment, increment error counter, continue

### Logging

- **Level**: INFO for normal operations
- **Format**: Timestamp, module, level, message
- **Progress**: Every 10 comments during batch processing

## LangSmith Integration

### Tracing

All LangChain operations are decorated with `@traceable`:
- `categorize_comment`: Individual comment classification
- `categorize_batch`: Batch categorization
- `chunk_text`: Text chunking
- `embed_chunks`: Embedding generation
- `load_document`: Full document loading

### Configuration

```python
# In reggie/config.py
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "reggie"
```

### Benefits

- Monitor token usage
- Debug classification accuracy
- Optimize batch sizes
- Track performance over time

## Performance Optimizations

### 1. Batch Processing

- **Categorization**: Process 10 comments in parallel
- **Embedding**: Embed 100 chunks per API call
- **API Fetching**: Fetch comment details in batches of 10

### 2. Connection Pooling

- Reuse HTTP connections with httpx
- Single database connection per load operation

### 3. Async Generators

- Stream comments instead of loading all into memory
- Process chunks as they're generated

### 4. Database Indexes

- B-tree indexes on category, sentiment
- Composite index on (category, sentiment)
- IVFFlat index on embeddings for similarity search

## Future Enhancements

### Part 2: RAG Agent

The discuss command will use:
- LangGraph for agent state management
- Tools for querying database:
  - `count_by_category_sentiment`
  - `semantic_search`
  - `get_example_comments`
- LangChain's PGVector for vector search
- Conversational memory for multi-turn dialogue

### Potential Optimizations

- Connection pooling for concurrent loads
- Background workers for long-running jobs
- Caching for frequently accessed documents
- Streaming responses in discuss mode

## Design Principles

1. **Async-First**: Everything that does I/O is async
2. **Type Safety**: Pydantic models and type hints everywhere
3. **Fail Gracefully**: Errors are logged, not fatal
4. **Observable**: LangSmith tracing and comprehensive logging
5. **Idempotent**: Re-running load updates existing data
6. **Modular**: Clear separation of concerns
7. **Production-Ready**: Error handling, retries, rate limiting
8. **User-Friendly**: Rich CLI with progress indicators

## Technology Choices

| Technology | Why |
|------------|-----|
| LangChain | Structured output, async support, ecosystem |
| GPT-5-nano | Very cheap for high-volume categorization |
| text-embedding-3-small | Good quality, 1536-dim, cost-effective |
| PostgreSQL | Robust, JSONB support, wide ecosystem |
| pgvector | Native vector support, good performance |
| httpx | Modern async HTTP client |
| Click | Industry standard for CLIs |
| Rich | Beautiful terminal output |
| asyncio | Native Python async support |
| Pydantic | Runtime type checking and validation |

## Testing Strategy (Future)

1. **Unit Tests**: Individual pipeline components
2. **Integration Tests**: Database operations
3. **E2E Tests**: Full load → query workflow
4. **LangSmith Datasets**: Evaluation datasets for classification
5. **Mock API**: Test without hitting regulations.gov
