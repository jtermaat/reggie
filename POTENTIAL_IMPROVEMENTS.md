# Potential Improvements for Reggie Codebase

This document outlines areas where the reggie codebase can be improved for better modularity, readability, and adherence to LangChain/LangGraph best practices.

## 1. Prompts Consolidation

### 1.1 Scattered Prompt Strings

**Problem:** Prompt strings are scattered across multiple files:
- `agent/discussion.py` (lines 27-39): System message for discussion agent
- `agent/rag_graph.py` (lines 82-92): Summarization prompt
- `agent/rag_graph.py` (lines 99-113): Generation prompt
- `pipeline/categorizer.py` (lines 35-61): Categorization prompt

**Why it's a problem:**
- Hard to maintain consistency across prompts
- No version control for prompt changes
- Can't easily A/B test different prompts
- Difficult to reuse prompt components
- No type safety for prompt variables

**Recommended fix:**
Create `reggie/prompts.py` using LangChain PromptTemplate:

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Dict

class ReggiePrompts:
    """Centralized prompt management for reggie application."""

    # Agent prompts
    DISCUSSION_SYSTEM = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant helping users explore and analyze public comments on federal regulations.

Your capabilities:
1. Search for relevant comments using semantic search
2. Provide analysis and insights about comment patterns
3. Answer questions about specific comments or document statistics

Guidelines:
- Be precise and cite specific comments when relevant
- If you don't find relevant information, say so clearly
- Use the search_comments tool to find relevant comments before answering
- Provide comment IDs for reference when discussing specific comments"""),
        ("placeholder", "{messages}")
    ])

    # RAG prompts
    SUMMARIZE_COMMENTS = PromptTemplate.from_template(
        """Given the following comments related to the query "{query}",
provide a concise summary of the key themes and patterns:

Comments:
{comments}

Summary:"""
    )

    GENERATE_ANSWER = ChatPromptTemplate.from_messages([
        ("system", "You are analyzing public comments on federal regulations. Provide accurate, well-sourced answers."),
        ("human", """Query: {query}

Relevant comments found:
{context}

Please provide a comprehensive answer to the query based on these comments.
Include specific comment IDs when referencing particular comments.""")
    ])

    # Pipeline prompts
    CATEGORIZE_COMMENT = PromptTemplate.from_template(
        """Analyze the following public comment and categorize it.

Comment: {comment_text}

Provide:
1. Primary category (support/oppose/question/suggestion/concern/other)
2. Key topics mentioned (list)
3. Sentiment (positive/negative/neutral/mixed)
4. Brief reasoning for categorization

Format your response as JSON:
{{
  "category": "...",
  "topics": [...],
  "sentiment": "...",
  "reasoning": "..."
}}"""
    )

# Create singleton instance
prompts = ReggiePrompts()
```

Then update all files to import and use `prompts.DISCUSSION_SYSTEM`, etc.

### 1.2 Tool Docstrings as Prompts

**Problem:** In `agent/rag_graph.py` (line 35), the tool's docstring serves as both documentation and LLM instruction.

**Why it's a problem:**
- Mixing concerns (developer docs vs LLM prompts)
- Docstrings should follow Python conventions, not LLM requirements
- Can't optimize LLM prompts without affecting code documentation

**Recommended fix:**
Separate tool documentation from LLM descriptions:

```python
def search_comments(query: str, limit: int = 5) -> str:
    """Search for relevant comments using semantic similarity.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Formatted string containing matching comments
    """
    # Implementation...

# In prompts.py or when creating tools
from langchain_core.tools import StructuredTool

search_tool = StructuredTool.from_function(
    func=search_comments,
    name="search_comments",
    description="Search public comments database using semantic similarity. Use this when you need to find comments relevant to a specific topic or question. Returns up to {limit} most relevant comments with their IDs and text."
)
```

## 2. Agent Architecture

### 2.1 State Management Not Following LangGraph Best Practices

**Problem:** `agent/rag_graph.py` uses Pydantic `BaseModel` for `RAGState` (line 17) instead of `TypedDict`.

**Why it's a problem:**
- LangGraph documentation recommends `TypedDict` for state schemas
- Pydantic models add unnecessary overhead for state management
- `TypedDict` provides better integration with LangGraph's state handling
- Can cause issues with state updates and merging

**Recommended fix:**
In `agent/rag_graph.py`, replace:

```python
from pydantic import BaseModel, Field
from typing import List

class RAGState(BaseModel):
    query: str
    context: str = Field(default="")
    _relevant_comment_ids: List[str] = Field(default_factory=list)
```

With:

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class RAGState(TypedDict):
    """State for RAG graph processing."""
    query: str
    context: str
    relevant_comment_ids: list[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

### 2.2 Unused State Field

**Problem:** `_relevant_comment_ids` field in `RAGState` is populated but never used (line 90).

**Why it's a problem:**
- Dead code increases cognitive load
- Wastes memory and processing
- Suggests incomplete feature implementation
- Makes it unclear if this was intentional or a bug

**Recommended fix:**
Either:
1. Remove the field if truly unused
2. Use it for citation tracking:

```python
def generate_answer(state: RAGState) -> dict:
    """Generate final answer with citations."""
    # ... existing code ...

    return {
        "context": response,
        "relevant_comment_ids": state["relevant_comment_ids"],  # Pass through for citation
        "messages": [AIMessage(content=response)]
    }
```

### 2.3 Graph Creation Not Cached

**Problem:** `create_rag_graph()` creates a new graph on every call (line 117), including compiling the graph.

**Why it's a problem:**
- Graph compilation is expensive
- Creates unnecessary overhead for each query
- Wastes memory with duplicate graph instances
- Slower response times

**Recommended fix:**
Use module-level caching:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def create_rag_graph() -> CompiledGraph:
    """Create and compile the RAG graph (cached)."""
    workflow = StateGraph(RAGState)
    # ... build graph ...
    return workflow.compile()

# Or use a singleton pattern
_compiled_graph: CompiledGraph | None = None

def get_rag_graph() -> CompiledGraph:
    """Get the compiled RAG graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_rag_graph()
    return _compiled_graph
```

### 2.4 No Error Recovery in Graph

**Problem:** Graph has no error handling nodes; errors in retrieval or generation will crash the entire pipeline.

**Why it's a problem:**
- Poor user experience when errors occur
- No graceful degradation
- Difficult to debug issues in production
- Can't provide partial results when something fails

**Recommended fix:**
Add error recovery nodes:

```python
def handle_retrieval_error(state: RAGState) -> dict:
    """Handle errors during comment retrieval."""
    logger.error(f"Retrieval failed for query: {state['query']}")
    return {
        "context": "Unable to retrieve comments due to an error.",
        "messages": [AIMessage(content="I encountered an error searching for relevant comments. Please try rephrasing your question.")]
    }

def should_retry(state: RAGState) -> str:
    """Decide whether to retry after error."""
    # Could check retry count in state
    return "fallback" if state.get("retry_count", 0) > 2 else "retry"

# Add to graph
workflow.add_node("handle_error", handle_retrieval_error)
workflow.add_conditional_edges(
    "retrieve_comments",
    lambda s: "error" if not s.get("context") else "success",
    {"error": "handle_error", "success": "generate_answer"}
)
```

### 2.5 Discussion Agent Not Persisting History

**Problem:** `agent/discussion.py` creates agent with memory (line 45) but doesn't persist conversation history between invocations.

**Why it's a problem:**
- Each query loses context from previous questions
- Users can't have meaningful multi-turn conversations
- The memory setup is misleading (appears to work but doesn't persist)

**Recommended fix:**
Use LangGraph's checkpointing for persistence:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def create_discussion_agent(document_id: str, checkpoint_dir: str = ".checkpoints") -> CompiledGraph:
    """Create discussion agent with persistent memory."""
    # ... existing setup ...

    # Add checkpointing
    memory = SqliteSaver.from_conn_string(f"{checkpoint_dir}/discussion_{document_id}.db")
    return graph.compile(checkpointer=memory)

# When invoking
def run_discussion(document_id: str, query: str, session_id: str) -> str:
    """Run discussion with session persistence."""
    agent = create_discussion_agent(document_id)
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    return result["messages"][-1].content
```

### 2.6 Hardcoded Model Configuration

**Problem:** Models hardcoded in agent files (e.g., `agent/rag_graph.py` line 47, `agent/discussion.py` line 16).

**Why it's a problem:**
- Can't easily switch models for testing
- No way to use different models for different tasks
- Configuration not centralized

**Recommended fix:**
Add model config to config.py and use throughout:

```python
# In config.py
class AgentConfig(BaseSettings):
    """Agent configuration."""
    discussion_model: str = Field(default="gpt-4o-mini")
    rag_model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4000)

# In agent files
from reggie.config import AgentConfig

def create_discussion_agent(document_id: str):
    config = AgentConfig()
    llm = ChatOpenAI(
        model=config.discussion_model,
        temperature=config.temperature
    )
```

## 3. Pipeline Architecture

### 3.1 CommentProcessor Doing Too Much

**Problem:** `pipeline/processor.py` `CommentProcessor` class handles both orchestration and processing logic (lines 33-153).

**Why it's a problem:**
- Violates Single Responsibility Principle
- Hard to test individual processing steps
- Difficult to modify processing flow
- Can't reuse individual steps in different contexts

**Recommended fix:**
Split into orchestrator and processor:

```python
# pipeline/stages.py
from abc import ABC, abstractmethod

class PipelineStage(ABC):
    """Base class for pipeline stages."""

    @abstractmethod
    async def process(self, comment: Comment) -> Comment:
        """Process a comment through this stage."""
        pass

class CategorizationStage(PipelineStage):
    def __init__(self, categorizer: CommentCategorizer):
        self.categorizer = categorizer

    async def process(self, comment: Comment) -> Comment:
        category_data = await self.categorizer.categorize(comment.text)
        comment.category_data = category_data
        return comment

class EmbeddingStage(PipelineStage):
    def __init__(self, embedder: CommentEmbedder):
        self.embedder = embedder

    async def process(self, comment: Comment) -> Comment:
        chunks = self.embedder.embed_comment(comment)
        comment.chunks = chunks
        return comment

# pipeline/orchestrator.py
class PipelineOrchestrator:
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages

    async def process_comment(self, comment: Comment) -> Comment:
        for stage in self.stages:
            comment = await stage.process(comment)
        return comment
```

### 3.2 Database Connection Management

**Problem:** Direct connection management in processor (line 42) instead of using context managers consistently.

**Why it's a problem:**
- Connections might not be closed on errors
- Inconsistent with repository pattern
- Resource leaks possible

**Recommended fix:**
Use context manager pattern:

```python
# In db/connection.py
from contextlib import contextmanager

@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

# In processor.py
def process_document(self, document_id: str):
    with get_connection() as conn:
        doc_repo = DocumentRepository(conn)
        comment_repo = CommentRepository(conn)
        # ... process ...
```

### 3.3 No Pipeline Metrics

**Problem:** No telemetry or metrics collection for pipeline stages.

**Why it's a problem:**
- Can't identify bottlenecks
- No visibility into processing times
- Difficult to optimize
- Can't track failure rates

**Recommended fix:**
Add metrics decorator:

```python
from functools import wraps
import time
from typing import Callable

def track_stage_metrics(stage_name: str) -> Callable:
    """Decorator to track pipeline stage metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{stage_name} completed", extra={
                    "stage": stage_name,
                    "duration_ms": duration * 1000,
                    "status": "success"
                })
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{stage_name} failed", extra={
                    "stage": stage_name,
                    "duration_ms": duration * 1000,
                    "status": "error",
                    "error": str(e)
                })
                raise
        return wrapper
    return decorator

class CategorizationStage(PipelineStage):
    @track_stage_metrics("categorization")
    async def process(self, comment: Comment) -> Comment:
        # ... processing ...
```

### 3.4 No Stage Composition

**Problem:** Pipeline stages not composable; can't easily create different pipelines for different use cases.

**Why it's a problem:**
- Can't create lightweight pipelines for testing
- All comments must go through all stages
- Can't optimize for different document types

**Recommended fix:**
Make pipeline configuration-driven:

```python
# pipeline/builder.py
from dataclasses import dataclass
from typing import List

@dataclass
class PipelineConfig:
    """Configuration for pipeline stages."""
    categorize: bool = True
    embed: bool = True
    store: bool = True
    batch_size: int = 100

class PipelineBuilder:
    """Builder for creating custom pipelines."""

    def __init__(self):
        self._stages: List[PipelineStage] = []

    def with_categorization(self, categorizer: CommentCategorizer) -> 'PipelineBuilder':
        self._stages.append(CategorizationStage(categorizer))
        return self

    def with_embedding(self, embedder: CommentEmbedder) -> 'PipelineBuilder':
        self._stages.append(EmbeddingStage(embedder))
        return self

    def with_storage(self, repository: CommentRepository) -> 'PipelineBuilder':
        self._stages.append(StorageStage(repository))
        return self

    def build(self) -> PipelineOrchestrator:
        return PipelineOrchestrator(self._stages)

# Usage
pipeline = (PipelineBuilder()
    .with_categorization(categorizer)
    .with_embedding(embedder)
    .with_storage(repo)
    .build())
```

## 4. Code Organization

### 4.1 Inconsistent Dependency Injection

**Problem:** Mixed patterns - some classes take dependencies in constructor, others create them internally (compare `pipeline/processor.py` line 37 vs `agent/discussion.py` line 42).

**Why it's a problem:**
- Hard to test with mocks
- Tight coupling
- Inconsistent patterns across codebase
- Can't easily swap implementations

**Recommended fix:**
Standardize on constructor injection:

```python
# Before (in discussion.py)
def create_discussion_agent(document_id: str):
    config = DatabaseConfig()
    conn = get_db_connection()
    # ...

# After
def create_discussion_agent(
    document_id: str,
    comment_repo: CommentRepository,
    config: AgentConfig
):
    """Create discussion agent with injected dependencies."""
    # Use provided dependencies
```

### 4.2 Mixed Abstraction Levels

**Problem:** Functions mix high-level orchestration with low-level details (e.g., `pipeline/processor.py` `process_document` method).

**Why it's a problem:**
- Hard to understand at a glance
- Difficult to test specific logic
- Violates Single Level of Abstraction Principle

**Recommended fix:**
Extract low-level operations:

```python
# Before
def process_document(self, document_id: str):
    comments = self.comment_repo.get_comments_by_document(document_id)
    for comment in comments:
        # Complex categorization logic
        # Complex embedding logic
        # Complex storage logic

# After
def process_document(self, document_id: str):
    """Process all comments for a document."""
    comments = self._fetch_comments(document_id)
    processed = self._process_batch(comments)
    self._store_results(processed)

def _fetch_comments(self, document_id: str) -> List[Comment]:
    """Fetch comments for processing."""
    return self.comment_repo.get_comments_by_document(document_id)

def _process_batch(self, comments: List[Comment]) -> List[ProcessedComment]:
    """Process a batch of comments through pipeline."""
    # Processing logic

def _store_results(self, processed: List[ProcessedComment]):
    """Store processed results."""
    # Storage logic
```

### 4.3 Repository Returns Tuples

**Problem:** Some repository methods return tuples instead of Pydantic models (e.g., `db/repository.py` `search_similar_comments`).

**Why it's a problem:**
- No type safety
- Easy to mix up tuple positions
- Not consistent with rest of codebase
- Doesn't follow project guidelines

**Recommended fix:**
Create result models:

```python
# In models/agent.py or models/comment.py
from pydantic import BaseModel

class CommentSearchResult(BaseModel):
    """Result from comment similarity search."""
    comment_id: str
    comment_text: str
    similarity_score: float
    metadata: Dict[str, Any] = {}

# In repository.py
def search_similar_comments(
    self,
    document_id: str,
    query_embedding: List[float],
    limit: int = 5
) -> List[CommentSearchResult]:
    """Search for similar comments using vector similarity."""
    # ... query logic ...
    return [
        CommentSearchResult(
            comment_id=row[0],
            comment_text=row[1],
            similarity_score=row[2]
        )
        for row in cursor.fetchall()
    ]
```

## 5. LangChain Patterns

### 5.1 Not Using Runnable Interface

**Problem:** Custom functions don't implement Runnable interface, missing benefits of LangChain's composition patterns.

**Why it's a problem:**
- Can't use `.stream()` for streaming responses
- Can't compose with pipes `|`
- Can't use batch processing
- Missing LangChain's built-in error handling

**Recommended fix:**
Convert to Runnables:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Instead of plain functions, create runnables
retrieve_comments = RunnableLambda(
    lambda x: retrieve_comments_impl(x["query"])
).with_config({"run_name": "retrieve_comments"})

generate_answer = RunnableLambda(
    lambda x: generate_answer_impl(x["query"], x["context"])
).with_config({"run_name": "generate_answer"})

# Chain them
rag_chain = (
    {"query": RunnablePassthrough()}
    | retrieve_comments
    | generate_answer
)

# Now you can stream
for chunk in rag_chain.stream("What are the main concerns?"):
    print(chunk)
```

### 5.2 Missing Output Parsers

**Problem:** Manual parsing of LLM outputs (e.g., `pipeline/categorizer.py` line 67), no structured output with fallbacks.

**Why it's a problem:**
- Fragile JSON parsing
- No validation of output format
- Can't handle malformed responses gracefully
- Missing benefits of structured outputs

**Recommended fix:**
Use LangChain output parsers with retry:

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

class CategoryOutput(BaseModel):
    """Structured output for comment categorization."""
    category: str = Field(description="Primary category")
    topics: List[str] = Field(description="Key topics mentioned")
    sentiment: str = Field(description="Overall sentiment")
    reasoning: str = Field(description="Brief reasoning")

parser = PydanticOutputParser(pydantic_object=CategoryOutput)

categorization_chain = (
    prompts.CATEGORIZE_COMMENT
    | llm
    | parser
).with_fallbacks([
    # Fallback to simpler parsing if structured fails
    prompts.CATEGORIZE_COMMENT | llm
])
```

### 5.3 No Proper Chain Composition

**Problem:** Logic duplicated across agent and RAG graph instead of composing reusable chains.

**Why it's a problem:**
- Code duplication
- Inconsistent behavior
- Hard to maintain
- Can't test chains independently

**Recommended fix:**
Create reusable chain library:

```python
# chains/search.py
from langchain_core.runnables import Runnable

def create_search_chain(embedder: CommentEmbedder, repo: CommentRepository) -> Runnable:
    """Create reusable comment search chain."""
    def search_impl(query: str) -> List[CommentSearchResult]:
        embedding = embedder.embed_text(query)
        return repo.search_similar_comments(
            document_id=query.get("document_id"),
            query_embedding=embedding,
            limit=5
        )

    return RunnableLambda(search_impl)

# Use in both agent and RAG graph
search_chain = create_search_chain(embedder, repo)
```

### 5.4 Not Using LangChain Memory

**Problem:** Custom memory implementation in discussion agent instead of using LangChain's memory classes.

**Why it's a problem:**
- Reinventing the wheel
- Missing features like summarization, token limits
- Inconsistent with LangChain patterns
- More code to maintain

**Recommended fix:**
Use LangChain memory:

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# For short conversations
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# For long conversations with summarization
memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    return_messages=True,
    memory_key="chat_history"
)

# Use with agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)
```

## 6. Configuration Management

### 6.1 Direct os.environ Modification

**Problem:** `config.py` `setup_langsmith()` function directly modifies `os.environ` (lines 72-78), violating project guidelines.

**Why it's a problem:**
- Violates stated project policy against direct os.environ access
- Side effects not obvious from function signature
- Can cause issues in testing
- Not following own config class pattern

**Recommended fix:**
Make LangSmith config part of Pydantic config and set in one place:

```python
class LangSmithConfig(BaseSettings):
    """LangSmith configuration."""
    enabled: bool = Field(default=False)
    project: str = Field(default="reggie")
    tracing_enabled: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")

    def apply(self):
        """Apply LangSmith configuration to environment."""
        if self.enabled and self.api_key:
            import os
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.tracing_enabled).lower()
            os.environ["LANGCHAIN_PROJECT"] = self.project
            os.environ["LANGSMITH_API_KEY"] = self.api_key

# Usage at app startup
langsmith_config = LangSmithConfig()
langsmith_config.apply()
```

### 6.2 Multiple Config Classes

**Problem:** Four separate config classes (APIConfig, DatabaseConfig, EmbeddingConfig, ProcessingConfig) instead of unified configuration.

**Why it's a problem:**
- Have to import multiple classes
- No single source of truth
- Can't easily serialize full config
- Harder to override for testing

**Recommended fix:**
Create unified config with nested models:

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class ReggieConfig(BaseSettings):
    """Unified configuration for reggie application."""

    # Database
    db_connection_string: str = Field(alias="DATABASE_URL")
    db_pool_size: int = Field(default=10)

    # API
    reg_api_key: Optional[str] = Field(default=None, alias="REG_API_KEY")
    reg_api_base_url: str = Field(default="https://api.regulations.gov/v4")

    # Embeddings
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)

    # Processing
    batch_size: int = Field(default=100)
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)

    # Agent
    discussion_model: str = Field(default="gpt-4o-mini")
    rag_model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7)

    # LangSmith
    langsmith_enabled: bool = Field(default=False)
    langsmith_project: str = Field(default="reggie")
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton pattern
_config: Optional[ReggieConfig] = None

def get_config() -> ReggieConfig:
    """Get application configuration (singleton)."""
    global _config
    if _config is None:
        _config = ReggieConfig()
    return _config
```

## Priority Recommendations

### High Priority (Do First)
1. **Consolidate prompts** - Creates foundation for other improvements
2. **Fix RAGState to use TypedDict** - Prevents potential bugs
3. **Unify configuration** - Makes everything else easier to configure
4. **Add graph compilation caching** - Immediate performance improvement

### Medium Priority
5. **Standardize dependency injection** - Improves testability
6. **Extract pipeline stages** - Better modularity
7. **Add output parsers** - More robust LLM interactions
8. **Fix repository return types** - Type safety

### Lower Priority
9. **Implement pipeline metrics** - Helpful for optimization
10. **Convert to Runnables** - Better LangChain integration
11. **Add graph error recovery** - Improves user experience

## Implementation Notes

- Many of these changes are breaking changes - coordinate implementation
- Consider feature flags for gradual rollout
- Add integration tests before refactoring
- Update documentation as changes are made
- Consider creating migration guide for any API changes
