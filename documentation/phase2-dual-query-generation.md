# Phase 2: Context-Aware Dual Query Generation

## Overview

This document describes the implementation of context-aware dual query generation for hybrid search. Building on the aggregated keywords extracted in Phase 1, this phase modifies the query generation system to:

1. Generate two separate queries optimized for different search backends
2. Use actual keywords/entities from the document to ground query generation
3. Route the appropriate query to each search component

## Prerequisites

- Phase 1 must be complete (keyword extraction via classification pipeline)
- Documents must have been processed (or reprocessed) to have `aggregated_keywords` populated

---

## Problem Context

### Current Query Generation Flow

```
User Question: "What do doctors say about payment changes?"
                           │
                           ▼
              ┌────────────────────────┐
              │ RAG_GENERATE_QUERY     │
              │ Prompt                 │
              │                        │
              │ "Generate a verbose,   │
              │ detailed search query  │
              │ (complete phrases,     │
              │ not keywords)..."      │
              └────────────────────────┘
                           │
                           ▼
              Single query: "concerns from doctors about
              proposed payment changes and their impact
              on medical practices"
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    Vector Search (70%)            FTS/ts_rank_cd (30%)
    ✓ Works well with              ✗ plainto_tsquery ANDs
      verbose phrases                all 12+ words
                                   ✗ Few/no matches
                                   ✗ 30% weight wasted
```

### Issues

1. **Single query for dual backends**: The same verbose query goes to both vector search AND full-text search
2. **No document context**: LLM must guess at domain terminology
3. **FTS underperformance**: `plainto_tsquery('english', 'concerns from doctors about proposed payment changes...')` creates a query that ANDs 12+ words, dramatically reducing matches
4. **Guessed terminology**: LLM might say "doctors" when comments use "physicians", or "payment changes" when they use "RVU methodology"

### Solution: Dual Query with Document Context

```
User Question: "What do doctors say about payment changes?"
                           │
                           ▼
              ┌────────────────────────────────────────┐
              │ Fetch aggregated_keywords from         │
              │ documents table (from Phase 1)         │
              │                                        │
              │ keywords_phrases: ["medicare reimburse-│
              │   ment", "rvu", "conversion factor"...]│
              │ entities: ["American Medical Assoc.",  │
              │   "CMS", "MPFS CY2024", "CPT 99213"...│
              └────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────────────────────┐
              │ Context-Aware Dual Query Generation    │
              │                                        │
              │ Generate:                              │
              │ - semantic_query (verbose, 8-15 words) │
              │ - keyword_query (concise, 2-5 terms)   │
              │                                        │
              │ Using EXACT terms from keywords_phrases│
              └────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    semantic_query:              keyword_query:
    "physician concerns          "physician reimbursement
    about Medicare               RVU"
    reimbursement rate
    changes fee schedule
    impact on practice"
           │                               │
           ▼                               ▼
    Vector Search (70%)            FTS/ts_rank_cd (30%)
    ✓ Rich semantic context        ✓ Only 3 precise terms
    ✓ Captures related concepts    ✓ Terms from actual docs
                                   ✓ High recall with AND
```

---

## Architectural Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG Graph Flow                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │ User Question    │                                               │
│  │ + document_id    │                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    generate_query Node                        │   │
│  │                                                               │   │
│  │  1. Fetch aggregated_keywords from documents table           │   │
│  │     (or use cached from state)                               │   │
│  │  2. Format keywords as document_context string               │   │
│  │  3. Call query_generation_chain with:                        │   │
│  │     - question                                                │   │
│  │     - iteration_context                                       │   │
│  │     - document_context  <── NEW                              │   │
│  │  4. Store in state:                                          │   │
│  │     - current_semantic_query  <── NEW                        │   │
│  │     - current_keyword_query   <── NEW                        │   │
│  │     - aggregated_keywords (cached)                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    search_vectors Node                        │   │
│  │                                                               │   │
│  │  If hybrid mode:                                              │   │
│  │    search_chain.ainvoke({                                    │   │
│  │      "semantic_query": state.current_semantic_query,         │   │
│  │      "keyword_query": state.current_keyword_query            │   │
│  │    })                                                         │   │
│  │                                                               │   │
│  │  Inside hybrid_search_chain:                                 │   │
│  │    - Embed semantic_query → vector search                    │   │
│  │    - Pass keyword_query → ts_rank_cd                         │   │
│  │    - RRF fusion of results                                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                                                          │
│           ▼                                                          │
│       [assess → select → extract → END]                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Changes

| Component | Change | Purpose |
|-----------|--------|---------|
| `QueryGeneration` model | Replace `query` with `semantic_query` + `keyword_query` | Separate queries for each backend |
| `RAGState` | Add `current_semantic_query`, `current_keyword_query`, `aggregated_keywords` | Track dual queries and cached keywords |
| `RAG_GENERATE_QUERY` prompt | Add `{document_context}` variable, remove few-shot guessing | Ground generation in real content |
| `generate_query` node | Fetch aggregated_keywords, format context, extract dual queries | Provide context to LLM |
| `search_vectors` node | Pass dict with both queries to chain | Route queries appropriately |
| `create_hybrid_search_chain` | Accept dict input, route to correct search | Use right query for each backend |

---

## Implementation Details

### 1. Update QueryGeneration Model

**File**: `reggie/models/agent.py`

```python
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from reggie.models.comment import Sentiment, Category, Topic


class QueryGeneration(BaseModel):
    """
    Dual query and filter generation for hybrid RAG search.

    Generates two separate queries optimized for different search backends:
    - semantic_query: Verbose, contextual query for vector/embedding search
    - keyword_query: Concise, precise query for full-text search (ts_rank_cd)
    """

    semantic_query: str = Field(
        description=(
            "Verbose query (8-15 words) for vector/embedding similarity search. "
            "Use complete phrases that capture the full meaning and intent. "
            "Include related concepts, synonyms, and contextual language."
        )
    )

    keyword_query: str = Field(
        description=(
            "Concise query (2-5 terms) for PostgreSQL full-text search. "
            "Use EXACT terms from the document's keyword_phrases when relevant. "
            "These terms are ANDed together, so fewer precise terms = better recall."
        )
    )

    sentiment_filter: Optional[Sentiment] = Field(
        default=None,
        description="Optional filter for comment sentiment"
    )

    category_filter: Optional[Category] = Field(
        default=None,
        description="Optional filter for commenter category"
    )

    topics_filter: Optional[List[Topic]] = Field(
        default=None,
        description="Optional list of topics to filter by"
    )

    topic_filter_mode: Literal["any", "all"] = Field(
        default="any",
        description="When using topics_filter: 'any' matches comments with any listed topic, 'all' requires all topics"
    )

    reasoning: str = Field(
        description="Brief explanation of why these queries and filters were chosen"
    )
```

### 2. Update RAGState

**File**: `reggie/models/agent.py`

```python
from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add


class RAGState(TypedDict, total=False):
    """State for the RAG search graph."""

    # Document identification
    document_id: str

    # Document context (cached after first fetch)
    aggregated_keywords: Optional[dict]  # NEW - {"keywords_phrases": [...], "entities": [...]}

    # Conversation
    messages: Annotated[Sequence[BaseMessage], add]

    # Current search parameters - UPDATED
    current_semantic_query: str  # NEW (replaces current_query)
    current_keyword_query: str   # NEW
    filters: Dict[str, Any]
    topic_filter_mode: str

    # Search results
    search_results: List[Any]
    all_retrieved_chunks: Dict[str, Any]

    # Selection and extraction
    relevant_comment_ids: List[str]
    final_snippets: List[Any]

    # Iteration control
    iteration_count: int
    max_iterations: int
    has_enough_information: Optional[str]
```

### 3. Update Query Generation Prompt

**File**: `reggie/prompts.py`

```python
from langchain_core.prompts import ChatPromptTemplate


RAG_GENERATE_QUERY = ChatPromptTemplate.from_messages([
    ("system", """You are generating search queries to find relevant public comments about a healthcare regulation document.

Generate TWO queries optimized for hybrid search:

## 1. semantic_query (8-15 words)
For vector/embedding similarity search:
- Capture the full meaning and intent of the user's question
- Include related concepts and contextual language
- Can paraphrase and expand beyond the literal question
- Use domain terminology from the document context below

## 2. keyword_query (2-5 terms)
For PostgreSQL full-text search (ts_rank_cd):
- Use EXACT terms from the keywords_phrases list below
- These terms are ANDed together (ALL must match)
- Fewer, more precise terms = better recall
- Choose terms that would appear in relevant comments
- Avoid generic words like "the", "about", "concerns"

## DOCUMENT CONTEXT
{document_context}

The above shows keywords and entities extracted from all comments in this document.
Use these EXACT terms when formulating your keyword_query.

## METADATA FILTERS (optional, use when clearly helpful)
- sentiment_filter: 'for', 'against', 'mixed', 'unclear'
- category_filter: commenter category
- topics_filter: list of topic values
- topic_filter_mode: 'any' (has any topic) or 'all' (has all topics)

Only apply filters when they clearly narrow to relevant content."""),
    ("user", """Question: {question}

{iteration_context}

Generate semantic_query and keyword_query using the document context above.""")
])
```

### 4. Format Document Context

**File**: `reggie/agent/rag_graph.py` (new helper function)

```python
def format_document_context(aggregated_keywords: dict) -> str:
    """
    Format aggregated keywords for the query generation prompt.

    Args:
        aggregated_keywords: Dict with keywords_phrases and entities lists

    Returns:
        Formatted string for prompt context
    """
    lines = []

    keywords = aggregated_keywords.get("keywords_phrases", [])
    entities = aggregated_keywords.get("entities", [])

    if keywords:
        lines.append("### Keywords/Phrases Available")
        lines.append("Use these exact terms for keyword_query:")
        lines.append(", ".join(keywords[:50]))  # Limit for prompt size
        lines.append("")

    if entities:
        lines.append("### Entities Mentioned")
        lines.append(", ".join(entities[:30]))  # Limit for prompt size

    if not lines:
        return "No keywords available for this document."

    return "\n".join(lines)
```

### 5. Update generate_query Node

**File**: `reggie/agent/rag_graph.py`

```python
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from reggie.agent.chains import create_query_generation_chain
from reggie.db.unit_of_work import UnitOfWork
from reggie.models.agent import RAGState

logger = logging.getLogger(__name__)


async def generate_query(state: RAGState) -> Dict[str, Any]:
    """
    Generate dual search queries using document context.

    This node:
    1. Fetches aggregated_keywords from documents table (or uses cached)
    2. Formats keywords as context for the LLM
    3. Calls the query generation chain
    4. Returns both semantic and keyword queries

    Args:
        state: Current RAG state

    Returns:
        Updated state with queries and cached keywords
    """
    document_id = state["document_id"]
    logger.debug(f"Generating queries for document {document_id}")

    # Extract user question from messages
    user_messages = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    if not user_messages:
        raise ValueError("No user question found in state")
    question = user_messages[-1].content

    # Build iteration context
    iteration_count = state.get("iteration_count", 0)
    if iteration_count == 0:
        iteration_context = "This is the first search iteration."
    else:
        all_retrieved = state.get("all_retrieved_chunks", {})
        comment_count = len(all_retrieved)
        iteration_context = (
            f"Previous searches found {comment_count} comments. "
            "Generate DIFFERENT queries to find additional relevant content. "
            "Try varying the keyword_query terms or using different filters."
        )

    # Fetch aggregated keywords (use cached if available)
    aggregated_keywords = state.get("aggregated_keywords")
    if not aggregated_keywords:
        async with UnitOfWork() as uow:
            aggregated_keywords = await uow.documents.get_aggregated_keywords(document_id)

        if not aggregated_keywords:
            logger.warning(f"No keywords found for document {document_id}, using empty context")
            document_context = "No keywords available for this document."
        else:
            document_context = format_document_context(aggregated_keywords)
    else:
        document_context = format_document_context(aggregated_keywords)

    # Create and invoke the query generation chain
    query_generation_chain = create_query_generation_chain()

    query_gen = await query_generation_chain.ainvoke({
        "question": question,
        "iteration_context": iteration_context,
        "document_context": document_context,
    })

    logger.info(f"Generated semantic_query: {query_gen.semantic_query}")
    logger.info(f"Generated keyword_query: {query_gen.keyword_query}")
    logger.debug(f"Query reasoning: {query_gen.reasoning}")

    # Build filters dict
    filters = {}
    if query_gen.sentiment_filter:
        filters["sentiment"] = query_gen.sentiment_filter.value
    if query_gen.category_filter:
        filters["category"] = query_gen.category_filter.value
    if query_gen.topics_filter:
        filters["topics"] = [t.value for t in query_gen.topics_filter]

    return {
        "aggregated_keywords": aggregated_keywords,  # Cache for subsequent iterations
        "current_semantic_query": query_gen.semantic_query,
        "current_keyword_query": query_gen.keyword_query,
        "filters": filters,
        "topic_filter_mode": query_gen.topic_filter_mode,
        "iteration_count": iteration_count + 1,
        "messages": [
            AIMessage(content=(
                f"Searching with:\n"
                f"  semantic: '{query_gen.semantic_query}'\n"
                f"  keywords: '{query_gen.keyword_query}'\n"
                f"  reasoning: {query_gen.reasoning}"
            ))
        ],
    }
```

### 6. Update search_vectors Node

**File**: `reggie/agent/rag_graph.py`

```python
async def search_vectors(state: RAGState) -> Dict[str, Any]:
    """
    Search for relevant comment chunks using the generated queries.

    In hybrid mode, passes separate queries:
    - semantic_query → embedded for vector search
    - keyword_query → used for ts_rank_cd full-text search

    Args:
        state: Current RAG state with queries

    Returns:
        Updated state with search results
    """
    document_id = state["document_id"]
    semantic_query = state.get("current_semantic_query", "")
    keyword_query = state.get("current_keyword_query", "")
    filters = state.get("filters", {})
    topic_filter_mode = state.get("topic_filter_mode", "any")

    logger.debug(f"Searching with semantic='{semantic_query}', keywords='{keyword_query}'")

    config = get_config()

    # Create search chain based on mode
    if config.search_mode == "hybrid":
        search_chain = create_hybrid_search_chain(
            document_id=document_id,
            limit=config.vector_search_limit,
            vector_weight=config.hybrid_vector_weight,
            fts_weight=config.hybrid_fts_weight,
            rrf_k=config.hybrid_rrf_k,
            sentiment_filter=filters.get("sentiment"),
            category_filter=filters.get("category"),
            topics_filter=filters.get("topics"),
            topic_filter_mode=topic_filter_mode,
        )

        # Pass both queries as dict
        results = await search_chain.ainvoke({
            "semantic_query": semantic_query,
            "keyword_query": keyword_query,
        })

    elif config.search_mode == "vector":
        search_chain = create_vector_search_chain(
            document_id=document_id,
            limit=config.vector_search_limit,
            sentiment_filter=filters.get("sentiment"),
            category_filter=filters.get("category"),
            topics_filter=filters.get("topics"),
            topic_filter_mode=topic_filter_mode,
        )
        # Vector-only uses semantic query
        results = await search_chain.ainvoke(semantic_query)

    elif config.search_mode == "fts":
        search_chain = create_fulltext_search_chain(
            document_id=document_id,
            limit=config.vector_search_limit,
            sentiment_filter=filters.get("sentiment"),
            category_filter=filters.get("category"),
            topics_filter=filters.get("topics"),
            topic_filter_mode=topic_filter_mode,
        )
        # FTS-only uses keyword query
        results = await search_chain.ainvoke(keyword_query)

    else:
        raise ValueError(f"Unknown search mode: {config.search_mode}")

    logger.info(f"Search returned {len(results)} chunks")

    # Merge with existing results
    all_retrieved = dict(state.get("all_retrieved_chunks", {}))
    for result in results:
        comment_id = result.comment_id
        if comment_id not in all_retrieved:
            all_retrieved[comment_id] = []
        all_retrieved[comment_id].append(result)

    return {
        "search_results": results,
        "all_retrieved_chunks": all_retrieved,
    }
```

### 7. Update Hybrid Search Chain

**File**: `reggie/agent/chains.py`

```python
from typing import Dict, List, Union

from langchain_core.runnables import Runnable, RunnableLambda

from reggie.db.connection import UnitOfWork
from reggie.models.search import CommentChunkSearchResult


def create_hybrid_search_chain(
    document_id: str,
    limit: int = 10,
    vector_weight: float = 0.7,
    fts_weight: float = 0.3,
    rrf_k: int = 60,
    sentiment_filter: str = None,
    category_filter: str = None,
    topics_filter: List[str] = None,
    topic_filter_mode: str = "any",
) -> Runnable:
    """
    Create a hybrid search chain that accepts separate queries for vector and FTS.

    Args:
        document_id: Document to search within
        limit: Maximum results to return
        vector_weight: Weight for vector search in RRF (default 0.7)
        fts_weight: Weight for FTS in RRF (default 0.3)
        rrf_k: RRF ranking constant (default 60)
        sentiment_filter: Optional sentiment filter
        category_filter: Optional category filter
        topics_filter: Optional topics filter
        topic_filter_mode: 'any' or 'all' for topic matching

    Returns:
        Runnable that accepts dict with 'semantic_query' and 'keyword_query'
    """
    embedding_chain = create_embedding_chain()

    async def search_impl(
        query_input: Union[str, Dict[str, str]]
    ) -> List[CommentChunkSearchResult]:
        """
        Execute hybrid search with separate queries for each backend.

        Args:
            query_input: Either a string (backward compat) or dict with:
                - semantic_query: Query for vector embedding
                - keyword_query: Query for full-text search

        Returns:
            List of search results ranked by RRF fusion
        """
        # Handle both string and dict input for backward compatibility
        if isinstance(query_input, str):
            semantic_query = query_input
            keyword_query = query_input
        else:
            semantic_query = query_input.get("semantic_query", "")
            keyword_query = query_input.get("keyword_query", semantic_query)

        # Generate embedding from semantic query
        embedding = await embedding_chain.ainvoke(semantic_query)

        # Execute hybrid search with separate queries
        async with UnitOfWork() as uow:
            results = await uow.chunks.search_hybrid(
                document_id=document_id,
                query_embedding=embedding,
                query_text=keyword_query,  # Use keyword query for FTS
                limit=limit,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
                rrf_k=rrf_k,
                sentiment_filter=sentiment_filter,
                category_filter=category_filter,
                topics_filter=topics_filter,
                topic_filter_mode=topic_filter_mode,
            )

        return results

    return RunnableLambda(search_impl).with_config(
        {"run_name": f"hybrid_search_{document_id}"}
    )
```

### 8. Update Query Generation Chain

**File**: `reggie/agent/chains.py`

```python
from langchain_openai import ChatOpenAI

from reggie.config import get_config
from reggie.models.agent import QueryGeneration
from reggie.prompts import RAG_GENERATE_QUERY


def create_query_generation_chain() -> Runnable:
    """
    Create the query generation chain with structured output.

    This chain:
    1. Takes question, iteration_context, and document_context
    2. Generates QueryGeneration with semantic_query and keyword_query
    3. Returns structured output

    Returns:
        Runnable chain that outputs QueryGeneration
    """
    config = get_config()

    llm = ChatOpenAI(
        model=config.rag_model,
        temperature=0,
    )

    # Use structured output to ensure valid QueryGeneration
    structured_llm = llm.with_structured_output(QueryGeneration)

    return RAG_GENERATE_QUERY | structured_llm
```

---

## Testing Considerations

### Unit Tests

1. **QueryGeneration model**:
   - Verify both queries are required
   - Verify field descriptions match expectations

2. **format_document_context**:
   - Test with full aggregated_keywords dict
   - Test with empty lists
   - Verify keywords_phrases are clearly labeled

3. **generate_query node**:
   - Mock aggregated_keywords fetch
   - Verify context formatting
   - Verify dual queries in output

4. **search_vectors node**:
   - Verify dict passed to hybrid chain
   - Verify backward compat with string input

5. **create_hybrid_search_chain**:
   - Test with dict input
   - Test with string input (backward compat)
   - Verify semantic_query goes to embedding
   - Verify keyword_query goes to FTS

### Integration Tests

1. **End-to-end query generation**:
   - Load document with aggregated_keywords populated
   - Submit question
   - Verify two different queries generated
   - Verify keyword_query uses terms from aggregated_keywords

2. **Search routing**:
   - Verify semantic query embedded correctly
   - Verify keyword query passed to ts_rank_cd
   - Verify results are RRF-fused

3. **Iteration behavior**:
   - First query uses aggregated_keywords
   - Second query gets iteration context
   - Verify aggregated_keywords cached in state

---

## Migration Path

### Backward Compatibility

The implementation maintains backward compatibility:

1. **String input to search chains**: Still works, uses same query for both
2. **Existing prompts**: Can coexist until cutover
3. **Vector-only mode**: Uses semantic_query, ignores keyword_query
4. **FTS-only mode**: Uses keyword_query, ignores semantic_query

### Rollout Steps

1. Deploy Phase 1 (keyword extraction via classification pipeline)
2. Process documents to populate aggregated_keywords (or reprocess existing)
3. Deploy Phase 2 code (dual queries)
4. Verify in staging
5. Enable for production

### Rollback

If issues arise:
- Set `search_mode = "vector"` to bypass FTS
- Or revert to single-query prompt

---

## Success Criteria

Phase 2 is complete when:

1. [ ] `QueryGeneration` model has `semantic_query` and `keyword_query`
2. [ ] `RAGState` includes dual query fields and aggregated_keywords cache
3. [ ] `RAG_GENERATE_QUERY` prompt uses `{document_context}`
4. [ ] `format_document_context` produces readable context from aggregated_keywords
5. [ ] `generate_query` node fetches aggregated_keywords and passes context
6. [ ] `search_vectors` node routes queries correctly
7. [ ] `create_hybrid_search_chain` accepts dict input
8. [ ] Backward compatibility maintained for string input
9. [ ] Unit tests passing
10. [ ] Integration test verifying end-to-end flow

---

## Expected Improvements

After implementation, we expect:

1. **Better FTS recall**: Keyword queries with 2-5 precise terms vs 10+ verbose terms
2. **More relevant results**: Keywords match actual document terminology
3. **Improved RRF fusion**: Both search components contribute meaningful results
4. **Better filter suggestions**: LLM can see actual distributions
5. **Reduced hallucination**: No guessing at domain terms

---

## Monitoring

Track these metrics to validate improvement:

1. **FTS match rate**: % of searches where FTS returns results
2. **Query term overlap**: % of keyword_query terms found in aggregated_keywords.keywords_phrases
3. **Result diversity**: % of results from FTS vs vector
4. **User satisfaction**: Relevance ratings on search results
