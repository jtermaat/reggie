# Phase 1: Document Keyword Extraction via Classification Pipeline

## Overview

This document describes how we extract domain-specific keywords, phrases, and entities from regulation comments to enable precise RAG query generation. Instead of a separate document profile system, we leverage the existing comment classification pipeline to extract keywords during ingestion.

## Problem Context

The query generation LLM receives no information about the document being searched. When a user asks "What do doctors say about payment changes?", the LLM must guess at domain terminology. We need to provide it with actual phrases and context from the document.

## Solution: Integrated Keyword Extraction

Every comment is already processed through gpt-5-nano for classification (category, sentiment, topics, etc.). We extend this existing pipeline to also extract:

1. **Keywords & Phrases**: Domain-specific terms and multi-word phrases (e.g., "Medicare reimbursement", "RVU", "conversion factor")
2. **Entities**: Named entities like organizations, regulations, CPT codes, and programs

This approach has several advantages over a separate sampling-based approach:

- **No additional LLM calls** - we're already touching every comment
- **Complete coverage** - every comment contributes keywords, not just a truncated sample
- **More accurate** - each comment gets focused attention
- **Simpler architecture** - no separate ProfileService needed
- **Incremental** - works naturally with streaming ingestion

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Comment Classification Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For each comment, gpt-5-nano extracts:                     │
│    - category, sentiment, topics (existing)                 │
│    - keywords_phrases (NEW)                                 │
│    - entities (NEW)                                         │
│                                                              │
│  Stored in comments.keywords_entities JSONB column          │
│                                                              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                Document-Level Aggregation                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  After all comments processed:                               │
│    1. Collect keywords_phrases from all comments            │
│    2. Collect entities from all comments                    │
│    3. Normalize and deduplicate                             │
│    4. Store in documents.aggregated_keywords JSONB          │
│                                                              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Query Generation (Phase 2)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Question + Document Keywords                           │
│                                                              │
│  aggregated_keywords contains:                               │
│    keywords_phrases: ["medicare reimbursement", "rvu",      │
│      "conversion factor", "telehealth supervision", ...]    │
│                                                              │
│    entities: ["American Medical Association", "CMS",        │
│      "MPFS CY2024", "CPT 99213", ...]                       │
│                                                              │
│  LLM uses these EXACT terms for full-text search queries    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Comments Table - New Column

```sql
-- Added to comments table
keywords_entities JSONB DEFAULT '{"keywords_phrases": [], "entities": []}'
```

Example value:
```json
{
  "keywords_phrases": ["Medicare reimbursement", "conversion factor", "E/M codes"],
  "entities": ["American Medical Association", "CMS", "CPT 99213"]
}
```

### Documents Table - New Column

```sql
-- Added to documents table
aggregated_keywords JSONB DEFAULT '{"keywords_phrases": [], "entities": []}'
```

Example value (aggregated from all comments, deduplicated):
```json
{
  "keywords_phrases": ["medicare reimbursement", "rvu", "conversion factor", ...],
  "entities": ["American Medical Association", "CMS", "MPFS CY2024", ...]
}
```

---

## Implementation Details

### CommentClassification Model

The `CommentClassification` Pydantic model (used for structured LLM output) now includes:

```python
class CommentClassification(BaseModel):
    # ... existing fields ...
    keywords_phrases: List[str] = Field(
        default_factory=list,
        description="Domain-specific keywords and multi-word phrases (3-10 terms)"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Named entities (organizations, regulations, codes, programs)"
    )
```

### Categorization Prompt

The categorization prompt now instructs the LLM to extract:

**Keywords & Phrases** (3-10 terms):
- Acronyms and technical terms (RVU, CMS, MPFS, HCPCS, CPT, E/M)
- Multi-word domain phrases ("Medicare reimbursement", "conversion factor")
- Stakeholder/role terms that appear in the text
- Focus on terms useful for searching
- Avoid generic words like "concerns", "issues", "important"

**Entities**:
- Organization names (medical associations, companies, agencies)
- Regulation/program names (MPFS, MIPS, QPP)
- Code references (CPT codes, HCPCS codes)
- Specific named programs, policies, or initiatives
- Only include entities explicitly mentioned in the comment

### Aggregation Logic

After all comments are processed, the `aggregate_keywords` function:

1. Fetches `keywords_entities` from all comments for the document
2. Collects all keywords (normalized to lowercase for deduplication)
3. Collects all entities (preserves case but deduplicates)
4. Returns sorted, deduplicated lists
5. Stores in `documents.aggregated_keywords`

```python
async def aggregate_keywords(self, document_id: str) -> dict:
    # Fetch all keywords_entities from comments
    # Normalize, deduplicate, and return aggregated result
    return {
        "keywords_phrases": sorted(list(all_keywords)),
        "entities": sorted(list(all_entities)),
    }
```

### Pipeline Integration

**Orchestrator** (`reggie/pipeline/orchestrator.py`):
- After processing all comments, calls `_aggregate_document_keywords()`
- Aggregation runs within the same database transaction

**Streamer** (`reggie/pipeline/streamer.py`):
- After streaming completes, calls `_aggregate_document_keywords()`
- Handles the producer-consumer streaming pattern

Both pipelines now:
1. Extract keywords/entities during classification (per-comment)
2. Store in `comments.keywords_entities`
3. Aggregate to document level after all comments processed
4. Store in `documents.aggregated_keywords`

---

## Migration

For existing documents with already-classified comments:
- New documents processed after this change will have keywords
- Existing documents can be reprocessed with `reggie process <document_id>` to extract keywords

---

## What This Enables (Phase 2)

The RAG query generation prompt will receive the aggregated keywords:

```
DOCUMENT CONTEXT:
Keywords/phrases available in this document:
medicare reimbursement, rvu, conversion factor, telehealth supervision,
practice expense, fee schedule, mpfs, prior authorization, e/m codes, ...

Entities mentioned:
American Medical Association, CMS, MPFS CY2024, CPT 99213, ...
```

The LLM can then select EXACT terms from these lists when formulating full-text search queries, instead of guessing at terminology.

---

## Files Modified

| File | Changes |
|------|---------|
| `reggie/models/comment.py` | Added `keywords_phrases` and `entities` to `CommentClassification`; added `keywords_entities` to `Comment` |
| `reggie/prompts.py` | Extended categorization prompt with keyword/entity extraction instructions |
| `reggie/db/sql/schema/schema.sql` | Added `keywords_entities` to comments, `aggregated_keywords` to documents |
| `reggie/db/repositories/comment_repository.py` | Updated `store_comment()` and `update_comment_classification()` to handle keywords/entities |
| `reggie/db/repositories/document_repository.py` | Added `aggregate_keywords()`, `update_aggregated_keywords()`, `get_aggregated_keywords()` |
| `reggie/pipeline/stages.py` | Updated `CategorizationStage` and `BatchCategorizationStage` to pass new fields |
| `reggie/pipeline/orchestrator.py` | Added `_aggregate_document_keywords()` call after processing |
| `reggie/pipeline/streamer.py` | Added `_aggregate_document_keywords()` call after streaming |
