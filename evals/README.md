# Reggie Agent Evaluation Suite

This directory contains the evaluation framework for the Reggie discussion agent, using LangSmith.

## Overview

The evaluation suite tests the ReAct agent's ability to:
- Answer statistical queries using the `get_statistics` tool
- Perform semantic search using the `search_comments` tool with RAG
- Apply appropriate filters (category, sentiment, topics) based on query context
- Handle complex multi-step reasoning
- Deal with edge cases and challenging queries

## Files

- **`evaluation_dataset.py`**: 19 carefully crafted test cases based on actual database content
- **`evaluators.py`**: 5 custom evaluators (tool selection, content relevance, answer completeness, factual accuracy, filter application)
- **`run_evaluation.py`**: Main evaluation runner with LangSmith integration

## Usage

### View Dataset Summary

```bash
python -m evals.run_evaluation --report-only
```

### Run Full Evaluation

```bash
# With LangSmith (requires LANGSMITH_API_KEY in .env)
python -m evals.run_evaluation

# Quick test with limited examples
python -m evals.run_evaluation --limit 5

# Test with different model
python -m evals.run_evaluation --model gpt-5-mini

# Custom experiment name
python -m evals.run_evaluation --experiment-name "baseline-v1"
```

## Dataset Coverage

- **19 test cases** across 8 categories
- **Difficulty**: 3 easy, 7 medium, 9 hard
- **Query types**:
  - 6 statistical queries
  - 7 RAG queries
  - 2 multi-step queries
  - 7 filtered queries

## Evaluators

1. **Tool Selection** (rule-based): Checks if the agent used the expected tools
2. **Content Relevance** (rule-based): Verifies expected keywords/phrases in response
3. **Answer Completeness** (LLM-based): Evaluates how fully the query is answered
4. **Factual Accuracy** (LLM-based): Validates factual correctness of claims
5. **Filter Application** (LLM-based): Assesses proper filter inference and application

## Example Test Cases

**Easy**: "What do doctors generally think about this rule?"
- Tests basic sentiment breakdown by category

**Medium**: "How many physicians support the telehealth provisions?"
- Tests multi-filter application (category + sentiment + topic)

**Hard**: "Are there more comments for or against telehealth provisions, and what are the main arguments on each side?"
- Tests multi-step reasoning (statistics → RAG search)

**Precision**: "What specific arguments are made about the indirect PE floor and payment stability for psychological testing codes?"
- Tests finding very specific technical content (references to codes 96112, 96132)