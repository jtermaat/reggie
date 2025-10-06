# Testing Plan for Reggie

## Philosophy

This testing plan demonstrates **judicious testing** - focusing on high-value tests that catch real bugs, not just maximizing coverage numbers. For an MVP interview project with limited time, we prioritize:

1. **Critical path testing** - Ensure core workflows work end-to-end
2. **Edge case coverage** - Handle errors, empty data, and boundary conditions
3. **Smart mocking** - Mock external services (APIs), use real local resources (test DB)
4. **Maintainability** - Tests should be easy to understand and update

## Non-Negotiable: Zero External API Costs

**ALL external APIs are mocked in tests:**
- ✅ OpenAI API (categorization & embeddings) - mocked with `pytest-mock`
- ✅ Regulations.gov API - mocked with `httpx-mock`
- ✅ PostgreSQL - **real test database** (local, no cost, catches real SQL issues)

**We never make real API calls in tests.** This ensures fast, deterministic, cost-free test runs.

## Test Categories

### 1. Integration Tests (Highest Priority)
**Why:** These provide the most value - they test that components work together correctly.

#### 1.1 Database Integration Tests (`tests/integration/test_database.py`)
- **Setup:** Use a test PostgreSQL database (or pytest-postgresql)
- **Tests:**
  - ✅ Document CRUD operations (create, read, update via upsert)
  - ✅ Comment storage with classifications
  - ✅ Comment chunks with embeddings (vector operations)
  - ✅ Cascading deletes (delete document → comments → chunks)
  - ✅ Idempotent upserts (same document twice)
  - ✅ List documents with aggregated stats

**Why these matter:** Database operations are critical infrastructure. If these fail, nothing works.

#### 1.2 API Client Integration Tests (`tests/integration/test_api_client.py`)
- **Setup:** Use `httpx-mock` to mock ALL regulations.gov HTTP responses
- **Tests:**
  - ✅ Fetch document metadata (mocked response)
  - ✅ Paginate through comments (mock 2-3 pages of responses)
  - ✅ Handle API errors with retry logic (mock 429/500 errors)
  - ✅ Date-based windowing for >5000 comments (mock page 20 scenario)
  - ✅ Empty results (mock empty response)

**Why these matter:** API interactions are the data source. Pagination bugs would lose data silently.
**Note:** All HTTP calls are mocked - we never hit the real regulations.gov API.

#### 1.3 End-to-End Pipeline Tests (`tests/integration/test_pipeline.py`)
- **Setup:** Test database + mocked regulations.gov API + mocked OpenAI API
- **Tests:**
  - ✅ Load document flow: mocked API → real DB storage
  - ✅ Process comments flow: real DB → mocked categorization → mocked embedding → DB storage
  - ✅ Handle processing errors gracefully (mock OpenAI failures)

**Why these matter:** This is the user-facing workflow. Must work reliably.
**Note:** Both external APIs are mocked - zero costs, fast execution.

### 2. Unit Tests (Selected High-Value Areas)
**Why:** Test specific logic in isolation where it provides clear value. Avoid testing trivial code.

#### 2.1 Model Validation Tests (`tests/unit/test_models.py`)
- **Tests:**
  - ✅ Pydantic validation (required fields, type checking)
  - ✅ Enum constraints (Category, Sentiment values)
  - ✅ Invalid data rejection with clear errors

**Why these matter:** Models are our data contracts. Validation prevents bad data propagation.

#### 2.2 Embedder Logic Tests (`tests/unit/test_embedder.py`)
- **Setup:** Mock OpenAIEmbeddings (zero API costs)
- **Tests:**
  - ✅ Text chunking (verify chunk sizes, overlap)
  - ✅ Empty/edge case text handling
  - ✅ Error handling (zero vectors on mock failure)

**Why these matter:** Chunking logic directly affects retrieval quality. Must be correct.

#### 2.3 Categorizer Logic Tests (`tests/unit/test_categorizer.py`)
- **Setup:** Mock ChatOpenAI responses (zero API costs)
- **Tests:**
  - ✅ Context building (comment + metadata formatting)
  - ✅ Fallback to defaults on mocked API errors
  - ✅ Batch processing with mixed success/failures

**Why these matter:** Categorization is core business logic. Error handling is critical.

### 3. Repository Tests (`tests/unit/test_repositories.py`)
**Note:** Can be tested with real test DB (covered in integration) or mocked for pure unit testing.

- **Focus areas:**
  - ✅ Upsert logic (ON CONFLICT behavior)
  - ✅ Query result mapping to models
  - ✅ Edge cases (None values, empty results)

**Why these matter:** Repository bugs cause data corruption or SQL errors.
**Testing approach:** Covered primarily in integration tests with real DB to catch actual SQL issues.

### 4. CLI Tests (User-Facing)
**Why:** CLI is the user interface. Must provide good UX.

#### 4.1 CLI Command Tests (`tests/cli/test_commands.py`)
- **Setup:** Use Click's CliRunner
- **Tests:**
  - ✅ `reggie init` - successful initialization
  - ✅ `reggie load <doc_id>` - validates doc_id format
  - ✅ `reggie list` - displays documents correctly
  - ✅ `reggie process <doc_id>` - checks for OPENAI_API_KEY
  - ✅ Error messages are helpful

**Why these matter:** Poor CLI UX = frustrated users. Error messages especially important.

## What We're NOT Testing (And Why)

**Critical for limited-time MVP:** Focus on high-value tests only.

### ❌ Trivial Code
- **Skip:** Simple getters/setters, `__repr__` methods, property accessors
- **Why:** Low value, high maintenance cost. Time better spent elsewhere.

### ❌ Private Method Exhaustiveness
- **Skip:** Testing `_build_comment_context()` in isolation
- **Why:** It's tested implicitly through public API. Test behavior, not implementation.

### ❌ 100% Coverage Theater
- **Skip:** Every possible parameter combination, every edge case
- **Why:** Diminishing returns. We want ~80% coverage on critical paths, not 100% everywhere.

### ❌ Configuration Loading Details
- **Skip:** Testing every environment variable permutation
- **Why:** Pydantic handles this. Test validation logic, not the framework.

## Test Infrastructure

### Test Organization
```
tests/
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_categorizer.py
│   ├── test_embedder.py
│   └── test_repositories.py
├── integration/
│   ├── test_database.py
│   ├── test_api_client.py
│   └── test_pipeline.py
└── cli/
    └── test_commands.py
```

### Key Fixtures (`conftest.py`)
```python
@pytest.fixture
async def test_db():
    """Provides a clean test database for each test"""
    # Setup test DB, yield connection, teardown

@pytest.fixture
def mock_regulations_api(httpx_mock):
    """Mocks regulations.gov API - ZERO external calls"""
    # Mock document/comment responses with httpx_mock
    return httpx_mock

@pytest.fixture
def mock_openai(mocker):
    """Mocks OpenAI API - ZERO external calls"""
    # Mock ChatOpenAI.ainvoke() and OpenAIEmbeddings.aembed_documents()
    return mocker.patch(...)

@pytest.fixture
def sample_comment_data():
    """Realistic comment data for testing"""
    # Return dict with typical comment structure from regulations.gov
```

### Testing Dependencies
```
pytest>=7.4.0
pytest-asyncio>=0.21.0      # For async test support
pytest-postgresql>=5.0.0    # For test database fixtures
pytest-mock>=3.11.0         # For mocking OpenAI/internal methods
httpx>=0.27.0               # Already installed, needed for httpx_mock
pytest-httpx>=0.30.0        # For mocking httpx HTTP calls (regulations.gov API)
pytest-cov>=4.1.0           # For coverage reporting (optional)
```

## Success Criteria

### For One-Week MVP Interview Project:
1. **~70-80% coverage** of critical paths (quality over quantity)
2. **All tests passing** - proves the system works
3. **Clear test names** - tests serve as documentation
4. **Fast test suite** - under 20 seconds total
5. **Zero external API costs** - all mocked
6. **No flaky tests** - deterministic, reliable

### What This Demonstrates:
- ✅ **Judicious testing** - right tests, not all tests
- ✅ **Async testing competence** (pytest-asyncio)
- ✅ **Database testing skills** (test DB, transactions, cleanup)
- ✅ **API mocking patterns** (httpx-mock, zero costs)
- ✅ **Professional judgment** (what NOT to test is equally important)
- ✅ **Time management** (effective testing in limited time)

## Implementation Order

### Phase 1: Foundation (2-3 hours)
1. Set up test infrastructure (conftest.py, fixtures with mocking)
2. Database integration tests (schema, CRUD)
3. Model validation tests

### Phase 2: Core Components (2-3 hours)
4. API client tests (mocked httpx responses)
5. Embedder unit tests (mocked OpenAI)
6. Categorizer unit tests (mocked OpenAI)

### Phase 3: Integration (1-2 hours)
7. End-to-end pipeline tests (all APIs mocked, real test DB)
8. CLI tests (mocked services)

## Testing Anti-Patterns to Avoid

### ❌ Testing Implementation Details
```python
# BAD: Brittle, breaks when refactoring
def test_categorizer_calls_model_ainvoke():
    assert categorizer.model.ainvoke.called
```

```python
# GOOD: Test behavior
def test_categorizer_returns_classification():
    result = await categorizer.categorize("text")
    assert isinstance(result, CommentClassification)
    assert result.category in Category
```

### ❌ Mocking vs Real Resources
```python
# BAD: Mocking the database in integration tests
mock_db = Mock()
mock_db.execute = AsyncMock()
```

```python
# GOOD: Mock external APIs, use real test DB
async with test_db_connection() as conn:
    # Real DB catches SQL errors, schema issues
    await CommentRepository.store_comment(data, conn)
    # Verify by reading back
```

### ❌ Assertion Roulette
```python
# BAD: Which assertion failed?
assert result.category == "Physicians"
assert result.sentiment == "for"
assert result.reasoning != ""
```

```python
# GOOD: Clear failure messages
assert result.category == "Physicians", f"Expected Physicians, got {result.category}"
assert result.sentiment == "for", f"Expected 'for', got {result.sentiment}"
assert result.reasoning, "Reasoning should not be empty"
```

## Key Testing Principles for This Project

1. **Mock external services, use real local resources**
   - Mock: OpenAI API, regulations.gov API (zero costs, fast, deterministic)
   - Real: Test PostgreSQL database (catches SQL bugs, schema issues)

2. **Test the contract, not the implementation**
   - If we refactor repositories, tests shouldn't break
   - Focus on behavior and outcomes

3. **Realistic test data matters**
   - Use actual comment structures from regulations.gov
   - Edge cases: empty comments, missing metadata, long text

4. **Fast feedback loop**
   - Target: <20 second test suite
   - All external APIs mocked (no network I/O)
   - Parallel test execution where possible

5. **Tests as documentation**
   - Test names explain behavior: `test_upsert_updates_existing_comment_on_conflict`
   - Anyone can understand the system from reading tests

## Measuring Success

### Quantitative Metrics:
- ✅ 70-80% coverage on critical paths (not 100% everywhere)
- ✅ <20 second test suite runtime
- ✅ Zero external API costs
- ✅ Zero flaky tests

### Qualitative Metrics:
- ✅ Tests catch real bugs (not just increase coverage %)
- ✅ Tests enable refactoring with confidence
- ✅ Test names clearly document behavior
- ✅ Interviewer sees strong testing judgment

## Notes for Interview Discussion

**Be prepared to explain:**
- **Mocking strategy:** Why we mock external APIs (cost, speed, determinism) but use real test DB (catch SQL bugs)
- **What NOT to test:** Demonstrating judgment by skipping trivial code
- **Async testing:** pytest-asyncio patterns, handling async context managers
- **Time constraints:** How to test effectively with limited time (this is an MVP)
- **Coverage goals:** Why 70-80% on critical paths > 100% on everything

**This testing approach demonstrates:**
- ✅ Professional judgment about test value vs. cost
- ✅ Async Python testing competence
- ✅ Database testing skills (fixtures, transactions, cleanup)
- ✅ Zero-cost testing (all external APIs mocked)
- ✅ Focus on maintainability and clear documentation
- ✅ Understanding of MVP constraints (time-boxed, high-value tests only)
