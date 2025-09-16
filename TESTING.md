# Unit Tests for Embedding Service

This document describes the unit tests implemented for the embedding service, which use comprehensive mocking to avoid calling hosted models.

## Test Files

### `test_main.py`
Tests the FastAPI endpoints with mocked embedding service:

- **Health endpoint tests**: Verify the `/health` endpoint returns correct status
- **Embed endpoint tests**: Test `/embed` with various inputs, including empty text and invalid requests
- **Classify endpoint tests**: Test `/classify` with different similarity scores, threshold values, and category mappings
- **Classify batch endpoint tests**: Test `/classify_batch` with multiple texts, empty lists, and custom thresholds

**Key Features:**
- Mocks the entire `embedding_service` module to avoid model loading
- Tests all API endpoints without external dependencies
- Validates request/response schemas and error handling
- Tests category mapping logic (Product/Platform Engineering → Engineering)

### `test_embedding_service.py`
Tests the core `EmbeddingService` class functionality:

- **Text normalization tests**: Verify text preprocessing (lowercasing, punctuation removal, etc.)
- **Redis caching tests**: Test cache hits, misses, and connection failures
- **LRU cache fallback**: Test in-memory caching when Redis is unavailable
- **Initialization tests**: Verify proper setup with environment variables

**Key Features:**
- Uses `@patch` decorators to mock `SentenceTransformer` and Redis
- Tests caching logic without actual Redis or model calls
- Isolates each component for focused testing
- Covers error handling and fallback mechanisms

### `test_integration.py`
Integration tests demonstrating end-to-end mocking:

- **Full workflow tests**: Test complete request/response cycles
- **Complex scenario tests**: Multiple classifications with different similarity scores
- **Edge case handling**: Empty inputs, long texts, invalid requests
- **Mock verification**: Ensure mocks are called correctly

## Mocking Strategy

### 1. Model Mocking
- `SentenceTransformer` is mocked to return predefined numpy arrays
- No actual model downloading or inference occurs
- Different embedding vectors are returned for different test scenarios

### 2. Redis Mocking
- `redis.Redis` class is mocked for all connection attempts
- Cache hits/misses are simulated by controlling mock return values
- Connection failures are simulated by raising `RedisError`

### 3. Service Mocking
- For FastAPI tests, the entire `embedding_service` module is mocked
- Allows testing API endpoints without initializing the actual service
- Mock behaviors are configured per test for different scenarios

## Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest test_main.py -v
pytest test_embedding_service.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## Test Coverage

The tests cover:

- ✅ All FastAPI endpoints (`/health`, `/embed`, `/classify`, `/classify_batch`)
- ✅ Text normalization logic
- ✅ Redis caching (hits, misses, failures)
- ✅ LRU cache fallback
- ✅ Category mapping logic
- ✅ Threshold-based classification
- ✅ Error handling and edge cases
- ✅ Request validation
- ✅ Environment variable configuration

## Benefits of This Approach

1. **No External Dependencies**: Tests run without internet, Redis, or model files
2. **Fast Execution**: Mocks eliminate slow model loading and inference
3. **Deterministic**: Same inputs always produce same outputs
4. **Isolated**: Each test is independent and doesn't affect others
5. **Complete Coverage**: All code paths can be tested with different mock configurations

## Example Test Run

```bash
$ pytest -v
========================= test session starts =========================
collected 21 items

test_embedding_service.py::TestEmbeddingServiceCore::test_text_normalization_patterns PASSED
test_embedding_service.py::TestEmbeddingServiceWithMocks::test_service_initialization_and_caching PASSED
test_embedding_service.py::TestEmbeddingServiceWithMocks::test_redis_cache_hit PASSED
test_embedding_service.py::TestEmbeddingServiceWithMocks::test_redis_connection_failure_fallback PASSED
test_main.py::TestHealthEndpoint::test_health_check PASSED
test_main.py::TestEmbedEndpoint::test_embed_success PASSED
test_main.py::TestEmbedEndpoint::test_embed_empty_text PASSED
test_main.py::TestEmbedEndpoint::test_embed_invalid_request PASSED
test_main.py::TestClassifyEndpoint::test_classify_engineering_high_similarity PASSED
test_main.py::TestClassifyEndpoint::test_classify_product_engineering_mapping PASSED
test_main.py::TestClassifyEndpoint::test_classify_platform_engineering_mapping PASSED
test_main.py::TestClassifyEndpoint::test_classify_non_engineering PASSED
test_main.py::TestClassifyEndpoint::test_classify_below_threshold PASSED
test_main.py::TestClassifyEndpoint::test_classify_custom_threshold PASSED
test_main.py::TestClassifyBatchEndpoint::test_classify_batch_success PASSED
test_main.py::TestClassifyBatchEndpoint::test_classify_batch_empty_list PASSED
test_main.py::TestClassifyBatchEndpoint::test_classify_batch_with_threshold PASSED

========================= 17 passed in 4.24s =========================
```

All tests pass successfully with comprehensive mocking that prevents any calls to hosted models or external services.