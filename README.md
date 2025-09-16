# Embedding Service

A FastAPI-based web service for text embeddings and classification using machine learning models.

## Features

- **Text Embeddings**: Generate vector representations of text using sentence-transformers
- **Text Classification**: Classify text into Engineering/Non-Engineering categories
- **Batch Processing**: Handle multiple texts efficiently in single requests
- **Caching**: LRU cache for improved performance on repeated requests
- **Docker Support**: Containerized deployment ready

## Quick Start

### Using uv (recommended)
```bash
uv sync
python main.py
```

### Using pip
```bash
pip install fastapi numpy sentence-transformers uvicorn
python main.py
```

### Using Docker
```bash
docker build -t embedding-service .
docker run -p 8000:8000 embedding-service
```

## API Endpoints

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### GET /health
Health check endpoint:
```bash
curl http://localhost:8000/health
```

### POST /embed
Generate embedding vector for text:
```bash
curl -X POST "http://localhost:8000/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text here"}'
```

### POST /classify  
Classify text into categories:
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "Machine learning algorithms"}'
```

### POST /classify_batch
Process multiple texts:
```bash
curl -X POST "http://localhost:8000/classify_batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Python programming", "Marketing strategy"]}'
```

## Technology Stack

- **Python 3.13**
- **FastAPI** - Modern web framework
- **sentence-transformers** - Text embedding models
- **numpy** - Numerical computations
- **uvicorn** - ASGI server

## Development

The service uses the `all-MiniLM-L6-v2` model for embeddings and includes caching for performance optimization. See [Copilot Instructions](.copilot-instructions.md) for detailed development guidelines.

## Testing

The project includes comprehensive unit tests with mocking to avoid calling hosted models. Tests run quickly without external dependencies.

### Running Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest test_main.py test_embedding_service.py -v

# Run tests with coverage
pip install pytest-cov
pytest test_main.py test_embedding_service.py --cov=. --cov-report=html
```

### GitHub Actions

Tests are automatically run on every push and pull request via GitHub Actions. The workflow:

- Tests on Python 3.12 and 3.13
- Uses comprehensive mocking to prevent model downloads
- Runs offline to avoid external API calls
- Generates coverage reports

See [TESTING.md](TESTING.md) for detailed testing documentation.
