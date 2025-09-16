"""
Unit tests for the embedding service FastAPI endpoints.
Tests use mocks to avoid calling actual hosted models.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import sys

# Mock the embedding service module before any imports
mock_embedding_service = Mock()
mock_embedding_service.embed_or_cache.return_value = np.array([0.1, 0.2, 0.3])
mock_embedding_service.anchor_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]

# Mock the entire embedding_service module
sys.modules['embedding_service'] = Mock(embedding_service=mock_embedding_service)

# Now we can safely import main
from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "embedding-service"}


class TestEmbedEndpoint:
    """Test the embed endpoint with mocked embedding service."""
    
    @patch('main.embedding_service')
    def test_embed_success(self, mock_embedding_service):
        """Test successful embedding generation."""
        # Mock the embedding result
        mock_vector = np.array([0.1, 0.2, 0.3, 0.4])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        response = client.post("/embed", json={"text": "test text"})
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert data["embedding"] == [0.1, 0.2, 0.3, 0.4]
        mock_embedding_service.embed_or_cache.assert_called_once_with("test text")
    
    @patch('main.embedding_service')
    def test_embed_empty_text(self, mock_embedding_service):
        """Test embedding with empty text."""
        mock_vector = np.array([0.0, 0.0, 0.0, 0.0])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        response = client.post("/embed", json={"text": ""})
        
        assert response.status_code == 200
        data = response.json()
        assert data["embedding"] == [0.0, 0.0, 0.0, 0.0]
        mock_embedding_service.embed_or_cache.assert_called_once_with("")
    
    def test_embed_invalid_request(self):
        """Test embedding with invalid request body."""
        response = client.post("/embed", json={"invalid": "field"})
        assert response.status_code == 422  # Validation error


class TestClassifyEndpoint:
    """Test the classify endpoint with mocked embedding service."""
    
    @patch('main.embedding_service')
    def test_classify_engineering_high_similarity(self, mock_embedding_service):
        """Test classification as Engineering with high similarity."""
        # Mock embedding and anchor vectors
        mock_vector = np.array([1.0, 0.0, 0.0, 0.0])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        # Mock anchor vectors - first one should have highest similarity
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Engineering
            [0.0, 1.0, 0.0, 0.0],  # Product Engineering
            [0.0, 0.0, 1.0, 0.0],  # Platform Engineering
            [0.0, 0.0, 0.0, 1.0],  # Non-Engineering
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        response = client.post("/classify", json={"text": "Python programming"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Engineering"
        assert data["similarity"] == 1.0
        mock_embedding_service.embed_or_cache.assert_called_once_with("Python programming")
    
    @patch('main.embedding_service')
    def test_classify_product_engineering_mapping(self, mock_embedding_service):
        """Test that Product Engineering gets mapped to Engineering."""
        mock_vector = np.array([0.0, 1.0, 0.0, 0.0])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Engineering
            [0.0, 1.0, 0.0, 0.0],  # Product Engineering - this should match
            [0.0, 0.0, 1.0, 0.0],  # Platform Engineering
            [0.0, 0.0, 0.0, 1.0],  # Non-Engineering
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        response = client.post("/classify", json={"text": "Product development"})
        
        assert response.status_code == 200
        data = response.json()
        # Product Engineering should be mapped to Engineering
        assert data["category"] == "Engineering"
        assert data["similarity"] == 1.0
    
    @patch('main.embedding_service')
    def test_classify_platform_engineering_mapping(self, mock_embedding_service):
        """Test that Platform Engineering gets mapped to Engineering."""
        mock_vector = np.array([0.0, 0.0, 1.0, 0.0])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Engineering
            [0.0, 1.0, 0.0, 0.0],  # Product Engineering
            [0.0, 0.0, 1.0, 0.0],  # Platform Engineering - this should match
            [0.0, 0.0, 0.0, 1.0],  # Non-Engineering
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        response = client.post("/classify", json={"text": "Infrastructure setup"})
        
        assert response.status_code == 200
        data = response.json()
        # Platform Engineering should be mapped to Engineering
        assert data["category"] == "Engineering"
        assert data["similarity"] == 1.0
    
    @patch('main.embedding_service')
    def test_classify_non_engineering(self, mock_embedding_service):
        """Test classification as Non-Engineering."""
        mock_vector = np.array([0.0, 0.0, 0.0, 1.0])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Engineering
            [0.0, 1.0, 0.0, 0.0],  # Product Engineering
            [0.0, 0.0, 1.0, 0.0],  # Platform Engineering
            [0.0, 0.0, 0.0, 1.0],  # Non-Engineering - this should match
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        response = client.post("/classify", json={"text": "Marketing strategy"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Non-Engineering"
        assert data["similarity"] == 1.0
    
    @patch('main.embedding_service')
    def test_classify_below_threshold(self, mock_embedding_service):
        """Test classification below threshold returns Unsure."""
        mock_vector = np.array([0.1, 0.1, 0.1, 0.1])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        # All similarities will be 0.4, below default threshold of 0.7
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 1.0, 1.0, 1.0],  # Engineering
            [1.0, 1.0, 1.0, 1.0],  # Product Engineering
            [1.0, 1.0, 1.0, 1.0],  # Platform Engineering
            [1.0, 1.0, 1.0, 1.0],  # Non-Engineering
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        response = client.post("/classify", json={"text": "ambiguous text"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Unsure"
        assert data["similarity"] == 0.4
    
    @patch('main.embedding_service')  
    def test_classify_custom_threshold(self, mock_embedding_service):
        """Test classification with custom threshold."""
        mock_vector = np.array([0.5, 0.0, 0.0, 0.0])
        mock_embedding_service.embed_or_cache.return_value = mock_vector
        
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Engineering - similarity will be 0.5
            [0.0, 1.0, 0.0, 0.0],  # Product Engineering
            [0.0, 0.0, 1.0, 0.0],  # Platform Engineering
            [0.0, 0.0, 0.0, 1.0],  # Non-Engineering
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        # With threshold 0.3, should classify as Engineering
        response = client.post("/classify?threshold=0.3", json={"text": "coding"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Engineering"
        assert data["similarity"] == 0.5


class TestClassifyBatchEndpoint:
    """Test the classify_batch endpoint with mocked embedding service."""
    
    @patch('main.embedding_service')
    def test_classify_batch_success(self, mock_embedding_service):
        """Test successful batch classification."""
        # Mock embeddings for two texts
        mock_vectors = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # First text - Engineering
            np.array([0.0, 0.0, 0.0, 1.0]),  # Second text - Non-Engineering
        ]
        mock_embedding_service.embed_or_cache.side_effect = mock_vectors
        
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Engineering
            [0.0, 1.0, 0.0, 0.0],  # Product Engineering
            [0.0, 0.0, 1.0, 0.0],  # Platform Engineering
            [0.0, 0.0, 0.0, 1.0],  # Non-Engineering
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        texts = ["Python programming", "Marketing strategy"]
        response = client.post("/classify_batch", json={"texts": texts})
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        results = data["results"]
        assert len(results) == 2
        
        # First result
        assert results[0]["text"] == "Python programming"
        assert results[0]["category"] == "Engineering"
        assert results[0]["similarity"] == 1.0
        
        # Second result  
        assert results[1]["text"] == "Marketing strategy"
        assert results[1]["category"] == "Non-Engineering"
        assert results[1]["similarity"] == 1.0
        
        assert mock_embedding_service.embed_or_cache.call_count == 2
    
    @patch('main.embedding_service')
    def test_classify_batch_empty_list(self, mock_embedding_service):
        """Test batch classification with empty list."""
        response = client.post("/classify_batch", json={"texts": []})
        
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        mock_embedding_service.embed_or_cache.assert_not_called()
    
    @patch('main.embedding_service')
    def test_classify_batch_with_threshold(self, mock_embedding_service):
        """Test batch classification with custom threshold."""
        mock_vectors = [
            np.array([0.3, 0.0, 0.0, 0.0]),  # Low similarity
        ]
        mock_embedding_service.embed_or_cache.side_effect = mock_vectors
        
        mock_embedding_service.anchor_vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Engineering - similarity will be 0.3
            [0.0, 1.0, 0.0, 0.0],  # Product Engineering
            [0.0, 0.0, 1.0, 0.0],  # Platform Engineering  
            [0.0, 0.0, 0.0, 1.0],  # Non-Engineering
        ])
        mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]
        
        # With threshold 0.5, should return Unsure
        response = client.post("/classify_batch?threshold=0.5", json={"texts": ["ambiguous text"]})
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        assert len(results) == 1
        assert results[0]["category"] == "Unsure"
        assert results[0]["similarity"] == 0.3