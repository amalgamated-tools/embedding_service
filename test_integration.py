"""
Additional integration tests to demonstrate the full mocking capability.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
import sys

# Set up comprehensive mocking for integration test
mock_embedding_service = Mock()

# Configure the mock embedding service with realistic behavior
mock_embedding_service.embed_or_cache.return_value = np.array([0.1, 0.2, 0.3, 0.4])
mock_embedding_service.anchor_vecs = np.array([
    [1.0, 0.0, 0.0, 0.0],  # Engineering - high similarity 
    [0.0, 1.0, 0.0, 0.0],  # Product Engineering
    [0.0, 0.0, 1.0, 0.0],  # Platform Engineering
    [0.0, 0.0, 0.0, 1.0],  # Non-Engineering
])
mock_embedding_service.anchors = ["Engineering", "Product Engineering", "Platform Engineering", "Non-Engineering"]

# Mock the module
sys.modules['embedding_service'] = Mock(embedding_service=mock_embedding_service)

# Now import main after mocking
from main import app

client = TestClient(app)


class TestIntegrationMocking:
    """Integration tests that verify the complete mocking setup works."""
    
    def test_full_workflow_mock_verification(self):
        """Test that demonstrates the complete mocked workflow."""
        
        # Configure mock for a specific scenario
        input_text = "Python machine learning algorithm"
        expected_embedding = np.array([0.8, 0.6, 0.4, 0.2])
        mock_embedding_service.embed_or_cache.return_value = expected_embedding
        
        # Test embed endpoint
        response = client.post("/embed", json={"text": input_text})
        assert response.status_code == 200
        
        result = response.json()
        assert "embedding" in result
        assert result["embedding"] == expected_embedding.tolist()
        
        # Verify the mock was called correctly
        mock_embedding_service.embed_or_cache.assert_called_with(input_text)
    
    def test_classification_with_different_similarity_scores(self):
        """Test classification with different similarity scores."""
        
        # Test high similarity to Engineering
        high_similarity_vector = np.array([0.9, 0.1, 0.1, 0.1])  # Close to Engineering anchor
        mock_embedding_service.embed_or_cache.return_value = high_similarity_vector
        
        response = client.post("/classify", json={"text": "Software development"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Engineering"
        assert result["similarity"] == 0.9  # np.dot([1,0,0,0], [0.9,0.1,0.1,0.1]) = 0.9
        
        # Test low similarity (below threshold)
        low_similarity_vector = np.array([0.3, 0.3, 0.3, 0.3])  # Low similarity to all anchors
        mock_embedding_service.embed_or_cache.return_value = low_similarity_vector
        
        response = client.post("/classify", json={"text": "Ambiguous content"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Unsure"  # Below default threshold of 0.7
        assert result["similarity"] == 0.3
    
    def test_batch_classification_mock(self):
        """Test batch classification with mocked responses."""
        
        # Set up mock to return different embeddings for different texts
        def side_effect_embeddings(text):
            if "programming" in text.lower():
                return np.array([1.0, 0.0, 0.0, 0.0])  # Engineering
            elif "marketing" in text.lower():
                return np.array([0.0, 0.0, 0.0, 1.0])  # Non-Engineering
            else:
                return np.array([0.5, 0.5, 0.5, 0.5])  # Neutral
        
        mock_embedding_service.embed_or_cache.side_effect = side_effect_embeddings
        
        texts = ["Python programming", "Marketing campaign", "General text"]
        response = client.post("/classify_batch", json={"texts": texts})
        assert response.status_code == 200
        
        result = response.json()
        assert "results" in result
        results = result["results"]
        assert len(results) == 3
        
        # Verify classifications
        assert results[0]["category"] == "Engineering"
        assert results[0]["similarity"] == 1.0
        
        assert results[1]["category"] == "Non-Engineering"
        assert results[1]["similarity"] == 1.0
        
        # Third result should be below threshold
        assert results[2]["similarity"] == 0.5
        # Could be any category since all have same similarity, but should be below threshold for "Unsure"
    
    def test_product_and_platform_engineering_mapping(self):
        """Test that Product and Platform Engineering map to Engineering."""
        
        # Reset mock state
        mock_embedding_service.reset_mock()
        
        # Test Product Engineering mapping
        product_eng_vector = np.array([0.0, 1.0, 0.0, 0.0])  # High similarity to Product Engineering
        mock_embedding_service.embed_or_cache.return_value = product_eng_vector
        
        response = client.post("/classify", json={"text": "Product development strategy"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Engineering"  # Should be mapped from Product Engineering
        assert result["similarity"] == 1.0
        
        # Test Platform Engineering mapping
        platform_eng_vector = np.array([0.0, 0.0, 1.0, 0.0])  # High similarity to Platform Engineering
        mock_embedding_service.embed_or_cache.return_value = platform_eng_vector
        
        response = client.post("/classify", json={"text": "Infrastructure automation"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Engineering"  # Should be mapped from Platform Engineering
        assert result["similarity"] == 1.0
    
    def test_custom_threshold_behavior(self):
        """Test classification with custom threshold values."""
        
        # Reset mock state
        mock_embedding_service.reset_mock()
        
        # Set up a vector with moderate similarity
        moderate_vector = np.array([0.6, 0.0, 0.0, 0.0])  # 0.6 similarity to Engineering
        mock_embedding_service.embed_or_cache.return_value = moderate_vector
        
        # Test with default threshold (0.7) - should be "Unsure"
        response = client.post("/classify", json={"text": "Moderate similarity text"})
        result = response.json()
        assert result["category"] == "Unsure"
        assert result["similarity"] == 0.6
        
        # Test with lower threshold (0.5) - should classify as Engineering
        response = client.post("/classify?threshold=0.5", json={"text": "Moderate similarity text"})
        result = response.json()
        assert result["category"] == "Engineering"
        assert result["similarity"] == 0.6
        
        # Test with higher threshold (0.8) - should still be "Unsure"
        response = client.post("/classify?threshold=0.8", json={"text": "Moderate similarity text"})
        result = response.json()
        assert result["category"] == "Unsure"
        assert result["similarity"] == 0.6


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases with mocks."""
    
    def test_empty_and_edge_case_inputs(self):
        """Test handling of empty and edge case inputs."""
        
        # Reset mock state
        mock_embedding_service.reset_mock()
        
        # Test empty text
        empty_vector = np.array([0.0, 0.0, 0.0, 0.0])
        mock_embedding_service.embed_or_cache.return_value = empty_vector
        
        response = client.post("/embed", json={"text": ""})
        assert response.status_code == 200
        assert response.json()["embedding"] == [0.0, 0.0, 0.0, 0.0]
        
        # Test very long text (mock should handle it)
        long_text = "A" * 1000
        long_text_vector = np.array([0.5, 0.5, 0.5, 0.5])
        mock_embedding_service.embed_or_cache.return_value = long_text_vector
        
        response = client.post("/classify", json={"text": long_text})
        assert response.status_code == 200
        result = response.json()
        assert "category" in result
        assert "similarity" in result
    
    def test_batch_with_empty_list(self):
        """Test batch processing with empty list."""
        # Reset mock to clean state
        mock_embedding_service.reset_mock()
        
        response = client.post("/classify_batch", json={"texts": []})
        assert response.status_code == 200
        
        result = response.json()
        assert result["results"] == []
        
        # Verify mock was not called for empty list
        mock_embedding_service.embed_or_cache.assert_not_called()
    
    def test_invalid_request_bodies(self):
        """Test handling of invalid request bodies."""
        
        # Test missing required field
        response = client.post("/embed", json={"wrong_field": "value"})
        assert response.status_code == 422  # Validation error
        
        response = client.post("/classify", json={"not_text": "value"})
        assert response.status_code == 422  # Validation error
        
        # Test invalid JSON
        response = client.post("/embed", data="not json")
        assert response.status_code == 422
    
    def test_mock_service_call_verification(self):
        """Verify that our mocks are actually being called as expected."""
        
        # Reset mock to track calls
        mock_embedding_service.reset_mock()
        
        # Make several API calls
        test_texts = ["text1", "text2", "text3"]
        
        for text in test_texts:
            mock_embedding_service.embed_or_cache.return_value = np.array([0.7, 0.3, 0.2, 0.1])
            response = client.post("/embed", json={"text": text})
            assert response.status_code == 200
        
        # Verify mock was called correct number of times
        assert mock_embedding_service.embed_or_cache.call_count == len(test_texts)
        
        # Verify it was called with correct arguments
        call_args = [call.args[0] for call in mock_embedding_service.embed_or_cache.call_args_list]
        assert call_args == test_texts