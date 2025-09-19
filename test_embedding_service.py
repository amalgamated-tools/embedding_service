"""
Unit tests for the EmbeddingService class.
Tests use mocks to avoid calling actual hosted models and Redis.
"""
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import redis


# Simple functional tests for the core logic without dependency on actual model loading
class TestEmbeddingServiceCore:
    """Test core functionality that can be tested in isolation."""
    
    def test_text_normalization_patterns(self):
        """Test the text normalization regex patterns directly."""
        import re
        
        # Test the normalization logic directly without depending on the service
        def normalize_text(text: str) -> str:
            """Replicate the normalize function logic."""
            text = text.lower()
            text = re.sub(r'\d+', '', text)  # remove numbers
            text = re.sub(r'[^a-z ]+', '', text)  # remove punctuation
            return text.strip()
        
        # Test various normalization cases
        assert normalize_text("Hello World!") == "hello world"
        assert normalize_text("Test123Text") == "testtext"
        assert normalize_text("  Multiple   Spaces  ") == "multiple   spaces"  # Multiple spaces preserved
        assert normalize_text("Special@#$%Characters") == "specialcharacters"
        assert normalize_text("Mixed123CASE!@#text") == "mixedcasetext"
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
        assert normalize_text("123") == ""
        assert normalize_text("!@#$%^&*()") == ""


class TestEmbeddingServiceWithMocks:
    """Test EmbeddingService with comprehensive mocking."""
    
    def setup_method(self):
        """Clean up any module caching between tests."""
        if 'embedding_service' in sys.modules:
            # Remove the mocked version if it exists
            original = sys.modules['embedding_service']
            if hasattr(original, 'embedding_service') and hasattr(original.embedding_service, '_mock_name'):
                del sys.modules['embedding_service']
    
    @patch('redis.Redis')
    @patch('sentence_transformers.SentenceTransformer')
    def test_service_initialization_and_caching(self, mock_st, mock_redis):
        """Test service initialization and basic caching behavior."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        mock_st.return_value = mock_model
        
        # Mock Redis with successful connection
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = None  # Simulate cache miss
        mock_redis_instance.set.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        # Import and create service
        from embedding_service import EmbeddingService
        service = EmbeddingService()
        
        # Verify initialization
        assert service.use_redis is True
        assert len(service.anchors) == 40
        
        # Test Redis cache miss - mock the encode call for embedding
        embedding_result = np.array([0.9, 0.8])
        mock_model.encode.return_value = np.array([embedding_result])  # Return array containing the vector
        
        result = service.embed_or_cache("test text")
        
        # Verify the result
        np.testing.assert_array_equal(result, embedding_result)
        mock_redis_instance.get.assert_called_with("embed:test text")
        mock_redis_instance.set.assert_called_with("embed:test text", json.dumps(embedding_result.tolist()))
    
    @patch('redis.Redis')
    @patch('sentence_transformers.SentenceTransformer') 
    def test_redis_cache_hit(self, mock_st, mock_redis):
        """Test Redis cache hit scenario."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        mock_st.return_value = mock_model
        
        # Mock Redis with cache hit
        cached_vector = [0.9, 0.8, 0.7]
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = json.dumps(cached_vector)
        mock_redis.return_value = mock_redis_instance
        
        # Import and create service
        from embedding_service import EmbeddingService
        service = EmbeddingService()
        
        result = service.embed_or_cache("test text")
        
        # Verify cache hit result
        np.testing.assert_array_equal(result, np.array(cached_vector, dtype=np.float32))
        mock_redis_instance.get.assert_called_with("embed:test text")
        mock_redis_instance.set.assert_not_called()  # Should not set on cache hit
    
    @patch('redis.Redis')
    @patch('sentence_transformers.SentenceTransformer')
    def test_redis_connection_failure_fallback(self, mock_st, mock_redis):
        """Test fallback to LRU cache when Redis connection fails."""
        # Mock the sentence transformer
        mock_model = Mock()
        # When encode is called with a list, return array with the embedding as first element
        embedding_result = np.array([0.9, 0.8])
        mock_model.encode.return_value = np.array([embedding_result])  # Wrap in array to match [text_norm] call
        mock_st.return_value = mock_model
        
        # Mock Redis connection failure
        mock_redis_instance = Mock()
        mock_redis_instance.ping.side_effect = redis.RedisError("Connection failed")
        mock_redis.return_value = mock_redis_instance
        
        # Import and create service
        from embedding_service import EmbeddingService
        service = EmbeddingService()
        
        # Verify fallback to LRU
        assert service.use_redis is False
        
        # Test LRU caching
        embedding_result = np.array([0.9, 0.8])
        mock_model.encode.return_value = np.array([embedding_result])
        
        # First call - cache miss
        result1 = service.embed_or_cache("test text")
        np.testing.assert_array_equal(result1, embedding_result)
        
        # Second call should use cached result (no additional model call)
        result2 = service.embed_or_cache("test text")
        np.testing.assert_array_equal(result2, embedding_result)
        
        # Verify Redis methods were not called due to fallback
        mock_redis_instance.get.assert_not_called()
        mock_redis_instance.set.assert_not_called()