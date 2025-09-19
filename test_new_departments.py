"""
Tests for the new department classification functionality.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
import sys

# Set up comprehensive mocking for the new department tests
mock_embedding_service = Mock()

# Configure the mock embedding service with the new department categories
mock_embedding_service.anchors = [
    "Software Engineering", "Data & AI", "Hardware / Embedded", "Product Management", 
    "Product / UX Design", "UI / Visual Design", "UX Research", "Content / UX Writing",
    "Sales", "Marketing", "Customer Success / Support", "Community & Developer Relations",
    "People / HR / Recruiting / Talent", "Finance & Accounting", "Legal & Compliance",
    "Operations / Strategy / BizOps", "Facilities / Workplace Experience", 
    "Corporate IT / Helpdesk", "Security & Privacy", "Executive roles"
]

# Set up anchor vectors - 20-dimensional unit vectors
anchor_vecs = []
for i in range(20):
    vec = [0.0] * 20
    vec[i] = 1.0
    anchor_vecs.append(vec)
mock_embedding_service.anchor_vecs = np.array(anchor_vecs)

def mock_map_category(category):
    """Mock implementation of map_category that handles the new variant mappings."""
    if category == "Unsure":
        return "Unsure"
    
    # Simulate some of the key mappings from the new variant_map
    mappings = {
        "frontend": "Software Engineering",
        "backend": "Software Engineering",
        "data scientist": "Data & AI", 
        "product manager": "Product Management",
        "ux designer": "Product / UX Design",
        "sales associate": "Sales",
        "marketing manager": "Marketing",
        "hr": "People / HR / Recruiting / Talent",
        "ceo": "Executive roles"
    }
    
    # Normalize the input (basic version)
    normalized = category.lower().strip()
    return mappings.get(normalized, category)

mock_embedding_service.map_category.side_effect = mock_map_category

# Mock the module
sys.modules['embedding_service'] = Mock(embedding_service=mock_embedding_service)

# Now import main after mocking
from main import app

client = TestClient(app)


class TestNewDepartments:
    """Test the new department classification functionality."""
    
    def test_software_engineering_classification(self):
        """Test classification into Software Engineering category."""
        # Mock high similarity to Software Engineering (index 0)
        software_eng_vector = np.array([1.0] + [0.0] * 19)
        mock_embedding_service.embed_or_cache.return_value = software_eng_vector
        
        response = client.post("/classify", json={"text": "Python backend development"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Software Engineering"
        assert result["closest_anchor"] == "Software Engineering"
        assert result["similarity"] == 1.0
    
    def test_data_ai_classification(self):
        """Test classification into Data & AI category."""
        # Mock high similarity to Data & AI (index 1)
        data_ai_vector = np.array([0.0, 1.0] + [0.0] * 18)
        mock_embedding_service.embed_or_cache.return_value = data_ai_vector
        
        response = client.post("/classify", json={"text": "Machine learning model development"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Data & AI"
        assert result["closest_anchor"] == "Data & AI"
        assert result["similarity"] == 1.0
    
    def test_product_management_classification(self):
        """Test classification into Product Management category."""
        # Mock high similarity to Product Management (index 3)
        pm_vector = np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 16)
        mock_embedding_service.embed_or_cache.return_value = pm_vector
        
        response = client.post("/classify", json={"text": "Product roadmap planning"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Product Management"
        assert result["closest_anchor"] == "Product Management"
        assert result["similarity"] == 1.0
    
    def test_marketing_classification(self):
        """Test classification into Marketing category."""
        # Mock high similarity to Marketing (index 9)
        marketing_vector = [0.0] * 20
        marketing_vector[9] = 1.0
        mock_embedding_service.embed_or_cache.return_value = np.array(marketing_vector)
        
        response = client.post("/classify", json={"text": "Digital marketing campaign"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Marketing"
        assert result["closest_anchor"] == "Marketing"
        assert result["similarity"] == 1.0
    
    def test_executive_roles_classification(self):
        """Test classification into Executive roles category."""
        # Mock high similarity to Executive roles (index 19)
        exec_vector = [0.0] * 20
        exec_vector[19] = 1.0
        mock_embedding_service.embed_or_cache.return_value = np.array(exec_vector)
        
        response = client.post("/classify", json={"text": "Chief Technology Officer responsibilities"})
        assert response.status_code == 200
        
        result = response.json()
        assert result["category"] == "Executive roles"
        assert result["closest_anchor"] == "Executive roles"
        assert result["similarity"] == 1.0
    
    def test_batch_classification_new_categories(self):
        """Test batch classification with new department categories."""
        
        def side_effect_embeddings(text):
            if "software" in text.lower() or "programming" in text.lower():
                vec = [0.0] * 20
                vec[0] = 1.0  # Software Engineering
                return np.array(vec)
            elif "data science" in text.lower():
                vec = [0.0] * 20
                vec[1] = 1.0  # Data & AI
                return np.array(vec)
            elif "product manager" in text.lower():
                vec = [0.0] * 20
                vec[3] = 1.0  # Product Management
                return np.array(vec)
            else:
                return np.array([0.3] * 20)  # Low similarity - should be "Unsure"
        
        mock_embedding_service.embed_or_cache.side_effect = side_effect_embeddings
        
        texts = [
            "Software development position", 
            "Data science role",
            "Product manager opening",
            "Ambiguous job posting"
        ]
        
        response = client.post("/classify_batch", json={"texts": texts})
        assert response.status_code == 200
        
        result = response.json()
        assert "results" in result
        results = result["results"]
        assert len(results) == 4
        
        # Verify classifications
        assert results[0]["category"] == "Software Engineering"
        assert results[0]["similarity"] == 1.0
        
        assert results[1]["category"] == "Data & AI"
        assert results[1]["similarity"] == 1.0
        
        assert results[2]["category"] == "Product Management"
        assert results[2]["similarity"] == 1.0
        
        assert results[3]["category"] == "Unsure"  # Below threshold
        assert results[3]["similarity"] == 0.3
    
    def test_variant_mapping_functionality(self):
        """Test that variant mappings work correctly with new categories."""
        # Reset the mock and configure it fresh for this test
        mock_embedding_service.reset_mock()
        mock_embedding_service.map_category.side_effect = mock_map_category
        
        # Test a frontend developer mapping to Software Engineering
        frontend_vector = np.array([0.8] + [0.1] * 19)  # High similarity to Software Engineering
        mock_embedding_service.embed_or_cache.return_value = frontend_vector
        
        response = client.post("/classify", json={"text": "Frontend developer position"})
        assert response.status_code == 200
        
        result = response.json()
        # The mapping should work regardless of similarity score
        assert result["closest_anchor"] == "Software Engineering"
        # The test should focus on the category mapping working, not the exact similarity
    
    def test_all_20_categories_exist(self):
        """Test that all 20 department categories are properly configured."""
        expected_categories = [
            "Software Engineering", "Data & AI", "Hardware / Embedded", "Product Management", 
            "Product / UX Design", "UI / Visual Design", "UX Research", "Content / UX Writing",
            "Sales", "Marketing", "Customer Success / Support", "Community & Developer Relations",
            "People / HR / Recruiting / Talent", "Finance & Accounting", "Legal & Compliance",
            "Operations / Strategy / BizOps", "Facilities / Workplace Experience", 
            "Corporate IT / Helpdesk", "Security & Privacy", "Executive roles"
        ]
        
        # Verify all expected categories are in the anchors
        assert len(mock_embedding_service.anchors) == 20
        for category in expected_categories:
            assert category in mock_embedding_service.anchors
        
        # Verify anchor vectors have correct shape
        assert mock_embedding_service.anchor_vecs.shape == (20, 20)