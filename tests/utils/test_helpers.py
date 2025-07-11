"""
Test helper functions and utilities for the rental ML system test suite.
"""

import time
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from unittest.mock import Mock, MagicMock
from contextlib import contextmanager
import tempfile
import os
from uuid import uuid4
from datetime import datetime, timedelta

from domain.entities.property import Property
from domain.entities.user import User, UserPreferences, UserInteraction
from domain.entities.search_query import SearchQuery


class TestDataGenerator:
    """Generate test data for various testing scenarios."""
    
    @staticmethod
    def generate_properties(count: int, seed: int = 42) -> List[Property]:
        """Generate a list of test properties."""
        np.random.seed(seed)
        
        neighborhoods = ['Downtown', 'Mission', 'SoMa', 'Castro', 'Marina', 'Richmond', 'Sunset', 'Haight']
        property_types = ['apartment', 'condo', 'house', 'studio', 'loft']
        amenities_pool = ['parking', 'gym', 'pool', 'rooftop', 'laundry', 'dishwasher', 'balcony', 'garden', 'elevator', 'concierge']
        
        properties = []
        for i in range(count):
            # Generate realistic property data
            bedrooms = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.25, 0.05])
            bathrooms = round(np.random.uniform(1.0, min(bedrooms + 1, 3.5)), 1)
            square_feet = int(np.random.uniform(500, 2500))
            base_price = square_feet * np.random.uniform(1.5, 4.0)
            
            # Add location premium
            location_premium = {
                'Downtown': 1.3, 'Mission': 1.1, 'SoMa': 1.4, 'Castro': 1.2,
                'Marina': 1.5, 'Richmond': 0.9, 'Sunset': 0.8, 'Haight': 1.0
            }
            neighborhood = np.random.choice(neighborhoods)
            price = base_price * location_premium.get(neighborhood, 1.0)
            
            # Generate amenities
            num_amenities = np.random.randint(2, 6)
            amenities = np.random.choice(amenities_pool, size=num_amenities, replace=False).tolist()
            
            property_obj = Property.create(
                title=f"Beautiful {bedrooms}BR {property_types[i % len(property_types)]} in {neighborhood}",
                description=f"Stunning {bedrooms} bedroom, {bathrooms} bathroom {property_types[i % len(property_types)]} "
                          f"with {', '.join(amenities[:3])} and more!",
                price=round(price),
                location=f"{neighborhood}, San Francisco, CA",
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                square_feet=square_feet,
                amenities=amenities,
                contact_info={'phone': f'555-{1000+i:04d}', 'email': f'contact{i}@example.com'},
                images=[f'image{i}_1.jpg', f'image{i}_2.jpg'],
                property_type=property_types[i % len(property_types)]
            )
            properties.append(property_obj)
        
        return properties
    
    @staticmethod
    def generate_users(count: int, seed: int = 42) -> List[User]:
        """Generate a list of test users with preferences."""
        np.random.seed(seed)
        
        users = []
        for i in range(count):
            preferences = UserPreferences(
                min_price=np.random.randint(1000, 2500),
                max_price=np.random.randint(3000, 6000),
                min_bedrooms=np.random.randint(1, 3),
                max_bedrooms=np.random.randint(2, 5),
                preferred_locations=np.random.choice(
                    ['Downtown', 'Mission', 'SoMa', 'Castro', 'Marina'], 
                    size=np.random.randint(1, 4), 
                    replace=False
                ).tolist(),
                required_amenities=np.random.choice(
                    ['parking', 'gym', 'pool', 'laundry'], 
                    size=np.random.randint(0, 3), 
                    replace=False
                ).tolist(),
                property_types=np.random.choice(
                    ['apartment', 'condo', 'house'], 
                    size=np.random.randint(1, 3), 
                    replace=False
                ).tolist()
            )
            
            user = User.create(
                email=f'testuser{i}@example.com',
                preferences=preferences
            )
            users.append(user)
        
        return users
    
    @staticmethod
    def generate_user_interactions(users: List[User], properties: List[Property], 
                                 interaction_rate: float = 0.1, seed: int = 42) -> None:
        """Generate realistic user interactions."""
        np.random.seed(seed)
        
        interaction_types = ['view', 'like', 'inquiry', 'save']
        type_probabilities = [0.6, 0.2, 0.15, 0.05]
        
        for user in users:
            # Each user interacts with a subset of properties
            num_interactions = int(len(properties) * interaction_rate * np.random.uniform(0.5, 2.0))
            interacted_properties = np.random.choice(properties, size=num_interactions, replace=False)
            
            for prop in interacted_properties:
                interaction_type = np.random.choice(interaction_types, p=type_probabilities)
                duration = np.random.randint(30, 300) if interaction_type == 'view' else None
                
                interaction = UserInteraction.create(
                    property_id=prop.id,
                    interaction_type=interaction_type,
                    duration_seconds=duration
                )
                user.add_interaction(interaction)
    
    @staticmethod
    def generate_user_item_matrix(num_users: int, num_items: int, 
                                density: float = 0.1, seed: int = 42) -> np.ndarray:
        """Generate a sparse user-item interaction matrix."""
        np.random.seed(seed)
        
        matrix = np.zeros((num_users, num_items))
        
        # Generate interactions with realistic patterns
        for user_id in range(num_users):
            # Each user has different activity levels
            activity_level = np.random.beta(2, 5)  # Most users are low-activity
            num_interactions = int(num_items * density * activity_level)
            
            if num_interactions > 0:
                # Users tend to interact with similar properties
                preferred_items = np.random.choice(num_items, size=num_interactions, replace=False)
                
                for item_id in preferred_items:
                    # Interaction strength (1-5 rating scale, but we'll use binary)
                    matrix[user_id, item_id] = 1
        
        return matrix
    
    @staticmethod
    def generate_property_features_dict(properties: List[Property]) -> List[Dict]:
        """Convert Property entities to feature dictionaries for ML models."""
        property_features = []
        
        for prop in properties:
            features = {
                'property_id': str(prop.id),
                'neighborhood': prop.location.split(',')[0].strip(),
                'city': 'San Francisco',  # Assuming all properties are in SF
                'price': prop.price,
                'bedrooms': prop.bedrooms,
                'bathrooms': prop.bathrooms,
                'square_feet': prop.square_feet,
                'amenities': prop.amenities,
                'property_type': prop.property_type
            }
            property_features.append(features)
        
        return property_features


class MLTestHelpers:
    """Helper functions for ML model testing."""
    
    @staticmethod
    def assert_valid_predictions(predictions: np.ndarray, expected_shape: tuple = None):
        """Assert that ML predictions are valid."""
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert not np.isnan(predictions).any(), "Predictions should not contain NaN values"
        assert not np.isinf(predictions).any(), "Predictions should not contain infinite values"
        
        if expected_shape:
            assert predictions.shape == expected_shape, f"Expected shape {expected_shape}, got {predictions.shape}"
    
    @staticmethod
    def assert_recommendations_quality(recommendations: List, min_count: int = 1, max_rating: float = 1.0):
        """Assert that recommendations meet quality criteria."""
        assert len(recommendations) >= min_count, f"Should have at least {min_count} recommendations"
        
        for i, rec in enumerate(recommendations):
            assert hasattr(rec, 'item_id'), f"Recommendation {i} missing item_id"
            assert hasattr(rec, 'predicted_rating'), f"Recommendation {i} missing predicted_rating"
            assert hasattr(rec, 'confidence_score'), f"Recommendation {i} missing confidence_score"
            
            assert 0 <= rec.predicted_rating <= max_rating, f"Rating {rec.predicted_rating} out of range [0, {max_rating}]"
            assert 0 <= rec.confidence_score <= 1.0, f"Confidence {rec.confidence_score} out of range [0, 1]"
        
        # Check if recommendations are sorted by rating (descending)
        ratings = [rec.predicted_rating for rec in recommendations]
        assert ratings == sorted(ratings, reverse=True), "Recommendations should be sorted by rating (descending)"
    
    @staticmethod
    def create_mock_ml_model(model_type: str = "content_based"):
        """Create a mock ML model for testing."""
        mock_model = Mock()
        mock_model.model_type = model_type
        mock_model.is_trained = True
        mock_model.predict = Mock(return_value=np.array([0.8, 0.6, 0.7]))
        mock_model.fit = Mock(return_value={'loss': 0.1, 'accuracy': 0.95})
        mock_model.save_model = Mock()
        mock_model.load_model = Mock()
        mock_model.get_model_info = Mock(return_value={'model_type': model_type, 'is_trained': True})
        
        return mock_model


class APITestHelpers:
    """Helper functions for API testing."""
    
    @staticmethod
    def assert_response_structure(response_data: Dict, required_fields: List[str]):
        """Assert that API response has required structure."""
        for field in required_fields:
            assert field in response_data, f"Response missing required field: {field}"
    
    @staticmethod
    def assert_pagination_response(response_data: Dict):
        """Assert that paginated response has correct structure."""
        required_fields = ['items', 'total', 'page', 'per_page', 'pages']
        APITestHelpers.assert_response_structure(response_data, required_fields)
        
        assert isinstance(response_data['items'], list), "Items should be a list"
        assert isinstance(response_data['total'], int), "Total should be an integer"
        assert response_data['total'] >= 0, "Total should be non-negative"
    
    @staticmethod
    def create_mock_request():
        """Create a mock FastAPI request object."""
        mock_request = Mock()
        mock_request.app = Mock()
        mock_request.app.state = Mock()
        mock_request.app.state.repository_factory = Mock()
        return mock_request


class DatabaseTestHelpers:
    """Helper functions for database testing."""
    
    @staticmethod
    def create_test_tables(engine):
        """Create test database tables."""
        # This would create the actual database schema
        # For now, we'll use mocks
        pass
    
    @staticmethod
    def cleanup_test_data(session):
        """Clean up test data from database."""
        # This would clean up test data
        # For now, we'll use transaction rollback in conftest.py
        pass


class PerformanceTestHelpers:
    """Helper functions for performance testing."""
    
    @staticmethod
    @contextmanager
    def measure_time():
        """Context manager to measure execution time."""
        start_time = time.time()
        yield lambda: time.time() - start_time
    
    @staticmethod
    def assert_performance_threshold(elapsed_time: float, threshold: float, operation: str):
        """Assert that operation completed within time threshold."""
        assert elapsed_time <= threshold, (
            f"{operation} took {elapsed_time:.3f}s, "
            f"expected <= {threshold:.3f}s"
        )
    
    @staticmethod
    def generate_load_test_data(num_requests: int = 100):
        """Generate data for load testing."""
        return [
            {
                'user_id': np.random.randint(0, 100),
                'num_recommendations': np.random.randint(5, 20),
                'timestamp': datetime.now() + timedelta(milliseconds=i*10)
            }
            for i in range(num_requests)
        ]


class MockHelpers:
    """Helper functions for creating mocks."""
    
    @staticmethod
    def create_mock_repository(repo_type: str):
        """Create a mock repository."""
        mock_repo = Mock()
        
        if repo_type == "property":
            mock_repo.get_by_id = Mock()
            mock_repo.get_all_active = Mock(return_value=[])
            mock_repo.get_by_location = Mock(return_value=[])
            mock_repo.get_by_price_range = Mock(return_value=[])
            mock_repo.save = Mock()
            mock_repo.delete = Mock()
            mock_repo.get_count = Mock(return_value=0)
            mock_repo.get_active_count = Mock(return_value=0)
        
        elif repo_type == "user":
            mock_repo.get_by_id = Mock()
            mock_repo.get_by_email = Mock()
            mock_repo.save = Mock()
            mock_repo.delete = Mock()
            mock_repo.get_user_interactions = Mock(return_value=[])
        
        elif repo_type == "model":
            mock_repo.save_model = Mock()
            mock_repo.load_model = Mock()
            mock_repo.get_model_metadata = Mock()
        
        return mock_repo
    
    @staticmethod
    def create_mock_cache():
        """Create a mock cache (Redis)."""
        mock_cache = Mock()
        mock_cache.get = Mock(return_value=None)
        mock_cache.set = Mock(return_value=True)
        mock_cache.delete = Mock(return_value=True)
        mock_cache.exists = Mock(return_value=False)
        mock_cache.expire = Mock(return_value=True)
        return mock_cache


# Decorator for async tests
def async_test(timeout: float = 30.0):
    """Decorator for async test functions with timeout."""
    def decorator(test_func):
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(test_func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


# Custom pytest markers
def slow_test(func):
    """Mark test as slow."""
    import pytest
    return pytest.mark.slow(func)


def ml_test(func):
    """Mark test as ML-related."""
    import pytest
    return pytest.mark.ml(func)


def integration_test(func):
    """Mark test as integration test."""
    import pytest
    return pytest.mark.integration(func)


def performance_test(func):
    """Mark test as performance test."""
    import pytest
    return pytest.mark.performance(pytest.mark.slow(func))


# Test data constants
TEST_CONSTANTS = {
    'DEFAULT_PROPERTY_COUNT': 10,
    'DEFAULT_USER_COUNT': 5,
    'DEFAULT_INTERACTION_RATE': 0.1,
    'ML_MODEL_TIMEOUT': 60.0,
    'API_TIMEOUT': 5.0,
    'DB_TIMEOUT': 10.0,
    'PERFORMANCE_THRESHOLDS': {
        'ml_prediction': 1.0,  # seconds
        'db_query': 0.5,       # seconds
        'api_response': 2.0,   # seconds
        'model_training': 30.0  # seconds
    }
}