"""
Global pytest configuration and fixtures for the rental ML system test suite.

This module provides test configuration, database setup, ML model fixtures,
and shared utilities for all test modules.
"""

import asyncio
import os
import pytest
import tempfile
import shutil
import sqlite3
from typing import Dict, List, Generator, AsyncGenerator
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis

# Test environment setup
os.environ["TESTING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

# Configure TensorFlow for testing
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

# Import project modules for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domain.entities.property import Property
from domain.entities.user import User, UserPreferences, UserInteraction
from domain.entities.search_query import SearchQuery
from infrastructure.data.config import DatabaseConfig


# =======================
# Test Configuration
# =======================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Ensure deterministic behavior for ML tests
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create test output directories
    os.makedirs("test_outputs", exist_ok=True)
    os.makedirs("test_outputs/models", exist_ok=True)
    os.makedirs("test_outputs/logs", exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "test_ml" in str(item.fspath):
            item.add_marker(pytest.mark.ml)
        if "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        if "test_db" in str(item.fspath) or "repository" in str(item.fspath):
            item.add_marker(pytest.mark.db)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# =======================
# Event Loop Configuration
# =======================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =======================
# Database Fixtures
# =======================

@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Provide test database URL."""
    return "sqlite:///test_rental_ml.db"


@pytest.fixture(scope="session")
def test_engine(test_db_url):
    """Create test database engine."""
    engine = create_engine(
        test_db_url,
        echo=False,
        connect_args={"check_same_thread": False}
    )
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_engine):
    """Create a test database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    
    # Create a session bound to the connection
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.expire.return_value = True
    mock_redis.flushdb.return_value = True
    return mock_redis


# =======================
# Repository Fixtures
# =======================

@pytest.fixture
def mock_repository_factory():
    """Mock repository factory for testing."""
    factory = Mock()
    
    # Mock repositories
    property_repo = Mock()
    user_repo = Mock()
    model_repo = Mock()
    
    factory.get_property_repository.return_value = property_repo
    factory.get_user_repository.return_value = user_repo
    factory.get_model_repository.return_value = model_repo
    
    return factory


# =======================
# ML Model Fixtures
# =======================

@pytest.fixture
def small_ml_model():
    """Create a small TensorFlow model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


@pytest.fixture
def sample_user_item_matrix():
    """Sample user-item interaction matrix for testing."""
    return np.array([
        [1, 0, 1, 0, 1, 0],  # User 0: liked items 0, 2, 4
        [0, 1, 0, 1, 0, 1],  # User 1: liked items 1, 3, 5
        [1, 1, 0, 0, 1, 1],  # User 2: liked items 0, 1, 4, 5
        [0, 0, 0, 0, 0, 0],  # User 3: new user, no interactions
        [1, 0, 0, 1, 0, 0],  # User 4: liked items 0, 3
    ])


@pytest.fixture
def sample_property_data():
    """Sample property data for testing ML models."""
    return [
        {
            'property_id': 0,
            'neighborhood': 'Downtown',
            'city': 'San Francisco',
            'price': 3000,
            'bedrooms': 2,
            'bathrooms': 1.5,
            'amenities': ['parking', 'gym', 'pool'],
            'property_type': 'apartment',
            'square_feet': 1200
        },
        {
            'property_id': 1,
            'neighborhood': 'Mission',
            'city': 'San Francisco',
            'price': 2500,
            'bedrooms': 1,
            'bathrooms': 1.0,
            'amenities': ['gym', 'rooftop'],
            'property_type': 'studio',
            'square_feet': 800
        },
        {
            'property_id': 2,
            'neighborhood': 'SoMa',
            'city': 'San Francisco',
            'price': 3500,
            'bedrooms': 3,
            'bathrooms': 2.0,
            'amenities': ['parking', 'balcony', 'dishwasher'],
            'property_type': 'apartment',
            'square_feet': 1500
        },
        {
            'property_id': 3,
            'neighborhood': 'Castro',
            'city': 'San Francisco',
            'price': 2800,
            'bedrooms': 2,
            'bathrooms': 1.0,
            'amenities': ['gym', 'pet_friendly'],
            'property_type': 'condo',
            'square_feet': 1100
        },
        {
            'property_id': 4,
            'neighborhood': 'Marina',
            'city': 'San Francisco',
            'price': 4000,
            'bedrooms': 3,
            'bathrooms': 2.5,
            'amenities': ['parking', 'pool', 'gym', 'concierge'],
            'property_type': 'luxury_apartment',
            'square_feet': 1800
        },
        {
            'property_id': 5,
            'neighborhood': 'Richmond',
            'city': 'San Francisco',
            'price': 2200,
            'bedrooms': 2,
            'bathrooms': 1.0,
            'amenities': ['laundry', 'garden'],
            'property_type': 'house',
            'square_feet': 1000
        }
    ]


# =======================
# Entity Fixtures
# =======================

@pytest.fixture
def sample_property():
    """Create a sample Property entity for testing."""
    return Property.create(
        title="Luxury Downtown Apartment",
        description="Beautiful 2-bedroom apartment in the heart of downtown",
        price=3000.0,
        location="Downtown San Francisco",
        bedrooms=2,
        bathrooms=1.5,
        square_feet=1200,
        amenities=["parking", "gym", "pool", "concierge"],
        contact_info={"phone": "555-0123", "email": "contact@example.com"},
        images=["image1.jpg", "image2.jpg"],
        property_type="apartment"
    )


@pytest.fixture
def sample_user():
    """Create a sample User entity for testing."""
    preferences = UserPreferences(
        min_price=2000.0,
        max_price=4000.0,
        min_bedrooms=1,
        max_bedrooms=3,
        preferred_locations=["Downtown", "Mission"],
        required_amenities=["parking"],
        property_types=["apartment", "condo"]
    )
    return User.create(
        email="test@example.com",
        preferences=preferences
    )


@pytest.fixture
def sample_user_with_interactions(sample_user, sample_property):
    """Create a user with sample interactions."""
    # Add some interactions
    sample_user.add_interaction(
        UserInteraction.create(
            property_id=sample_property.id,
            interaction_type="view",
            duration_seconds=120
        )
    )
    sample_user.add_interaction(
        UserInteraction.create(
            property_id=sample_property.id,
            interaction_type="like"
        )
    )
    return sample_user


@pytest.fixture
def sample_search_query():
    """Create a sample SearchQuery entity for testing."""
    return SearchQuery.create(
        user_id=uuid4(),
        query_text="2 bedroom apartment downtown parking",
        location="San Francisco",
        min_price=2000.0,
        max_price=4000.0,
        bedrooms=2,
        amenities=["parking"],
        sort_by="price"
    )


# =======================
# API Testing Fixtures
# =======================

@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI application for testing."""
    from fastapi import FastAPI
    from unittest.mock import Mock
    
    app = FastAPI()
    app.state = Mock()
    app.state.repository_factory = Mock()
    
    return app


@pytest.fixture
def test_client(mock_fastapi_app):
    """Create test client for API testing."""
    from fastapi.testclient import TestClient
    return TestClient(mock_fastapi_app)


# =======================
# Performance Testing Fixtures
# =======================

@pytest.fixture
def performance_timer():
    """Context manager for timing test execution."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, *args):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    num_properties = 1000
    num_users = 100
    
    # Generate properties
    properties = []
    for i in range(num_properties):
        properties.append({
            'property_id': i,
            'neighborhood': f'Neighborhood_{i % 20}',
            'city': 'San Francisco',
            'price': np.random.uniform(1500, 6000),
            'bedrooms': np.random.randint(1, 5),
            'bathrooms': np.random.uniform(1, 3),
            'amenities': np.random.choice(['parking', 'gym', 'pool', 'rooftop', 'laundry'], 
                                        size=np.random.randint(1, 4), replace=False).tolist(),
            'property_type': np.random.choice(['apartment', 'condo', 'house']),
            'square_feet': np.random.randint(600, 2500)
        })
    
    # Generate user-item matrix
    user_item_matrix = np.random.choice([0, 1], size=(num_users, num_properties), p=[0.95, 0.05])
    
    return {
        'properties': properties,
        'user_item_matrix': user_item_matrix,
        'num_users': num_users,
        'num_properties': num_properties
    }


# =======================
# Temporary File Fixtures
# =======================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_model_file(temp_dir):
    """Create temporary file path for model saving/loading tests."""
    return os.path.join(temp_dir, "test_model.h5")


# =======================
# Logging Fixtures
# =======================

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


# =======================
# Scraping Test Fixtures
# =======================

@pytest.fixture
def mock_web_response():
    """Mock web response for scraping tests."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = """
    <html>
        <div class="property-card">
            <h2>Test Property</h2>
            <span class="price">$2500</span>
            <span class="bedrooms">2 bed</span>
            <span class="bathrooms">1 bath</span>
        </div>
    </html>
    """
    return mock_response


# =======================
# Data Factory Fixtures
# =======================

@pytest.fixture
def property_factory():
    """Factory for creating test properties."""
    def create_property(**kwargs):
        defaults = {
            'title': f'Test Property {uuid4().hex[:8]}',
            'description': 'A beautiful test property',
            'price': 2500.0,
            'location': 'Test Location',
            'bedrooms': 2,
            'bathrooms': 1.0,
            'square_feet': 1000,
            'amenities': ['parking'],
            'property_type': 'apartment'
        }
        defaults.update(kwargs)
        return Property.create(**defaults)
    
    return create_property


@pytest.fixture
def user_factory():
    """Factory for creating test users."""
    def create_user(**kwargs):
        defaults = {
            'email': f'test_{uuid4().hex[:8]}@example.com',
            'preferences': UserPreferences()
        }
        defaults.update(kwargs)
        return User.create(**defaults)
    
    return create_user


# =======================
# Cleanup Functions
# =======================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test."""
    yield
    
    # Cleanup TensorFlow models from memory
    tf.keras.backend.clear_session()
    
    # Cleanup any test files
    test_files = [
        "test_model.h5",
        "test_model.weights.h5",
        "test_model.json"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except OSError:
                pass


# =======================
# Pytest Hooks
# =======================

def pytest_runtest_setup(item):
    """Setup for each test item."""
    # Set random seeds for reproducible tests
    np.random.seed(42)
    tf.random.set_seed(42)


def pytest_runtest_teardown(item, nextitem):
    """Teardown after each test item."""
    # Clear TensorFlow session
    tf.keras.backend.clear_session()


# =======================
# Custom Assertions
# =======================

class CustomAssertions:
    """Custom assertion helpers for the test suite."""
    
    @staticmethod
    def assert_ml_model_trained(model):
        """Assert that an ML model is properly trained."""
        assert hasattr(model, 'is_trained'), "Model should have is_trained attribute"
        assert model.is_trained, "Model should be marked as trained"
    
    @staticmethod
    def assert_valid_recommendations(recommendations, min_count=1):
        """Assert that recommendations are valid."""
        assert len(recommendations) >= min_count, f"Should have at least {min_count} recommendations"
        for rec in recommendations:
            assert hasattr(rec, 'item_id'), "Recommendation should have item_id"
            assert hasattr(rec, 'predicted_rating'), "Recommendation should have predicted_rating"
            assert 0 <= rec.predicted_rating <= 1, "Rating should be between 0 and 1"
    
    @staticmethod
    def assert_performance_threshold(elapsed_time, threshold_seconds):
        """Assert that execution time is within threshold."""
        assert elapsed_time <= threshold_seconds, f"Execution took {elapsed_time:.2f}s, expected <= {threshold_seconds}s"


@pytest.fixture
def custom_assertions():
    """Provide custom assertion helpers."""
    return CustomAssertions


# =======================
# Async Testing Utilities
# =======================

@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 30.0  # seconds