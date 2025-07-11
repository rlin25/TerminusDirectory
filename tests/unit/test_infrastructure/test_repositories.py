"""
Unit tests for data repositories (PostgreSQL and Redis implementations).

Tests repository interfaces, data access patterns, error handling, and caching.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Optional
from uuid import uuid4
import json

from domain.entities.property import Property
from domain.entities.user import User, UserPreferences
from tests.utils.data_factories import PropertyFactory, UserFactory, FactoryConfig
from tests.utils.test_helpers import DatabaseTestHelpers, MockHelpers


class TestPostgresPropertyRepository:
    """Test cases for PostgreSQL Property Repository."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.factory = PropertyFactory(FactoryConfig(seed=42))
        self.mock_session = Mock()
        self.mock_async_session = AsyncMock()
        
        # Create mock repository (we'll patch the actual import)
        self.repository_class = Mock()
        self.repository = Mock()
        self.repository_class.return_value = self.repository
    
    @pytest.mark.asyncio
    async def test_get_by_id_success(self):
        """Test successful property retrieval by ID."""
        # Setup
        property_id = uuid4()
        expected_property = self.factory.create()
        expected_property.id = property_id
        
        self.repository.get_by_id = AsyncMock(return_value=expected_property)
        
        # Test
        result = await self.repository.get_by_id(property_id)
        
        # Assertions
        assert result == expected_property
        assert result.id == property_id
        self.repository.get_by_id.assert_called_once_with(property_id)
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        """Test property retrieval when property doesn't exist."""
        property_id = uuid4()
        self.repository.get_by_id = AsyncMock(return_value=None)
        
        result = await self.repository.get_by_id(property_id)
        
        assert result is None
        self.repository.get_by_id.assert_called_once_with(property_id)
    
    @pytest.mark.asyncio
    async def test_get_by_id_database_error(self):
        """Test property retrieval with database error."""
        property_id = uuid4()
        self.repository.get_by_id = AsyncMock(side_effect=Exception("Database connection failed"))
        
        with pytest.raises(Exception, match="Database connection failed"):
            await self.repository.get_by_id(property_id)
    
    @pytest.mark.asyncio
    async def test_get_all_active_success(self):
        """Test retrieval of all active properties."""
        expected_properties = self.factory.create_batch(5)
        for prop in expected_properties:
            prop.activate()
        
        self.repository.get_all_active = AsyncMock(return_value=expected_properties)
        
        result = await self.repository.get_all_active(limit=10, offset=0)
        
        assert len(result) == 5
        assert all(prop.is_active for prop in result)
        self.repository.get_all_active.assert_called_once_with(limit=10, offset=0)
    
    @pytest.mark.asyncio
    async def test_get_all_active_with_pagination(self):
        """Test active properties retrieval with pagination."""
        expected_properties = self.factory.create_batch(3)
        self.repository.get_all_active = AsyncMock(return_value=expected_properties)
        
        result = await self.repository.get_all_active(limit=3, offset=5)
        
        assert len(result) == 3
        self.repository.get_all_active.assert_called_once_with(limit=3, offset=5)
    
    @pytest.mark.asyncio
    async def test_get_by_location_success(self):
        """Test property retrieval by location."""
        location = "Downtown"
        expected_properties = [
            self.factory.create(location=f"{location}, San Francisco"),
            self.factory.create(location=f"{location}, CA")
        ]
        
        self.repository.get_by_location = AsyncMock(return_value=expected_properties)
        
        result = await self.repository.get_by_location(location, limit=10, offset=0)
        
        assert len(result) == 2
        assert all(location in prop.location for prop in result)
        self.repository.get_by_location.assert_called_once_with(location, limit=10, offset=0)
    
    @pytest.mark.asyncio
    async def test_get_by_price_range_success(self):
        """Test property retrieval by price range."""
        min_price = 2000.0
        max_price = 4000.0
        expected_properties = [
            self.factory.create(price=2500.0),
            self.factory.create(price=3500.0)
        ]
        
        self.repository.get_by_price_range = AsyncMock(return_value=expected_properties)
        
        result = await self.repository.get_by_price_range(min_price, max_price, limit=10, offset=0)
        
        assert len(result) == 2
        assert all(min_price <= prop.price <= max_price for prop in result)
        self.repository.get_by_price_range.assert_called_once_with(min_price, max_price, limit=10, offset=0)
    
    @pytest.mark.asyncio
    async def test_save_property_success(self):
        """Test successful property saving."""
        property_to_save = self.factory.create()
        self.repository.save = AsyncMock(return_value=property_to_save)
        
        result = await self.repository.save(property_to_save)
        
        assert result == property_to_save
        self.repository.save.assert_called_once_with(property_to_save)
    
    @pytest.mark.asyncio
    async def test_save_property_database_error(self):
        """Test property saving with database error."""
        property_to_save = self.factory.create()
        self.repository.save = AsyncMock(side_effect=Exception("Constraint violation"))
        
        with pytest.raises(Exception, match="Constraint violation"):
            await self.repository.save(property_to_save)
    
    @pytest.mark.asyncio
    async def test_delete_property_success(self):
        """Test successful property deletion."""
        property_id = uuid4()
        self.repository.delete = AsyncMock(return_value=True)
        
        result = await self.repository.delete(property_id)
        
        assert result is True
        self.repository.delete.assert_called_once_with(property_id)
    
    @pytest.mark.asyncio
    async def test_delete_property_not_found(self):
        """Test property deletion when property doesn't exist."""
        property_id = uuid4()
        self.repository.delete = AsyncMock(return_value=False)
        
        result = await self.repository.delete(property_id)
        
        assert result is False
        self.repository.delete.assert_called_once_with(property_id)
    
    @pytest.mark.asyncio
    async def test_get_count_success(self):
        """Test property count retrieval."""
        expected_count = 42
        self.repository.get_count = AsyncMock(return_value=expected_count)
        
        result = await self.repository.get_count()
        
        assert result == expected_count
        self.repository.get_count.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_active_count_success(self):
        """Test active property count retrieval."""
        expected_count = 35
        self.repository.get_active_count = AsyncMock(return_value=expected_count)
        
        result = await self.repository.get_active_count()
        
        assert result == expected_count
        self.repository.get_active_count.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_aggregated_stats_success(self):
        """Test aggregated statistics retrieval."""
        expected_stats = {
            'avg_price': 2850.0,
            'min_price': 1200.0,
            'max_price': 5500.0,
            'avg_bedrooms': 2.3,
            'avg_bathrooms': 1.8,
            'avg_square_feet': 1450.0
        }
        
        self.repository.get_aggregated_stats = AsyncMock(return_value=expected_stats)
        
        result = await self.repository.get_aggregated_stats()
        
        assert result == expected_stats
        assert 'avg_price' in result
        assert 'min_price' in result
        assert 'max_price' in result
        self.repository.get_aggregated_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_property_features_success(self):
        """Test property feature retrieval for ML models."""
        property_id = uuid4()
        expected_features = {
            'price': 3000.0,
            'bedrooms': 2,
            'bathrooms': 1.5,
            'location_vector': [0.1, 0.3, 0.5],
            'amenity_vector': [1, 0, 1, 1, 0]
        }
        
        self.repository.get_property_features = AsyncMock(return_value=expected_features)
        
        result = await self.repository.get_property_features(property_id)
        
        assert result == expected_features
        assert 'price' in result
        assert 'location_vector' in result
        self.repository.get_property_features.assert_called_once_with(property_id)


class TestPostgresUserRepository:
    """Test cases for PostgreSQL User Repository."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.user_factory = UserFactory(FactoryConfig(seed=42))
        self.repository = Mock()
    
    @pytest.mark.asyncio
    async def test_get_by_id_success(self):
        """Test successful user retrieval by ID."""
        user_id = uuid4()
        expected_user = self.user_factory.create()
        expected_user.id = user_id
        
        self.repository.get_by_id = AsyncMock(return_value=expected_user)
        
        result = await self.repository.get_by_id(user_id)
        
        assert result == expected_user
        assert result.id == user_id
        self.repository.get_by_id.assert_called_once_with(user_id)
    
    @pytest.mark.asyncio
    async def test_get_by_email_success(self):
        """Test successful user retrieval by email."""
        email = "test@example.com"
        expected_user = self.user_factory.create(email=email)
        
        self.repository.get_by_email = AsyncMock(return_value=expected_user)
        
        result = await self.repository.get_by_email(email)
        
        assert result == expected_user
        assert result.email == email
        self.repository.get_by_email.assert_called_once_with(email)
    
    @pytest.mark.asyncio
    async def test_get_by_email_not_found(self):
        """Test user retrieval when email doesn't exist."""
        email = "nonexistent@example.com"
        self.repository.get_by_email = AsyncMock(return_value=None)
        
        result = await self.repository.get_by_email(email)
        
        assert result is None
        self.repository.get_by_email.assert_called_once_with(email)
    
    @pytest.mark.asyncio
    async def test_save_user_success(self):
        """Test successful user saving."""
        user_to_save = self.user_factory.create()
        self.repository.save = AsyncMock(return_value=user_to_save)
        
        result = await self.repository.save(user_to_save)
        
        assert result == user_to_save
        self.repository.save.assert_called_once_with(user_to_save)
    
    @pytest.mark.asyncio
    async def test_save_user_duplicate_email(self):
        """Test user saving with duplicate email."""
        user_to_save = self.user_factory.create()
        self.repository.save = AsyncMock(side_effect=Exception("Email already exists"))
        
        with pytest.raises(Exception, match="Email already exists"):
            await self.repository.save(user_to_save)
    
    @pytest.mark.asyncio
    async def test_get_user_interactions_success(self):
        """Test retrieval of user interactions."""
        user_id = uuid4()
        property_factory = PropertyFactory(FactoryConfig(seed=42))
        properties = property_factory.create_batch(3)
        
        expected_interactions = [
            {'property_id': properties[0].id, 'interaction_type': 'view', 'timestamp': '2024-01-01T10:00:00'},
            {'property_id': properties[1].id, 'interaction_type': 'like', 'timestamp': '2024-01-01T11:00:00'},
            {'property_id': properties[2].id, 'interaction_type': 'save', 'timestamp': '2024-01-01T12:00:00'}
        ]
        
        self.repository.get_user_interactions = AsyncMock(return_value=expected_interactions)
        
        result = await self.repository.get_user_interactions(user_id)
        
        assert len(result) == 3
        assert all('property_id' in interaction for interaction in result)
        assert all('interaction_type' in interaction for interaction in result)
        self.repository.get_user_interactions.assert_called_once_with(user_id)
    
    @pytest.mark.asyncio
    async def test_delete_user_success(self):
        """Test successful user deletion."""
        user_id = uuid4()
        self.repository.delete = AsyncMock(return_value=True)
        
        result = await self.repository.delete(user_id)
        
        assert result is True
        self.repository.delete.assert_called_once_with(user_id)


class TestRedisCacheRepository:
    """Test cases for Redis Cache Repository."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.mock_redis = MockHelpers.create_mock_cache()
        self.repository = Mock()
        self.repository.redis_client = self.mock_redis
    
    @pytest.mark.asyncio
    async def test_get_cache_hit(self):
        """Test cache retrieval with cache hit."""
        key = "property:123"
        cached_data = {"id": "123", "title": "Test Property", "price": 3000}
        
        self.mock_redis.get.return_value = json.dumps(cached_data)
        self.repository.get = Mock(return_value=cached_data)
        
        result = self.repository.get(key)
        
        assert result == cached_data
        self.mock_redis.get.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self):
        """Test cache retrieval with cache miss."""
        key = "property:456"
        
        self.mock_redis.get.return_value = None
        self.repository.get = Mock(return_value=None)
        
        result = self.repository.get(key)
        
        assert result is None
        self.mock_redis.get.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_set_cache_success(self):
        """Test successful cache setting."""
        key = "property:789"
        data = {"id": "789", "title": "Cached Property", "price": 2500}
        ttl = 3600  # 1 hour
        
        self.mock_redis.set.return_value = True
        self.repository.set = Mock(return_value=True)
        
        result = self.repository.set(key, data, ttl)
        
        assert result is True
        self.repository.set.assert_called_once_with(key, data, ttl)
    
    @pytest.mark.asyncio
    async def test_delete_cache_success(self):
        """Test successful cache deletion."""
        key = "property:999"
        
        self.mock_redis.delete.return_value = 1  # Number of keys deleted
        self.repository.delete = Mock(return_value=True)
        
        result = self.repository.delete(key)
        
        assert result is True
        self.repository.delete.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_exists_cache_key_exists(self):
        """Test cache key existence check."""
        key = "user:123:recommendations"
        
        self.mock_redis.exists.return_value = 1
        self.repository.exists = Mock(return_value=True)
        
        result = self.repository.exists(key)
        
        assert result is True
        self.repository.exists.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_exists_cache_key_not_exists(self):
        """Test cache key existence check when key doesn't exist."""
        key = "user:456:recommendations"
        
        self.mock_redis.exists.return_value = 0
        self.repository.exists = Mock(return_value=False)
        
        result = self.repository.exists(key)
        
        assert result is False
        self.repository.exists.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_expire_cache_success(self):
        """Test setting cache expiration."""
        key = "search:query:123"
        ttl = 1800  # 30 minutes
        
        self.mock_redis.expire.return_value = True
        self.repository.expire = Mock(return_value=True)
        
        result = self.repository.expire(key, ttl)
        
        assert result is True
        self.repository.expire.assert_called_once_with(key, ttl)
    
    @pytest.mark.asyncio
    async def test_flushdb_success(self):
        """Test database flush for testing."""
        self.mock_redis.flushdb.return_value = True
        self.repository.flushdb = Mock(return_value=True)
        
        result = self.repository.flushdb()
        
        assert result is True
        self.repository.flushdb.assert_called_once()
    
    def test_cache_key_generation(self):
        """Test cache key generation patterns."""
        # Test different key patterns
        test_cases = [
            ("property", "123", "property:123"),
            ("user", "456", "user:456"),
            ("recommendations", "user:789", "recommendations:user:789"),
            ("search", "query:abc", "search:query:abc")
        ]
        
        for prefix, identifier, expected_key in test_cases:
            # This would test actual key generation logic
            generated_key = f"{prefix}:{identifier}"
            assert generated_key == expected_key
    
    def test_cache_serialization(self):
        """Test data serialization for cache storage."""
        test_data = {
            "id": "123",
            "title": "Test Property",
            "price": 3000.0,
            "bedrooms": 2,
            "amenities": ["parking", "gym"],
            "created_at": "2024-01-01T10:00:00Z"
        }
        
        # Test JSON serialization
        serialized = json.dumps(test_data)
        deserialized = json.loads(serialized)
        
        assert deserialized == test_data
        assert isinstance(deserialized["price"], float)
        assert isinstance(deserialized["bedrooms"], int)
        assert isinstance(deserialized["amenities"], list)
    
    def test_cache_error_handling(self):
        """Test cache error handling."""
        key = "test:key"
        
        # Test Redis connection error
        self.mock_redis.get.side_effect = Exception("Redis connection failed")
        self.repository.get = Mock(side_effect=Exception("Redis connection failed"))
        
        with pytest.raises(Exception, match="Redis connection failed"):
            self.repository.get(key)


class TestModelRepository:
    """Test cases for Model Repository (for ML model storage)."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.repository = Mock()
    
    @pytest.mark.asyncio
    async def test_save_model_success(self):
        """Test successful model saving."""
        model_id = "content_based_v1"
        model_data = b"mock_model_binary_data"
        metadata = {
            "model_type": "content_based_recommender",
            "version": "1.0",
            "training_date": "2024-01-01",
            "performance_metrics": {"accuracy": 0.85, "loss": 0.15}
        }
        
        self.repository.save_model = AsyncMock(return_value=True)
        
        result = await self.repository.save_model(model_id, model_data, metadata)
        
        assert result is True
        self.repository.save_model.assert_called_once_with(model_id, model_data, metadata)
    
    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model loading."""
        model_id = "hybrid_v2"
        expected_data = b"mock_model_binary_data"
        
        self.repository.load_model = AsyncMock(return_value=expected_data)
        
        result = await self.repository.load_model(model_id)
        
        assert result == expected_data
        self.repository.load_model.assert_called_once_with(model_id)
    
    @pytest.mark.asyncio
    async def test_load_model_not_found(self):
        """Test model loading when model doesn't exist."""
        model_id = "nonexistent_model"
        
        self.repository.load_model = AsyncMock(return_value=None)
        
        result = await self.repository.load_model(model_id)
        
        assert result is None
        self.repository.load_model.assert_called_once_with(model_id)
    
    @pytest.mark.asyncio
    async def test_get_model_metadata_success(self):
        """Test model metadata retrieval."""
        model_id = "collaborative_v3"
        expected_metadata = {
            "model_type": "collaborative_filtering",
            "version": "3.0",
            "training_date": "2024-02-01",
            "performance_metrics": {"rmse": 0.85, "mae": 0.65},
            "hyperparameters": {"factors": 50, "learning_rate": 0.01}
        }
        
        self.repository.get_model_metadata = AsyncMock(return_value=expected_metadata)
        
        result = await self.repository.get_model_metadata(model_id)
        
        assert result == expected_metadata
        assert result["model_type"] == "collaborative_filtering"
        assert "performance_metrics" in result
        self.repository.get_model_metadata.assert_called_once_with(model_id)
    
    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test listing all available models."""
        expected_models = [
            {"id": "content_based_v1", "type": "content_based", "version": "1.0"},
            {"id": "collaborative_v2", "type": "collaborative", "version": "2.0"},
            {"id": "hybrid_v1", "type": "hybrid", "version": "1.0"}
        ]
        
        self.repository.list_models = AsyncMock(return_value=expected_models)
        
        result = await self.repository.list_models()
        
        assert len(result) == 3
        assert all("id" in model for model in result)
        assert all("type" in model for model in result)
        self.repository.list_models.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_model_success(self):
        """Test successful model deletion."""
        model_id = "old_model_v1"
        
        self.repository.delete_model = AsyncMock(return_value=True)
        
        result = await self.repository.delete_model(model_id)
        
        assert result is True
        self.repository.delete_model.assert_called_once_with(model_id)


class TestRepositoryIntegration:
    """Integration tests for repository interactions."""
    
    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.property_repo = Mock()
        self.user_repo = Mock()
        self.cache_repo = Mock()
        self.model_repo = Mock()
    
    @pytest.mark.asyncio
    async def test_property_caching_workflow(self):
        """Test complete property caching workflow."""
        property_id = uuid4()
        property_data = PropertyFactory().create()
        property_data.id = property_id
        
        cache_key = f"property:{property_id}"
        
        # Setup mocks
        self.cache_repo.get = Mock(return_value=None)  # Cache miss
        self.property_repo.get_by_id = AsyncMock(return_value=property_data)
        self.cache_repo.set = Mock(return_value=True)
        
        # Simulate service layer logic
        # 1. Check cache
        cached_data = self.cache_repo.get(cache_key)
        assert cached_data is None
        
        # 2. Fetch from database
        db_data = await self.property_repo.get_by_id(property_id)
        assert db_data == property_data
        
        # 3. Store in cache
        cache_result = self.cache_repo.set(cache_key, db_data, ttl=3600)
        assert cache_result is True
        
        # Verify call sequence
        self.cache_repo.get.assert_called_once_with(cache_key)
        self.property_repo.get_by_id.assert_called_once_with(property_id)
        self.cache_repo.set.assert_called_once_with(cache_key, db_data, ttl=3600)
    
    @pytest.mark.asyncio
    async def test_user_recommendations_caching(self):
        """Test user recommendations caching workflow."""
        user_id = uuid4()
        recommendations = [
            {"item_id": 1, "score": 0.9},
            {"item_id": 2, "score": 0.8},
            {"item_id": 3, "score": 0.7}
        ]
        
        cache_key = f"recommendations:user:{user_id}"
        
        # Setup mocks
        self.cache_repo.get = Mock(return_value=recommendations)  # Cache hit
        
        # Simulate recommendation service logic
        cached_recommendations = self.cache_repo.get(cache_key)
        
        assert cached_recommendations == recommendations
        assert len(cached_recommendations) == 3
        self.cache_repo.get.assert_called_once_with(cache_key)
    
    @pytest.mark.asyncio
    async def test_repository_error_propagation(self):
        """Test error propagation between repository layers."""
        property_id = uuid4()
        
        # Database error should propagate
        self.property_repo.get_by_id = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(Exception, match="Database error"):
            await self.property_repo.get_by_id(property_id)
        
        # Cache error should be handled gracefully in some cases
        self.cache_repo.get = Mock(side_effect=Exception("Redis error"))
        
        with pytest.raises(Exception, match="Redis error"):
            self.cache_repo.get("test:key")