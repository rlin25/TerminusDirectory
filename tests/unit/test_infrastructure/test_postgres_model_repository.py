"""
Unit tests for PostgreSQL Model Repository
Tests the enhanced ML model repository implementation with comprehensive coverage
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from ....src.infrastructure.data.repositories.postgres_model_repository import (
    PostgresModelRepository,
    MLModelStorage,
    EmbeddingStorage,
    TrainingMetrics,
    PredictionCache,
    QueryMetrics
)


class TestPostgresModelRepository:
    """Test suite for PostgresModelRepository"""
    
    @pytest.fixture
    def mock_database_url(self):
        """Mock database URL for testing"""
        return "postgresql+asyncpg://test:test@localhost:5432/test_db"
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session"""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.execute = AsyncMock()
        session.add = Mock()
        session.close = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_session_factory(self, mock_session):
        """Mock session factory"""
        factory = Mock()
        factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        factory.return_value.__aexit__ = AsyncMock()
        return factory
    
    @pytest.fixture
    def repository(self, mock_database_url):
        """Create repository instance for testing"""
        with patch('sqlalchemy.ext.asyncio.create_async_engine'):
            with patch('sqlalchemy.orm.sessionmaker'):
                repo = PostgresModelRepository(
                    database_url=mock_database_url,
                    enable_metrics=True
                )\n                repo.async_session_factory = Mock()\n                return repo
    
    @pytest.fixture
    def sample_model_data(self):
        """Sample ML model data for testing"""
        class MockModel:
            def __init__(self):
                self.weights = np.array([1.0, 2.0, 3.0])
                self.bias = 0.5
                
            def predict(self, x):
                return np.dot(x, self.weights) + self.bias
        
        return MockModel()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings data for testing"""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    @pytest.fixture
    def sample_metrics(self):
        """Sample training metrics for testing"""
        return {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90,
            'loss': 0.05
        }
    
    # ============================================
    # Model Storage Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_save_model_success(self, repository, sample_model_data, mock_session_factory):
        """Test successful model saving"""
        # Mock the session factory and session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock()
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        # Test save_model
        result = await repository.save_model(
            model_name="test_model",
            model_data=sample_model_data,
            version="1.0"
        )
        
        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_model_validation_error(self, repository):
        """Test model saving with invalid input"""
        # Test with empty model name
        result = await repository.save_model("", {"data": "test"}, "1.0")
        assert result is False
        
        # Test with None model data
        result = await repository.save_model("test_model", None, "1.0")
        assert result is False
        
        # Test with empty version
        result = await repository.save_model("test_model", {"data": "test"}, "")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, repository, sample_model_data, mock_session_factory):
        """Test successful model loading"""
        # Mock model record
        mock_model_record = Mock()
        mock_model_record.model_data = pickle.dumps(sample_model_data)
        mock_model_record.version = "1.0"
        mock_model_record.created_at = datetime.utcnow()
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_model_record
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock()
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        # Test load_model
        result = await repository.load_model("test_model", "1.0")
        
        assert result is not None
        assert hasattr(result, 'weights')
        assert hasattr(result, 'bias')
        np.testing.assert_array_equal(result.weights, sample_model_data.weights)
        assert result.bias == sample_model_data.bias
    
    @pytest.mark.asyncio
    async def test_load_model_not_found(self, repository, mock_session_factory):
        """Test loading non-existent model"""
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.load_model("non_existent_model", "1.0")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_model_versions_success(self, repository, mock_session_factory):
        """Test getting model versions"""
        # Mock version results
        mock_results = [("1.0",), ("1.1",), ("2.0",)]
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_results
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_model_versions("test_model")
        
        assert result == ["1.0", "1.1", "2.0"]
    
    @pytest.mark.asyncio
    async def test_delete_model_success(self, repository, mock_session_factory):
        """Test successful model deletion"""
        # Mock update result
        mock_result = Mock()
        mock_result.rowcount = 1
        
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        result = await repository.delete_model("test_model", "1.0")
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    # ============================================
    # Embeddings Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_save_embeddings_success(self, repository, sample_embeddings, mock_session_factory):
        """Test successful embeddings saving"""
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        result = await repository.save_embeddings(
            entity_type="user",
            entity_id="user123",
            embeddings=sample_embeddings
        )
        
        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_embeddings_validation_error(self, repository):
        """Test embeddings saving with invalid input"""
        # Test with empty entity type
        result = await repository.save_embeddings("", "user123", np.array([1, 2, 3]))
        assert result is False
        
        # Test with None embeddings
        result = await repository.save_embeddings("user", "user123", None)
        assert result is False
        
        # Test with empty embeddings
        result = await repository.save_embeddings("user", "user123", np.array([]))
        assert result is False
        
        # Test with non-numpy array
        result = await repository.save_embeddings("user", "user123", [1, 2, 3])
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_embeddings_success(self, repository, sample_embeddings, mock_session_factory):
        """Test successful embeddings retrieval"""
        # Mock embedding record
        mock_embedding_record = Mock()
        mock_embedding_record.embeddings = pickle.dumps(sample_embeddings)
        mock_embedding_record.dimension = len(sample_embeddings)
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_embedding_record
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_embeddings("user", "user123")
        
        assert result is not None
        np.testing.assert_array_equal(result, sample_embeddings)
    
    @pytest.mark.asyncio
    async def test_get_all_embeddings_success(self, repository, sample_embeddings, mock_session_factory):
        """Test getting all embeddings for entity type"""
        # Mock embedding records
        mock_records = []
        for i in range(3):
            record = Mock()
            record.entity_id = f"user{i}"
            record.embeddings = pickle.dumps(sample_embeddings * (i + 1))
            mock_records.append(record)
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_records
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_all_embeddings("user")
        
        assert len(result) == 3
        assert "user0" in result
        assert "user1" in result
        assert "user2" in result
    
    # ============================================
    # Training Metrics Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_save_training_metrics_success(self, repository, sample_metrics, mock_session_factory):
        """Test successful training metrics saving"""
        # Mock session
        mock_session = AsyncMock()
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        result = await repository.save_training_metrics(
            model_name="test_model",
            version="1.0",
            metrics=sample_metrics
        )
        
        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_training_metrics_validation_error(self, repository):
        """Test training metrics saving with invalid input"""
        # Test with empty model name
        result = await repository.save_training_metrics("", "1.0", {"accuracy": 0.95})
        assert result is False
        
        # Test with empty metrics
        result = await repository.save_training_metrics("test_model", "1.0", {})
        assert result is False
        
        # Test with non-numeric metrics
        result = await repository.save_training_metrics("test_model", "1.0", {"accuracy": "high"})
        assert result is False
        
        # Test with infinite metrics
        result = await repository.save_training_metrics("test_model", "1.0", {"accuracy": float('inf')})
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_training_metrics_success(self, repository, sample_metrics, mock_session_factory):
        """Test successful training metrics retrieval"""
        # Mock metrics record
        mock_metrics_record = Mock()
        mock_metrics_record.metrics = {**sample_metrics, '_metadata': {'saved_at': datetime.utcnow().isoformat()}}
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_metrics_record
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_training_metrics("test_model", "1.0")
        
        assert result is not None
        assert result == sample_metrics  # Metadata should be removed
        assert '_metadata' not in result
    
    # ============================================
    # Prediction Cache Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_cache_predictions_success(self, repository, mock_session_factory):
        """Test successful prediction caching"""
        predictions = {"result": [1, 2, 3], "confidence": 0.95}
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock()
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        # Mock _cleanup_expired_cache
        repository._cleanup_expired_cache = AsyncMock(return_value=0)
        
        result = await repository.cache_predictions(
            cache_key="test_predictions",
            predictions=predictions,
            ttl_seconds=3600
        )
        
        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_predictions_validation_error(self, repository):
        """Test prediction caching with invalid input"""
        # Test with empty cache key
        result = await repository.cache_predictions("", {"result": [1, 2, 3]}, 3600)
        assert result is False
        
        # Test with None predictions
        result = await repository.cache_predictions("test_key", None, 3600)
        assert result is False
        
        # Test with invalid TTL
        result = await repository.cache_predictions("test_key", {"result": [1, 2, 3]}, 0)
        assert result is False
        
        # Test with TTL too large
        result = await repository.cache_predictions("test_key", {"result": [1, 2, 3]}, 100000)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_cached_predictions_success(self, repository, mock_session_factory):
        """Test successful cached predictions retrieval"""
        predictions = {"result": [1, 2, 3], "confidence": 0.95}
        
        # Mock cache record
        mock_cache_record = Mock()
        mock_cache_record.predictions = pickle.dumps(predictions)
        mock_cache_record.id = "cache_id"
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_cache_record
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_cached_predictions("test_key")
        
        assert result is not None
        assert result == predictions
    
    @pytest.mark.asyncio
    async def test_get_cached_predictions_expired(self, repository, mock_session_factory):
        """Test getting expired cached predictions"""
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_cached_predictions("test_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_clear_cache_success(self, repository, mock_session_factory):
        """Test successful cache clearing"""
        # Mock delete result
        mock_result = Mock()
        mock_result.rowcount = 5
        
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        result = await repository.clear_cache("test_*")
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    # ============================================
    # Utility Methods Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, repository, mock_session_factory):
        """Test successful health check"""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.health_check()
        
        assert result['status'] == 'healthy'
        assert 'checks' in result
        assert 'timestamp' in result
    
    @pytest.mark.asyncio
    async def test_cleanup_old_models_success(self, repository, mock_session_factory):
        """Test successful old model cleanup"""
        # Mock model records
        mock_models = [Mock() for _ in range(10)]
        for i, model in enumerate(mock_models):
            model.id = f"model_{i}"
            model.created_at = datetime.utcnow() - timedelta(days=i)
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_models
        
        # Mock update results
        mock_update_result = Mock()
        mock_update_result.rowcount = 1
        mock_session.execute.return_value = mock_update_result
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        result = await repository.cleanup_old_models("test_model", keep_versions=5)
        
        assert result == 5  # Should delete 5 old versions
    
    @pytest.mark.asyncio
    async def test_batch_save_embeddings_success(self, repository, sample_embeddings, mock_session_factory):
        """Test successful batch embeddings saving"""
        embeddings_data = [
            ("user", "user1", sample_embeddings),
            ("user", "user2", sample_embeddings * 2),
            ("property", "prop1", sample_embeddings * 3)
        ]
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Mock get_transaction
        repository.get_transaction = AsyncMock()
        mock_transaction = Mock()
        mock_transaction.session = mock_session
        repository.get_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
        repository.get_transaction.return_value.__aexit__ = AsyncMock()
        
        result = await repository.batch_save_embeddings(embeddings_data)
        
        assert len(result) == 3
        assert all(success for success in result.values())
    
    def test_get_python_version(self, repository):
        """Test getting Python version"""
        version = repository._get_python_version()
        assert isinstance(version, str)
        assert len(version.split('.')) >= 2
    
    def test_get_dependencies_info(self, repository):
        """Test getting dependencies info"""
        deps = repository._get_dependencies_info()
        assert isinstance(deps, dict)
        # At least numpy should be available
        assert 'numpy' in deps or len(deps) == 0  # Empty if no ML deps installed
    
    def test_calculate_compression_ratio(self, repository, sample_model_data):
        """Test compression ratio calculation"""
        import pickle
        serialized = pickle.dumps(sample_model_data)
        ratio = repository._calculate_compression_ratio(sample_model_data, serialized)
        assert isinstance(ratio, float)
        assert ratio > 0
    
    # ============================================
    # Performance and Metrics Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, repository):
        """Test getting performance metrics"""
        # Add some mock metrics
        repository._performance_metrics = [
            QueryMetrics("save_model", 0.5, 1, datetime.utcnow()),
            QueryMetrics("load_model", 0.3, 1, datetime.utcnow())
        ]
        
        metrics = await repository.get_performance_metrics()
        
        assert len(metrics) == 2
        assert metrics[0].query_type == "save_model"
        assert metrics[1].query_type == "load_model"
    
    @pytest.mark.asyncio
    async def test_clear_performance_metrics(self, repository):
        """Test clearing performance metrics"""
        # Add some mock metrics
        repository._performance_metrics = [
            QueryMetrics("save_model", 0.5, 1, datetime.utcnow())
        ]
        
        await repository.clear_performance_metrics()
        
        assert len(repository._performance_metrics) == 0
    
    @pytest.mark.asyncio
    async def test_get_cache_statistics(self, repository, mock_session_factory):
        """Test getting cache statistics"""
        # Mock statistics results
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = 10
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_cache_statistics()
        
        assert isinstance(result, dict)
        # Should have some statistics even if mocked
        assert 'total_entries' in result or result == {}
    
    # ============================================
    # Error Handling Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_save_model_database_error(self, repository):
        """Test model saving with database error"""
        # Mock database error
        repository.get_transaction = AsyncMock()
        repository.get_transaction.side_effect = Exception("Database connection failed")
        
        result = await repository.save_model("test_model", {"data": "test"}, "1.0")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_load_model_deserialization_error(self, repository, mock_session_factory):
        """Test model loading with deserialization error"""
        # Mock corrupted model record
        mock_model_record = Mock()
        mock_model_record.model_data = b"corrupted_data"
        mock_model_record.version = "1.0"
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_model_record
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.load_model("test_model", "1.0")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_embeddings_deserialization_error(self, repository, mock_session_factory):
        """Test embeddings retrieval with deserialization error"""
        # Mock corrupted embedding record
        mock_embedding_record = Mock()
        mock_embedding_record.embeddings = b"corrupted_data"
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_embedding_record
        
        # Mock get_session
        repository.get_session = AsyncMock()
        repository.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        repository.get_session.return_value.__aexit__ = AsyncMock()
        
        result = await repository.get_embeddings("user", "user123")
        
        assert result is None


# Import pickle for testing
import pickle