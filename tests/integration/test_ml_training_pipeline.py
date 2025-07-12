"""
Integration tests for the ML training pipeline components.

These tests validate the complete ML training pipeline including
feature engineering, model training, monitoring, and deployment.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.application.ml_training import (
    ProductionTrainingPipeline,
    ModelRegistry,
    FeatureEngineeringPipeline,
    ModelMonitoringService,
    TrainingJobConfig,
    ModelType,
    TrainingStatus,
    ModelVersion,
    ModelStatus,
    AlertLevel,
    DriftDetectionMethod
)
from src.domain.entities.property import Property
from src.domain.entities.user import User, UserPreferences
from src.infrastructure.ml.training.data_loader import MLDataset, TrainingDataBatch


class TestMLTrainingPipelineIntegration:
    """Integration tests for ML training pipeline"""
    
    @pytest.fixture
    async def mock_repositories(self):
        """Create mock repositories for testing"""
        mock_model_repo = AsyncMock()
        mock_property_repo = AsyncMock()
        mock_user_repo = AsyncMock()
        
        # Setup mock responses
        mock_model_repo.save_model.return_value = True
        mock_model_repo.load_model.return_value = None
        mock_model_repo.get_model_versions.return_value = []
        mock_model_repo.save_training_metrics.return_value = True
        
        return {
            'model_repository': mock_model_repo,
            'property_repository': mock_property_repo,
            'user_repository': mock_user_repo
        }
    
    @pytest.fixture
    async def temp_directories(self):
        """Create temporary directories for testing"""
        temp_dir = Path(tempfile.mkdtemp())
        models_dir = temp_dir / "models"
        artifacts_dir = temp_dir / "artifacts"
        
        models_dir.mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)
        
        yield {
            'temp_dir': temp_dir,
            'models_dir': str(models_dir),
            'artifacts_dir': str(artifacts_dir)
        }
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    async def model_registry(self, mock_repositories):
        """Create a model registry for testing"""
        registry = ModelRegistry(mock_repositories['model_repository'])
        await registry.initialize()
        return registry
    
    @pytest.fixture
    async def feature_pipeline(self, mock_repositories):
        """Create a feature engineering pipeline for testing"""
        pipeline = FeatureEngineeringPipeline(
            property_repository=mock_repositories['property_repository'],
            user_repository=mock_repositories['user_repository']
        )
        await pipeline.initialize()
        return pipeline
    
    @pytest.fixture
    async def monitoring_service(self, model_registry, mock_repositories):
        """Create a monitoring service for testing"""
        service = ModelMonitoringService(
            model_registry=model_registry,
            model_repository=mock_repositories['model_repository']
        )
        await service.initialize()
        return service
    
    @pytest.fixture
    async def training_pipeline(self, temp_directories):
        """Create a training pipeline for testing"""
        pipeline = ProductionTrainingPipeline(
            database_url="sqlite:///:memory:",
            models_dir=temp_directories['models_dir'],
            artifacts_dir=temp_directories['artifacts_dir'],
            enable_monitoring=False,  # Disable for testing
            enable_scheduling=False
        )
        return pipeline
    
    @pytest.mark.asyncio
    async def test_model_registry_basic_operations(self, model_registry):
        """Test basic model registry operations"""
        # Test model registration
        model_id = await model_registry.register_model(
            model_name="test_model",
            version="1.0.0",
            model_path="/tmp/test_model.pkl",
            metadata={"test": True},
            performance_metrics={"accuracy": 0.85},
            description="Test model"
        )
        
        assert model_id is not None
        assert "test_model_1.0.0" in model_id
        
        # Test model retrieval
        model_version = await model_registry.get_model_version("test_model", "1.0.0")
        assert model_version is not None
        assert model_version.model_name == "test_model"
        assert model_version.version == "1.0.0"
        assert model_version.performance_metrics["accuracy"] == 0.85
        
        # Test model listing
        versions = await model_registry.list_model_versions("test_model")
        assert len(versions) == 1
        assert versions[0].version == "1.0.0"
        
        # Test status update
        success = await model_registry.update_model_status(
            "test_model", "1.0.0", ModelStatus.STAGING
        )
        assert success is True
    
    @pytest.mark.asyncio
    async def test_model_deployment_workflow(self, model_registry):
        """Test model deployment workflow"""
        # Register a model first
        await model_registry.register_model(
            model_name="deploy_test",
            version="1.0.0",
            model_path="/tmp/deploy_test.pkl",
            metadata={"test": True},
            performance_metrics={"accuracy": 0.9}
        )
        
        # Test staging deployment
        success = await model_registry.deploy_to_staging(
            "deploy_test", "1.0.0"
        )
        assert success is True
        
        # Test production deployment
        success = await model_registry.deploy_to_production(
            "deploy_test", "1.0.0"
        )
        assert success is True
        
        # Test canary deployment
        success = await model_registry.deploy_canary(
            "deploy_test", "1.0.0", traffic_percentage=10.0
        )
        assert success is True
        
        # Test deployment status
        status = await model_registry.get_deployment_status("deploy_test")
        assert status["model_name"] == "deploy_test"
        assert "environments" in status
    
    @pytest.mark.asyncio
    async def test_model_comparison(self, model_registry):
        """Test model comparison functionality"""
        # Register two model versions
        await model_registry.register_model(
            model_name="compare_test",
            version="1.0.0",
            model_path="/tmp/compare_test_v1.pkl",
            metadata={"test": True},
            performance_metrics={"accuracy": 0.85, "precision": 0.8}
        )
        
        await model_registry.register_model(
            model_name="compare_test",
            version="2.0.0",
            model_path="/tmp/compare_test_v2.pkl",
            metadata={"test": True},
            performance_metrics={"accuracy": 0.9, "precision": 0.88}
        )
        
        # Compare models
        comparison = await model_registry.compare_models(
            "compare_test", "1.0.0", "2.0.0"
        )
        
        assert comparison.baseline_version == "1.0.0"
        assert comparison.candidate_version == "2.0.0"
        assert comparison.improvement_percentage["accuracy"] > 0
        assert comparison.recommendation in ["deploy", "reject", "needs_more_data"]
    
    @pytest.mark.asyncio
    async def test_feature_engineering_pipeline(self, feature_pipeline):
        """Test feature engineering pipeline"""
        # Create mock dataset
        mock_dataset = MLDataset(
            train_data=TrainingDataBatch(
                user_item_matrix=np.random.rand(100, 50),
                property_features=np.random.rand(50, 10),
                user_features=np.random.rand(100, 5),
                property_metadata=[
                    {
                        'price': 1000 + i * 100,
                        'bedrooms': 1 + (i % 4),
                        'bathrooms': 1 + (i % 3),
                        'square_feet': 500 + i * 50,
                        'location': f'Location_{i % 10}',
                        'amenities': [f'amenity_{j}' for j in range(i % 5)],
                        'scraped_at': datetime.utcnow()
                    }
                    for i in range(50)
                ],
                user_metadata=[
                    {
                        'min_price': 800,
                        'max_price': 1500,
                        'min_bedrooms': 1,
                        'max_bedrooms': 3,
                        'preferred_locations': [f'Location_{i % 5}'],
                        'required_amenities': [f'amenity_{i % 3}']
                    }
                    for i in range(100)
                ],
                feature_names=['feature_' + str(i) for i in range(15)],
                batch_info={'batch_size': 150}
            ),
            validation_data=TrainingDataBatch(
                user_item_matrix=np.random.rand(20, 10),
                property_features=np.random.rand(10, 10),
                user_features=np.random.rand(20, 5),
                property_metadata=[],
                user_metadata=[],
                feature_names=['feature_' + str(i) for i in range(15)],
                batch_info={'batch_size': 30}
            ),
            test_data=None,
            metadata={
                'total_users': 100,
                'total_properties': 50,
                'total_interactions': 5000,
                'data_quality': {'interaction_sparsity': 0.1}
            }
        )
        
        # Process features
        result = await feature_pipeline.process_features(
            dataset=mock_dataset,
            feature_set_name="test_features"
        )
        
        assert result.feature_set_name == "test_features"
        assert result.processed_features.shape[0] > 0
        assert len(result.feature_names) > 0
        assert result.processing_time_seconds > 0
        assert 'feature_count' in result.feature_metadata
    
    @pytest.mark.asyncio
    async def test_real_time_feature_processing(self, feature_pipeline):
        """Test real-time feature processing"""
        # Mock the pipeline as fitted
        feature_pipeline.is_fitted = True
        feature_pipeline.scalers['fitted'] = MagicMock()
        feature_pipeline.scalers['fitted'].transform = MagicMock(return_value=np.array([[1.0, 2.0, 3.0]]))
        
        # Test real-time processing
        property_data = {
            'price': 1200,
            'square_feet': 800,
            'bedrooms': 2,
            'bathrooms': 1.5
        }
        
        user_data = {
            'min_price': 1000,
            'max_price': 1500
        }
        
        features = await feature_pipeline.process_real_time_features(
            property_data=property_data,
            user_data=user_data,
            feature_set_name="test_features"
        )
        
        assert features.shape[0] == 1  # Single sample
        assert features.shape[1] > 0  # Has features
    
    @pytest.mark.asyncio
    async def test_monitoring_service_setup(self, monitoring_service):
        """Test monitoring service setup"""
        # Setup monitoring for a model
        await monitoring_service.setup_model_monitoring(
            model_name="monitor_test",
            version="1.0.0",
            model_type="collaborative_filtering",
            monitoring_config={
                'data_drift_threshold': 0.1,
                'performance_thresholds': {'accuracy': 0.8},
                'check_frequency': 'hourly'
            }
        )
        
        # Check that monitoring is active
        monitor_key = "monitor_test:1.0.0"
        assert monitor_key in monitoring_service.active_monitors
        
        monitor_config = monitoring_service.active_monitors[monitor_key]
        assert monitor_config['model_name'] == "monitor_test"
        assert monitor_config['version'] == "1.0.0"
        assert monitor_config['is_active'] is True
    
    @pytest.mark.asyncio
    async def test_prediction_logging_and_monitoring(self, monitoring_service):
        """Test prediction logging and monitoring"""
        # Setup monitoring first
        await monitoring_service.setup_model_monitoring(
            model_name="pred_test",
            version="1.0.0",
            model_type="collaborative_filtering",
            monitoring_config={}
        )
        
        # Log some predictions
        for i in range(10):
            await monitoring_service.log_prediction(
                model_name="pred_test",
                version="1.0.0",
                input_features=np.random.rand(5),
                prediction=0.8 + 0.1 * np.random.rand(),
                actual_value=0.75 + 0.2 * np.random.rand(),
                latency_ms=50 + 10 * np.random.rand()
            )
        
        # Check performance
        snapshot = await monitoring_service.check_model_performance("pred_test", "1.0.0")
        
        assert snapshot.model_name == "pred_test"
        assert snapshot.model_version == "1.0.0"
        assert snapshot.predictions_count == 10
        assert len(snapshot.metrics) > 0
    
    @pytest.mark.asyncio
    async def test_data_drift_detection(self, monitoring_service):
        """Test data drift detection"""
        # Setup monitoring first
        await monitoring_service.setup_model_monitoring(
            model_name="drift_test",
            version="1.0.0",
            model_type="collaborative_filtering",
            monitoring_config={}
        )
        
        # Setup baseline data
        monitor_key = "drift_test:1.0.0"
        baseline_features = np.random.normal(0, 1, (1000, 5))
        monitoring_service.drift_baselines[monitor_key] = {
            'features': baseline_features,
            'feature_stats': {}
        }
        
        # Add current features (with drift)
        current_features = np.random.normal(0.5, 1.2, (500, 5))  # Shifted distribution
        for features in current_features:
            await monitoring_service.log_prediction(
                model_name="drift_test",
                version="1.0.0",
                input_features=features,
                prediction=0.8,
                latency_ms=50
            )
        
        # Detect drift
        drift_results = await monitoring_service.detect_data_drift(
            "drift_test", "1.0.0", DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        )
        
        assert len(drift_results) > 0
        for result in drift_results:
            assert result.method_used == DriftDetectionMethod.KOLMOGOROV_SMIRNOV
            assert result.drift_score >= 0
            assert 0 <= result.p_value <= 1
    
    @pytest.mark.asyncio
    async def test_ab_testing_workflow(self, monitoring_service):
        """Test A/B testing workflow"""
        # Start an A/B test
        test_id = await monitoring_service.start_ab_test(
            model_name="ab_test",
            control_version="1.0.0",
            treatment_version="1.1.0",
            traffic_split={"1.0.0": 50.0, "1.1.0": 50.0},
            success_metrics=["accuracy", "precision"],
            duration_days=7
        )
        
        assert test_id is not None
        assert test_id in monitoring_service.active_tests
        
        test_config = monitoring_service.active_tests[test_id]
        assert test_config.model_name == "ab_test"
        assert test_config.control_version == "1.0.0"
        assert test_config.treatment_version == "1.1.0"
        
        # Setup mock monitoring for both versions
        await monitoring_service.setup_model_monitoring(
            "ab_test", "1.0.0", "collaborative_filtering", {}
        )
        await monitoring_service.setup_model_monitoring(
            "ab_test", "1.1.0", "collaborative_filtering", {}
        )
        
        # Mock some performance data
        monitoring_service.prediction_buffer["ab_test:1.0.0"].extend([
            {'timestamp': datetime.utcnow(), 'prediction': 0.8, 'actual_value': 0.75, 'metadata': {}},
            {'timestamp': datetime.utcnow(), 'prediction': 0.85, 'actual_value': 0.8, 'metadata': {}},
        ])
        
        monitoring_service.prediction_buffer["ab_test:1.1.0"].extend([
            {'timestamp': datetime.utcnow(), 'prediction': 0.9, 'actual_value': 0.85, 'metadata': {}},
            {'timestamp': datetime.utcnow(), 'prediction': 0.92, 'actual_value': 0.88, 'metadata': {}},
        ])
        
        # Analyze A/B test
        result = await monitoring_service.analyze_ab_test(test_id)
        
        assert result.test_id == test_id
        assert "control_metrics" in result.__dict__
        assert "treatment_metrics" in result.__dict__
        assert result.recommendation in ["promote", "continue", "stop"]
    
    @pytest.mark.asyncio
    async def test_training_job_configuration(self, training_pipeline):
        """Test training job configuration"""
        # Create training job config
        config = training_pipeline.create_training_job_config(
            model_type=ModelType.COLLABORATIVE_FILTERING,
            model_name="test_cf_model",
            version="1.0.0",
            epochs=10,
            batch_size=128,
            learning_rate=0.001,
            hyperparameter_optimization=False,
            deployment_target="staging"
        )
        
        assert config.model_type == ModelType.COLLABORATIVE_FILTERING
        assert config.model_name == "test_cf_model"
        assert config.version == "1.0.0"
        assert config.training_config.epochs == 10
        assert config.training_config.batch_size == 128
        assert config.deployment_target == "staging"
        assert config.job_id is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics(self, training_pipeline):
        """Test pipeline metrics collection"""
        # Get initial metrics
        metrics = await training_pipeline.get_pipeline_metrics()
        
        assert "total_jobs" in metrics
        assert "completed_jobs" in metrics
        assert "failed_jobs" in metrics
        assert "active_jobs" in metrics
        assert "success_rate" in metrics
        assert "average_duration_seconds" in metrics
        
        # Initially should have no jobs
        assert metrics["total_jobs"] == 0
        assert metrics["active_jobs"] == 0
        assert metrics["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_dashboard_data_generation(self, monitoring_service):
        """Test dashboard data generation"""
        # Setup monitoring
        await monitoring_service.setup_model_monitoring(
            model_name="dashboard_test",
            version="1.0.0",
            model_type="collaborative_filtering",
            monitoring_config={}
        )
        
        # Add some performance data
        snapshot = await monitoring_service.check_model_performance("dashboard_test", "1.0.0")
        
        # Get dashboard data
        dashboard_data = await monitoring_service.get_model_dashboard_data(
            "dashboard_test", "1.0.0"
        )
        
        assert dashboard_data["model_name"] == "dashboard_test"
        assert dashboard_data["version"] == "1.0.0"
        assert "status" in dashboard_data
        assert "current_metrics" in dashboard_data
        assert "performance_history" in dashboard_data
    
    @pytest.mark.asyncio
    async def test_cleanup_and_resource_management(self, monitoring_service, training_pipeline):
        """Test proper cleanup and resource management"""
        # Test monitoring service cleanup
        await monitoring_service.close()
        
        # Test training pipeline cleanup
        await training_pipeline.close()
        
        # Verify resources are cleaned up
        assert len(training_pipeline.active_jobs) == 0


class TestMLTrainingPipelineErrors:
    """Test error handling in ML training pipeline"""
    
    @pytest.mark.asyncio
    async def test_invalid_model_registration(self):
        """Test error handling for invalid model registration"""
        mock_repo = AsyncMock()
        mock_repo.save_model.side_effect = Exception("Database error")
        
        registry = ModelRegistry(mock_repo)
        await registry.initialize()
        
        with pytest.raises(Exception):
            await registry.register_model(
                model_name="invalid_model",
                version="1.0.0",
                model_path="/nonexistent/path",
                metadata={},
                performance_metrics={}
            )
    
    @pytest.mark.asyncio
    async def test_monitoring_with_invalid_data(self):
        """Test monitoring with invalid data"""
        mock_model_repo = AsyncMock()
        mock_registry = AsyncMock()
        
        service = ModelMonitoringService(mock_registry, mock_model_repo)
        await service.initialize()
        
        # Try to log prediction for non-monitored model
        await service.log_prediction(
            model_name="nonexistent",
            version="1.0.0",
            input_features=None,
            prediction=None
        )
        
        # Should not raise exception, just log warning
        assert True  # Test passes if no exception raised
    
    @pytest.mark.asyncio
    async def test_feature_processing_with_empty_dataset(self, mock_repositories):
        """Test feature processing with empty dataset"""
        pipeline = FeatureEngineeringPipeline(
            property_repository=mock_repositories['property_repository'],
            user_repository=mock_repositories['user_repository']
        )
        await pipeline.initialize()
        
        # Create empty dataset
        empty_dataset = MLDataset(
            train_data=TrainingDataBatch(
                user_item_matrix=np.array([]),
                property_features=np.array([]),
                user_features=np.array([]),
                property_metadata=[],
                user_metadata=[],
                feature_names=[],
                batch_info={}
            ),
            validation_data=TrainingDataBatch(
                user_item_matrix=np.array([]),
                property_features=np.array([]),
                user_features=np.array([]),
                property_metadata=[],
                user_metadata=[],
                feature_names=[],
                batch_info={}
            ),
            test_data=None,
            metadata={'total_users': 0, 'total_properties': 0, 'total_interactions': 0}
        )
        
        # Should handle empty dataset gracefully
        result = await pipeline.process_features(empty_dataset, feature_set_name="empty_test")
        
        assert result.feature_set_name == "empty_test"
        assert result.processed_features.size == 0 or result.processed_features.shape[0] == 0


@pytest.fixture
async def mock_repositories():
    """Create mock repositories for testing"""
    mock_model_repo = AsyncMock()
    mock_property_repo = AsyncMock()
    mock_user_repo = AsyncMock()
    
    # Setup mock responses
    mock_model_repo.save_model.return_value = True
    mock_model_repo.load_model.return_value = None
    mock_model_repo.get_model_versions.return_value = []
    mock_model_repo.save_training_metrics.return_value = True
    
    return {
        'model_repository': mock_model_repo,
        'property_repository': mock_property_repo,
        'user_repository': mock_user_repo
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])