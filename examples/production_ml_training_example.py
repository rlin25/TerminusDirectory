#!/usr/bin/env python3
"""
Production ML Training Pipeline Example

This example demonstrates how to use the complete ML training pipeline
including feature engineering, model training, monitoring, and deployment.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.ml_training import (
    ProductionTrainingPipeline,
    ModelRegistry,
    FeatureEngineeringPipeline,
    ModelMonitoringService,
    TrainingJobConfig,
    ModelType,
    TrainingStatus
)
from src.infrastructure.data.config import DataConfig
from src.infrastructure.data.repository_factory import get_repository_factory


async def example_training_pipeline():
    """Example of using the production training pipeline"""
    print("=== Production ML Training Pipeline Example ===\n")
    
    try:
        # 1. Setup configuration
        print("1. Setting up configuration...")
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rental_ml")
        models_dir = "/tmp/rental_ml_models"
        artifacts_dir = "/tmp/rental_ml_artifacts"
        
        # 2. Initialize repository factory
        print("2. Initializing repository factory...")
        try:
            repository_factory = await get_repository_factory()
            model_repository = repository_factory.get_model_repository()
            property_repository = repository_factory.get_property_repository()
            user_repository = repository_factory.get_user_repository()
            print("   ✓ Repository factory initialized")
        except Exception as e:
            print(f"   ✗ Repository factory failed: {e}")
            print("   Note: This is expected if database is not running")
            # Create mock repositories for demo
            model_repository = None
            property_repository = None
            user_repository = None
        
        # 3. Initialize core components
        print("3. Initializing core components...")
        
        # Model Registry
        if model_repository:
            model_registry = ModelRegistry(model_repository)
            await model_registry.initialize()
            print("   ✓ Model registry initialized")
        else:
            print("   ⚠ Model registry skipped (no database connection)")
            model_registry = None
        
        # Feature Engineering Pipeline
        if property_repository and user_repository:
            feature_pipeline = FeatureEngineeringPipeline(
                property_repository=property_repository,
                user_repository=user_repository
            )
            await feature_pipeline.initialize()
            print("   ✓ Feature engineering pipeline initialized")
        else:
            print("   ⚠ Feature pipeline skipped (no database connection)")
            feature_pipeline = None
        
        # Model Monitoring Service
        if model_registry and model_repository:
            monitoring_service = ModelMonitoringService(
                model_registry=model_registry,
                model_repository=model_repository
            )
            await monitoring_service.initialize()
            print("   ✓ Model monitoring service initialized")
        else:
            print("   ⚠ Monitoring service skipped (no database connection)")
            monitoring_service = None
        
        # Production Training Pipeline
        training_pipeline = ProductionTrainingPipeline(
            database_url=database_url,
            models_dir=models_dir,
            artifacts_dir=artifacts_dir,
            enable_monitoring=monitoring_service is not None
        )
        
        try:
            await training_pipeline.initialize()
            print("   ✓ Training pipeline initialized")
        except Exception as e:
            print(f"   ⚠ Training pipeline initialization failed: {e}")
            print("   Note: This is expected without a complete database setup")
        
        # 4. Create training job configuration
        print("4. Creating training job configuration...")
        
        job_config = training_pipeline.create_training_job_config(
            model_type=ModelType.COLLABORATIVE_FILTERING,
            model_name="rental_cf_model",
            version="1.0.0",
            epochs=50,
            batch_size=256,
            learning_rate=0.001,
            hyperparameter_optimization=True,
            deployment_target="staging",
            monitoring_enabled=True
        )
        
        print(f"   ✓ Training job created: {job_config.job_id}")
        print(f"   ✓ Model type: {job_config.model_type.value}")
        print(f"   ✓ Model name: {job_config.model_name}")
        print(f"   ✓ Version: {job_config.version}")
        
        # 5. Demonstrate model registry functionality
        if model_registry:
            print("5. Demonstrating model registry functionality...")
            
            # Create a sample model version
            sample_metadata = {
                "training_time": datetime.utcnow().isoformat(),
                "framework": "tensorflow",
                "dataset_size": 10000
            }
            
            sample_metrics = {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            }
            
            try:
                model_id = await model_registry.register_model(
                    model_name="sample_model",
                    version="1.0.0",
                    model_path="/tmp/sample_model.pkl",
                    metadata=sample_metadata,
                    performance_metrics=sample_metrics,
                    description="Sample model for demonstration"
                )
                print(f"   ✓ Model registered with ID: {model_id}")
                
                # List model versions
                versions = await model_registry.list_model_versions("sample_model")
                print(f"   ✓ Found {len(versions)} model versions")
                
                # Get deployment status
                deployment_status = await model_registry.get_deployment_status("sample_model")
                print(f"   ✓ Deployment status retrieved")
                
            except Exception as e:
                print(f"   ⚠ Model registry operations failed: {e}")
        else:
            print("5. Model registry functionality skipped (no database connection)")
        
        # 6. Demonstrate feature engineering
        if feature_pipeline:
            print("6. Demonstrating feature engineering functionality...")
            
            try:
                # Get feature definitions
                feature_info = feature_pipeline.get_feature_set_info("default")
                if feature_info:
                    print(f"   ✓ Feature set info retrieved: {feature_info.get('name', 'Unknown')}")
                else:
                    print("   ⚠ No feature set found")
                
                # Get feature importance (if available)
                importance = feature_pipeline.get_feature_importance()
                print(f"   ✓ Feature importance: {len(importance)} features")
                
            except Exception as e:
                print(f"   ⚠ Feature engineering operations failed: {e}")
        else:
            print("6. Feature engineering functionality skipped (no database connection)")
        
        # 7. Demonstrate monitoring functionality
        if monitoring_service:
            print("7. Demonstrating monitoring functionality...")
            
            try:
                # Setup monitoring for a model
                await monitoring_service.setup_model_monitoring(
                    model_name="sample_model",
                    version="1.0.0",
                    model_type="collaborative_filtering",
                    monitoring_config={
                        'data_drift_threshold': 0.1,
                        'performance_thresholds': {'accuracy': 0.8},
                        'check_frequency': 'hourly'
                    }
                )
                print("   ✓ Model monitoring setup completed")
                
                # Get dashboard data
                dashboard_data = await monitoring_service.get_model_dashboard_data(
                    "sample_model", "1.0.0"
                )
                print(f"   ✓ Dashboard data retrieved: Status = {dashboard_data.get('status', 'unknown')}")
                
            except Exception as e:
                print(f"   ⚠ Monitoring operations failed: {e}")
        else:
            print("7. Monitoring functionality skipped (no database connection)")
        
        # 8. Get pipeline metrics
        print("8. Getting pipeline metrics...")
        
        try:
            metrics = await training_pipeline.get_pipeline_metrics()
            print(f"   ✓ Pipeline metrics retrieved:")
            print(f"      - Total jobs: {metrics.get('total_jobs', 0)}")
            print(f"      - Success rate: {metrics.get('success_rate', 0):.2%}")
            print(f"      - Active jobs: {metrics.get('active_jobs', 0)}")
        except Exception as e:
            print(f"   ⚠ Pipeline metrics failed: {e}")
        
        # 9. Cleanup
        print("9. Cleaning up resources...")
        
        try:
            await training_pipeline.close()
            print("   ✓ Training pipeline closed")
        except Exception as e:
            print(f"   ⚠ Training pipeline cleanup failed: {e}")
        
        if monitoring_service:
            try:
                await monitoring_service.close()
                print("   ✓ Monitoring service closed")
            except Exception as e:
                print(f"   ⚠ Monitoring service cleanup failed: {e}")
        
        if repository_factory and repository_factory.is_initialized():
            try:
                await repository_factory.close()
                print("   ✓ Repository factory closed")
            except Exception as e:
                print(f"   ⚠ Repository factory cleanup failed: {e}")
        
        print("\n=== Example completed successfully! ===")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


async def example_ab_testing():
    """Example of A/B testing functionality"""
    print("\n=== A/B Testing Example ===\n")
    
    try:
        # Create a mock monitoring service for A/B testing
        print("1. Setting up A/B testing...")
        
        # This would normally be connected to a real monitoring service
        print("   ⚠ A/B testing requires a full monitoring setup")
        print("   This is a simplified demonstration")
        
        # Example A/B test configuration
        test_config = {
            'test_id': 'model_comparison_test_001',
            'model_name': 'rental_recommender',
            'control_version': '1.0.0',
            'treatment_version': '1.1.0',
            'traffic_split': {'1.0.0': 50.0, '1.1.0': 50.0},
            'success_metrics': ['accuracy', 'precision', 'recall'],
            'duration_days': 7
        }
        
        print(f"   ✓ A/B test configured: {test_config['test_id']}")
        print(f"   ✓ Control: {test_config['control_version']}")
        print(f"   ✓ Treatment: {test_config['treatment_version']}")
        print(f"   ✓ Traffic split: {test_config['traffic_split']}")
        print(f"   ✓ Success metrics: {test_config['success_metrics']}")
        print(f"   ✓ Duration: {test_config['duration_days']} days")
        
        print("\n=== A/B Testing example completed! ===")
        
    except Exception as e:
        print(f"\n❌ A/B Testing example failed: {e}")


async def main():
    """Main example function"""
    print("Production ML Training Pipeline - Complete Example")
    print("=" * 60)
    
    # Run the main training pipeline example
    await example_training_pipeline()
    
    # Run the A/B testing example
    await example_ab_testing()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nNext Steps:")
    print("1. Set up PostgreSQL database with the schema")
    print("2. Set up Redis for caching")
    print("3. Configure environment variables")
    print("4. Run actual training jobs with real data")
    print("5. Set up monitoring dashboards")


if __name__ == "__main__":
    asyncio.run(main())