# Production ML Training Pipeline Documentation

This document provides comprehensive documentation for the production-ready ML training pipeline implemented for the rental property recommendation system.

## Overview

The ML training pipeline consists of four main components:

1. **Production Training Pipeline** - Orchestrates the entire ML training lifecycle
2. **Model Registry** - Manages model versions, deployment, and metadata
3. **Feature Engineering Pipeline** - Handles feature extraction, transformation, and real-time processing
4. **Model Monitoring Service** - Provides comprehensive monitoring, alerting, and A/B testing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Production Training Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Feature         │  │ Model           │  │ Model           │ │
│  │ Engineering     │  │ Registry        │  │ Monitoring      │ │
│  │ Pipeline        │  │                 │  │ Service         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Domain Layer Integration                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Property        │  │ User            │  │ SearchQuery     │ │
│  │ Repository      │  │ Repository      │  │ Repository      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ PostgreSQL      │  │ Redis           │  │ ML Models       │ │
│  │ Database        │  │ Cache           │  │ (TensorFlow)    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Production Training Pipeline

**File**: `src/application/ml_training/production_training_pipeline.py`

The main orchestrator that handles the complete ML training lifecycle:

#### Key Features:
- **Automated Training Jobs**: Submit and track training jobs with comprehensive configuration
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning
- **Model Validation**: Multiple validation strategies (holdout, cross-validation, time-series)
- **Deployment Integration**: Automatic deployment to staging/production environments
- **MLflow Integration**: Experiment tracking and model versioning
- **Resource Management**: Efficient memory and compute resource utilization

#### Usage Example:
```python
from src.application.ml_training import ProductionTrainingPipeline, ModelType

# Initialize pipeline
pipeline = ProductionTrainingPipeline(
    database_url="postgresql://user:pass@host:5432/rental_ml",
    models_dir="/path/to/models",
    enable_monitoring=True
)
await pipeline.initialize()

# Create training job
config = pipeline.create_training_job_config(
    model_type=ModelType.COLLABORATIVE_FILTERING,
    model_name="rental_cf_model",
    version="1.0.0",
    epochs=100,
    hyperparameter_optimization=True,
    deployment_target="staging"
)

# Submit job
job_id = await pipeline.submit_training_job(config)

# Monitor progress
status = await pipeline.get_training_job_status(job_id)
```

#### Training Job Lifecycle:
1. **Data Preparation**: Load and validate training data
2. **Feature Engineering**: Process features using the feature pipeline
3. **Hyperparameter Optimization**: Optimize model parameters (if enabled)
4. **Model Training**: Train the model with full configuration
5. **Model Validation**: Validate performance using specified strategy
6. **Model Registration**: Register model in the registry
7. **Model Deployment**: Deploy to target environment
8. **Monitoring Setup**: Configure monitoring for deployed model

### 2. Model Registry

**File**: `src/application/ml_training/model_registry.py`

Comprehensive model version management and deployment tracking:

#### Key Features:
- **Version Management**: Semantic versioning with lineage tracking
- **Metadata Tracking**: Performance metrics, training data, hyperparameters
- **Deployment Management**: Track deployments across environments
- **Model Comparison**: Statistical comparison between model versions
- **A/B Testing Support**: Configure and manage A/B tests
- **Rollback Capabilities**: Quick rollback to previous versions
- **Model Signing**: Security and integrity validation

#### Usage Example:
```python
from src.application.ml_training import ModelRegistry

# Initialize registry
registry = ModelRegistry(model_repository)
await registry.initialize()

# Register a model
model_id = await registry.register_model(
    model_name="recommendation_model",
    version="1.2.0",
    model_path="/path/to/model.pkl",
    metadata={"framework": "tensorflow", "dataset_size": 100000},
    performance_metrics={"accuracy": 0.87, "precision": 0.84},
    parent_version="1.1.0"  # For lineage tracking
)

# Deploy to staging
await registry.deploy_to_staging("recommendation_model", "1.2.0")

# Compare with previous version
comparison = await registry.compare_models(
    "recommendation_model", "1.1.0", "1.2.0"
)

# Create A/B test
test_id = await registry.create_experiment(
    model_name="recommendation_model",
    baseline_version="1.1.0",
    candidate_version="1.2.0",
    traffic_split={"1.1.0": 70.0, "1.2.0": 30.0},
    success_criteria={"accuracy": 0.85, "latency": 100}
)
```

#### Model States:
- **REGISTERED**: Model is registered but not deployed
- **STAGING**: Model is deployed to staging environment
- **PRODUCTION**: Model is deployed to production environment
- **CANARY**: Model is deployed with limited traffic
- **ARCHIVED**: Model is archived but kept for reference
- **DEPRECATED**: Model is deprecated and should not be used

### 3. Feature Engineering Pipeline

**File**: `src/application/ml_training/feature_engineering.py`

Production-ready feature processing for both batch and real-time scenarios:

#### Key Features:
- **Automated Feature Extraction**: Extract features from property and user data
- **Feature Scaling**: Multiple scaling methods (StandardScaler, MinMaxScaler, RobustScaler)
- **Feature Selection**: Automated feature selection using statistical methods
- **Real-time Processing**: Process features for individual predictions
- **Feature Store Integration**: Connect to external feature stores
- **Quality Validation**: Comprehensive feature quality checks
- **Caching**: Intelligent caching for performance optimization

#### Usage Example:
```python
from src.application.ml_training import FeatureEngineeringPipeline

# Initialize pipeline
pipeline = FeatureEngineeringPipeline(
    property_repository=property_repo,
    user_repository=user_repo
)
await pipeline.initialize()

# Process features for training
result = await pipeline.process_features(
    dataset=training_dataset,
    feature_set_name="training_features"
)

# Real-time feature processing
features = await pipeline.process_real_time_features(
    property_data={"price": 1200, "bedrooms": 2, "bathrooms": 1.5},
    user_data={"min_price": 1000, "max_price": 1500},
    feature_set_name="inference_features"
)

# Get feature importance
importance = pipeline.get_feature_importance("training_features")
```

#### Feature Types Supported:
- **Numerical**: Price, square footage, ratings
- **Categorical**: Property type, location, amenities
- **Text**: Property descriptions, amenity lists (TF-IDF)
- **Temporal**: Listing age, seasonal features
- **Geospatial**: Location clustering, distance features
- **Interaction**: User-property interaction patterns

### 4. Model Monitoring Service

**File**: `src/application/ml_training/model_monitoring.py`

Comprehensive monitoring for deployed ML models:

#### Key Features:
- **Real-time Monitoring**: Track predictions, latency, and errors
- **Data Drift Detection**: Statistical methods for detecting feature drift
- **Performance Alerting**: Configurable alerts for model degradation
- **A/B Testing Framework**: Complete A/B testing workflow
- **Dashboard Integration**: Generate data for monitoring dashboards
- **Automated Remediation**: Trigger retraining when thresholds are exceeded

#### Usage Example:
```python
from src.application.ml_training import ModelMonitoringService

# Initialize service
service = ModelMonitoringService(
    model_registry=registry,
    model_repository=model_repo
)
await service.initialize()

# Setup monitoring
await service.setup_model_monitoring(
    model_name="recommendation_model",
    version="1.2.0",
    model_type="collaborative_filtering",
    monitoring_config={
        'data_drift_threshold': 0.1,
        'performance_thresholds': {'accuracy': 0.8},
        'check_frequency': 'hourly'
    }
)

# Log predictions
await service.log_prediction(
    model_name="recommendation_model",
    version="1.2.0",
    input_features=feature_vector,
    prediction=model_output,
    actual_value=ground_truth,
    latency_ms=response_time
)

# Check performance
snapshot = await service.check_model_performance(
    "recommendation_model", "1.2.0"
)

# Detect data drift
drift_results = await service.detect_data_drift(
    "recommendation_model", "1.2.0"
)
```

#### Monitoring Metrics:
- **Performance**: Accuracy, precision, recall, F1-score, MSE, MAE
- **Operational**: Latency percentiles, throughput, error rates
- **Data Quality**: Missing values, feature variance, correlations
- **Data Drift**: KS test, Population Stability Index, Jensen-Shannon divergence

## Integration with Existing System

### Domain Layer Integration

The ML training pipeline integrates seamlessly with the existing domain entities:

- **Property Entity**: Used for feature extraction and training data
- **User Entity**: User preferences and interaction history for personalization
- **SearchQuery Entity**: Query patterns for search ranking models

### Repository Integration

- **PropertyRepository**: Access to property data for training
- **UserRepository**: User data and interaction history
- **ModelRepository**: Store and retrieve trained models

### Infrastructure Integration

- **PostgreSQL**: Store training data, model metadata, and monitoring data
- **Redis**: Cache features and model predictions
- **ML Models**: Integrate with existing collaborative filtering, content-based, and hybrid models

## Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rental_ml
DB_USERNAME=postgres
DB_PASSWORD=password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ML Training Configuration
ML_MODELS_DIR=/data/ml_models
ML_ARTIFACTS_DIR=/data/ml_artifacts
MLFLOW_TRACKING_URI=http://localhost:5000

# Monitoring Configuration
MONITORING_CHECK_FREQUENCY=300
ALERT_RETENTION_DAYS=30
PERFORMANCE_RETENTION_DAYS=90
```

### Training Configuration

```python
from src.application.ml_training import TrainingJobConfig, ModelType

config = TrainingJobConfig(
    job_id="training_job_001",
    model_type=ModelType.HYBRID,
    model_name="rental_hybrid_model",
    version="2.0.0",
    training_config=TrainingConfig(
        epochs=100,
        batch_size=256,
        learning_rate=0.001,
        validation_split=0.2,
        early_stopping_patience=10
    ),
    hyperparameter_optimization=True,
    deployment_target="staging",
    monitoring_enabled=True,
    performance_threshold={"accuracy": 0.85, "latency": 100}
)
```

## Testing

### Unit Tests
- Individual component testing
- Mock repository integration
- Error handling validation

### Integration Tests
- End-to-end pipeline testing
- Database integration validation
- Real-time processing verification

### Performance Tests
- Load testing for high-volume predictions
- Memory usage optimization
- Latency benchmarking

### Running Tests

```bash
# Run all ML training pipeline tests
pytest tests/integration/test_ml_training_pipeline.py -v

# Run with coverage
pytest tests/integration/test_ml_training_pipeline.py --cov=src.application.ml_training

# Run specific test categories
pytest tests/integration/test_ml_training_pipeline.py::TestMLTrainingPipelineIntegration -v
```

## Deployment

### Production Deployment

1. **Infrastructure Setup**:
   ```bash
   # Deploy PostgreSQL with extensions
   kubectl apply -f k8s/05-postgres-deployment.yaml
   
   # Deploy Redis
   kubectl apply -f k8s/07-redis-deployment.yaml
   
   # Deploy ML training workers
   kubectl apply -f k8s/09-worker-deployment.yaml
   ```

2. **Configuration**:
   ```bash
   # Create configuration secrets
   kubectl create secret generic ml-training-config \
     --from-env-file=config/ml-training.env
   ```

3. **Scaling**:
   ```yaml
   # k8s/13-hpa.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: ml-training-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: ml-training-worker
     minReplicas: 2
     maxReplicas: 10
   ```

### Monitoring and Observability

1. **Prometheus Metrics**:
   - Training job success/failure rates
   - Model performance metrics
   - Data drift scores
   - Prediction latency

2. **Grafana Dashboards**:
   - Model performance trends
   - Training pipeline health
   - Resource utilization
   - Alert status

3. **Logging**:
   - Structured logging with correlation IDs
   - Error tracking and alerting
   - Audit logs for model deployments

## Performance Considerations

### Scalability
- **Horizontal Scaling**: Multiple training workers
- **Vertical Scaling**: GPU support for large models
- **Data Partitioning**: Distribute training data across workers

### Optimization
- **Caching**: Feature caching for repeated training jobs
- **Batch Processing**: Efficient batch prediction processing
- **Memory Management**: Streaming data processing for large datasets

### Security
- **Model Signing**: Cryptographic signatures for model integrity
- **Access Control**: Role-based access to training pipeline
- **Data Privacy**: PII handling and anonymization

## Troubleshooting

### Common Issues

1. **Training Job Failures**:
   - Check data quality and availability
   - Verify resource limits and allocation
   - Review hyperparameter constraints

2. **Monitoring Alerts**:
   - Investigate data drift sources
   - Check model performance degradation
   - Verify infrastructure health

3. **Deployment Issues**:
   - Validate model compatibility
   - Check environment configuration
   - Verify database connectivity

### Debug Commands

```bash
# Check training job status
python -c "
import asyncio
from src.application.ml_training import ProductionTrainingPipeline
async def check_job():
    pipeline = ProductionTrainingPipeline('postgresql://...')
    await pipeline.initialize()
    status = await pipeline.get_training_job_status('job_id')
    print(status)
asyncio.run(check_job())
"

# Validate feature processing
python -c "
import asyncio
from src.application.ml_training import FeatureEngineeringPipeline
# ... validation code
"

# Check monitoring health
python -c "
import asyncio
from src.application.ml_training import ModelMonitoringService
# ... health check code
"
```

## Future Enhancements

### Planned Features
1. **AutoML Integration**: Automated model selection and tuning
2. **Federated Learning**: Distributed training across data sources
3. **Real-time Streaming**: Kafka integration for streaming data
4. **Advanced Monitoring**: Explainability and fairness metrics
5. **Multi-tenant Support**: Isolated training environments

### Performance Improvements
1. **GPU Acceleration**: CUDA support for training
2. **Distributed Training**: Multi-node training capabilities
3. **Model Compression**: Quantization and pruning
4. **Edge Deployment**: Model optimization for edge devices

## Support and Documentation

### Additional Resources
- [Model Architecture Documentation](ML_MODELS_ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Performance Tuning](PERFORMANCE_TUNING.md)

### Community
- GitHub Issues for bug reports
- Documentation contributions welcome
- Performance benchmarking results

---

This production ML training pipeline provides a comprehensive, scalable, and maintainable solution for training, deploying, and monitoring machine learning models in the rental property recommendation system. The clean architecture ensures easy testing, maintenance, and future enhancements.