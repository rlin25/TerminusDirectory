# Production ML System Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the enterprise-grade ML infrastructure for the rental property recommendation system. The system is designed to handle millions of properties and users with real-time inference capabilities.

## Architecture Overview

### Core Components

1. **Production ML Training Pipeline** (`src/ml/production/`)
   - Distributed training with Ray and Horovod
   - Hyperparameter optimization with Optuna/Ray Tune
   - Advanced validation and cross-validation
   - MLflow integration for experiment tracking

2. **Model Serving Infrastructure** (`src/ml/serving/`)
   - FastAPI-based real-time inference API
   - TensorFlow Serving integration
   - A/B testing framework
   - Intelligent caching and load balancing

3. **MLOps Pipeline** (`mlops/`)
   - MLflow model registry and lifecycle management
   - Automated training schedules and triggers
   - Model monitoring and drift detection
   - CI/CD integration for model deployment

4. **Feature Store** (`src/ml/features/`)
   - Real-time and offline feature serving
   - Feature versioning and lineage tracking
   - Quality monitoring and validation
   - Cross-model feature sharing

5. **Model Optimization** (`src/ml/optimization/`)
   - Quantization and pruning for production
   - TensorFlow Lite and ONNX conversion
   - Hardware acceleration optimization
   - Performance benchmarking

6. **Data Pipeline** (`src/ml/data/`)
   - Kafka-based streaming data processing
   - Real-time feature extraction
   - Data validation and quality monitoring
   - Incremental learning capabilities

## Prerequisites

### System Requirements

- **CPU**: 16+ cores recommended
- **RAM**: 32GB+ recommended
- **Storage**: 500GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Network**: High-bandwidth connection for distributed training

### Software Dependencies

```bash
# Python 3.9+
python --version

# Docker and Docker Compose
docker --version
docker-compose --version

# Kubernetes (for production deployment)
kubectl version
helm version
```

### Required Services

- PostgreSQL 14+
- Redis 7+
- Apache Kafka 3.0+
- MLflow Server
- Prometheus + Grafana (monitoring)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rental-ml-system

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/prod.txt
```

### 2. Environment Configuration

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rental_ml

# Redis
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Kafka
KAFKA_BROKERS=localhost:9092

# API Keys (if needed)
OPENAI_API_KEY=your_key_here
```

### 3. Docker Deployment

```bash
# Start the complete ML infrastructure
docker-compose -f docker-compose.production-ml.yml up -d

# Verify services are running
docker-compose -f docker-compose.production-ml.yml ps

# Check logs
docker-compose -f docker-compose.production-ml.yml logs -f ml-serving
```

### 4. Initialize Database

```bash
# Run migrations
python scripts/init_database.py

# Seed with sample data (optional)
python database/seeds/production_data_seeder.py
```

## Detailed Deployment

### Production Training Pipeline

#### Configuration

```python
from src.ml.production.training_pipeline import ProductionMLPipeline, TrainingConfig

# Create training configuration
config = TrainingConfig(
    model_types=['collaborative', 'content', 'hybrid'],
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    use_distributed=True,
    use_hyperopt=True,
    hyperopt_trials=50,
    use_cross_validation=True,
    cv_folds=5
)
```

#### Running Training

```python
import asyncio

async def train_models():
    pipeline = ProductionMLPipeline(
        database_url=DATABASE_URL,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI
    )
    
    await pipeline.initialize()
    results = await pipeline.train_models(config)
    print("Training Results:", results)

# Run training
asyncio.run(train_models())
```

#### Distributed Training Setup

```bash
# Start Ray cluster
ray start --head --port=6379

# On worker nodes
ray start --address='head-node-ip:6379'

# Run distributed training
python scripts/distributed_training.py
```

### Model Serving

#### Starting the Model Server

```python
from src.ml.serving.model_server import ProductionModelServer

server = ProductionModelServer(
    database_url=DATABASE_URL,
    redis_url=REDIS_URL,
    models_dir="/app/models"
)

# Run server
server.run(host="0.0.0.0", port=8000, workers=4)
```

#### API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
prediction_request = {
    "user_id": "user123",
    "num_recommendations": 10,
    "model_type": "hybrid",
    "include_explanations": True
}

response = requests.post(
    "http://localhost:8000/predict",
    json=prediction_request
)
print(response.json())
```

### Feature Store Setup

#### Initialize Feature Store

```python
from src.ml.features.feature_store import FeatureStore, FeatureGroup
from src.ml.features.feature_registry import FeatureDefinition

# Create feature store
feature_store = FeatureStore(
    database_url=DATABASE_URL,
    redis_url=REDIS_URL,
    storage_path="/app/feature-store"
)

await feature_store.initialize()

# Register feature group
feature_group = FeatureGroup(
    name="user_features",
    entity_key="user_id",
    features=[
        FeatureDefinition(name="age", dtype="int"),
        FeatureDefinition(name="income", dtype="float"),
        FeatureDefinition(name="location", dtype="string")
    ],
    description="User demographic features",
    owner="ml-team"
)

await feature_store.register_feature_group(feature_group)
```

#### Feature Usage

```python
# Write features
features_df = pd.DataFrame({
    "user_id": ["user1", "user2"],
    "age": [25, 30],
    "income": [50000, 60000],
    "location": ["NYC", "SF"]
})

await feature_store.write_features("user_features", features_df)

# Read features for training
training_features = await feature_store.get_features_for_training(
    entity_ids=["user1", "user2"],
    feature_groups=["user_features"],
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

# Read features for inference
inference_features = await feature_store.get_features_for_inference(
    entity_ids=["user1"],
    feature_groups=["user_features"]
)
```

### MLOps Pipeline

#### MLflow Setup

```python
from mlops.mlflow_manager import MLflowManager, ExperimentConfig

# Initialize MLflow manager
mlflow_manager = MLflowManager(
    tracking_uri="http://localhost:5000"
)

await mlflow_manager.initialize()

# Create experiment
experiment_config = ExperimentConfig(
    name="rental_ml_production",
    description="Production ML training for rental recommendations",
    tags={"project": "rental-ml", "environment": "production"}
)

experiment_id = await mlflow_manager.create_experiment_if_not_exists(experiment_config)
```

#### Model Registry

```python
from mlops.mlflow_manager import ModelRegistryConfig

# Register model
registry_config = ModelRegistryConfig(
    model_name="rental_hybrid_recommender",
    stage="Staging",
    description="Hybrid recommendation model for rental properties"
)

version = await mlflow_manager.register_model(
    model_path="/app/models/hybrid_model.h5",
    config=registry_config
)

# Promote to production
await mlflow_manager.promote_model(
    model_name="rental_hybrid_recommender",
    version=version,
    stage="Production"
)
```

### Streaming Data Pipeline

#### Kafka Setup

```bash
# Create topics
kafka-topics --create --topic property_data --bootstrap-server localhost:9092
kafka-topics --create --topic user_interactions --bootstrap-server localhost:9092
kafka-topics --create --topic processed_data --bootstrap-server localhost:9092
```

#### Pipeline Configuration

```python
from src.ml.data.streaming_pipeline import StreamingDataPipeline, StreamConfig

config = StreamConfig(
    kafka_brokers=["localhost:9092"],
    input_topics=["property_data", "user_interactions"],
    output_topics=["processed_data"],
    batch_size=100,
    enable_feature_extraction=True,
    enable_data_validation=True
)

pipeline = StreamingDataPipeline(
    config=config,
    database_url=DATABASE_URL,
    redis_url=REDIS_URL
)

await pipeline.initialize()
await pipeline.start()
```

### Model Optimization

#### Optimization Pipeline

```python
from src.ml.optimization.optimization_pipeline import OptimizationPipeline, OptimizationTarget, OptimizationConfig

# Create optimization pipeline
optimizer = OptimizationPipeline(
    workspace_dir="/tmp/optimization",
    enable_gpu=True
)

# Define target
target = OptimizationTarget(
    target_platform="mobile",
    max_latency_ms=100,
    max_memory_mb=50,
    target_format="tflite"
)

# Configure optimization
config = OptimizationConfig(
    enable_quantization=True,
    quantization_type="int8",
    enable_pruning=True,
    target_sparsity=0.7,
    export_tflite=True
)

# Optimize model
result = await optimizer.optimize_model(
    model_path="/app/models/hybrid_model.h5",
    target=target,
    config=config,
    validation_data=(X_val, y_val)
)

print("Optimization Result:", result.success)
print("Optimized Models:", result.optimized_models)
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl and helm
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl https://get.helm.sh/helm-v3.10.0-linux-amd64.tar.gz | tar xz
```

### Deploy with Helm

```bash
# Add repository
helm repo add rental-ml ./k8s/helm/rental-ml

# Install
helm install rental-ml-prod rental-ml/rental-ml \
  --namespace ml-system \
  --create-namespace \
  --values k8s/helm/rental-ml/values-prod.yaml

# Check deployment
kubectl get pods -n ml-system
kubectl get services -n ml-system
```

### Auto-scaling Configuration

```yaml
# k8s/13-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-serving'
    static_configs:
      - targets: ['ml-serving:8000']
    metrics_path: '/metrics'
    
  - job_name: 'feature-store'
    static_configs:
      - targets: ['feature-store:8001']
    metrics_path: '/metrics'

  - job_name: 'streaming-pipeline'
    static_configs:
      - targets: ['streaming-pipeline:8002']
    metrics_path: '/metrics'
```

### Grafana Dashboards

Key metrics to monitor:

- **Model Serving**:
  - Request latency (p50, p95, p99)
  - Throughput (requests/second)
  - Error rate
  - Cache hit rate
  - Model load time

- **Feature Store**:
  - Feature retrieval latency
  - Feature quality scores
  - Storage utilization
  - Cache performance

- **Training Pipeline**:
  - Training time
  - Model accuracy metrics
  - Resource utilization
  - Data processing speed

- **Streaming Pipeline**:
  - Message throughput
  - Processing latency
  - Error rate
  - Queue depth

### Alerting Rules

```yaml
# monitoring/prometheus/alerts/ml-alerts.yml
groups:
- name: ml_system_alerts
  rules:
  - alert: HighInferenceLatency
    expr: histogram_quantile(0.95, ml_inference_duration_seconds) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High inference latency detected"
      
  - alert: LowCacheHitRate
    expr: ml_cache_hit_rate < 0.7
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Cache hit rate is below threshold"
      
  - alert: HighErrorRate
    expr: rate(ml_inference_requests_total{status="failed"}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate in ML inference"
```

## Performance Tuning

### Model Serving Optimization

```python
# Optimize for high throughput
server_config = {
    "workers": 8,  # Number of worker processes
    "max_batch_size": 64,  # Batch requests
    "cache_ttl": 300,  # Cache TTL in seconds
    "prefetch_size": 100,  # Prefetch queue size
    "enable_tf_serving": True,  # Use TensorFlow Serving
}
```

### Database Optimization

```sql
-- Create indexes for faster queries
CREATE INDEX CONCURRENTLY idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX CONCURRENTLY idx_properties_location ON properties USING GIN(location);
CREATE INDEX CONCURRENTLY idx_features_timestamp ON features(created_at);

-- Partition large tables
CREATE TABLE user_interactions_2024 PARTITION OF user_interactions 
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### Redis Optimization

```bash
# Redis configuration for ML workloads
echo "maxmemory 8gb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
echo "save 900 1" >> /etc/redis/redis.conf
echo "stop-writes-on-bgsave-error no" >> /etc/redis/redis.conf
```

## Security Configuration

### API Security

```python
from fastapi.security import HTTPBearer
from fastapi import Depends, HTTPException

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Implement JWT verification
    if not verify_jwt_token(token.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token
```

### Database Security

```yaml
# Environment variables for secrets
environment:
  - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
  - REDIS_PASSWORD_FILE=/run/secrets/redis_password
  - JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret

secrets:
  postgres_password:
    external: true
  redis_password:
    external: true
  jwt_secret:
    external: true
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Reduce batch size
   export BATCH_SIZE=64
   
   # Enable model quantization
   export ENABLE_QUANTIZATION=true
   ```

2. **Slow Inference**
   ```bash
   # Enable GPU acceleration
   export CUDA_VISIBLE_DEVICES=0
   
   # Use TensorFlow Serving
   export ENABLE_TF_SERVING=true
   
   # Optimize model
   python src/ml/optimization/optimize_model.py
   ```

3. **Training Failures**
   ```bash
   # Check logs
   docker logs ml-training
   
   # Reduce learning rate
   export LEARNING_RATE=0.0001
   
   # Enable gradient clipping
   export GRADIENT_CLIPPING=1.0
   ```

### Log Analysis

```bash
# Check service logs
docker-compose -f docker-compose.production-ml.yml logs -f ml-serving

# Monitor system metrics
docker exec -it ml-serving python -c "
import psutil
print('CPU:', psutil.cpu_percent())
print('Memory:', psutil.virtual_memory().percent)
"

# Check model performance
curl http://localhost:8000/metrics | grep ml_inference
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
pg_dump -h localhost -U postgres rental_ml > backup_$(date +%Y%m%d).sql

# Restore backup
psql -h localhost -U postgres rental_ml < backup_20241213.sql
```

### Model Backup

```bash
# Backup models and artifacts
tar -czf models_backup_$(date +%Y%m%d).tar.gz /app/models /app/mlflow-artifacts

# Restore models
tar -xzf models_backup_20241213.tar.gz -C /
```

## Scaling Guidelines

### Horizontal Scaling

- **API Servers**: Scale based on CPU/memory usage (target: 70% CPU)
- **Training Workers**: Scale based on queue depth and training load
- **Feature Store**: Scale read replicas for high query load

### Vertical Scaling

- **Memory**: 32GB+ for training, 16GB+ for serving
- **CPU**: 16+ cores for training, 8+ cores for serving
- **GPU**: Multiple GPUs for distributed training

### Load Testing

```python
import asyncio
import aiohttp
import time

async def load_test():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(1000):
            task = session.post(
                'http://localhost:8000/predict',
                json={'user_id': f'user_{i}', 'num_recommendations': 10}
            )
            tasks.append(task)
        
        start = time.time()
        responses = await asyncio.gather(*tasks)
        end = time.time()
        
        print(f"Processed {len(responses)} requests in {end-start:.2f}s")
        print(f"Throughput: {len(responses)/(end-start):.2f} req/s")

asyncio.run(load_test())
```

## Maintenance

### Regular Tasks

1. **Model Retraining** (Weekly)
   ```bash
   # Automated retraining
   python scripts/automated_retraining.py
   ```

2. **Feature Store Cleanup** (Daily)
   ```bash
   # Clean old features
   python src/ml/features/cleanup_old_features.py
   ```

3. **Cache Optimization** (Daily)
   ```bash
   # Redis memory optimization
   redis-cli MEMORY PURGE
   ```

4. **Log Rotation** (Daily)
   ```bash
   # Rotate logs
   logrotate /etc/logrotate.d/ml_system
   ```

### Performance Monitoring

- Monitor key metrics continuously
- Set up automated alerts for anomalies
- Regular performance reviews and optimization
- Capacity planning based on usage trends

## Support and Documentation

- **API Documentation**: Available at `http://localhost:8000/docs`
- **Monitoring Dashboard**: Available at `http://localhost:3000`
- **MLflow UI**: Available at `http://localhost:5000`
- **Feature Store API**: Available at `http://localhost:8001/docs`

For additional support, please refer to the comprehensive documentation in the `docs/` directory or contact the ML engineering team.