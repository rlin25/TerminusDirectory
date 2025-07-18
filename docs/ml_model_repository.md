# ML Model Repository Implementation

## Overview

The PostgreSQL Model Repository provides a comprehensive, production-ready implementation for storing and managing ML models, embeddings, training metrics, and prediction caching. This implementation includes advanced features like connection pooling, performance monitoring, batch operations, and comprehensive error handling.

## Key Features

### 1. Model Storage and Versioning
- **Versioned Model Storage**: Store multiple versions of ML models with metadata
- **Soft Deletion**: Models are marked inactive rather than physically deleted
- **Automatic Cleanup**: Remove old model versions while keeping specified number of recent versions
- **Metadata Tracking**: Store model size, dependencies, creation time, and custom metadata

### 2. Embeddings Management
- **Entity Embeddings**: Store embeddings for users, properties, and other entities
- **Batch Operations**: Efficiently save multiple embeddings in a single transaction
- **Dimension Validation**: Ensure embedding dimensions are consistent
- **Update Tracking**: Track when embeddings were last updated

### 3. Prediction Caching
- **TTL-based Caching**: Cache predictions with configurable time-to-live
- **Automatic Expiration**: Expired cache entries are automatically cleaned up
- **Access Tracking**: Monitor cache hit patterns and usage statistics
- **Pattern-based Clearing**: Clear cache entries matching specific patterns

### 4. Training Metrics
- **Metrics Storage**: Store training metrics (accuracy, loss, etc.) for each model version
- **Validation**: Ensure metrics are numeric and finite
- **Historical Tracking**: Maintain history of training metrics across versions

### 5. Performance Features
- **Connection Pooling**: Optimized database connection management
- **Query Performance Monitoring**: Track query execution times and identify slow operations
- **Batch Processing**: Efficient bulk operations for large datasets
- **Database Indexes**: Optimized indexes for fast lookups and aggregations

### 6. Production-Ready Features
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Logging**: Detailed logging for debugging and monitoring
- **Health Checks**: Built-in health check functionality
- **Database Optimization**: Automatic database maintenance and optimization

## Usage Examples

### Basic Model Operations

```python
from src.infrastructure.data.repositories.postgres_model_repository import PostgresModelRepository

# Initialize repository
repo = PostgresModelRepository(
    database_url="postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=40,
    enable_metrics=True
)

# Save a model
model_data = trained_model  # Your trained ML model
success = await repo.save_model(
    model_name="recommendation_model",
    model_data=model_data,
    version="1.0"
)

# Load a model
model = await repo.load_model("recommendation_model", "1.0")
# Or load latest version
model = await repo.load_model("recommendation_model", "latest")

# Get all versions
versions = await repo.get_model_versions("recommendation_model")

# Delete a model version
await repo.delete_model("recommendation_model", "1.0")
```

### Embeddings Operations

```python
import numpy as np

# Save embeddings for a user
user_embeddings = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
success = await repo.save_embeddings(
    entity_type="user",
    entity_id="user123",
    embeddings=user_embeddings
)

# Get embeddings for a specific entity
embeddings = await repo.get_embeddings("user", "user123")

# Get all embeddings for an entity type
all_user_embeddings = await repo.get_all_embeddings("user")

# Batch save embeddings
embeddings_data = [
    ("user", "user1", np.array([0.1, 0.2, 0.3])),
    ("user", "user2", np.array([0.4, 0.5, 0.6])),
    ("property", "prop1", np.array([0.7, 0.8, 0.9]))
]
results = await repo.batch_save_embeddings(embeddings_data)
```

### Prediction Caching

```python
# Cache predictions
predictions = {
    "recommendations": [1, 2, 3, 4, 5],
    "scores": [0.9, 0.8, 0.7, 0.6, 0.5]
}

success = await repo.cache_predictions(
    cache_key="user123_recommendations",
    predictions=predictions,
    ttl_seconds=3600  # 1 hour
)

# Get cached predictions
cached_predictions = await repo.get_cached_predictions("user123_recommendations")

# Clear cache
await repo.clear_cache("user123_*")  # Clear all cache entries for user123
await repo.clear_cache("*")  # Clear all cache entries
```

### Training Metrics

```python
# Save training metrics
metrics = {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90,
    "loss": 0.05
}

success = await repo.save_training_metrics(
    model_name="recommendation_model",
    version="1.0",
    metrics=metrics
)

# Get training metrics
metrics = await repo.get_training_metrics("recommendation_model", "1.0")
```

### Monitoring and Maintenance

```python
# Health check
health_status = await repo.health_check()

# Get performance metrics
performance_metrics = await repo.get_performance_metrics()

# Get cache statistics
cache_stats = await repo.get_cache_statistics()

# Get model storage statistics
model_stats = await repo.get_model_storage_stats()

# Get embeddings statistics
embeddings_stats = await repo.get_embeddings_stats()

# Clean up old models
deleted_count = await repo.cleanup_old_models("recommendation_model", keep_versions=5)

# Optimize database
optimization_results = await repo.optimize_database()
```

## Database Schema

### ML Models Table (`ml_models`)
```sql
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_data BYTEA NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT uq_model_name_version UNIQUE (model_name, version)
);
```

### Embeddings Table (`embeddings`)
```sql
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    embeddings BYTEA NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_entity_type_id UNIQUE (entity_type, entity_id)
);
```

### Prediction Cache Table (`prediction_cache`)
```sql
CREATE TABLE prediction_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(512) NOT NULL UNIQUE,
    predictions BYTEA NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT prediction_cache_expires_after_cached CHECK (expires_at > cached_at)
);
```

### Training Metrics Table (`training_metrics`)
```sql
CREATE TABLE training_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    training_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## Performance Optimizations

### Database Indexes
- **Composite Indexes**: Optimized for common query patterns
- **GIN Indexes**: For JSON/JSONB metadata searching
- **Partial Indexes**: For active records only
- **Covering Indexes**: Include frequently accessed columns

### Connection Pooling
- **Async Connection Pool**: Efficient connection management
- **Pool Monitoring**: Track connection usage and performance
- **Connection Recycling**: Prevent stale connections

### Query Optimization
- **Batch Operations**: Minimize database round trips
- **Prepared Statements**: Improved query performance
- **Query Timeout**: Prevent long-running queries
- **Connection Timeout**: Handle connection failures gracefully

## Error Handling

### Retry Mechanisms
- **Exponential Backoff**: Intelligent retry strategies
- **Transient Error Detection**: Retry only recoverable errors
- **Maximum Retry Limits**: Prevent infinite retry loops

### Validation
- **Input Validation**: Validate all input parameters
- **Data Type Checking**: Ensure correct data types
- **Constraint Validation**: Check business rules

### Logging
- **Structured Logging**: Consistent log format
- **Error Context**: Include relevant context in error messages
- **Performance Logging**: Track slow operations

## Monitoring

### Performance Metrics
- **Query Performance**: Track execution times
- **Resource Usage**: Monitor memory and CPU usage
- **Error Rates**: Track error frequency

### Health Checks
- **Database Connectivity**: Verify database connection
- **Table Access**: Test table accessibility
- **Query Performance**: Check query response times

### Statistics
- **Cache Hit Rates**: Monitor cache effectiveness
- **Storage Usage**: Track database storage consumption
- **Model Usage**: Monitor model access patterns

## Migration

To set up the database schema, run the migration script:

```bash
psql -d your_database -f migrations/005_ml_prediction_cache.sql
```

This will create all necessary tables, indexes, and functions.

## Best Practices

1. **Model Versioning**: Use semantic versioning for models
2. **Cache Keys**: Use descriptive, hierarchical cache keys
3. **Cleanup**: Regularly clean up old models and expired cache entries
4. **Monitoring**: Set up monitoring for performance and errors
5. **Testing**: Test with realistic data volumes
6. **Backup**: Regular backups of model data and metrics

## Configuration

### Environment Variables
```bash
# Database configuration
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30

# Cache configuration
CACHE_DEFAULT_TTL=3600
CACHE_MAX_TTL=86400

# Performance configuration
QUERY_TIMEOUT=30
ENABLE_METRICS=true
```

### Python Configuration
```python
repo = PostgresModelRepository(
    database_url=os.getenv("DATABASE_URL"),
    pool_size=int(os.getenv("DB_POOL_SIZE", 20)),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", 40)),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", 30)),
    enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true"
)
```

## Troubleshooting

### Common Issues

1. **Connection Pool Exhaustion**
   - Increase pool size or max overflow
   - Check for connection leaks
   - Verify proper session cleanup

2. **Slow Queries**
   - Review query execution plans
   - Check index usage
   - Consider query optimization

3. **Memory Issues**
   - Monitor model and cache sizes
   - Implement cleanup strategies
   - Consider compression

4. **Cache Misses**
   - Review cache key patterns
   - Adjust TTL values
   - Monitor cache statistics

### Debugging

Enable detailed logging:
```python
import logging
logging.getLogger('src.infrastructure.data.repositories.postgres_model_repository').setLevel(logging.DEBUG)
```

Check performance metrics:
```python
metrics = await repo.get_performance_metrics()
for metric in metrics:
    print(f"{metric.query_type}: {metric.execution_time:.2f}s")
```

Run health checks:
```python
health = await repo.health_check()
print(f"Status: {health['status']}")
for check, result in health['checks'].items():
    print(f"  {check}: {result}")
```