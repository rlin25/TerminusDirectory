# Comprehensive Collaborative Filtering Model

## Overview

This document describes the comprehensive neural collaborative filtering model implementation for the rental ML system. The model implements state-of-the-art techniques for recommendation systems with production-ready features.

## Architecture

### Neural Collaborative Filtering (NCF)

The model combines two complementary approaches:

1. **Neural Matrix Factorization (NMF)**: Element-wise product of user and item embeddings
2. **Multi-Layer Perceptron (MLP)**: Deep learning layers for complex interaction modeling

```
User ID ─┐
         ├─→ Embedding ─┐
Item ID ─┘              ├─→ NMF (element-wise) ─┐
                        │                       ├─→ Concatenate ─→ Dense ─→ Output
         ├─→ Embedding ─┘                       │
         │                                      │
         └─→ MLP (dense layers) ────────────────┘
```

### Key Components

#### 1. Data Preprocessing (`DataPreprocessor`)
- **User/Item ID Encoding**: Converts external IDs to internal indices
- **Rating Normalization**: Standardizes rating scales
- **Negative Sampling**: Generates implicit feedback for unobserved interactions
- **Statistics Calculation**: Computes user and item statistics for cold start handling

#### 2. Model Architecture (`CollaborativeFilteringModel`)
- **Embedding Layers**: Learn dense representations for users and items
- **Neural MF Branch**: Captures linear interactions
- **MLP Branch**: Models complex non-linear interactions
- **Regularization**: L2 regularization and dropout for overfitting prevention
- **Batch Normalization**: Improves training stability

#### 3. Training Pipeline
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Saves best model during training
- **Comprehensive Logging**: Tracks training progress and metrics

#### 4. Evaluation Metrics (`ModelEvaluator`)
- **Accuracy Metrics**: RMSE, MAE
- **Ranking Metrics**: Precision@K, Recall@K, NDCG@K
- **Diversity Metrics**: Coverage, diversity, novelty
- **System Metrics**: Latency, memory usage, throughput

#### 5. Cold Start Handling (`ColdStartHandler`)
- **Cold Users**: Popularity-based recommendations
- **Cold Items**: Content-based fallback
- **Hybrid Approach**: Combines multiple strategies

## Features

### 1. Advanced Model Architecture

```python
config = TrainingConfig(
    embedding_dim=128,
    hidden_layers=[256, 128, 64],
    dropout_rate=0.2,
    regularization=1e-6
)

model = CollaborativeFilteringModel(
    num_users=10000,
    num_items=5000,
    config=config
)
```

### 2. Flexible Data Input

```python
# From DataFrame
model.fit(interactions_df=df)

# From matrix
model.fit(user_item_matrix=matrix)

# With validation data
model.fit(interactions_df=df, validation_data=(val_users, val_items, val_ratings))
```

### 3. Comprehensive Evaluation

```python
# Automatic evaluation during training
evaluation_metrics = model.evaluator.evaluate_model(
    model, test_data, user_item_matrix
)

# Performance monitoring
performance_info = model.monitor_performance(test_data)
```

### 4. Production-Ready Features

```python
# Model persistence
model.save_model('/path/to/model.h5')
model.load_model('/path/to/model.h5')

# Batch predictions
predictions = model.predict_batch(user_ids, item_ids)

# Cold start handling
recommendations = model.recommend(new_user_id, num_recommendations=10)
```

## Usage Examples

### Basic Training

```python
import pandas as pd
from src.infrastructure.ml.models.collaborative_filter import (
    CollaborativeFilteringModel, TrainingConfig
)

# Load data
interactions_df = pd.read_csv('interactions.csv')

# Configure model
config = TrainingConfig(
    epochs=100,
    batch_size=512,
    embedding_dim=128,
    early_stopping_patience=10
)

# Initialize and train model
model = CollaborativeFilteringModel(
    num_users=interactions_df['user_id'].nunique(),
    num_items=interactions_df['item_id'].nunique(),
    config=config
)

results = model.fit(interactions_df=interactions_df)
```

### Advanced Configuration

```python
# Custom architecture
config = TrainingConfig(
    epochs=50,
    batch_size=1024,
    learning_rate=0.001,
    embedding_dim=256,
    hidden_layers=[512, 256, 128, 64],
    dropout_rate=0.3,
    regularization=1e-5,
    negative_sampling_ratio=0.6
)

# Training with validation
train_df = interactions_df.sample(frac=0.8)
val_df = interactions_df.drop(train_df.index)

model = CollaborativeFilteringModel(num_users=1000, num_items=500, config=config)
results = model.fit(
    interactions_df=train_df,
    validation_data=prepare_validation_data(val_df)
)
```

### Making Recommendations

```python
# Get recommendations for a user
recommendations = model.recommend(
    user_id=42,
    num_recommendations=10,
    exclude_seen=True
)

# Handle cold start users
cold_recommendations = model.recommend(
    user_id=new_user_id,
    user_features={'age': 25, 'location': 'NYC'},
    num_recommendations=10
)

# Batch predictions
user_ids = np.array([1, 2, 3, 4, 5])
item_ids = np.array([10, 20, 30, 40, 50])
predictions = model.predict_batch(user_ids, item_ids)
```

### Model Analysis

```python
# Get similar users and items
similar_users = model.get_similar_users(user_id=42, num_similar=5)
similar_items = model.get_similar_items(item_id=123, num_similar=5)

# Detailed explanation
explanation = model.explain_recommendation(user_id=42, item_id=123)

# Feature importance
importance = model.get_feature_importance()

# Performance monitoring
performance = model.monitor_performance(test_data)
```

## Configuration Options

### TrainingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 100 | Number of training epochs |
| `batch_size` | int | 256 | Training batch size |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `embedding_dim` | int | 128 | Embedding dimension |
| `hidden_layers` | List[int] | [256, 128, 64] | MLP hidden layer sizes |
| `dropout_rate` | float | 0.2 | Dropout rate |
| `regularization` | float | 1e-6 | L2 regularization strength |
| `early_stopping_patience` | int | 10 | Early stopping patience |
| `reduce_lr_patience` | int | 5 | LR reduction patience |
| `negative_sampling_ratio` | float | 0.5 | Negative sampling ratio |

## Evaluation Metrics

### Accuracy Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

### Ranking Metrics
- **Precision@K**: Proportion of relevant items in top-K recommendations
- **Recall@K**: Proportion of relevant items retrieved in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain

### Diversity Metrics
- **Coverage**: Proportion of items recommended across all users
- **Diversity**: Average pairwise dissimilarity in recommendations
- **Novelty**: Average popularity of recommended items (inverted)

## Cold Start Strategies

### Cold Users
1. **Popularity-based**: Recommend popular items
2. **Content-based**: Use user features if available
3. **Hybrid**: Combine multiple approaches

### Cold Items
1. **Content-based**: Use item features for prediction
2. **Average rating**: Fall back to global average
3. **Similar items**: Use content similarity

## Integration with Model Repository

The model integrates seamlessly with the existing model serving infrastructure:

```python
# Model server integration
from src.infrastructure.ml.serving.model_server import ModelServer

server = ModelServer(database_url="postgresql://...")
await server.deploy_model(
    model_type="collaborative",
    model_path="/path/to/model.h5",
    version="v1.0.0",
    performance_metrics=evaluation_metrics
)
```

## Performance Optimization

### Memory Optimization
- **Batch processing**: Process recommendations in batches
- **Model caching**: Cache frequently used models
- **Embedding compression**: Reduce embedding dimensions for deployment

### Computation Optimization
- **GPU acceleration**: Use GPU for training and inference
- **Quantization**: Reduce model precision for faster inference
- **Pruning**: Remove unnecessary connections

## Monitoring and Logging

### Training Monitoring
- Loss curves and validation metrics
- Learning rate schedules
- Model checkpointing

### Production Monitoring
- Prediction latency
- Memory usage
- Recommendation quality metrics
- A/B testing results

## Best Practices

### Data Preparation
1. **Clean data**: Remove duplicates and outliers
2. **Feature engineering**: Create meaningful user/item features
3. **Balanced sampling**: Ensure representative training data
4. **Temporal splitting**: Use time-based train/test splits

### Model Training
1. **Hyperparameter tuning**: Use grid search or Bayesian optimization
2. **Cross-validation**: Validate on multiple folds
3. **Regularization**: Prevent overfitting with dropout and L2
4. **Early stopping**: Monitor validation loss

### Production Deployment
1. **Model versioning**: Track model versions and metadata
2. **A/B testing**: Compare model performance
3. **Monitoring**: Track key metrics in production
4. **Fallback strategies**: Handle edge cases gracefully

## Troubleshooting

### Common Issues

#### Training Issues
- **Overfitting**: Increase regularization, reduce model complexity
- **Underfitting**: Increase model capacity, reduce regularization
- **Slow convergence**: Adjust learning rate, batch size

#### Prediction Issues
- **Poor cold start**: Improve fallback strategies
- **Low diversity**: Adjust recommendation algorithm
- **High latency**: Optimize model architecture, use caching

#### Memory Issues
- **OOM during training**: Reduce batch size, model size
- **Large model size**: Use embedding compression
- **Memory leaks**: Proper cleanup of TensorFlow sessions

### Performance Tuning

#### For Accuracy
- Increase embedding dimensions
- Add more hidden layers
- Tune regularization parameters
- Collect more training data

#### For Speed
- Reduce embedding dimensions
- Use fewer hidden layers
- Implement model quantization
- Use approximate methods

#### For Scalability
- Implement distributed training
- Use model sharding
- Optimize data loading
- Implement caching strategies

## Future Enhancements

### Planned Features
1. **Multi-task learning**: Joint optimization of multiple objectives
2. **Attention mechanisms**: Improve interaction modeling
3. **Graph neural networks**: Leverage user-item graph structure
4. **Reinforcement learning**: Optimize long-term user engagement
5. **Federated learning**: Privacy-preserving collaborative filtering

### Research Directions
1. **Explainable AI**: Better recommendation explanations
2. **Fairness**: Bias mitigation in recommendations
3. **Temporal dynamics**: Model time-evolving preferences
4. **Cross-domain**: Transfer learning across domains
5. **Automated ML**: Automated architecture search