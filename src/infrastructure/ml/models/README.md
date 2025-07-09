# Hybrid Recommendation System

This directory contains the implementation of a production-ready hybrid recommendation system that combines collaborative filtering and content-based approaches for rental property recommendations.

## Architecture Overview

The hybrid recommendation system consists of three main components:

1. **CollaborativeFilteringModel** (`collaborative_filter.py`) - Neural collaborative filtering using TensorFlow
2. **ContentBasedRecommender** (`content_recommender.py`) - Content-based recommendations using property features
3. **HybridRecommendationSystem** (`hybrid_recommender.py`) - Combines both approaches with configurable weighting

## Features

### HybridRecommendationSystem

The main `HybridRecommendationSystem` class provides:

- **Configurable Weighting**: Adjust the balance between collaborative filtering (CF) and content-based (CB) recommendations
- **Cold Start Handling**: Automatically falls back to content-based recommendations for new users
- **Comprehensive Explanations**: Provides detailed explanations for recommendations at multiple detail levels
- **Diversity Filtering**: Applies diversity constraints to avoid over-similar recommendations
- **Robust Error Handling**: Gracefully handles model failures and missing data
- **Model Persistence**: Save and load trained models for production deployment

### Key Classes

#### HybridRecommendationResult
Extended recommendation result with hybrid scoring details:
```python
@dataclass
class HybridRecommendationResult:
    item_id: int
    predicted_rating: float
    confidence_score: float
    explanation: str
    cf_score: Optional[float] = None
    cb_score: Optional[float] = None
    hybrid_method: str = "weighted_average"
    feature_importance: Optional[Dict[str, float]] = None
```

## Usage Examples

### Basic Usage

```python
from src.infrastructure.ml.models.hybrid_recommender import HybridRecommendationSystem
import numpy as np

# Initialize the hybrid system
hybrid_system = HybridRecommendationSystem(
    cf_weight=0.6,  # 60% collaborative filtering
    cb_weight=0.4,  # 40% content-based
    min_cf_interactions=5,  # Minimum interactions for CF
    fallback_to_content=True,  # Use CB for new users
    explanation_detail_level="detailed"
)

# Initialize the underlying models
hybrid_system.initialize_models(
    num_users=1000,
    num_items=500,
    cf_embedding_dim=50,
    cb_embedding_dim=128
)

# Prepare training data
user_item_matrix = np.random.rand(1000, 500)  # Replace with real data
property_data = [
    {
        'neighborhood': 'Downtown',
        'city': 'San Francisco',
        'state': 'CA',
        'price': 3000,
        'bedrooms': 2,
        'bathrooms': 1,
        'amenities': ['parking', 'gym', 'pool'],
        'property_type': 'apartment'
    },
    # ... more property data
]

# Train the hybrid system
training_results = hybrid_system.fit(
    user_item_matrix=user_item_matrix,
    property_data=property_data,
    cf_epochs=100,
    cb_epochs=100
)

# Generate recommendations
recommendations = hybrid_system.recommend(
    user_id=42,
    num_recommendations=10,
    exclude_seen=True,
    include_explanations=True
)

# Print recommendations
for rec in recommendations:
    print(f"Property {rec.item_id}: {rec.predicted_rating:.3f} - {rec.explanation}")
```

### Advanced Usage

```python
# Get detailed explanation for a specific recommendation
explanation = hybrid_system.explain_recommendation(
    user_id=42,
    item_id=123,
    include_feature_importance=True
)

print(f"Hybrid Score: {explanation['hybrid_score']:.3f}")
print(f"CF Score: {explanation.get('cf_score', 'N/A')}")
print(f"CB Score: {explanation.get('cb_score', 'N/A')}")

# Update weights dynamically
hybrid_system.update_weights(cf_weight=0.7, cb_weight=0.3)

# Get system performance metrics
metrics = hybrid_system.get_model_performance_metrics()
print(f"CF Training Loss: {metrics['collaborative_filtering']['final_loss']}")
print(f"CB Training Accuracy: {metrics['content_based']['final_accuracy']}")

# Save trained models
hybrid_system.save_models(
    cf_model_path="models/cf_model.h5",
    cb_model_path="models/cb_model.h5"
)
```

## Configuration Options

### Initialization Parameters

- `cf_weight` (float): Weight for collaborative filtering recommendations (0.0 to 1.0)
- `cb_weight` (float): Weight for content-based recommendations (0.0 to 1.0)
- `min_cf_interactions` (int): Minimum user interactions required for CF recommendations
- `fallback_to_content` (bool): Whether to fall back to content-based for new users
- `explanation_detail_level` (str): Level of detail for explanations ("simple", "detailed", "technical")

### Training Parameters

- `cf_epochs` (int): Number of epochs for collaborative filtering training
- `cb_epochs` (int): Number of epochs for content-based training
- `cf_batch_size` (int): Batch size for CF training
- `cb_batch_size` (int): Batch size for CB training
- `validation_split` (float): Validation split for training

## Cold Start Problem Handling

The system handles cold start scenarios in several ways:

1. **New Users**: Automatically falls back to content-based recommendations
2. **New Items**: Content-based model can recommend new properties immediately
3. **Sparse Data**: Combines both approaches to provide robust recommendations

## Error Handling

The system includes comprehensive error handling:

- **Model Failures**: Graceful fallback between models
- **Invalid Inputs**: Validates user and item IDs
- **Missing Data**: Handles missing property features
- **Training Failures**: Provides detailed error messages and logs

## Performance Considerations

- **Caching**: Implement caching for frequently requested recommendations
- **Batch Processing**: Use batch prediction for better performance
- **Model Serving**: Consider using TensorFlow Serving for production deployment
- **Monitoring**: Track model performance and recommendation quality metrics

## Integration with Domain Services

The hybrid recommender integrates with the domain layer through:

- **RecommendationService**: Provides high-level recommendation API
- **PropertyRepository**: Accesses property data and features
- **UserRepository**: Manages user interactions and preferences
- **ModelRepository**: Handles model persistence and caching

## Testing

The system includes comprehensive testing:

- **Unit Tests**: Test individual components and methods
- **Integration Tests**: Test the complete recommendation pipeline
- **Performance Tests**: Measure latency and throughput
- **A/B Testing**: Compare different weighting strategies

## Deployment

For production deployment:

1. **Model Training**: Train models on historical data
2. **Model Validation**: Validate model performance on test data
3. **Model Deployment**: Deploy using TensorFlow Serving or similar
4. **Monitoring**: Set up monitoring for model performance and drift
5. **Scaling**: Configure auto-scaling based on traffic patterns

## Monitoring and Metrics

Key metrics to monitor:

- **Recommendation Quality**: Precision@K, Recall@K, NDCG
- **User Engagement**: Click-through rates, conversion rates
- **System Performance**: Latency, throughput, error rates
- **Model Performance**: Training loss, validation metrics
- **Business Impact**: Revenue, user satisfaction, retention

## Troubleshooting

Common issues and solutions:

1. **Poor Recommendations**: Check data quality and model training
2. **Slow Performance**: Optimize batch processing and caching
3. **Memory Issues**: Reduce embedding dimensions or batch sizes
4. **Cold Start Issues**: Increase content-based weight for new users
5. **Training Failures**: Check data format and model parameters

## Future Enhancements

Potential improvements:

- **Deep Learning**: Advanced neural architectures (autoencoders, VAEs)
- **Real-time Learning**: Online learning for real-time adaptation
- **Multi-armed Bandits**: Exploration vs exploitation optimization
- **Contextual Recommendations**: Time, location, and seasonal factors
- **Fairness**: Bias detection and mitigation techniques