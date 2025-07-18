#!/usr/bin/env python3
"""
Example usage of the comprehensive collaborative filtering model.

This example demonstrates:
1. Data preprocessing
2. Model training with advanced features
3. Evaluation metrics
4. Cold start handling
5. Model monitoring
6. Model saving and loading
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.ml.models.collaborative_filter import (
    CollaborativeFilteringModel,
    TrainingConfig,
    RecommendationResult,
    EvaluationMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample interaction data for demonstration"""
    np.random.seed(42)
    
    # Create sample user-item interactions
    num_users = 1000
    num_items = 500
    num_interactions = 10000
    
    interactions = []
    for _ in range(num_interactions):
        user_id = np.random.randint(0, num_users)
        item_id = np.random.randint(0, num_items)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
        timestamp = datetime.now()
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': timestamp
        })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Remove duplicates, keep the latest interaction
    interactions_df = interactions_df.drop_duplicates(
        subset=['user_id', 'item_id'], 
        keep='last'
    )
    
    logger.info(f"Created {len(interactions_df)} unique interactions")
    return interactions_df

def main():
    """Main example function"""
    logger.info("Starting collaborative filtering example")
    
    # Create sample data
    interactions_df = create_sample_data()
    
    # Configure model training
    config = TrainingConfig(
        epochs=50,
        batch_size=512,
        learning_rate=0.001,
        embedding_dim=64,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        regularization=1e-5,
        early_stopping_patience=5,
        negative_sampling_ratio=0.4
    )
    
    # Split data for training and testing
    train_df = interactions_df.sample(frac=0.8, random_state=42)
    test_df = interactions_df.drop(train_df.index)
    
    logger.info(f"Training set: {len(train_df)} interactions")
    logger.info(f"Test set: {len(test_df)} interactions")
    
    # Initialize model
    num_users = interactions_df['user_id'].nunique()
    num_items = interactions_df['item_id'].nunique()
    
    model = CollaborativeFilteringModel(
        num_users=num_users,
        num_items=num_items,
        config=config
    )
    
    logger.info("Model initialized successfully")
    
    # Train the model
    logger.info("Starting model training...")
    training_results = model.fit(interactions_df=train_df)
    
    logger.info("Training completed!")
    logger.info(f"Final training loss: {training_results['final_loss']:.4f}")
    logger.info(f"Final validation loss: {training_results['val_loss']:.4f}")
    
    # Evaluate the model
    logger.info("Evaluating model...")
    
    # Prepare test data for evaluation
    test_data = {
        'users': model.preprocessor.user_encoder.transform(test_df['user_id']),
        'items': model.preprocessor.item_encoder.transform(test_df['item_id']),
        'ratings': test_df['rating'].values
    }
    
    # Monitor performance
    performance_info = model.monitor_performance(test_data)
    logger.info(f"Model evaluation completed")
    logger.info(f"RMSE: {performance_info['evaluation_metrics'].rmse:.4f}")
    logger.info(f"MAE: {performance_info['evaluation_metrics'].mae:.4f}")
    logger.info(f"Precision@10: {performance_info['evaluation_metrics'].precision_at_k.get(10, 0):.4f}")
    logger.info(f"Recall@10: {performance_info['evaluation_metrics'].recall_at_k.get(10, 0):.4f}")
    
    # Get recommendations for a sample user
    sample_user_id = 0
    logger.info(f"Getting recommendations for user {sample_user_id}")
    
    recommendations = model.recommend(
        user_id=sample_user_id,
        num_recommendations=10,
        exclude_seen=True
    )
    
    logger.info(f"Top 10 recommendations for user {sample_user_id}:")
    for i, rec in enumerate(recommendations[:5], 1):
        logger.info(f"{i}. Item {rec.item_id}: {rec.predicted_rating:.3f} (confidence: {rec.confidence_score:.3f})")
        logger.info(f"   Explanation: {rec.explanation}")
    
    # Demonstrate cold start handling
    logger.info("Testing cold start handling...")
    
    # Cold start user
    cold_user_recommendations = model.recommend(
        user_id=num_users + 1,  # New user
        num_recommendations=5
    )
    
    logger.info(f"Cold start recommendations: {len(cold_user_recommendations)} items")
    
    # Get detailed explanation for a recommendation
    if recommendations:
        explanation = model.explain_recommendation(sample_user_id, recommendations[0].item_id)
        logger.info(f"Detailed explanation for top recommendation:")
        logger.info(f"  Predicted rating: {explanation['predicted_rating']:.3f}")
        logger.info(f"  Confidence: {explanation['confidence']:.3f}")
        logger.info(f"  Similar items: {len(explanation['similar_items'])}")
        logger.info(f"  Similar users: {len(explanation['similar_users'])}")
    
    # Find similar users and items
    similar_users = model.get_similar_users(sample_user_id, num_similar=5)
    similar_items = model.get_similar_items(0, num_similar=5)
    
    logger.info(f"Similar users to user {sample_user_id}:")
    for user_id, similarity in similar_users:
        logger.info(f"  User {user_id}: {similarity:.3f}")
    
    logger.info(f"Similar items to item 0:")
    for item_id, similarity in similar_items:
        logger.info(f"  Item {item_id}: {similarity:.3f}")
    
    # Get model information
    model_info = model.get_model_info()
    logger.info(f"Model information:")
    logger.info(f"  Users: {model_info['num_users']}")
    logger.info(f"  Items: {model_info['num_items']}")
    logger.info(f"  Embedding dimension: {model_info['embedding_dim']}")
    logger.info(f"  Total parameters: {model_info['model_complexity']['total_params']:,}")
    
    # Save the model
    model_path = "/tmp/collaborative_filtering_model.h5"
    logger.info(f"Saving model to {model_path}")
    model.save_model(model_path)
    
    # Load the model
    logger.info("Loading model from saved file")
    new_model = CollaborativeFilteringModel(
        num_users=num_users,
        num_items=num_items,
        config=config
    )
    new_model.load_model(model_path)
    
    # Test loaded model
    test_recommendations = new_model.recommend(sample_user_id, num_recommendations=5)
    logger.info(f"Loaded model recommendations: {len(test_recommendations)} items")
    
    # Feature importance analysis
    feature_importance = model.get_feature_importance()
    logger.info(f"Feature importance analysis:")
    logger.info(f"  User embedding mean norm: {feature_importance['user_embedding_stats']['mean_norm']:.3f}")
    logger.info(f"  Item embedding mean norm: {feature_importance['item_embedding_stats']['mean_norm']:.3f}")
    logger.info(f"  Top user dimensions: {feature_importance['most_important_user_dims'][:5]}")
    logger.info(f"  Top item dimensions: {feature_importance['most_important_item_dims'][:5]}")
    
    logger.info("Collaborative filtering example completed successfully!")

if __name__ == "__main__":
    main()