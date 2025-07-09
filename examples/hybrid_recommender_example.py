#!/usr/bin/env python3
"""
Hybrid Recommendation System Example
====================================

This example demonstrates how to use the HybridRecommendationSystem
for rental property recommendations in a production environment.

Requirements:
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Pandas (for data handling)

Usage:
    python examples/hybrid_recommender_example.py
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.ml.models.hybrid_recommender import HybridRecommendationSystem, HybridRecommendationResult


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_sample_data(num_users: int = 1000, num_properties: int = 500, 
                        interaction_rate: float = 0.05) -> tuple:
    """
    Generate sample data for demonstration purposes.
    
    Args:
        num_users: Number of users in the system
        num_properties: Number of properties in the system  
        interaction_rate: Probability of user-property interaction
        
    Returns:
        Tuple of (user_item_matrix, property_data)
    """
    print(f"Generating sample data for {num_users} users and {num_properties} properties...")
    
    # Generate user-item interaction matrix
    user_item_matrix = np.random.binomial(1, interaction_rate, (num_users, num_properties))
    
    # Generate property data
    cities = ['San Francisco', 'New York', 'Los Angeles', 'Chicago', 'Boston', 'Seattle']
    neighborhoods = ['Downtown', 'Mission', 'SoMa', 'Castro', 'Marina', 'Nob Hill']
    property_types = ['apartment', 'house', 'condo', 'townhouse', 'studio']
    amenities_list = [
        ['parking', 'gym', 'pool'],
        ['parking', 'laundry', 'gym'],
        ['pool', 'gym', 'concierge'],
        ['parking', 'balcony', 'gym'],
        ['laundry', 'gym', 'rooftop'],
        ['parking', 'pool', 'pet_friendly'],
        ['gym', 'balcony', 'dishwasher'],
        ['parking', 'laundry', 'pool']
    ]
    
    property_data = []
    for i in range(num_properties):
        property_data.append({
            'property_id': i,
            'neighborhood': np.random.choice(neighborhoods),
            'city': np.random.choice(cities),
            'state': 'CA',
            'price': np.random.randint(1500, 5000),
            'bedrooms': np.random.randint(1, 4),
            'bathrooms': np.random.randint(1, 3),
            'square_feet': np.random.randint(500, 2000),
            'amenities': np.random.choice(amenities_list).tolist(),
            'property_type': np.random.choice(property_types),
            'pet_friendly': np.random.choice([True, False]),
            'parking': np.random.choice([True, False])
        })
    
    print(f"Generated {len(property_data)} property records")
    return user_item_matrix, property_data


def train_hybrid_system(user_item_matrix: np.ndarray, 
                       property_data: List[Dict]) -> HybridRecommendationSystem:
    """
    Train the hybrid recommendation system.
    
    Args:
        user_item_matrix: User-item interaction matrix
        property_data: List of property dictionaries
        
    Returns:
        Trained HybridRecommendationSystem
    """
    print("Initializing hybrid recommendation system...")
    
    # Initialize the hybrid system with balanced weights
    hybrid_system = HybridRecommendationSystem(
        cf_weight=0.6,  # 60% collaborative filtering
        cb_weight=0.4,  # 40% content-based
        min_cf_interactions=3,  # Minimum 3 interactions for CF
        fallback_to_content=True,  # Use CB for new users
        explanation_detail_level="detailed"
    )
    
    # Initialize the underlying models
    num_users, num_items = user_item_matrix.shape
    hybrid_system.initialize_models(
        num_users=num_users,
        num_items=num_items,
        cf_embedding_dim=32,  # Smaller for demo
        cb_embedding_dim=64,  # Smaller for demo
        cf_reg_lambda=1e-5,
        cb_reg_lambda=1e-4,
        location_vocab_size=100,
        amenity_vocab_size=200
    )
    
    print("Training hybrid recommendation system...")
    
    # Train the system
    training_results = hybrid_system.fit(
        user_item_matrix=user_item_matrix,
        property_data=property_data,
        cf_epochs=20,  # Reduced for demo
        cb_epochs=20,  # Reduced for demo
        cf_batch_size=128,
        cb_batch_size=64,
        validation_split=0.2
    )
    
    print("Training completed successfully!")
    print(f"CF Training Loss: {training_results['cf_results']['final_loss']:.4f}")
    print(f"CB Training Accuracy: {training_results['cb_results']['final_accuracy']:.4f}")
    
    return hybrid_system


def demonstrate_recommendations(hybrid_system: HybridRecommendationSystem, 
                              user_item_matrix: np.ndarray,
                              property_data: List[Dict]):
    """
    Demonstrate various recommendation scenarios.
    
    Args:
        hybrid_system: Trained hybrid recommendation system
        user_item_matrix: User-item interaction matrix
        property_data: List of property dictionaries
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: HYBRID RECOMMENDATIONS")
    print("="*60)
    
    # Scenario 1: Active user with sufficient interaction history
    print("\n1. ACTIVE USER RECOMMENDATIONS (CF + CB)")
    print("-" * 50)
    
    # Find an active user (user with many interactions)
    user_interactions = np.sum(user_item_matrix, axis=1)
    active_user_id = np.argmax(user_interactions)
    
    print(f"User {active_user_id} has {user_interactions[active_user_id]} interactions")
    
    recommendations = hybrid_system.recommend(
        user_id=active_user_id,
        num_recommendations=5,
        exclude_seen=True,
        include_explanations=True
    )
    
    print(f"Top 5 recommendations for active user {active_user_id}:")
    for i, rec in enumerate(recommendations, 1):
        prop = property_data[rec.item_id]
        print(f"{i}. Property {rec.item_id}: {prop['city']}, {prop['neighborhood']}")
        print(f"   Price: ${prop['price']}, Bedrooms: {prop['bedrooms']}, Type: {prop['property_type']}")
        print(f"   Score: {rec.predicted_rating:.3f} (CF: {rec.cf_score:.3f}, CB: {rec.cb_score:.3f})")
        print(f"   Confidence: {rec.confidence_score:.3f}")
        print(f"   Explanation: {rec.explanation}")
        print()
    
    # Scenario 2: New user with no interaction history
    print("\n2. NEW USER RECOMMENDATIONS (CB Only)")
    print("-" * 50)
    
    # Find a user with no interactions
    new_user_id = np.argmin(user_interactions)
    
    print(f"User {new_user_id} has {user_interactions[new_user_id]} interactions")
    
    recommendations = hybrid_system.recommend(
        user_id=new_user_id,
        num_recommendations=5,
        exclude_seen=True,
        include_explanations=True
    )
    
    print(f"Top 5 recommendations for new user {new_user_id}:")
    for i, rec in enumerate(recommendations, 1):
        prop = property_data[rec.item_id]
        print(f"{i}. Property {rec.item_id}: {prop['city']}, {prop['neighborhood']}")
        print(f"   Price: ${prop['price']}, Bedrooms: {prop['bedrooms']}, Type: {prop['property_type']}")
        print(f"   Score: {rec.predicted_rating:.3f} (Method: {rec.hybrid_method})")
        print(f"   Confidence: {rec.confidence_score:.3f}")
        print(f"   Explanation: {rec.explanation}")
        print()
    
    # Scenario 3: Detailed explanation for a specific recommendation
    print("\n3. DETAILED RECOMMENDATION EXPLANATION")
    print("-" * 50)
    
    if recommendations:
        target_item = recommendations[0].item_id
        explanation = hybrid_system.explain_recommendation(
            user_id=active_user_id,
            item_id=target_item,
            include_feature_importance=True
        )
        
        print(f"Detailed explanation for Property {target_item} recommended to User {active_user_id}:")
        print(f"Hybrid Score: {explanation.get('hybrid_score', 'N/A'):.3f}")
        print(f"Hybrid Method: {explanation.get('hybrid_method', 'N/A')}")
        print(f"CF Weight: {explanation.get('cf_weight', 'N/A'):.2f}")
        print(f"CB Weight: {explanation.get('cb_weight', 'N/A'):.2f}")
        
        print("Component explanations:")
        for exp in explanation.get('explanations', []):
            print(f"  - {exp['type']}: {exp['description']}")
            print(f"    Score: {exp['score']:.3f}, Weight: {exp['weight']:.2f}, "
                  f"Contribution: {exp['contribution']:.3f}")
    
    # Scenario 4: System performance metrics
    print("\n4. SYSTEM PERFORMANCE METRICS")
    print("-" * 50)
    
    metrics = hybrid_system.get_model_performance_metrics()
    print("Training Performance:")
    if 'collaborative_filtering' in metrics:
        cf_metrics = metrics['collaborative_filtering']
        print(f"  CF Final Loss: {cf_metrics.get('final_loss', 'N/A'):.4f}")
        print(f"  CF Final MAE: {cf_metrics.get('final_mae', 'N/A'):.4f}")
        print(f"  CF Training Samples: {cf_metrics.get('training_samples', 'N/A')}")
    
    if 'content_based' in metrics:
        cb_metrics = metrics['content_based']
        print(f"  CB Final Loss: {cb_metrics.get('final_loss', 'N/A'):.4f}")
        print(f"  CB Final Accuracy: {cb_metrics.get('final_accuracy', 'N/A'):.4f}")
        print(f"  CB Training Samples: {cb_metrics.get('training_samples', 'N/A')}")
    
    # Scenario 5: Dynamic weight adjustment
    print("\n5. DYNAMIC WEIGHT ADJUSTMENT")
    print("-" * 50)
    
    print("Testing different weight configurations:")
    
    # Test more CB-focused approach
    print("\nTesting CB-focused approach (CF: 0.3, CB: 0.7)")
    hybrid_system.update_weights(cf_weight=0.3, cb_weight=0.7)
    
    cb_focused_recs = hybrid_system.recommend(
        user_id=active_user_id,
        num_recommendations=3,
        exclude_seen=True
    )
    
    for i, rec in enumerate(cb_focused_recs, 1):
        prop = property_data[rec.item_id]
        print(f"  {i}. Property {rec.item_id}: Score {rec.predicted_rating:.3f} "
              f"(CF: {rec.cf_score:.3f}, CB: {rec.cb_score:.3f})")
    
    # Test more CF-focused approach
    print("\nTesting CF-focused approach (CF: 0.8, CB: 0.2)")
    hybrid_system.update_weights(cf_weight=0.8, cb_weight=0.2)
    
    cf_focused_recs = hybrid_system.recommend(
        user_id=active_user_id,
        num_recommendations=3,
        exclude_seen=True
    )
    
    for i, rec in enumerate(cf_focused_recs, 1):
        prop = property_data[rec.item_id]
        print(f"  {i}. Property {rec.item_id}: Score {rec.predicted_rating:.3f} "
              f"(CF: {rec.cf_score:.3f}, CB: {rec.cb_score:.3f})")
    
    # Reset to balanced weights
    hybrid_system.update_weights(cf_weight=0.6, cb_weight=0.4)


def demonstrate_model_persistence(hybrid_system: HybridRecommendationSystem):
    """
    Demonstrate model saving and loading capabilities.
    
    Args:
        hybrid_system: Trained hybrid recommendation system
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: MODEL PERSISTENCE")
    print("="*60)
    
    # Create models directory if it doesn't exist
    import os
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save models
    cf_model_path = os.path.join(models_dir, "cf_model.h5")
    cb_model_path = os.path.join(models_dir, "cb_model.h5")
    
    print(f"Saving models to {models_dir}/...")
    try:
        hybrid_system.save_models(cf_model_path, cb_model_path)
        print("Models saved successfully!")
        
        # Show model files
        if os.path.exists(cf_model_path):
            cf_size = os.path.getsize(cf_model_path)
            print(f"CF Model: {cf_model_path} ({cf_size:,} bytes)")
        
        if os.path.exists(cb_model_path):
            cb_size = os.path.getsize(cb_model_path)
            print(f"CB Model: {cb_model_path} ({cb_size:,} bytes)")
            
    except Exception as e:
        print(f"Model saving failed: {e}")
        print("This is expected in the demo environment without full TensorFlow setup")


def main():
    """
    Main demonstration function.
    """
    print("HYBRID RECOMMENDATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    try:
        # Generate sample data
        user_item_matrix, property_data = generate_sample_data(
            num_users=200,  # Smaller for demo
            num_properties=100,  # Smaller for demo
            interaction_rate=0.1
        )
        
        # Train the hybrid system
        hybrid_system = train_hybrid_system(user_item_matrix, property_data)
        
        # Demonstrate various recommendation scenarios
        demonstrate_recommendations(hybrid_system, user_item_matrix, property_data)
        
        # Demonstrate model persistence
        demonstrate_model_persistence(hybrid_system)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nThis demo requires TensorFlow and other ML libraries.")
        print("To run this demo, install the required packages:")
        print("  pip install tensorflow numpy scikit-learn pandas")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()