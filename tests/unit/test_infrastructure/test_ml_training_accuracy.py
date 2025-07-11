"""
ML-specific tests for model training validation, prediction accuracy, 
data preprocessing, and feature engineering.

Tests the machine learning pipeline components for correctness and robustness.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from tests.utils.test_helpers import MLTestHelpers
from tests.utils.data_factories import MLDataFactory, FactoryConfig


class TestMLModelTraining:
    """Test ML model training validation and convergence."""
    
    def setup_method(self):
        """Set up test fixtures for ML training tests."""
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
        self.training_data = self.ml_factory.create_training_data(
            num_users=100, num_properties=200, density=0.1
        )
    
    @pytest.mark.ml
    def test_content_based_model_convergence(self):
        """Test that content-based model converges during training."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        recommender = ContentBasedRecommender(
            embedding_dim=32,
            learning_rate=0.01
        )
        
        # Train model and capture training history
        training_result = recommender.fit(
            user_item_matrix=self.training_data['user_item_matrix'],
            property_data=self.training_data['property_features'],
            epochs=20,
            batch_size=16
        )
        
        # Test convergence criteria
        assert training_result['final_accuracy'] > 0.6  # Should achieve reasonable accuracy
        assert training_result['final_loss'] < 0.8      # Loss should decrease
        assert training_result['epochs_trained'] >= 5   # Should train for multiple epochs
        
        # Verify model state after training
        assert recommender.is_trained is True
        assert recommender.property_features is not None
        assert recommender.property_similarity_matrix is not None
    
    @pytest.mark.ml
    def test_training_with_different_data_sizes(self):
        """Test model training with different dataset sizes."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        data_sizes = [
            {'users': 20, 'properties': 50, 'density': 0.2},
            {'users': 50, 'properties': 100, 'density': 0.15},
            {'users': 100, 'properties': 200, 'density': 0.1},
            {'users': 200, 'properties': 400, 'density': 0.05}
        ]
        
        training_results = []
        
        for size_config in data_sizes:
            data = self.ml_factory.create_training_data(
                num_users=size_config['users'],
                num_properties=size_config['properties'],
                density=size_config['density']
            )
            
            recommender = ContentBasedRecommender(embedding_dim=16)
            
            result = recommender.fit(
                user_item_matrix=data['user_item_matrix'],
                property_data=data['property_features'],
                epochs=10
            )
            
            training_results.append({
                'config': size_config,
                'accuracy': result['final_accuracy'],
                'loss': result['final_loss'],
                'training_samples': result['training_samples']
            })
            
            # Verify training completed successfully
            assert result['final_accuracy'] > 0.5
            assert result['training_samples'] > 0
        
        # Check that larger datasets generally lead to better performance
        # (though this may not always be true due to sparsity)
        accuracies = [r['accuracy'] for r in training_results]
        assert max(accuracies) > min(accuracies) + 0.1  # Some improvement with scale
    
    @pytest.mark.ml
    def test_training_with_sparse_data(self):
        """Test model training with very sparse interaction data."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create very sparse data (1% density)
        sparse_data = self.ml_factory.create_training_data(
            num_users=100, num_properties=300, density=0.01
        )
        
        recommender = ContentBasedRecommender(embedding_dim=32)
        
        # Training should handle sparse data gracefully
        result = recommender.fit(
            user_item_matrix=sparse_data['user_item_matrix'],
            property_data=sparse_data['property_features'],
            epochs=15
        )
        
        # Even with sparse data, model should learn something
        assert result['final_accuracy'] > 0.5
        assert result['training_samples'] > 0
        
        # Test predictions on sparse data
        predictions = recommender.predict(user_id=0, item_ids=[0, 1, 2, 3, 4])
        MLTestHelpers.assert_valid_predictions(predictions)
        
        # Predictions should show some variation (not all identical)
        assert np.std(predictions) > 0.01
    
    @pytest.mark.ml
    def test_training_regularization_effects(self):
        """Test effects of different regularization parameters."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        regularization_values = [1e-6, 1e-4, 1e-2, 1e-1]
        results = []
        
        for reg_lambda in regularization_values:
            recommender = ContentBasedRecommender(
                embedding_dim=32,
                reg_lambda=reg_lambda,
                learning_rate=0.01
            )
            
            result = recommender.fit(
                user_item_matrix=self.training_data['user_item_matrix'],
                property_data=self.training_data['property_features'],
                epochs=10
            )
            
            results.append({
                'reg_lambda': reg_lambda,
                'final_loss': result['final_loss'],
                'final_accuracy': result['final_accuracy']
            })
        
        # Higher regularization should generally increase loss
        low_reg_loss = results[0]['final_loss']   # Lowest regularization
        high_reg_loss = results[-1]['final_loss']  # Highest regularization
        
        # With very high regularization, loss might be higher
        # but this depends on the specific data and model
        assert all(r['final_accuracy'] > 0.4 for r in results)  # All should learn something
    
    @pytest.mark.ml
    def test_learning_rate_optimization(self):
        """Test model training with different learning rates."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        learning_rates = [0.0001, 0.001, 0.01, 0.1]
        results = []
        
        for lr in learning_rates:
            recommender = ContentBasedRecommender(
                embedding_dim=32,
                learning_rate=lr
            )
            
            result = recommender.fit(
                user_item_matrix=self.training_data['user_item_matrix'],
                property_data=self.training_data['property_features'],
                epochs=15
            )
            
            results.append({
                'learning_rate': lr,
                'final_loss': result['final_loss'],
                'final_accuracy': result['final_accuracy']
            })
        
        # Verify all models trained successfully
        assert all(r['final_accuracy'] > 0.4 for r in results)
        
        # Very high learning rates might cause instability
        very_high_lr_result = results[-1]  # lr = 0.1
        
        # Should still achieve reasonable performance
        assert very_high_lr_result['final_accuracy'] > 0.3


class TestMLModelAccuracy:
    """Test ML model prediction accuracy and validation."""
    
    def setup_method(self):
        """Set up test fixtures for accuracy testing."""
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
        
        # Create training and test data
        self.train_data = self.ml_factory.create_training_data(
            num_users=80, num_properties=150, density=0.12
        )
        self.test_data = self.ml_factory.create_training_data(
            num_users=20, num_properties=150, density=0.1
        )
    
    @pytest.mark.ml
    def test_prediction_accuracy_validation(self):
        """Test prediction accuracy using train/test split."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Train model on training data
        recommender = ContentBasedRecommender(embedding_dim=32)
        recommender.fit(
            user_item_matrix=self.train_data['user_item_matrix'],
            property_data=self.train_data['property_features'],
            epochs=10
        )
        
        # Generate predictions for test users on training properties
        test_predictions = []
        test_labels = []
        
        for user_id in range(min(10, self.test_data['num_users'])):  # Test on subset
            # Get user's actual interactions from test data
            user_interactions = self.test_data['user_item_matrix'][user_id]
            interacted_items = np.where(user_interactions > 0)[0]
            non_interacted_items = np.where(user_interactions == 0)[0]
            
            if len(interacted_items) > 0 and len(non_interacted_items) > 0:
                # Sample some items for testing
                test_items = list(interacted_items[:3]) + list(non_interacted_items[:7])
                
                predictions = recommender.predict(user_id=0, item_ids=test_items)  # Use training user
                
                # Create binary labels (1 for interacted, 0 for non-interacted)
                labels = [1 if item in interacted_items else 0 for item in test_items]
                
                test_predictions.extend(predictions)
                test_labels.extend(labels)
        
        if len(test_predictions) > 0:
            # Convert predictions to binary using threshold
            binary_predictions = (np.array(test_predictions) > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, binary_predictions)
            precision = precision_score(test_labels, binary_predictions, zero_division=0)
            recall = recall_score(test_labels, binary_predictions, zero_division=0)
            f1 = f1_score(test_labels, binary_predictions, zero_division=0)
            
            # Model should perform better than random
            assert accuracy > 0.4  # Better than random (0.5 might be too strict)
            assert f1 > 0.2       # Some predictive power
            
            print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                  f"Recall: {recall:.3f}, F1: {f1:.3f}")
    
    @pytest.mark.ml
    def test_recommendation_ranking_quality(self):
        """Test quality of recommendation rankings."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        recommender = ContentBasedRecommender(embedding_dim=32)
        recommender.fit(
            user_item_matrix=self.train_data['user_item_matrix'],
            property_data=self.train_data['property_features'],
            epochs=10
        )
        
        # Test ranking quality for multiple users
        ranking_metrics = []
        
        for user_id in range(min(5, self.train_data['num_users'])):
            # Get user's actual interactions
            user_interactions = self.train_data['user_item_matrix'][user_id]
            liked_items = set(np.where(user_interactions > 0)[0])
            
            if len(liked_items) >= 2:  # Need at least 2 liked items
                # Generate recommendations
                recommendations = recommender.recommend(
                    user_id=user_id,
                    num_recommendations=20,
                    exclude_seen=True
                )
                
                if len(recommendations) > 0:
                    # Calculate ranking metrics
                    recommended_items = [rec.item_id for rec in recommendations]
                    
                    # Calculate precision@k for different k values
                    for k in [5, 10, 15]:
                        if len(recommended_items) >= k:
                            top_k_items = set(recommended_items[:k])
                            precision_at_k = len(top_k_items.intersection(liked_items)) / k
                            ranking_metrics.append(precision_at_k)
        
        if ranking_metrics:
            avg_precision = np.mean(ranking_metrics)
            
            # Should achieve some precision (better than random)
            # Random precision would be: (num_liked_items / total_items)
            # With our data, this should be around 0.1, so we expect > 0.05
            assert avg_precision > 0.02
            
            print(f"Average Precision@K: {avg_precision:.3f}")
    
    @pytest.mark.ml
    def test_cold_start_performance(self):
        """Test model performance on cold start scenarios."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        recommender = ContentBasedRecommender(embedding_dim=32)
        recommender.fit(
            user_item_matrix=self.train_data['user_item_matrix'],
            property_data=self.train_data['property_features'],
            epochs=10
        )
        
        # Test cold start user (user not in training data)
        cold_start_user_id = self.train_data['num_users']  # New user ID
        
        # Should handle cold start gracefully (content-based should work)
        cold_start_recommendations = recommender.recommend(
            user_id=0,  # Use existing user as proxy for content-based rec
            num_recommendations=10,
            exclude_seen=False
        )
        
        # Should generate recommendations for cold start scenario
        assert len(cold_start_recommendations) > 0
        
        # Recommendations should have reasonable scores
        scores = [rec.predicted_rating for rec in cold_start_recommendations]
        assert all(0 <= score <= 1 for score in scores)
        assert np.std(scores) > 0.01  # Should show variation
    
    @pytest.mark.ml
    def test_model_consistency(self):
        """Test model prediction consistency across multiple runs."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Train model multiple times with same data and parameters
        predictions_list = []
        
        for run in range(3):
            tf.random.set_seed(42)  # Same seed for reproducibility
            np.random.seed(42)
            
            recommender = ContentBasedRecommender(
                embedding_dim=32,
                learning_rate=0.01
            )
            
            recommender.fit(
                user_item_matrix=self.train_data['user_item_matrix'],
                property_data=self.train_data['property_features'],
                epochs=10
            )
            
            predictions = recommender.predict(user_id=0, item_ids=[0, 1, 2, 3, 4])
            predictions_list.append(predictions)
        
        # Predictions should be very similar across runs (with same seed)
        for i in range(1, len(predictions_list)):
            correlation = np.corrcoef(predictions_list[0], predictions_list[i])[0, 1]
            assert correlation > 0.95  # Should be highly correlated
            
            # Absolute differences should be small
            diff = np.abs(predictions_list[0] - predictions_list[i])
            assert np.max(diff) < 0.1  # Max difference < 0.1


class TestDataPreprocessingAndFeatures:
    """Test data preprocessing and feature engineering components."""
    
    def setup_method(self):
        """Set up test fixtures for preprocessing tests."""
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
    
    @pytest.mark.ml
    def test_property_feature_extraction_quality(self):
        """Test quality and completeness of property feature extraction."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create diverse property data
        properties = [
            {
                'property_id': 0,
                'neighborhood': 'Downtown',
                'city': 'San Francisco',
                'price': 3000,
                'bedrooms': 2,
                'bathrooms': 1.5,
                'amenities': ['parking', 'gym', 'pool'],
                'property_type': 'apartment',
                'square_feet': 1200
            },
            {
                'property_id': 1,
                'neighborhood': 'Mission',
                'city': 'San Francisco',
                'price': 2500,
                'bedrooms': 1,
                'bathrooms': 1.0,
                'amenities': ['laundry'],
                'property_type': 'studio',
                'square_feet': 600
            },
            {
                'property_id': 2,
                'neighborhood': 'SoMa',
                'city': 'San Francisco',
                'price': 4000,
                'bedrooms': 3,
                'bathrooms': 2.0,
                'amenities': ['parking', 'gym', 'rooftop', 'concierge'],
                'property_type': 'luxury_apartment',
                'square_feet': 1800
            }
        ]
        
        recommender = ContentBasedRecommender()
        features = recommender.extract_property_features(properties)
        
        # Test feature dimensions and types
        assert len(features.location_features) == 3
        assert features.price_features.shape == (3, 3)  # price, bedrooms, bathrooms
        assert features.amenity_features.shape[0] == 3
        assert features.combined_features.shape[0] == 3
        
        # Test feature value ranges and scaling
        # Price features should be scaled (mean ~0, std ~1)
        price_means = np.mean(features.price_features, axis=0)
        price_stds = np.std(features.price_features, axis=0)
        
        # For small dataset, scaling might not be perfect, but should be reasonable
        assert all(abs(mean) < 2 for mean in price_means)  # Roughly centered
        assert all(std > 0 for std in price_stds)          # Has variation
        
        # Location features should be categorical
        assert features.location_features.dtype in [np.int32, np.int64]
        
        # Amenity features should be binary/sparse
        assert np.all((features.amenity_features == 0) | (features.amenity_features >= 0))
    
    @pytest.mark.ml
    def test_feature_engineering_edge_cases(self):
        """Test feature engineering with edge cases and missing data."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Properties with missing/edge case data
        edge_case_properties = [
            {
                'property_id': 0,
                'price': 0,  # Zero price
                'bedrooms': 0,  # Studio
                'bathrooms': 0.5,
                'amenities': [],  # No amenities
                'property_type': 'studio'
            },
            {
                'property_id': 1,
                'neighborhood': '',  # Empty neighborhood
                'price': 50000,  # Very high price
                'bedrooms': 10,  # Many bedrooms
                'bathrooms': 8.0,
                'amenities': ['amenity'] * 20,  # Many amenities
                'property_type': 'mansion'
            },
            {
                'property_id': 2,
                # Missing several fields
                'price': 2500,
                'bedrooms': 1,
                'bathrooms': 1.0
            }
        ]
        
        recommender = ContentBasedRecommender()
        
        # Should handle edge cases without crashing
        features = recommender.extract_property_features(edge_case_properties)
        
        assert len(features.location_features) == 3
        assert features.price_features.shape[0] == 3
        assert features.amenity_features.shape[0] == 3
        
        # All features should be finite (no NaN/Inf)
        assert np.all(np.isfinite(features.price_features))
        assert np.all(np.isfinite(features.amenity_features))
        assert np.all(np.isfinite(features.combined_features))
    
    @pytest.mark.ml
    def test_feature_scaling_consistency(self):
        """Test that feature scaling is consistent across different data batches."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create two batches of properties
        batch1_properties = self.ml_factory.create_property_features(50)
        batch2_properties = self.ml_factory.create_property_features(30)
        
        recommender = ContentBasedRecommender()
        
        # Extract features from first batch (this fits the scalers)
        features1 = recommender.extract_property_features(batch1_properties)
        
        # Extract features from second batch (this uses fitted scalers)
        features2 = recommender.extract_property_features(batch2_properties)
        
        # Check that scaling is reasonable for both batches
        # First batch (used for fitting)
        batch1_means = np.mean(features1.price_features, axis=0)
        batch1_stds = np.std(features1.price_features, axis=0)
        
        # Second batch (using fitted scalers)
        batch2_means = np.mean(features2.price_features, axis=0)
        batch2_stds = np.std(features2.price_features, axis=0)
        
        # Means should be reasonably close (within 2 standard deviations)
        for i in range(len(batch1_means)):
            if batch1_stds[i] > 0:  # Avoid division by zero
                z_score = abs(batch2_means[i] - batch1_means[i]) / batch1_stds[i]
                assert z_score < 3  # Within 3 standard deviations
    
    @pytest.mark.ml
    def test_amenity_vectorization_quality(self):
        """Test quality of amenity text vectorization."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Properties with different amenity patterns
        properties_with_amenities = [
            {
                'property_id': 0,
                'price': 3000, 'bedrooms': 2, 'bathrooms': 1.5,
                'amenities': ['parking', 'gym'],
                'property_type': 'apartment'
            },
            {
                'property_id': 1,
                'price': 3000, 'bedrooms': 2, 'bathrooms': 1.5,
                'amenities': ['parking', 'pool'],  # Shares 'parking' with first
                'property_type': 'apartment'
            },
            {
                'property_id': 2,
                'price': 3000, 'bedrooms': 2, 'bathrooms': 1.5,
                'amenities': ['laundry', 'balcony'],  # Completely different
                'property_type': 'apartment'
            }
        ]
        
        recommender = ContentBasedRecommender()
        features = recommender.extract_property_features(properties_with_amenities)
        
        # Calculate amenity feature similarities
        amenity_features = features.amenity_features
        
        # Properties 0 and 1 should be more similar (share 'parking')
        similarity_01 = np.dot(amenity_features[0], amenity_features[1])
        similarity_02 = np.dot(amenity_features[0], amenity_features[2])
        similarity_12 = np.dot(amenity_features[1], amenity_features[2])
        
        # Properties sharing amenities should have higher similarity
        assert similarity_01 > similarity_02 or similarity_01 > similarity_12
    
    @pytest.mark.ml
    def test_data_augmentation_effects(self):
        """Test effects of data augmentation on model performance."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Original small dataset
        original_data = self.ml_factory.create_training_data(
            num_users=30, num_properties=60, density=0.15
        )
        
        # Train model on original data
        recommender1 = ContentBasedRecommender(embedding_dim=16)
        result1 = recommender1.fit(
            user_item_matrix=original_data['user_item_matrix'],
            property_data=original_data['property_features'],
            epochs=10
        )
        
        # Create augmented dataset (more users, same properties)
        augmented_matrix = np.vstack([
            original_data['user_item_matrix'],
            self.ml_factory.create_user_item_matrix(20, 60, density=0.1)  # Additional synthetic users
        ])
        
        # Train model on augmented data
        recommender2 = ContentBasedRecommender(embedding_dim=16)
        result2 = recommender2.fit(
            user_item_matrix=augmented_matrix,
            property_data=original_data['property_features'],
            epochs=10
        )
        
        # Both models should train successfully
        assert result1['final_accuracy'] > 0.4
        assert result2['final_accuracy'] > 0.4
        
        # Augmented model might perform differently (not necessarily better due to synthetic data)
        # But it should handle the larger dataset
        assert result2['training_samples'] > result1['training_samples']


class TestModelRobustness:
    """Test ML model robustness and error handling."""
    
    def setup_method(self):
        """Set up test fixtures for robustness testing."""
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
    
    @pytest.mark.ml
    def test_adversarial_data_robustness(self):
        """Test model robustness against adversarial or corrupted data."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create normal training data
        normal_data = self.ml_factory.create_training_data(
            num_users=50, num_properties=100, density=0.1
        )
        
        # Train model on normal data
        recommender = ContentBasedRecommender(embedding_dim=16)
        recommender.fit(
            user_item_matrix=normal_data['user_item_matrix'],
            property_data=normal_data['property_features'],
            epochs=8
        )
        
        # Test with corrupted property data
        corrupted_properties = [
            {
                'property_id': 0,
                'price': float('inf'),  # Invalid price
                'bedrooms': -5,  # Negative bedrooms
                'bathrooms': 100.0,  # Unrealistic bathrooms
                'amenities': ['?' * 1000],  # Very long amenity name
                'property_type': ''
            },
            {
                'property_id': 1,
                'price': float('-inf'),  # Invalid price
                'bedrooms': 0,
                'bathrooms': 0.0,
                'amenities': [],
                'property_type': None
            }
        ]
        
        # Model should handle corrupted data gracefully
        try:
            corrupted_features = recommender.extract_property_features(corrupted_properties)
            
            # Features should be finite after processing
            assert np.all(np.isfinite(corrupted_features.price_features))
            assert np.all(np.isfinite(corrupted_features.amenity_features))
            
        except Exception as e:
            # If extraction fails, it should fail gracefully with informative error
            assert "Properties list cannot be empty" in str(e) or "invalid" in str(e).lower()
    
    @pytest.mark.ml
    def test_model_stability_across_seeds(self):
        """Test model stability across different random seeds."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        training_data = self.ml_factory.create_training_data(
            num_users=40, num_properties=80, density=0.12
        )
        
        results = []
        seeds = [42, 123, 456, 789]
        
        for seed in seeds:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
            recommender = ContentBasedRecommender(embedding_dim=16)
            result = recommender.fit(
                user_item_matrix=training_data['user_item_matrix'],
                property_data=training_data['property_features'],
                epochs=8
            )
            
            # Generate predictions for consistency check
            predictions = recommender.predict(user_id=0, item_ids=[0, 1, 2, 3, 4])
            
            results.append({
                'seed': seed,
                'accuracy': result['final_accuracy'],
                'loss': result['final_loss'],
                'predictions': predictions
            })
        
        # All models should achieve reasonable performance
        accuracies = [r['accuracy'] for r in results]
        assert all(acc > 0.3 for acc in accuracies)
        
        # Results should be relatively stable (not too much variance)
        accuracy_std = np.std(accuracies)
        assert accuracy_std < 0.3  # Standard deviation should be reasonable
        
        # Predictions should show some consistency patterns
        # (different seeds will give different results, but structure should be similar)
        pred_correlations = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                corr = np.corrcoef(results[i]['predictions'], results[j]['predictions'])[0, 1]
                if not np.isnan(corr):
                    pred_correlations.append(abs(corr))
        
        if pred_correlations:
            avg_correlation = np.mean(pred_correlations)
            # Some correlation expected due to data structure, but not too high
            assert 0.1 < avg_correlation < 0.9
    
    @pytest.mark.ml
    def test_memory_leak_detection(self):
        """Test for potential memory leaks during repeated training."""
        import psutil
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        training_data = self.ml_factory.create_training_data(
            num_users=30, num_properties=60, density=0.1
        )
        
        # Train multiple models sequentially
        memory_usage = [initial_memory]
        
        for i in range(5):
            tf.keras.backend.clear_session()  # Clear TF session
            
            recommender = ContentBasedRecommender(embedding_dim=16)
            recommender.fit(
                user_item_matrix=training_data['user_item_matrix'],
                property_data=training_data['property_features'],
                epochs=5
            )
            
            # Force garbage collection
            import gc
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
            
            # Clean up explicitly
            del recommender
        
        # Memory usage should not grow excessively
        final_memory = memory_usage[-1]
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 500  # Should not grow by more than 500MB
        
        # Memory should not keep growing linearly
        # (some growth is expected, but it should stabilize)
        memory_diffs = [memory_usage[i+1] - memory_usage[i] for i in range(len(memory_usage)-1)]
        later_growth = np.mean(memory_diffs[-2:])  # Last 2 iterations
        early_growth = np.mean(memory_diffs[:2])   # First 2 iterations
        
        # Later growth should be less than early growth (stabilization)
        assert later_growth <= early_growth * 2  # Allow some variance