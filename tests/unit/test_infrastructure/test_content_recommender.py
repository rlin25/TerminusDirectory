"""
Unit tests for the ContentBasedRecommender ML model.

Tests model initialization, training, prediction, and recommendation methods.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

from infrastructure.ml.models.content_recommender import ContentBasedRecommender, PropertyFeatures
from infrastructure.ml.models.collaborative_filter import RecommendationResult
from tests.utils.test_helpers import MLTestHelpers, PerformanceTestHelpers
from tests.utils.data_factories import MLDataFactory, FactoryConfig


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        # Ensure deterministic behavior
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.recommender = ContentBasedRecommender(
            embedding_dim=64,  # Smaller for testing
            location_vocab_size=100,
            amenity_vocab_size=50,
            reg_lambda=1e-4,
            learning_rate=0.01
        )
        
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
        
        # Sample test data
        self.sample_properties = [
            {
                'property_id': 0,
                'neighborhood': 'Downtown',
                'city': 'San Francisco',
                'price': 3000,
                'bedrooms': 2,
                'bathrooms': 1.5,
                'amenities': ['parking', 'gym'],
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
                'amenities': ['gym', 'rooftop'],
                'property_type': 'studio',
                'square_feet': 800
            },
            {
                'property_id': 2,
                'neighborhood': 'SoMa',
                'city': 'San Francisco',
                'price': 3500,
                'bedrooms': 3,
                'bathrooms': 2.0,
                'amenities': ['parking', 'pool'],
                'property_type': 'condo',
                'square_feet': 1500
            }
        ]
        
        self.sample_user_item_matrix = np.array([
            [1, 0, 1],  # User 0 liked properties 0, 2
            [0, 1, 0],  # User 1 liked property 1
            [1, 1, 1],  # User 2 liked all properties
        ])
    
    def test_initialization(self):
        """Test ContentBasedRecommender initialization."""
        recommender = ContentBasedRecommender(
            embedding_dim=128,
            location_vocab_size=500,
            amenity_vocab_size=200,
            reg_lambda=1e-5,
            learning_rate=0.001
        )
        
        assert recommender.embedding_dim == 128
        assert recommender.location_vocab_size == 500
        assert recommender.amenity_vocab_size == 200
        assert recommender.reg_lambda == 1e-5
        assert recommender.learning_rate == 0.001
        assert recommender.is_trained is False
        assert recommender.model is not None
        assert recommender.location_encoder is not None
        assert recommender.amenity_vectorizer is not None
        assert recommender.price_scaler is not None
    
    def test_model_architecture(self):
        """Test that the neural network model is built correctly."""
        model = self.recommender.model
        
        # Check model structure
        assert len(model.inputs) == 3  # location, price, amenity inputs
        assert len(model.outputs) == 1  # preference output
        
        # Check input shapes
        location_input, price_input, amenity_input = model.inputs
        assert location_input.shape[1:] == ()  # Scalar input for location
        assert price_input.shape[1:] == (3,)  # Price, bedrooms, bathrooms
        assert amenity_input.shape[1:] == (50,)  # Amenity features
        
        # Check output shape
        output = model.outputs[0]
        assert output.shape[1:] == (1,)  # Single preference score
        
        # Check model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
        assert model.metrics is not None
    
    def test_extract_property_features(self):
        """Test property feature extraction."""
        features = self.recommender.extract_property_features(self.sample_properties)
        
        assert isinstance(features, PropertyFeatures)
        assert len(features.location_features) == 3
        assert features.price_features.shape == (3, 3)  # 3 properties, 3 price features
        assert features.amenity_features.shape[0] == 3  # 3 properties
        assert features.combined_features.shape[0] == 3  # 3 properties
        assert len(features.feature_names) > 0
        
        # Check that location features are encoded
        assert np.issubdtype(features.location_features.dtype, np.integer)
        
        # Check that price features are scaled
        assert np.all(np.isfinite(features.price_features))
        
        # Check that amenity features are sparse
        assert features.amenity_features.shape[1] <= 50  # Max amenity vocab size
    
    def test_extract_property_features_edge_cases(self):
        """Test property feature extraction with edge cases."""
        # Empty properties list
        with pytest.raises(ValueError, match="Properties list cannot be empty"):
            self.recommender.extract_property_features([])
        
        # Properties with missing fields
        incomplete_properties = [
            {
                'property_id': 0,
                'price': 2000,
                'bedrooms': 1,
                'bathrooms': 1.0
                # Missing other fields
            }
        ]
        
        features = self.recommender.extract_property_features(incomplete_properties)
        assert isinstance(features, PropertyFeatures)
        assert len(features.location_features) == 1
        
        # Properties with empty amenities
        empty_amenities_properties = [
            {
                'property_id': 0,
                'neighborhood': 'Downtown',
                'city': 'San Francisco',
                'price': 2000,
                'bedrooms': 1,
                'bathrooms': 1.0,
                'amenities': [],
                'property_type': 'studio'
            }
        ]
        
        features = self.recommender.extract_property_features(empty_amenities_properties)
        assert isinstance(features, PropertyFeatures)
        assert features.amenity_features.shape[0] == 1
    
    def test_feature_processing_consistency(self):
        """Test that feature processing is consistent between calls."""
        # First extraction
        features1 = self.recommender.extract_property_features(self.sample_properties)
        
        # Second extraction with same data
        features2 = self.recommender.extract_property_features(self.sample_properties)
        
        # Features should be identical
        np.testing.assert_array_equal(features1.location_features, features2.location_features)
        np.testing.assert_array_equal(features1.price_features, features2.price_features)
        np.testing.assert_array_equal(features1.amenity_features, features2.amenity_features)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        # First extract features
        self.recommender.property_features = self.recommender.extract_property_features(
            self.sample_properties
        )
        
        # Prepare training data
        X_train, y_train = self.recommender._prepare_training_data(self.sample_user_item_matrix)
        
        assert len(X_train) == 3  # location, price, amenity inputs
        assert len(X_train[0]) == len(y_train)  # Same number of samples
        assert len(X_train[1]) == len(y_train)
        assert len(X_train[2]) == len(y_train)
        
        # Check input shapes
        assert X_train[0].shape[1:] == ()  # Location features
        assert X_train[1].shape[1:] == (3,)  # Price features
        assert X_train[2].shape[1:] == (50,)  # Amenity features
        
        # Check labels
        assert np.all((y_train == 0) | (y_train == 1))  # Binary labels
        assert np.sum(y_train == 1) > 0  # Some positive samples
        assert np.sum(y_train == 0) > 0  # Some negative samples
    
    @pytest.mark.slow
    def test_model_training(self):
        """Test model training process."""
        training_result = self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=5,  # Small number for testing
            batch_size=8,
            validation_split=0.2
        )
        
        assert self.recommender.is_trained is True
        assert isinstance(training_result, dict)
        assert 'final_loss' in training_result
        assert 'final_accuracy' in training_result
        assert 'epochs_trained' in training_result
        assert 'training_samples' in training_result
        
        # Check that loss is reasonable
        assert training_result['final_loss'] >= 0
        assert training_result['final_accuracy'] >= 0
        assert training_result['final_accuracy'] <= 1
    
    def test_training_without_data(self):
        """Test that training fails without proper data."""
        with pytest.raises(Exception):  # Should raise some exception
            self.recommender.fit(
                user_item_matrix=np.array([]),
                property_data=[]
            )
    
    def test_prediction_before_training(self):
        """Test that prediction fails before training."""
        with pytest.raises(ValueError, match="Model must be trained"):
            self.recommender.predict(user_id=0, item_ids=[0, 1])
    
    def test_prediction_after_training(self):
        """Test prediction after training."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2,
            batch_size=4
        )
        
        # Make predictions
        predictions = self.recommender.predict(user_id=0, item_ids=[0, 1, 2])
        
        MLTestHelpers.assert_valid_predictions(predictions, expected_shape=(3,))
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
    
    def test_prediction_with_invalid_item_ids(self):
        """Test prediction with invalid item IDs."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        # Test with out-of-range item IDs
        predictions = self.recommender.predict(user_id=0, item_ids=[0, 1, 10])  # 10 is out of range
        
        # Should return predictions for valid IDs only
        assert len(predictions) == 2  # Only 0 and 1 are valid
    
    def test_prediction_with_empty_item_list(self):
        """Test prediction with empty item list."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        predictions = self.recommender.predict(user_id=0, item_ids=[])
        assert len(predictions) == 0
    
    def test_recommendations_generation(self):
        """Test recommendation generation."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        recommendations = self.recommender.recommend(user_id=0, num_recommendations=2)
        
        MLTestHelpers.assert_recommendations_quality(recommendations, min_count=1)
        assert len(recommendations) <= 2
        
        # Check recommendation structure
        for rec in recommendations:
            assert isinstance(rec, RecommendationResult)
            assert hasattr(rec, 'item_id')
            assert hasattr(rec, 'predicted_rating')
            assert hasattr(rec, 'confidence_score')
            assert hasattr(rec, 'explanation')
    
    def test_recommendations_exclude_seen(self):
        """Test that recommendations exclude seen items when requested."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        # User 0 has interacted with items 0 and 2
        recommendations = self.recommender.recommend(user_id=0, exclude_seen=True)
        
        recommended_item_ids = [rec.item_id for rec in recommendations]
        assert 0 not in recommended_item_ids  # Should be excluded
        assert 2 not in recommended_item_ids  # Should be excluded
    
    def test_recommendations_include_seen(self):
        """Test that recommendations include seen items when requested."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        recommendations = self.recommender.recommend(user_id=0, exclude_seen=False)
        
        # Should include all items
        assert len(recommendations) <= 3
    
    def test_recommendations_for_invalid_user(self):
        """Test recommendations for invalid user ID."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        with pytest.raises(ValueError, match="User ID .* is out of range"):
            self.recommender.recommend(user_id=10)  # Out of range
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        # Test confidence calculation
        confidence = self.recommender._calculate_confidence(user_id=0, item_id=1, prediction=0.8)
        
        assert 0 <= confidence <= 1
        assert confidence >= 0.8  # Should be at least the prediction value (or adjusted)
    
    def test_explanation_generation(self):
        """Test explanation generation."""
        explanations = []
        predictions = [0.9, 0.7, 0.5, 0.3]
        
        for pred in predictions:
            explanation = self.recommender._generate_explanation(user_id=0, item_id=0, prediction=pred)
            explanations.append(explanation)
            assert isinstance(explanation, str)
            assert len(explanation) > 0
        
        # Check that different prediction scores generate different explanations
        assert len(set(explanations)) > 1
    
    def test_property_similarity(self):
        """Test property similarity calculation."""
        # Train the model to compute similarity matrix
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        similar_properties = self.recommender.get_property_similarity(item_id=0, num_similar=2)
        
        assert len(similar_properties) <= 2
        for item_id, similarity in similar_properties:
            assert isinstance(item_id, int)
            assert isinstance(similarity, float)
            assert 0 <= similarity <= 1
            assert item_id != 0  # Should not include the query item itself
    
    def test_property_similarity_invalid_item(self):
        """Test property similarity with invalid item ID."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        with pytest.raises(ValueError, match="Item ID .* is out of range"):
            self.recommender.get_property_similarity(item_id=10)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        importance = self.recommender.get_feature_importance(item_id=0)
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # Check that importance values sum to approximately 1
        total_importance = sum(importance.values())
        assert 0.95 <= total_importance <= 1.05
        
        # Check that all importance values are non-negative
        assert all(value >= 0 for value in importance.values())
    
    def test_feature_importance_invalid_item(self):
        """Test feature importance with invalid item ID."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        with pytest.raises(Exception):  # Should raise an exception for out-of-range item
            self.recommender.get_feature_importance(item_id=10)
    
    def test_model_save_and_load(self, temp_model_file):
        """Test model saving and loading."""
        # Train the model
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        # Save model
        self.recommender.save_model(temp_model_file)
        
        # Create new recommender and load model
        new_recommender = ContentBasedRecommender()
        new_recommender.load_model(temp_model_file)
        
        assert new_recommender.is_trained is True
        assert new_recommender.model is not None
    
    def test_save_model_before_training(self, temp_model_file):
        """Test that saving fails before training."""
        with pytest.raises(ValueError, match="Model must be trained"):
            self.recommender.save_model(temp_model_file)
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.recommender.get_model_info()
        
        assert isinstance(info, dict)
        assert info['model_type'] == 'content_based_recommender'
        assert info['embedding_dim'] == 64
        assert info['location_vocab_size'] == 100
        assert info['amenity_vocab_size'] == 50
        assert info['is_trained'] is False
        assert info['num_properties'] == 0
        
        # After training
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        info_after_training = self.recommender.get_model_info()
        assert info_after_training['is_trained'] is True
        assert info_after_training['num_properties'] == 3
    
    @pytest.mark.performance
    def test_prediction_performance(self):
        """Test prediction performance with larger dataset."""
        # Create larger dataset
        large_data = self.ml_factory.create_training_data(
            num_users=50, num_properties=100, density=0.1
        )
        
        # Train model
        self.recommender.fit(
            user_item_matrix=large_data['user_item_matrix'],
            property_data=large_data['property_features'],
            epochs=3
        )
        
        # Test prediction performance
        with PerformanceTestHelpers.measure_time() as timer:
            predictions = self.recommender.predict(user_id=0, item_ids=list(range(100)))
        
        elapsed_time = timer()
        PerformanceTestHelpers.assert_performance_threshold(
            elapsed_time, threshold=2.0, operation="Model prediction"
        )
        
        assert len(predictions) == 100
    
    @pytest.mark.performance
    def test_recommendation_performance(self):
        """Test recommendation generation performance."""
        # Create larger dataset
        large_data = self.ml_factory.create_training_data(
            num_users=50, num_properties=200, density=0.1
        )
        
        # Train model
        self.recommender.fit(
            user_item_matrix=large_data['user_item_matrix'],
            property_data=large_data['property_features'],
            epochs=3
        )
        
        # Test recommendation performance
        with PerformanceTestHelpers.measure_time() as timer:
            recommendations = self.recommender.recommend(user_id=0, num_recommendations=20)
        
        elapsed_time = timer()
        PerformanceTestHelpers.assert_performance_threshold(
            elapsed_time, threshold=3.0, operation="Recommendation generation"
        )
        
        assert len(recommendations) <= 20
    
    def test_error_handling_during_training(self):
        """Test error handling during training process."""
        # Test with invalid user-item matrix
        invalid_matrix = np.array([[1, 2, 3]])  # Non-binary values
        
        with pytest.raises(Exception):
            self.recommender.fit(
                user_item_matrix=invalid_matrix,
                property_data=self.sample_properties,
                epochs=1
            )
    
    def test_error_handling_during_prediction(self):
        """Test error handling during prediction."""
        # Train model first
        self.recommender.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=2
        )
        
        # Test with invalid user ID
        with pytest.raises(ValueError):
            self.recommender.predict(user_id=-1, item_ids=[0, 1])
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        # Train first model
        tf.random.set_seed(42)
        np.random.seed(42)
        
        recommender1 = ContentBasedRecommender(
            embedding_dim=32, learning_rate=0.01
        )
        recommender1.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=3
        )
        predictions1 = recommender1.predict(user_id=0, item_ids=[0, 1, 2])
        
        # Train second model with same seed
        tf.random.set_seed(42)
        np.random.seed(42)
        
        recommender2 = ContentBasedRecommender(
            embedding_dim=32, learning_rate=0.01
        )
        recommender2.fit(
            user_item_matrix=self.sample_user_item_matrix,
            property_data=self.sample_properties,
            epochs=3
        )
        predictions2 = recommender2.predict(user_id=0, item_ids=[0, 1, 2])
        
        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(predictions1, predictions2, decimal=2)


class TestPropertyFeatures:
    """Test cases for PropertyFeatures dataclass."""
    
    def test_property_features_creation(self):
        """Test PropertyFeatures creation."""
        location_features = np.array([0, 1, 2])
        price_features = np.array([[1000, 2, 1], [2000, 3, 2]])
        bedroom_features = np.array([[2], [3]])
        bathroom_features = np.array([[1], [2]])
        amenity_features = np.array([[1, 0, 1], [0, 1, 1]])
        combined_features = np.array([[0, 1000, 2, 1, 1, 0, 1], [1, 2000, 3, 2, 0, 1, 1]])
        feature_names = ['location', 'price', 'bedrooms', 'bathrooms', 'amenity_0', 'amenity_1', 'amenity_2']
        
        features = PropertyFeatures(
            location_features=location_features,
            price_features=price_features,
            bedroom_features=bedroom_features,
            bathroom_features=bathroom_features,
            amenity_features=amenity_features,
            combined_features=combined_features,
            feature_names=feature_names
        )
        
        assert np.array_equal(features.location_features, location_features)
        assert np.array_equal(features.price_features, price_features)
        assert np.array_equal(features.bedroom_features, bedroom_features)
        assert np.array_equal(features.bathroom_features, bathroom_features)
        assert np.array_equal(features.amenity_features, amenity_features)
        assert np.array_equal(features.combined_features, combined_features)
        assert features.feature_names == feature_names


class TestContentBasedRecommenderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_model_with_minimal_data(self):
        """Test model with minimal training data."""
        recommender = ContentBasedRecommender(embedding_dim=16)
        
        # Minimal data
        minimal_properties = [{
            'property_id': 0,
            'neighborhood': 'Test',
            'city': 'Test City',
            'price': 1000,
            'bedrooms': 1,
            'bathrooms': 1.0,
            'amenities': ['test'],
            'property_type': 'apartment'
        }]
        
        minimal_matrix = np.array([[1]])
        
        # Should handle minimal data gracefully
        recommender.fit(
            user_item_matrix=minimal_matrix,
            property_data=minimal_properties,
            epochs=1
        )
        
        assert recommender.is_trained is True
    
    def test_model_with_zero_interactions(self):
        """Test model behavior with zero interactions."""
        recommender = ContentBasedRecommender()
        
        # Matrix with no interactions
        zero_matrix = np.zeros((3, 3))
        
        properties = [
            {'property_id': i, 'neighborhood': f'Area{i}', 'city': 'City',
             'price': 1000+i*500, 'bedrooms': 1+i, 'bathrooms': 1.0,
             'amenities': ['test'], 'property_type': 'apartment'}
            for i in range(3)
        ]
        
        # Should handle zero interactions
        recommender.fit(
            user_item_matrix=zero_matrix,
            property_data=properties,
            epochs=1
        )
        
        assert recommender.is_trained is True
    
    def test_model_with_extreme_values(self):
        """Test model with extreme property values."""
        recommender = ContentBasedRecommender()
        
        extreme_properties = [
            {
                'property_id': 0,
                'neighborhood': 'Test',
                'city': 'Test City',
                'price': 1,  # Very low price
                'bedrooms': 0,
                'bathrooms': 0.5,
                'amenities': [],
                'property_type': 'studio'
            },
            {
                'property_id': 1,
                'neighborhood': 'Luxury',
                'city': 'Test City',
                'price': 100000,  # Very high price
                'bedrooms': 20,
                'bathrooms': 15.0,
                'amenities': ['luxury'] * 50,  # Many amenities
                'property_type': 'mansion'
            }
        ]
        
        matrix = np.array([[1, 0], [0, 1]])
        
        # Should handle extreme values
        recommender.fit(
            user_item_matrix=matrix,
            property_data=extreme_properties,
            epochs=2
        )
        
        predictions = recommender.predict(user_id=0, item_ids=[0, 1])
        MLTestHelpers.assert_valid_predictions(predictions)