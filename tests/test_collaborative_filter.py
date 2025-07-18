#!/usr/bin/env python3
"""
Comprehensive test suite for the collaborative filtering model.

Tests cover:
1. Data preprocessing functionality
2. Model training and validation
3. Prediction accuracy
4. Cold start handling
5. Model persistence
6. Performance monitoring
7. Integration with model repository
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import logging
from unittest.mock import patch, MagicMock

# Suppress TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.ml.models.collaborative_filter import (
    CollaborativeFilteringModel,
    TrainingConfig,
    DataPreprocessor,
    ModelEvaluator,
    ColdStartHandler,
    RecommendationResult,
    EvaluationMetrics
)

class TestDataPreprocessor(unittest.TestCase):
    """Test data preprocessing functionality"""
    
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.preprocessor = DataPreprocessor(self.logger)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
            'item_id': [1, 2, 1, 3, 2, 3, 1, 4],
            'rating': [5, 4, 3, 5, 4, 2, 5, 3],
            'timestamp': pd.date_range('2023-01-01', periods=8)
        })
    
    def test_fit_transform_interactions(self):
        """Test interaction data preprocessing"""
        user_item_matrix, training_data = self.preprocessor.fit_transform_interactions(
            self.sample_data
        )
        
        self.assertIsInstance(user_item_matrix, np.ndarray)
        self.assertIsInstance(training_data, dict)
        self.assertIn('users', training_data)
        self.assertIn('items', training_data)
        self.assertIn('ratings', training_data)
        self.assertIn('num_users', training_data)
        self.assertIn('num_items', training_data)
        
        # Check dimensions
        self.assertEqual(user_item_matrix.shape[0], training_data['num_users'])
        self.assertEqual(user_item_matrix.shape[1], training_data['num_items'])
        self.assertTrue(self.preprocessor.is_fitted)
    
    def test_add_negative_samples(self):
        """Test negative sampling"""
        user_item_matrix, training_data = self.preprocessor.fit_transform_interactions(
            self.sample_data
        )
        
        original_length = len(training_data['users'])
        training_data = self.preprocessor.add_negative_samples(training_data, negative_ratio=0.5)
        
        self.assertGreater(len(training_data['users']), original_length)
        self.assertIn('negative_samples', training_data)
        self.assertGreater(training_data['negative_samples'], 0)
    
    def test_id_transformations(self):
        """Test ID encoding and decoding"""
        user_item_matrix, training_data = self.preprocessor.fit_transform_interactions(
            self.sample_data
        )
        
        # Test user ID transformation
        original_user_id = 1
        encoded_user_id = self.preprocessor.transform_user_id(original_user_id)
        decoded_user_id = self.preprocessor.inverse_transform_user_id(encoded_user_id)
        
        self.assertEqual(original_user_id, decoded_user_id)
        
        # Test item ID transformation
        original_item_id = 2
        encoded_item_id = self.preprocessor.transform_item_id(original_item_id)
        decoded_item_id = self.preprocessor.inverse_transform_item_id(encoded_item_id)
        
        self.assertEqual(original_item_id, decoded_item_id)
    
    def test_unknown_id_handling(self):
        """Test handling of unknown IDs"""
        user_item_matrix, training_data = self.preprocessor.fit_transform_interactions(
            self.sample_data
        )
        
        # Test unknown user ID
        unknown_user_encoded = self.preprocessor.transform_user_id(999)
        self.assertEqual(unknown_user_encoded, -1)
        
        # Test unknown item ID
        unknown_item_encoded = self.preprocessor.transform_item_id(999)
        self.assertEqual(unknown_item_encoded, -1)


class TestCollaborativeFilteringModel(unittest.TestCase):
    """Test collaborative filtering model functionality"""
    
    def setUp(self):
        self.config = TrainingConfig(
            epochs=2,  # Small number for testing
            batch_size=32,
            embedding_dim=16,
            hidden_layers=[32, 16],
            early_stopping_patience=1
        )
        
        self.model = CollaborativeFilteringModel(
            num_users=50,
            num_items=30,
            config=self.config
        )
        
        # Create sample training data
        self.sample_interactions = pd.DataFrame({
            'user_id': np.random.randint(0, 50, 200),
            'item_id': np.random.randint(0, 30, 200),
            'rating': np.random.randint(1, 6, 200),
            'timestamp': pd.date_range('2023-01-01', periods=200)
        })
        
        # Remove duplicates
        self.sample_interactions = self.sample_interactions.drop_duplicates(
            subset=['user_id', 'item_id'],
            keep='last'
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model.model)
        self.assertFalse(self.model.is_trained)
        self.assertIsInstance(self.model.preprocessor, DataPreprocessor)
        self.assertIsInstance(self.model.evaluator, ModelEvaluator)
        self.assertIsInstance(self.model.cold_start_handler, ColdStartHandler)
    
    def test_model_training(self):
        """Test model training"""
        training_results = self.model.fit(interactions_df=self.sample_interactions)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsInstance(training_results, dict)
        self.assertIn('final_loss', training_results)
        self.assertIn('val_loss', training_results)
        self.assertIn('epochs_trained', training_results)
        self.assertGreater(training_results['epochs_trained'], 0)
    
    def test_predictions(self):
        """Test model predictions"""
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        # Test single prediction
        predictions = self.model.predict(user_id=0, item_ids=[0, 1, 2])
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
        
        # Test batch predictions
        batch_predictions = self.model.predict_batch(
            user_ids=np.array([0, 1, 2]),
            item_ids=np.array([0, 1, 2])
        )
        self.assertIsInstance(batch_predictions, np.ndarray)
        self.assertEqual(len(batch_predictions), 3)
    
    def test_recommendations(self):
        """Test recommendation generation"""
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        # Test recommendations
        recommendations = self.model.recommend(user_id=0, num_recommendations=5)
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        # Check recommendation structure
        if recommendations:
            rec = recommendations[0]
            self.assertIsInstance(rec, RecommendationResult)
            self.assertIsInstance(rec.item_id, int)
            self.assertIsInstance(rec.predicted_rating, float)
            self.assertIsInstance(rec.confidence_score, float)
            self.assertIsInstance(rec.explanation, str)
    
    def test_cold_start_handling(self):
        """Test cold start user handling"""
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        # Test cold start user
        cold_recommendations = self.model.recommend(user_id=999, num_recommendations=5)
        self.assertIsInstance(cold_recommendations, list)
        # Should still provide recommendations using cold start handler
        
        # Test cold start item prediction
        cold_predictions = self.model.predict(user_id=0, item_ids=[999])
        self.assertIsInstance(cold_predictions, np.ndarray)
        self.assertEqual(len(cold_predictions), 1)
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            model_path = tmp_file.name
            
        try:
            self.model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test loading
            new_model = CollaborativeFilteringModel(
                num_users=50,
                num_items=30,
                config=self.config
            )
            new_model.load_model(model_path)
            
            self.assertTrue(new_model.is_trained)
            
            # Test that loaded model can make predictions
            predictions = new_model.predict(user_id=0, item_ids=[0, 1])
            self.assertIsInstance(predictions, np.ndarray)
            self.assertEqual(len(predictions), 2)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
            metadata_path = model_path.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
            preprocessor_path = model_path.replace('.h5', '_preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                os.unlink(preprocessor_path)
    
    def test_similarity_functions(self):
        """Test user and item similarity functions"""
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        # Test similar users
        similar_users = self.model.get_similar_users(user_id=0, num_similar=3)
        self.assertIsInstance(similar_users, list)
        self.assertLessEqual(len(similar_users), 3)
        
        if similar_users:
            user_id, similarity = similar_users[0]
            self.assertIsInstance(user_id, int)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, -1)
            self.assertLessEqual(similarity, 1)
        
        # Test similar items
        similar_items = self.model.get_similar_items(item_id=0, num_similar=3)
        self.assertIsInstance(similar_items, list)
        self.assertLessEqual(len(similar_items), 3)
        
        if similar_items:
            item_id, similarity = similar_items[0]
            self.assertIsInstance(item_id, int)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, -1)
            self.assertLessEqual(similarity, 1)
    
    def test_model_info(self):
        """Test model information retrieval"""
        model_info = self.model.get_model_info()
        
        self.assertIsInstance(model_info, dict)
        self.assertIn('model_type', model_info)
        self.assertIn('num_users', model_info)
        self.assertIn('num_items', model_info)
        self.assertIn('embedding_dim', model_info)
        self.assertIn('is_trained', model_info)
        self.assertIn('config', model_info)
        
        self.assertEqual(model_info['model_type'], 'neural_collaborative_filtering')
        self.assertEqual(model_info['num_users'], 50)
        self.assertEqual(model_info['num_items'], 30)
    
    def test_feature_importance(self):
        """Test feature importance analysis"""
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        feature_importance = self.model.get_feature_importance()
        
        self.assertIsInstance(feature_importance, dict)
        self.assertIn('user_embedding_stats', feature_importance)
        self.assertIn('item_embedding_stats', feature_importance)
        self.assertIn('most_important_user_dims', feature_importance)
        self.assertIn('most_important_item_dims', feature_importance)
    
    def test_recommendation_explanation(self):
        """Test detailed recommendation explanation"""
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        explanation = self.model.explain_recommendation(user_id=0, item_id=0)
        
        self.assertIsInstance(explanation, dict)
        self.assertIn('user_id', explanation)
        self.assertIn('item_id', explanation)
        self.assertIn('predicted_rating', explanation)
        self.assertIn('confidence', explanation)
        self.assertIn('similar_items', explanation)
        self.assertIn('similar_users', explanation)
    
    @patch('psutil.Process')
    @patch('psutil.cpu_percent')
    def test_performance_monitoring(self, mock_cpu_percent, mock_process):
        """Test performance monitoring"""
        # Mock system metrics
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_cpu_percent.return_value = 15.5
        
        # Train model first
        self.model.fit(interactions_df=self.sample_interactions)
        
        # Prepare test data
        test_data = {
            'users': np.array([0, 1, 2]),
            'items': np.array([0, 1, 2]),
            'ratings': np.array([4.0, 3.0, 5.0])
        }
        
        performance_info = self.model.monitor_performance(test_data)
        
        self.assertIsInstance(performance_info, dict)
        self.assertIn('evaluation_metrics', performance_info)
        self.assertIn('system_metrics', performance_info)
        self.assertIn('model_health', performance_info)
        self.assertIn('timestamp', performance_info)
        
        # Check system metrics
        self.assertIn('memory_usage_mb', performance_info['system_metrics'])
        self.assertIn('cpu_percent', performance_info['system_metrics'])
        self.assertIn('prediction_latency_ms', performance_info['system_metrics'])


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation functionality"""
    
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.evaluator = ModelEvaluator(self.logger)
        
        # Create a simple mock model
        self.mock_model = MagicMock()
        self.mock_model.predict_batch.return_value = np.array([0.8, 0.6, 0.4])
        self.mock_model.recommend.return_value = [
            RecommendationResult(item_id=0, predicted_rating=0.8, confidence_score=0.9, explanation="test"),
            RecommendationResult(item_id=1, predicted_rating=0.6, confidence_score=0.7, explanation="test")
        ]
        
        # Sample test data
        self.test_data = {
            'users': np.array([0, 1, 2]),
            'items': np.array([0, 1, 2]),
            'ratings': np.array([0.9, 0.5, 0.3])
        }
        
        self.user_item_matrix = np.array([
            [0.9, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.3]
        ])
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        rmse = self.evaluator._calculate_rmse(self.mock_model, self.test_data)
        self.assertIsInstance(rmse, float)
        self.assertGreater(rmse, 0)
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        mae = self.evaluator._calculate_mae(self.mock_model, self.test_data)
        self.assertIsInstance(mae, float)
        self.assertGreater(mae, 0)
    
    def test_coverage_calculation(self):
        """Test coverage calculation"""
        coverage = self.evaluator._calculate_coverage(
            self.mock_model, 
            self.test_data, 
            self.user_item_matrix
        )
        self.assertIsInstance(coverage, float)
        self.assertGreaterEqual(coverage, 0)
        self.assertLessEqual(coverage, 1)


class TestColdStartHandler(unittest.TestCase):
    """Test cold start handling functionality"""
    
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.cold_start_handler = ColdStartHandler(self.logger)
        
        # Set up popularity model
        user_item_matrix = np.array([
            [5, 0, 4],
            [0, 3, 0],
            [4, 0, 5]
        ])
        self.cold_start_handler.set_popularity_model(user_item_matrix)
    
    def test_cold_user_handling(self):
        """Test cold user recommendation"""
        recommendations = self.cold_start_handler.handle_cold_user(
            user_features={},
            num_recommendations=5
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        if recommendations:
            self.assertIsInstance(recommendations[0], RecommendationResult)
    
    def test_cold_item_handling(self):
        """Test cold item rating prediction"""
        rating = self.cold_start_handler.handle_cold_item(
            item_features={},
            user_id=0
        )
        
        self.assertIsInstance(rating, float)
        self.assertGreaterEqual(rating, 0)
        self.assertLessEqual(rating, 1)


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.embedding_dim, 128)
        self.assertEqual(config.hidden_layers, [256, 128, 64])
        self.assertEqual(config.dropout_rate, 0.2)
        self.assertEqual(config.regularization, 1e-6)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            epochs=50,
            batch_size=512,
            embedding_dim=64,
            hidden_layers=[128, 64],
            dropout_rate=0.3
        )
        
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.batch_size, 512)
        self.assertEqual(config.embedding_dim, 64)
        self.assertEqual(config.hidden_layers, [128, 64])
        self.assertEqual(config.dropout_rate, 0.3)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)