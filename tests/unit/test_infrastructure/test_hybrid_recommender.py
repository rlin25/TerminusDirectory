"""
Unit tests for the HybridRecommendationSystem
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from infrastructure.ml.models.hybrid_recommender import (
    HybridRecommendationSystem,
    HybridRecommendationResult
)
from infrastructure.ml.models.collaborative_filter import RecommendationResult


class TestHybridRecommendationSystem(unittest.TestCase):
    """Test cases for HybridRecommendationSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hybrid_system = HybridRecommendationSystem(
            cf_weight=0.6,
            cb_weight=0.4,
            min_cf_interactions=3,
            fallback_to_content=True
        )
        
        # Mock models
        self.hybrid_system.cf_model = Mock()
        self.hybrid_system.cb_model = Mock()
        
        # Sample data
        self.user_item_matrix = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0]  # New user with no interactions
        ])
        
        self.property_data = [
            {
                'property_id': 0,
                'neighborhood': 'Downtown',
                'city': 'San Francisco',
                'price': 3000,
                'bedrooms': 2,
                'bathrooms': 1,
                'amenities': ['parking', 'gym']
            },
            {
                'property_id': 1,
                'neighborhood': 'Mission',
                'city': 'San Francisco',
                'price': 2500,
                'bedrooms': 1,
                'bathrooms': 1,
                'amenities': ['gym', 'pool']
            },
            {
                'property_id': 2,
                'neighborhood': 'SoMa',
                'city': 'San Francisco',
                'price': 3500,
                'bedrooms': 3,
                'bathrooms': 2,
                'amenities': ['parking', 'balcony']
            },
            {
                'property_id': 3,
                'neighborhood': 'Castro',
                'city': 'San Francisco',
                'price': 2800,
                'bedrooms': 2,
                'bathrooms': 1,
                'amenities': ['gym', 'rooftop']
            },
            {
                'property_id': 4,
                'neighborhood': 'Marina',
                'city': 'San Francisco',
                'price': 4000,
                'bedrooms': 3,
                'bathrooms': 2,
                'amenities': ['parking', 'pool', 'gym']
            }
        ]
    
    def test_initialization(self):
        """Test hybrid system initialization"""
        # Test normal initialization
        system = HybridRecommendationSystem(cf_weight=0.7, cb_weight=0.3)
        self.assertEqual(system.cf_weight, 0.7)
        self.assertEqual(system.cb_weight, 0.3)
        self.assertFalse(system.is_trained)
        
        # Test weight normalization
        system = HybridRecommendationSystem(cf_weight=0.8, cb_weight=0.6)
        self.assertAlmostEqual(system.cf_weight, 0.8 / 1.4, places=5)
        self.assertAlmostEqual(system.cb_weight, 0.6 / 1.4, places=5)
        
        # Test invalid weights
        with self.assertRaises(ValueError):
            HybridRecommendationSystem(cf_weight=-0.1, cb_weight=0.5)
        
        with self.assertRaises(ValueError):
            HybridRecommendationSystem(cf_weight=0.0, cb_weight=0.0)
    
    def test_should_use_collaborative_filtering(self):
        """Test collaborative filtering availability check"""
        # Setup mock CF model
        self.hybrid_system.cf_model.is_trained = True
        self.hybrid_system.cf_model.user_item_matrix = self.user_item_matrix
        
        # Test user with sufficient interactions
        result = self.hybrid_system._should_use_collaborative_filtering(0)
        self.assertTrue(result)  # User 0 has 3 interactions
        
        # Test user with insufficient interactions
        result = self.hybrid_system._should_use_collaborative_filtering(3)
        self.assertFalse(result)  # User 3 has 0 interactions
        
        # Test with untrained model
        self.hybrid_system.cf_model.is_trained = False
        result = self.hybrid_system._should_use_collaborative_filtering(0)
        self.assertFalse(result)
    
    def test_predict_hybrid_scores(self):
        """Test hybrid prediction generation"""
        # Setup mock models
        self.hybrid_system.cf_model.is_trained = True
        self.hybrid_system.cb_model.is_trained = True
        self.hybrid_system.cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # Mock predictions
        self.hybrid_system.cf_model.predict.return_value = np.array([0.8, 0.6, 0.7])
        self.hybrid_system.cb_model.predict.return_value = np.array([0.5, 0.9, 0.4])
        
        # Test hybrid prediction
        predictions = self.hybrid_system.predict(user_id=0, item_ids=[1, 2, 3])
        
        # Expected: 0.6 * [0.8, 0.6, 0.7] + 0.4 * [0.5, 0.9, 0.4]
        expected = np.array([0.68, 0.72, 0.58])
        np.testing.assert_array_almost_equal(predictions, expected, decimal=5)
        
        # Test with untrained system
        self.hybrid_system.is_trained = False
        with self.assertRaises(ValueError):
            self.hybrid_system.predict(user_id=0, item_ids=[1, 2, 3])
    
    def test_predict_content_based_only(self):
        """Test prediction with content-based only (new user)"""
        # Setup for new user (no CF)
        self.hybrid_system.cf_model.is_trained = True
        self.hybrid_system.cb_model.is_trained = True
        self.hybrid_system.cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # Mock CB predictions only
        self.hybrid_system.cb_model.predict.return_value = np.array([0.5, 0.9, 0.4])
        
        # Test prediction for new user (user 3 has no interactions)
        predictions = self.hybrid_system.predict(user_id=3, item_ids=[1, 2, 4])
        
        # Should use only content-based
        expected = np.array([0.5, 0.9, 0.4])
        np.testing.assert_array_almost_equal(predictions, expected, decimal=5)
    
    def test_combine_recommendations(self):
        """Test recommendation combination logic"""
        # Create mock recommendations
        cf_recs = [
            RecommendationResult(item_id=1, predicted_rating=0.8, confidence_score=0.9, explanation="CF rec 1"),
            RecommendationResult(item_id=2, predicted_rating=0.6, confidence_score=0.8, explanation="CF rec 2"),
            RecommendationResult(item_id=3, predicted_rating=0.7, confidence_score=0.85, explanation="CF rec 3")
        ]
        
        cb_recs = [
            RecommendationResult(item_id=1, predicted_rating=0.5, confidence_score=0.7, explanation="CB rec 1"),
            RecommendationResult(item_id=2, predicted_rating=0.9, confidence_score=0.95, explanation="CB rec 2"),
            RecommendationResult(item_id=4, predicted_rating=0.4, confidence_score=0.6, explanation="CB rec 4")
        ]
        
        # Test combination
        combined = self.hybrid_system._combine_recommendations(cf_recs, cb_recs, num_recommendations=5)
        
        # Should have 4 unique items (1, 2, 3, 4)
        self.assertEqual(len(combined), 4)
        
        # Check hybrid scores
        item_1 = next(rec for rec in combined if rec.item_id == 1)
        expected_score = 0.6 * 0.8 + 0.4 * 0.5  # 0.68
        self.assertAlmostEqual(item_1.predicted_rating, expected_score, places=5)
        
        item_2 = next(rec for rec in combined if rec.item_id == 2)
        expected_score = 0.6 * 0.6 + 0.4 * 0.9  # 0.72
        self.assertAlmostEqual(item_2.predicted_rating, expected_score, places=5)
        
        # Check that recommendations are sorted by score
        scores = [rec.predicted_rating for rec in combined]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_convert_to_hybrid_results(self):
        """Test conversion of single-model results to hybrid format"""
        single_recs = [
            RecommendationResult(item_id=1, predicted_rating=0.8, confidence_score=0.9, explanation="Single rec 1"),
            RecommendationResult(item_id=2, predicted_rating=0.6, confidence_score=0.8, explanation="Single rec 2")
        ]
        
        # Test CF-only conversion
        cf_only = self.hybrid_system._convert_to_hybrid_results(single_recs, "collaborative_filtering_only")
        self.assertEqual(len(cf_only), 2)
        self.assertEqual(cf_only[0].cf_score, 0.8)
        self.assertIsNone(cf_only[0].cb_score)
        self.assertEqual(cf_only[0].hybrid_method, "collaborative_filtering_only")
        
        # Test CB-only conversion
        cb_only = self.hybrid_system._convert_to_hybrid_results(single_recs, "content_based_only")
        self.assertEqual(len(cb_only), 2)
        self.assertIsNone(cb_only[0].cf_score)
        self.assertEqual(cb_only[0].cb_score, 0.8)
        self.assertEqual(cb_only[0].hybrid_method, "content_based_only")
    
    def test_calculate_hybrid_confidence(self):
        """Test hybrid confidence calculation"""
        # Test with both CF and CB
        cf_rec = RecommendationResult(item_id=1, predicted_rating=0.8, confidence_score=0.9, explanation="CF")
        cb_rec = RecommendationResult(item_id=1, predicted_rating=0.7, confidence_score=0.8, explanation="CB")
        
        confidence = self.hybrid_system._calculate_hybrid_confidence(cf_rec, cb_rec)
        
        # Expected: weighted average + agreement boost
        expected_base = 0.6 * 0.9 + 0.4 * 0.8  # 0.86
        agreement = 1.0 - abs(0.8 - 0.7)  # 0.9
        agreement_boost = min(agreement * 0.1, 0.2)  # 0.09
        expected = min(expected_base + agreement_boost, 1.0)
        
        self.assertAlmostEqual(confidence, expected, places=5)
        
        # Test with only CF
        confidence_cf_only = self.hybrid_system._calculate_hybrid_confidence(cf_rec, None)
        self.assertAlmostEqual(confidence_cf_only, 0.9 * 0.9, places=5)
        
        # Test with only CB
        confidence_cb_only = self.hybrid_system._calculate_hybrid_confidence(None, cb_rec)
        self.assertAlmostEqual(confidence_cb_only, 0.8 * 0.9, places=5)
    
    def test_generate_hybrid_explanation(self):
        """Test hybrid explanation generation"""
        cf_rec = RecommendationResult(item_id=1, predicted_rating=0.8, confidence_score=0.9, explanation="CF explanation")
        cb_rec = RecommendationResult(item_id=1, predicted_rating=0.7, confidence_score=0.8, explanation="CB explanation")
        
        # Test detailed explanation
        self.hybrid_system.explanation_detail_level = "detailed"
        explanation = self.hybrid_system._generate_hybrid_explanation(cf_rec, cb_rec)
        self.assertIn("similar users", explanation)
        self.assertIn("property features", explanation)
        self.assertIn("0.80", explanation)
        self.assertIn("0.70", explanation)
        
        # Test simple explanation
        self.hybrid_system.explanation_detail_level = "simple"
        explanation = self.hybrid_system._generate_hybrid_explanation(cf_rec, cb_rec)
        self.assertEqual(explanation, "Recommended based on similar users and property features")
        
        # Test technical explanation
        self.hybrid_system.explanation_detail_level = "technical"
        explanation = self.hybrid_system._generate_hybrid_explanation(cf_rec, cb_rec)
        self.assertIn("CF(0.800)", explanation)
        self.assertIn("CB(0.700)", explanation)
        self.assertIn("0.60", explanation)
        self.assertIn("0.40", explanation)
    
    def test_update_weights(self):
        """Test dynamic weight updates"""
        # Test valid weight update
        self.hybrid_system.update_weights(cf_weight=0.8, cb_weight=0.2)
        self.assertEqual(self.hybrid_system.cf_weight, 0.8)
        self.assertEqual(self.hybrid_system.cb_weight, 0.2)
        
        # Test weight normalization
        self.hybrid_system.update_weights(cf_weight=0.6, cb_weight=0.9)
        self.assertAlmostEqual(self.hybrid_system.cf_weight, 0.6 / 1.5, places=5)
        self.assertAlmostEqual(self.hybrid_system.cb_weight, 0.9 / 1.5, places=5)
        
        # Test invalid weights
        with self.assertRaises(ValueError):
            self.hybrid_system.update_weights(cf_weight=-0.1, cb_weight=0.5)
        
        with self.assertRaises(ValueError):
            self.hybrid_system.update_weights(cf_weight=0.0, cb_weight=0.0)
    
    def test_get_system_info(self):
        """Test system information retrieval"""
        # Setup mock model info
        self.hybrid_system.cf_model.get_model_info.return_value = {
            'model_type': 'collaborative_filtering',
            'is_trained': True
        }
        self.hybrid_system.cb_model.get_model_info.return_value = {
            'model_type': 'content_based',
            'is_trained': True
        }
        
        info = self.hybrid_system.get_system_info()
        
        self.assertEqual(info['system_type'], 'hybrid_recommendation_system')
        self.assertEqual(info['cf_weight'], 0.6)
        self.assertEqual(info['cb_weight'], 0.4)
        self.assertEqual(info['min_cf_interactions'], 3)
        self.assertTrue(info['fallback_to_content'])
        self.assertIn('cf_model_info', info)
        self.assertIn('cb_model_info', info)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with uninitialized models
        system = HybridRecommendationSystem()
        with self.assertRaises(ValueError):
            system.predict(user_id=0, item_ids=[1, 2, 3])
        
        # Test with empty item list
        self.hybrid_system.is_trained = True
        predictions = self.hybrid_system.predict(user_id=0, item_ids=[])
        self.assertEqual(len(predictions), 0)
        
        # Test CF failure fallback
        self.hybrid_system.cf_model.is_trained = True
        self.hybrid_system.cb_model.is_trained = True
        self.hybrid_system.cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # Mock CF to fail, CB to succeed
        self.hybrid_system.cf_model.predict.side_effect = Exception("CF failed")
        self.hybrid_system.cb_model.predict.return_value = np.array([0.5, 0.9, 0.4])
        
        predictions = self.hybrid_system.predict(user_id=0, item_ids=[1, 2, 3])
        np.testing.assert_array_almost_equal(predictions, np.array([0.5, 0.9, 0.4]), decimal=5)


class TestHybridRecommendationResult(unittest.TestCase):
    """Test cases for HybridRecommendationResult dataclass"""
    
    def test_hybrid_result_creation(self):
        """Test hybrid recommendation result creation"""
        result = HybridRecommendationResult(
            item_id=123,
            predicted_rating=0.85,
            confidence_score=0.9,
            explanation="Test explanation",
            cf_score=0.8,
            cb_score=0.7,
            hybrid_method="weighted_average"
        )
        
        self.assertEqual(result.item_id, 123)
        self.assertEqual(result.predicted_rating, 0.85)
        self.assertEqual(result.confidence_score, 0.9)
        self.assertEqual(result.explanation, "Test explanation")
        self.assertEqual(result.cf_score, 0.8)
        self.assertEqual(result.cb_score, 0.7)
        self.assertEqual(result.hybrid_method, "weighted_average")
    
    def test_hybrid_result_defaults(self):
        """Test default values in hybrid result"""
        result = HybridRecommendationResult(
            item_id=123,
            predicted_rating=0.85,
            confidence_score=0.9,
            explanation="Test explanation"
        )
        
        self.assertIsNone(result.cf_score)
        self.assertIsNone(result.cb_score)
        self.assertEqual(result.hybrid_method, "weighted_average")
        self.assertIsNone(result.feature_importance)


if __name__ == '__main__':
    unittest.main()