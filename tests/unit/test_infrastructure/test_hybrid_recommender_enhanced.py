"""
Enhanced unit tests for the HybridRecommendationSystem.

Comprehensive tests covering all aspects of the hybrid recommendation system
including error handling, performance, and edge cases.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

from infrastructure.ml.models.hybrid_recommender import (
    HybridRecommendationSystem,
    HybridRecommendationResult
)
from infrastructure.ml.models.collaborative_filter import RecommendationResult
from tests.utils.test_helpers import MLTestHelpers, PerformanceTestHelpers
from tests.utils.data_factories import MLDataFactory, FactoryConfig


class TestHybridRecommendationSystemEnhanced:
    """Enhanced test cases for HybridRecommendationSystem."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        np.random.seed(42)
        
        self.hybrid_system = HybridRecommendationSystem(
            cf_weight=0.6,
            cb_weight=0.4,
            min_cf_interactions=3,
            fallback_to_content=True,
            explanation_detail_level="detailed"
        )
        
        # Mock the component models
        self.mock_cf_model = Mock()
        self.mock_cb_model = Mock()
        
        self.hybrid_system.cf_model = self.mock_cf_model
        self.hybrid_system.cb_model = self.mock_cb_model
        
        # Sample data
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
        self.sample_training_data = self.ml_factory.create_training_data(
            num_users=10, num_properties=20, density=0.15
        )
        
        # Sample user-item matrix for testing
        self.user_item_matrix = np.array([
            [1, 0, 1, 0, 1, 0],  # User 0: active user
            [0, 1, 0, 1, 0, 1],  # User 1: active user
            [1, 1, 0, 0, 1, 1],  # User 2: very active user
            [0, 0, 0, 0, 0, 0],  # User 3: new user with no interactions
            [1, 0, 0, 1, 0, 0],  # User 4: minimal interactions
        ])
    
    def test_initialization_with_various_configurations(self):
        """Test hybrid system initialization with different configurations."""
        # Test different weight combinations
        configs = [
            (0.7, 0.3), (0.5, 0.5), (0.8, 0.2), (0.1, 0.9)
        ]
        
        for cf_weight, cb_weight in configs:
            system = HybridRecommendationSystem(cf_weight=cf_weight, cb_weight=cb_weight)
            expected_total = cf_weight + cb_weight
            
            assert abs(system.cf_weight - cf_weight / expected_total) < 1e-6
            assert abs(system.cb_weight - cb_weight / expected_total) < 1e-6
            assert abs(system.cf_weight + system.cb_weight - 1.0) < 1e-6
    
    def test_initialization_edge_cases(self):
        """Test initialization with edge case parameters."""
        # Very small weights
        system = HybridRecommendationSystem(cf_weight=1e-10, cb_weight=1.0)
        assert system.cf_weight > 0
        assert system.cb_weight > 0
        
        # Large weights that need normalization
        system = HybridRecommendationSystem(cf_weight=1000, cb_weight=2000)
        assert abs(system.cf_weight - 1000/3000) < 1e-6
        assert abs(system.cb_weight - 2000/3000) < 1e-6
        
        # Test invalid configurations
        with pytest.raises(ValueError):
            HybridRecommendationSystem(cf_weight=-0.1, cb_weight=0.5)
        
        with pytest.raises(ValueError):
            HybridRecommendationSystem(cf_weight=0.0, cb_weight=0.0)
    
    def test_component_model_training(self):
        """Test training of component models."""
        # Setup mock return values
        self.mock_cf_model.fit.return_value = {'loss': 0.1, 'accuracy': 0.9}
        self.mock_cb_model.fit.return_value = {'loss': 0.15, 'accuracy': 0.85}
        
        training_result = self.hybrid_system.fit(
            user_item_matrix=self.sample_training_data['user_item_matrix'],
            property_data=self.sample_training_data['property_features'],
            epochs=5
        )
        
        # Check that both models were trained
        self.mock_cf_model.fit.assert_called_once()
        self.mock_cb_model.fit.assert_called_once()
        
        # Check training result
        assert isinstance(training_result, dict)
        assert 'cf_training_result' in training_result
        assert 'cb_training_result' in training_result
        assert self.hybrid_system.is_trained is True
    
    def test_collaborative_filtering_availability_logic(self):
        """Test the logic for determining CF availability."""
        self.mock_cf_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        
        # Test different users with different interaction counts
        test_cases = [
            (0, True),   # User 0 has 3 interactions (meets threshold)
            (1, True),   # User 1 has 3 interactions
            (2, True),   # User 2 has 4 interactions
            (3, False),  # User 3 has 0 interactions
            (4, False),  # User 4 has 2 interactions (below threshold)
        ]
        
        for user_id, expected in test_cases:
            result = self.hybrid_system._should_use_collaborative_filtering(user_id)
            assert result == expected, f"User {user_id}: expected {expected}, got {result}"
    
    def test_collaborative_filtering_with_untrained_model(self):
        """Test CF availability when model is not trained."""
        self.mock_cf_model.is_trained = False
        
        # Should return False even for active users
        result = self.hybrid_system._should_use_collaborative_filtering(0)
        assert result is False
    
    def test_hybrid_prediction_with_both_models(self):
        """Test hybrid prediction when both models are available."""
        # Setup
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # Mock predictions
        cf_predictions = np.array([0.8, 0.6, 0.7])
        cb_predictions = np.array([0.5, 0.9, 0.4])
        
        self.mock_cf_model.predict.return_value = cf_predictions
        self.mock_cb_model.predict.return_value = cb_predictions
        
        # Test prediction
        predictions = self.hybrid_system.predict(user_id=0, item_ids=[1, 2, 3])
        
        # Expected: 0.6 * cf + 0.4 * cb
        expected = 0.6 * cf_predictions + 0.4 * cb_predictions
        np.testing.assert_array_almost_equal(predictions, expected, decimal=5)
    
    def test_hybrid_prediction_content_only(self):
        """Test hybrid prediction for new users (content-based only)."""
        # Setup for new user
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        cb_predictions = np.array([0.5, 0.9, 0.4])
        self.mock_cb_model.predict.return_value = cb_predictions
        
        # User 3 has no interactions - should use CB only
        predictions = self.hybrid_system.predict(user_id=3, item_ids=[1, 2, 4])
        
        np.testing.assert_array_almost_equal(predictions, cb_predictions, decimal=5)
        
        # CF should not be called for new users
        self.mock_cf_model.predict.assert_not_called()
    
    def test_prediction_error_handling(self):
        """Test prediction error handling and fallbacks."""
        # Setup
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # CF fails, CB succeeds
        self.mock_cf_model.predict.side_effect = Exception("CF failed")
        cb_predictions = np.array([0.5, 0.9, 0.4])
        self.mock_cb_model.predict.return_value = cb_predictions
        
        predictions = self.hybrid_system.predict(user_id=0, item_ids=[1, 2, 3])
        
        # Should fallback to content-based only
        np.testing.assert_array_almost_equal(predictions, cb_predictions, decimal=5)
    
    def test_prediction_both_models_fail(self):
        """Test prediction when both models fail."""
        # Setup
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # Both models fail
        self.mock_cf_model.predict.side_effect = Exception("CF failed")
        self.mock_cb_model.predict.side_effect = Exception("CB failed")
        
        predictions = self.hybrid_system.predict(user_id=0, item_ids=[1, 2, 3])
        
        # Should return empty array
        assert len(predictions) == 0
    
    def test_recommendation_generation_hybrid(self):
        """Test recommendation generation using hybrid approach."""
        # Setup mocks
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # Mock recommendation results
        cf_recs = [
            RecommendationResult(1, 0.8, 0.9, "CF rec 1"),
            RecommendationResult(2, 0.7, 0.8, "CF rec 2"),
        ]
        cb_recs = [
            RecommendationResult(1, 0.6, 0.7, "CB rec 1"),
            RecommendationResult(3, 0.5, 0.6, "CB rec 3"),
        ]
        
        self.mock_cf_model.recommend.return_value = cf_recs
        self.mock_cb_model.recommend.return_value = cb_recs
        
        recommendations = self.hybrid_system.recommend(user_id=0, num_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(rec, HybridRecommendationResult) for rec in recommendations)
        
        # Check that hybrid method is set correctly
        for rec in recommendations:
            assert rec.hybrid_method in ["weighted_average", "collaborative_filtering_only", "content_based_only"]
    
    def test_recommendation_generation_content_only(self):
        """Test recommendation generation for new users (content-based only)."""
        # Setup for new user
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        cb_recs = [
            RecommendationResult(1, 0.8, 0.9, "CB rec 1"),
            RecommendationResult(2, 0.7, 0.8, "CB rec 2"),
        ]
        
        self.mock_cb_model.recommend.return_value = cb_recs
        
        # User 3 has no interactions
        recommendations = self.hybrid_system.recommend(user_id=3, num_recommendations=2)
        
        assert len(recommendations) == 2
        assert all(rec.hybrid_method == "content_based_only" for rec in recommendations)
        assert all(rec.cf_score is None for rec in recommendations)
        assert all(rec.cb_score is not None for rec in recommendations)
    
    def test_recommendation_combination_logic(self):
        """Test the logic for combining recommendations from different models."""
        cf_recs = [
            RecommendationResult(1, 0.8, 0.9, "CF rec 1"),
            RecommendationResult(2, 0.6, 0.8, "CF rec 2"),
            RecommendationResult(3, 0.7, 0.85, "CF rec 3")
        ]
        
        cb_recs = [
            RecommendationResult(1, 0.5, 0.7, "CB rec 1"),
            RecommendationResult(2, 0.9, 0.95, "CB rec 2"),
            RecommendationResult(4, 0.4, 0.6, "CB rec 4")
        ]
        
        combined = self.hybrid_system._combine_recommendations(cf_recs, cb_recs, num_recommendations=5)
        
        # Should have 4 unique items (1, 2, 3, 4)
        assert len(combined) == 4
        
        # Items 1 and 2 should have hybrid scores
        item_1 = next(rec for rec in combined if rec.item_id == 1)
        item_2 = next(rec for rec in combined if rec.item_id == 2)
        
        expected_score_1 = 0.6 * 0.8 + 0.4 * 0.5  # 0.68
        expected_score_2 = 0.6 * 0.6 + 0.4 * 0.9  # 0.72
        
        assert abs(item_1.predicted_rating - expected_score_1) < 1e-5
        assert abs(item_2.predicted_rating - expected_score_2) < 1e-5
        
        # Should be sorted by predicted rating
        ratings = [rec.predicted_rating for rec in combined]
        assert ratings == sorted(ratings, reverse=True)
    
    def test_confidence_calculation_variations(self):
        """Test different confidence calculation scenarios."""
        # Both models agree
        cf_rec = RecommendationResult(1, 0.8, 0.9, "CF")
        cb_rec = RecommendationResult(1, 0.8, 0.8, "CB")
        
        confidence = self.hybrid_system._calculate_hybrid_confidence(cf_rec, cb_rec)
        
        # Should get agreement boost for similar predictions
        base_confidence = 0.6 * 0.9 + 0.4 * 0.8  # 0.86
        agreement = 1.0 - abs(0.8 - 0.8)  # 1.0
        agreement_boost = min(agreement * 0.1, 0.2)  # 0.1
        expected = min(base_confidence + agreement_boost, 1.0)
        
        assert abs(confidence - expected) < 1e-5
        
        # Models disagree
        cf_rec_disagree = RecommendationResult(1, 0.9, 0.9, "CF")
        cb_rec_disagree = RecommendationResult(1, 0.3, 0.8, "CB")
        
        confidence_disagree = self.hybrid_system._calculate_hybrid_confidence(cf_rec_disagree, cb_rec_disagree)
        
        # Should get less boost due to disagreement
        base_confidence_disagree = 0.6 * 0.9 + 0.4 * 0.8
        agreement_disagree = 1.0 - abs(0.9 - 0.3)  # 0.4
        
        assert confidence_disagree < confidence  # Should be lower due to disagreement
    
    def test_explanation_generation_variations(self):
        """Test explanation generation for different detail levels."""
        cf_rec = RecommendationResult(1, 0.8, 0.9, "Similar users liked this")
        cb_rec = RecommendationResult(1, 0.7, 0.8, "Matches your preferences")
        
        # Test different detail levels
        detail_levels = ["simple", "detailed", "technical"]
        
        for level in detail_levels:
            self.hybrid_system.explanation_detail_level = level
            explanation = self.hybrid_system._generate_hybrid_explanation(cf_rec, cb_rec)
            
            assert isinstance(explanation, str)
            assert len(explanation) > 0
            
            if level == "simple":
                assert "similar users" in explanation.lower()
                assert "property features" in explanation.lower()
            elif level == "detailed":
                assert "0.80" in explanation or "0.70" in explanation
            elif level == "technical":
                assert "CF(" in explanation
                assert "CB(" in explanation
    
    def test_weight_updates(self):
        """Test dynamic weight updating."""
        original_cf_weight = self.hybrid_system.cf_weight
        original_cb_weight = self.hybrid_system.cb_weight
        
        # Update weights
        self.hybrid_system.update_weights(cf_weight=0.8, cb_weight=0.2)
        
        assert self.hybrid_system.cf_weight == 0.8
        assert self.hybrid_system.cb_weight == 0.2
        assert abs(self.hybrid_system.cf_weight + self.hybrid_system.cb_weight - 1.0) < 1e-6
        
        # Test normalization during update
        self.hybrid_system.update_weights(cf_weight=3.0, cb_weight=1.0)
        
        assert abs(self.hybrid_system.cf_weight - 0.75) < 1e-6
        assert abs(self.hybrid_system.cb_weight - 0.25) < 1e-6
    
    def test_system_info_retrieval(self):
        """Test system information and metadata retrieval."""
        # Setup mock model info
        self.mock_cf_model.get_model_info.return_value = {
            'model_type': 'collaborative_filtering',
            'is_trained': True,
            'num_users': 100,
            'num_items': 200
        }
        self.mock_cb_model.get_model_info.return_value = {
            'model_type': 'content_based',
            'is_trained': True,
            'num_properties': 200
        }
        
        info = self.hybrid_system.get_system_info()
        
        expected_fields = [
            'system_type', 'cf_weight', 'cb_weight', 'min_cf_interactions',
            'fallback_to_content', 'explanation_detail_level', 'is_trained',
            'cf_model_info', 'cb_model_info'
        ]
        
        for field in expected_fields:
            assert field in info
        
        assert info['system_type'] == 'hybrid_recommendation_system'
        assert info['cf_weight'] == 0.6
        assert info['cb_weight'] == 0.4
    
    @pytest.mark.performance
    def test_prediction_performance_large_scale(self):
        """Test prediction performance with large datasets."""
        # Create large dataset
        large_data = self.ml_factory.create_training_data(
            num_users=200, num_properties=1000, density=0.05
        )
        
        # Setup mocks for large scale
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = large_data['user_item_matrix']
        self.hybrid_system.is_trained = True
        
        # Mock large predictions
        cf_predictions = np.random.uniform(0, 1, 500)
        cb_predictions = np.random.uniform(0, 1, 500)
        
        self.mock_cf_model.predict.return_value = cf_predictions
        self.mock_cb_model.predict.return_value = cb_predictions
        
        # Test performance
        with PerformanceTestHelpers.measure_time() as timer:
            predictions = self.hybrid_system.predict(user_id=0, item_ids=list(range(500)))
        
        elapsed_time = timer()
        PerformanceTestHelpers.assert_performance_threshold(
            elapsed_time, threshold=2.0, operation="Hybrid prediction (large scale)"
        )
        
        assert len(predictions) == 500
    
    @pytest.mark.performance
    def test_recommendation_performance_large_scale(self):
        """Test recommendation performance with large datasets."""
        # Setup for large scale recommendations
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = np.zeros((100, 1000))  # Large sparse matrix
        self.hybrid_system.is_trained = True
        
        # Mock large recommendation lists
        cf_recs = [
            RecommendationResult(i, np.random.uniform(0.5, 1.0), np.random.uniform(0.7, 1.0), f"CF rec {i}")
            for i in range(100)
        ]
        cb_recs = [
            RecommendationResult(i+50, np.random.uniform(0.3, 0.9), np.random.uniform(0.6, 0.9), f"CB rec {i}")
            for i in range(100)
        ]
        
        self.mock_cf_model.recommend.return_value = cf_recs
        self.mock_cb_model.recommend.return_value = cb_recs
        
        # Test performance
        with PerformanceTestHelpers.measure_time() as timer:
            recommendations = self.hybrid_system.recommend(user_id=0, num_recommendations=50)
        
        elapsed_time = timer()
        PerformanceTestHelpers.assert_performance_threshold(
            elapsed_time, threshold=3.0, operation="Hybrid recommendation (large scale)"
        )
        
        assert len(recommendations) <= 50
    
    def test_memory_usage_optimization(self):
        """Test that the system handles memory efficiently."""
        # This test would be more comprehensive with actual memory profiling
        # For now, we test that large operations complete without issues
        
        large_item_ids = list(range(10000))
        
        # Setup mocks
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = np.zeros((10, 10000))
        self.hybrid_system.is_trained = True
        
        # Should handle large item lists without memory issues
        self.mock_cf_model.predict.return_value = np.random.uniform(0, 1, 10000)
        self.mock_cb_model.predict.return_value = np.random.uniform(0, 1, 10000)
        
        predictions = self.hybrid_system.predict(user_id=0, item_ids=large_item_ids)
        assert len(predictions) == 10000
    
    def test_concurrent_prediction_safety(self):
        """Test thread safety for concurrent predictions."""
        import threading
        import queue
        
        # Setup
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = True
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        self.hybrid_system.is_trained = True
        
        # Mock consistent predictions
        self.mock_cf_model.predict.return_value = np.array([0.8, 0.6, 0.7])
        self.mock_cb_model.predict.return_value = np.array([0.5, 0.9, 0.4])
        
        results_queue = queue.Queue()
        
        def predict_worker():
            try:
                predictions = self.hybrid_system.predict(user_id=0, item_ids=[1, 2, 3])
                results_queue.put(predictions)
            except Exception as e:
                results_queue.put(e)
        
        # Run multiple concurrent predictions
        threads = [threading.Thread(target=predict_worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check all results are consistent
        results = []
        while not results_queue.empty():
            result = results_queue.get()
            assert not isinstance(result, Exception), f"Concurrent prediction failed: {result}"
            results.append(result)
        
        assert len(results) == 5
        
        # All results should be identical
        for result in results[1:]:
            np.testing.assert_array_almost_equal(results[0], result, decimal=5)
    
    def test_fallback_scenarios(self):
        """Test various fallback scenarios."""
        # CF model not available, CB available
        self.mock_cf_model.is_trained = False
        self.mock_cb_model.is_trained = True
        self.hybrid_system.is_trained = True
        self.hybrid_system.fallback_to_content = True
        
        cb_recs = [RecommendationResult(1, 0.8, 0.9, "CB only")]
        self.mock_cb_model.recommend.return_value = cb_recs
        
        recommendations = self.hybrid_system.recommend(user_id=0)
        assert len(recommendations) == 1
        assert recommendations[0].hybrid_method == "content_based_only"
        
        # CB model not available, CF available
        self.mock_cf_model.is_trained = True
        self.mock_cb_model.is_trained = False
        self.mock_cf_model.user_item_matrix = self.user_item_matrix
        
        cf_recs = [RecommendationResult(1, 0.8, 0.9, "CF only")]
        self.mock_cf_model.recommend.return_value = cf_recs
        
        recommendations = self.hybrid_system.recommend(user_id=0)
        assert len(recommendations) == 1
        assert recommendations[0].hybrid_method == "collaborative_filtering_only"
        
        # Neither model available
        self.mock_cf_model.is_trained = False
        self.mock_cb_model.is_trained = False
        
        recommendations = self.hybrid_system.recommend(user_id=0)
        assert len(recommendations) == 0
    
    def test_edge_case_user_behaviors(self):
        """Test edge cases in user behavior patterns."""
        # User with exactly threshold interactions
        threshold_matrix = np.zeros((5, 6))
        threshold_matrix[0, [0, 1, 2]] = 1  # Exactly 3 interactions
        
        self.mock_cf_model.is_trained = True
        self.mock_cf_model.user_item_matrix = threshold_matrix
        
        # Should use CF for user at threshold
        result = self.hybrid_system._should_use_collaborative_filtering(0)
        assert result is True
        
        # User with one less than threshold
        threshold_matrix[1, [0, 1]] = 1  # Only 2 interactions
        result = self.hybrid_system._should_use_collaborative_filtering(1)
        assert result is False
    
    def test_recommendation_diversity(self):
        """Test that recommendations maintain diversity."""
        # Setup overlapping but diverse recommendations
        cf_recs = [RecommendationResult(i, 0.8 - i*0.1, 0.9, f"CF {i}") for i in range(5)]
        cb_recs = [RecommendationResult(i+3, 0.7 - i*0.1, 0.8, f"CB {i}") for i in range(5)]
        
        combined = self.hybrid_system._combine_recommendations(cf_recs, cb_recs, num_recommendations=10)
        
        # Should have diverse item IDs
        item_ids = [rec.item_id for rec in combined]
        assert len(set(item_ids)) == len(item_ids)  # All unique
        
        # Should include items from both models
        cf_only_items = set(range(3))  # Items 0, 1, 2 only in CF
        cb_only_items = set(range(6, 8))  # Items 6, 7 only in CB
        
        combined_item_ids = set(item_ids)
        assert len(cf_only_items.intersection(combined_item_ids)) > 0
        assert len(cb_only_items.intersection(combined_item_ids)) > 0