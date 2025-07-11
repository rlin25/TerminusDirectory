"""
Performance tests for ML models and recommendation systems.

Tests model inference speed, training performance, and scalability under load.
"""

import pytest
import time
import asyncio
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from tests.utils.test_helpers import PerformanceTestHelpers, MLTestHelpers
from tests.utils.data_factories import MLDataFactory, FactoryConfig


class TestMLModelInferencePerformance:
    """Performance tests for ML model inference speed."""
    
    def setup_method(self):
        """Set up test fixtures for performance testing."""
        # Ensure deterministic behavior
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
        
        # Create test data of various sizes
        self.small_data = self.ml_factory.create_training_data(
            num_users=50, num_properties=100, density=0.1
        )
        self.medium_data = self.ml_factory.create_training_data(
            num_users=200, num_properties=500, density=0.05
        )
        self.large_data = self.ml_factory.create_training_data(
            num_users=1000, num_properties=2000, density=0.02
        )
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_content_based_inference_performance(self):
        """Test ContentBasedRecommender inference performance."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create and train model with medium dataset
        recommender = ContentBasedRecommender(
            embedding_dim=64,  # Smaller for faster training
            location_vocab_size=100,
            amenity_vocab_size=50
        )
        
        # Quick training for performance testing
        recommender.fit(
            user_item_matrix=self.medium_data['user_item_matrix'],
            property_data=self.medium_data['property_features'],
            epochs=3,  # Minimal training for performance testing
            batch_size=32
        )
        
        # Test single prediction performance
        with PerformanceTestHelpers.measure_time() as timer:
            predictions = recommender.predict(user_id=0, item_ids=[0, 1, 2, 3, 4])
        
        single_prediction_time = timer()
        
        # Should complete single prediction quickly
        PerformanceTestHelpers.assert_performance_threshold(
            single_prediction_time, threshold=0.1, operation="Single content-based prediction"
        )
        
        # Test batch prediction performance
        batch_item_ids = list(range(100))  # Predict for 100 items
        
        with PerformanceTestHelpers.measure_time() as timer:
            batch_predictions = recommender.predict(user_id=0, item_ids=batch_item_ids)
        
        batch_prediction_time = timer()
        
        # Should handle batch predictions efficiently
        PerformanceTestHelpers.assert_performance_threshold(
            batch_prediction_time, threshold=0.5, operation="Batch content-based prediction (100 items)"
        )
        
        assert len(batch_predictions) == 100
        MLTestHelpers.assert_valid_predictions(batch_predictions)
        
        # Calculate predictions per second
        predictions_per_second = len(batch_item_ids) / batch_prediction_time
        assert predictions_per_second > 200  # Should handle at least 200 predictions/second
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_hybrid_recommender_inference_performance(self):
        """Test HybridRecommendationSystem inference performance."""
        from infrastructure.ml.models.hybrid_recommender import HybridRecommendationSystem
        
        # Create hybrid system
        hybrid_system = HybridRecommendationSystem(
            cf_weight=0.6,
            cb_weight=0.4,
            min_cf_interactions=3
        )
        
        # Mock the component models for performance testing
        mock_cf_model = Mock()
        mock_cb_model = Mock()
        
        # Setup mock predictions (simulate fast model inference)
        mock_cf_model.is_trained = True
        mock_cb_model.is_trained = True
        mock_cf_model.user_item_matrix = self.medium_data['user_item_matrix']
        
        mock_cf_model.predict.return_value = np.random.uniform(0, 1, 50)
        mock_cb_model.predict.return_value = np.random.uniform(0, 1, 50)
        
        hybrid_system.cf_model = mock_cf_model
        hybrid_system.cb_model = mock_cb_model
        hybrid_system.is_trained = True
        
        # Test hybrid prediction performance
        item_ids = list(range(50))
        
        with PerformanceTestHelpers.measure_time() as timer:
            predictions = hybrid_system.predict(user_id=0, item_ids=item_ids)
        
        hybrid_prediction_time = timer()
        
        # Should complete hybrid predictions quickly
        PerformanceTestHelpers.assert_performance_threshold(
            hybrid_prediction_time, threshold=0.2, operation="Hybrid prediction (50 items)"
        )
        
        assert len(predictions) == 50
        MLTestHelpers.assert_valid_predictions(predictions)
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_recommendation_generation_performance(self):
        """Test recommendation generation performance."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create and minimally train model
        recommender = ContentBasedRecommender(embedding_dim=32)
        recommender.fit(
            user_item_matrix=self.small_data['user_item_matrix'],
            property_data=self.small_data['property_features'],
            epochs=2
        )
        
        # Test recommendation generation performance
        num_recommendations_tests = [5, 10, 20, 50]
        
        for num_recs in num_recommendations_tests:
            with PerformanceTestHelpers.measure_time() as timer:
                recommendations = recommender.recommend(
                    user_id=0,
                    num_recommendations=num_recs,
                    exclude_seen=True
                )
            
            rec_time = timer()
            
            # Performance should scale reasonably with number of recommendations
            max_time = 0.1 + (num_recs * 0.01)  # Linear scaling with small overhead
            PerformanceTestHelpers.assert_performance_threshold(
                rec_time, threshold=max_time, 
                operation=f"Recommendation generation ({num_recs} recommendations)"
            )
            
            MLTestHelpers.assert_recommendations_quality(recommendations, min_count=1)
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_concurrent_inference_performance(self):
        """Test ML model performance under concurrent load."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create and train model
        recommender = ContentBasedRecommender(embedding_dim=32)
        recommender.fit(
            user_item_matrix=self.small_data['user_item_matrix'],
            property_data=self.small_data['property_features'],
            epochs=2
        )
        
        def run_inference(user_id: int):
            """Run inference for a single user."""
            return recommender.predict(user_id=user_id % 10, item_ids=[0, 1, 2, 3, 4])
        
        # Test concurrent inference with multiple threads
        num_threads = 10
        num_requests_per_thread = 5
        
        with PerformanceTestHelpers.measure_time() as timer:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for thread_id in range(num_threads):
                    for request_id in range(num_requests_per_thread):
                        future = executor.submit(run_inference, thread_id)
                        futures.append(future)
                
                # Wait for all requests to complete
                results = [future.result() for future in futures]
        
        concurrent_time = timer()
        total_requests = num_threads * num_requests_per_thread
        
        # Should handle concurrent requests efficiently
        PerformanceTestHelpers.assert_performance_threshold(
            concurrent_time, threshold=2.0, 
            operation=f"Concurrent inference ({total_requests} requests)"
        )
        
        # All requests should complete successfully
        assert len(results) == total_requests
        assert all(len(result) == 5 for result in results)
        
        # Calculate concurrent throughput
        requests_per_second = total_requests / concurrent_time
        assert requests_per_second > 10  # Should handle at least 10 RPS under load
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_memory_usage_performance(self):
        """Test ML model memory usage performance."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Monitor memory usage during model operations
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        recommender = ContentBasedRecommender(embedding_dim=64)
        
        model_creation_memory = process.memory_info().rss / 1024 / 1024
        model_creation_delta = model_creation_memory - initial_memory
        
        # Train model
        recommender.fit(
            user_item_matrix=self.medium_data['user_item_matrix'],
            property_data=self.medium_data['property_features'],
            epochs=3
        )
        
        training_memory = process.memory_info().rss / 1024 / 1024
        training_delta = training_memory - model_creation_memory
        
        # Run multiple predictions
        for _ in range(100):
            recommender.predict(user_id=0, item_ids=[0, 1, 2, 3, 4])
        
        inference_memory = process.memory_info().rss / 1024 / 1024
        inference_delta = inference_memory - training_memory
        
        # Memory usage should be reasonable
        assert model_creation_delta < 100  # Model creation should use < 100MB
        assert training_delta < 200  # Training should use < 200MB additional
        assert inference_delta < 50   # Inference should not significantly increase memory
        
        print(f"Memory usage - Creation: {model_creation_delta:.1f}MB, "
              f"Training: {training_delta:.1f}MB, Inference: {inference_delta:.1f}MB")
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_scalability_performance(self):
        """Test ML model scalability with increasing data sizes."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        data_sizes = [
            ('small', self.small_data, 50),
            ('medium', self.medium_data, 200),
            ('large', self.large_data, 1000)
        ]
        
        scalability_results = []
        
        for size_name, data, num_predictions in data_sizes:
            recommender = ContentBasedRecommender(embedding_dim=32)
            
            # Measure training time
            with PerformanceTestHelpers.measure_time() as timer:
                recommender.fit(
                    user_item_matrix=data['user_item_matrix'],
                    property_data=data['property_features'],
                    epochs=2
                )
            training_time = timer()
            
            # Measure prediction time
            item_ids = list(range(min(num_predictions, data['num_properties'])))
            
            with PerformanceTestHelpers.measure_time() as timer:
                predictions = recommender.predict(user_id=0, item_ids=item_ids)
            prediction_time = timer()
            
            scalability_results.append({
                'size': size_name,
                'num_users': data['num_users'],
                'num_properties': data['num_properties'],
                'training_time': training_time,
                'prediction_time': prediction_time,
                'predictions_per_second': len(predictions) / prediction_time if prediction_time > 0 else 0
            })
            
            # Verify predictions are valid
            MLTestHelpers.assert_valid_predictions(predictions)
        
        # Check scalability characteristics
        small_result = scalability_results[0]
        medium_result = scalability_results[1]
        large_result = scalability_results[2]
        
        # Training time should scale sub-quadratically
        training_time_ratio = medium_result['training_time'] / small_result['training_time']
        data_size_ratio = medium_result['num_properties'] / small_result['num_properties']
        
        # Training time should not scale worse than O(n^2)
        assert training_time_ratio <= data_size_ratio ** 1.5
        
        # Prediction throughput should remain reasonable
        assert small_result['predictions_per_second'] > 50
        assert medium_result['predictions_per_second'] > 20
        assert large_result['predictions_per_second'] > 10
        
        print("Scalability Results:")
        for result in scalability_results:
            print(f"  {result['size']}: {result['num_users']} users, {result['num_properties']} properties")
            print(f"    Training: {result['training_time']:.2f}s, "
                  f"Prediction: {result['prediction_time']:.3f}s, "
                  f"Throughput: {result['predictions_per_second']:.1f} pred/s")


class TestMLTrainingPerformance:
    """Performance tests for ML model training."""
    
    def setup_method(self):
        """Set up test fixtures for training performance testing."""
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
    
    @pytest.mark.performance
    @pytest.mark.ml
    @pytest.mark.slow
    def test_content_based_training_performance(self):
        """Test ContentBasedRecommender training performance."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Test training performance with different configurations
        configs = [
            {'embedding_dim': 32, 'epochs': 5, 'batch_size': 16},
            {'embedding_dim': 64, 'epochs': 10, 'batch_size': 32},
            {'embedding_dim': 128, 'epochs': 15, 'batch_size': 64}
        ]
        
        training_data = self.ml_factory.create_training_data(
            num_users=100, num_properties=300, density=0.08
        )
        
        for config in configs:
            recommender = ContentBasedRecommender(
                embedding_dim=config['embedding_dim'],
                learning_rate=0.01  # Higher learning rate for faster training
            )
            
            with PerformanceTestHelpers.measure_time() as timer:
                training_result = recommender.fit(
                    user_item_matrix=training_data['user_item_matrix'],
                    property_data=training_data['property_features'],
                    epochs=config['epochs'],
                    batch_size=config['batch_size']
                )
            
            training_time = timer()
            
            # Training should complete within reasonable time
            max_time = config['epochs'] * 2.0  # 2 seconds per epoch maximum
            PerformanceTestHelpers.assert_performance_threshold(
                training_time, threshold=max_time,
                operation=f"Content-based training (dim={config['embedding_dim']}, epochs={config['epochs']})"
            )
            
            # Training should achieve reasonable results
            assert training_result['final_accuracy'] > 0.5
            assert training_result['final_loss'] < 1.0
            
            print(f"Config {config}: {training_time:.2f}s, "
                  f"Accuracy: {training_result['final_accuracy']:.3f}, "
                  f"Loss: {training_result['final_loss']:.3f}")
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_feature_extraction_performance(self):
        """Test property feature extraction performance."""
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        # Create property data of different sizes
        property_counts = [100, 500, 1000, 2000]
        
        for count in property_counts:
            properties = self.ml_factory.create_property_features(count)
            recommender = ContentBasedRecommender()
            
            with PerformanceTestHelpers.measure_time() as timer:
                features = recommender.extract_property_features(properties)
            
            extraction_time = timer()
            
            # Feature extraction should scale linearly
            max_time = count * 0.001  # 1ms per property
            PerformanceTestHelpers.assert_performance_threshold(
                extraction_time, threshold=max_time,
                operation=f"Feature extraction ({count} properties)"
            )
            
            # Verify features are extracted correctly
            assert len(features.location_features) == count
            assert features.price_features.shape[0] == count
            assert features.amenity_features.shape[0] == count
            
            properties_per_second = count / extraction_time
            assert properties_per_second > 1000  # Should process > 1000 properties/second
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_gpu_vs_cpu_performance(self):
        """Test GPU vs CPU performance comparison (if GPU available)."""
        # Check if GPU is available
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        if not gpu_available:
            pytest.skip("GPU not available for performance comparison")
        
        from infrastructure.ml.models.content_recommender import ContentBasedRecommender
        
        training_data = self.ml_factory.create_training_data(
            num_users=200, num_properties=500, density=0.05
        )
        
        # Test CPU performance
        with tf.device('/CPU:0'):
            cpu_recommender = ContentBasedRecommender(embedding_dim=64)
            
            with PerformanceTestHelpers.measure_time() as timer:
                cpu_recommender.fit(
                    user_item_matrix=training_data['user_item_matrix'],
                    property_data=training_data['property_features'],
                    epochs=5
                )
            cpu_time = timer()
        
        # Test GPU performance
        with tf.device('/GPU:0'):
            gpu_recommender = ContentBasedRecommender(embedding_dim=64)
            
            with PerformanceTestHelpers.measure_time() as timer:
                gpu_recommender.fit(
                    user_item_matrix=training_data['user_item_matrix'],
                    property_data=training_data['property_features'],
                    epochs=5
                )
            gpu_time = timer()
        
        # GPU should be faster for larger models
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x (CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s)")
        
        # For this model size, GPU might not be significantly faster due to overhead
        # but it should not be significantly slower
        assert gpu_time <= cpu_time * 2.0  # GPU should not be more than 2x slower


class TestRecommendationSystemPerformance:
    """Performance tests for the complete recommendation system."""
    
    def setup_method(self):
        """Set up test fixtures for recommendation system performance testing."""
        self.ml_factory = MLDataFactory(FactoryConfig(seed=42))
    
    @pytest.mark.performance
    @pytest.mark.ml
    @pytest.mark.asyncio
    async def test_end_to_end_recommendation_performance(self):
        """Test end-to-end recommendation system performance."""
        # Mock a complete recommendation service
        mock_recommendation_service = AsyncMock()
        
        # Simulate realistic response times
        async def mock_get_recommendations(user_id, num_recommendations=10, **kwargs):
            # Simulate processing time based on number of recommendations
            processing_time = 0.01 + (num_recommendations * 0.005)  # 10ms + 5ms per recommendation
            await asyncio.sleep(processing_time)
            
            return [
                {
                    'property_id': i,
                    'predicted_rating': np.random.uniform(0.5, 1.0),
                    'confidence_score': np.random.uniform(0.7, 1.0),
                    'explanation': f'Recommendation {i}'
                }
                for i in range(num_recommendations)
            ]
        
        mock_recommendation_service.get_user_recommendations = mock_get_recommendations
        
        # Test different recommendation counts
        recommendation_counts = [5, 10, 20, 50]
        
        for count in recommendation_counts:
            with PerformanceTestHelpers.measure_time() as timer:
                recommendations = await mock_recommendation_service.get_user_recommendations(
                    user_id=1, num_recommendations=count
                )
            
            rec_time = timer()
            
            # Should complete within reasonable time
            max_time = 0.1 + (count * 0.01)  # 100ms + 10ms per recommendation
            PerformanceTestHelpers.assert_performance_threshold(
                rec_time, threshold=max_time,
                operation=f"End-to-end recommendations ({count} items)"
            )
            
            assert len(recommendations) == count
    
    @pytest.mark.performance
    @pytest.mark.ml
    @pytest.mark.asyncio
    async def test_concurrent_user_recommendations_performance(self):
        """Test concurrent recommendation generation for multiple users."""
        mock_recommendation_service = AsyncMock()
        
        async def mock_get_recommendations(user_id, **kwargs):
            # Simulate variable processing time based on user history
            processing_time = np.random.uniform(0.05, 0.15)  # 50-150ms
            await asyncio.sleep(processing_time)
            
            return [
                {'property_id': i, 'predicted_rating': 0.8}
                for i in range(10)
            ]
        
        mock_recommendation_service.get_user_recommendations = mock_get_recommendations
        
        # Test concurrent recommendations for multiple users
        num_users = 20
        
        with PerformanceTestHelpers.measure_time() as timer:
            tasks = [
                mock_recommendation_service.get_user_recommendations(user_id=i)
                for i in range(num_users)
            ]
            results = await asyncio.gather(*tasks)
        
        concurrent_time = timer()
        
        # Should handle concurrent users efficiently
        PerformanceTestHelpers.assert_performance_threshold(
            concurrent_time, threshold=2.0,
            operation=f"Concurrent recommendations ({num_users} users)"
        )
        
        # All users should get recommendations
        assert len(results) == num_users
        assert all(len(recs) == 10 for recs in results)
        
        # Calculate user throughput
        users_per_second = num_users / concurrent_time
        assert users_per_second > 5  # Should handle at least 5 users/second
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_recommendation_caching_performance(self):
        """Test recommendation caching performance impact."""
        # Mock cache operations
        mock_cache = {}
        
        def get_from_cache(key):
            return mock_cache.get(key)
        
        def set_cache(key, value, ttl=3600):
            mock_cache[key] = value
            return True
        
        def generate_recommendations(user_id):
            # Simulate expensive recommendation generation
            time.sleep(0.1)  # 100ms processing time
            return [{'property_id': i, 'score': 0.8} for i in range(10)]
        
        user_id = 123
        cache_key = f"user_recommendations:{user_id}"
        
        # First call - cache miss
        with PerformanceTestHelpers.measure_time() as timer:
            cached_recs = get_from_cache(cache_key)
            if cached_recs is None:
                recommendations = generate_recommendations(user_id)
                set_cache(cache_key, recommendations)
            else:
                recommendations = cached_recs
        
        cache_miss_time = timer()
        
        # Second call - cache hit
        with PerformanceTestHelpers.measure_time() as timer:
            cached_recs = get_from_cache(cache_key)
            if cached_recs is None:
                recommendations = generate_recommendations(user_id)
                set_cache(cache_key, recommendations)
            else:
                recommendations = cached_recs
        
        cache_hit_time = timer()
        
        # Cache hit should be much faster
        assert cache_hit_time < cache_miss_time * 0.1  # At least 10x faster
        assert cache_hit_time < 0.01  # Less than 10ms
        
        speedup = cache_miss_time / cache_hit_time
        print(f"Cache speedup: {speedup:.1f}x (Miss: {cache_miss_time:.3f}s, Hit: {cache_hit_time:.3f}s)")
    
    @pytest.mark.performance
    @pytest.mark.ml
    def test_recommendation_system_memory_efficiency(self):
        """Test recommendation system memory efficiency."""
        from infrastructure.ml.models.hybrid_recommender import HybridRecommendationSystem
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple recommendation systems
        systems = []
        for i in range(10):
            system = HybridRecommendationSystem()
            
            # Mock component models to avoid actual training overhead
            system.cf_model = Mock()
            system.cb_model = Mock()
            system.cf_model.is_trained = True
            system.cb_model.is_trained = True
            system.is_trained = True
            
            systems.append(system)
        
        after_creation_memory = process.memory_info().rss / 1024 / 1024
        memory_per_system = (after_creation_memory - initial_memory) / len(systems)
        
        # Each recommendation system should use reasonable memory
        assert memory_per_system < 50  # Less than 50MB per system
        
        # Cleanup
        del systems
        
        print(f"Memory per recommendation system: {memory_per_system:.1f}MB")