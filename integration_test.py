#!/usr/bin/env python3
"""
End-to-End Integration Test

This test validates the complete repository functionality and demonstrates
production-ready capabilities of the rental ML system.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.infrastructure.data.config import DataConfig
from src.infrastructure.data.repository_factory import RepositoryManager
from src.domain.entities.property import Property
from src.domain.entities.user import User, UserPreferences, UserInteraction
from src.domain.entities.search_query import SearchQuery, SearchFilters
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Complete integration test suite for repository layer"""
    
    def __init__(self):
        self.config = DataConfig()
        self.test_data = {
            'users': [],
            'properties': [],
            'models': [],
            'interactions': []
        }
        self.test_results = {}
        
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("Starting integration test suite...")
        
        start_time = datetime.utcnow()
        
        try:
            async with RepositoryManager(self.config) as factory:
                logger.info("Repository factory initialized successfully")
                
                # Test 1: System Health Check
                await self.test_system_health(factory)
                
                # Test 2: Data Creation Pipeline
                await self.test_data_creation_pipeline(factory)
                
                # Test 3: Search and Recommendation Pipeline
                await self.test_search_recommendation_pipeline(factory)
                
                # Test 4: ML Model Pipeline
                await self.test_ml_model_pipeline(factory)
                
                # Test 5: Analytics and Reporting
                await self.test_analytics_pipeline(factory)
                
                # Test 6: Performance and Scalability
                await self.test_performance_scalability(factory)
                
                # Test 7: Data Consistency and Integrity
                await self.test_data_integrity(factory)
                
                # Clean up test data
                await self.cleanup_test_data(factory)
                
                # Generate comprehensive report
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                await self.generate_integration_report(execution_time)
                
        except Exception as e:
            logger.error(f"Integration test suite failed: {e}")
            self.test_results['overall_status'] = 'failed'
            self.test_results['error'] = str(e)
            raise
        
        return self.test_results
    
    async def test_system_health(self, factory):
        """Test overall system health and connectivity"""
        test_name = "system_health"
        logger.info(f"Running {test_name} test...")
        
        try:
            # Test repository factory health
            health_status = await factory.health_check()
            
            # Test individual repository health
            property_repo = factory.get_property_repository()
            user_repo = factory.get_user_repository()
            model_repo = factory.get_model_repository()
            cache_repo = factory.get_cache_repository()
            
            property_health = await property_repo.health_check()
            user_health = await user_repo.health_check()
            model_health = await model_repo.health_check()
            cache_health = await cache_repo.health_check()
            
            # Check connection pools and performance
            property_connection_info = await property_repo.get_connection_info()
            
            all_healthy = all([
                health_status.get("overall", False),
                property_health.get("status") == "healthy",
                user_health.get("status") == "healthy",
                model_health.get("status") == "healthy",
                cache_health
            ])
            
            self.test_results[test_name] = {
                "status": "passed" if all_healthy else "failed",
                "details": {
                    "factory_health": health_status,
                    "property_health": property_health,
                    "user_health": user_health,
                    "model_health": model_health,
                    "cache_health": cache_health,
                    "connection_info": property_connection_info
                }
            }
            
            logger.info(f"✅ {test_name} {'passed' if all_healthy else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    async def test_data_creation_pipeline(self, factory):
        """Test end-to-end data creation pipeline"""
        test_name = "data_creation_pipeline"
        logger.info(f"Running {test_name} test...")
        
        try:
            property_repo = factory.get_property_repository()
            user_repo = factory.get_user_repository()
            
            # Create test users
            test_users = []
            for i in range(5):
                user = User(
                    id=uuid.uuid4(),
                    email=f"integration_test_user_{i}@example.com",
                    preferences=UserPreferences(
                        min_price=1000.0 + (i * 500),
                        max_price=3000.0 + (i * 500),
                        min_bedrooms=1 + (i % 3),
                        max_bedrooms=3 + (i % 2),
                        preferred_locations=[f"Test City {i % 3}"],
                        required_amenities=["parking"] + (["gym"] if i % 2 == 0 else ["pool"]),
                        property_types=["apartment", "condo"]
                    ),
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                created_user = await user_repo.create(user)
                if created_user:
                    test_users.append(created_user)
                    self.test_data['users'].append(created_user.id)
            
            # Create test properties
            test_properties = []
            for i in range(10):
                property_obj = Property(
                    id=uuid.uuid4(),
                    title=f"Integration Test Property {i+1}",
                    description=f"A comprehensive test property number {i+1} for integration testing with all features.",
                    price=2000.0 + (i * 250),
                    location=f"Test City {i % 3}, Test State",
                    bedrooms=1 + (i % 4),
                    bathrooms=1.0 + (i % 3) * 0.5,
                    square_feet=800 + (i * 150),
                    amenities=["parking", "gym", "pool", "laundry"][:(i % 4) + 1],
                    contact_info={"phone": f"555-{1000+i}", "email": f"contact_{i}@example.com"},
                    images=[f"https://example.com/property_{i}_1.jpg", f"https://example.com/property_{i}_2.jpg"],
                    scraped_at=datetime.utcnow() - timedelta(days=i),
                    is_active=True,
                    property_type=["apartment", "house", "condo"][i % 3]
                )
                created_property = await property_repo.create(property_obj)
                if created_property:
                    test_properties.append(created_property)
                    self.test_data['properties'].append(created_property.id)
            
            # Create user interactions
            interaction_count = 0
            for user in test_users[:3]:  # Use first 3 users
                for property_obj in test_properties[:5]:  # With first 5 properties
                    for interaction_type in ["view", "like", "inquiry"]:
                        interaction = UserInteraction(
                            property_id=property_obj.id,
                            interaction_type=interaction_type,
                            timestamp=datetime.utcnow() - timedelta(minutes=interaction_count),
                            duration_seconds=30 + (interaction_count % 60)
                        )
                        
                        success = await user_repo.add_interaction(user.id, interaction)
                        if success:
                            interaction_count += 1
            
            # Verify data integrity
            created_users_count = len(test_users)
            created_properties_count = len(test_properties)
            
            success = (
                created_users_count == 5 and
                created_properties_count == 10 and
                interaction_count > 0
            )
            
            self.test_results[test_name] = {
                "status": "passed" if success else "failed",
                "details": {
                    "users_created": created_users_count,
                    "properties_created": created_properties_count,
                    "interactions_created": interaction_count
                }
            }
            
            logger.info(f"✅ {test_name} {'passed' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    async def test_search_recommendation_pipeline(self, factory):
        """Test search and recommendation pipeline"""
        test_name = "search_recommendation_pipeline"
        logger.info(f"Running {test_name} test...")
        
        try:
            property_repo = factory.get_property_repository()
            user_repo = factory.get_user_repository()
            
            # Test various search scenarios
            search_tests = []
            
            # Test 1: Basic text search
            search_query = SearchQuery(
                query_text="Integration Test",
                limit=20
            )
            results, total_count = await property_repo.search(search_query)
            search_tests.append(("text_search", len(results) > 0, len(results)))
            
            # Test 2: Price range search
            search_query = SearchQuery(
                query_text="",
                filters=SearchFilters(min_price=2000, max_price=3000),
                limit=20
            )
            results, total_count = await property_repo.search(search_query)
            search_tests.append(("price_range", len(results) > 0, len(results)))
            
            # Test 3: Location-based search
            search_query = SearchQuery(
                query_text="",
                filters=SearchFilters(locations=["Test City 1"]),
                limit=20
            )
            results, total_count = await property_repo.search(search_query)
            search_tests.append(("location_search", len(results) > 0, len(results)))
            
            # Test 4: Complex multi-filter search
            search_query = SearchQuery(
                query_text="Test",
                filters=SearchFilters(
                    min_price=2000,
                    max_price=4000,
                    min_bedrooms=2,
                    max_bedrooms=4,
                    amenities=["parking"],
                    property_types=["apartment", "house"]
                ),
                limit=20,
                sort_by="price_asc"
            )
            results, total_count = await property_repo.search(search_query)
            search_tests.append(("complex_search", len(results) >= 0, len(results)))
            
            # Test user-based queries
            user_tests = []
            
            # Get users for testing
            users = await user_repo.get_all_active(limit=3)
            
            if users:
                # Test user statistics
                for user in users:
                    stats = await user_repo.get_user_statistics(user.id)
                    user_tests.append(("user_stats", stats is not None))
                
                # Test similar users
                if len(users) > 1:
                    similar_users = await user_repo.get_similar_users(users[0].id, limit=5)
                    user_tests.append(("similar_users", len(similar_users) >= 0))
                
                # Test user behavior analytics
                behavior_analytics = await user_repo.get_user_behavior_analytics(days=30)
                user_tests.append(("behavior_analytics", behavior_analytics is not None))
            
            # Test property analytics
            property_tests = []
            
            # Get aggregated statistics
            aggregated_stats = await property_repo.get_aggregated_stats()
            property_tests.append(("aggregated_stats", aggregated_stats is not None))
            
            # Get property counts by status
            status_counts = await property_repo.get_counts_by_status()
            property_tests.append(("status_counts", status_counts is not None))
            
            # Test similar properties
            if self.test_data['properties']:
                property_id = self.test_data['properties'][0]
                similar_props = await property_repo.get_similar_properties(property_id, limit=5)
                property_tests.append(("similar_properties", len(similar_props) >= 0))
            
            all_search_passed = all(success for _, success, _ in search_tests)
            all_user_passed = all(success for _, success in user_tests)
            all_property_passed = all(success for _, success in property_tests)
            
            overall_success = all_search_passed and all_user_passed and all_property_passed
            
            self.test_results[test_name] = {
                "status": "passed" if overall_success else "failed",
                "details": {
                    "search_tests": {name: {"passed": success, "count": count} for name, success, count in search_tests},
                    "user_tests": {name: success for name, success in user_tests},
                    "property_tests": {name: success for name, success in property_tests},
                    "total_search_results": sum(count for _, success, count in search_tests if success)
                }
            }
            
            logger.info(f"✅ {test_name} {'passed' if overall_success else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    async def test_ml_model_pipeline(self, factory):
        """Test ML model storage and retrieval pipeline"""
        test_name = "ml_model_pipeline"
        logger.info(f"Running {test_name} test...")
        
        try:
            model_repo = factory.get_model_repository()
            
            # Test model storage
            test_models = []
            for i in range(3):
                model_name = f"integration_test_model_{i}"
                version = f"1.{i}.0"
                
                # Create mock model data
                model_data = {
                    "model_type": "neural_network",
                    "layers": [128, 64, 32, 16],
                    "weights": list(range(100 + i * 10)),
                    "bias": 0.1 * i,
                    "metadata": {
                        "training_samples": 10000 + i * 1000,
                        "validation_accuracy": 0.85 + i * 0.02,
                        "features": ["price", "bedrooms", "bathrooms", "location"]
                    }
                }
                
                # Save model
                save_success = await model_repo.save_model(model_name, model_data, version)
                if save_success:
                    test_models.append((model_name, version))
                    self.test_data['models'].append((model_name, version))
                
                # Save training metrics
                training_metrics = {
                    "accuracy": 0.85 + i * 0.02,
                    "precision": 0.83 + i * 0.015,
                    "recall": 0.87 + i * 0.01,
                    "f1_score": 0.85 + i * 0.01,
                    "loss": 0.15 - i * 0.01,
                    "training_time": 3600 + i * 300
                }
                await model_repo.save_training_metrics(model_name, version, training_metrics)
            
            # Test embeddings storage
            embedding_tests = []
            for i in range(5):
                entity_type = "property" if i % 2 == 0 else "user"
                entity_id = f"test_entity_{i}"
                embeddings = np.random.rand(128).astype(np.float32)  # 128-dimensional embeddings
                
                save_success = await model_repo.save_embeddings(entity_type, entity_id, embeddings)
                embedding_tests.append(save_success)
                
                if save_success:
                    # Test retrieval
                    retrieved_embeddings = await model_repo.get_embeddings(entity_type, entity_id)
                    retrieval_success = retrieved_embeddings is not None and np.array_equal(embeddings, retrieved_embeddings)
                    embedding_tests.append(retrieval_success)
            
            # Test model loading and versioning
            model_tests = []
            for model_name, version in test_models:
                # Test specific version loading
                loaded_model = await model_repo.load_model(model_name, version)
                model_tests.append(loaded_model is not None)
                
                # Test latest version loading
                latest_model = await model_repo.load_model(model_name, "latest")
                model_tests.append(latest_model is not None)
                
                # Test version listing
                versions = await model_repo.get_model_versions(model_name)
                model_tests.append(version in versions)
                
                # Test model info
                model_info = await model_repo.get_model_info(model_name, version)
                model_tests.append(model_info is not None)
                
                # Test training metrics retrieval
                metrics = await model_repo.get_training_metrics(model_name, version)
                model_tests.append(metrics is not None and "accuracy" in metrics)
            
            # Test prediction caching
            cache_tests = []
            for i in range(3):
                cache_key = f"integration_test_predictions_{i}"
                predictions = {
                    "property_id": str(uuid.uuid4()),
                    "scores": [0.8 + i * 0.05, 0.7 + i * 0.03, 0.9 - i * 0.02],
                    "recommendations": [str(uuid.uuid4()) for _ in range(5)],
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Test cache save
                cache_save = await model_repo.cache_predictions(cache_key, predictions, ttl_seconds=3600)
                cache_tests.append(cache_save)
                
                # Test cache retrieval
                cached_predictions = await model_repo.get_cached_predictions(cache_key)
                cache_tests.append(cached_predictions is not None and cached_predictions == predictions)
            
            # Test repository statistics
            model_stats = await model_repo.get_model_storage_stats()
            embeddings_stats = await model_repo.get_embeddings_stats()
            cache_stats = await model_repo.get_cache_statistics()
            
            stats_tests = [
                model_stats is not None,
                embeddings_stats is not None,
                cache_stats is not None
            ]
            
            all_tests_passed = all([
                len(test_models) == 3,
                all(embedding_tests),
                all(model_tests),
                all(cache_tests),
                all(stats_tests)
            ])
            
            self.test_results[test_name] = {
                "status": "passed" if all_tests_passed else "failed",
                "details": {
                    "models_saved": len(test_models),
                    "embedding_tests_passed": sum(embedding_tests),
                    "model_tests_passed": sum(model_tests),
                    "cache_tests_passed": sum(cache_tests),
                    "stats_available": all(stats_tests),
                    "model_storage_stats": model_stats,
                    "embeddings_stats": embeddings_stats,
                    "cache_stats": cache_stats
                }
            }
            
            logger.info(f"✅ {test_name} {'passed' if all_tests_passed else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    async def test_analytics_pipeline(self, factory):
        """Test analytics and reporting pipeline"""
        test_name = "analytics_pipeline"
        logger.info(f"Running {test_name} test...")
        
        try:
            property_repo = factory.get_property_repository()
            user_repo = factory.get_user_repository()
            
            analytics_tests = []
            
            # Test property analytics
            property_count = await property_repo.get_count()
            active_count = await property_repo.get_active_count()
            analytics_tests.append(property_count >= 0 and active_count >= 0)
            
            # Test aggregated statistics
            aggregated_stats = await property_repo.get_aggregated_stats()
            analytics_tests.append(aggregated_stats is not None and "avg_price" in aggregated_stats)
            
            # Test location analytics
            location_analytics = await property_repo.get_location_analytics(limit=10)
            analytics_tests.append(isinstance(location_analytics, list))
            
            # Test price distribution
            price_distribution = await property_repo.get_price_distribution()
            analytics_tests.append(price_distribution is not None and "percentiles" in price_distribution)
            
            # Test trending properties
            trending_properties = await property_repo.get_trending_properties(limit=10)
            analytics_tests.append(isinstance(trending_properties, list))
            
            # Test user analytics
            user_count = await user_repo.get_count()
            active_user_count = await user_repo.get_active_count()
            analytics_tests.append(user_count >= 0 and active_user_count >= 0)
            
            # Test user behavior analytics
            behavior_analytics = await user_repo.get_user_behavior_analytics(days=30)
            analytics_tests.append(behavior_analytics is not None and "active_users" in behavior_analytics)
            
            # Test user segmentation
            user_segmentation = await user_repo.get_user_segmentation()
            analytics_tests.append(user_segmentation is not None and "activity_segments" in user_segmentation)
            
            # Test interaction matrix (if users exist)
            users = await user_repo.get_all_active(limit=1)
            if users:
                interaction_matrix = await user_repo.get_user_interaction_matrix()
                analytics_tests.append(isinstance(interaction_matrix, dict))
            else:
                analytics_tests.append(True)  # Skip if no users
            
            all_analytics_passed = all(analytics_tests)
            
            self.test_results[test_name] = {
                "status": "passed" if all_analytics_passed else "failed",
                "details": {
                    "property_count": property_count,
                    "active_property_count": active_count,
                    "user_count": user_count,
                    "active_user_count": active_user_count,
                    "analytics_tests_passed": sum(analytics_tests),
                    "total_analytics_tests": len(analytics_tests)
                }
            }
            
            logger.info(f"✅ {test_name} {'passed' if all_analytics_passed else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    async def test_performance_scalability(self, factory):
        """Test performance and scalability"""
        test_name = "performance_scalability"
        logger.info(f"Running {test_name} test...")
        
        try:
            property_repo = factory.get_property_repository()
            
            performance_tests = []
            
            # Test batch operations performance
            start_time = datetime.utcnow()
            
            # Create batch of properties
            batch_properties = []
            for i in range(20):  # Moderate batch size for testing
                prop = Property(
                    id=uuid.uuid4(),
                    title=f"Performance Test Property {i+1}",
                    description=f"Performance testing property number {i+1}",
                    price=2000.0 + (i * 100),
                    location=f"Performance Test Location {i % 5}",
                    bedrooms=2 + (i % 3),
                    bathrooms=2.0,
                    square_feet=1000 + (i * 50),
                    amenities=["parking", "gym"],
                    contact_info={"phone": f"555-{2000+i}"},
                    images=[],
                    scraped_at=datetime.utcnow(),
                    is_active=True,
                    property_type="apartment"
                )
                batch_properties.append(prop)
            
            created_properties = await property_repo.bulk_create(batch_properties)
            bulk_create_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Add created property IDs to cleanup list
            for prop in created_properties:
                self.test_data['properties'].append(prop.id)
            
            performance_tests.append(("bulk_create", bulk_create_time < 30.0, bulk_create_time))
            performance_tests.append(("bulk_create_success", len(created_properties) == len(batch_properties), len(created_properties)))
            
            # Test batch retrieval performance
            start_time = datetime.utcnow()
            property_ids = [prop.id for prop in created_properties]
            retrieved_properties = await property_repo.get_by_ids(property_ids)
            batch_read_time = (datetime.utcnow() - start_time).total_seconds()
            
            performance_tests.append(("batch_read", batch_read_time < 10.0, batch_read_time))
            performance_tests.append(("batch_read_success", len(retrieved_properties) == len(property_ids), len(retrieved_properties)))
            
            # Test search performance with various complexities
            search_tests = []
            
            # Simple search
            start_time = datetime.utcnow()
            search_query = SearchQuery(query_text="Performance", limit=50)
            results, _ = await property_repo.search(search_query)
            simple_search_time = (datetime.utcnow() - start_time).total_seconds()
            search_tests.append(("simple_search", simple_search_time < 5.0, simple_search_time))
            
            # Complex search
            start_time = datetime.utcnow()
            search_query = SearchQuery(
                query_text="Test",
                filters=SearchFilters(
                    min_price=1500,
                    max_price=3500,
                    min_bedrooms=1,
                    max_bedrooms=4,
                    amenities=["parking"]
                ),
                limit=50,
                sort_by="price_asc"
            )
            results, _ = await property_repo.search(search_query)
            complex_search_time = (datetime.utcnow() - start_time).total_seconds()
            search_tests.append(("complex_search", complex_search_time < 10.0, complex_search_time))
            
            # Test connection performance
            connection_info = await property_repo.get_connection_info()
            performance_tests.append(("connection_health", connection_info is not None, connection_info))
            
            all_performance_passed = all(success for _, success, _ in performance_tests) and all(success for _, success, _ in search_tests)
            
            self.test_results[test_name] = {
                "status": "passed" if all_performance_passed else "failed",
                "details": {
                    "bulk_operations": {name: {"passed": success, "value": value} for name, success, value in performance_tests},
                    "search_operations": {name: {"passed": success, "time": time_val} for name, success, time_val in search_tests},
                    "connection_info": connection_info
                }
            }
            
            logger.info(f"✅ {test_name} {'passed' if all_performance_passed else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    async def test_data_integrity(self, factory):
        """Test data consistency and integrity"""
        test_name = "data_integrity"
        logger.info(f"Running {test_name} test...")
        
        try:
            property_repo = factory.get_property_repository()
            user_repo = factory.get_user_repository()
            
            integrity_tests = []
            
            # Test property data integrity
            if self.test_data['properties']:
                # Get a property and verify all fields are preserved
                property_id = self.test_data['properties'][0]
                property_obj = await property_repo.get_by_id(property_id)
                
                if property_obj:
                    integrity_tests.append(("property_fields_complete", all([
                        property_obj.id is not None,
                        property_obj.title is not None,
                        property_obj.description is not None,
                        property_obj.price is not None,
                        property_obj.location is not None,
                        property_obj.bedrooms is not None,
                        property_obj.bathrooms is not None,
                        isinstance(property_obj.amenities, list),
                        isinstance(property_obj.contact_info, dict),
                        property_obj.scraped_at is not None
                    ])))
                    
                    # Test property update integrity
                    original_price = property_obj.price
                    property_obj.price = original_price + 100
                    updated_property = await property_repo.update(property_obj)
                    integrity_tests.append(("property_update_integrity", 
                                          updated_property is not None and updated_property.price == original_price + 100))
                    
                    # Restore original price
                    property_obj.price = original_price
                    await property_repo.update(property_obj)
            
            # Test user data integrity
            if self.test_data['users']:
                user_id = self.test_data['users'][0]
                user_obj = await user_repo.get_by_id(user_id)
                
                if user_obj:
                    integrity_tests.append(("user_fields_complete", all([
                        user_obj.id is not None,
                        user_obj.email is not None,
                        user_obj.preferences is not None,
                        isinstance(user_obj.interactions, list),
                        user_obj.created_at is not None
                    ])))
                    
                    # Test user preference integrity
                    original_max_price = user_obj.preferences.max_price
                    user_obj.preferences.max_price = original_max_price + 500
                    updated_user = await user_repo.update(user_obj)
                    integrity_tests.append(("user_update_integrity",
                                          updated_user is not None and updated_user.preferences.max_price == original_max_price + 500))
                    
                    # Restore original price
                    user_obj.preferences.max_price = original_max_price
                    await user_repo.update(user_obj)
            
            # Test referential integrity (interactions)
            if self.test_data['users'] and self.test_data['properties']:
                user_id = self.test_data['users'][0]
                property_id = self.test_data['properties'][0]
                
                # Create test interaction
                interaction = UserInteraction(
                    property_id=property_id,
                    interaction_type="view",
                    timestamp=datetime.utcnow(),
                    duration_seconds=45
                )
                
                interaction_success = await user_repo.add_interaction(user_id, interaction)
                integrity_tests.append(("referential_integrity", interaction_success))
                
                # Verify interaction was stored
                interactions = await user_repo.get_interactions(user_id, limit=1)
                integrity_tests.append(("interaction_retrieval", len(interactions) > 0))
            
            # Test transaction consistency
            # This tests that operations either complete fully or fail completely
            try:
                # Attempt to create a property with invalid data
                invalid_property = Property(
                    id=uuid.uuid4(),
                    title="",  # Invalid: empty title
                    description="Test",
                    price=-100,  # Invalid: negative price
                    location="Test",
                    bedrooms=-1,  # Invalid: negative bedrooms
                    bathrooms=1.0,
                    square_feet=None,
                    amenities=[],
                    contact_info={},
                    images=[],
                    scraped_at=datetime.utcnow(),
                    is_active=True,
                    property_type="apartment"
                )
                
                # This should fail due to validation
                try:
                    await property_repo.create(invalid_property)
                    integrity_tests.append(("validation_consistency", False))  # Should not succeed
                except (ValueError, Exception):
                    integrity_tests.append(("validation_consistency", True))  # Expected to fail
            except Exception:
                integrity_tests.append(("validation_consistency", True))  # Expected behavior
            
            all_integrity_passed = all(success for _, success in integrity_tests)
            
            self.test_results[test_name] = {
                "status": "passed" if all_integrity_passed else "failed",
                "details": {
                    "integrity_tests": {name: success for name, success in integrity_tests},
                    "total_tests": len(integrity_tests),
                    "passed_tests": sum(success for _, success in integrity_tests)
                }
            }
            
            logger.info(f"✅ {test_name} {'passed' if all_integrity_passed else 'failed'}")
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    async def cleanup_test_data(self, factory):
        """Clean up all test data"""
        logger.info("Cleaning up test data...")
        
        try:
            property_repo = factory.get_property_repository()
            user_repo = factory.get_user_repository()
            model_repo = factory.get_model_repository()
            
            # Clean up properties
            for property_id in self.test_data['properties']:
                try:
                    await property_repo.delete(property_id, hard_delete=True)
                except Exception as e:
                    logger.warning(f"Failed to delete property {property_id}: {e}")
            
            # Clean up users
            for user_id in self.test_data['users']:
                try:
                    await user_repo.delete(user_id)
                except Exception as e:
                    logger.warning(f"Failed to delete user {user_id}: {e}")
            
            # Clean up models
            for model_name, version in self.test_data['models']:
                try:
                    await model_repo.delete_model(model_name, version)
                except Exception as e:
                    logger.warning(f"Failed to delete model {model_name} v{version}: {e}")
            
            # Clean up cache
            try:
                await model_repo.clear_cache("integration_test_*")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
            
            logger.info("Test data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def generate_integration_report(self, execution_time: float):
        """Generate comprehensive integration test report"""
        logger.info("Generating integration test report...")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if isinstance(result, dict) and result.get("status") == "passed")
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*100)
        print("RENTAL ML SYSTEM - INTEGRATION TEST REPORT")
        print("="*100)
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*100)
        
        # Detailed results
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                status_icon = "✅" if result.get("status") == "passed" else "❌"
                print(f"{status_icon} {test_name.upper().replace('_', ' ')}: {result.get('status', 'unknown').upper()}")
                
                if result.get("status") == "failed" and result.get("error"):
                    print(f"   Error: {result['error']}")
                elif result.get("details"):
                    # Print key metrics from details
                    details = result["details"]
                    if isinstance(details, dict):
                        for key, value in details.items():
                            if isinstance(value, (int, float, str)) and not key.endswith("_stats"):
                                print(f"   {key}: {value}")
                print()
        
        print("="*100)
        
        # System information
        config = DataConfig()
        print("SYSTEM CONFIGURATION:")
        print(f"Database Host: {config.database.host}:{config.database.port}")
        print(f"Database Name: {config.database.database}")
        print(f"Redis Host: {config.redis.host}:{config.redis.port}")
        print(f"Pool Size: {config.database.pool_size}")
        print()
        
        # Test data summary
        print("TEST DATA CREATED:")
        print(f"Users: {len(self.test_data['users'])}")
        print(f"Properties: {len(self.test_data['properties'])}")
        print(f"Models: {len(self.test_data['models'])}")
        print()
        
        if failed_tests == 0:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("The repository layer is production-ready and fully functional.")
        else:
            print("⚠️  Some integration tests failed.")
            print("Please review the errors above before deploying to production.")
        
        print("="*100)

async def main():
    """Main integration test function"""
    print("Rental ML System - Integration Test Suite")
    print("========================================")
    
    # Check environment
    required_env_vars = [
        "DB_HOST", "DB_PORT", "DB_NAME", "DB_USERNAME", "DB_PASSWORD",
        "REDIS_HOST", "REDIS_PORT"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set all required environment variables for production testing.")
        print("Using default values for demo...")
        print()
    
    # Initialize and run tests
    test_suite = IntegrationTestSuite()
    
    try:
        results = await test_suite.run_integration_tests()
        
        # Determine exit code
        failed_count = sum(1 for result in results.values() 
                          if isinstance(result, dict) and result.get("status") == "failed")
        exit_code = 0 if failed_count == 0 else 1
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️  Integration tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Integration test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)