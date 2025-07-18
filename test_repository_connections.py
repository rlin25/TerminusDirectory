#!/usr/bin/env python3
"""
Repository Connection Test Script

This script tests the repository layer with real database connections
to ensure production readiness and validate CRUD operations.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Optional

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

class RepositoryTester:
    """Comprehensive repository testing suite"""
    
    def __init__(self):
        self.config = DataConfig()
        self.test_results = {}
        
    async def run_all_tests(self) -> dict:
        """Run all repository tests"""
        logger.info("Starting repository connection tests...")
        
        # Test database connection
        await self.test_database_connection()
        
        # Test Redis connection
        await self.test_redis_connection()
        
        # Test repository initialization
        await self.test_repository_initialization()
        
        # Test CRUD operations
        await self.test_property_crud()
        await self.test_user_crud()
        await self.test_model_crud()
        
        # Test complex operations
        await self.test_property_search()
        await self.test_user_interactions()
        await self.test_performance()
        
        # Generate test report
        await self.generate_test_report()
        
        return self.test_results
    
    async def test_database_connection(self):
        """Test basic database connectivity"""
        test_name = "database_connection"
        logger.info(f"Testing {test_name}...")
        
        try:
            async with RepositoryManager(self.config) as factory:
                # Test health check
                health = await factory.health_check()
                
                self.test_results[test_name] = {
                    "status": "passed" if health["overall"] else "failed",
                    "details": health,
                    "error": None
                }
                
                if health["overall"]:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed: {health}")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
    
    async def test_redis_connection(self):
        """Test Redis connectivity"""
        test_name = "redis_connection"
        logger.info(f"Testing {test_name}...")
        
        try:
            async with RepositoryManager(self.config) as factory:
                cache_repo = factory.get_cache_repository()
                
                # Test cache operations
                test_key = "test_connection"
                test_value = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
                
                # Set and get test data
                await cache_repo.set(test_key, test_value, ttl=60)
                retrieved_value = await cache_repo.get(test_key)
                
                # Health check
                health_check = await cache_repo.health_check()
                
                success = (
                    retrieved_value is not None and 
                    retrieved_value.get("test") == "data" and
                    health_check
                )
                
                self.test_results[test_name] = {
                    "status": "passed" if success else "failed",
                    "details": {
                        "cache_write": test_value,
                        "cache_read": retrieved_value,
                        "health_check": health_check
                    },
                    "error": None
                }
                
                # Cleanup
                await cache_repo.delete(test_key)
                
                if success:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
    
    async def test_repository_initialization(self):
        """Test repository factory initialization"""
        test_name = "repository_initialization"
        logger.info(f"Testing {test_name}...")
        
        try:
            async with RepositoryManager(self.config) as factory:
                # Test all repository getters
                user_repo = factory.get_user_repository()
                property_repo = factory.get_property_repository()
                model_repo = factory.get_model_repository()
                cache_repo = factory.get_cache_repository()
                
                # Test if repositories are properly initialized
                repos_initialized = all([
                    user_repo is not None,
                    property_repo is not None,
                    model_repo is not None,
                    cache_repo is not None
                ])
                
                self.test_results[test_name] = {
                    "status": "passed" if repos_initialized else "failed",
                    "details": {
                        "user_repository": user_repo.__class__.__name__ if user_repo else None,
                        "property_repository": property_repo.__class__.__name__ if property_repo else None,
                        "model_repository": model_repo.__class__.__name__ if model_repo else None,
                        "cache_repository": cache_repo.__class__.__name__ if cache_repo else None
                    },
                    "error": None
                }
                
                if repos_initialized:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
    
    async def test_property_crud(self):
        """Test property CRUD operations"""
        test_name = "property_crud"
        logger.info(f"Testing {test_name}...")
        
        test_property_id = uuid.uuid4()
        
        try:
            async with RepositoryManager(self.config) as factory:
                property_repo = factory.get_property_repository()
                
                # Create test property
                test_property = Property(
                    id=test_property_id,
                    title="Test Property for CRUD",
                    description="A test property for validating CRUD operations",
                    price=2500.0,
                    location="Test City, Test State",
                    bedrooms=2,
                    bathrooms=2.0,
                    square_feet=1200,
                    amenities=["parking", "gym", "pool"],
                    contact_info={"phone": "555-0123", "email": "test@example.com"},
                    images=["https://example.com/image1.jpg"],
                    scraped_at=datetime.utcnow(),
                    is_active=True,
                    property_type="apartment"
                )
                
                # Test CREATE
                created_property = await property_repo.create(test_property)
                create_success = created_property is not None and created_property.id == test_property_id
                
                # Test READ
                retrieved_property = await property_repo.get_by_id(test_property_id)
                read_success = (
                    retrieved_property is not None and 
                    retrieved_property.id == test_property_id and
                    retrieved_property.title == test_property.title
                )
                
                # Test UPDATE
                if retrieved_property:
                    retrieved_property.price = 2600.0
                    retrieved_property.title = "Updated Test Property"
                    updated_property = await property_repo.update(retrieved_property)
                    update_success = (
                        updated_property is not None and
                        updated_property.price == 2600.0 and
                        updated_property.title == "Updated Test Property"
                    )
                else:
                    update_success = False
                
                # Test search functionality
                search_query = SearchQuery(
                    query_text="Test",
                    filters=SearchFilters(min_price=2000, max_price=3000),
                    limit=10
                )
                search_results, total_count = await property_repo.search(search_query)
                search_success = any(p.id == test_property_id for p in search_results)
                
                # Test DELETE
                delete_success = await property_repo.delete(test_property_id)
                
                # Verify deletion
                deleted_property = await property_repo.get_by_id(test_property_id)
                deletion_verified = deleted_property is None
                
                overall_success = all([
                    create_success, read_success, update_success, 
                    search_success, delete_success, deletion_verified
                ])
                
                self.test_results[test_name] = {
                    "status": "passed" if overall_success else "failed",
                    "details": {
                        "create": create_success,
                        "read": read_success,
                        "update": update_success,
                        "search": search_success,
                        "delete": delete_success,
                        "deletion_verified": deletion_verified,
                        "search_results_count": len(search_results) if search_results else 0
                    },
                    "error": None
                }
                
                if overall_success:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
            
            # Cleanup on error
            try:
                async with RepositoryManager(self.config) as factory:
                    property_repo = factory.get_property_repository()
                    await property_repo.delete(test_property_id)
            except:
                pass
    
    async def test_user_crud(self):
        """Test user CRUD operations"""
        test_name = "user_crud"
        logger.info(f"Testing {test_name}...")
        
        test_user_id = uuid.uuid4()
        test_email = f"test_{test_user_id}@example.com"
        
        try:
            async with RepositoryManager(self.config) as factory:
                user_repo = factory.get_user_repository()
                
                # Create test user
                test_preferences = UserPreferences(
                    min_price=1000.0,
                    max_price=3000.0,
                    min_bedrooms=1,
                    max_bedrooms=3,
                    min_bathrooms=1.0,
                    max_bathrooms=2.5,
                    preferred_locations=["Test City"],
                    required_amenities=["parking"],
                    property_types=["apartment", "condo"]
                )
                
                test_user = User(
                    id=test_user_id,
                    email=test_email,
                    preferences=test_preferences,
                    interactions=[],
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                
                # Test CREATE
                created_user = await user_repo.create(test_user)
                create_success = created_user is not None and created_user.id == test_user_id
                
                # Test READ by ID
                retrieved_user = await user_repo.get_by_id(test_user_id)
                read_success = (
                    retrieved_user is not None and 
                    retrieved_user.id == test_user_id and
                    retrieved_user.email == test_email
                )
                
                # Test READ by email
                retrieved_by_email = await user_repo.get_by_email(test_email)
                email_read_success = (
                    retrieved_by_email is not None and
                    retrieved_by_email.id == test_user_id
                )
                
                # Test UPDATE
                if retrieved_user:
                    retrieved_user.preferences.max_price = 3500.0
                    updated_user = await user_repo.update(retrieved_user)
                    update_success = (
                        updated_user is not None and
                        updated_user.preferences.max_price == 3500.0
                    )
                else:
                    update_success = False
                
                # Test user statistics
                stats = await user_repo.get_user_statistics(test_user_id)
                stats_success = stats is not None and isinstance(stats, dict)
                
                # Test DELETE
                delete_success = await user_repo.delete(test_user_id)
                
                # Verify deletion
                deleted_user = await user_repo.get_by_id(test_user_id)
                deletion_verified = deleted_user is None
                
                overall_success = all([
                    create_success, read_success, email_read_success,
                    update_success, stats_success, delete_success, deletion_verified
                ])
                
                self.test_results[test_name] = {
                    "status": "passed" if overall_success else "failed",
                    "details": {
                        "create": create_success,
                        "read": read_success,
                        "email_read": email_read_success,
                        "update": update_success,
                        "statistics": stats_success,
                        "delete": delete_success,
                        "deletion_verified": deletion_verified
                    },
                    "error": None
                }
                
                if overall_success:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
            
            # Cleanup on error
            try:
                async with RepositoryManager(self.config) as factory:
                    user_repo = factory.get_user_repository()
                    await user_repo.delete(test_user_id)
            except:
                pass
    
    async def test_model_crud(self):
        """Test ML model repository operations"""
        test_name = "model_crud"
        logger.info(f"Testing {test_name}...")
        
        try:
            async with RepositoryManager(self.config) as factory:
                model_repo = factory.get_model_repository()
                
                # Test data
                test_model_name = "test_model"
                test_version = "1.0.0"
                test_model_data = {"weights": [1, 2, 3], "bias": 0.5, "metadata": "test"}
                test_embeddings = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
                test_metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.92}
                
                # Test model save/load
                save_success = await model_repo.save_model(test_model_name, test_model_data, test_version)
                loaded_model = await model_repo.load_model(test_model_name, test_version)
                load_success = loaded_model is not None and loaded_model == test_model_data
                
                # Test model versions
                versions = await model_repo.get_model_versions(test_model_name)
                versions_success = test_version in versions
                
                # Test embeddings
                embeddings_save_success = await model_repo.save_embeddings(
                    "test_entity", "test_123", test_embeddings
                )
                loaded_embeddings = await model_repo.get_embeddings("test_entity", "test_123")
                embeddings_load_success = (
                    loaded_embeddings is not None and
                    np.array_equal(loaded_embeddings, test_embeddings)
                )
                
                # Test training metrics
                metrics_save_success = await model_repo.save_training_metrics(
                    test_model_name, test_version, test_metrics
                )
                loaded_metrics = await model_repo.get_training_metrics(test_model_name, test_version)
                metrics_load_success = (
                    loaded_metrics is not None and
                    loaded_metrics.get("accuracy") == 0.95
                )
                
                # Test prediction caching
                cache_key = "test_predictions"
                test_predictions = {"predictions": [0.8, 0.9, 0.7], "model_version": test_version}
                cache_save_success = await model_repo.cache_predictions(cache_key, test_predictions)
                cached_predictions = await model_repo.get_cached_predictions(cache_key)
                cache_load_success = (
                    cached_predictions is not None and
                    cached_predictions == test_predictions
                )
                
                # Test cleanup
                delete_success = await model_repo.delete_model(test_model_name, test_version)
                clear_cache_success = await model_repo.clear_cache("test_*")
                
                overall_success = all([
                    save_success, load_success, versions_success,
                    embeddings_save_success, embeddings_load_success,
                    metrics_save_success, metrics_load_success,
                    cache_save_success, cache_load_success,
                    delete_success, clear_cache_success
                ])
                
                self.test_results[test_name] = {
                    "status": "passed" if overall_success else "failed",
                    "details": {
                        "model_save": save_success,
                        "model_load": load_success,
                        "versions": versions_success,
                        "embeddings_save": embeddings_save_success,
                        "embeddings_load": embeddings_load_success,
                        "metrics_save": metrics_save_success,
                        "metrics_load": metrics_load_success,
                        "cache_save": cache_save_success,
                        "cache_load": cache_load_success,
                        "delete": delete_success,
                        "clear_cache": clear_cache_success
                    },
                    "error": None
                }
                
                if overall_success:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
    
    async def test_property_search(self):
        """Test advanced property search functionality"""
        test_name = "property_search"
        logger.info(f"Testing {test_name}...")
        
        try:
            async with RepositoryManager(self.config) as factory:
                property_repo = factory.get_property_repository()
                
                # Create multiple test properties for search testing
                test_properties = []
                for i in range(5):
                    prop = Property(
                        id=uuid.uuid4(),
                        title=f"Search Test Property {i+1}",
                        description=f"Test property number {i+1} for search functionality",
                        price=2000.0 + (i * 500),  # Prices: 2000, 2500, 3000, 3500, 4000
                        location=f"Test Location {i % 2 + 1}",  # Alternating locations
                        bedrooms=i % 3 + 1,  # 1, 2, 3, 1, 2 bedrooms
                        bathrooms=1.0 + (i % 2),  # 1.0, 2.0 bathrooms
                        square_feet=800 + (i * 200),  # Square feet: 800, 1000, 1200, 1400, 1600
                        amenities=["parking"] + (["gym"] if i % 2 == 0 else ["pool"]),
                        contact_info={"phone": f"555-010{i}"},
                        images=[],
                        scraped_at=datetime.utcnow(),
                        is_active=True,
                        property_type="apartment"
                    )
                    created_prop = await property_repo.create(prop)
                    if created_prop:
                        test_properties.append(prop)
                
                search_tests = []
                
                # Test 1: Basic text search
                search_query1 = SearchQuery(query_text="Search Test", limit=10)
                results1, count1 = await property_repo.search(search_query1)
                search_tests.append(("text_search", len(results1) >= 1))
                
                # Test 2: Price range filter
                search_query2 = SearchQuery(
                    query_text="",
                    filters=SearchFilters(min_price=2300, max_price=3200),
                    limit=10
                )
                results2, count2 = await property_repo.search(search_query2)
                search_tests.append(("price_filter", len(results2) >= 1))
                
                # Test 3: Bedroom filter
                search_query3 = SearchQuery(
                    query_text="",
                    filters=SearchFilters(min_bedrooms=2, max_bedrooms=3),
                    limit=10
                )
                results3, count3 = await property_repo.search(search_query3)
                search_tests.append(("bedroom_filter", len(results3) >= 1))
                
                # Test 4: Location filter
                search_query4 = SearchQuery(
                    query_text="",
                    filters=SearchFilters(locations=["Test Location 1"]),
                    limit=10
                )
                results4, count4 = await property_repo.search(search_query4)
                search_tests.append(("location_filter", len(results4) >= 1))
                
                # Test 5: Combined filters
                search_query5 = SearchQuery(
                    query_text="Test",
                    filters=SearchFilters(
                        min_price=2000,
                        max_price=4000,
                        min_bedrooms=1,
                        amenities=["parking"]
                    ),
                    limit=10,
                    sort_by="price_asc"
                )
                results5, count5 = await property_repo.search(search_query5)
                search_tests.append(("combined_filters", len(results5) >= 1))
                
                # Cleanup test properties
                cleanup_success = True
                for prop in test_properties:
                    if not await property_repo.delete(prop.id):
                        cleanup_success = False
                
                # Check if all search tests passed
                all_passed = all(result for _, result in search_tests) and cleanup_success
                
                self.test_results[test_name] = {
                    "status": "passed" if all_passed else "failed",
                    "details": {
                        "search_tests": dict(search_tests),
                        "cleanup": cleanup_success,
                        "test_properties_created": len(test_properties)
                    },
                    "error": None
                }
                
                if all_passed:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
    
    async def test_user_interactions(self):
        """Test user interaction functionality"""
        test_name = "user_interactions"
        logger.info(f"Testing {test_name}...")
        
        test_user_id = uuid.uuid4()
        test_property_id = uuid.uuid4()
        
        try:
            async with RepositoryManager(self.config) as factory:
                user_repo = factory.get_user_repository()
                
                # Create test user
                test_user = User(
                    id=test_user_id,
                    email=f"interaction_test_{test_user_id}@example.com",
                    preferences=UserPreferences(
                        min_price=1000.0,
                        max_price=3000.0,
                        property_types=["apartment"]
                    ),
                    interactions=[],
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                
                user_created = await user_repo.create(test_user)
                
                if user_created:
                    # Test adding interactions
                    interaction_types = ["view", "like", "inquiry", "save"]
                    interactions_added = []
                    
                    for interaction_type in interaction_types:
                        interaction = UserInteraction(
                            property_id=test_property_id,
                            interaction_type=interaction_type,
                            timestamp=datetime.utcnow(),
                            duration_seconds=30
                        )
                        
                        add_success = await user_repo.add_interaction(test_user_id, interaction)
                        interactions_added.append(add_success)
                    
                    # Test retrieving interactions
                    all_interactions = await user_repo.get_interactions(test_user_id)
                    specific_interactions = await user_repo.get_interactions(
                        test_user_id, interaction_type="like"
                    )
                    
                    # Test user statistics
                    stats = await user_repo.get_user_statistics(test_user_id)
                    
                    # Test behavior analytics
                    behavior_analytics = await user_repo.get_user_behavior_analytics(days=30)
                    
                    test_success = all([
                        all(interactions_added),
                        len(all_interactions) == len(interaction_types),
                        len(specific_interactions) == 1,
                        stats is not None,
                        behavior_analytics is not None,
                        stats.get("total_interactions", 0) > 0
                    ])
                    
                    # Cleanup
                    await user_repo.delete(test_user_id)
                    
                    self.test_results[test_name] = {
                        "status": "passed" if test_success else "failed",
                        "details": {
                            "interactions_added": sum(interactions_added),
                            "all_interactions_count": len(all_interactions),
                            "specific_interactions_count": len(specific_interactions),
                            "stats_available": stats is not None,
                            "analytics_available": behavior_analytics is not None
                        },
                        "error": None
                    }
                    
                    if test_success:
                        logger.info(f"✅ {test_name} passed")
                    else:
                        logger.error(f"❌ {test_name} failed")
                else:
                    raise Exception("Failed to create test user")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
            
            # Cleanup on error
            try:
                async with RepositoryManager(self.config) as factory:
                    user_repo = factory.get_user_repository()
                    await user_repo.delete(test_user_id)
            except:
                pass
    
    async def test_performance(self):
        """Test repository performance and concurrency"""
        test_name = "performance"
        logger.info(f"Testing {test_name}...")
        
        try:
            async with RepositoryManager(self.config) as factory:
                property_repo = factory.get_property_repository()
                
                # Test batch operations performance
                start_time = datetime.utcnow()
                
                # Create multiple properties for performance testing
                test_properties = []
                for i in range(10):  # Small batch for testing
                    prop = Property(
                        id=uuid.uuid4(),
                        title=f"Performance Test Property {i+1}",
                        description=f"Test property for performance testing",
                        price=2000.0 + (i * 100),
                        location="Performance Test Location",
                        bedrooms=2,
                        bathrooms=2.0,
                        square_feet=1000,
                        amenities=["parking", "gym"],
                        contact_info={"phone": "555-0123"},
                        images=[],
                        scraped_at=datetime.utcnow(),
                        is_active=True,
                        property_type="apartment"
                    )
                    test_properties.append(prop)
                
                # Test bulk creation
                created_properties = await property_repo.bulk_create(test_properties)
                bulk_create_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Test batch retrieval
                start_time = datetime.utcnow()
                property_ids = [p.id for p in created_properties]
                retrieved_properties = await property_repo.get_by_ids(property_ids)
                batch_read_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Test search performance
                start_time = datetime.utcnow()
                search_query = SearchQuery(
                    query_text="Performance Test",
                    filters=SearchFilters(min_price=2000, max_price=3000),
                    limit=50
                )
                search_results, _ = await property_repo.search(search_query)
                search_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Test connection info
                connection_info = await property_repo.get_connection_info()
                
                # Cleanup
                cleanup_success = True
                for prop in created_properties:
                    if not await property_repo.delete(prop.id):
                        cleanup_success = False
                
                performance_acceptable = (
                    bulk_create_time < 10.0 and  # Should create 10 properties in under 10 seconds
                    batch_read_time < 5.0 and    # Should read 10 properties in under 5 seconds
                    search_time < 5.0 and        # Should search in under 5 seconds
                    len(created_properties) == len(test_properties) and
                    len(retrieved_properties) == len(property_ids) and
                    cleanup_success
                )
                
                self.test_results[test_name] = {
                    "status": "passed" if performance_acceptable else "failed",
                    "details": {
                        "bulk_create_time": bulk_create_time,
                        "batch_read_time": batch_read_time,
                        "search_time": search_time,
                        "properties_created": len(created_properties),
                        "properties_retrieved": len(retrieved_properties),
                        "search_results": len(search_results),
                        "connection_info": connection_info,
                        "cleanup": cleanup_success
                    },
                    "error": None
                }
                
                if performance_acceptable:
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.error(f"❌ {test_name} failed")
                    
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            self.test_results[test_name] = {
                "status": "failed",
                "details": None,
                "error": str(e)
            }
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "passed")
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*80)
        print("REPOSITORY CONNECTION TEST REPORT")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*80)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"{status_icon} {test_name.upper()}: {result['status'].upper()}")
            
            if result["status"] == "failed" and result["error"]:
                print(f"   Error: {result['error']}")
            elif result["details"]:
                print(f"   Details: {result['details']}")
            print()
        
        print("="*80)
        
        # Database configuration summary
        print("DATABASE CONFIGURATION:")
        print(f"Host: {self.config.database.host}")
        print(f"Port: {self.config.database.port}")
        print(f"Database: {self.config.database.database}")
        print(f"Pool Size: {self.config.database.pool_size}")
        print()
        
        print("REDIS CONFIGURATION:")
        print(f"Host: {self.config.redis.host}")
        print(f"Port: {self.config.redis.port}")
        print(f"DB: {self.config.redis.db}")
        print()
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Repository layer is production-ready.")
        else:
            print("⚠️  Some tests failed. Please review the errors above.")
        
        print("="*80)

async def main():
    """Main test function"""
    print("Repository Connection Test Suite")
    print("================================")
    
    # Check environment variables
    required_env_vars = [
        "DB_HOST", "DB_PORT", "DB_NAME", "DB_USERNAME", "DB_PASSWORD",
        "REDIS_HOST", "REDIS_PORT"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Using default values for testing...")
        print()
    
    # Run tests
    tester = RepositoryTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Exit with appropriate code
        failed_count = sum(1 for result in results.values() if result["status"] == "failed")
        exit_code = 0 if failed_count == 0 else 1
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed with unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)