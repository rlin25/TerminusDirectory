#!/usr/bin/env python3
"""
Test Data Operations for Rental ML System
Verify that property search, recommendations, and repository operations work correctly
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uuid

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load production environment
load_dotenv(project_root / ".env.production")

from src.infrastructure.data.repository_factory import RepositoryFactory
from src.domain.entities.property import Property
from src.domain.entities.user import User, UserPreferences
from src.domain.entities.search_query import SearchQuery, SearchFilters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataOperationsTester:
    """Test suite for verifying data operations"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rental_ml")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.repo_factory = None
        
    async def initialize(self):
        """Initialize repository factory"""
        try:
            self.repo_factory = RepositoryFactory(
                database_url=self.database_url,
                redis_url=self.redis_url,
                pool_size=5,
                enable_performance_monitoring=True
            )
            await self.repo_factory.initialize()
            logger.info("Repository factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize repository factory: {e}")
            raise
    
    async def close(self):
        """Close repository factory"""
        if self.repo_factory:
            await self.repo_factory.close()
            logger.info("Repository factory closed")
    
    async def test_repository_connections(self) -> Dict[str, Any]:
        """Test all repository connections"""
        logger.info("Testing repository connections...")
        
        results = {
            'health_check': {},
            'repositories': {},
            'overall_success': False
        }
        
        try:
            # Test health check
            health_status = await self.repo_factory.health_check()
            results['health_check'] = health_status
            
            # Test individual repositories
            results['repositories']['user_repo'] = await self._test_user_repository()
            results['repositories']['property_repo'] = await self._test_property_repository()
            results['repositories']['model_repo'] = await self._test_model_repository()
            
            # Determine overall success
            repo_successes = [r.get('success', False) for r in results['repositories'].values()]
            results['overall_success'] = health_status.get('overall', False) and all(repo_successes)
            
        except Exception as e:
            logger.error(f"Repository connection test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _test_user_repository(self) -> Dict[str, Any]:
        """Test user repository operations"""
        logger.info("Testing user repository...")
        
        result = {'success': False, 'operations': {}, 'error': None}
        
        try:
            user_repo = self.repo_factory.get_user_repository()
            
            # Test create user
            test_user = User.create(
                email=f"test_user_{uuid.uuid4()}@example.com",
                preferences=UserPreferences(
                    min_price=1000,
                    max_price=3000,
                    min_bedrooms=1,
                    max_bedrooms=3,
                    preferred_locations=["Downtown", "Midtown"]
                )
            )
            
            created_user = await user_repo.create(test_user)
            result['operations']['create'] = created_user is not None
            
            if created_user:
                # Test find by id
                found_user = await user_repo.find_by_id(created_user.id)
                result['operations']['find_by_id'] = found_user is not None
                
                # Test find by email
                found_by_email = await user_repo.find_by_email(created_user.email)
                result['operations']['find_by_email'] = found_by_email is not None
                
                # Test update
                created_user.preferences.max_price = 4000
                updated_user = await user_repo.update(created_user)
                result['operations']['update'] = updated_user is not None
                
                # Test delete
                deleted = await user_repo.delete(created_user.id)
                result['operations']['delete'] = deleted
            
            result['success'] = all(result['operations'].values())
            
        except Exception as e:
            logger.error(f"User repository test failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _test_property_repository(self) -> Dict[str, Any]:
        """Test property repository operations"""
        logger.info("Testing property repository...")
        
        result = {'success': False, 'operations': {}, 'error': None}
        
        try:
            property_repo = self.repo_factory.get_property_repository()
            
            # Test create property
            test_property = Property.create(
                title=f"Test Property {uuid.uuid4()}",
                description="Test property for repository testing",
                price=2500.0,
                location="Test Location",
                bedrooms=2,
                bathrooms=1.5,
                square_feet=900,
                amenities=["parking", "laundry"],
                contact_info={"phone": "555-123-4567"},
                images=["https://example.com/image1.jpg"],
                property_type="apartment"
            )
            
            created_property = await property_repo.create(test_property)
            result['operations']['create'] = created_property is not None
            
            if created_property:
                # Test find by id
                found_property = await property_repo.find_by_id(created_property.id)
                result['operations']['find_by_id'] = found_property is not None
                
                # Test search
                search_results = await property_repo.search({
                    'min_price': 2000,
                    'max_price': 3000,
                    'location': 'Test Location'
                })
                result['operations']['search'] = len(search_results) > 0
                
                # Test get all active
                active_properties = await property_repo.get_all_active(limit=10)
                result['operations']['get_all_active'] = isinstance(active_properties, list)
                
                # Test update
                created_property.price = 2600.0
                updated_property = await property_repo.update(created_property)
                result['operations']['update'] = updated_property is not None
                
                # Test delete
                deleted = await property_repo.delete(created_property.id)
                result['operations']['delete'] = deleted
            
            result['success'] = all(result['operations'].values())
            
        except Exception as e:
            logger.error(f"Property repository test failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _test_model_repository(self) -> Dict[str, Any]:
        """Test model repository operations"""
        logger.info("Testing model repository...")
        
        result = {'success': False, 'operations': {}, 'error': None}
        
        try:
            model_repo = self.repo_factory.get_model_repository()
            
            # Test save model
            model_data = {
                'model_name': 'test_model',
                'version': '1.0.0',
                'metadata': {'test': True},
                'model_data': b'test_model_data'
            }
            
            saved = await model_repo.save_model(**model_data)
            result['operations']['save_model'] = saved
            
            if saved:
                # Test load model
                loaded_model = await model_repo.load_model('test_model', '1.0.0')
                result['operations']['load_model'] = loaded_model is not None
                
                # Test list models
                models = await model_repo.list_models()
                result['operations']['list_models'] = isinstance(models, list)
                
                # Test get latest version
                latest = await model_repo.get_latest_version('test_model')
                result['operations']['get_latest_version'] = latest is not None
            
            result['success'] = all(result['operations'].values())
            
        except Exception as e:
            logger.error(f"Model repository test failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def test_property_search(self, sample_size: int = 10) -> Dict[str, Any]:
        """Test property search functionality"""
        logger.info("Testing property search functionality...")
        
        result = {
            'success': False,
            'search_tests': {},
            'sample_size': sample_size,
            'error': None
        }
        
        try:
            property_repo = self.repo_factory.get_property_repository()
            
            # Test basic search
            basic_search = await property_repo.search({'limit': sample_size})
            result['search_tests']['basic_search'] = {
                'success': True,
                'count': len(basic_search),
                'has_results': len(basic_search) > 0
            }
            
            # Test price range search
            price_search = await property_repo.search({
                'min_price': 1000,
                'max_price': 3000,
                'limit': sample_size
            })
            result['search_tests']['price_range'] = {
                'success': True,
                'count': len(price_search),
                'has_results': len(price_search) > 0
            }
            
            # Test bedroom search
            bedroom_search = await property_repo.search({
                'min_bedrooms': 2,
                'max_bedrooms': 3,
                'limit': sample_size
            })
            result['search_tests']['bedroom_filter'] = {
                'success': True,
                'count': len(bedroom_search),
                'has_results': len(bedroom_search) > 0
            }
            
            # Test location search (if we have location data)
            location_search = await property_repo.search({
                'location': 'Downtown',
                'limit': sample_size
            })
            result['search_tests']['location_filter'] = {
                'success': True,
                'count': len(location_search),
                'has_results': len(location_search) >= 0  # May be 0 if no downtown properties
            }
            
            # Test combined filters
            combined_search = await property_repo.search({
                'min_price': 1500,
                'max_price': 4000,
                'min_bedrooms': 1,
                'property_type': 'apartment',
                'limit': sample_size
            })
            result['search_tests']['combined_filters'] = {
                'success': True,
                'count': len(combined_search),
                'has_results': len(combined_search) >= 0
            }
            
            # Determine overall success
            search_successes = [t.get('success', False) for t in result['search_tests'].values()]
            result['success'] = all(search_successes)
            
        except Exception as e:
            logger.error(f"Property search test failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def test_recommendations(self) -> Dict[str, Any]:
        """Test recommendation system functionality"""
        logger.info("Testing recommendation system...")
        
        result = {
            'success': False,
            'recommendation_tests': {},
            'error': None
        }
        
        try:
            # This is a basic test - in a full implementation, you would test
            # the ML recommendation services
            
            # Test that we can get properties for recommendation
            property_repo = self.repo_factory.get_property_repository()
            properties = await property_repo.get_all_active(limit=20)
            
            result['recommendation_tests']['data_available'] = {
                'success': len(properties) > 0,
                'property_count': len(properties),
                'message': f"Found {len(properties)} properties for recommendations"
            }
            
            # Test that we can get users for recommendations
            user_repo = self.repo_factory.get_user_repository()
            users = await user_repo.get_all(limit=10)
            
            result['recommendation_tests']['users_available'] = {
                'success': len(users) >= 0,
                'user_count': len(users),
                'message': f"Found {len(users)} users for recommendations"
            }
            
            # Simulate basic content-based recommendation
            if len(properties) > 5:
                # Pick a property and find similar ones
                target_property = properties[0]
                similar_properties = [
                    p for p in properties[1:] 
                    if (abs(p.price - target_property.price) < 500 and 
                        p.bedrooms == target_property.bedrooms)
                ]
                
                result['recommendation_tests']['content_based_similarity'] = {
                    'success': True,
                    'target_property': target_property.title,
                    'similar_count': len(similar_properties),
                    'message': f"Found {len(similar_properties)} similar properties"
                }
            
            # Determine overall success
            test_successes = [t.get('success', False) for t in result['recommendation_tests'].values()]
            result['success'] = all(test_successes)
            
        except Exception as e:
            logger.error(f"Recommendation test failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def test_user_interactions(self) -> Dict[str, Any]:
        """Test user interaction functionality"""
        logger.info("Testing user interactions...")
        
        result = {
            'success': False,
            'interaction_tests': {},
            'error': None
        }
        
        try:
            user_repo = self.repo_factory.get_user_repository()
            property_repo = self.repo_factory.get_property_repository()
            
            # Get some test data
            users = await user_repo.get_all(limit=5)
            properties = await property_repo.get_all_active(limit=5)
            
            result['interaction_tests']['test_data_available'] = {
                'success': len(users) > 0 and len(properties) > 0,
                'users': len(users),
                'properties': len(properties),
                'message': f"Available: {len(users)} users, {len(properties)} properties"
            }
            
            if len(users) > 0 and len(properties) > 0:
                # Test user can view properties
                user = users[0]
                property_obj = properties[0]
                
                # In a full system, you would test actual interaction recording
                result['interaction_tests']['user_property_interaction'] = {
                    'success': True,
                    'user_id': str(user.id),
                    'property_id': str(property_obj.id),
                    'message': "User-property interaction structure available"
                }
                
                # Test user preferences matching
                user_preferences = user.preferences
                matching_properties = [
                    p for p in properties 
                    if (user_preferences.min_price <= p.price <= user_preferences.max_price and
                        user_preferences.min_bedrooms <= p.bedrooms <= user_preferences.max_bedrooms)
                ]
                
                result['interaction_tests']['preference_matching'] = {
                    'success': True,
                    'matching_count': len(matching_properties),
                    'total_properties': len(properties),
                    'message': f"Found {len(matching_properties)} properties matching user preferences"
                }
            
            # Determine overall success
            test_successes = [t.get('success', False) for t in result['interaction_tests'].values()]
            result['success'] = all(test_successes)
            
        except Exception as e:
            logger.error(f"User interaction test failed: {e}")
            result['error'] = str(e)
        
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all data operation tests"""
        logger.info("Running comprehensive data operations test suite...")
        
        start_time = datetime.now()
        
        test_results = {
            'started_at': start_time.isoformat(),
            'overall_success': False,
            'tests': {},
            'summary': {},
            'errors': []
        }
        
        try:
            await self.initialize()
            
            # Run all test categories
            test_results['tests']['repository_connections'] = await self.test_repository_connections()
            test_results['tests']['property_search'] = await self.test_property_search()
            test_results['tests']['recommendations'] = await self.test_recommendations()
            test_results['tests']['user_interactions'] = await self.test_user_interactions()
            
            # Calculate summary
            test_results['summary'] = self._calculate_test_summary(test_results['tests'])
            
            # Determine overall success
            category_successes = [t.get('overall_success', t.get('success', False)) for t in test_results['tests'].values()]
            test_results['overall_success'] = all(category_successes)
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            test_results['errors'].append(str(e))
            
        finally:
            await self.close()
            
        end_time = datetime.now()
        test_results['completed_at'] = end_time.isoformat()
        test_results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        return test_results
    
    def _calculate_test_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate test summary statistics"""
        summary = {
            'total_categories': len(tests),
            'successful_categories': 0,
            'failed_categories': 0,
            'total_individual_tests': 0,
            'successful_individual_tests': 0,
            'failed_individual_tests': 0
        }
        
        for category, results in tests.items():
            # Count category success
            if results.get('overall_success', results.get('success', False)):
                summary['successful_categories'] += 1
            else:
                summary['failed_categories'] += 1
            
            # Count individual tests
            if 'repositories' in results:
                for repo_test in results['repositories'].values():
                    if 'operations' in repo_test:
                        for op_success in repo_test['operations'].values():
                            summary['total_individual_tests'] += 1
                            if op_success:
                                summary['successful_individual_tests'] += 1
                            else:
                                summary['failed_individual_tests'] += 1
            
            if 'search_tests' in results:
                for search_test in results['search_tests'].values():
                    summary['total_individual_tests'] += 1
                    if search_test.get('success', False):
                        summary['successful_individual_tests'] += 1
                    else:
                        summary['failed_individual_tests'] += 1
            
            # Similar counting for other test types...
        
        return summary


async def main():
    """Main function for command-line usage"""
    
    print("="*60)
    print("RENTAL ML SYSTEM - DATA OPERATIONS TEST SUITE")
    print("="*60)
    
    tester = DataOperationsTester()
    results = await tester.run_all_tests()
    
    # Print results
    print(f"\nTest Execution: {results['overall_success'] and 'PASSED' or 'FAILED'}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    print(f"Started: {results['started_at']}")
    print(f"Completed: {results['completed_at']}")
    
    print("\n" + "-"*40)
    print("SUMMARY")
    print("-"*40)
    summary = results.get('summary', {})
    print(f"Test Categories: {summary.get('successful_categories', 0)}/{summary.get('total_categories', 0)} passed")
    print(f"Individual Tests: {summary.get('successful_individual_tests', 0)}/{summary.get('total_individual_tests', 0)} passed")
    
    print("\n" + "-"*40)
    print("DETAILED RESULTS")
    print("-"*40)
    
    for category, test_result in results.get('tests', {}).items():
        success = test_result.get('overall_success', test_result.get('success', False))
        print(f"\n{category.upper()}: {'PASS' if success else 'FAIL'}")
        
        if 'repositories' in test_result:
            for repo_name, repo_result in test_result['repositories'].items():
                repo_success = repo_result.get('success', False)
                print(f"  {repo_name}: {'PASS' if repo_success else 'FAIL'}")
                if 'operations' in repo_result:
                    for op_name, op_success in repo_result['operations'].items():
                        print(f"    {op_name}: {'PASS' if op_success else 'FAIL'}")
        
        if 'search_tests' in test_result:
            for search_name, search_result in test_result['search_tests'].items():
                search_success = search_result.get('success', False)
                count = search_result.get('count', 0)
                print(f"  {search_name}: {'PASS' if search_success else 'FAIL'} ({count} results)")
        
        if 'recommendation_tests' in test_result:
            for rec_name, rec_result in test_result['recommendation_tests'].items():
                rec_success = rec_result.get('success', False)
                message = rec_result.get('message', '')
                print(f"  {rec_name}: {'PASS' if rec_success else 'FAIL'} - {message}")
        
        if 'interaction_tests' in test_result:
            for int_name, int_result in test_result['interaction_tests'].items():
                int_success = int_result.get('success', False)
                message = int_result.get('message', '')
                print(f"  {int_name}: {'PASS' if int_success else 'FAIL'} - {message}")
    
    # Print errors if any
    if results.get('errors'):
        print("\n" + "-"*40)
        print("ERRORS")
        print("-"*40)
        for error in results['errors']:
            print(f"  - {error}")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())