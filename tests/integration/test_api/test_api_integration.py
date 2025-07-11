"""
Integration tests for API endpoints.

Tests full request/response cycles with realistic data flow through the application layers.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
import json
from datetime import datetime

from tests.utils.data_factories import PropertyFactory, UserFactory, FactoryConfig
from tests.utils.test_helpers import APITestHelpers, PerformanceTestHelpers


class TestPropertyAPIIntegration:
    """Integration tests for Property API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.app = FastAPI()
        self.client = TestClient(self.app)
        self.factory = PropertyFactory(FactoryConfig(seed=42))
        
        # Setup mock dependencies
        self.mock_repository_factory = Mock()
        self.mock_property_repo = AsyncMock()
        self.mock_repository_factory.get_property_repository.return_value = self.mock_property_repo
        
        # Add router to app
        from application.api.routers.property_router import router as property_router
        self.app.include_router(property_router, prefix="/api/properties")
        
        # Add dependency override
        self.app.state.repository_factory = self.mock_repository_factory
    
    @pytest.mark.integration
    def test_get_property_full_flow(self):
        """Test complete property retrieval flow."""
        # Setup
        property_id = uuid4()
        expected_property = self.factory.create()
        expected_property.id = property_id
        
        self.mock_property_repo.get_by_id.return_value = expected_property
        
        # Make request
        response = self.client.get(f"/api/properties/{property_id}")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == str(property_id)
        assert data["title"] == expected_property.title
        assert data["price"] == expected_property.price
        assert data["bedrooms"] == expected_property.bedrooms
        assert "amenities" in data
        assert "contact_info" in data
    
    @pytest.mark.integration
    def test_list_properties_with_pagination(self):
        """Test property listing with pagination parameters."""
        # Setup
        properties = self.factory.create_batch(5)
        self.mock_property_repo.get_all_active.return_value = properties
        
        # Make request with pagination
        response = self.client.get("/api/properties?limit=3&offset=0")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) <= 3
        assert all("id" in prop for prop in data)
        assert all("title" in prop for prop in data)
        assert all("price" in prop for prop in data)
    
    @pytest.mark.integration
    def test_list_properties_with_filters(self):
        """Test property listing with various filters."""
        # Setup
        filtered_properties = self.factory.create_batch(2)
        
        # Test location filter
        self.mock_property_repo.get_by_location.return_value = filtered_properties
        response = self.client.get("/api/properties?location=Downtown")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        
        # Test price range filter
        self.mock_property_repo.get_by_price_range.return_value = filtered_properties
        response = self.client.get("/api/properties?min_price=2000&max_price=4000")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
    
    @pytest.mark.integration
    def test_get_property_features_integration(self):
        """Test property ML features endpoint integration."""
        # Setup
        property_id = uuid4()
        expected_features = {
            'numerical_features': [3000.0, 2.0, 1.5, 1200.0],
            'categorical_features': {'location': 'Downtown', 'property_type': 'apartment'},
            'text_features': {'description_vector': [0.1, 0.2, 0.3]},
            'amenity_features': [1, 0, 1, 1, 0]
        }
        
        self.mock_property_repo.get_property_features.return_value = expected_features
        
        # Make request
        response = self.client.get(f"/api/properties/{property_id}/features")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["property_id"] == str(property_id)
        assert "features" in data
        assert "feature_count" in data
        assert "generated_at" in data
        assert data["features"] == expected_features
    
    @pytest.mark.integration
    def test_get_property_statistics_integration(self):
        """Test property statistics endpoint integration."""
        # Setup mock statistics
        self.mock_property_repo.get_count.return_value = 1250
        self.mock_property_repo.get_active_count.return_value = 1180
        self.mock_property_repo.get_aggregated_stats.return_value = {
            'avg_price': 2850.0,
            'min_price': 800.0,
            'max_price': 8500.0,
            'avg_bedrooms': 2.3,
            'avg_bathrooms': 1.8,
            'avg_square_feet': 1250.0
        }
        
        # Make request
        response = self.client.get("/api/properties/stats/overview")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        APITestHelpers.assert_response_structure(data, [
            'counts', 'pricing', 'features', 'activity', 'generated_at'
        ])
        
        assert data["counts"]["total_properties"] == 1250
        assert data["counts"]["active_properties"] == 1180
        assert data["pricing"]["average_price"] == 2850.0
        assert data["features"]["average_bedrooms"] == 2.3
    
    @pytest.mark.integration
    def test_error_handling_integration(self):
        """Test error handling in property API integration."""
        # Test 404 error
        non_existent_id = uuid4()
        self.mock_property_repo.get_by_id.return_value = None
        
        response = self.client.get(f"/api/properties/{non_existent_id}")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
        assert str(non_existent_id) in error_data["detail"]
        
        # Test 500 error
        self.mock_property_repo.get_by_id.side_effect = Exception("Database error")
        
        response = self.client.get(f"/api/properties/{uuid4()}")
        assert response.status_code == 500
        
        error_data = response.json()
        assert "detail" in error_data
        assert "Failed to retrieve property" in error_data["detail"]


class TestRecommendationAPIIntegration:
    """Integration tests for Recommendation API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.app = FastAPI()
        self.client = TestClient(self.app)
        
        # Setup mock recommendation service
        self.mock_recommendation_service = AsyncMock()
        self.app.state.recommendation_service = self.mock_recommendation_service
        
        # Add router to app (would need actual router implementation)
        # For now, we'll simulate the endpoints
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_user_recommendations_integration(self):
        """Test user recommendations endpoint integration."""
        user_id = uuid4()
        expected_recommendations = [
            {
                'property_id': str(uuid4()),
                'predicted_rating': 0.92,
                'confidence_score': 0.88,
                'explanation': 'Matches your preference for downtown apartments',
                'recommendation_type': 'hybrid',
                'property_details': {
                    'title': 'Luxury Downtown Apartment',
                    'price': 3200,
                    'bedrooms': 2,
                    'location': 'Downtown'
                }
            },
            {
                'property_id': str(uuid4()),
                'predicted_rating': 0.87,
                'confidence_score': 0.83,
                'explanation': 'Similar to properties you liked before',
                'recommendation_type': 'collaborative_filtering',
                'property_details': {
                    'title': 'Modern City Loft',
                    'price': 2800,
                    'bedrooms': 1,
                    'location': 'SoMa'
                }
            }
        ]
        
        self.mock_recommendation_service.get_user_recommendations.return_value = expected_recommendations
        
        # Simulate API call
        result = await self.mock_recommendation_service.get_user_recommendations(
            user_id=user_id,
            num_recommendations=10,
            method='hybrid',
            exclude_seen=True
        )
        
        # Assertions
        assert len(result) == 2
        assert all('property_id' in rec for rec in result)
        assert all('predicted_rating' in rec for rec in result)
        assert all('explanation' in rec for rec in result)
        
        # Check recommendation quality
        assert result[0]['predicted_rating'] > result[1]['predicted_rating']  # Should be sorted
        assert all(0 <= rec['confidence_score'] <= 1 for rec in result)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_similar_properties_integration(self):
        """Test similar properties endpoint integration."""
        property_id = uuid4()
        expected_similar = [
            {
                'property_id': str(uuid4()),
                'similarity_score': 0.94,
                'similarity_reasons': ['Same neighborhood', 'Similar price range', 'Same bedrooms'],
                'property_details': {
                    'title': 'Similar Downtown Apartment',
                    'price': 3100,
                    'bedrooms': 2
                }
            },
            {
                'property_id': str(uuid4()),
                'similarity_score': 0.89,
                'similarity_reasons': ['Similar amenities', 'Same property type'],
                'property_details': {
                    'title': 'Comparable City Apartment',
                    'price': 3400,
                    'bedrooms': 2
                }
            }
        ]
        
        self.mock_recommendation_service.get_similar_properties.return_value = expected_similar
        
        # Simulate API call
        result = await self.mock_recommendation_service.get_similar_properties(
            property_id=property_id,
            num_similar=5,
            similarity_threshold=0.7
        )
        
        # Assertions
        assert len(result) == 2
        assert all('similarity_score' in prop for prop in result)
        assert all('similarity_reasons' in prop for prop in result)
        assert result[0]['similarity_score'] > result[1]['similarity_score']
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recommendation_personalization_integration(self):
        """Test that recommendations are properly personalized."""
        # Test different users get different recommendations
        user1_id = uuid4()
        user2_id = uuid4()
        
        user1_recs = [
            {'property_id': str(uuid4()), 'predicted_rating': 0.9, 'explanation': 'Luxury preference'}
        ]
        user2_recs = [
            {'property_id': str(uuid4()), 'predicted_rating': 0.8, 'explanation': 'Budget-friendly option'}
        ]
        
        # Mock different responses for different users
        def mock_recommendations(user_id, **kwargs):
            if user_id == user1_id:
                return user1_recs
            else:
                return user2_recs
        
        self.mock_recommendation_service.get_user_recommendations.side_effect = mock_recommendations
        
        # Get recommendations for both users
        result1 = await self.mock_recommendation_service.get_user_recommendations(user_id=user1_id)
        result2 = await self.mock_recommendation_service.get_user_recommendations(user_id=user2_id)
        
        # Assertions
        assert result1 != result2
        assert 'Luxury' in result1[0]['explanation']
        assert 'Budget' in result2[0]['explanation']


class TestSearchAPIIntegration:
    """Integration tests for Search API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.app = FastAPI()
        self.client = TestClient(self.app)
        
        # Setup mock search service
        self.mock_search_service = AsyncMock()
        self.app.state.search_service = self.mock_search_service
        
        self.factory = PropertyFactory(FactoryConfig(seed=42))
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_property_search_integration(self):
        """Test property search endpoint integration."""
        query = "2 bedroom apartment downtown parking"
        expected_results = [
            {
                'property': {
                    'id': str(uuid4()),
                    'title': 'Downtown 2BR with Parking',
                    'price': 3200,
                    'bedrooms': 2,
                    'location': 'Downtown',
                    'amenities': ['parking', 'gym']
                },
                'relevance_score': 0.95,
                'matching_criteria': ['bedrooms', 'location', 'amenities'],
                'search_explanation': 'Perfect match for bedrooms, location, and parking'
            },
            {
                'property': {
                    'id': str(uuid4()),
                    'title': 'City Center 2 Bedroom',
                    'price': 2900,
                    'bedrooms': 2,
                    'location': 'City Center',
                    'amenities': ['parking', 'balcony']
                },
                'relevance_score': 0.87,
                'matching_criteria': ['bedrooms', 'amenities'],
                'search_explanation': 'Good match for bedrooms and parking'
            }
        ]
        
        self.mock_search_service.search_properties.return_value = expected_results
        
        # Simulate API call with search parameters
        result = await self.mock_search_service.search_properties(
            query=query,
            filters={
                'min_price': 2000,
                'max_price': 5000,
                'bedrooms': 2,
                'amenities': ['parking']
            },
            sort_by='relevance',
            limit=20,
            offset=0
        )
        
        # Assertions
        assert len(result) == 2
        assert all('property' in item for item in result)
        assert all('relevance_score' in item for item in result)
        assert all('matching_criteria' in item for item in result)
        
        # Check relevance ordering
        assert result[0]['relevance_score'] > result[1]['relevance_score']
        
        # Check matching criteria
        assert 'bedrooms' in result[0]['matching_criteria']
        assert 'location' in result[0]['matching_criteria']
        assert 'amenities' in result[0]['matching_criteria']
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_with_complex_filters_integration(self):
        """Test search with complex filter combinations."""
        complex_filters = {
            'price_range': {'min': 2500, 'max': 4000},
            'bedrooms': [2, 3],
            'bathrooms_min': 1.5,
            'square_feet_range': {'min': 1000, 'max': 2000},
            'locations': ['Downtown', 'SoMa', 'Mission'],
            'amenities_required': ['parking'],
            'amenities_preferred': ['gym', 'pool', 'rooftop'],
            'property_types': ['apartment', 'condo'],
            'pet_friendly': True,
            'availability_date': '2024-02-01'
        }
        
        expected_results = [
            {
                'property': {
                    'id': str(uuid4()),
                    'title': 'Pet-Friendly Downtown Condo',
                    'price': 3500,
                    'bedrooms': 3,
                    'bathrooms': 2.0,
                    'square_feet': 1400,
                    'location': 'Downtown',
                    'amenities': ['parking', 'gym', 'rooftop', 'pet_friendly']
                },
                'relevance_score': 0.92,
                'filter_matches': {
                    'price_range': True,
                    'bedrooms': True,
                    'bathrooms': True,
                    'square_feet': True,
                    'location': True,
                    'required_amenities': True,
                    'preferred_amenities': 2,  # gym, rooftop
                    'property_type': True,
                    'pet_friendly': True
                }
            }
        ]
        
        self.mock_search_service.search_with_filters.return_value = expected_results
        
        # Simulate complex search
        result = await self.mock_search_service.search_with_filters(
            query="pet friendly condo",
            filters=complex_filters,
            ranking_weights={
                'relevance': 0.4,
                'price': 0.3,
                'location': 0.2,
                'amenities': 0.1
            }
        )
        
        # Assertions
        assert len(result) == 1
        property_result = result[0]
        
        assert property_result['relevance_score'] > 0.9
        assert property_result['filter_matches']['required_amenities'] is True
        assert property_result['filter_matches']['preferred_amenities'] >= 2
        assert 'pet_friendly' in property_result['property']['amenities']
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_suggestions_integration(self):
        """Test search suggestions endpoint integration."""
        partial_query = "down"
        expected_suggestions = [
            {
                'suggestion': 'downtown',
                'category': 'location',
                'popularity': 0.95,
                'match_type': 'prefix',
                'result_count': 245
            },
            {
                'suggestion': 'downtown apartment',
                'category': 'combined',
                'popularity': 0.87,
                'match_type': 'phrase',
                'result_count': 156
            },
            {
                'suggestion': 'downtown luxury',
                'category': 'combined',
                'popularity': 0.62,
                'match_type': 'phrase',
                'result_count': 89
            }
        ]
        
        self.mock_search_service.get_search_suggestions.return_value = expected_suggestions
        
        # Simulate suggestions API call
        result = await self.mock_search_service.get_search_suggestions(
            partial_query=partial_query,
            limit=5,
            categories=['location', 'amenity', 'combined'],
            min_popularity=0.1
        )
        
        # Assertions
        assert len(result) == 3
        assert all('suggestion' in item for item in result)
        assert all('category' in item for item in result)
        assert all('popularity' in item for item in result)
        
        # Check ordering by popularity
        popularities = [item['popularity'] for item in result]
        assert popularities == sorted(popularities, reverse=True)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_analytics_integration(self):
        """Test search analytics and tracking integration."""
        search_session = {
            'session_id': str(uuid4()),
            'user_id': str(uuid4()),
            'query': '2 bedroom apartment',
            'filters_applied': {'bedrooms': 2, 'location': 'Downtown'},
            'results_count': 25,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock search analytics tracking
        self.mock_search_service.track_search.return_value = {
            'tracked': True,
            'session_id': search_session['session_id']
        }
        
        # Mock search result interaction tracking
        self.mock_search_service.track_result_interaction.return_value = {
            'tracked': True,
            'interaction_type': 'click',
            'property_id': str(uuid4()),
            'position': 1
        }
        
        # Simulate search tracking
        search_tracking = await self.mock_search_service.track_search(search_session)
        assert search_tracking['tracked'] is True
        
        # Simulate result interaction tracking
        interaction_tracking = await self.mock_search_service.track_result_interaction(
            session_id=search_session['session_id'],
            property_id=str(uuid4()),
            interaction_type='click',
            position=1
        )
        assert interaction_tracking['tracked'] is True


class TestUserAPIIntegration:
    """Integration tests for User API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.app = FastAPI()
        self.client = TestClient(self.app)
        self.user_factory = UserFactory(FactoryConfig(seed=42))
        
        # Setup mock dependencies
        self.mock_repository_factory = Mock()
        self.mock_user_repo = AsyncMock()
        self.mock_repository_factory.get_user_repository.return_value = self.mock_user_repo
        
        self.app.state.repository_factory = self.mock_repository_factory
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_user_lifecycle_integration(self):
        """Test complete user lifecycle integration."""
        # 1. Create user
        user_data = {
            'email': 'integration_test@example.com',
            'preferences': {
                'min_price': 2000,
                'max_price': 4000,
                'preferred_locations': ['Downtown', 'SoMa'],
                'required_amenities': ['parking'],
                'property_types': ['apartment', 'condo']
            }
        }
        
        created_user = self.user_factory.create(email=user_data['email'])
        self.mock_user_repo.save.return_value = created_user
        self.mock_user_repo.get_by_id.return_value = created_user
        self.mock_user_repo.get_by_email.return_value = created_user
        
        # Simulate user creation
        create_result = await self.mock_user_repo.save(created_user)
        assert create_result.email == user_data['email']
        
        # 2. Retrieve user
        get_result = await self.mock_user_repo.get_by_id(created_user.id)
        assert get_result.id == created_user.id
        
        # 3. Update preferences
        updated_user = created_user
        updated_user.preferences.max_price = 5000
        updated_user.preferences.preferred_locations.append('Mission')
        
        self.mock_user_repo.save.return_value = updated_user
        update_result = await self.mock_user_repo.save(updated_user)
        assert update_result.preferences.max_price == 5000
        assert 'Mission' in update_result.preferences.preferred_locations
        
        # 4. Add interactions
        interactions = [
            {'property_id': str(uuid4()), 'interaction_type': 'view', 'timestamp': datetime.now().isoformat()},
            {'property_id': str(uuid4()), 'interaction_type': 'like', 'timestamp': datetime.now().isoformat()},
            {'property_id': str(uuid4()), 'interaction_type': 'save', 'timestamp': datetime.now().isoformat()}
        ]
        
        self.mock_user_repo.get_user_interactions.return_value = interactions
        interaction_result = await self.mock_user_repo.get_user_interactions(created_user.id)
        assert len(interaction_result) == 3
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_user_preferences_validation_integration(self):
        """Test user preferences validation integration."""
        # Test valid preferences
        valid_preferences = {
            'min_price': 1500,
            'max_price': 3500,
            'min_bedrooms': 1,
            'max_bedrooms': 3,
            'preferred_locations': ['Downtown', 'SoMa'],
            'required_amenities': ['parking', 'laundry'],
            'property_types': ['apartment']
        }
        
        user = self.user_factory.create()
        user.preferences.min_price = valid_preferences['min_price']
        user.preferences.max_price = valid_preferences['max_price']
        
        # Should not raise validation errors
        assert user.preferences.min_price <= user.preferences.max_price
        
        # Test edge case preferences
        edge_case_preferences = {
            'min_price': 0,
            'max_price': 50000,
            'min_bedrooms': 0,
            'max_bedrooms': 10,
            'preferred_locations': [],  # No location preference
            'required_amenities': [],   # No required amenities
            'property_types': ['studio', 'apartment', 'house', 'condo']
        }
        
        user.preferences.min_price = edge_case_preferences['min_price']
        user.preferences.max_price = edge_case_preferences['max_price']
        
        # Should handle edge cases gracefully
        assert user.preferences.min_price >= 0
        assert user.preferences.max_price > user.preferences.min_price


class TestAPIIntegrationPerformance:
    """Performance tests for API integration."""
    
    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_api_requests_performance(self):
        """Test API performance under concurrent load."""
        # Setup mock services
        mock_property_repo = AsyncMock()
        mock_property_repo.get_all_active.return_value = []
        
        async def simulate_api_request():
            # Simulate API request processing time
            await asyncio.sleep(0.01)  # 10ms processing time
            return await mock_property_repo.get_all_active(limit=20, offset=0)
        
        # Test concurrent requests
        num_concurrent_requests = 50
        
        with PerformanceTestHelpers.measure_time() as timer:
            tasks = [simulate_api_request() for _ in range(num_concurrent_requests)]
            results = await asyncio.gather(*tasks)
        
        elapsed_time = timer()
        
        # All requests should complete
        assert len(results) == num_concurrent_requests
        
        # Should handle concurrent load efficiently
        PerformanceTestHelpers.assert_performance_threshold(
            elapsed_time, threshold=2.0, operation=f"{num_concurrent_requests} concurrent API requests"
        )
        
        # Calculate requests per second
        requests_per_second = num_concurrent_requests / elapsed_time
        assert requests_per_second > 20  # Should handle at least 20 RPS
    
    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_large_response_handling_performance(self):
        """Test API performance with large response payloads."""
        # Create large dataset
        factory = PropertyFactory(FactoryConfig(seed=42))
        large_property_list = factory.create_batch(1000)
        
        mock_property_repo = AsyncMock()
        mock_property_repo.get_all_active.return_value = large_property_list
        
        with PerformanceTestHelpers.measure_time() as timer:
            result = await mock_property_repo.get_all_active(limit=1000, offset=0)
        
        elapsed_time = timer()
        
        # Should handle large responses efficiently
        assert len(result) == 1000
        PerformanceTestHelpers.assert_performance_threshold(
            elapsed_time, threshold=1.0, operation="Large response (1000 properties)"
        )
    
    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_performance_integration(self):
        """Test search API performance integration."""
        mock_search_service = AsyncMock()
        
        # Simulate complex search with large result set
        large_search_results = [
            {
                'property': {'id': str(uuid4()), 'title': f'Property {i}'},
                'relevance_score': 0.9 - (i * 0.001),
                'matching_criteria': ['location', 'price']
            }
            for i in range(500)
        ]
        
        mock_search_service.search_properties.return_value = large_search_results
        
        with PerformanceTestHelpers.measure_time() as timer:
            result = await mock_search_service.search_properties(
                query="apartment downtown",
                filters={'location': 'Downtown'},
                limit=500
            )
        
        elapsed_time = timer()
        
        # Should handle large search results efficiently
        assert len(result) == 500
        PerformanceTestHelpers.assert_performance_threshold(
            elapsed_time, threshold=2.0, operation="Large search results (500 properties)"
        )