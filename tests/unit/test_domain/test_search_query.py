"""
Unit tests for the SearchQuery domain entity and SearchFilters.

Tests SearchQuery, SearchFilters creation, validation, and methods.
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4
from typing import List, Dict

from domain.entities.search_query import SearchQuery, SearchFilters
from tests.utils.data_factories import SearchQueryFactory, FactoryConfig


class TestSearchFilters:
    """Test cases for SearchFilters entity."""
    
    def test_search_filters_creation_with_defaults(self):
        """Test SearchFilters creation with default values."""
        filters = SearchFilters()
        
        assert filters.min_price is None
        assert filters.max_price is None
        assert filters.min_bedrooms is None
        assert filters.max_bedrooms is None
        assert filters.min_bathrooms is None
        assert filters.max_bathrooms is None
        assert filters.locations == []
        assert filters.amenities == []
        assert filters.property_types == []
        assert filters.min_square_feet is None
        assert filters.max_square_feet is None
    
    def test_search_filters_creation_with_values(self):
        """Test SearchFilters creation with specific values."""
        filters = SearchFilters(
            min_price=2000.0,
            max_price=5000.0,
            min_bedrooms=2,
            max_bedrooms=4,
            min_bathrooms=1.5,
            max_bathrooms=3.0,
            locations=["Downtown", "Mission"],
            amenities=["parking", "gym"],
            property_types=["apartment", "condo"],
            min_square_feet=800,
            max_square_feet=2000
        )
        
        assert filters.min_price == 2000.0
        assert filters.max_price == 5000.0
        assert filters.min_bedrooms == 2
        assert filters.max_bedrooms == 4
        assert filters.min_bathrooms == 1.5
        assert filters.max_bathrooms == 3.0
        assert filters.locations == ["Downtown", "Mission"]
        assert filters.amenities == ["parking", "gym"]
        assert filters.property_types == ["apartment", "condo"]
        assert filters.min_square_feet == 800
        assert filters.max_square_feet == 2000
    
    def test_search_filters_post_init(self):
        """Test SearchFilters __post_init__ method."""
        # Test with None values - should be converted to empty lists
        filters = SearchFilters(
            min_price=1000.0,
            locations=None,
            amenities=None,
            property_types=None
        )
        
        assert filters.min_price == 1000.0
        assert filters.locations == []
        assert filters.amenities == []
        assert filters.property_types == []
    
    def test_search_filters_partial_specification(self):
        """Test SearchFilters with partial specification."""
        filters = SearchFilters(
            min_price=1500.0,
            max_bedrooms=3,
            locations=["SoMa"],
            amenities=["parking"]
        )
        
        assert filters.min_price == 1500.0
        assert filters.max_price is None
        assert filters.min_bedrooms is None
        assert filters.max_bedrooms == 3
        assert filters.locations == ["SoMa"]
        assert filters.amenities == ["parking"]
        assert filters.property_types == []
    
    def test_search_filters_edge_cases(self):
        """Test edge cases for SearchFilters."""
        # Zero values
        filters = SearchFilters(
            min_price=0.0,
            max_price=0.0,
            min_bedrooms=0,
            max_bedrooms=0,
            min_square_feet=0,
            max_square_feet=0
        )
        
        assert filters.min_price == 0.0
        assert filters.max_price == 0.0
        assert filters.min_bedrooms == 0
        assert filters.max_bedrooms == 0
        assert filters.min_square_feet == 0
        assert filters.max_square_feet == 0
        
        # Negative values (currently allowed)
        filters = SearchFilters(
            min_price=-100.0,
            min_bedrooms=-1,
            min_square_feet=-500
        )
        
        assert filters.min_price == -100.0
        assert filters.min_bedrooms == -1
        assert filters.min_square_feet == -500
        
        # Very large values
        filters = SearchFilters(
            max_price=1000000.0,
            max_bedrooms=100,
            max_square_feet=50000
        )
        
        assert filters.max_price == 1000000.0
        assert filters.max_bedrooms == 100
        assert filters.max_square_feet == 50000


class TestSearchQuery:
    """Test cases for SearchQuery entity."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.factory = SearchQueryFactory(FactoryConfig(seed=42))
    
    def test_search_query_creation_with_factory(self):
        """Test search query creation using factory."""
        query = self.factory.create()
        
        assert isinstance(query, SearchQuery)
        assert isinstance(query.id, UUID)
        assert query.query_text
        assert isinstance(query.filters, SearchFilters)
        assert isinstance(query.created_at, datetime)
        assert query.limit > 0
        assert query.offset >= 0
        assert query.sort_by in ["relevance", "price", "price_desc", "bedrooms", "date_added"]
    
    def test_search_query_create_class_method(self):
        """Test SearchQuery.create() class method."""
        user_id = uuid4()
        filters = SearchFilters(
            min_price=2000.0,
            max_price=4000.0,
            locations=["Downtown"]
        )
        
        query = SearchQuery.create(
            query_text="luxury apartment downtown",
            user_id=user_id,
            filters=filters,
            limit=20,
            offset=10,
            sort_by="price_desc"
        )
        
        assert query.query_text == "luxury apartment downtown"
        assert query.user_id == user_id
        assert query.filters == filters
        assert query.limit == 20
        assert query.offset == 10
        assert query.sort_by == "price_desc"
        assert isinstance(query.id, UUID)
        assert isinstance(query.created_at, datetime)
    
    def test_search_query_create_with_defaults(self):
        """Test SearchQuery.create() with default values."""
        query = SearchQuery.create(query_text="apartment")
        
        assert query.query_text == "apartment"
        assert query.user_id is None
        assert isinstance(query.filters, SearchFilters)
        assert query.limit == 50
        assert query.offset == 0
        assert query.sort_by == "relevance"
        assert isinstance(query.id, UUID)
        assert isinstance(query.created_at, datetime)
    
    def test_search_query_create_minimal(self):
        """Test SearchQuery.create() with minimal parameters."""
        query = SearchQuery.create("")
        
        assert query.query_text == ""
        assert query.user_id is None
        assert isinstance(query.filters, SearchFilters)
        assert query.filters.locations == []
        assert query.filters.amenities == []
        assert query.filters.property_types == []
    
    def test_get_normalized_query(self):
        """Test query text normalization."""
        # Test basic normalization
        query = SearchQuery.create("  Luxury APARTMENT Downtown  ")
        normalized = query.get_normalized_query()
        assert normalized == "luxury apartment downtown"
        
        # Test empty query
        query = SearchQuery.create("")
        normalized = query.get_normalized_query()
        assert normalized == ""
        
        # Test query with special characters
        query = SearchQuery.create("  2-BR Apt. w/ Parking!  ")
        normalized = query.get_normalized_query()
        assert normalized == "2-br apt. w/ parking!"
        
        # Test only whitespace
        query = SearchQuery.create("   ")
        normalized = query.get_normalized_query()
        assert normalized == ""
    
    def test_has_location_filter(self):
        """Test location filter detection."""
        # With location filter
        filters = SearchFilters(locations=["Downtown", "Mission"])
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_location_filter() is True
        
        # Without location filter (empty list)
        filters = SearchFilters(locations=[])
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_location_filter() is False
        
        # Without location filter (None)
        filters = SearchFilters(locations=None)  # Will be converted to []
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_location_filter() is False
        
        # Default filters (no locations)
        query = SearchQuery.create("apartment")
        assert query.has_location_filter() is False
    
    def test_has_price_filter(self):
        """Test price filter detection."""
        # With min price only
        filters = SearchFilters(min_price=2000.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_price_filter() is True
        
        # With max price only
        filters = SearchFilters(max_price=5000.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_price_filter() is True
        
        # With both min and max price
        filters = SearchFilters(min_price=2000.0, max_price=5000.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_price_filter() is True
        
        # With zero prices (should still be considered a filter)
        filters = SearchFilters(min_price=0.0, max_price=0.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_price_filter() is True
        
        # Without price filter
        filters = SearchFilters()
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_price_filter() is False
        
        # Default filters (no prices)
        query = SearchQuery.create("apartment")
        assert query.has_price_filter() is False
    
    def test_has_size_filter(self):
        """Test size filter detection."""
        # With min bedrooms only
        filters = SearchFilters(min_bedrooms=2)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_size_filter() is True
        
        # With max bedrooms only
        filters = SearchFilters(max_bedrooms=4)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_size_filter() is True
        
        # With min bathrooms only
        filters = SearchFilters(min_bathrooms=1.5)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_size_filter() is True
        
        # With max bathrooms only
        filters = SearchFilters(max_bathrooms=3.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_size_filter() is True
        
        # With multiple size filters
        filters = SearchFilters(min_bedrooms=2, max_bedrooms=4, min_bathrooms=1.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_size_filter() is True
        
        # With zero size filters (should still be considered a filter)
        filters = SearchFilters(min_bedrooms=0, max_bathrooms=0.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_size_filter() is True
        
        # Without size filters
        filters = SearchFilters()
        query = SearchQuery.create("apartment", filters=filters)
        assert query.has_size_filter() is False
        
        # Default filters (no sizes)
        query = SearchQuery.create("apartment")
        assert query.has_size_filter() is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        user_id = uuid4()
        filters = SearchFilters(
            min_price=2000.0,
            max_price=5000.0,
            min_bedrooms=2,
            max_bedrooms=4,
            locations=["Downtown", "Mission"],
            amenities=["parking", "gym"],
            property_types=["apartment"],
            min_square_feet=800,
            max_square_feet=2000
        )
        
        query = SearchQuery.create(
            query_text="luxury apartment",
            user_id=user_id,
            filters=filters,
            limit=25,
            offset=5,
            sort_by="price_desc"
        )
        
        result_dict = query.to_dict()
        
        # Check top-level fields
        assert result_dict["id"] == str(query.id)
        assert result_dict["user_id"] == str(user_id)
        assert result_dict["query_text"] == "luxury apartment"
        assert result_dict["limit"] == 25
        assert result_dict["offset"] == 5
        assert result_dict["sort_by"] == "price_desc"
        assert "created_at" in result_dict
        
        # Check filters
        filters_dict = result_dict["filters"]
        assert filters_dict["min_price"] == 2000.0
        assert filters_dict["max_price"] == 5000.0
        assert filters_dict["min_bedrooms"] == 2
        assert filters_dict["max_bedrooms"] == 4
        assert filters_dict["locations"] == ["Downtown", "Mission"]
        assert filters_dict["amenities"] == ["parking", "gym"]
        assert filters_dict["property_types"] == ["apartment"]
        assert filters_dict["min_square_feet"] == 800
        assert filters_dict["max_square_feet"] == 2000
        
        # Check None values
        assert filters_dict["min_bathrooms"] is None
        assert filters_dict["max_bathrooms"] is None
    
    def test_to_dict_with_none_user_id(self):
        """Test to_dict() with None user_id."""
        query = SearchQuery.create("apartment")
        result_dict = query.to_dict()
        
        assert result_dict["user_id"] is None
    
    def test_to_dict_with_empty_filters(self):
        """Test to_dict() with empty filters."""
        query = SearchQuery.create("apartment")
        result_dict = query.to_dict()
        
        filters_dict = result_dict["filters"]
        assert filters_dict["min_price"] is None
        assert filters_dict["max_price"] is None
        assert filters_dict["min_bedrooms"] is None
        assert filters_dict["max_bedrooms"] is None
        assert filters_dict["min_bathrooms"] is None
        assert filters_dict["max_bathrooms"] is None
        assert filters_dict["locations"] == []
        assert filters_dict["amenities"] == []
        assert filters_dict["property_types"] == []
        assert filters_dict["min_square_feet"] is None
        assert filters_dict["max_square_feet"] is None
    
    def test_search_query_factory_specific_searches(self):
        """Test factory methods for specific search types."""
        # Budget search
        budget_query = self.factory.create_specific_search("budget_search")
        assert "affordable" in budget_query.query_text or "studio" in budget_query.query_text
        assert budget_query.sort_by == "price"
        
        # Luxury search
        luxury_query = self.factory.create_specific_search("luxury_search")
        assert "luxury" in luxury_query.query_text or "penthouse" in luxury_query.query_text
        assert luxury_query.sort_by == "price_desc"
        
        # Family search
        family_query = self.factory.create_specific_search("family_search")
        assert "family" in family_query.query_text or "house" in family_query.query_text
        
        # Empty search
        empty_query = self.factory.create_specific_search("empty_search")
        assert empty_query.query_text == ""
    
    def test_search_query_batch_creation(self):
        """Test creating multiple search queries."""
        queries = self.factory.create_batch(5)
        
        assert len(queries) == 5
        assert all(isinstance(query, SearchQuery) for query in queries)
        
        # Check that all queries have unique IDs
        query_ids = [query.id for query in queries]
        assert len(set(query_ids)) == 5
    
    def test_search_query_created_at_timestamp(self):
        """Test that created_at timestamp is recent."""
        query = self.factory.create()
        
        time_diff = datetime.now() - query.created_at
        assert time_diff.total_seconds() < 5  # Should be created within 5 seconds
    
    def test_search_query_sort_options(self):
        """Test various sort options."""
        sort_options = ["relevance", "price_asc", "price_desc", "date_new", "date_old"]
        
        for sort_option in sort_options:
            query = SearchQuery.create("apartment", sort_by=sort_option)
            assert query.sort_by == sort_option
    
    def test_search_query_pagination_parameters(self):
        """Test pagination parameters."""
        # Test different limit values
        limits = [1, 10, 50, 100, 1000]
        for limit in limits:
            query = SearchQuery.create("apartment", limit=limit)
            assert query.limit == limit
        
        # Test different offset values
        offsets = [0, 10, 50, 100, 1000]
        for offset in offsets:
            query = SearchQuery.create("apartment", offset=offset)
            assert query.offset == offset
        
        # Test large pagination
        query = SearchQuery.create("apartment", limit=1000, offset=5000)
        assert query.limit == 1000
        assert query.offset == 5000
    
    def test_search_query_edge_cases(self):
        """Test edge cases for search queries."""
        # Very long query text
        long_query = "a" * 10000
        query = SearchQuery.create(long_query)
        assert query.query_text == long_query
        
        # Query with special characters
        special_query = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        query = SearchQuery.create(special_query)
        assert query.query_text == special_query
        
        # Unicode query text
        unicode_query = "апартамент в центре города"
        query = SearchQuery.create(unicode_query)
        assert query.query_text == unicode_query
        
        # Zero limit (currently allowed)
        query = SearchQuery.create("apartment", limit=0)
        assert query.limit == 0
        
        # Negative offset (currently allowed)
        query = SearchQuery.create("apartment", offset=-10)
        assert query.offset == -10
        
        # Invalid sort option (currently allowed)
        query = SearchQuery.create("apartment", sort_by="invalid_sort")
        assert query.sort_by == "invalid_sort"
    
    def test_complex_search_filters(self):
        """Test complex filter combinations."""
        filters = SearchFilters(
            min_price=2000.0,
            max_price=8000.0,
            min_bedrooms=2,
            max_bedrooms=4,
            min_bathrooms=1.5,
            max_bathrooms=3.0,
            locations=["Downtown", "Mission", "SoMa", "Castro"],
            amenities=["parking", "gym", "pool", "rooftop", "concierge"],
            property_types=["apartment", "condo", "luxury_apartment"],
            min_square_feet=1000,
            max_square_feet=3000
        )
        
        query = SearchQuery.create(
            "luxury apartment with amenities",
            filters=filters
        )
        
        assert query.has_location_filter() is True
        assert query.has_price_filter() is True
        assert query.has_size_filter() is True
        
        # Check all filter methods work correctly
        assert len(query.filters.locations) == 4
        assert len(query.filters.amenities) == 5
        assert len(query.filters.property_types) == 3
        assert query.filters.min_price < query.filters.max_price
        assert query.filters.min_bedrooms < query.filters.max_bedrooms
        assert query.filters.min_bathrooms < query.filters.max_bathrooms
        assert query.filters.min_square_feet < query.filters.max_square_feet
    
    def test_search_query_immutability_concern(self):
        """Test potential immutability concerns with lists in filters."""
        original_locations = ["Downtown", "Mission"]
        original_amenities = ["parking", "gym"]
        
        filters = SearchFilters(
            locations=original_locations.copy(),
            amenities=original_amenities.copy()
        )
        
        query = SearchQuery.create("apartment", filters=filters)
        
        # Modify the original lists
        original_locations.append("SoMa")
        original_amenities.append("pool")
        
        # Query filters should not be affected
        assert len(query.filters.locations) == 2
        assert len(query.filters.amenities) == 2
        assert "SoMa" not in query.filters.locations
        assert "pool" not in query.filters.amenities
    
    def test_filter_validation_edge_cases(self):
        """Test edge cases that might need validation."""
        # Inverted price range (min > max)
        filters = SearchFilters(min_price=5000.0, max_price=2000.0)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.filters.min_price > query.filters.max_price  # Currently allowed
        
        # Inverted bedroom range
        filters = SearchFilters(min_bedrooms=4, max_bedrooms=2)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.filters.min_bedrooms > query.filters.max_bedrooms  # Currently allowed
        
        # Inverted square feet range
        filters = SearchFilters(min_square_feet=2000, max_square_feet=1000)
        query = SearchQuery.create("apartment", filters=filters)
        assert query.filters.min_square_feet > query.filters.max_square_feet  # Currently allowed
        
        # Very large lists
        large_locations = [f"Location_{i}" for i in range(1000)]
        large_amenities = [f"Amenity_{i}" for i in range(500)]
        
        filters = SearchFilters(
            locations=large_locations,
            amenities=large_amenities
        )
        query = SearchQuery.create("apartment", filters=filters)
        assert len(query.filters.locations) == 1000
        assert len(query.filters.amenities) == 500