from typing import List, Optional, Tuple
from uuid import UUID

from ..entities.property import Property
from ..entities.search_query import SearchQuery, SearchFilters
from ..repositories.property_repository import PropertyRepository
from ..repositories.model_repository import ModelRepository


class SearchService:
    def __init__(self, property_repository: PropertyRepository, model_repository: ModelRepository):
        self.property_repository = property_repository
        self.model_repository = model_repository
    
    async def search_properties(self, query: SearchQuery) -> Tuple[List[Property], int]:
        """
        Search for properties based on query and filters
        Returns (properties, total_count)
        """
        # Validate query
        if not query.query_text.strip() and not self._has_meaningful_filters(query.filters):
            raise ValueError("Search query must have text or meaningful filters")
        
        # Get properties from repository
        properties, total_count = await self.property_repository.search(query)
        
        # If we have a text query, we might want to rank results using ML
        if query.query_text.strip():
            # Check if we have cached ML rankings
            cache_key = f"search_ranking:{hash(query.query_text)}:{query.limit}:{query.offset}"
            cached_rankings = await self.model_repository.get_cached_predictions(cache_key)
            
            if cached_rankings:
                # Use cached rankings to reorder results
                properties = self._apply_cached_rankings(properties, cached_rankings)
            else:
                # For now, return basic results - ML ranking would be implemented later
                pass
        
        return properties, total_count
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        # This would typically use a suggestion service or cache
        # For now, return empty list - would be implemented with autocomplete logic
        return []
    
    async def get_popular_searches(self, limit: int = 10) -> List[str]:
        """Get popular search terms"""
        # This would typically be based on analytics/logs
        # For now, return some common searches
        return [
            "apartment near downtown",
            "2 bedroom with parking",
            "pet friendly",
            "luxury apartments",
            "studio apartment",
            "apartments with gym",
            "furnished apartment",
            "apartments with pool"
        ][:limit]
    
    def _has_meaningful_filters(self, filters: SearchFilters) -> bool:
        """Check if filters contain meaningful search criteria"""
        return (
            filters.min_price is not None or
            filters.max_price is not None or
            filters.min_bedrooms is not None or
            filters.max_bedrooms is not None or
            filters.min_bathrooms is not None or
            filters.max_bathrooms is not None or
            len(filters.locations) > 0 or
            len(filters.amenities) > 0 or
            len(filters.property_types) > 0 or
            filters.min_square_feet is not None or
            filters.max_square_feet is not None
        )
    
    def _apply_cached_rankings(self, properties: List[Property], rankings: dict) -> List[Property]:
        """Apply cached ML rankings to reorder properties"""
        # Create a mapping of property ID to ranking score
        property_scores = {}
        for prop in properties:
            property_scores[str(prop.id)] = rankings.get(str(prop.id), 0.0)
        
        # Sort properties by ranking score
        return sorted(properties, key=lambda p: property_scores.get(str(p.id), 0.0), reverse=True)
    
    async def validate_search_query(self, query: SearchQuery) -> List[str]:
        """Validate search query and return any validation errors"""
        errors = []
        
        # Check query text length
        if len(query.query_text) > 500:
            errors.append("Search query too long (max 500 characters)")
        
        # Check filters
        if query.filters.min_price is not None and query.filters.min_price < 0:
            errors.append("Minimum price cannot be negative")
        
        if query.filters.max_price is not None and query.filters.max_price < 0:
            errors.append("Maximum price cannot be negative")
        
        if (query.filters.min_price is not None and query.filters.max_price is not None and
            query.filters.min_price > query.filters.max_price):
            errors.append("Minimum price cannot be greater than maximum price")
        
        if query.filters.min_bedrooms is not None and query.filters.min_bedrooms < 0:
            errors.append("Minimum bedrooms cannot be negative")
        
        if query.filters.max_bedrooms is not None and query.filters.max_bedrooms < 0:
            errors.append("Maximum bedrooms cannot be negative")
        
        if (query.filters.min_bedrooms is not None and query.filters.max_bedrooms is not None and
            query.filters.min_bedrooms > query.filters.max_bedrooms):
            errors.append("Minimum bedrooms cannot be greater than maximum bedrooms")
        
        # Check limit and offset
        if query.limit <= 0 or query.limit > 100:
            errors.append("Limit must be between 1 and 100")
        
        if query.offset < 0:
            errors.append("Offset cannot be negative")
        
        return errors