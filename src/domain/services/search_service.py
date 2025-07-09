"""
Search Service - Handles search business logic for rental properties.
"""
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID
import re
from datetime import datetime, timedelta

from ..entities.property import Property
from ..entities.search_query import SearchQuery, SearchFilters
from ..entities.user import User, UserPreferences
from ..repositories.property_repository import PropertyRepository
from ..repositories.user_repository import UserRepository
from ..repositories.model_repository import ModelRepository


class SearchService:
    """Service for handling property search business logic."""
    
    def __init__(
        self,
        property_repository: PropertyRepository,
        user_repository: UserRepository,
        model_repository: ModelRepository
    ):
        self.property_repository = property_repository
        self.user_repository = user_repository
        self.model_repository = model_repository
    
    async def search_properties(
        self,
        query: SearchQuery,
        use_ml_ranking: bool = True
    ) -> Tuple[List[Property], int]:
        """
        Search for properties based on the query.
        
        Args:
            query: The search query with filters and parameters
            use_ml_ranking: Whether to use ML-based ranking for results
            
        Returns:
            Tuple of (properties, total_count)
        """
        # Validate search query
        self._validate_search_query(query)
        
        # Apply any query preprocessing
        processed_query = await self._preprocess_query(query)
        
        # Execute the search
        properties, total_count = await self.property_repository.search(processed_query)
        
        # Apply ML ranking if enabled
        if use_ml_ranking and properties:
            properties = await self._apply_ml_ranking(properties, query)
        
        # Track search query for analytics
        if query.user_id:
            await self._track_search_query(query)
        
        return properties, total_count
    
    async def search_with_user_preferences(
        self,
        user_id: UUID,
        query_text: str,
        additional_filters: Optional[SearchFilters] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Property], int]:
        """
        Search for properties using user preferences as base filters.
        
        Args:
            user_id: The user's ID
            query_text: Text search query
            additional_filters: Additional filters to apply
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (properties, total_count)
        """
        # Get user preferences
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Convert user preferences to search filters
        base_filters = self._convert_preferences_to_filters(user.preferences)
        
        # Merge with additional filters
        merged_filters = self._merge_filters(base_filters, additional_filters)
        
        # Create search query
        search_query = SearchQuery.create(
            query_text=query_text,
            user_id=user_id,
            filters=merged_filters,
            limit=limit,
            offset=offset
        )
        
        return await self.search_properties(search_query)
    
    async def get_similar_properties(
        self,
        property_id: UUID,
        limit: int = 10,
        user_id: Optional[UUID] = None
    ) -> List[Property]:
        """
        Get properties similar to the given property.
        
        Args:
            property_id: The reference property ID
            limit: Maximum number of similar properties to return
            user_id: Optional user ID for personalized similarity
            
        Returns:
            List of similar properties
        """
        # Get the reference property
        reference_property = await self.property_repository.get_by_id(property_id)
        if not reference_property:
            raise ValueError(f"Property with ID {property_id} not found")
        
        # Get similar properties using repository
        similar_properties = await self.property_repository.get_similar_properties(
            property_id, limit
        )
        
        # If user provided, personalize the similarity
        if user_id and similar_properties:
            similar_properties = await self._personalize_similarity(
                similar_properties, user_id
            )
        
        return similar_properties
    
    async def auto_complete_search(
        self,
        query_text: str,
        limit: int = 10
    ) -> List[str]:
        """
        Provide auto-complete suggestions for search queries.
        
        Args:
            query_text: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of search suggestions
        """
        suggestions = []
        
        # Location-based suggestions
        if len(query_text) >= 2:
            location_suggestions = await self._get_location_suggestions(query_text, limit // 2)
            suggestions.extend(location_suggestions)
        
        # Amenity-based suggestions
        amenity_suggestions = await self._get_amenity_suggestions(query_text, limit // 2)
        suggestions.extend(amenity_suggestions)
        
        # Remove duplicates and limit results
        unique_suggestions = list(dict.fromkeys(suggestions))[:limit]
        
        return unique_suggestions
    
    async def get_search_analytics(
        self,
        user_id: Optional[UUID] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get search analytics data.
        
        Args:
            user_id: Optional user ID for user-specific analytics
            days_back: Number of days to look back for analytics
            
        Returns:
            Analytics data dictionary
        """
        cache_key = f"search_analytics_{user_id}_{days_back}"
        
        # Try to get from cache first
        cached_analytics = await self.model_repository.get_cached_predictions(cache_key)
        if cached_analytics:
            return cached_analytics
        
        # Generate analytics
        analytics = {
            "total_searches": 0,
            "avg_results_per_search": 0,
            "most_common_filters": {},
            "popular_locations": [],
            "avg_price_range": {"min": 0, "max": 0},
            "search_trends": []
        }
        
        # Cache the results
        await self.model_repository.cache_predictions(
            cache_key, analytics, ttl_seconds=3600
        )
        
        return analytics
    
    def _validate_search_query(self, query: SearchQuery) -> None:
        """Validate search query parameters."""
        if not query.query_text and not any([
            query.filters.min_price,
            query.filters.max_price,
            query.filters.locations,
            query.filters.amenities,
            query.filters.property_types
        ]):
            raise ValueError("Query must have either text or filters")
        
        if query.filters.min_price and query.filters.max_price:
            if query.filters.min_price > query.filters.max_price:
                raise ValueError("min_price cannot be greater than max_price")
        
        if query.filters.min_bedrooms and query.filters.max_bedrooms:
            if query.filters.min_bedrooms > query.filters.max_bedrooms:
                raise ValueError("min_bedrooms cannot be greater than max_bedrooms")
        
        if query.limit < 1 or query.limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        
        if query.offset < 0:
            raise ValueError("offset must be non-negative")
    
    async def _preprocess_query(self, query: SearchQuery) -> SearchQuery:
        """Preprocess the search query for better results."""
        # Normalize query text
        normalized_text = self._normalize_query_text(query.query_text)
        
        # Extract implicit filters from query text
        implicit_filters = self._extract_implicit_filters(normalized_text)
        
        # Merge implicit filters with explicit filters
        merged_filters = self._merge_filters(query.filters, implicit_filters)
        
        # Create new query with processed data
        processed_query = SearchQuery(
            id=query.id,
            user_id=query.user_id,
            query_text=normalized_text,
            filters=merged_filters,
            created_at=query.created_at,
            limit=query.limit,
            offset=query.offset,
            sort_by=query.sort_by
        )
        
        return processed_query
    
    def _normalize_query_text(self, query_text: str) -> str:
        """Normalize query text for better matching."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query_text.strip())
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove special characters but keep spaces and alphanumeric
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        return normalized
    
    def _extract_implicit_filters(self, query_text: str) -> SearchFilters:
        """Extract implicit filters from query text."""
        filters = SearchFilters()
        
        # Extract price information
        price_patterns = [
            r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:to|-)?\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'under\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'over\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'below\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'above\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, query_text)
            if matches:
                # Handle different price pattern types
                if len(matches[0]) == 2:  # Range pattern
                    filters.min_price = float(matches[0][0].replace(',', ''))
                    filters.max_price = float(matches[0][1].replace(',', ''))
                elif 'under' in pattern or 'below' in pattern:
                    filters.max_price = float(matches[0].replace(',', ''))
                elif 'over' in pattern or 'above' in pattern:
                    filters.min_price = float(matches[0].replace(',', ''))
                break
        
        # Extract bedroom/bathroom information
        bedroom_match = re.search(r'(\d+)\s*(?:bed|bedroom|br)', query_text)
        if bedroom_match:
            filters.min_bedrooms = int(bedroom_match.group(1))
        
        bathroom_match = re.search(r'(\d+(?:\.\d)?)\s*(?:bath|bathroom|ba)', query_text)
        if bathroom_match:
            filters.min_bathrooms = float(bathroom_match.group(1))
        
        # Extract common amenities
        amenity_keywords = {
            'parking': ['parking', 'garage', 'car'],
            'pool': ['pool', 'swimming'],
            'gym': ['gym', 'fitness', 'workout'],
            'laundry': ['laundry', 'washer', 'dryer'],
            'pet': ['pet', 'dog', 'cat'],
            'balcony': ['balcony', 'deck', 'patio'],
            'ac': ['ac', 'air conditioning', 'cooling'],
            'heating': ['heating', 'heat']
        }
        
        found_amenities = []
        for amenity, keywords in amenity_keywords.items():
            if any(keyword in query_text for keyword in keywords):
                found_amenities.append(amenity)
        
        if found_amenities:
            filters.amenities = found_amenities
        
        return filters
    
    def _convert_preferences_to_filters(self, preferences: UserPreferences) -> SearchFilters:
        """Convert user preferences to search filters."""
        return SearchFilters(
            min_price=preferences.min_price,
            max_price=preferences.max_price,
            min_bedrooms=preferences.min_bedrooms,
            max_bedrooms=preferences.max_bedrooms,
            min_bathrooms=preferences.min_bathrooms,
            max_bathrooms=preferences.max_bathrooms,
            locations=preferences.preferred_locations.copy(),
            amenities=preferences.required_amenities.copy(),
            property_types=preferences.property_types.copy()
        )
    
    def _merge_filters(
        self,
        base_filters: SearchFilters,
        additional_filters: Optional[SearchFilters]
    ) -> SearchFilters:
        """Merge two sets of search filters."""
        if not additional_filters:
            return base_filters
        
        return SearchFilters(
            min_price=additional_filters.min_price or base_filters.min_price,
            max_price=additional_filters.max_price or base_filters.max_price,
            min_bedrooms=additional_filters.min_bedrooms or base_filters.min_bedrooms,
            max_bedrooms=additional_filters.max_bedrooms or base_filters.max_bedrooms,
            min_bathrooms=additional_filters.min_bathrooms or base_filters.min_bathrooms,
            max_bathrooms=additional_filters.max_bathrooms or base_filters.max_bathrooms,
            locations=list(set(base_filters.locations + additional_filters.locations)),
            amenities=list(set(base_filters.amenities + additional_filters.amenities)),
            property_types=list(set(base_filters.property_types + additional_filters.property_types)),
            min_square_feet=additional_filters.min_square_feet or base_filters.min_square_feet,
            max_square_feet=additional_filters.max_square_feet or base_filters.max_square_feet
        )
    
    async def _apply_ml_ranking(
        self,
        properties: List[Property],
        query: SearchQuery
    ) -> List[Property]:
        """Apply ML-based ranking to search results."""
        if not properties or not query.user_id:
            return properties
        
        # Get user for personalization
        user = await self.user_repository.get_by_id(query.user_id)
        if not user:
            return properties
        
        # Generate ranking scores (placeholder for ML model)
        scored_properties = []
        for prop in properties:
            score = await self._calculate_relevance_score(prop, query, user)
            scored_properties.append((prop, score))
        
        # Sort by score (descending)
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return [prop for prop, score in scored_properties]
    
    async def _calculate_relevance_score(
        self,
        property: Property,
        query: SearchQuery,
        user: User
    ) -> float:
        """Calculate relevance score for a property."""
        score = 0.0
        
        # Text relevance score
        text_score = self._calculate_text_relevance(property, query.query_text)
        score += text_score * 0.3
        
        # User preference score
        preference_score = self._calculate_preference_score(property, user.preferences)
        score += preference_score * 0.4
        
        # Interaction history score
        interaction_score = await self._calculate_interaction_score(property, user)
        score += interaction_score * 0.3
        
        return score
    
    def _calculate_text_relevance(self, property: Property, query_text: str) -> float:
        """Calculate text relevance score."""
        if not query_text:
            return 0.0
        
        property_text = property.get_full_text().lower()
        query_words = query_text.lower().split()
        
        # Simple word matching score
        matches = sum(1 for word in query_words if word in property_text)
        return matches / len(query_words) if query_words else 0.0
    
    def _calculate_preference_score(
        self,
        property: Property,
        preferences: UserPreferences
    ) -> float:
        """Calculate preference-based score."""
        score = 0.0
        
        # Price preference
        if preferences.min_price and preferences.max_price:
            if preferences.min_price <= property.price <= preferences.max_price:
                score += 0.3
        
        # Bedroom preference
        if preferences.min_bedrooms and preferences.max_bedrooms:
            if preferences.min_bedrooms <= property.bedrooms <= preferences.max_bedrooms:
                score += 0.2
        
        # Bathroom preference
        if preferences.min_bathrooms and preferences.max_bathrooms:
            if preferences.min_bathrooms <= property.bathrooms <= preferences.max_bathrooms:
                score += 0.2
        
        # Location preference
        if preferences.preferred_locations:
            if any(loc.lower() in property.location.lower() for loc in preferences.preferred_locations):
                score += 0.15
        
        # Amenity preference
        if preferences.required_amenities:
            matching_amenities = set(preferences.required_amenities) & set(property.amenities)
            score += (len(matching_amenities) / len(preferences.required_amenities)) * 0.15
        
        return score
    
    async def _calculate_interaction_score(
        self,
        property: Property,
        user: User
    ) -> float:
        """Calculate interaction-based score."""
        # Get user's interaction history
        interactions = await self.user_repository.get_interactions(user.id)
        
        # Get properties user has interacted with
        interacted_properties = await self.property_repository.get_by_ids(
            [i.property_id for i in interactions]
        )
        
        # Calculate similarity to previously liked properties
        liked_properties = [p for p in interacted_properties 
                          if any(i.interaction_type == 'like' and i.property_id == p.id 
                                for i in interactions)]
        
        if not liked_properties:
            return 0.0
        
        # Simple similarity based on property type and location
        similarities = []
        for liked_prop in liked_properties:
            similarity = 0.0
            
            # Property type similarity
            if property.property_type == liked_prop.property_type:
                similarity += 0.3
            
            # Location similarity
            if property.location == liked_prop.location:
                similarity += 0.4
            
            # Price similarity (normalized)
            price_diff = abs(property.price - liked_prop.price)
            max_price = max(property.price, liked_prop.price)
            if max_price > 0:
                price_similarity = 1 - (price_diff / max_price)
                similarity += price_similarity * 0.3
            
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    async def _personalize_similarity(
        self,
        properties: List[Property],
        user_id: UUID
    ) -> List[Property]:
        """Personalize similarity based on user preferences."""
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            return properties
        
        # Score properties based on user preferences
        scored_properties = []
        for prop in properties:
            score = self._calculate_preference_score(prop, user.preferences)
            scored_properties.append((prop, score))
        
        # Sort by score
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return [prop for prop, score in scored_properties]
    
    async def _get_location_suggestions(
        self,
        query_text: str,
        limit: int
    ) -> List[str]:
        """Get location-based suggestions."""
        # This would typically query a location database or service
        # For now, return some common location suggestions
        common_locations = [
            "Downtown", "Midtown", "Uptown", "Suburbs", "City Center",
            "Waterfront", "Historic District", "Business District"
        ]
        
        query_lower = query_text.lower()
        suggestions = [loc for loc in common_locations 
                      if query_lower in loc.lower()]
        
        return suggestions[:limit]
    
    async def _get_amenity_suggestions(
        self,
        query_text: str,
        limit: int
    ) -> List[str]:
        """Get amenity-based suggestions."""
        common_amenities = [
            "parking", "pool", "gym", "laundry", "pet-friendly",
            "balcony", "air conditioning", "heating", "dishwasher",
            "elevator", "security", "concierge"
        ]
        
        query_lower = query_text.lower()
        suggestions = [amenity for amenity in common_amenities 
                      if query_lower in amenity.lower()]
        
        return suggestions[:limit]
    
    async def _track_search_query(self, query: SearchQuery) -> None:
        """Track search query for analytics."""
        # This would typically save to an analytics database
        # For now, just cache the query
        cache_key = f"search_query_{query.id}"
        await self.model_repository.cache_predictions(
            cache_key, query.to_dict(), ttl_seconds=86400
        )