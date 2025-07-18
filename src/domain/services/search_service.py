from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID
import asyncio
import math
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
from dataclasses import dataclass, asdict

from ..entities.property import Property
from ..entities.search_query import SearchQuery, SearchFilters
from ..repositories.property_repository import PropertyRepository
from ..repositories.model_repository import ModelRepository


@dataclass
class SearchResult:
    """Search result with relevance scoring."""
    property: Property
    relevance_score: float
    ranking_factors: Dict[str, float]
    

@dataclass
class SearchAnalytics:
    """Search analytics data."""
    search_id: UUID
    user_id: Optional[UUID]
    query_text: str
    filters_applied: Dict[str, Any]
    results_count: int
    search_duration_ms: float
    clicked_properties: List[UUID]
    timestamp: datetime
    result_positions: Dict[str, int]
    

class SearchService:
    """Advanced Search Service with ranking, caching, and analytics."""
    
    def __init__(self, property_repository: PropertyRepository, model_repository: ModelRepository):
        self.property_repository = property_repository
        self.model_repository = model_repository
        self.logger = logging.getLogger(__name__)
        
        # Search analytics storage
        self.search_analytics: List[SearchAnalytics] = []
        
        # Popular search terms cache
        self._popular_searches_cache = None
        self._popular_searches_cache_time = None
        self._cache_ttl = 3600  # 1 hour
        
        # Text processing patterns
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
    async def search_properties(self, query: SearchQuery) -> Tuple[List[Property], int, Dict[str, Any]]:
        """
        Search for properties with advanced ranking and analytics.
        Returns (properties, total_count, search_metadata)
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate query
            validation_errors = await self.validate_search_query(query)
            if validation_errors:
                raise ValueError(f"Invalid search query: {', '.join(validation_errors)}")
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_result = await self._get_cached_search_result(cache_key)
            
            if cached_result:
                self.logger.info(f"Cache hit for search query: {query.query_text[:50]}")
                await self._track_search_analytics(query, cached_result, start_time, cache_hit=True)
                return cached_result
            
            # Perform search
            raw_properties, total_count = await self.property_repository.search(query)
            
            # Apply advanced ranking if we have text query
            if query.query_text.strip():
                ranked_results = await self._rank_search_results(raw_properties, query)
                properties = [result.property for result in ranked_results]
                ranking_metadata = {
                    'ranking_applied': True,
                    'ranking_factors': [result.ranking_factors for result in ranked_results[:10]]
                }
            else:
                properties = await self._apply_filter_based_ranking(raw_properties, query)
                ranking_metadata = {'ranking_applied': False, 'sort_by': query.sort_by}
            
            # Apply sorting if specified
            properties = await self._apply_sorting(properties, query.sort_by)
            
            # Prepare search metadata
            search_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            search_metadata = {
                'total_count': total_count,
                'search_duration_ms': search_duration,
                'cache_hit': False,
                'query_processed': self._preprocess_query_text(query.query_text),
                **ranking_metadata
            }
            
            # Cache the result
            result = (properties, total_count, search_metadata)
            await self._cache_search_result(cache_key, result)
            
            # Track analytics
            await self._track_search_analytics(query, result, start_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query.query_text}': {e}")
            error_metadata = {
                'error': str(e),
                'search_duration_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            return [], 0, error_metadata
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get intelligent search suggestions based on partial query."""
        try:
            suggestions = []
            normalized_query = partial_query.lower().strip()
            
            if len(normalized_query) < 2:
                return []
            
            # Get suggestions from multiple sources
            popular_suggestions = await self._get_popular_query_suggestions(normalized_query, limit)
            location_suggestions = await self._get_location_suggestions(normalized_query, limit)
            amenity_suggestions = await self._get_amenity_suggestions(normalized_query, limit)
            
            # Combine and rank suggestions
            all_suggestions = []
            all_suggestions.extend([(s, 'popular') for s in popular_suggestions])
            all_suggestions.extend([(s, 'location') for s in location_suggestions])
            all_suggestions.extend([(s, 'amenity') for s in amenity_suggestions])
            
            # Remove duplicates and score
            seen = set()
            for suggestion, source in all_suggestions:
                if suggestion not in seen and len(suggestions) < limit:
                    suggestions.append(suggestion)
                    seen.add(suggestion)
            
            return suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    async def get_popular_searches(self, limit: int = 10, time_window_days: int = 7) -> List[Dict[str, Any]]:
        """Get popular search terms with analytics data."""
        try:
            # Check cache
            if (self._popular_searches_cache and 
                self._popular_searches_cache_time and
                (datetime.utcnow() - self._popular_searches_cache_time).seconds < self._cache_ttl):
                return self._popular_searches_cache[:limit]
            
            # Calculate popular searches from analytics
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            recent_searches = [
                analytics for analytics in self.search_analytics
                if analytics.timestamp >= cutoff_date and analytics.query_text.strip()
            ]
            
            # Count search frequency and calculate metrics
            search_counter = Counter()
            click_rates = defaultdict(list)
            
            for search in recent_searches:
                normalized_query = search.query_text.lower().strip()
                search_counter[normalized_query] += 1
                
                # Calculate click-through rate
                ctr = len(search.clicked_properties) / max(search.results_count, 1)
                click_rates[normalized_query].append(ctr)
            
            # Build popular searches with metrics
            popular_searches = []
            for query, count in search_counter.most_common(limit * 2):
                avg_ctr = sum(click_rates[query]) / len(click_rates[query]) if click_rates[query] else 0
                
                popular_searches.append({
                    'query': query,
                    'search_count': count,
                    'avg_click_through_rate': round(avg_ctr, 3),
                    'popularity_score': count * (1 + avg_ctr)  # Boost queries with higher CTR
                })
            
            # Sort by popularity score and cache
            popular_searches.sort(key=lambda x: x['popularity_score'], reverse=True)
            
            # Fallback to default searches if no analytics data
            if not popular_searches:
                popular_searches = [
                    {'query': 'apartment near downtown', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0},
                    {'query': '2 bedroom with parking', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0},
                    {'query': 'pet friendly', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0},
                    {'query': 'luxury apartments', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0},
                    {'query': 'studio apartment', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0},
                    {'query': 'apartments with gym', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0},
                    {'query': 'furnished apartment', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0},
                    {'query': 'apartments with pool', 'search_count': 0, 'avg_click_through_rate': 0, 'popularity_score': 0}
                ]
            
            # Cache results
            self._popular_searches_cache = popular_searches
            self._popular_searches_cache_time = datetime.utcnow()
            
            return popular_searches[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get popular searches: {e}")
            return []
    
    def _has_meaningful_filters(self, filters: SearchFilters) -> bool:
        """Check if filters contain meaningful search criteria."""
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
        """Apply cached ML rankings to reorder properties."""
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
    
    # Advanced Search Methods
    
    async def _rank_search_results(self, properties: List[Property], query: SearchQuery) -> List[SearchResult]:
        """Apply advanced ranking algorithm to search results."""
        query_terms = self._preprocess_query_text(query.query_text)
        results = []
        
        for prop in properties:
            # Calculate various relevance factors
            text_relevance = self._calculate_text_relevance(prop, query_terms)
            location_relevance = self._calculate_location_relevance(prop, query)
            price_relevance = self._calculate_price_relevance(prop, query)
            amenity_relevance = self._calculate_amenity_relevance(prop, query)
            freshness_score = self._calculate_freshness_score(prop)
            
            # Combine relevance factors with weights
            ranking_factors = {
                'text_relevance': text_relevance,
                'location_relevance': location_relevance,
                'price_relevance': price_relevance,
                'amenity_relevance': amenity_relevance,
                'freshness_score': freshness_score
            }
            
            # Calculate overall relevance score
            relevance_score = (
                text_relevance * 0.35 +
                location_relevance * 0.25 +
                price_relevance * 0.15 +
                amenity_relevance * 0.15 +
                freshness_score * 0.10
            )
            
            results.append(SearchResult(
                property=prop,
                relevance_score=relevance_score,
                ranking_factors=ranking_factors
            ))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    async def _apply_filter_based_ranking(self, properties: List[Property], query: SearchQuery) -> List[Property]:
        """Apply ranking when no text query is provided."""
        # For filter-only queries, apply basic ranking based on property quality indicators
        
        def calculate_quality_score(prop: Property) -> float:
            score = 0.0
            
            # Boost properties with more amenities
            score += len(prop.amenities) * 0.1
            
            # Boost properties with images
            score += len(prop.images) * 0.05
            
            # Boost properties with complete information
            if prop.square_feet:
                score += 0.2
            
            # Boost newer listings
            days_old = (datetime.utcnow() - prop.scraped_at).days
            if days_old < 7:
                score += 0.3
            elif days_old < 30:
                score += 0.1
            
            return score
        
        # Sort by quality score
        return sorted(properties, key=calculate_quality_score, reverse=True)
    
    async def _apply_sorting(self, properties: List[Property], sort_by: str) -> List[Property]:
        """Apply user-specified sorting to properties."""
        if sort_by == 'price_asc':
            return sorted(properties, key=lambda p: p.price)
        elif sort_by == 'price_desc':
            return sorted(properties, key=lambda p: p.price, reverse=True)
        elif sort_by == 'date_new':
            return sorted(properties, key=lambda p: p.scraped_at, reverse=True)
        elif sort_by == 'date_old':
            return sorted(properties, key=lambda p: p.scraped_at)
        else:  # relevance or default
            return properties  # Already sorted by relevance
    
    def _preprocess_query_text(self, query_text: str) -> List[str]:
        """Preprocess query text for better matching."""
        if not query_text:
            return []
        
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_text.lower())
        
        # Split into words and remove stop words
        words = [word.strip() for word in normalized.split() if word.strip()]
        meaningful_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return meaningful_words
    
    def _calculate_text_relevance(self, property: Property, query_terms: List[str]) -> float:
        """Calculate text relevance score between property and query terms."""
        if not query_terms:
            return 0.0
        
        # Get property text fields
        title_text = property.title.lower()
        desc_text = property.description.lower()
        location_text = property.location.lower()
        amenities_text = ' '.join(property.amenities).lower()
        
        score = 0.0
        total_terms = len(query_terms)
        
        for term in query_terms:
            term_score = 0.0
            
            # Title matches (highest weight)
            if term in title_text:
                term_score += 1.0
            
            # Description matches
            if term in desc_text:
                term_score += 0.6
            
            # Location matches
            if term in location_text:
                term_score += 0.8
            
            # Amenity matches
            if term in amenities_text:
                term_score += 0.4
            
            score += term_score
        
        # Normalize by number of terms
        return min(score / total_terms, 1.0)
    
    def _calculate_location_relevance(self, property: Property, query: SearchQuery) -> float:
        """Calculate location-based relevance."""
        if not query.filters.locations:
            return 1.0  # No location filter, all equally relevant
        
        property_location = property.location.lower()
        
        for filter_location in query.filters.locations:
            if filter_location.lower() in property_location:
                return 1.0
        
        return 0.5  # Partial match or no match
    
    def _calculate_price_relevance(self, property: Property, query: SearchQuery) -> float:
        """Calculate price-based relevance."""
        if not query.filters.min_price and not query.filters.max_price:
            return 1.0
        
        score = 1.0
        
        # Penalty for being significantly outside price range
        if query.filters.min_price and property.price < query.filters.min_price * 0.8:
            score *= 0.7
        
        if query.filters.max_price and property.price > query.filters.max_price * 1.2:
            score *= 0.7
        
        return score
    
    def _calculate_amenity_relevance(self, property: Property, query: SearchQuery) -> float:
        """Calculate amenity-based relevance."""
        if not query.filters.amenities:
            return 1.0
        
        property_amenities = set(amenity.lower() for amenity in property.amenities)
        query_amenities = set(amenity.lower() for amenity in query.filters.amenities)
        
        if not query_amenities:
            return 1.0
        
        # Calculate overlap
        overlap = len(property_amenities.intersection(query_amenities))
        return overlap / len(query_amenities)
    
    def _calculate_freshness_score(self, property: Property) -> float:
        """Calculate freshness score based on when property was scraped."""
        days_old = (datetime.utcnow() - property.scraped_at).days
        
        if days_old <= 1:
            return 1.0
        elif days_old <= 7:
            return 0.8
        elif days_old <= 30:
            return 0.6
        elif days_old <= 90:
            return 0.4
        else:
            return 0.2
    
    # Caching Methods
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        return f"search_v2:{query.get_cache_key()}"
    
    async def _get_cached_search_result(self, cache_key: str) -> Optional[Tuple[List[Property], int, Dict[str, Any]]]:
        """Get cached search result."""
        try:
            cached_data = await self.model_repository.get_cached_predictions(cache_key)
            if cached_data:
                # Deserialize cached result
                return cached_data.get('result')
        except Exception as e:
            self.logger.warning(f"Failed to get cached search result: {e}")
        return None
    
    async def _cache_search_result(self, cache_key: str, result: Tuple[List[Property], int, Dict[str, Any]], ttl: int = 300) -> None:
        """Cache search result."""
        try:
            # Serialize result for caching
            cache_data = {
                'result': result,
                'cached_at': datetime.utcnow().isoformat()
            }
            await self.model_repository.cache_predictions(cache_key, cache_data, ttl)
        except Exception as e:
            self.logger.warning(f"Failed to cache search result: {e}")
    
    # Analytics Methods
    
    async def _track_search_analytics(self, query: SearchQuery, result: Tuple[List[Property], int, Dict[str, Any]], 
                                    start_time: datetime, cache_hit: bool = False) -> None:
        """Track search analytics."""
        try:
            properties, total_count, metadata = result
            
            # Create analytics record
            analytics = SearchAnalytics(
                search_id=query.id,
                user_id=query.user_id,
                query_text=query.query_text,
                filters_applied=asdict(query.filters),
                results_count=total_count,
                search_duration_ms=metadata.get('search_duration_ms', 0),
                clicked_properties=[],  # Will be updated when clicks are tracked
                timestamp=start_time,
                result_positions={str(prop.id): idx for idx, prop in enumerate(properties[:20])}
            )
            
            # Store analytics (in-memory for now, would use analytics warehouse in production)
            self.search_analytics.append(analytics)
            
            # Keep only recent analytics (last 10000 searches)
            if len(self.search_analytics) > 10000:
                self.search_analytics = self.search_analytics[-10000:]
            
        except Exception as e:
            self.logger.error(f"Failed to track search analytics: {e}")
    
    async def track_property_click(self, search_id: UUID, property_id: UUID, position: int) -> None:
        """Track when a user clicks on a property from search results."""
        try:
            # Find the corresponding search analytics record
            for analytics in reversed(self.search_analytics):  # Start from most recent
                if analytics.search_id == search_id:
                    analytics.clicked_properties.append(property_id)
                    break
            
            self.logger.info(f"Tracked property click: search_id={search_id}, property_id={property_id}, position={position}")
            
        except Exception as e:
            self.logger.error(f"Failed to track property click: {e}")
    
    # Suggestion Methods
    
    async def _get_popular_query_suggestions(self, partial_query: str, limit: int) -> List[str]:
        """Get suggestions from popular search queries."""
        suggestions = []
        
        # Get recent popular searches
        popular_searches = await self.get_popular_searches(limit * 2)
        
        for search_data in popular_searches:
            query = search_data['query']
            if partial_query in query.lower() and query not in suggestions:
                suggestions.append(query)
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    async def _get_location_suggestions(self, partial_query: str, limit: int) -> List[str]:
        """Get location-based suggestions."""
        # Common locations that might match the partial query
        common_locations = [
            'downtown', 'city center', 'uptown', 'midtown', 'financial district',
            'near university', 'near subway', 'near park', 'waterfront', 'historic district'
        ]
        
        suggestions = []
        for location in common_locations:
            if partial_query in location.lower():
                suggestions.append(f"apartments {location}")
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    async def _get_amenity_suggestions(self, partial_query: str, limit: int) -> List[str]:
        """Get amenity-based suggestions."""
        common_amenities = [
            'parking', 'gym', 'pool', 'pet friendly', 'furnished', 'balcony',
            'dishwasher', 'washer dryer', 'air conditioning', 'heating'
        ]
        
        suggestions = []
        for amenity in common_amenities:
            if partial_query in amenity.lower():
                suggestions.append(f"apartments with {amenity}")
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    # Analytics and Reporting Methods
    
    async def get_search_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get search analytics summary for the specified time period."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            recent_searches = [
                analytics for analytics in self.search_analytics
                if analytics.timestamp >= cutoff_date
            ]
            
            if not recent_searches:
                return {
                    'total_searches': 0,
                    'unique_users': 0,
                    'avg_results_per_search': 0,
                    'avg_search_duration_ms': 0,
                    'top_queries': [],
                    'click_through_rate': 0
                }
            
            # Calculate metrics
            total_searches = len(recent_searches)
            unique_users = len(set(s.user_id for s in recent_searches if s.user_id))
            avg_results = sum(s.results_count for s in recent_searches) / total_searches
            avg_duration = sum(s.search_duration_ms for s in recent_searches) / total_searches
            
            # Calculate click-through rate
            searches_with_clicks = sum(1 for s in recent_searches if s.clicked_properties)
            ctr = searches_with_clicks / total_searches if total_searches > 0 else 0
            
            # Top queries
            query_counter = Counter(s.query_text for s in recent_searches if s.query_text.strip())
            top_queries = [{'query': q, 'count': c} for q, c in query_counter.most_common(10)]
            
            return {
                'total_searches': total_searches,
                'unique_users': unique_users,
                'avg_results_per_search': round(avg_results, 2),
                'avg_search_duration_ms': round(avg_duration, 2),
                'top_queries': top_queries,
                'click_through_rate': round(ctr, 3)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get search analytics summary: {e}")
            return {}
    
    async def optimize_search_performance(self) -> Dict[str, Any]:
        """Analyze search performance and suggest optimizations."""
        try:
            analytics_summary = await self.get_search_analytics_summary()
            
            recommendations = []
            
            # Check average search duration
            avg_duration = analytics_summary.get('avg_search_duration_ms', 0)
            if avg_duration > 1000:  # More than 1 second
                recommendations.append({
                    'type': 'performance',
                    'message': f'Average search duration is {avg_duration:.0f}ms. Consider optimizing database queries or adding more caching.',
                    'priority': 'high'
                })
            
            # Check click-through rate
            ctr = analytics_summary.get('click_through_rate', 0)
            if ctr < 0.3:  # Less than 30% CTR
                recommendations.append({
                    'type': 'relevance',
                    'message': f'Click-through rate is {ctr:.1%}. Consider improving search ranking algorithm.',
                    'priority': 'medium'
                })
            
            # Check for queries with no results
            no_result_queries = [
                analytics for analytics in self.search_analytics
                if analytics.results_count == 0 and analytics.timestamp >= datetime.utcnow() - timedelta(days=7)
            ]
            
            if no_result_queries:
                recommendations.append({
                    'type': 'coverage',
                    'message': f'{len(no_result_queries)} searches returned no results in the last 7 days.',
                    'priority': 'medium'
                })
            
            return {
                'analytics_summary': analytics_summary,
                'recommendations': recommendations,
                'cache_hit_rate': 'Not tracked in this implementation',
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize search performance: {e}")
            return {'error': str(e)}