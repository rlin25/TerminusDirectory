"""
Search API router for property search functionality.

This module provides endpoints for property search, search suggestions,
and search analytics.
"""

import time
import logging
import asyncio
import math
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import json
import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ....domain.entities.search_query import SearchQuery, SearchFilters
from ....infrastructure.data import get_property_repository, get_cache_repository
from ....infrastructure.ml.models.search_ranker import NLPSearchRanker, create_search_ranker
from ....infrastructure.data_warehouse.analytics_warehouse import AnalyticsWarehouse
from ...dto.search_dto import (
    SearchRequest, SearchResponse, PropertyResponse,
    SearchSuggestionRequest, SearchSuggestionResponse,
    PopularSearchRequest, PopularSearchResponse,
    SearchErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize search ranker (will be loaded from config in production)
search_ranker = None
analytics_warehouse = None

# Search analytics tracking
search_analytics = {
    "total_searches": 0,
    "successful_searches": 0,
    "failed_searches": 0,
    "average_response_time": 0.0,
    "popular_queries": {},
    "last_updated": datetime.now()
}


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


def get_search_ranker() -> NLPSearchRanker:
    """Dependency to get search ranker instance"""
    global search_ranker
    if search_ranker is None:
        search_ranker = create_search_ranker()
    return search_ranker


def get_analytics_warehouse(request: Request) -> Optional[AnalyticsWarehouse]:
    """Dependency to get analytics warehouse from app state"""
    return getattr(request.app.state, 'analytics_warehouse', None)


async def track_search_analytics(query: str, result_count: int, response_time: float, success: bool = True):
    """Track search analytics for monitoring and optimization"""
    global search_analytics
    
    search_analytics["total_searches"] += 1
    
    if success:
        search_analytics["successful_searches"] += 1
    else:
        search_analytics["failed_searches"] += 1
    
    # Update average response time
    current_avg = search_analytics["average_response_time"]
    total_searches = search_analytics["total_searches"]
    search_analytics["average_response_time"] = (
        (current_avg * (total_searches - 1) + response_time) / total_searches
    )
    
    # Track popular queries
    query_lower = query.lower().strip()
    if query_lower:
        if query_lower in search_analytics["popular_queries"]:
            search_analytics["popular_queries"][query_lower] += 1
        else:
            search_analytics["popular_queries"][query_lower] = 1
    
    search_analytics["last_updated"] = datetime.now()


async def calculate_relevance_scores(
    query: str, 
    properties: List[Dict], 
    ranker: NLPSearchRanker
) -> List[Tuple[Dict, float]]:
    """Calculate relevance scores for properties using ML ranker"""
    try:
        if not properties:
            return []
        
        # Convert property objects to dictionaries if needed
        prop_dicts = []
        for prop in properties:
            if hasattr(prop, '__dict__'):
                prop_dict = {
                    'id': str(prop.id),
                    'title': prop.title,
                    'description': prop.description,
                    'location': prop.location,
                    'amenities': prop.amenities,
                    'price': prop.price,
                    'bedrooms': prop.bedrooms,
                    'bathrooms': prop.bathrooms,
                    'property_type': prop.property_type
                }
            else:
                prop_dict = prop
            prop_dicts.append(prop_dict)
        
        # Get rankings from ML model
        ranking_results = ranker.rank_properties(query, prop_dicts)
        
        # Create property-score pairs
        scored_properties = []
        for i, prop in enumerate(properties):
            # Find corresponding ranking result
            prop_id = str(prop.id) if hasattr(prop, 'id') else str(i)
            score = 0.5  # Default score
            
            for result in ranking_results:
                if result.property_id == prop_id:
                    score = result.relevance_score
                    break
            
            scored_properties.append((prop, score))
        
        # Sort by relevance score (descending)
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return scored_properties
        
    except Exception as e:
        logger.error(f"Failed to calculate relevance scores: {e}")
        # Return properties with default scores
        return [(prop, 0.5) for prop in properties]


async def apply_advanced_filtering(
    properties: List, 
    filters: SearchFilters,
    query: str
) -> List:
    """Apply comprehensive filtering logic to properties"""
    if not properties:
        return properties
    
    filtered_properties = []
    
    for prop in properties:
        # Price filtering
        if filters.min_price is not None and prop.price < filters.min_price:
            continue
        if filters.max_price is not None and prop.price > filters.max_price:
            continue
        
        # Bedroom filtering
        if filters.min_bedrooms is not None and prop.bedrooms < filters.min_bedrooms:
            continue
        if filters.max_bedrooms is not None and prop.bedrooms > filters.max_bedrooms:
            continue
        
        # Bathroom filtering
        if filters.min_bathrooms is not None and prop.bathrooms < filters.min_bathrooms:
            continue
        if filters.max_bathrooms is not None and prop.bathrooms > filters.max_bathrooms:
            continue
        
        # Location filtering
        if filters.locations:
            location_match = any(
                location.lower() in prop.location.lower() 
                for location in filters.locations
            )
            if not location_match:
                continue
        
        # Amenities filtering
        if filters.amenities:
            property_amenities = [amenity.lower() for amenity in prop.amenities]
            required_amenities = [amenity.lower() for amenity in filters.amenities]
            
            if not all(
                any(req_amenity in prop_amenity for prop_amenity in property_amenities)
                for req_amenity in required_amenities
            ):
                continue
        
        # Property type filtering
        if filters.property_types:
            if prop.property_type.lower() not in [pt.lower() for pt in filters.property_types]:
                continue
        
        # Square feet filtering
        if prop.square_feet:
            if filters.min_square_feet is not None and prop.square_feet < filters.min_square_feet:
                continue
            if filters.max_square_feet is not None and prop.square_feet > filters.max_square_feet:
                continue
        
        filtered_properties.append(prop)
    
    return filtered_properties


@router.post("/", response_model=SearchResponse)
async def search_properties(
    search_request: SearchRequest,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Search for rental properties using natural language queries and filters.
    
    This endpoint provides intelligent property search with:
    - Natural language query processing
    - Advanced filtering options
    - Relevance-based ranking
    - Personalized results (when user_id provided)
    - Pagination support
    
    Returns a list of properties ranked by relevance with detailed metadata.
    """
    start_time = time.time()
    
    try:
        # Get repositories
        property_repo = repository_factory.get_property_repository()
        cache_repo = repository_factory.get_cache_repository()
        
        # Create domain search filters
        filters = SearchFilters(
            min_price=search_request.filters.min_price if search_request.filters else None,
            max_price=search_request.filters.max_price if search_request.filters else None,
            min_bedrooms=search_request.filters.min_bedrooms if search_request.filters else None,
            max_bedrooms=search_request.filters.max_bedrooms if search_request.filters else None,
            min_bathrooms=search_request.filters.min_bathrooms if search_request.filters else None,
            max_bathrooms=search_request.filters.max_bathrooms if search_request.filters else None,
            locations=search_request.filters.locations if search_request.filters else [],
            amenities=search_request.filters.amenities if search_request.filters else [],
            property_types=search_request.filters.property_types if search_request.filters else [],
            min_square_feet=search_request.filters.min_square_feet if search_request.filters else None,
            max_square_feet=search_request.filters.max_square_feet if search_request.filters else None
        )
        
        # Create domain search query
        domain_query = SearchQuery(
            query_text=search_request.query,
            filters=filters,
            user_id=search_request.user_id,
            limit=search_request.limit,
            offset=search_request.offset,
            sort_by=search_request.sort_by
        )
        
        # Check cache first
        cache_key = f"search:{domain_query.get_cache_key()}"
        cached_result = await cache_repo.get_cached_search_results(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached search results for query: {search_request.query}")
            search_time_ms = (time.time() - start_time) * 1000
            
            # Convert cached results to response format
            properties = [PropertyResponse(**prop) for prop in cached_result]
            return SearchResponse(
                properties=properties,
                total_count=len(properties),
                page=(search_request.offset // search_request.limit) + 1,
                page_size=search_request.limit,
                total_pages=(len(properties) + search_request.limit - 1) // search_request.limit,
                query=search_request.query,
                filters=search_request.filters,
                sort_by=search_request.sort_by,
                search_time_ms=search_time_ms,
                suggestions=None
            )
        
        # Perform search
        logger.info(f"Executing search for query: {search_request.query}")
        properties, total_count = await property_repo.search(domain_query)
        
        # Apply advanced filtering
        filtered_properties = await apply_advanced_filtering(
            properties, 
            filters, 
            search_request.query
        )
        
        # Get search ranker for relevance scoring
        ranker = get_search_ranker()
        
        # Calculate relevance scores using ML
        scored_properties = await calculate_relevance_scores(
            search_request.query,
            filtered_properties,
            ranker
        )
        
        # Convert to response format with real relevance scores
        property_responses = []
        for prop, relevance_score in scored_properties:
            property_response = PropertyResponse(
                id=prop.id,
                title=prop.title,
                description=prop.description,
                price=prop.price,
                location=prop.location,
                bedrooms=prop.bedrooms,
                bathrooms=prop.bathrooms,
                square_feet=prop.square_feet,
                amenities=prop.amenities,
                contact_info=prop.contact_info,
                images=prop.images,
                property_type=prop.property_type,
                scraped_at=prop.scraped_at,
                is_active=prop.is_active,
                price_per_sqft=prop.get_price_per_sqft() if prop.square_feet else None,
                relevance_score=relevance_score
            )
            property_responses.append(property_response)
        
        # Update total count to reflect filtering
        total_count = len(property_responses)
        
        # Calculate pagination info
        page = (search_request.offset // search_request.limit) + 1
        total_pages = (total_count + search_request.limit - 1) // search_request.limit
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = SearchResponse(
            properties=property_responses,
            total_count=total_count,
            page=page,
            page_size=search_request.limit,
            total_pages=total_pages,
            query=search_request.query,
            filters=search_request.filters,
            sort_by=search_request.sort_by,
            search_time_ms=search_time_ms,
            suggestions=await _generate_search_suggestions(search_request.query)
        )
        
        # Cache results for future requests
        cache_data = [prop.dict() for prop in property_responses]
        await cache_repo.cache_search_results(cache_key, cache_data, ttl_seconds=300)
        
        # Track search analytics
        await track_search_analytics(
            search_request.query,
            len(property_responses),
            search_time_ms,
            success=True
        )
        
        # Store search analytics in warehouse if available
        warehouse = get_analytics_warehouse(request)
        if warehouse:
            try:
                search_data = {
                    "query": search_request.query,
                    "user_id": str(search_request.user_id) if search_request.user_id else None,
                    "result_count": len(property_responses),
                    "response_time_ms": search_time_ms,
                    "filters_used": {
                        "has_price_filter": domain_query.has_price_filter(),
                        "has_location_filter": domain_query.has_location_filter(),
                        "has_size_filter": domain_query.has_size_filter(),
                    },
                    "timestamp": datetime.now(),
                    "sort_by": search_request.sort_by
                }
                
                # Store in analytics warehouse asynchronously
                asyncio.create_task(
                    warehouse.store_analytics_data(
                        "search_events",
                        pd.DataFrame([search_data])
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to store search analytics: {e}")
        
        logger.info(f"Search completed: {len(property_responses)} results in {search_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Search failed for query '{search_request.query}': {e}")
        
        # Track failed search
        search_time_ms = (time.time() - start_time) * 1000
        await track_search_analytics(
            search_request.query,
            0,
            search_time_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Search operation failed: {str(e)}"
        )


@router.get("/suggestions", response_model=SearchSuggestionResponse)
async def get_search_suggestions(
    query: str = Query(..., min_length=1, max_length=100, description="Partial search query"),
    limit: int = Query(default=10, ge=1, le=20, description="Number of suggestions"),
    user_id: Optional[UUID] = Query(None, description="User ID for personalized suggestions"),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get search suggestions based on partial query input.
    
    This endpoint provides intelligent search suggestions including:
    - Auto-completion for location names
    - Popular search terms
    - Personalized suggestions based on user history
    - Property type suggestions
    """
    start_time = time.time()
    
    try:
        suggestions = await _generate_search_suggestions(query, limit, user_id)
        response_time_ms = (time.time() - start_time) * 1000
        
        return SearchSuggestionResponse(
            suggestions=suggestions,
            query=query,
            response_time_ms=response_time_ms
        )
        
    except Exception as e:
        logger.error(f"Failed to get search suggestions for query '{query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search suggestions: {str(e)}"
        )


@router.get("/popular", response_model=PopularSearchResponse)
async def get_popular_searches(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50, description="Number of popular terms"),
    time_range: str = Query(default="24h", description="Time range for popularity"),
    category: Optional[str] = Query(None, description="Category filter"),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get popular search terms and trending queries.
    
    Returns trending search terms based on:
    - Search frequency
    - Recent activity
    - Category filters
    - Time range analysis
    """
    start_time = time.time()
    
    try:
        # Get analytics data
        global search_analytics
        popular_queries = search_analytics.get("popular_queries", {})
        
        # Get analytics warehouse for historical data
        warehouse = get_analytics_warehouse(request)
        
        if warehouse:
            try:
                # Query analytics warehouse for popular searches
                query = """
                SELECT 
                    query,
                    COUNT(*) as search_count,
                    AVG(response_time_ms) as avg_response_time,
                    MAX(timestamp) as last_searched
                FROM search_events 
                WHERE timestamp >= NOW() - INTERVAL %s
                GROUP BY query
                ORDER BY search_count DESC
                LIMIT %s
                """
                
                # Convert time range to SQL interval
                interval_map = {
                    "1h": "1 HOUR",
                    "24h": "1 DAY", 
                    "7d": "7 DAYS",
                    "30d": "30 DAYS"
                }
                interval = interval_map.get(time_range, "1 DAY")
                
                warehouse_result = await warehouse.query_analytics_data(
                    query,
                    parameters={"interval": interval, "limit": limit * 2},
                    use_cache=True,
                    cache_ttl=600
                )
                
                if warehouse_result["status"] == "success" and warehouse_result["data"]:
                    warehouse_popular = warehouse_result["data"]
                else:
                    warehouse_popular = []
                    
            except Exception as e:
                logger.warning(f"Failed to get warehouse data for popular searches: {e}")
                warehouse_popular = []
        else:
            warehouse_popular = []
        
        # Combine current analytics with warehouse data
        combined_popular = {}
        
        # Add current session data
        for query, count in popular_queries.items():
            combined_popular[query] = {
                "term": query,
                "count": count,
                "category": _categorize_search_term(query),
                "trend": "stable",
                "source": "current"
            }
        
        # Add warehouse data
        for item in warehouse_popular:
            query = item.get("query", "")
            count = item.get("search_count", 0)
            
            if query in combined_popular:
                # Combine counts
                combined_popular[query]["count"] += count
                combined_popular[query]["trend"] = _calculate_trend(
                    combined_popular[query]["count"], 
                    count
                )
            else:
                combined_popular[query] = {
                    "term": query,
                    "count": count,
                    "category": _categorize_search_term(query),
                    "trend": "stable",
                    "source": "warehouse"
                }
        
        # Convert to list and sort by count
        popular_terms = list(combined_popular.values())
        popular_terms.sort(key=lambda x: x["count"], reverse=True)
        
        # Filter by category if specified
        if category:
            popular_terms = [term for term in popular_terms if term["category"] == category]
        
        # Limit results
        popular_terms = popular_terms[:limit]
        
        # If we don't have enough real data, supplement with smart defaults
        if len(popular_terms) < limit:
            default_terms = [
                {"term": "downtown apartment", "count": 100, "category": "location", "trend": "up"},
                {"term": "2 bedroom", "count": 95, "category": "bedrooms", "trend": "stable"},
                {"term": "gym amenities", "count": 85, "category": "amenities", "trend": "up"},
                {"term": "pet friendly", "count": 80, "category": "amenities", "trend": "up"},
                {"term": "parking included", "count": 75, "category": "amenities", "trend": "stable"},
                {"term": "near metro", "count": 70, "category": "location", "trend": "up"},
                {"term": "luxury condo", "count": 65, "category": "property_type", "trend": "up"},
                {"term": "studio apartment", "count": 60, "category": "property_type", "trend": "stable"},
                {"term": "washer dryer", "count": 55, "category": "amenities", "trend": "stable"},
                {"term": "under $2000", "count": 50, "category": "price", "trend": "down"}
            ]
            
            existing_terms = {term["term"] for term in popular_terms}
            for default_term in default_terms:
                if len(popular_terms) >= limit:
                    break
                if default_term["term"] not in existing_terms:
                    if not category or default_term["category"] == category:
                        popular_terms.append(default_term)
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return PopularSearchResponse(
            terms=popular_terms,
            time_range=time_range,
            category=category,
            response_time_ms=response_time_ms
        )
        
    except Exception as e:
        logger.error(f"Failed to get popular searches: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get popular searches: {str(e)}"
        )


def _categorize_search_term(term: str) -> str:
    """Categorize a search term based on its content"""
    term_lower = term.lower()
    
    # Price category
    if any(word in term_lower for word in ["$", "under", "over", "price", "cheap", "expensive", "affordable"]):
        return "price"
    
    # Location category  
    if any(word in term_lower for word in ["downtown", "city", "near", "metro", "suburb", "center", "district"]):
        return "location"
    
    # Bedrooms category
    if any(word in term_lower for word in ["bedroom", "bed", "br", "studio"]):
        return "bedrooms"
    
    # Amenities category
    if any(word in term_lower for word in ["gym", "pool", "parking", "pet", "laundry", "balcony", "garden", "dishwasher"]):
        return "amenities"
    
    # Property type category
    if any(word in term_lower for word in ["apartment", "condo", "house", "loft", "townhouse", "luxury"]):
        return "property_type"
    
    return "general"


def _calculate_trend(current_count: int, historical_count: int) -> str:
    """Calculate trend based on current vs historical counts"""
    if historical_count == 0:
        return "up"
    
    ratio = current_count / historical_count
    
    if ratio > 1.2:
        return "up"
    elif ratio < 0.8:
        return "down"
    else:
        return "stable"


@router.get("/autocomplete")
async def autocomplete_search(
    query: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=20),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get autocomplete suggestions for search input.
    
    Provides real-time suggestions including:
    - Location names
    - Property types
    - Amenity names
    - Popular search completions
    """
    try:
        suggestions = await _generate_autocomplete_suggestions(query, limit)
        
        return {
            "suggestions": suggestions,
            "query": query,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Autocomplete failed for query '{query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Autocomplete failed: {str(e)}"
        )


async def _generate_search_suggestions(
    query: str, 
    limit: int = 10, 
    user_id: Optional[UUID] = None
) -> List[str]:
    """Generate intelligent search suggestions using ML and analytics"""
    try:
        query_lower = query.lower().strip()
        suggestions = []
        
        # Get suggestions from popular queries in analytics
        global search_analytics
        popular_queries = search_analytics.get("popular_queries", {})
        
        # Find similar queries from analytics data
        similar_queries = []
        for popular_query, count in sorted(popular_queries.items(), key=lambda x: x[1], reverse=True):
            if query_lower in popular_query or popular_query.startswith(query_lower):
                similar_queries.append(popular_query)
        
        # Add top similar queries
        suggestions.extend(similar_queries[:limit//2])
        
        # Generate context-aware suggestions
        context_suggestions = []
        
        # Location-based suggestions
        if any(word in query_lower for word in ["downtown", "center", "city", "near"]):
            context_suggestions.extend([
                f"{query_lower} apartment with gym",
                f"{query_lower} luxury condo",
                f"{query_lower} 2 bedroom",
                f"{query_lower} with parking"
            ])
        
        # Bedroom-based suggestions
        if any(word in query_lower for word in ["bedroom", "bed", "br", "studio"]):
            context_suggestions.extend([
                f"{query_lower} apartment",
                f"{query_lower} with parking",
                f"{query_lower} near metro",
                f"{query_lower} pet friendly"
            ])
        
        # Amenity-based suggestions
        if any(word in query_lower for word in ["gym", "pool", "parking", "pet"]):
            context_suggestions.extend([
                f"{query_lower} apartment",
                f"{query_lower} downtown",
                f"{query_lower} luxury",
                f"{query_lower} 2 bedroom"
            ])
        
        # Price-based suggestions
        if any(word in query_lower for word in ["under", "below", "cheap", "affordable", "$"]):
            context_suggestions.extend([
                f"{query_lower} apartment",
                f"{query_lower} studio",
                f"{query_lower} 1 bedroom",
                f"{query_lower} near transit"
            ])
        
        # Add context suggestions
        suggestions.extend(context_suggestions[:limit//2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen and suggestion != query_lower:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        # If we don't have enough suggestions, add smart defaults
        if len(unique_suggestions) < limit:
            smart_defaults = [
                "2 bedroom downtown apartment",
                "luxury apartment with gym",
                "pet friendly apartment with parking",
                "studio apartment under $1500",
                "3 bedroom house with garden",
                "condo with pool and gym",
                "furnished apartment near metro",
                "apartment with balcony downtown",
                "modern loft with parking",
                "family home with yard"
            ]
            
            for default in smart_defaults:
                if len(unique_suggestions) >= limit:
                    break
                if default not in seen and query_lower not in default:
                    unique_suggestions.append(default)
                    seen.add(default)
        
        return unique_suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Failed to generate suggestions: {e}")
        return []


async def _generate_autocomplete_suggestions(query: str, limit: int = 10) -> List[str]:
    """Generate autocomplete suggestions"""
    try:
        query_lower = query.lower()
        
        # Predefined completions
        completions = [
            "downtown apartment", "luxury condo", "studio apartment",
            "2 bedroom", "3 bedroom", "1 bedroom",
            "pet friendly", "gym amenities", "parking included",
            "near metro", "city center", "suburb",
            "under $2000", "under $1500", "under $3000",
            "furnished", "unfurnished", "utilities included",
            "balcony", "terrace", "garden", "pool",
            "dishwasher", "washer dryer", "air conditioning"
        ]
        
        # Filter completions that start with or contain the query
        matches = []
        
        # Exact prefix matches first
        for completion in completions:
            if completion.startswith(query_lower):
                matches.append(completion)
        
        # Contains matches second
        for completion in completions:
            if query_lower in completion and completion not in matches:
                matches.append(completion)
        
        return matches[:limit]
        
    except Exception as e:
        logger.error(f"Failed to generate autocomplete: {e}")
        return []


@router.get("/analytics")
async def get_search_analytics(
    request: Request,
    time_range: str = Query(default="24h", description="Time range for analytics"),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get comprehensive search analytics and performance metrics.
    
    Returns detailed analytics including:
    - Search volume and trends
    - Performance metrics
    - Popular queries and categories
    - User behavior patterns
    - System performance statistics
    """
    start_time = time.time()
    
    try:
        global search_analytics
        
        # Get current session analytics
        current_analytics = {
            "session_stats": search_analytics.copy(),
            "performance_metrics": {
                "total_searches": search_analytics.get("total_searches", 0),
                "success_rate": (
                    search_analytics.get("successful_searches", 0) / 
                    max(search_analytics.get("total_searches", 1), 1)
                ) * 100,
                "failure_rate": (
                    search_analytics.get("failed_searches", 0) / 
                    max(search_analytics.get("total_searches", 1), 1)
                ) * 100,
                "average_response_time_ms": search_analytics.get("average_response_time", 0) * 1000
            }
        }
        
        # Get analytics warehouse data
        warehouse = get_analytics_warehouse(request)
        historical_analytics = {}
        
        if warehouse:
            try:
                # Query for search volume trends
                volume_query = """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as search_count,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(CASE WHEN result_count > 0 THEN 1 END) as successful_searches
                FROM search_events 
                WHERE timestamp >= NOW() - INTERVAL %s
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC
                """
                
                interval_map = {
                    "1h": "1 HOUR",
                    "24h": "1 DAY", 
                    "7d": "7 DAYS",
                    "30d": "30 DAYS"
                }
                interval = interval_map.get(time_range, "1 DAY")
                
                volume_result = await warehouse.query_analytics_data(
                    volume_query,
                    parameters={"interval": interval},
                    use_cache=True,
                    cache_ttl=300
                )
                
                # Query for user behavior patterns
                behavior_query = """
                SELECT 
                    user_id,
                    COUNT(*) as search_count,
                    AVG(result_count) as avg_results_per_search,
                    COUNT(DISTINCT query) as unique_queries
                FROM search_events 
                WHERE timestamp >= NOW() - INTERVAL %s
                    AND user_id IS NOT NULL
                GROUP BY user_id
                ORDER BY search_count DESC
                LIMIT 100
                """
                
                behavior_result = await warehouse.query_analytics_data(
                    behavior_query,
                    parameters={"interval": interval},
                    use_cache=True,
                    cache_ttl=300
                )
                
                # Query for query performance analysis
                performance_query = """
                SELECT 
                    query,
                    COUNT(*) as frequency,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(result_count) as avg_result_count,
                    MAX(response_time_ms) as max_response_time,
                    MIN(response_time_ms) as min_response_time
                FROM search_events 
                WHERE timestamp >= NOW() - INTERVAL %s
                GROUP BY query
                HAVING COUNT(*) >= 2
                ORDER BY frequency DESC
                LIMIT 50
                """
                
                performance_result = await warehouse.query_analytics_data(
                    performance_query,
                    parameters={"interval": interval},
                    use_cache=True,
                    cache_ttl=300
                )
                
                historical_analytics = {
                    "volume_trends": volume_result.get("data", []) if volume_result["status"] == "success" else [],
                    "user_behavior": behavior_result.get("data", []) if behavior_result["status"] == "success" else [],
                    "query_performance": performance_result.get("data", []) if performance_result["status"] == "success" else []
                }
                
            except Exception as e:
                logger.warning(f"Failed to get historical analytics: {e}")
                historical_analytics = {
                    "volume_trends": [],
                    "user_behavior": [],
                    "query_performance": []
                }
        
        # Calculate advanced metrics
        advanced_metrics = _calculate_advanced_search_metrics(
            current_analytics,
            historical_analytics
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "time_range": time_range,
            "current_session": current_analytics,
            "historical_data": historical_analytics,
            "advanced_metrics": advanced_metrics,
            "response_time_ms": response_time_ms,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get search analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search analytics: {str(e)}"
        )


@router.get("/metrics")
async def get_search_metrics(
    request: Request,
    metric_type: str = Query(default="all", description="Type of metrics to return"),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get specific search metrics for monitoring and alerting.
    
    Metric types:
    - performance: Response times, throughput, error rates
    - usage: Search volume, popular terms, user patterns
    - quality: Relevance scores, result satisfaction
    - system: Resource usage, cache hit rates
    """
    start_time = time.time()
    
    try:
        global search_analytics
        
        metrics = {}
        
        if metric_type in ["all", "performance"]:
            metrics["performance"] = {
                "avg_response_time_ms": search_analytics.get("average_response_time", 0) * 1000,
                "total_requests": search_analytics.get("total_searches", 0),
                "successful_requests": search_analytics.get("successful_searches", 0),
                "failed_requests": search_analytics.get("failed_searches", 0),
                "success_rate_percent": (
                    search_analytics.get("successful_searches", 0) / 
                    max(search_analytics.get("total_searches", 1), 1)
                ) * 100,
                "error_rate_percent": (
                    search_analytics.get("failed_searches", 0) / 
                    max(search_analytics.get("total_searches", 1), 1)
                ) * 100
            }
        
        if metric_type in ["all", "usage"]:
            popular_queries = search_analytics.get("popular_queries", {})
            metrics["usage"] = {
                "total_unique_queries": len(popular_queries),
                "most_popular_query": max(popular_queries.items(), key=lambda x: x[1]) if popular_queries else None,
                "query_diversity": _calculate_query_diversity(popular_queries),
                "search_volume_trend": "stable"  # Would be calculated from historical data
            }
        
        if metric_type in ["all", "quality"]:
            # Would integrate with ranker metrics in production
            metrics["quality"] = {
                "avg_relevance_score": 0.75,  # Placeholder
                "ranker_model_accuracy": 0.82,  # Placeholder
                "user_satisfaction_score": 0.78  # Placeholder - from click-through rates
            }
        
        if metric_type in ["all", "system"]:
            # Would integrate with system monitoring in production
            metrics["system"] = {
                "cache_hit_rate_percent": 65.0,  # Placeholder
                "avg_db_query_time_ms": 45.0,  # Placeholder
                "ml_model_inference_time_ms": 25.0,  # Placeholder
                "memory_usage_mb": 256.0  # Placeholder
            }
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "metric_type": metric_type,
            "metrics": metrics,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get search metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search metrics: {str(e)}"
        )


def _calculate_advanced_search_metrics(
    current_analytics: Dict[str, Any],
    historical_analytics: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate advanced search metrics from analytics data"""
    try:
        metrics = {}
        
        # Search velocity metrics
        total_searches = current_analytics.get("session_stats", {}).get("total_searches", 0)
        time_since_start = (
            datetime.now() - 
            current_analytics.get("session_stats", {}).get("last_updated", datetime.now())
        ).total_seconds()
        
        metrics["search_velocity"] = {
            "searches_per_hour": (total_searches / max(time_since_start / 3600, 0.1)),
            "peak_search_hour": "N/A",  # Would be calculated from historical data
            "search_distribution": "even"  # Would be calculated from time-series data
        }
        
        # Query complexity metrics
        popular_queries = current_analytics.get("session_stats", {}).get("popular_queries", {})
        if popular_queries:
            avg_query_length = sum(len(query.split()) for query in popular_queries.keys()) / len(popular_queries)
            metrics["query_complexity"] = {
                "avg_query_length_words": avg_query_length,
                "complex_query_ratio": len([q for q in popular_queries.keys() if len(q.split()) > 3]) / len(popular_queries),
                "filter_usage_rate": 0.3  # Placeholder - would track filter usage
            }
        
        # Performance trends
        historical_volume = historical_analytics.get("volume_trends", [])
        if historical_volume:
            recent_avg_time = sum(
                item.get("avg_response_time", 0) for item in historical_volume[:5]
            ) / min(len(historical_volume), 5)
            
            metrics["performance_trends"] = {
                "response_time_trend": "improving" if recent_avg_time < 100 else "stable",
                "throughput_trend": "increasing",
                "error_trend": "decreasing"
            }
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Failed to calculate advanced metrics: {e}")
        return {}


def _calculate_query_diversity(popular_queries: Dict[str, int]) -> float:
    """Calculate query diversity using Shannon entropy"""
    try:
        if not popular_queries:
            return 0.0
        
        total_queries = sum(popular_queries.values())
        entropy = 0.0
        
        for count in popular_queries.values():
            probability = count / total_queries
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize to 0-1 range
        max_entropy = math.log2(len(popular_queries))
        return entropy / max_entropy if max_entropy > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Failed to calculate query diversity: {e}")
        return 0.0