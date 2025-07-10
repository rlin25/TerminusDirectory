"""
Search API router for property search functionality.

This module provides endpoints for property search, search suggestions,
and search analytics.
"""

import time
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ....domain.entities.search_query import SearchQuery, SearchFilters
from ....infrastructure.data import get_property_repository, get_cache_repository
from ...dto.search_dto import (
    SearchRequest, SearchResponse, PropertyResponse,
    SearchSuggestionRequest, SearchSuggestionResponse,
    PopularSearchRequest, PopularSearchResponse,
    SearchErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


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
        
        # Convert to response format
        property_responses = []
        for prop in properties:
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
                relevance_score=0.95  # TODO: Implement actual relevance scoring
            )
            property_responses.append(property_response)
        
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
        
        logger.info(f"Search completed: {len(properties)} results in {search_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Search failed for query '{search_request.query}': {e}")
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
        # TODO: Implement actual popular search analytics
        # For now, return mock data
        popular_terms = [
            {"term": "downtown apartment", "count": 1250, "category": "location", "trend": "up"},
            {"term": "2 bedroom", "count": 980, "category": "bedrooms", "trend": "stable"},
            {"term": "gym amenities", "count": 750, "category": "amenities", "trend": "up"},
            {"term": "under $2000", "count": 680, "category": "price", "trend": "down"},
            {"term": "pet friendly", "count": 620, "category": "amenities", "trend": "up"},
            {"term": "parking included", "count": 580, "category": "amenities", "trend": "stable"},
            {"term": "near metro", "count": 520, "category": "location", "trend": "up"},
            {"term": "luxury condo", "count": 480, "category": "property_type", "trend": "up"},
            {"term": "studio apartment", "count": 420, "category": "property_type", "trend": "stable"},
            {"term": "washer dryer", "count": 380, "category": "amenities", "trend": "stable"}
        ]
        
        # Filter by category if specified
        if category:
            popular_terms = [term for term in popular_terms if term["category"] == category]
        
        # Limit results
        popular_terms = popular_terms[:limit]
        
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
    """Generate intelligent search suggestions"""
    try:
        # TODO: Implement ML-based suggestion generation
        # For now, return relevant suggestions based on query keywords
        
        query_lower = query.lower()
        suggestions = []
        
        # Location suggestions
        if any(word in query_lower for word in ["downtown", "center", "city", "near"]):
            suggestions.extend([
                "downtown apartment with gym",
                "city center luxury condo",
                "near metro station",
                "downtown 2 bedroom"
            ])
        
        # Bedroom suggestions
        if any(word in query_lower for word in ["bedroom", "bed", "br"]):
            suggestions.extend([
                "1 bedroom apartment",
                "2 bedroom with parking",
                "3 bedroom house",
                "studio apartment"
            ])
        
        # Amenity suggestions
        if any(word in query_lower for word in ["gym", "pool", "parking", "pet"]):
            suggestions.extend([
                "gym and pool amenities",
                "pet friendly with parking",
                "luxury amenities included",
                "fitness center access"
            ])
        
        # Price suggestions
        if any(word in query_lower for word in ["under", "below", "cheap", "affordable"]):
            suggestions.extend([
                "under $2000 per month",
                "affordable 1 bedroom",
                "budget friendly options",
                "under $1500 studio"
            ])
        
        # Default suggestions if no matches
        if not suggestions:
            suggestions = [
                "2 bedroom downtown",
                "luxury apartment with gym",
                "pet friendly parking",
                "studio under $1500",
                "3 bedroom house",
                "condo with amenities"
            ]
        
        return suggestions[:limit]
        
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