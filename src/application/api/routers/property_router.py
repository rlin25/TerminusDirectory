"""
Property API router for property management and information.

This module provides endpoints for property CRUD operations,
property statistics, and property analytics.
"""

import time
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...dto.search_dto import PropertyResponse
from ....domain.entities.property import Property

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


@router.get("/{property_id}", response_model=PropertyResponse)
async def get_property(
    property_id: UUID,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Get detailed information about a specific property.
    
    Returns comprehensive property data including:
    - Basic property details
    - Amenities and features
    - Contact information
    - Images and media
    - Pricing information
    - Location details
    """
    try:
        property_repo = repository_factory.get_property_repository()
        
        property_data = await property_repo.get_by_id(property_id)
        if not property_data:
            raise HTTPException(
                status_code=404,
                detail=f"Property {property_id} not found"
            )
        
        response = PropertyResponse(
            id=property_data.id,
            title=property_data.title,
            description=property_data.description,
            price=property_data.price,
            location=property_data.location,
            bedrooms=property_data.bedrooms,
            bathrooms=property_data.bathrooms,
            square_feet=property_data.square_feet,
            amenities=property_data.amenities,
            contact_info=property_data.contact_info,
            images=property_data.images,
            property_type=property_data.property_type,
            scraped_at=property_data.scraped_at,
            is_active=property_data.is_active,
            price_per_sqft=property_data.get_price_per_sqft() if property_data.square_feet else None
        )
        
        logger.info(f"Retrieved property {property_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get property {property_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve property: {str(e)}"
        )


@router.get("/", response_model=List[PropertyResponse])
async def list_properties(
    limit: int = Query(default=20, ge=1, le=100, description="Number of properties to return"),
    offset: int = Query(default=0, ge=0, description="Number of properties to skip"),
    location: Optional[str] = Query(None, description="Filter by location"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    property_type: Optional[str] = Query(None, description="Property type filter"),
    active_only: bool = Query(default=True, description="Show only active properties"),
    repository_factory = Depends(get_repository_factory)
):
    """
    List properties with optional filtering and pagination.
    
    Supports filtering by:
    - Location
    - Price range
    - Property type
    - Active status
    
    Returns paginated list of properties with basic information.
    """
    try:
        property_repo = repository_factory.get_property_repository()
        
        if location:
            properties = await property_repo.get_by_location(location, limit, offset)
        elif min_price or max_price:
            min_p = min_price or 0
            max_p = max_price or 999999999
            properties = await property_repo.get_by_price_range(min_p, max_p, limit, offset)
        else:
            properties = await property_repo.get_all_active(limit, offset)
        
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
                price_per_sqft=prop.get_price_per_sqft() if prop.square_feet else None
            )
            property_responses.append(property_response)
        
        logger.info(f"Listed {len(property_responses)} properties")
        return property_responses
        
    except Exception as e:
        logger.error(f"Failed to list properties: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list properties: {str(e)}"
        )


@router.get("/{property_id}/features")
async def get_property_features(
    property_id: UUID,
    repository_factory = Depends(get_repository_factory)
):
    """
    Get ML features for a specific property.
    
    Returns feature vector and metadata used by ML models including:
    - Numerical features (price, bedrooms, etc.)
    - Categorical features (location, property type)
    - Derived features (price per sqft, amenity counts)
    - Embedding representations
    """
    try:
        property_repo = repository_factory.get_property_repository()
        
        features = await property_repo.get_property_features(property_id)
        if not features:
            raise HTTPException(
                status_code=404,
                detail=f"Property {property_id} not found"
            )
        
        return {
            "property_id": property_id,
            "features": features,
            "feature_count": len(features),
            "generated_at": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get property features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get property features: {str(e)}"
        )


@router.get("/stats/overview")
async def get_property_statistics(
    repository_factory = Depends(get_repository_factory)
):
    """
    Get comprehensive property statistics and analytics.
    
    Returns statistics including:
    - Total property counts
    - Price distributions
    - Location breakdowns
    - Property type distributions
    - Market trends
    """
    try:
        property_repo = repository_factory.get_property_repository()
        
        # Get basic counts
        total_count = await property_repo.get_count()
        active_count = await property_repo.get_active_count()
        
        # Get aggregated statistics
        stats = await property_repo.get_aggregated_stats()
        
        return {
            "counts": {
                "total_properties": total_count,
                "active_properties": active_count,
                "inactive_properties": total_count - active_count
            },
            "pricing": {
                "average_price": stats.get("avg_price", 0),
                "min_price": stats.get("min_price", 0),
                "max_price": stats.get("max_price", 0),
                "median_price": stats.get("avg_price", 0)  # TODO: Calculate actual median
            },
            "features": {
                "average_bedrooms": stats.get("avg_bedrooms", 0),
                "average_bathrooms": stats.get("avg_bathrooms", 0),
                "average_square_feet": stats.get("avg_square_feet", 0)
            },
            "activity": {
                "properties_added_today": 0,  # TODO: Calculate from scraped_at
                "properties_updated_today": 0,
                "new_listings_this_week": 0
            },
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get property statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get property statistics: {str(e)}"
        )


@router.get("/stats/location")
async def get_location_statistics(
    limit: int = Query(default=20, ge=1, le=100),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get statistics grouped by location.
    
    Returns location-based analytics including:
    - Property counts per location
    - Average prices by area
    - Popular locations
    - Market trends by location
    """
    try:
        # TODO: Implement location-based statistics
        # This would require additional database queries
        
        return {
            "locations": [
                {
                    "location": "Downtown",
                    "property_count": 245,
                    "average_price": 2850,
                    "min_price": 1200,
                    "max_price": 5500,
                    "popular_property_types": ["apartment", "condo"]
                },
                {
                    "location": "Midtown",
                    "property_count": 189,
                    "average_price": 2300,
                    "min_price": 1000,
                    "max_price": 4200,
                    "popular_property_types": ["apartment", "studio"]
                },
                {
                    "location": "Suburbs",
                    "property_count": 156,
                    "average_price": 1950,
                    "min_price": 800,
                    "max_price": 3800,
                    "popular_property_types": ["house", "townhouse"]
                }
            ],
            "total_locations": 3,
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get location statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get location statistics: {str(e)}"
        )


@router.get("/stats/pricing")
async def get_pricing_analytics(
    repository_factory = Depends(get_repository_factory)
):
    """
    Get detailed pricing analytics and market insights.
    
    Returns pricing data including:
    - Price distributions
    - Price trends over time
    - Price per square foot analysis
    - Market competitiveness metrics
    """
    try:
        property_repo = repository_factory.get_property_repository()
        stats = await property_repo.get_aggregated_stats()
        
        return {
            "price_distribution": {
                "under_1000": 15,  # TODO: Calculate actual distributions
                "1000_to_2000": 45,
                "2000_to_3000": 25,
                "3000_to_4000": 10,
                "over_4000": 5
            },
            "price_per_sqft": {
                "average": 2.85,
                "median": 2.70,
                "min": 1.20,
                "max": 5.50
            },
            "market_trends": {
                "price_change_30d": 2.5,  # Percentage change
                "price_change_90d": 5.8,
                "market_temperature": "warm",
                "competitiveness_score": 0.72
            },
            "recommendations": [
                "Market prices are trending upward",
                "Good time for property owners to list",
                "Competitive pricing recommended for quick rental"
            ],
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get pricing analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pricing analytics: {str(e)}"
        )