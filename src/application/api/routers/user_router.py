"""
User API router for user management and preferences.

This module provides endpoints for user CRUD operations,
user preferences management, and user analytics.
"""

import time
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...dto.user_dto import UserResponse, UserCreateRequest, UserUpdateRequest, UserPreferencesResponse
from ....domain.entities.user import User, UserPreferences

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Get detailed information about a specific user.
    
    Returns user data including:
    - Basic user information
    - User preferences
    - Interaction history summary
    - Account status
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        # Get user statistics
        user_stats = await user_repo.get_user_statistics(user_id)
        
        response = UserResponse(
            id=user.id,
            email=user.email,
            preferences=user.preferences,
            created_at=user.created_at,
            is_active=user.is_active,
            total_interactions=user_stats.get('total_interactions', 0),
            last_activity=user_stats.get('last_activity')
        )
        
        logger.info(f"Retrieved user {user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user: {str(e)}"
        )


@router.post("/", response_model=UserResponse)
async def create_user(
    user_request: UserCreateRequest,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Create a new user account.
    
    Creates a new user with:
    - Email address
    - Initial preferences
    - Account setup
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        # Check if user already exists
        existing_user = await user_repo.get_by_email(user_request.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail=f"User with email {user_request.email} already exists"
            )
        
        # Create user preferences
        preferences = UserPreferences(
            min_price=user_request.preferences.min_price if user_request.preferences else None,
            max_price=user_request.preferences.max_price if user_request.preferences else None,
            min_bedrooms=user_request.preferences.min_bedrooms if user_request.preferences else None,
            max_bedrooms=user_request.preferences.max_bedrooms if user_request.preferences else None,
            min_bathrooms=user_request.preferences.min_bathrooms if user_request.preferences else None,
            max_bathrooms=user_request.preferences.max_bathrooms if user_request.preferences else None,
            preferred_locations=user_request.preferences.preferred_locations if user_request.preferences else [],
            required_amenities=user_request.preferences.required_amenities if user_request.preferences else [],
            property_types=user_request.preferences.property_types if user_request.preferences else ["apartment"]
        )
        
        # Create user
        user = User(
            email=user_request.email,
            preferences=preferences,
            interactions=[]
        )
        
        # Save user
        success = await user_repo.save(user)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to create user"
            )
        
        response = UserResponse(
            id=user.id,
            email=user.email,
            preferences=user.preferences,
            created_at=user.created_at,
            is_active=user.is_active,
            total_interactions=0,
            last_activity=None
        )
        
        logger.info(f"Created user {user.id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create user: {str(e)}"
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_request: UserUpdateRequest,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Update user information and preferences.
    
    Updates user data including:
    - Email address
    - Preferences
    - Account settings
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        # Get existing user
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        # Update user data
        if user_request.email:
            user.email = user_request.email
        
        if user_request.preferences:
            user.preferences.min_price = user_request.preferences.min_price
            user.preferences.max_price = user_request.preferences.max_price
            user.preferences.min_bedrooms = user_request.preferences.min_bedrooms
            user.preferences.max_bedrooms = user_request.preferences.max_bedrooms
            user.preferences.min_bathrooms = user_request.preferences.min_bathrooms
            user.preferences.max_bathrooms = user_request.preferences.max_bathrooms
            user.preferences.preferred_locations = user_request.preferences.preferred_locations or []
            user.preferences.required_amenities = user_request.preferences.required_amenities or []
            user.preferences.property_types = user_request.preferences.property_types or ["apartment"]
        
        # Save updated user
        success = await user_repo.save(user)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update user"
            )
        
        # Clear recommendation cache for this user
        cache_repo = repository_factory.get_cache_repository()
        await cache_repo.clear_cache(f"recommendations:{user_id}:*")
        
        # Get updated statistics
        user_stats = await user_repo.get_user_statistics(user_id)
        
        response = UserResponse(
            id=user.id,
            email=user.email,
            preferences=user.preferences,
            created_at=user.created_at,
            is_active=user.is_active,
            total_interactions=user_stats.get('total_interactions', 0),
            last_activity=user_stats.get('last_activity')
        )
        
        logger.info(f"Updated user {user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update user: {str(e)}"
        )


@router.get("/{user_id}/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    user_id: UUID,
    repository_factory = Depends(get_repository_factory)
):
    """
    Get detailed user preferences and settings.
    
    Returns comprehensive preference data including:
    - Price preferences
    - Location preferences
    - Property type preferences
    - Amenity requirements
    - Search history insights
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        # Get user interaction insights
        interactions = await user_repo.get_interactions(user_id, limit=100)
        
        # Analyze preferences from interactions
        liked_properties = [i.property_id for i in interactions if i.interaction_type == "like"]
        viewed_properties = [i.property_id for i in interactions if i.interaction_type == "view"]
        
        response = UserPreferencesResponse(
            user_id=user_id,
            preferences=user.preferences,
            derived_insights={
                "most_viewed_price_range": "2000-3000",  # TODO: Calculate from actual data
                "favorite_locations": user.preferences.preferred_locations[:3],
                "preferred_amenities": user.preferences.required_amenities,
                "interaction_patterns": {
                    "total_likes": len(liked_properties),
                    "total_views": len(viewed_properties),
                    "engagement_rate": len(liked_properties) / max(len(viewed_properties), 1)
                }
            },
            last_updated=user.created_at  # TODO: Track actual preference update time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user preferences: {str(e)}"
        )


@router.get("/{user_id}/interactions")
async def get_user_interactions(
    user_id: UUID,
    interaction_type: Optional[str] = Query(None, description="Filter by interaction type"),
    limit: int = Query(default=50, ge=1, le=200, description="Number of interactions to return"),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get user interaction history.
    
    Returns interaction data including:
    - Property views and interactions
    - Likes and dislikes
    - Inquiries and contacts
    - Timestamps and durations
    - Interaction patterns
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        # Check if user exists
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        # Get interactions
        interactions = await user_repo.get_interactions(
            user_id, interaction_type, limit
        )
        
        # Convert to response format
        interaction_data = []
        for interaction in interactions:
            interaction_data.append({
                "property_id": str(interaction.property_id),
                "interaction_type": interaction.interaction_type,
                "timestamp": interaction.timestamp.isoformat(),
                "duration_seconds": interaction.duration_seconds
            })
        
        return {
            "user_id": str(user_id),
            "interactions": interaction_data,
            "total_count": len(interaction_data),
            "interaction_type_filter": interaction_type,
            "generated_at": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user interactions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user interactions: {str(e)}"
        )


@router.get("/{user_id}/statistics")
async def get_user_statistics(
    user_id: UUID,
    repository_factory = Depends(get_repository_factory)
):
    """
    Get comprehensive user statistics and analytics.
    
    Returns detailed user metrics including:
    - Activity statistics
    - Engagement metrics
    - Preference evolution
    - Recommendation performance
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        # Get user statistics
        stats = await user_repo.get_user_statistics(user_id)
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        return {
            "user_id": str(user_id),
            "activity": {
                "total_interactions": stats.get('total_interactions', 0),
                "unique_properties": stats.get('unique_properties', 0),
                "views": stats.get('views', 0),
                "likes": stats.get('likes', 0),
                "inquiries": stats.get('inquiries', 0),
                "last_activity": stats.get('last_activity').isoformat() if stats.get('last_activity') else None,
                "first_activity": stats.get('first_activity').isoformat() if stats.get('first_activity') else None
            },
            "engagement": {
                "avg_session_duration": 180,  # TODO: Calculate from actual data
                "properties_per_session": 5.2,
                "conversion_rate": (stats.get('inquiries', 0) / max(stats.get('views', 1), 1)) * 100,
                "return_user": stats.get('total_interactions', 0) > 10
            },
            "preferences": {
                "preference_stability": 0.85,  # TODO: Calculate preference changes over time
                "exploration_score": 0.65,     # TODO: Calculate how much user explores different types
                "specificity_score": 0.75      # TODO: Calculate how specific user preferences are
            },
            "generated_at": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user statistics: {str(e)}"
        )


@router.delete("/{user_id}")
async def delete_user(
    user_id: UUID,
    repository_factory = Depends(get_repository_factory)
):
    """
    Delete a user account (soft delete).
    
    Deactivates the user account while preserving:
    - Historical interaction data
    - Analytics data
    - ML model training data
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        success = await user_repo.delete(user_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found or already deleted"
            )
        
        # Clear all caches for this user
        cache_repo = repository_factory.get_cache_repository()
        await cache_repo.clear_cache(f"*{user_id}*")
        
        logger.info(f"Deleted user {user_id}")
        return {
            "success": True,
            "message": f"User {user_id} has been deactivated",
            "user_id": str(user_id),
            "deleted_at": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete user: {str(e)}"
        )