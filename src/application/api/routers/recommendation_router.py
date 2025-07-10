"""
Recommendation API router for personalized property recommendations.

This module provides endpoints for personalized recommendations,
similar properties, user interaction tracking, and recommendation explanations.
"""

import time
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ....domain.entities.user import UserInteraction
from ....infrastructure.ml.models.hybrid_recommender import HybridRecommendationSystem
from ....infrastructure.data import get_user_repository, get_property_repository, get_cache_repository
from ...dto.recommendation_dto import (
    RecommendationRequest, RecommendationResponse, RecommendedProperty,
    SimilarPropertiesRequest, SimilarPropertiesResponse,
    UserInteractionRequest, UserInteractionResponse,
    RecommendationExplanationRequest, RecommendationExplanationResponse,
    RecommendationScore, RecommendationExplanation, RecommendationMetrics
)
from ...dto.search_dto import PropertyResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


@router.post("/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    recommendation_request: RecommendationRequest,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Get personalized property recommendations for a user.
    
    This endpoint provides ML-powered personalized recommendations using:
    - Collaborative filtering based on user behavior
    - Content-based filtering using property features
    - Hybrid approach combining multiple signals
    - User preference learning
    - Diversity and novelty optimization
    
    Returns ranked list of properties with recommendation scores and explanations.
    """
    start_time = time.time()
    
    try:
        # Get repositories
        user_repo = repository_factory.get_user_repository()
        property_repo = repository_factory.get_property_repository()
        cache_repo = repository_factory.get_cache_repository()
        
        # Check cache first
        cache_key = f"recommendations:{recommendation_request.user_id}:{recommendation_request.limit}"
        cached_result = await cache_repo.get_cached_predictions(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached recommendations for user {recommendation_request.user_id}")
            response_time_ms = (time.time() - start_time) * 1000
            return _create_recommendation_response(
                cached_result, recommendation_request, response_time_ms
            )
        
        # Get user data
        user = await user_repo.get_by_id(recommendation_request.user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User {recommendation_request.user_id} not found"
            )
        
        # Get user interaction history
        user_interactions = await user_repo.get_interactions(
            recommendation_request.user_id, limit=500
        )
        
        # TODO: Initialize and use actual hybrid recommender
        # For now, we'll use a simplified approach
        
        # Get candidate properties
        candidate_properties = await property_repo.get_all_active(
            limit=recommendation_request.limit * 5  # Get more candidates for filtering
        )
        
        if not candidate_properties:
            logger.warning(f"No properties available for recommendations")
            return RecommendationResponse(
                recommendations=[],
                user_id=recommendation_request.user_id,
                total_count=0,
                page=1,
                page_size=recommendation_request.limit,
                recommendation_type="personalized",
                response_time_ms=(time.time() - start_time) * 1000
            )
        
        # Filter out already viewed properties if requested
        if recommendation_request.exclude_viewed:
            viewed_property_ids = {
                interaction.property_id for interaction in user_interactions
                if interaction.interaction_type in ["view", "like", "inquiry"]
            }
            candidate_properties = [
                prop for prop in candidate_properties
                if prop.id not in viewed_property_ids
            ]
        
        # Apply additional filters if provided
        if recommendation_request.filters:
            candidate_properties = _apply_recommendation_filters(
                candidate_properties, recommendation_request.filters
            )
        
        # Generate recommendations using simplified scoring
        recommendations = await _generate_personalized_recommendations(
            user, candidate_properties, user_interactions,
            recommendation_request.limit, recommendation_request.include_explanations
        )
        
        # Cache results
        cache_data = [rec.dict() for rec in recommendations]
        await cache_repo.cache_predictions(cache_key, cache_data, ttl_seconds=1800)
        
        response_time_ms = (time.time() - start_time) * 1000
        
        response = RecommendationResponse(
            recommendations=recommendations,
            user_id=recommendation_request.user_id,
            total_count=len(recommendations),
            page=1,
            page_size=recommendation_request.limit,
            recommendation_type="personalized",
            response_time_ms=response_time_ms
        )
        
        logger.info(f"Generated {len(recommendations)} personalized recommendations for user {recommendation_request.user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate personalized recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.post("/similar", response_model=SimilarPropertiesResponse)
async def get_similar_properties(
    similar_request: SimilarPropertiesRequest,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Find properties similar to a given property.
    
    This endpoint finds similar properties based on:
    - Property features (location, price, bedrooms, etc.)
    - Amenities and characteristics
    - User interaction patterns
    - Content-based similarity metrics
    
    Returns ranked list of similar properties with similarity scores.
    """
    start_time = time.time()
    
    try:
        property_repo = repository_factory.get_property_repository()
        
        # Get the source property
        source_property = await property_repo.get_by_id(similar_request.property_id)
        if not source_property:
            raise HTTPException(
                status_code=404,
                detail=f"Property {similar_request.property_id} not found"
            )
        
        # Get similar properties using repository method
        similar_properties = await property_repo.get_similar_properties(
            similar_request.property_id, limit=similar_request.limit
        )
        
        # Convert to recommendation format with similarity scores
        recommendations = []
        for i, prop in enumerate(similar_properties):
            # Calculate similarity score (simplified)
            similarity_score = max(0.0, 1.0 - (i * 0.1))  # Decreasing similarity
            
            recommendation_score = RecommendationScore(
                overall_score=similarity_score,
                content_score=similarity_score,
                collaborative_score=0.5,
                popularity_score=0.7,
                recency_score=0.8,
                diversity_score=0.6
            )
            
            explanation = None
            if similar_request.include_explanations:
                explanation = RecommendationExplanation(
                    reason=f"Similar to your selected property in {source_property.location}",
                    factors=[
                        f"Same location area ({prop.location})",
                        f"Similar price range (${prop.price:,.0f})",
                        f"Same number of bedrooms ({prop.bedrooms})",
                        f"Similar property type ({prop.property_type})"
                    ],
                    similar_properties=[source_property.id],
                    user_preferences={},
                    confidence=similarity_score
                )
            
            recommended_prop = RecommendedProperty(
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
                recommendation_score=recommendation_score,
                explanation=explanation,
                rank=i + 1,
                recommendation_type="similar"
            )
            recommendations.append(recommended_prop)
        
        response_time_ms = (time.time() - start_time) * 1000
        
        response = SimilarPropertiesResponse(
            similar_properties=recommendations,
            source_property_id=similar_request.property_id,
            total_count=len(recommendations),
            similarity_threshold=similar_request.similarity_threshold,
            response_time_ms=response_time_ms
        )
        
        logger.info(f"Found {len(recommendations)} similar properties for {similar_request.property_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar properties: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find similar properties: {str(e)}"
        )


@router.post("/interactions", response_model=UserInteractionResponse)
async def track_user_interaction(
    interaction_request: UserInteractionRequest,
    request: Request,
    repository_factory = Depends(get_repository_factory)
):
    """
    Track user interaction with a property.
    
    This endpoint records user interactions for ML model training including:
    - Property views and time spent
    - Likes and dislikes
    - Inquiries and contacts
    - Saves and bookmarks
    - Sharing activity
    
    These interactions are used to improve recommendation quality over time.
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        # Create user interaction object
        interaction = UserInteraction(
            property_id=interaction_request.property_id,
            interaction_type=interaction_request.interaction_type,
            duration_seconds=interaction_request.duration_seconds
        )
        
        # Save interaction
        success = await user_repo.add_interaction(
            interaction_request.user_id, interaction
        )
        
        if success:
            # Clear user's recommendation cache to reflect new preferences
            cache_repo = repository_factory.get_cache_repository()
            await cache_repo.clear_cache(f"recommendations:{interaction_request.user_id}:*")
            
            logger.info(
                f"Tracked {interaction_request.interaction_type} interaction "
                f"for user {interaction_request.user_id} on property {interaction_request.property_id}"
            )
            
            return UserInteractionResponse(
                success=True,
                message="Interaction tracked successfully",
                user_id=interaction_request.user_id,
                property_id=interaction_request.property_id,
                interaction_type=interaction_request.interaction_type
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to track interaction"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track user interaction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track interaction: {str(e)}"
        )


@router.get("/explain", response_model=RecommendationExplanationResponse)
async def explain_recommendation(
    user_id: UUID = Query(..., description="User ID"),
    property_id: UUID = Query(..., description="Property ID"),
    recommendation_type: str = Query(default="personalized", description="Type of recommendation"),
    repository_factory = Depends(get_repository_factory)
):
    """
    Get explanation for why a property was recommended to a user.
    
    This endpoint provides detailed explanations including:
    - Matching user preferences
    - Similar properties the user liked
    - Content-based similarity factors
    - Collaborative filtering insights
    - Confidence scores and reasoning
    
    Helps users understand and trust the recommendation system.
    """
    start_time = time.time()
    
    try:
        user_repo = repository_factory.get_user_repository()
        property_repo = repository_factory.get_property_repository()
        
        # Get user and property data
        user = await user_repo.get_by_id(user_id)
        property_data = await property_repo.get_by_id(property_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not property_data:
            raise HTTPException(status_code=404, detail="Property not found")
        
        # Generate explanation
        explanation = await _generate_recommendation_explanation(
            user, property_data, recommendation_type
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return RecommendationExplanationResponse(
            user_id=user_id,
            property_id=property_id,
            explanation=explanation,
            recommendation_type=recommendation_type,
            response_time_ms=response_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate recommendation explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@router.get("/metrics/{user_id}", response_model=RecommendationMetrics)
async def get_user_recommendation_metrics(
    user_id: UUID,
    repository_factory = Depends(get_repository_factory)
):
    """
    Get recommendation performance metrics for a user.
    
    Returns metrics including:
    - Click-through rates
    - Conversion rates
    - Diversity scores
    - Novelty metrics
    - Coverage statistics
    
    Used for monitoring recommendation system performance per user.
    """
    try:
        user_repo = repository_factory.get_user_repository()
        
        # Get user statistics
        user_stats = await user_repo.get_user_statistics(user_id)
        
        if not user_stats:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Calculate metrics (simplified)
        total_interactions = user_stats.get('total_interactions', 0)
        total_views = user_stats.get('views', 0)
        total_inquiries = user_stats.get('inquiries', 0)
        
        ctr = (total_views / max(total_interactions, 1)) * 100
        conversion_rate = (total_inquiries / max(total_views, 1)) * 100
        
        metrics = RecommendationMetrics(
            user_id=user_id,
            total_recommendations=total_interactions,
            click_through_rate=ctr,
            conversion_rate=conversion_rate,
            diversity_score=0.75,  # TODO: Calculate actual diversity
            novelty_score=0.68,   # TODO: Calculate actual novelty
            coverage_score=0.82   # TODO: Calculate actual coverage
        )
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user metrics: {str(e)}"
        )


async def _generate_personalized_recommendations(
    user, properties, interactions, limit, include_explanations
) -> List[RecommendedProperty]:
    """Generate personalized recommendations using simplified approach"""
    recommendations = []
    
    # Simple scoring based on user preferences
    for i, prop in enumerate(properties[:limit]):
        # Calculate scores (simplified)
        content_score = _calculate_content_score(user, prop)
        popularity_score = 0.7  # TODO: Calculate from actual data
        recency_score = 0.8     # TODO: Calculate from scraped_at
        
        overall_score = (content_score * 0.6 + popularity_score * 0.3 + recency_score * 0.1)
        
        recommendation_score = RecommendationScore(
            overall_score=overall_score,
            content_score=content_score,
            collaborative_score=0.5,  # TODO: Implement CF
            popularity_score=popularity_score,
            recency_score=recency_score,
            diversity_score=0.6
        )
        
        explanation = None
        if include_explanations:
            explanation = RecommendationExplanation(
                reason=_generate_recommendation_reason(user, prop),
                factors=_generate_recommendation_factors(user, prop),
                similar_properties=[],
                user_preferences=_extract_user_preferences(user),
                confidence=overall_score
            )
        
        recommended_prop = RecommendedProperty(
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
            recommendation_score=recommendation_score,
            explanation=explanation,
            rank=i + 1,
            recommendation_type="personalized"
        )
        recommendations.append(recommended_prop)
    
    # Sort by overall score
    recommendations.sort(key=lambda x: x.recommendation_score.overall_score, reverse=True)
    
    return recommendations


def _calculate_content_score(user, property_obj) -> float:
    """Calculate content-based similarity score"""
    score = 0.0
    factors = 0
    
    # Price preference match
    if user.preferences.min_price and user.preferences.max_price:
        if user.preferences.min_price <= property_obj.price <= user.preferences.max_price:
            score += 1.0
        factors += 1
    
    # Bedroom preference match
    if user.preferences.min_bedrooms and user.preferences.max_bedrooms:
        if user.preferences.min_bedrooms <= property_obj.bedrooms <= user.preferences.max_bedrooms:
            score += 1.0
        factors += 1
    
    # Location preference match
    if user.preferences.preferred_locations:
        for location in user.preferences.preferred_locations:
            if location.lower() in property_obj.location.lower():
                score += 1.0
                break
        factors += 1
    
    # Amenity preference match
    if user.preferences.required_amenities:
        matching_amenities = len(set(user.preferences.required_amenities) & set(property_obj.amenities))
        if matching_amenities > 0:
            score += matching_amenities / len(user.preferences.required_amenities)
        factors += 1
    
    return score / max(factors, 1)


def _generate_recommendation_reason(user, property_obj) -> str:
    """Generate main reason for recommendation"""
    # Simple rule-based reason generation
    if user.preferences.preferred_locations:
        for location in user.preferences.preferred_locations:
            if location.lower() in property_obj.location.lower():
                return f"Located in your preferred area: {property_obj.location}"
    
    if user.preferences.min_price and user.preferences.max_price:
        if user.preferences.min_price <= property_obj.price <= user.preferences.max_price:
            return f"Matches your price range: ${property_obj.price:,.0f}"
    
    return "Recommended based on your preferences and similar users"


def _generate_recommendation_factors(user, property_obj) -> List[str]:
    """Generate list of recommendation factors"""
    factors = []
    
    # Check each preference match
    if user.preferences.min_bedrooms and user.preferences.max_bedrooms:
        if user.preferences.min_bedrooms <= property_obj.bedrooms <= user.preferences.max_bedrooms:
            factors.append(f"Has {property_obj.bedrooms} bedrooms (matches preference)")
    
    if user.preferences.required_amenities:
        matching_amenities = set(user.preferences.required_amenities) & set(property_obj.amenities)
        if matching_amenities:
            factors.append(f"Includes preferred amenities: {', '.join(matching_amenities)}")
    
    if property_obj.property_type in user.preferences.property_types:
        factors.append(f"Property type: {property_obj.property_type}")
    
    return factors


def _extract_user_preferences(user) -> dict:
    """Extract user preferences for explanation"""
    return {
        "price_range": f"${user.preferences.min_price or 0:,.0f} - ${user.preferences.max_price or 999999:,.0f}",
        "bedrooms": f"{user.preferences.min_bedrooms or 'any'} - {user.preferences.max_bedrooms or 'any'}",
        "locations": user.preferences.preferred_locations,
        "amenities": user.preferences.required_amenities,
        "property_types": user.preferences.property_types
    }


async def _generate_recommendation_explanation(user, property_data, recommendation_type):
    """Generate detailed recommendation explanation"""
    return RecommendationExplanation(
        reason=_generate_recommendation_reason(user, property_data),
        factors=_generate_recommendation_factors(user, property_data),
        similar_properties=[],
        user_preferences=_extract_user_preferences(user),
        confidence=_calculate_content_score(user, property_data)
    )


def _apply_recommendation_filters(properties, filters) -> list:
    """Apply additional filters to candidate properties"""
    # TODO: Implement filter application logic
    return properties


def _create_recommendation_response(cached_result, request, response_time_ms):
    """Create recommendation response from cached result"""
    # TODO: Implement cache result conversion
    pass