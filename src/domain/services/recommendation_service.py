from typing import List, Optional, Dict, Tuple
from uuid import UUID

from ..entities.property import Property
from ..entities.user import User, UserInteraction
from ..repositories.property_repository import PropertyRepository
from ..repositories.user_repository import UserRepository
from ..repositories.model_repository import ModelRepository


class RecommendationService:
    def __init__(self, property_repository: PropertyRepository, 
                 user_repository: UserRepository, model_repository: ModelRepository):
        self.property_repository = property_repository
        self.user_repository = user_repository
        self.model_repository = model_repository
    
    async def get_recommendations_for_user(self, user_id: UUID, limit: int = 10) -> List[Dict]:
        """
        Get personalized property recommendations for a user
        Returns list of dicts with property and recommendation metadata
        """
        # Get user data
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Check cache first
        cache_key = f"user_recommendations:{user_id}:{limit}"
        cached_recommendations = await self.model_repository.get_cached_predictions(cache_key)
        
        if cached_recommendations:
            return cached_recommendations
        
        # Get user interactions
        interactions = await self.user_repository.get_interactions(user_id)
        
        # Get properties user has already interacted with
        interacted_property_ids = {interaction.property_id for interaction in interactions}
        
        # Generate recommendations based on user preferences and interactions
        recommendations = []
        
        # 1. Content-based recommendations based on user preferences
        content_based_recs = await self._get_content_based_recommendations(user, interacted_property_ids, limit // 2)
        recommendations.extend(content_based_recs)
        
        # 2. Collaborative filtering recommendations
        if len(interactions) > 0:  # Only if user has interaction history
            collaborative_recs = await self._get_collaborative_recommendations(user_id, interacted_property_ids, limit // 2)
            recommendations.extend(collaborative_recs)
        
        # 3. If we don't have enough recommendations, add popular properties
        if len(recommendations) < limit:
            popular_recs = await self._get_popular_recommendations(interacted_property_ids, limit - len(recommendations))
            recommendations.extend(popular_recs)
        
        # Remove duplicates and limit results
        seen_property_ids = set()
        final_recommendations = []
        for rec in recommendations:
            if rec['property_id'] not in seen_property_ids:
                seen_property_ids.add(rec['property_id'])
                final_recommendations.append(rec)
                if len(final_recommendations) >= limit:
                    break
        
        # Cache the results
        await self.model_repository.cache_predictions(cache_key, final_recommendations, ttl_seconds=3600)
        
        return final_recommendations
    
    async def get_similar_properties(self, property_id: UUID, limit: int = 5) -> List[Dict]:
        """Get properties similar to the given property"""
        # Check cache first
        cache_key = f"similar_properties:{property_id}:{limit}"
        cached_similar = await self.model_repository.get_cached_predictions(cache_key)
        
        if cached_similar:
            return cached_similar
        
        # Get the base property
        base_property = await self.property_repository.get_by_id(property_id)
        if not base_property:
            raise ValueError(f"Property {property_id} not found")
        
        # Get similar properties from repository
        similar_properties = await self.property_repository.get_similar_properties(property_id, limit)
        
        # Format as recommendation results
        similar_recommendations = []
        for prop in similar_properties:
            similar_recommendations.append({
                'property_id': prop.id,
                'property': prop,
                'score': 0.8,  # Default similarity score
                'reason': 'similar_properties',
                'explanation': f"Similar to {base_property.title}"
            })
        
        # Cache the results
        await self.model_repository.cache_predictions(cache_key, similar_recommendations, ttl_seconds=3600)
        
        return similar_recommendations
    
    async def record_user_interaction(self, user_id: UUID, property_id: UUID, 
                                    interaction_type: str, duration_seconds: Optional[int] = None):
        """Record a user interaction with a property"""
        # Validate interaction type
        valid_types = ['view', 'like', 'dislike', 'inquiry', 'save', 'share']
        if interaction_type not in valid_types:
            raise ValueError(f"Invalid interaction type: {interaction_type}")
        
        # Create interaction
        interaction = UserInteraction.create(property_id, interaction_type, duration_seconds)
        
        # Save interaction
        await self.user_repository.add_interaction(user_id, interaction)
        
        # Invalidate user's recommendation cache
        cache_pattern = f"user_recommendations:{user_id}:*"
        await self.model_repository.clear_cache(cache_pattern)
    
    async def _get_content_based_recommendations(self, user: User, 
                                               excluded_property_ids: set, limit: int) -> List[Dict]:
        """Get content-based recommendations based on user preferences"""
        recommendations = []
        
        # Build search criteria based on user preferences
        preferences = user.preferences
        
        # Get properties matching user preferences
        if preferences.preferred_locations:
            for location in preferences.preferred_locations[:2]:  # Limit to 2 locations
                location_properties = await self.property_repository.get_by_location(location, limit)
                for prop in location_properties:
                    if prop.id not in excluded_property_ids:
                        recommendations.append({
                            'property_id': prop.id,
                            'property': prop,
                            'score': 0.7,
                            'reason': 'preferred_location',
                            'explanation': f"Matches your preferred location: {location}"
                        })
        
        # Get properties in user's price range
        if preferences.min_price is not None or preferences.max_price is not None:
            min_price = preferences.min_price or 0
            max_price = preferences.max_price or float('inf')
            price_properties = await self.property_repository.get_by_price_range(min_price, max_price, limit)
            for prop in price_properties:
                if prop.id not in excluded_property_ids:
                    recommendations.append({
                        'property_id': prop.id,
                        'property': prop,
                        'score': 0.6,
                        'reason': 'price_range',
                        'explanation': f"Within your budget: ${prop.price:,.0f}"
                    })
        
        return recommendations[:limit]
    
    async def _get_collaborative_recommendations(self, user_id: UUID, 
                                               excluded_property_ids: set, limit: int) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        recommendations = []
        
        # Get similar users
        similar_users = await self.user_repository.get_similar_users(user_id, 10)
        
        # Get properties liked by similar users
        for similar_user in similar_users:
            liked_properties = similar_user.get_liked_properties()
            for property_id in liked_properties:
                if property_id not in excluded_property_ids:
                    property_obj = await self.property_repository.get_by_id(property_id)
                    if property_obj:
                        recommendations.append({
                            'property_id': property_id,
                            'property': property_obj,
                            'score': 0.8,
                            'reason': 'similar_users',
                            'explanation': f"Liked by users with similar preferences"
                        })
        
        return recommendations[:limit]
    
    async def _get_popular_recommendations(self, excluded_property_ids: set, limit: int) -> List[Dict]:
        """Get popular properties as fallback recommendations"""
        recommendations = []
        
        # Get active properties (this would be enhanced with actual popularity metrics)
        popular_properties = await self.property_repository.get_all_active(limit * 2)
        
        for prop in popular_properties:
            if prop.id not in excluded_property_ids:
                recommendations.append({
                    'property_id': prop.id,
                    'property': prop,
                    'score': 0.5,
                    'reason': 'popular',
                    'explanation': "Popular property in your area"
                })
                
                if len(recommendations) >= limit:
                    break
        
        return recommendations
    
    async def get_recommendation_explanation(self, user_id: UUID, property_id: UUID) -> Dict:
        """Get detailed explanation for why a property was recommended"""
        user = await self.user_repository.get_by_id(user_id)
        property_obj = await self.property_repository.get_by_id(property_id)
        
        if not user or not property_obj:
            return {'explanation': 'No explanation available'}
        
        # Analyze why this property matches user preferences
        explanations = []
        
        # Check location match
        if property_obj.location in user.preferences.preferred_locations:
            explanations.append(f"Located in your preferred area: {property_obj.location}")
        
        # Check price range
        if (user.preferences.min_price is None or property_obj.price >= user.preferences.min_price) and \
           (user.preferences.max_price is None or property_obj.price <= user.preferences.max_price):
            explanations.append(f"Within your budget range")
        
        # Check bedrooms
        if (user.preferences.min_bedrooms is None or property_obj.bedrooms >= user.preferences.min_bedrooms) and \
           (user.preferences.max_bedrooms is None or property_obj.bedrooms <= user.preferences.max_bedrooms):
            explanations.append(f"Has {property_obj.bedrooms} bedrooms as preferred")
        
        # Check amenities
        matching_amenities = set(property_obj.amenities) & set(user.preferences.required_amenities)
        if matching_amenities:
            explanations.append(f"Includes desired amenities: {', '.join(matching_amenities)}")
        
        return {
            'property_id': property_id,
            'user_id': user_id,
            'explanations': explanations,
            'overall_explanation': '; '.join(explanations) if explanations else 'Recommended based on general preferences'
        }