"""
Recommendation Service - Handles ML-based recommendation logic for rental properties.
"""
from typing import List, Dict, Optional, Tuple, Any
from uuid import UUID
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

from ..entities.property import Property
from ..entities.user import User, UserInteraction
from ..entities.search_query import SearchQuery, SearchFilters
from ..repositories.property_repository import PropertyRepository
from ..repositories.user_repository import UserRepository
from ..repositories.model_repository import ModelRepository


class RecommendationService:
    """Service for handling ML-based property recommendations."""
    
    def __init__(
        self,
        property_repository: PropertyRepository,
        user_repository: UserRepository,
        model_repository: ModelRepository
    ):
        self.property_repository = property_repository
        self.user_repository = user_repository
        self.model_repository = model_repository
    
    async def get_personalized_recommendations(
        self,
        user_id: UUID,
        limit: int = 10,
        exclude_viewed: bool = True,
        exclude_liked: bool = False
    ) -> List[Property]:
        """
        Get personalized property recommendations for a user.
        
        Args:
            user_id: The user's ID
            limit: Maximum number of recommendations
            exclude_viewed: Whether to exclude already viewed properties
            exclude_liked: Whether to exclude already liked properties
            
        Returns:
            List of recommended properties
        """
        # Get user data
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Get user's interaction history
        interactions = await self.user_repository.get_interactions(user_id)
        
        # Get properties to exclude
        exclude_property_ids = set()
        if exclude_viewed:
            exclude_property_ids.update(
                i.property_id for i in interactions if i.interaction_type == 'view'
            )
        if exclude_liked:
            exclude_property_ids.update(
                i.property_id for i in interactions if i.interaction_type == 'like'
            )
        
        # Try multiple recommendation strategies
        recommendations = []
        
        # Strategy 1: Collaborative filtering
        collab_recs = await self._get_collaborative_recommendations(
            user_id, limit, exclude_property_ids
        )
        recommendations.extend(collab_recs)
        
        # Strategy 2: Content-based filtering
        content_recs = await self._get_content_based_recommendations(
            user, limit, exclude_property_ids
        )
        recommendations.extend(content_recs)
        
        # Strategy 3: Hybrid approach
        hybrid_recs = await self._get_hybrid_recommendations(
            user, limit, exclude_property_ids
        )
        recommendations.extend(hybrid_recs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for prop in recommendations:
            if prop.id not in seen:
                seen.add(prop.id)
                unique_recommendations.append(prop)
        
        # Limit results
        return unique_recommendations[:limit]
    
    async def get_similar_user_recommendations(
        self,
        user_id: UUID,
        limit: int = 10
    ) -> List[Property]:
        """
        Get recommendations based on similar users' preferences.
        
        Args:
            user_id: The user's ID
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended properties
        """
        # Get similar users
        similar_users = await self.user_repository.get_similar_users(user_id, limit=20)
        
        if not similar_users:
            return []
        
        # Get properties liked by similar users
        property_scores = defaultdict(float)
        
        for similar_user in similar_users:
            liked_properties = similar_user.get_liked_properties()
            similarity_score = await self._calculate_user_similarity(user_id, similar_user.id)
            
            for property_id in liked_properties:
                property_scores[property_id] += similarity_score
        
        # Sort by score and get top properties
        sorted_properties = sorted(
            property_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get property objects
        property_ids = [prop_id for prop_id, _ in sorted_properties[:limit]]
        properties = await self.property_repository.get_by_ids(property_ids)
        
        return properties
    
    async def get_trending_recommendations(
        self,
        limit: int = 10,
        days_back: int = 7
    ) -> List[Property]:
        """
        Get trending property recommendations based on recent activity.
        
        Args:
            limit: Maximum number of recommendations
            days_back: Number of days to look back for trending analysis
            
        Returns:
            List of trending properties
        """
        cache_key = f"trending_recommendations_{limit}_{days_back}"
        
        # Try to get from cache
        cached_recommendations = await self.model_repository.get_cached_predictions(cache_key)
        if cached_recommendations:
            # Convert back to Property objects
            property_ids = [UUID(prop_id) for prop_id in cached_recommendations]
            return await self.property_repository.get_by_ids(property_ids)
        
        # Calculate trending properties
        trending_properties = await self._calculate_trending_properties(days_back)
        
        # Cache the results
        property_ids = [str(prop.id) for prop in trending_properties[:limit]]
        await self.model_repository.cache_predictions(
            cache_key, property_ids, ttl_seconds=3600
        )
        
        return trending_properties[:limit]
    
    async def get_location_based_recommendations(
        self,
        user_id: UUID,
        location: str,
        limit: int = 10
    ) -> List[Property]:
        """
        Get recommendations for properties in a specific location.
        
        Args:
            user_id: The user's ID
            location: The location to search in
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended properties in the location
        """
        # Get user preferences
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Get properties in the location
        properties = await self.property_repository.get_by_location(location, limit=limit*2)
        
        # Score properties based on user preferences
        scored_properties = []
        for prop in properties:
            score = await self._calculate_user_property_score(user, prop)
            scored_properties.append((prop, score))
        
        # Sort by score and return top results
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return [prop for prop, score in scored_properties[:limit]]
    
    async def get_price_based_recommendations(
        self,
        user_id: UUID,
        min_price: float,
        max_price: float,
        limit: int = 10
    ) -> List[Property]:
        """
        Get recommendations for properties within a price range.
        
        Args:
            user_id: The user's ID
            min_price: Minimum price
            max_price: Maximum price
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended properties in the price range
        """
        # Get user preferences
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Get properties in the price range
        properties = await self.property_repository.get_by_price_range(
            min_price, max_price, limit=limit*2
        )
        
        # Score properties based on user preferences
        scored_properties = []
        for prop in properties:
            score = await self._calculate_user_property_score(user, prop)
            scored_properties.append((prop, score))
        
        # Sort by score and return top results
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return [prop for prop, score in scored_properties[:limit]]
    
    async def get_cold_start_recommendations(
        self,
        user_id: UUID,
        limit: int = 10
    ) -> List[Property]:
        """
        Get recommendations for new users with no interaction history.
        
        Args:
            user_id: The user's ID
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended properties for new users
        """
        # Get user preferences
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Get popular properties as base recommendations
        popular_properties = await self._get_popular_properties(limit * 2)
        
        # If user has preferences, filter based on them
        if user.preferences:
            filtered_properties = []
            for prop in popular_properties:
                if self._matches_user_preferences(prop, user.preferences):
                    filtered_properties.append(prop)
            
            if filtered_properties:
                return filtered_properties[:limit]
        
        # Fall back to general popular properties
        return popular_properties[:limit]
    
    async def update_recommendation_feedback(
        self,
        user_id: UUID,
        property_id: UUID,
        feedback_type: str,
        rating: Optional[float] = None
    ) -> bool:
        """
        Update recommendation feedback for model improvement.
        
        Args:
            user_id: The user's ID
            property_id: The property ID
            feedback_type: Type of feedback ('like', 'dislike', 'view', 'inquiry')
            rating: Optional rating score
            
        Returns:
            True if feedback was recorded successfully
        """
        # Create user interaction
        interaction = UserInteraction.create(
            property_id=property_id,
            interaction_type=feedback_type
        )
        
        # Add interaction to user
        success = await self.user_repository.add_interaction(user_id, interaction)
        
        if success:
            # Update recommendation model with feedback
            await self._update_recommendation_model(user_id, property_id, feedback_type, rating)
        
        return success
    
    async def get_recommendation_explanation(
        self,
        user_id: UUID,
        property_id: UUID
    ) -> Dict[str, Any]:
        """
        Get explanation for why a property was recommended.
        
        Args:
            user_id: The user's ID
            property_id: The property ID
            
        Returns:
            Dictionary containing recommendation explanation
        """
        user = await self.user_repository.get_by_id(user_id)
        property = await self.property_repository.get_by_id(property_id)
        
        if not user or not property:
            return {}
        
        explanation = {
            "property_id": str(property_id),
            "user_id": str(user_id),
            "reasons": [],
            "confidence_score": 0.0
        }
        
        # Analyze preference matches
        preference_matches = self._analyze_preference_matches(user.preferences, property)
        if preference_matches:
            explanation["reasons"].extend(preference_matches)
        
        # Analyze similarity to liked properties
        similarity_reasons = await self._analyze_similarity_reasons(user, property)
        if similarity_reasons:
            explanation["reasons"].extend(similarity_reasons)
        
        # Analyze collaborative filtering reasons
        collab_reasons = await self._analyze_collaborative_reasons(user_id, property)
        if collab_reasons:
            explanation["reasons"].extend(collab_reasons)
        
        # Calculate confidence score
        explanation["confidence_score"] = len(explanation["reasons"]) * 0.2
        
        return explanation
    
    async def _get_collaborative_recommendations(
        self,
        user_id: UUID,
        limit: int,
        exclude_property_ids: set
    ) -> List[Property]:
        """Get recommendations using collaborative filtering."""
        # Get user interaction matrix
        interaction_matrix = await self.user_repository.get_user_interaction_matrix()
        
        if not interaction_matrix:
            return []
        
        # Get similar users
        similar_users = await self.user_repository.get_similar_users(user_id, limit=50)
        
        # Get properties liked by similar users
        property_scores = defaultdict(float)
        
        for similar_user in similar_users:
            similarity_score = await self._calculate_user_similarity(user_id, similar_user.id)
            liked_properties = similar_user.get_liked_properties()
            
            for property_id in liked_properties:
                if property_id not in exclude_property_ids:
                    property_scores[property_id] += similarity_score
        
        # Sort by score and get top properties
        sorted_properties = sorted(
            property_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        property_ids = [prop_id for prop_id, _ in sorted_properties[:limit]]
        return await self.property_repository.get_by_ids(property_ids)
    
    async def _get_content_based_recommendations(
        self,
        user: User,
        limit: int,
        exclude_property_ids: set
    ) -> List[Property]:
        """Get recommendations using content-based filtering."""
        # Get user's liked properties to build profile
        liked_property_ids = user.get_liked_properties()
        
        if not liked_property_ids:
            return []
        
        liked_properties = await self.property_repository.get_by_ids(liked_property_ids)
        
        # Build user profile from liked properties
        user_profile = self._build_user_profile(liked_properties)
        
        # Get all active properties
        all_properties = await self.property_repository.get_all_active(limit=1000)
        
        # Score properties based on user profile
        scored_properties = []
        for prop in all_properties:
            if prop.id not in exclude_property_ids:
                score = self._calculate_content_similarity(user_profile, prop)
                scored_properties.append((prop, score))
        
        # Sort by score and return top results
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return [prop for prop, score in scored_properties[:limit]]
    
    async def _get_hybrid_recommendations(
        self,
        user: User,
        limit: int,
        exclude_property_ids: set
    ) -> List[Property]:
        """Get recommendations using hybrid approach."""
        # Get a mix of strategies
        collab_recs = await self._get_collaborative_recommendations(
            user.id, limit // 2, exclude_property_ids
        )
        content_recs = await self._get_content_based_recommendations(
            user, limit // 2, exclude_property_ids
        )
        
        # Combine and score
        all_recommendations = collab_recs + content_recs
        
        # Score each property using hybrid approach
        scored_properties = []
        for prop in all_recommendations:
            collab_score = 0.5 if prop in collab_recs else 0.0
            content_score = 0.5 if prop in content_recs else 0.0
            preference_score = await self._calculate_user_property_score(user, prop)
            
            hybrid_score = (collab_score * 0.4) + (content_score * 0.3) + (preference_score * 0.3)
            scored_properties.append((prop, hybrid_score))
        
        # Sort by score and return top results
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        
        return [prop for prop, score in scored_properties[:limit]]
    
    async def _calculate_user_similarity(self, user_id1: UUID, user_id2: UUID) -> float:
        """Calculate similarity between two users."""
        # Get user interactions
        interactions1 = await self.user_repository.get_interactions(user_id1)
        interactions2 = await self.user_repository.get_interactions(user_id2)
        
        # Get liked properties for both users
        liked1 = set(i.property_id for i in interactions1 if i.interaction_type == 'like')
        liked2 = set(i.property_id for i in interactions2 if i.interaction_type == 'like')
        
        # Calculate Jaccard similarity
        if not liked1 or not liked2:
            return 0.0
        
        intersection = len(liked1 & liked2)
        union = len(liked1 | liked2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _calculate_trending_properties(self, days_back: int) -> List[Property]:
        """Calculate trending properties based on recent activity."""
        # This is a simplified trending calculation
        # In a real system, you'd analyze view counts, likes, inquiries, etc.
        
        # Get recent active properties
        properties = await self.property_repository.get_all_active(limit=1000)
        
        # Simple trending score based on recent activity
        # This would be replaced with actual analytics data
        trending_scores = []
        for prop in properties:
            # Simulate trending score based on property attributes
            score = 0.0
            
            # More recent properties get higher score
            days_old = (datetime.now() - prop.scraped_at).days
            if days_old <= days_back:
                score += (days_back - days_old) / days_back
            
            # Properties with more amenities get higher score
            score += len(prop.amenities) * 0.1
            
            # Properties with competitive pricing get higher score
            # This would use market data in a real system
            score += 0.5
            
            trending_scores.append((prop, score))
        
        # Sort by score and return
        trending_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [prop for prop, score in trending_scores]
    
    async def _calculate_user_property_score(self, user: User, property: Property) -> float:
        """Calculate how well a property matches a user's preferences."""
        score = 0.0
        preferences = user.preferences
        
        # Price preference
        if preferences.min_price and preferences.max_price:
            if preferences.min_price <= property.price <= preferences.max_price:
                score += 0.3
            else:
                # Penalty for being outside price range
                score -= 0.1
        
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
        
        return max(0.0, score)  # Ensure non-negative score
    
    async def _get_popular_properties(self, limit: int) -> List[Property]:
        """Get popular properties for cold start recommendations."""
        # This would typically use analytics data
        # For now, return recent active properties
        return await self.property_repository.get_all_active(limit=limit)
    
    def _matches_user_preferences(self, property: Property, preferences) -> bool:
        """Check if a property matches user preferences."""
        # Price check
        if preferences.min_price and property.price < preferences.min_price:
            return False
        if preferences.max_price and property.price > preferences.max_price:
            return False
        
        # Bedroom check
        if preferences.min_bedrooms and property.bedrooms < preferences.min_bedrooms:
            return False
        if preferences.max_bedrooms and property.bedrooms > preferences.max_bedrooms:
            return False
        
        # Bathroom check
        if preferences.min_bathrooms and property.bathrooms < preferences.min_bathrooms:
            return False
        if preferences.max_bathrooms and property.bathrooms > preferences.max_bathrooms:
            return False
        
        # Location check
        if preferences.preferred_locations:
            if not any(loc.lower() in property.location.lower() for loc in preferences.preferred_locations):
                return False
        
        # Property type check
        if preferences.property_types:
            if property.property_type not in preferences.property_types:
                return False
        
        return True
    
    def _build_user_profile(self, liked_properties: List[Property]) -> Dict[str, Any]:
        """Build user profile from liked properties."""
        profile = {
            "avg_price": 0.0,
            "common_amenities": [],
            "preferred_locations": [],
            "preferred_property_types": [],
            "avg_bedrooms": 0.0,
            "avg_bathrooms": 0.0
        }
        
        if not liked_properties:
            return profile
        
        # Calculate averages
        profile["avg_price"] = sum(p.price for p in liked_properties) / len(liked_properties)
        profile["avg_bedrooms"] = sum(p.bedrooms for p in liked_properties) / len(liked_properties)
        profile["avg_bathrooms"] = sum(p.bathrooms for p in liked_properties) / len(liked_properties)
        
        # Find common amenities
        amenity_counts = defaultdict(int)
        for prop in liked_properties:
            for amenity in prop.amenities:
                amenity_counts[amenity] += 1
        
        # Get amenities that appear in at least 50% of liked properties
        threshold = len(liked_properties) * 0.5
        profile["common_amenities"] = [
            amenity for amenity, count in amenity_counts.items() 
            if count >= threshold
        ]
        
        # Get preferred locations
        location_counts = defaultdict(int)
        for prop in liked_properties:
            location_counts[prop.location] += 1
        
        profile["preferred_locations"] = [
            location for location, count in location_counts.items()
        ]
        
        # Get preferred property types
        type_counts = defaultdict(int)
        for prop in liked_properties:
            type_counts[prop.property_type] += 1
        
        profile["preferred_property_types"] = [
            prop_type for prop_type, count in type_counts.items()
        ]
        
        return profile
    
    def _calculate_content_similarity(self, user_profile: Dict[str, Any], property: Property) -> float:
        """Calculate content similarity between user profile and property."""
        score = 0.0
        
        # Price similarity
        if user_profile["avg_price"] > 0:
            price_diff = abs(property.price - user_profile["avg_price"])
            price_similarity = 1 - (price_diff / user_profile["avg_price"])
            score += max(0, price_similarity) * 0.3
        
        # Amenity similarity
        if user_profile["common_amenities"]:
            matching_amenities = set(user_profile["common_amenities"]) & set(property.amenities)
            amenity_similarity = len(matching_amenities) / len(user_profile["common_amenities"])
            score += amenity_similarity * 0.3
        
        # Location similarity
        if user_profile["preferred_locations"]:
            location_match = any(
                loc.lower() in property.location.lower() 
                for loc in user_profile["preferred_locations"]
            )
            if location_match:
                score += 0.2
        
        # Property type similarity
        if user_profile["preferred_property_types"]:
            if property.property_type in user_profile["preferred_property_types"]:
                score += 0.2
        
        return score
    
    async def _update_recommendation_model(
        self,
        user_id: UUID,
        property_id: UUID,
        feedback_type: str,
        rating: Optional[float]
    ) -> None:
        """Update recommendation model with new feedback."""
        # This would update ML model weights/parameters
        # For now, just cache the feedback
        feedback_data = {
            "user_id": str(user_id),
            "property_id": str(property_id),
            "feedback_type": feedback_type,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        
        cache_key = f"recommendation_feedback_{user_id}_{property_id}"
        await self.model_repository.cache_predictions(
            cache_key, feedback_data, ttl_seconds=86400
        )
    
    def _analyze_preference_matches(self, preferences, property: Property) -> List[str]:
        """Analyze how property matches user preferences."""
        matches = []
        
        # Price match
        if preferences.min_price and preferences.max_price:
            if preferences.min_price <= property.price <= preferences.max_price:
                matches.append(f"Price ${property.price:,.0f} is within your preferred range")
        
        # Location match
        if preferences.preferred_locations:
            for loc in preferences.preferred_locations:
                if loc.lower() in property.location.lower():
                    matches.append(f"Located in your preferred area: {property.location}")
                    break
        
        # Amenity matches
        if preferences.required_amenities:
            matching_amenities = set(preferences.required_amenities) & set(property.amenities)
            if matching_amenities:
                matches.append(f"Has your preferred amenities: {', '.join(matching_amenities)}")
        
        return matches
    
    async def _analyze_similarity_reasons(self, user: User, property: Property) -> List[str]:
        """Analyze similarity to user's liked properties."""
        reasons = []
        
        # Get liked properties
        liked_property_ids = user.get_liked_properties()
        if not liked_property_ids:
            return reasons
        
        liked_properties = await self.property_repository.get_by_ids(liked_property_ids)
        
        # Find similarities
        for liked_prop in liked_properties:
            if liked_prop.property_type == property.property_type:
                reasons.append(f"Similar to {liked_prop.title} (same property type)")
                break
        
        return reasons
    
    async def _analyze_collaborative_reasons(self, user_id: UUID, property: Property) -> List[str]:
        """Analyze collaborative filtering reasons."""
        reasons = []
        
        # Get users who liked this property
        users_who_liked = await self.user_repository.get_users_who_liked_property(property.id)
        
        if users_who_liked:
            similar_users = await self.user_repository.get_similar_users(user_id, limit=10)
            
            # Check if any similar users liked this property
            similar_user_ids = {u.id for u in similar_users}
            users_who_liked_ids = {u.id for u in users_who_liked}
            
            overlap = similar_user_ids & users_who_liked_ids
            if overlap:
                reasons.append(f"Liked by {len(overlap)} users with similar preferences")
        
        return reasons