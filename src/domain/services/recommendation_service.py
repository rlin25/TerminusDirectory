from typing import List, Optional, Dict, Tuple, Any, Union
from uuid import UUID
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib

from ..entities.property import Property
from ..entities.user import User, UserInteraction, UserPreferences
from ..repositories.property_repository import PropertyRepository
from ..repositories.user_repository import UserRepository
from ..repositories.model_repository import ModelRepository

# Import ML models
try:
    from ...infrastructure.ml.models.collaborative_filter import (
        CollaborativeFilteringModel, RecommendationResult, DataPreprocessor, 
        ModelEvaluator, ColdStartHandler, TrainingConfig
    )
    from ...infrastructure.ml.models.content_recommender import (
        ContentBasedRecommender, PropertyFeatures, SimilarityConfig, FeatureConfig
    )
    from ...infrastructure.ml.models.hybrid_recommender import (
        HybridRecommender, FusionMethod, ColdStartStrategy, UserContext,
        HybridRecommendationResult
    )
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML models not available: {e}. Using fallback implementation.")
    ML_MODELS_AVAILABLE = False


class RecommendationStrategy(Enum):
    """Enumeration of recommendation strategies"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    POPULARITY = "popularity"
    COLD_START = "cold_start"


@dataclass
class RecommendationConfig:
    """Configuration for recommendation service"""
    default_strategy: RecommendationStrategy = RecommendationStrategy.HYBRID
    cf_weight: float = 0.6
    cb_weight: float = 0.4
    hybrid_fusion_method: str = "weighted_average"
    enable_diversity: bool = True
    diversity_lambda: float = 0.2
    enable_novelty: bool = True
    novelty_lambda: float = 0.1
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 10000
    cold_start_threshold: int = 5  # Min interactions to use CF
    similarity_threshold: float = 0.1
    enable_explanations: bool = True
    enable_confidence_scoring: bool = True
    parallel_execution: bool = True
    max_workers: int = 4


@dataclass
class RecommendationMetrics:
    """Metrics for recommendation performance"""
    total_recommendations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time_ms: float = 0.0
    cf_usage_count: int = 0
    cb_usage_count: int = 0
    hybrid_usage_count: int = 0
    cold_start_count: int = 0
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class RecommendationService:
    def __init__(self, 
                 property_repository: PropertyRepository, 
                 user_repository: UserRepository, 
                 model_repository: ModelRepository,
                 config: Optional[RecommendationConfig] = None):
        self.property_repository = property_repository
        self.user_repository = user_repository
        self.model_repository = model_repository
        self.config = config or RecommendationConfig()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models
        self.cf_model = None
        self.cb_model = None
        self.hybrid_model = None
        
        # Initialize utilities
        self.data_preprocessor = None
        self.model_evaluator = None
        self.cold_start_handler = None
        
        # Performance tracking
        self.metrics = RecommendationMetrics()
        self.cache = {}
        self.cache_access_times = deque(maxlen=self.config.max_cache_size)
        
        # Thread pool for parallel execution
        if self.config.parallel_execution:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = None
            
        # Initialize models asynchronously
        self._models_initialized = False
        self._initialization_lock = asyncio.Lock()

    async def get_recommendations_for_user(self, 
                                          user_id: UUID, 
                                          limit: int = 10,
                                          strategy: Optional[RecommendationStrategy] = None,
                                          context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Get personalized property recommendations for a user using advanced ML models
        
        Args:
            user_id: The user ID to get recommendations for
            limit: Maximum number of recommendations to return
            strategy: Override the default recommendation strategy
            context: Additional context for recommendations (location, time, etc.)
            
        Returns:
            List of recommendation dictionaries with metadata
        """
        start_time = time.time()
        
        try:
            # Initialize models if needed
            await self._ensure_models_initialized()
            
            # Get user data
            user = await self.user_repository.get_by_id(user_id)
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            # Check cache first
            cache_key = self._generate_cache_key(user_id, limit, strategy, context)
            cached_recommendations = await self._get_cached_recommendations(cache_key)
            
            if cached_recommendations:
                self.metrics.cache_hits += 1
                return cached_recommendations
            
            self.metrics.cache_misses += 1
            
            # Get user interactions and determine strategy
            interactions = await self.user_repository.get_interactions(user_id)
            effective_strategy = strategy or self._determine_strategy(user, interactions)
            
            # Get properties user has already interacted with
            interacted_property_ids = {interaction.property_id for interaction in interactions}
            
            # Generate recommendations based on strategy
            recommendations = await self._generate_recommendations(
                user=user,
                interactions=interactions,
                strategy=effective_strategy,
                limit=limit,
                interacted_property_ids=interacted_property_ids,
                context=context
            )
            
            # Post-process recommendations
            final_recommendations = await self._post_process_recommendations(
                recommendations=recommendations,
                user=user,
                limit=limit,
                interacted_property_ids=interacted_property_ids
            )
            
            # Cache the results
            await self._cache_recommendations(cache_key, final_recommendations)
            
            # Update metrics
            self.metrics.total_recommendations += 1
            response_time = (time.time() - start_time) * 1000
            self._update_response_time_metric(response_time)
            
            return final_recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed for user {user_id}: {e}")
            self.metrics.error_count += 1
            
            # Fallback to simple recommendations
            return await self._get_fallback_recommendations(user_id, limit)

    async def get_similar_properties(self, 
                                   property_id: UUID, 
                                   limit: int = 5,
                                   similarity_method: str = "hybrid") -> List[Dict]:
        """Get properties similar to the given property using advanced similarity computation"""
        try:
            await self._ensure_models_initialized()
            
            # Check cache first
            cache_key = f"similar_properties:{property_id}:{limit}:{similarity_method}"
            cached_similar = await self._get_cached_recommendations(cache_key)
            
            if cached_similar:
                return cached_similar
            
            # Get the base property
            base_property = await self.property_repository.get_by_id(property_id)
            if not base_property:
                raise ValueError(f"Property {property_id} not found")
            
            # Use content-based model if available for better similarity
            if self.cb_model and similarity_method in ["content", "hybrid"]:
                similar_recommendations = await self._get_content_similar_properties(
                    base_property, limit
                )
            else:
                # Fallback to repository method
                similar_properties = await self.property_repository.get_similar_properties(property_id, limit)
                similar_recommendations = self._format_similar_properties(
                    similar_properties, base_property
                )
            
            # Cache the results
            await self._cache_recommendations(cache_key, similar_recommendations)
            
            return similar_recommendations
            
        except Exception as e:
            self.logger.error(f"Similar properties generation failed for {property_id}: {e}")
            # Fallback to repository method
            similar_properties = await self.property_repository.get_similar_properties(property_id, limit)
            base_property = await self.property_repository.get_by_id(property_id)
            return self._format_similar_properties(similar_properties, base_property)

    async def record_user_interaction(self, 
                                     user_id: UUID, 
                                     property_id: UUID, 
                                     interaction_type: str, 
                                     duration_seconds: Optional[int] = None,
                                     metadata: Optional[Dict[str, Any]] = None):
        """Record a user interaction with a property and update models"""
        try:
            # Validate interaction type
            valid_types = ['view', 'like', 'dislike', 'inquiry', 'save', 'share', 'contact']
            if interaction_type not in valid_types:
                raise ValueError(f"Invalid interaction type: {interaction_type}")
            
            # Create interaction
            interaction = UserInteraction.create(property_id, interaction_type, duration_seconds)
            
            # Save interaction
            await self.user_repository.add_interaction(user_id, interaction)
            
            # Update models with new interaction if available
            if self._models_initialized:
                await self._update_models_with_interaction(
                    user_id, property_id, interaction_type, duration_seconds
                )
            
            # Invalidate user's recommendation cache
            await self._invalidate_user_cache(user_id)
            
            # Log interaction for analytics
            self.logger.info(
                f"Recorded interaction: user={user_id}, property={property_id}, "
                f"type={interaction_type}, duration={duration_seconds}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record interaction: {e}")
            raise

    async def get_recommendation_explanation(self, 
                                           user_id: UUID, 
                                           property_id: UUID,
                                           detailed: bool = False) -> Dict:
        """Get comprehensive explanation for why a property was recommended"""
        try:
            await self._ensure_models_initialized()
            
            user = await self.user_repository.get_by_id(user_id)
            property_obj = await self.property_repository.get_by_id(property_id)
            
            if not user or not property_obj:
                return {'explanation': 'No explanation available', 'available': False}
            
            # Generate multi-layered explanation
            explanation = {
                'property_id': str(property_id),
                'user_id': str(user_id),
                'available': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Basic preference matching
            basic_explanations = self._generate_basic_explanations(user, property_obj)
            explanation['basic_match'] = basic_explanations
            
            # Advanced ML model explanations
            if detailed and self._models_initialized:
                ml_explanations = await self._generate_ml_explanations(
                    user_id, property_id, user, property_obj
                )
                explanation.update(ml_explanations)
            
            # Generate overall explanation
            explanation['overall_explanation'] = self._generate_overall_explanation(
                basic_explanations, explanation.get('ml_explanations', {})
            )
            
            # Add confidence and feature importance
            explanation['confidence_score'] = self._calculate_explanation_confidence(
                basic_explanations, explanation.get('ml_explanations', {})
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {
                'explanation': 'Explanation temporarily unavailable',
                'available': False,
                'error': str(e)
            }

    # === MODEL INITIALIZATION AND MANAGEMENT ===
    
    async def _ensure_models_initialized(self):
        """Ensure ML models are initialized"""
        if self._models_initialized:
            return
        
        async with self._initialization_lock:
            if self._models_initialized:
                return
            
            try:
                await self._initialize_models()
                self._models_initialized = True
                self.logger.info("ML models initialized successfully")
            except Exception as e:
                self.logger.warning(f"ML model initialization failed: {e}. Using fallback methods.")
    
    async def _initialize_models(self):
        """Initialize all ML models"""
        if not ML_MODELS_AVAILABLE:
            self.logger.warning("ML models not available, using fallback implementations")
            return
            
        try:
            # Initialize data preprocessor
            self.data_preprocessor = DataPreprocessor(self.logger)
            
            # Initialize model evaluator
            self.model_evaluator = ModelEvaluator(self.logger)
            
            # Initialize cold start handler
            self.cold_start_handler = ColdStartHandler(self.logger)
            
            # Load or train collaborative filtering model
            await self._initialize_collaborative_model()
            
            # Load or train content-based model
            await self._initialize_content_model()
            
            # Initialize hybrid model
            await self._initialize_hybrid_model()
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    async def _initialize_collaborative_model(self):
        """Initialize collaborative filtering model"""
        try:
            # Try to load existing model
            model_data = await self.model_repository.load_model("collaborative_filter", "latest")
            
            if model_data:
                self.cf_model = model_data
                self.logger.info("Loaded existing collaborative filtering model")
            else:
                # Train new model if data is available
                interaction_matrix = await self.user_repository.get_user_interaction_matrix()
                if interaction_matrix:
                    await self._train_collaborative_model(interaction_matrix)
                else:
                    self.logger.warning("No interaction data available for collaborative filtering")
                    
        except Exception as e:
            self.logger.error(f"Collaborative model initialization failed: {e}")
    
    async def _initialize_content_model(self):
        """Initialize content-based model"""
        try:
            # Try to load existing model
            model_data = await self.model_repository.load_model("content_recommender", "latest")
            
            if model_data:
                self.cb_model = model_data
                self.logger.info("Loaded existing content-based model")
            else:
                # Initialize new content model
                if ML_MODELS_AVAILABLE:
                    config = FeatureConfig()
                    similarity_config = SimilarityConfig()
                    self.cb_model = ContentBasedRecommender(
                        feature_config=config,
                        similarity_config=similarity_config,
                        logger=self.logger
                    )
                    
                    # Train with available property data
                    await self._train_content_model()
                
        except Exception as e:
            self.logger.error(f"Content model initialization failed: {e}")
    
    async def _initialize_hybrid_model(self):
        """Initialize hybrid model"""
        try:
            if (self.cf_model or self.cb_model) and ML_MODELS_AVAILABLE:
                self.hybrid_model = HybridRecommender(
                    collaborative_model=self.cf_model,
                    content_model=self.cb_model,
                    logger=self.logger
                )
                self.logger.info("Initialized hybrid recommender")
            else:
                self.logger.warning("No base models available for hybrid recommender")
                
        except Exception as e:
            self.logger.error(f"Hybrid model initialization failed: {e}")

    # === RECOMMENDATION GENERATION ===
    
    def _determine_strategy(self, user: User, interactions: List[UserInteraction]) -> RecommendationStrategy:
        """Determine the best recommendation strategy for a user"""
        num_interactions = len(interactions)
        
        if num_interactions < self.config.cold_start_threshold:
            return RecommendationStrategy.COLD_START
        elif self.hybrid_model and self._models_initialized:
            return RecommendationStrategy.HYBRID
        elif self.cf_model and num_interactions >= 10:
            return RecommendationStrategy.COLLABORATIVE_FILTERING
        elif self.cb_model:
            return RecommendationStrategy.CONTENT_BASED
        else:
            return RecommendationStrategy.POPULARITY
    
    async def _generate_recommendations(self,
                                      user: User,
                                      interactions: List[UserInteraction],
                                      strategy: RecommendationStrategy,
                                      limit: int,
                                      interacted_property_ids: set,
                                      context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Generate recommendations based on strategy"""
        
        if strategy == RecommendationStrategy.HYBRID and self.hybrid_model:
            return await self._get_hybrid_recommendations(
                user, interactions, interacted_property_ids, limit, context
            )
        elif strategy == RecommendationStrategy.COLLABORATIVE_FILTERING:
            self.metrics.cf_usage_count += 1
            return await self._get_collaborative_recommendations(
                user.id, interacted_property_ids, limit, context
            )
        elif strategy == RecommendationStrategy.CONTENT_BASED:
            self.metrics.cb_usage_count += 1
            return await self._get_content_based_recommendations(
                user, interacted_property_ids, limit, context
            )
        elif strategy == RecommendationStrategy.COLD_START:
            self.metrics.cold_start_count += 1
            return await self._get_cold_start_recommendations(
                user, interacted_property_ids, limit, context
            )
        else:
            return await self._get_popular_recommendations(
                interacted_property_ids, limit, context
            )

    async def _get_content_based_recommendations(self, 
                                               user: User, 
                                               excluded_property_ids: set, 
                                               limit: int,
                                               context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Get advanced content-based recommendations using ML model"""
        try:
            if self.cb_model:
                # Use advanced content-based model
                user_profile = self._build_user_profile(user)
                cb_results = await self._run_content_based_model(
                    user_profile, excluded_property_ids, limit, context
                )
                return self._format_cb_recommendations(cb_results)
            else:
                # Fallback to simple content-based recommendations
                return await self._get_simple_content_recommendations(
                    user, excluded_property_ids, limit
                )
                
        except Exception as e:
            self.logger.error(f"Content-based recommendations failed: {e}")
            return await self._get_simple_content_recommendations(
                user, excluded_property_ids, limit
            )

    async def _get_collaborative_recommendations(self, 
                                               user_id: UUID, 
                                               excluded_property_ids: set, 
                                               limit: int,
                                               context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Get advanced collaborative filtering recommendations using neural CF"""
        try:
            if self.cf_model and hasattr(self.cf_model, 'is_trained') and self.cf_model.is_trained:
                # Use advanced collaborative filtering model
                cf_results = await self._run_collaborative_model(
                    user_id, excluded_property_ids, limit, context
                )
                self.metrics.cf_usage_count += 1
                return self._format_cf_recommendations(cf_results)
            else:
                # Fallback to simple collaborative filtering
                return await self._get_simple_collaborative_recommendations(
                    user_id, excluded_property_ids, limit
                )
                
        except Exception as e:
            self.logger.error(f"Collaborative filtering failed: {e}")
            return await self._get_simple_collaborative_recommendations(
                user_id, excluded_property_ids, limit
            )

    async def _get_hybrid_recommendations(self,
                                        user: User,
                                        interactions: List[UserInteraction],
                                        interacted_property_ids: set,
                                        limit: int,
                                        context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Get hybrid recommendations using advanced fusion"""
        try:
            if not self.hybrid_model:
                # Fallback to weighted combination
                return await self._get_manual_hybrid_recommendations(
                    user, interacted_property_ids, limit, context
                )
            
            # Use advanced hybrid model
            user_context = self._determine_user_context(user, interactions)
            
            hybrid_results = await self._run_hybrid_model(
                user, user_context, interacted_property_ids, limit, context
            )
            
            self.metrics.hybrid_usage_count += 1
            return self._format_hybrid_recommendations(hybrid_results)
            
        except Exception as e:
            self.logger.error(f"Hybrid recommendations failed: {e}")
            return await self._get_manual_hybrid_recommendations(
                user, interacted_property_ids, limit, context
            )

    async def _get_cold_start_recommendations(self,
                                            user: User,
                                            interacted_property_ids: set,
                                            limit: int,
                                            context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Get recommendations for new users"""
        try:
            recommendations = []
            
            # Use user preferences if available
            if user.preferences:
                content_recs = await self._get_content_based_recommendations(
                    user, interacted_property_ids, limit // 2, context
                )
                recommendations.extend(content_recs)
            
            # Fill with popular properties
            if len(recommendations) < limit:
                popular_recs = await self._get_popular_recommendations(
                    interacted_property_ids, limit - len(recommendations), context
                )
                recommendations.extend(popular_recs)
            
            # Add cold start explanation
            for rec in recommendations:
                rec['cold_start'] = True
                rec['explanation'] = f"New user recommendation: {rec.get('explanation', '')}"
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Cold start recommendations failed: {e}")
            return await self._get_popular_recommendations(
                interacted_property_ids, limit, context
            )

    async def _get_popular_recommendations(self, 
                                         excluded_property_ids: set, 
                                         limit: int,
                                         context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Get popular properties with enhanced popularity scoring"""
        try:
            # Get interaction matrix for popularity computation
            interaction_matrix = await self.user_repository.get_user_interaction_matrix()
            
            if interaction_matrix:
                # Calculate advanced popularity scores
                popularity_scores = self._calculate_popularity_scores(interaction_matrix)
                popular_properties = await self._get_properties_by_popularity(
                    popularity_scores, excluded_property_ids, limit * 2
                )
            else:
                # Fallback to simple active properties
                popular_properties = await self.property_repository.get_all_active(limit * 2)
            
            recommendations = []
            for prop in popular_properties:
                if prop.id not in excluded_property_ids:
                    score = self._calculate_property_popularity_score(prop, interaction_matrix)
                    recommendations.append({
                        'property_id': prop.id,
                        'property': prop,
                        'score': score,
                        'reason': 'popular',
                        'explanation': self._generate_popularity_explanation(prop, score),
                        'confidence': min(score * 1.2, 1.0)
                    })
                    
                    if len(recommendations) >= limit:
                        break
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Popular recommendations failed: {e}")
            # Simple fallback
            popular_properties = await self.property_repository.get_all_active(limit * 2)
            recommendations = []
            for prop in popular_properties:
                if prop.id not in excluded_property_ids:
                    recommendations.append({
                        'property_id': prop.id,
                        'property': prop,
                        'score': 0.5,
                        'reason': 'popular',
                        'explanation': "Popular property"
                    })
                    if len(recommendations) >= limit:
                        break
            return recommendations

    # === POST-PROCESSING ===
    
    async def _post_process_recommendations(self,
                                          recommendations: List[Dict],
                                          user: User,
                                          limit: int,
                                          interacted_property_ids: set) -> List[Dict]:
        """Post-process recommendations with deduplication, ranking, and enhancement"""
        
        # Remove duplicates
        seen_property_ids = set()
        deduplicated = []
        
        for rec in recommendations:
            property_id = rec.get('property_id')
            if property_id and property_id not in seen_property_ids:
                seen_property_ids.add(property_id)
                deduplicated.append(rec)
        
        # Sort by score (descending)
        deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Enhance with additional metadata
        enhanced = []
        for i, rec in enumerate(deduplicated[:limit]):
            rec['rank'] = i + 1
            rec['timestamp'] = datetime.now().isoformat()
            
            # Add confidence score if not present
            if 'confidence' not in rec:
                rec['confidence'] = self._calculate_recommendation_confidence(rec, user)
            
            enhanced.append(rec)
        
        # Enhanced diversity and novelty
        if self.config.enable_diversity or self.config.enable_novelty:
            enhanced = await self._apply_diversity_and_novelty(
                enhanced, user, limit
            )
        
        return enhanced

    async def _apply_diversity_and_novelty(self,
                                         recommendations: List[Dict],
                                         user: User,
                                         limit: int) -> List[Dict]:
        """Apply diversity and novelty constraints"""
        if not recommendations:
            return recommendations
        
        try:
            # Apply diversity if enabled
            if self.config.enable_diversity:
                recommendations = await self._apply_diversity_constraint(
                    recommendations, self.config.diversity_lambda
                )
            
            # Apply novelty if enabled
            if self.config.enable_novelty:
                recommendations = await self._apply_novelty_constraint(
                    recommendations, user, self.config.novelty_lambda
                )
            
            return recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Diversity/novelty application failed: {e}")
            return recommendations[:limit]

    # === CACHING ===
    
    def _generate_cache_key(self,
                           user_id: UUID,
                           limit: int,
                           strategy: Optional[RecommendationStrategy],
                           context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for recommendations"""
        key_parts = [str(user_id), str(limit)]
        
        if strategy:
            key_parts.append(strategy.value)
        
        if context:
            context_str = json.dumps(context, sort_keys=True)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
            key_parts.append(context_hash)
        
        return "user_recommendations:" + ":".join(key_parts)
    
    async def _get_cached_recommendations(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached recommendations"""
        try:
            # Try repository cache first
            cached = await self.model_repository.get_cached_predictions(cache_key)
            if cached:
                return cached
            
            # Try local cache
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.config.cache_ttl_seconds:
                    return cache_entry['data']
                else:
                    del self.cache[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_recommendations(self, cache_key: str, recommendations: List[Dict]):
        """Cache recommendations"""
        try:
            # Cache in repository
            await self.model_repository.cache_predictions(
                cache_key, recommendations, ttl_seconds=self.config.cache_ttl_seconds
            )
            
            # Cache locally
            if len(self.cache) >= self.config.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
            
            self.cache[cache_key] = {
                'data': recommendations,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Caching failed: {e}")
    
    async def _invalidate_user_cache(self, user_id: UUID):
        """Invalidate all cache entries for a user"""
        try:
            # Invalidate repository cache
            cache_pattern = f"user_recommendations:{user_id}:*"
            await self.model_repository.clear_cache(cache_pattern)
            
            # Invalidate local cache
            user_str = str(user_id)
            keys_to_remove = [
                key for key in self.cache.keys() 
                if key.startswith(f"user_recommendations:{user_str}:")
            ]
            
            for key in keys_to_remove:
                del self.cache[key]
                
        except Exception as e:
            self.logger.error(f"Cache invalidation failed: {e}")

    # === UTILITY METHODS ===
    
    def _calculate_recommendation_confidence(self, recommendation: Dict, user: User) -> float:
        """Calculate confidence score for a recommendation"""
        try:
            base_score = recommendation.get('score', 0.5)
            
            # Adjust based on user interaction history
            interaction_boost = min(len(user.interactions) / 20.0, 0.3)
            
            # Adjust based on preference match
            preference_boost = 0.0
            if 'reason' in recommendation:
                if recommendation['reason'] in ['preferred_location', 'price_range']:
                    preference_boost = 0.2
                elif recommendation['reason'] == 'similar_users':
                    preference_boost = 0.15
            
            confidence = min(base_score + interaction_boost + preference_boost, 1.0)
            return round(confidence, 3)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _update_response_time_metric(self, response_time_ms: float):
        """Update average response time metric"""
        if self.metrics.total_recommendations == 1:
            self.metrics.average_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.metrics.average_response_time_ms
            )
    
    async def _get_fallback_recommendations(self, user_id: UUID, limit: int) -> List[Dict]:
        """Get simple fallback recommendations when advanced methods fail"""
        try:
            # Get popular properties as fallback
            popular_properties = await self.property_repository.get_all_active(limit)
            
            recommendations = []
            for prop in popular_properties:
                recommendations.append({
                    'property_id': prop.id,
                    'property': prop,
                    'score': 0.4,
                    'reason': 'fallback',
                    'explanation': 'Fallback recommendation',
                    'confidence': 0.3
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Fallback recommendations failed: {e}")
            return []

    # === ANALYTICS AND MONITORING ===
    
    async def get_recommendation_metrics(self) -> Dict[str, Any]:
        """Get recommendation service metrics"""
        cache_hit_rate = (
            self.metrics.cache_hits / 
            (self.metrics.cache_hits + self.metrics.cache_misses)
            if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
        )
        
        return {
            'total_recommendations': self.metrics.total_recommendations,
            'cache_hit_rate': round(cache_hit_rate, 3),
            'average_response_time_ms': round(self.metrics.average_response_time_ms, 2),
            'strategy_usage': {
                'collaborative_filtering': self.metrics.cf_usage_count,
                'content_based': self.metrics.cb_usage_count,
                'hybrid': self.metrics.hybrid_usage_count,
                'cold_start': self.metrics.cold_start_count
            },
            'error_count': self.metrics.error_count,
            'models_initialized': self._models_initialized,
            'cache_size': len(self.cache),
            'timestamp': self.metrics.timestamp.isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the recommendation service"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_initialized': self._models_initialized,
            'components': {}
        }
        
        # Check model availability
        health['components']['collaborative_model'] = {
            'available': self.cf_model is not None,
            'trained': (hasattr(self.cf_model, 'is_trained') and self.cf_model.is_trained) if self.cf_model else False
        }
        
        health['components']['content_model'] = {
            'available': self.cb_model is not None
        }
        
        health['components']['hybrid_model'] = {
            'available': self.hybrid_model is not None
        }
        
        # Check cache
        health['components']['cache'] = {
            'local_size': len(self.cache),
            'max_size': self.config.max_cache_size
        }
        
        return health

    # === SIMPLE FALLBACK IMPLEMENTATIONS ===
    
    async def _get_simple_content_recommendations(self, user: User, excluded_property_ids: set, limit: int) -> List[Dict]:
        """Simple content-based recommendations using user preferences"""
        recommendations = []
        preferences = user.preferences
        
        # Get properties matching user preferences
        if preferences and preferences.preferred_locations:
            for location in preferences.preferred_locations[:2]:
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
        if preferences and (preferences.min_price is not None or preferences.max_price is not None):
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
    
    async def _get_simple_collaborative_recommendations(self, user_id: UUID, excluded_property_ids: set, limit: int) -> List[Dict]:
        """Simple collaborative filtering using repository methods"""
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
    
    async def _get_manual_hybrid_recommendations(self, user: User, excluded_property_ids: set, 
                                               limit: int, context: Optional[Dict[str, Any]]) -> List[Dict]:
        """Manual hybrid recommendations using weighted combination"""
        cf_recs = await self._get_collaborative_recommendations(user.id, excluded_property_ids, limit)
        cb_recs = await self._get_content_based_recommendations(user, excluded_property_ids, limit, context)
        
        # Combine with weights
        all_recs = []
        
        # Weight CF recommendations
        for rec in cf_recs:
            rec['score'] *= self.config.cf_weight
            rec['method'] = 'collaborative'
            all_recs.append(rec)
        
        # Weight CB recommendations
        for rec in cb_recs:
            rec['score'] *= self.config.cb_weight
            rec['method'] = 'content'
            all_recs.append(rec)
        
        # Sort by weighted score
        all_recs.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_recs[:limit]

    def _generate_basic_explanations(self, user: User, property_obj: Property) -> List[str]:
        """Generate basic preference matching explanations"""
        explanations = []
        preferences = user.preferences
        
        if not preferences:
            return explanations
        
        # Check location match
        if preferences.preferred_locations and property_obj.location in preferences.preferred_locations:
            explanations.append(f"Located in your preferred area: {property_obj.location}")
        
        # Check price range
        if ((preferences.min_price is None or property_obj.price >= preferences.min_price) and 
            (preferences.max_price is None or property_obj.price <= preferences.max_price)):
            explanations.append("Within your budget range")
        
        # Check bedrooms
        if ((preferences.min_bedrooms is None or property_obj.bedrooms >= preferences.min_bedrooms) and 
            (preferences.max_bedrooms is None or property_obj.bedrooms <= preferences.max_bedrooms)):
            explanations.append(f"Has {property_obj.bedrooms} bedrooms as preferred")
        
        # Check amenities
        if preferences.required_amenities:
            matching_amenities = set(property_obj.amenities) & set(preferences.required_amenities)
            if matching_amenities:
                explanations.append(f"Includes desired amenities: {', '.join(matching_amenities)}")
        
        return explanations

    def _generate_overall_explanation(self, basic_explanations: List[str], ml_explanations: Dict) -> str:
        """Generate overall explanation combining basic and ML explanations"""
        if basic_explanations:
            return '; '.join(basic_explanations)
        elif ml_explanations:
            return "Recommended based on advanced matching algorithms"
        else:
            return "Recommended based on general preferences"

    def _calculate_explanation_confidence(self, basic_explanations: List[str], ml_explanations: Dict) -> float:
        """Calculate confidence score for explanation"""
        base_confidence = 0.3
        
        # Boost for basic explanations
        basic_boost = min(len(basic_explanations) * 0.2, 0.5)
        
        # Boost for ML explanations
        ml_boost = 0.2 if ml_explanations else 0.0
        
        return min(base_confidence + basic_boost + ml_boost, 1.0)

    def _format_similar_properties(self, similar_properties: List[Property], base_property: Property) -> List[Dict]:
        """Format similar properties as recommendation results"""
        similar_recommendations = []
        for prop in similar_properties:
            similar_recommendations.append({
                'property_id': prop.id,
                'property': prop,
                'score': 0.8,
                'reason': 'similar_properties',
                'explanation': f"Similar to {base_property.title if base_property else 'selected property'}"
            })
        return similar_recommendations

    # === PLACEHOLDER METHODS FOR ADVANCED ML IMPLEMENTATIONS ===
    # These would be implemented with the actual ML model interfaces
    
    async def _train_collaborative_model(self, interaction_matrix): 
        """Train collaborative filtering model"""
        pass
    
    async def _train_content_model(self): 
        """Train content-based model"""
        pass
    
    async def _run_collaborative_model(self, user_id, excluded_property_ids, limit, context):
        """Run collaborative filtering model"""
        return []
    
    async def _run_content_based_model(self, user_profile, excluded_property_ids, limit, context):
        """Run content-based model"""
        return []
    
    async def _run_hybrid_model(self, user, user_context, excluded_property_ids, limit, context):
        """Run hybrid model"""
        return []
    
    def _build_user_profile(self, user: User) -> Dict:
        """Build user profile for ML models"""
        return {}
    
    def _determine_user_context(self, user: User, interactions: List) -> str:
        """Determine user context"""
        return "regular_user"
    
    def _format_cb_recommendations(self, results): 
        """Format content-based recommendations"""
        return []
    
    def _format_cf_recommendations(self, results): 
        """Format collaborative filtering recommendations"""
        return []
    
    def _format_hybrid_recommendations(self, results): 
        """Format hybrid recommendations"""
        return []
    
    def _calculate_popularity_scores(self, matrix): 
        """Calculate popularity scores"""
        return {}
    
    async def _get_properties_by_popularity(self, scores, excluded, limit): 
        """Get properties by popularity"""
        return []
    
    def _calculate_property_popularity_score(self, prop, matrix): 
        """Calculate property popularity score"""
        return 0.5
    
    def _generate_popularity_explanation(self, prop, score): 
        """Generate popularity explanation"""
        return "Popular property"
    
    async def _get_content_similar_properties(self, prop, limit): 
        """Get content-based similar properties"""
        return []
    
    async def _update_models_with_interaction(self, user_id, prop_id, interaction_type, duration): 
        """Update models with new interaction"""
        pass
    
    async def _generate_ml_explanations(self, user_id, prop_id, user, prop): 
        """Generate ML-based explanations"""
        return {}
    
    async def _apply_diversity_constraint(self, recs, lambda_val): 
        """Apply diversity constraint"""
        return recs
    
    async def _apply_novelty_constraint(self, recs, user, lambda_val): 
        """Apply novelty constraint"""
        return recs