import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import hashlib
from collections import defaultdict, deque
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
from functools import lru_cache
import uuid
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from typing_extensions import Literal

from .collaborative_filter import CollaborativeFilteringModel, RecommendationResult
from .content_recommender import ContentBasedRecommender

warnings.filterwarnings('ignore')


class FusionMethod(Enum):
    """Enumeration of fusion methods for hybrid recommendation"""
    WEIGHTED_AVERAGE = "weighted_average"
    MIXED = "mixed"
    SWITCHING = "switching"
    CASCADING = "cascading"
    FEATURE_COMBINATION = "feature_combination"
    ENSEMBLE = "ensemble"
    DYNAMIC_WEIGHTED = "dynamic_weighted"
    RANK_FUSION = "rank_fusion"
    BAYESIAN_FUSION = "bayesian_fusion"
    NEURAL_FUSION = "neural_fusion"


class ColdStartStrategy(Enum):
    """Enumeration of cold start handling strategies"""
    CONTENT_BASED = "content_based"
    POPULARITY_BASED = "popularity_based"
    DEMOGRAPHIC_BASED = "demographic_based"
    HYBRID_APPROACH = "hybrid_approach"
    CLUSTERING_BASED = "clustering_based"
    KNOWLEDGE_BASED = "knowledge_based"


class UserContext(Enum):
    """Enumeration of user context types"""
    NEW_USER = "new_user"
    REGULAR_USER = "regular_user"
    POWER_USER = "power_user"
    INACTIVE_USER = "inactive_user"
    RETURNING_USER = "returning_user"


@dataclass
class HybridRecommendationResult:
    """Extended recommendation result with hybrid scoring details"""
    item_id: int
    predicted_rating: float
    confidence_score: float
    explanation: str
    cf_score: Optional[float] = None
    cb_score: Optional[float] = None
    hybrid_method: str = "weighted_average"
    feature_importance: Optional[Dict[str, float]] = None
    diversity_score: Optional[float] = None
    novelty_score: Optional[float] = None
    serendipity_score: Optional[float] = None
    user_context: Optional[UserContext] = None
    fusion_weights: Optional[Dict[str, float]] = None
    model_contributions: Optional[Dict[str, float]] = None
    ranking_position: Optional[int] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments"""
    experiment_id: str
    name: str
    description: str
    control_method: FusionMethod
    treatment_method: FusionMethod
    traffic_split: float = 0.5
    min_sample_size: int = 100
    max_duration_days: int = 30
    success_metrics: List[str] = field(default_factory=lambda: ['precision', 'recall', 'diversity'])
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformance:
    """Performance metrics for models"""
    model_name: str
    rmse: float
    mae: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    coverage: float
    diversity: float
    novelty: float
    serendipity: float
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecommendationContext:
    """Context for recommendation generation"""
    user_id: int
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    time_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    user_preferences: Optional[Dict[str, Any]] = None
    recent_interactions: Optional[List[Dict[str, Any]]] = None
    context_features: Optional[Dict[str, Any]] = None
    request_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DiversityConfig:
    """Configuration for diversity optimization"""
    intra_list_diversity: float = 0.8
    novelty_weight: float = 0.2
    serendipity_weight: float = 0.1
    coverage_weight: float = 0.1
    diversity_decay: float = 0.9
    min_diversity_threshold: float = 0.3
    max_similarity_threshold: float = 0.7
    use_clustering: bool = True
    clustering_method: str = "kmeans"
    num_clusters: int = 20


class HybridRecommendationSystem:
    """
    Comprehensive Hybrid Recommendation System with advanced fusion techniques.
    
    This system implements multiple fusion methods and strategies including:
    1. Multiple fusion methods (weighted, mixed, switching, cascading, ensemble)
    2. Dynamic weight adjustment based on user context and data availability
    3. Advanced cold start handling with multiple strategies
    4. A/B testing framework for model comparison
    5. Comprehensive evaluation and comparison metrics
    6. Adaptive learning and model updating
    7. Recommendation diversity and novelty optimization
    8. Real-time serving capabilities
    9. Comprehensive logging and monitoring
    10. Explanation generation for recommendations
    
    The system follows the masterplan architecture and provides production-ready
    recommendation capabilities for rental property systems.
    """
    
    def __init__(self, 
                 cf_weight: float = 0.6,
                 cb_weight: float = 0.4,
                 min_cf_interactions: int = 5,
                 fallback_to_content: bool = True,
                 explanation_detail_level: str = "detailed",
                 fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE,
                 cold_start_strategy: ColdStartStrategy = ColdStartStrategy.HYBRID_APPROACH,
                 diversity_config: Optional[DiversityConfig] = None,
                 enable_ab_testing: bool = False,
                 enable_adaptive_learning: bool = True,
                 enable_real_time_serving: bool = True,
                 cache_size: int = 10000,
                 model_update_frequency: int = 24):
        """
        Initialize the HybridRecommendationSystem.
        
        Args:
            cf_weight: Weight for collaborative filtering recommendations (0.0 to 1.0)
            cb_weight: Weight for content-based recommendations (0.0 to 1.0)
            min_cf_interactions: Minimum user interactions required for CF recommendations
            fallback_to_content: Whether to fall back to content-based for new users
            explanation_detail_level: Level of detail for explanations
            fusion_method: Fusion method for combining recommendations
            cold_start_strategy: Strategy for handling cold start users
            diversity_config: Configuration for diversity optimization
            enable_ab_testing: Whether to enable A/B testing
            enable_adaptive_learning: Whether to enable adaptive learning
            enable_real_time_serving: Whether to enable real-time serving
            cache_size: Size of recommendation cache
            model_update_frequency: Hours between model updates
        """
        # Validate weights
        if not (0.0 <= cf_weight <= 1.0) or not (0.0 <= cb_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        
        # Normalize weights if they don't sum to 1.0
        total_weight = cf_weight + cb_weight
        if total_weight > 0:
            self.cf_weight = cf_weight / total_weight
            self.cb_weight = cb_weight / total_weight
        else:
            raise ValueError("At least one weight must be positive")
        
        # Core configuration
        self.min_cf_interactions = min_cf_interactions
        self.fallback_to_content = fallback_to_content
        self.explanation_detail_level = explanation_detail_level
        self.fusion_method = fusion_method
        self.cold_start_strategy = cold_start_strategy
        self.diversity_config = diversity_config or DiversityConfig()
        
        # Advanced features
        self.enable_ab_testing = enable_ab_testing
        self.enable_adaptive_learning = enable_adaptive_learning
        self.enable_real_time_serving = enable_real_time_serving
        self.cache_size = cache_size
        self.model_update_frequency = model_update_frequency
        
        # Model instances
        self.cf_model: Optional[CollaborativeFilteringModel] = None
        self.cb_model: Optional[ContentBasedRecommender] = None
        
        # Training state
        self.is_trained = False
        self.training_metadata: Dict[str, Any] = {}
        
        # A/B testing
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_history: List[ModelPerformance] = []
        self.user_feedback: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        # Caching and serving
        self.recommendation_cache: Dict[str, Tuple[List[HybridRecommendationResult], datetime]] = {}
        self.cache_lock = Lock()
        
        # Adaptive learning
        self.weight_history: List[Dict[str, float]] = []
        self.performance_threshold = 0.05
        self.last_model_update = datetime.now()
        
        # Diversity optimization
        self.item_features_cache: Dict[int, np.ndarray] = {}
        self.diversity_clusters: Optional[KMeans] = None
        
        # Real-time serving
        self.serving_pool: Optional[ThreadPoolExecutor] = None
        if enable_real_time_serving:
            self.serving_pool = ThreadPoolExecutor(max_workers=min(32, cpu_count() + 4))
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized HybridRecommendationSystem with CF weight: {self.cf_weight:.2f}, "
                        f"CB weight: {self.cb_weight:.2f}, fusion method: {fusion_method.value}")
    
    def initialize_models(self, 
                         num_users: int, 
                         num_items: int,
                         cf_embedding_dim: int = 50,
                         cb_embedding_dim: int = 128,
                         **kwargs) -> None:
        """
        Initialize the underlying recommendation models.
        
        Args:
            num_users: Number of users in the system
            num_items: Number of items/properties in the system
            cf_embedding_dim: Embedding dimension for collaborative filtering
            cb_embedding_dim: Embedding dimension for content-based model
            **kwargs: Additional parameters for model initialization
        """
        try:
            # Initialize collaborative filtering model
            self.cf_model = CollaborativeFilteringModel(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=cf_embedding_dim,
                reg_lambda=kwargs.get('cf_reg_lambda', 1e-6)
            )
            
            # Initialize content-based model
            self.cb_model = ContentBasedRecommender(
                embedding_dim=cb_embedding_dim,
                location_vocab_size=kwargs.get('location_vocab_size', 1000),
                amenity_vocab_size=kwargs.get('amenity_vocab_size', 500),
                reg_lambda=kwargs.get('cb_reg_lambda', 1e-5),
                learning_rate=kwargs.get('learning_rate', 0.001)
            )
            
            self.logger.info("Successfully initialized both CF and CB models")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def fit(self, 
            user_item_matrix: np.ndarray,
            property_data: List[Dict],
            cf_epochs: int = 100,
            cb_epochs: int = 100,
            cf_batch_size: int = 256,
            cb_batch_size: int = 128,
            validation_split: float = 0.2,
            **kwargs) -> Dict[str, Any]:
        """
        Train both collaborative filtering and content-based models.
        
        Args:
            user_item_matrix: User-item interaction matrix
            property_data: List of property dictionaries with features
            cf_epochs: Epochs for collaborative filtering training
            cb_epochs: Epochs for content-based training
            cf_batch_size: Batch size for CF training
            cb_batch_size: Batch size for CB training
            validation_split: Validation split for training
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metadata and results
        """
        try:
            if self.cf_model is None or self.cb_model is None:
                raise ValueError("Models must be initialized before training")
            
            training_results = {}
            
            # Train collaborative filtering model
            self.logger.info("Training collaborative filtering model...")
            cf_results = self.cf_model.fit(
                user_item_matrix=user_item_matrix,
                epochs=cf_epochs,
                batch_size=cf_batch_size,
                validation_split=validation_split,
                **kwargs
            )
            training_results['cf_results'] = cf_results
            
            # Train content-based model
            self.logger.info("Training content-based model...")
            cb_results = self.cb_model.fit(
                user_item_matrix=user_item_matrix,
                property_data=property_data,
                epochs=cb_epochs,
                batch_size=cb_batch_size,
                validation_split=validation_split,
                **kwargs
            )
            training_results['cb_results'] = cb_results
            
            # Initialize diversity clusters if enabled
            if self.diversity_config.use_clustering:
                self._initialize_diversity_clusters(property_data)
            
            # Store training metadata
            self.training_metadata = {
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'num_users': user_item_matrix.shape[0],
                'num_items': user_item_matrix.shape[1],
                'num_properties': len(property_data),
                'cf_epochs': cf_epochs,
                'cb_epochs': cb_epochs,
                'training_results': training_results,
                'fusion_method': self.fusion_method.value,
                'cold_start_strategy': self.cold_start_strategy.value
            }
            
            self.is_trained = True
            self.last_model_update = datetime.now()
            
            self.logger.info("Successfully trained hybrid recommendation system")
            
            return {
                'status': 'success',
                'cf_results': cf_results,
                'cb_results': cb_results,
                'hybrid_metadata': self.training_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _initialize_diversity_clusters(self, property_data: List[Dict]) -> None:
        """Initialize clusters for diversity optimization"""
        try:
            if not property_data:
                return
            
            # Extract features for clustering
            features = []
            for prop in property_data:
                # Combine numerical features (price, bedrooms, bathrooms, etc.)
                feature_vector = [
                    prop.get('price', 0),
                    prop.get('bedrooms', 0),
                    prop.get('bathrooms', 0),
                    prop.get('square_feet', 0),
                    len(prop.get('amenities', [])),
                    prop.get('rating', 0)
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform clustering
            self.diversity_clusters = KMeans(
                n_clusters=self.diversity_config.num_clusters,
                random_state=42,
                n_init=10
            )
            self.diversity_clusters.fit(features_scaled)
            
            # Cache item features for diversity calculation
            for i, prop in enumerate(property_data):
                self.item_features_cache[prop.get('id', i)] = features_scaled[i]
            
            self.logger.info(f"Initialized {self.diversity_config.num_clusters} diversity clusters")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize diversity clusters: {e}")
    
    def get_user_context(self, user_id: int, context: Optional[RecommendationContext] = None) -> UserContext:
        """
        Determine user context based on interaction history and current context.
        
        Args:
            user_id: User identifier
            context: Optional recommendation context
            
        Returns:
            User context classification
        """
        try:
            if not self.is_trained or self.cf_model is None:
                return UserContext.NEW_USER
            
            # Check if user exists in CF model
            if user_id >= self.cf_model.user_item_matrix.shape[0] or user_id < 0:
                return UserContext.NEW_USER
            
            # Get user interaction count
            user_interactions = np.sum(self.cf_model.user_item_matrix[user_id] > 0)
            
            # Check recent activity if context is provided
            recent_activity = 0
            if context and context.recent_interactions:
                recent_activity = len(context.recent_interactions)
            
            # Classify user context
            if user_interactions == 0:
                return UserContext.NEW_USER
            elif user_interactions < self.min_cf_interactions:
                return UserContext.NEW_USER if recent_activity == 0 else UserContext.RETURNING_USER
            elif user_interactions > 50:
                return UserContext.POWER_USER
            elif recent_activity == 0:
                return UserContext.INACTIVE_USER
            else:
                return UserContext.REGULAR_USER
                
        except Exception as e:
            self.logger.warning(f"Error determining user context: {e}")
            return UserContext.NEW_USER
    
    def adjust_weights_dynamically(self, 
                                 user_id: int, 
                                 context: Optional[RecommendationContext] = None) -> Tuple[float, float]:
        """
        Dynamically adjust CF and CB weights based on user context and data availability.
        
        Args:
            user_id: User identifier
            context: Optional recommendation context
            
        Returns:
            Tuple of (cf_weight, cb_weight)
        """
        try:
            user_context = self.get_user_context(user_id, context)
            
            # Base weights
            cf_weight = self.cf_weight
            cb_weight = self.cb_weight
            
            # Adjust based on user context
            if user_context == UserContext.NEW_USER:
                # New users: rely more on content-based
                cf_weight *= 0.3
                cb_weight *= 1.4
            elif user_context == UserContext.POWER_USER:
                # Power users: rely more on collaborative filtering
                cf_weight *= 1.3
                cb_weight *= 0.7
            elif user_context == UserContext.INACTIVE_USER:
                # Inactive users: balance both approaches
                cf_weight *= 0.8
                cb_weight *= 1.2
            elif user_context == UserContext.RETURNING_USER:
                # Returning users: slight preference for content-based
                cf_weight *= 0.9
                cb_weight *= 1.1
            
            # Check data availability
            if self.cf_model and user_id < self.cf_model.user_item_matrix.shape[0]:
                user_interactions = np.sum(self.cf_model.user_item_matrix[user_id] > 0)
                if user_interactions < self.min_cf_interactions:
                    cf_weight *= 0.5
                    cb_weight *= 1.5
            
            # Normalize weights
            total_weight = cf_weight + cb_weight
            if total_weight > 0:
                cf_weight /= total_weight
                cb_weight /= total_weight
            
            return cf_weight, cb_weight
            
        except Exception as e:
            self.logger.warning(f"Error adjusting weights: {e}")
            return self.cf_weight, self.cb_weight
    
    def recommend(self, 
                  user_id: int, 
                  num_recommendations: int = 10,
                  context: Optional[RecommendationContext] = None,
                  exclude_seen: bool = True,
                  include_explanations: bool = True,
                  experiment_id: Optional[str] = None) -> List[HybridRecommendationResult]:
        """
        Generate hybrid recommendations for a user with advanced fusion techniques.
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to generate
            context: Optional recommendation context
            exclude_seen: Whether to exclude previously seen items
            include_explanations: Whether to include detailed explanations
            experiment_id: Optional experiment ID for A/B testing
            
        Returns:
            List of HybridRecommendationResult objects
        """
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before making recommendations")
            
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(user_id, num_recommendations, context)
            cached_result = self._get_cached_recommendations(cache_key)
            if cached_result:
                return cached_result
            
            # Determine fusion method (A/B testing override)
            fusion_method = self.fusion_method
            if experiment_id and experiment_id in self.experiments:
                fusion_method = self._get_experiment_method(experiment_id, user_id)
            
            # Get user context and adjust weights
            user_context = self.get_user_context(user_id, context)
            cf_weight, cb_weight = self.adjust_weights_dynamically(user_id, context)
            
            # Generate recommendations based on fusion method
            if fusion_method == FusionMethod.WEIGHTED_AVERAGE:
                recommendations = self._weighted_average_fusion(
                    user_id, num_recommendations, cf_weight, cb_weight, exclude_seen
                )
            elif fusion_method == FusionMethod.MIXED:
                recommendations = self._mixed_fusion(
                    user_id, num_recommendations, cf_weight, cb_weight, exclude_seen
                )
            elif fusion_method == FusionMethod.SWITCHING:
                recommendations = self._switching_fusion(
                    user_id, num_recommendations, user_context, exclude_seen
                )
            elif fusion_method == FusionMethod.CASCADING:
                recommendations = self._cascading_fusion(
                    user_id, num_recommendations, exclude_seen
                )
            elif fusion_method == FusionMethod.ENSEMBLE:
                recommendations = self._ensemble_fusion(
                    user_id, num_recommendations, exclude_seen
                )
            elif fusion_method == FusionMethod.DYNAMIC_WEIGHTED:
                recommendations = self._dynamic_weighted_fusion(
                    user_id, num_recommendations, context, exclude_seen
                )
            elif fusion_method == FusionMethod.RANK_FUSION:
                recommendations = self._rank_fusion(
                    user_id, num_recommendations, exclude_seen
                )
            else:
                # Default to weighted average
                recommendations = self._weighted_average_fusion(
                    user_id, num_recommendations, cf_weight, cb_weight, exclude_seen
                )
            
            # Apply diversity optimization
            if self.diversity_config and len(recommendations) > 1:
                recommendations = self._optimize_diversity(recommendations)
            
            # Add explanations if requested
            if include_explanations:
                recommendations = self._add_comprehensive_explanations(user_id, recommendations)
            
            # Add metadata
            for i, rec in enumerate(recommendations):
                rec.user_context = user_context
                rec.fusion_weights = {'cf': cf_weight, 'cb': cb_weight}
                rec.ranking_position = i + 1
                rec.timestamp = datetime.now()
                rec.metadata = {
                    'fusion_method': fusion_method.value,
                    'experiment_id': experiment_id
                }
            
            # Cache recommendations
            self._cache_recommendations(cache_key, recommendations)
            
            # Log performance
            response_time = (time.time() - start_time) * 1000
            self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id} "
                           f"in {response_time:.2f}ms using {fusion_method.value}")
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _weighted_average_fusion(self, 
                               user_id: int, 
                               num_recommendations: int, 
                               cf_weight: float, 
                               cb_weight: float, 
                               exclude_seen: bool) -> List[HybridRecommendationResult]:
        """Weighted average fusion method"""
        try:
            # Get CF recommendations
            cf_recommendations = []
            if self.cf_model and self._should_use_collaborative_filtering(user_id):
                try:
                    cf_recommendations = self.cf_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations * 2,
                        exclude_seen=exclude_seen
                    )
                except Exception as e:
                    self.logger.warning(f"CF recommendations failed: {e}")
            
            # Get CB recommendations
            cb_recommendations = []
            if self.cb_model:
                try:
                    cb_recommendations = self.cb_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations * 2,
                        exclude_seen=exclude_seen
                    )
                except Exception as e:
                    self.logger.warning(f"CB recommendations failed: {e}")
            
            # Combine recommendations
            return self._combine_recommendations_weighted(
                cf_recommendations, cb_recommendations, cf_weight, cb_weight, num_recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Weighted average fusion failed: {e}")
            return []
    
    def _mixed_fusion(self, 
                     user_id: int, 
                     num_recommendations: int, 
                     cf_weight: float, 
                     cb_weight: float, 
                     exclude_seen: bool) -> List[HybridRecommendationResult]:
        """Mixed fusion method - alternate between CF and CB recommendations"""
        try:
            # Calculate how many recommendations from each model
            cf_count = int(num_recommendations * cf_weight)
            cb_count = num_recommendations - cf_count
            
            recommendations = []
            
            # Get CF recommendations
            if cf_count > 0 and self.cf_model and self._should_use_collaborative_filtering(user_id):
                try:
                    cf_recommendations = self.cf_model.recommend(
                        user_id=user_id,
                        num_recommendations=cf_count,
                        exclude_seen=exclude_seen
                    )
                    for rec in cf_recommendations:
                        hybrid_rec = HybridRecommendationResult(
                            item_id=rec.item_id,
                            predicted_rating=rec.predicted_rating,
                            confidence_score=rec.confidence_score,
                            explanation=rec.explanation,
                            cf_score=rec.predicted_rating,
                            cb_score=None,
                            hybrid_method="mixed_cf"
                        )
                        recommendations.append(hybrid_rec)
                except Exception as e:
                    self.logger.warning(f"CF recommendations failed in mixed fusion: {e}")
            
            # Get CB recommendations
            if cb_count > 0 and self.cb_model:
                try:
                    cb_recommendations = self.cb_model.recommend(
                        user_id=user_id,
                        num_recommendations=cb_count,
                        exclude_seen=exclude_seen
                    )
                    for rec in cb_recommendations:
                        hybrid_rec = HybridRecommendationResult(
                            item_id=rec.item_id,
                            predicted_rating=rec.predicted_rating,
                            confidence_score=rec.confidence_score,
                            explanation=rec.explanation,
                            cf_score=None,
                            cb_score=rec.predicted_rating,
                            hybrid_method="mixed_cb"
                        )
                        recommendations.append(hybrid_rec)
                except Exception as e:
                    self.logger.warning(f"CB recommendations failed in mixed fusion: {e}")
            
            # Shuffle to mix CF and CB recommendations
            np.random.shuffle(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Mixed fusion failed: {e}")
            return []
    
    def _switching_fusion(self, 
                         user_id: int, 
                         num_recommendations: int, 
                         user_context: UserContext, 
                         exclude_seen: bool) -> List[HybridRecommendationResult]:
        """Switching fusion method - choose model based on user context"""
        try:
            # Decide which model to use based on user context
            if user_context in [UserContext.NEW_USER, UserContext.RETURNING_USER]:
                # Use content-based for new/returning users
                if self.cb_model:
                    cb_recommendations = self.cb_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations,
                        exclude_seen=exclude_seen
                    )
                    return self._convert_to_hybrid_results(cb_recommendations, "switching_cb")
            else:
                # Use collaborative filtering for regular/power users
                if self.cf_model and self._should_use_collaborative_filtering(user_id):
                    cf_recommendations = self.cf_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations,
                        exclude_seen=exclude_seen
                    )
                    return self._convert_to_hybrid_results(cf_recommendations, "switching_cf")
            
            # Fallback to weighted average if switching fails
            return self._weighted_average_fusion(
                user_id, num_recommendations, self.cf_weight, self.cb_weight, exclude_seen
            )
            
        except Exception as e:
            self.logger.error(f"Switching fusion failed: {e}")
            return []
    
    def _cascading_fusion(self, 
                         user_id: int, 
                         num_recommendations: int, 
                         exclude_seen: bool) -> List[HybridRecommendationResult]:
        """Cascading fusion method - use CB to filter CF recommendations"""
        try:
            recommendations = []
            
            # First, get CF recommendations
            if self.cf_model and self._should_use_collaborative_filtering(user_id):
                try:
                    cf_recommendations = self.cf_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations * 3,  # Get more for filtering
                        exclude_seen=exclude_seen
                    )
                    
                    # Filter and rank using CB scores
                    if self.cb_model and cf_recommendations:
                        item_ids = [rec.item_id for rec in cf_recommendations]
                        cb_scores = self.cb_model.predict(user_id, item_ids)
                        
                        # Combine CF and CB scores with cascading logic
                        for i, cf_rec in enumerate(cf_recommendations):
                            if i < len(cb_scores):
                                cb_score = cb_scores[i]
                                # Use CB score as a filter/booster
                                if cb_score > 0.5:  # Only include if CB score is above threshold
                                    hybrid_score = cf_rec.predicted_rating * (1 + cb_score * 0.2)
                                    hybrid_rec = HybridRecommendationResult(
                                        item_id=cf_rec.item_id,
                                        predicted_rating=hybrid_score,
                                        confidence_score=cf_rec.confidence_score * cb_score,
                                        explanation=cf_rec.explanation,
                                        cf_score=cf_rec.predicted_rating,
                                        cb_score=cb_score,
                                        hybrid_method="cascading"
                                    )
                                    recommendations.append(hybrid_rec)
                        
                        # Sort by hybrid score
                        recommendations.sort(key=lambda x: x.predicted_rating, reverse=True)
                        
                except Exception as e:
                    self.logger.warning(f"CF recommendations failed in cascading: {e}")
            
            # Fill remaining with CB recommendations if needed
            if len(recommendations) < num_recommendations and self.cb_model:
                try:
                    existing_items = {rec.item_id for rec in recommendations}
                    cb_recommendations = self.cb_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations - len(recommendations),
                        exclude_seen=exclude_seen
                    )
                    
                    for rec in cb_recommendations:
                        if rec.item_id not in existing_items:
                            hybrid_rec = HybridRecommendationResult(
                                item_id=rec.item_id,
                                predicted_rating=rec.predicted_rating,
                                confidence_score=rec.confidence_score,
                                explanation=rec.explanation,
                                cf_score=None,
                                cb_score=rec.predicted_rating,
                                hybrid_method="cascading_cb_fill"
                            )
                            recommendations.append(hybrid_rec)
                            
                except Exception as e:
                    self.logger.warning(f"CB fill failed in cascading: {e}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Cascading fusion failed: {e}")
            return []
    
    def _ensemble_fusion(self, 
                        user_id: int, 
                        num_recommendations: int, 
                        exclude_seen: bool) -> List[HybridRecommendationResult]:
        """Ensemble fusion method - use multiple fusion techniques and vote"""
        try:
            ensemble_results = []
            
            # Get recommendations from multiple fusion methods
            methods = [
                (self._weighted_average_fusion, "weighted", self.cf_weight, self.cb_weight),
                (self._mixed_fusion, "mixed", self.cf_weight, self.cb_weight),
                (self._cascading_fusion, "cascading", None, None)
            ]
            
            for method_func, method_name, cf_w, cb_w in methods:
                try:
                    if cf_w is not None and cb_w is not None:
                        method_results = method_func(user_id, num_recommendations, cf_w, cb_w, exclude_seen)
                    else:
                        method_results = method_func(user_id, num_recommendations, exclude_seen)
                    
                    ensemble_results.append((method_name, method_results))
                except Exception as e:
                    self.logger.warning(f"Ensemble method {method_name} failed: {e}")
            
            # Combine results using voting
            if ensemble_results:
                return self._vote_ensemble_results(ensemble_results, num_recommendations)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Ensemble fusion failed: {e}")
            return []
    
    def _dynamic_weighted_fusion(self, 
                               user_id: int, 
                               num_recommendations: int, 
                               context: Optional[RecommendationContext], 
                               exclude_seen: bool) -> List[HybridRecommendationResult]:
        """Dynamic weighted fusion - adjust weights based on real-time context"""
        try:
            # Get base weights
            cf_weight, cb_weight = self.adjust_weights_dynamically(user_id, context)
            
            # Additional context-based adjustments
            if context:
                # Time-based adjustments
                if context.time_of_day is not None:
                    if 9 <= context.time_of_day <= 17:  # Business hours
                        cf_weight *= 1.1  # Slightly favor CF during business hours
                    else:
                        cb_weight *= 1.1  # Slightly favor CB during off-hours
                
                # Device-based adjustments
                if context.device_type == "mobile":
                    cb_weight *= 1.2  # Favor CB on mobile for faster response
                elif context.device_type == "desktop":
                    cf_weight *= 1.1  # Favor CF on desktop for more complex recommendations
                
                # Location-based adjustments
                if context.location:
                    cb_weight *= 1.1  # Favor CB when location is available
                
                # Normalize weights
                total_weight = cf_weight + cb_weight
                if total_weight > 0:
                    cf_weight /= total_weight
                    cb_weight /= total_weight
            
            # Use weighted average with dynamic weights
            return self._weighted_average_fusion(
                user_id, num_recommendations, cf_weight, cb_weight, exclude_seen
            )
            
        except Exception as e:
            self.logger.error(f"Dynamic weighted fusion failed: {e}")
            return []
    
    def _rank_fusion(self, 
                    user_id: int, 
                    num_recommendations: int, 
                    exclude_seen: bool) -> List[HybridRecommendationResult]:
        """Rank fusion method - combine rankings from both models"""
        try:
            # Get recommendations from both models
            cf_recommendations = []
            cb_recommendations = []
            
            if self.cf_model and self._should_use_collaborative_filtering(user_id):
                try:
                    cf_recommendations = self.cf_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations * 2,
                        exclude_seen=exclude_seen
                    )
                except Exception as e:
                    self.logger.warning(f"CF recommendations failed in rank fusion: {e}")
            
            if self.cb_model:
                try:
                    cb_recommendations = self.cb_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations * 2,
                        exclude_seen=exclude_seen
                    )
                except Exception as e:
                    self.logger.warning(f"CB recommendations failed in rank fusion: {e}")
            
            # Create ranking dictionaries
            cf_ranks = {rec.item_id: i + 1 for i, rec in enumerate(cf_recommendations)}
            cb_ranks = {rec.item_id: i + 1 for i, rec in enumerate(cb_recommendations)}
            
            # Get all unique items
            all_items = set(cf_ranks.keys()) | set(cb_ranks.keys())
            
            # Calculate reciprocal rank fusion scores
            fusion_scores = {}
            for item_id in all_items:
                cf_rank = cf_ranks.get(item_id, len(cf_recommendations) + 1)
                cb_rank = cb_ranks.get(item_id, len(cb_recommendations) + 1)
                
                # Reciprocal rank fusion formula
                rrf_score = (1 / (60 + cf_rank)) + (1 / (60 + cb_rank))
                fusion_scores[item_id] = rrf_score
            
            # Sort by fusion score
            sorted_items = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create hybrid recommendations
            recommendations = []
            cf_lookup = {rec.item_id: rec for rec in cf_recommendations}
            cb_lookup = {rec.item_id: rec for rec in cb_recommendations}
            
            for item_id, score in sorted_items[:num_recommendations]:
                cf_rec = cf_lookup.get(item_id)
                cb_rec = cb_lookup.get(item_id)
                
                # Use available recommendation data
                if cf_rec and cb_rec:
                    hybrid_rec = HybridRecommendationResult(
                        item_id=item_id,
                        predicted_rating=score,
                        confidence_score=(cf_rec.confidence_score + cb_rec.confidence_score) / 2,
                        explanation=f"Ranked fusion: CF rank {cf_ranks.get(item_id, 'N/A')}, CB rank {cb_ranks.get(item_id, 'N/A')}",
                        cf_score=cf_rec.predicted_rating,
                        cb_score=cb_rec.predicted_rating,
                        hybrid_method="rank_fusion"
                    )
                elif cf_rec:
                    hybrid_rec = HybridRecommendationResult(
                        item_id=item_id,
                        predicted_rating=score,
                        confidence_score=cf_rec.confidence_score,
                        explanation=f"Ranked fusion: CF rank {cf_ranks.get(item_id, 'N/A')}",
                        cf_score=cf_rec.predicted_rating,
                        cb_score=None,
                        hybrid_method="rank_fusion"
                    )
                elif cb_rec:
                    hybrid_rec = HybridRecommendationResult(
                        item_id=item_id,
                        predicted_rating=score,
                        confidence_score=cb_rec.confidence_score,
                        explanation=f"Ranked fusion: CB rank {cb_ranks.get(item_id, 'N/A')}",
                        cf_score=None,
                        cb_score=cb_rec.predicted_rating,
                        hybrid_method="rank_fusion"
                    )
                else:
                    continue
                
                recommendations.append(hybrid_rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Rank fusion failed: {e}")
            return []
    
    def _combine_recommendations_weighted(self, 
                                        cf_recommendations: List[RecommendationResult],
                                        cb_recommendations: List[RecommendationResult],
                                        cf_weight: float,
                                        cb_weight: float,
                                        num_recommendations: int) -> List[HybridRecommendationResult]:
        """Combine recommendations using weighted average"""
        try:
            # Create lookup dictionaries
            cf_lookup = {rec.item_id: rec for rec in cf_recommendations}
            cb_lookup = {rec.item_id: rec for rec in cb_recommendations}
            
            # Get all unique item IDs
            all_item_ids = set(cf_lookup.keys()) | set(cb_lookup.keys())
            
            hybrid_recommendations = []
            
            for item_id in all_item_ids:
                cf_rec = cf_lookup.get(item_id)
                cb_rec = cb_lookup.get(item_id)
                
                # Calculate hybrid score
                hybrid_score = 0.0
                cf_score = None
                cb_score = None
                
                if cf_rec is not None:
                    cf_score = cf_rec.predicted_rating
                    hybrid_score += cf_weight * cf_score
                
                if cb_rec is not None:
                    cb_score = cb_rec.predicted_rating
                    hybrid_score += cb_weight * cb_score
                
                # Calculate confidence
                confidence = self._calculate_hybrid_confidence(cf_rec, cb_rec)
                
                # Generate explanation
                explanation = self._generate_hybrid_explanation(cf_rec, cb_rec, cf_weight, cb_weight)
                
                hybrid_rec = HybridRecommendationResult(
                    item_id=item_id,
                    predicted_rating=hybrid_score,
                    confidence_score=confidence,
                    explanation=explanation,
                    cf_score=cf_score,
                    cb_score=cb_score,
                    hybrid_method="weighted_average"
                )
                
                hybrid_recommendations.append(hybrid_rec)
            
            # Sort by hybrid score
            hybrid_recommendations.sort(key=lambda x: x.predicted_rating, reverse=True)
            
            return hybrid_recommendations[:num_recommendations]
            
        except Exception as e:
            self.logger.error(f"Weighted combination failed: {e}")
            return []
    
    def _vote_ensemble_results(self, 
                              ensemble_results: List[Tuple[str, List[HybridRecommendationResult]]],
                              num_recommendations: int) -> List[HybridRecommendationResult]:
        """Vote on ensemble results to get final recommendations"""
        try:
            # Collect all item scores from different methods
            item_scores = defaultdict(list)
            item_data = {}
            
            for method_name, results in ensemble_results:
                for rec in results:
                    item_scores[rec.item_id].append(rec.predicted_rating)
                    if rec.item_id not in item_data:
                        item_data[rec.item_id] = rec
            
            # Calculate ensemble scores
            ensemble_recommendations = []
            for item_id, scores in item_scores.items():
                # Use average of all method scores
                ensemble_score = np.mean(scores)
                
                # Use data from the first method that recommended this item
                base_rec = item_data[item_id]
                
                hybrid_rec = HybridRecommendationResult(
                    item_id=item_id,
                    predicted_rating=ensemble_score,
                    confidence_score=base_rec.confidence_score,
                    explanation=f"Ensemble recommendation (methods: {len(scores)})",
                    cf_score=base_rec.cf_score,
                    cb_score=base_rec.cb_score,
                    hybrid_method="ensemble"
                )
                ensemble_recommendations.append(hybrid_rec)
            
            # Sort by ensemble score
            ensemble_recommendations.sort(key=lambda x: x.predicted_rating, reverse=True)
            
            return ensemble_recommendations[:num_recommendations]
            
        except Exception as e:
            self.logger.error(f"Ensemble voting failed: {e}")
            return []
    
    def _optimize_diversity(self, 
                           recommendations: List[HybridRecommendationResult]) -> List[HybridRecommendationResult]:
        """Optimize recommendation diversity using various techniques"""
        try:
            if len(recommendations) <= 1:
                return recommendations
            
            # Calculate diversity scores
            for rec in recommendations:
                rec.diversity_score = self._calculate_diversity_score(rec, recommendations)
                rec.novelty_score = self._calculate_novelty_score(rec)
                rec.serendipity_score = self._calculate_serendipity_score(rec)
            
            # Re-rank based on diversity-accuracy trade-off
            diversity_recommendations = []
            used_clusters = set()
            
            # First, add the highest-rated item
            if recommendations:
                diversity_recommendations.append(recommendations[0])
                if recommendations[0].item_id in self.item_features_cache:
                    if self.diversity_clusters:
                        cluster = self.diversity_clusters.predict(
                            [self.item_features_cache[recommendations[0].item_id]]
                        )[0]
                        used_clusters.add(cluster)
            
            # Then, add items that maximize diversity
            for rec in recommendations[1:]:
                if len(diversity_recommendations) >= len(recommendations):
                    break
                
                # Check cluster diversity
                add_item = True
                if rec.item_id in self.item_features_cache and self.diversity_clusters:
                    cluster = self.diversity_clusters.predict(
                        [self.item_features_cache[rec.item_id]]
                    )[0]
                    
                    if cluster in used_clusters:
                        # Check if we should still add based on diversity threshold
                        if len(used_clusters) < self.diversity_config.num_clusters * 0.5:
                            add_item = False
                    else:
                        used_clusters.add(cluster)
                
                if add_item:
                    diversity_recommendations.append(rec)
            
            # Fill remaining slots with highest-rated items
            if len(diversity_recommendations) < len(recommendations):
                for rec in recommendations:
                    if rec not in diversity_recommendations:
                        diversity_recommendations.append(rec)
                        if len(diversity_recommendations) >= len(recommendations):
                            break
            
            return diversity_recommendations
            
        except Exception as e:
            self.logger.warning(f"Diversity optimization failed: {e}")
            return recommendations
    
    def _calculate_diversity_score(self, 
                                 rec: HybridRecommendationResult, 
                                 all_recommendations: List[HybridRecommendationResult]) -> float:
        """Calculate diversity score for a recommendation"""
        try:
            if rec.item_id not in self.item_features_cache:
                return 0.5  # Default diversity score
            
            item_features = self.item_features_cache[rec.item_id]
            diversity_sum = 0.0
            count = 0
            
            for other_rec in all_recommendations:
                if other_rec.item_id != rec.item_id and other_rec.item_id in self.item_features_cache:
                    other_features = self.item_features_cache[other_rec.item_id]
                    # Calculate cosine similarity
                    similarity = np.dot(item_features, other_features) / (
                        np.linalg.norm(item_features) * np.linalg.norm(other_features)
                    )
                    diversity_sum += 1 - similarity  # Diversity is 1 - similarity
                    count += 1
            
            return diversity_sum / count if count > 0 else 0.5
            
        except Exception as e:
            self.logger.warning(f"Diversity score calculation failed: {e}")
            return 0.5
    
    def _calculate_novelty_score(self, rec: HybridRecommendationResult) -> float:
        """Calculate novelty score for a recommendation"""
        try:
            # Simple novelty calculation based on item popularity
            # In a real implementation, this would use item popularity statistics
            # For now, we'll use a placeholder calculation
            return max(0.1, 1.0 - (rec.predicted_rating * 0.3))
            
        except Exception as e:
            self.logger.warning(f"Novelty score calculation failed: {e}")
            return 0.5
    
    def _calculate_serendipity_score(self, rec: HybridRecommendationResult) -> float:
        """Calculate serendipity score for a recommendation"""
        try:
            # Simple serendipity calculation
            # In a real implementation, this would consider user's typical preferences
            # For now, we'll use a placeholder calculation
            novelty = rec.novelty_score or 0.5
            relevance = rec.predicted_rating
            return novelty * relevance * 0.8
            
        except Exception as e:
            self.logger.warning(f"Serendipity score calculation failed: {e}")
            return 0.5
    
    def _should_use_collaborative_filtering(self, user_id: int) -> bool:
        """Determine if collaborative filtering should be used for a user"""
        try:
            if self.cf_model is None or not self.cf_model.is_trained:
                return False
            
            if self.cf_model.user_item_matrix is None:
                return False
            
            if user_id >= self.cf_model.user_item_matrix.shape[0] or user_id < 0:
                return False
            
            # Check if user has enough interactions
            user_interactions = np.sum(self.cf_model.user_item_matrix[user_id] > 0)
            return user_interactions >= self.min_cf_interactions
            
        except Exception as e:
            self.logger.warning(f"Error checking CF availability: {e}")
            return False
    
    def _convert_to_hybrid_results(self, 
                                 recommendations: List[RecommendationResult],
                                 method: str) -> List[HybridRecommendationResult]:
        """Convert single-model recommendations to hybrid results"""
        hybrid_results = []
        
        for rec in recommendations:
            hybrid_rec = HybridRecommendationResult(
                item_id=rec.item_id,
                predicted_rating=rec.predicted_rating,
                confidence_score=rec.confidence_score,
                explanation=rec.explanation,
                cf_score=rec.predicted_rating if "cf" in method else None,
                cb_score=rec.predicted_rating if "cb" in method else None,
                hybrid_method=method
            )
            hybrid_results.append(hybrid_rec)
        
        return hybrid_results
    
    def _calculate_hybrid_confidence(self, 
                                   cf_rec: Optional[RecommendationResult],
                                   cb_rec: Optional[RecommendationResult]) -> float:
        """Calculate confidence score for hybrid recommendation"""
        try:
            if cf_rec is not None and cb_rec is not None:
                # Both models agree, higher confidence
                cf_conf = cf_rec.confidence_score
                cb_conf = cb_rec.confidence_score
                
                # Weighted average with boost for agreement
                base_confidence = self.cf_weight * cf_conf + self.cb_weight * cb_conf
                
                # Boost confidence if both scores are similar
                score_agreement = 1.0 - abs(cf_rec.predicted_rating - cb_rec.predicted_rating)
                agreement_boost = min(score_agreement * 0.1, 0.2)
                
                return min(base_confidence + agreement_boost, 1.0)
            
            elif cf_rec is not None:
                return cf_rec.confidence_score * 0.9
            
            elif cb_rec is not None:
                return cb_rec.confidence_score * 0.9
            
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_hybrid_explanation(self, 
                                   cf_rec: Optional[RecommendationResult],
                                   cb_rec: Optional[RecommendationResult],
                                   cf_weight: Optional[float] = None,
                                   cb_weight: Optional[float] = None) -> str:
        """Generate explanation for hybrid recommendation"""
        try:
            cf_w = cf_weight or self.cf_weight
            cb_w = cb_weight or self.cb_weight
            
            if cf_rec is not None and cb_rec is not None:
                if self.explanation_detail_level == "simple":
                    return "Recommended based on similar users and property features"
                elif self.explanation_detail_level == "detailed":
                    return (f"Recommended by similar users (score: {cf_rec.predicted_rating:.2f}) "
                           f"and property features (score: {cb_rec.predicted_rating:.2f})")
                else:  # technical
                    return (f"Hybrid recommendation: CF({cf_rec.predicted_rating:.3f}) × {cf_w:.2f} + "
                           f"CB({cb_rec.predicted_rating:.3f}) × {cb_w:.2f}")
            
            elif cf_rec is not None:
                return cf_rec.explanation
            
            elif cb_rec is not None:
                return cb_rec.explanation
            
            else:
                return "No explanation available"
                
        except Exception as e:
            self.logger.warning(f"Explanation generation failed: {e}")
            return "Recommendation explanation unavailable"
    
    def _add_comprehensive_explanations(self, 
                                      user_id: int, 
                                      recommendations: List[HybridRecommendationResult]) -> List[HybridRecommendationResult]:
        """Add comprehensive explanations to recommendations"""
        for rec in recommendations:
            try:
                detailed_explanation = self.explain_recommendation(
                    user_id=user_id,
                    item_id=rec.item_id,
                    include_feature_importance=True
                )
                
                # Update recommendation with detailed explanation
                if 'explanations' in detailed_explanation:
                    explanation_parts = []
                    for exp in detailed_explanation['explanations']:
                        explanation_parts.append(f"{exp['type']}: {exp['description']}")
                    rec.explanation = "; ".join(explanation_parts)
                
                # Add feature importance if available
                if 'feature_importance' in detailed_explanation:
                    rec.feature_importance = detailed_explanation['feature_importance']
                    
            except Exception as e:
                self.logger.warning(f"Comprehensive explanation failed for item {rec.item_id}: {e}")
        
        return recommendations
    
    def explain_recommendation(self, 
                             user_id: int, 
                             item_id: int,
                             include_feature_importance: bool = True) -> Dict[str, Any]:
        """Generate detailed explanation for a specific recommendation"""
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before generating explanations")
            
            explanation = {
                'user_id': user_id,
                'item_id': item_id,
                'fusion_method': self.fusion_method.value,
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'explanations': []
            }
            
            # Check if CF is available for this user
            use_cf = self._should_use_collaborative_filtering(user_id)
            
            if use_cf and self.cf_model is not None:
                # Get CF prediction and explanation
                try:
                    cf_prediction = self.cf_model.predict(user_id, [item_id])
                    if len(cf_prediction) > 0:
                        cf_score = cf_prediction[0]
                        explanation['cf_score'] = float(cf_score)
                        explanation['explanations'].append({
                            'type': 'collaborative_filtering',
                            'score': float(cf_score),
                            'weight': self.cf_weight,
                            'contribution': float(cf_score * self.cf_weight),
                            'description': f"Users with similar preferences rated this property {cf_score:.2f}/1.0"
                        })
                except Exception as e:
                    self.logger.warning(f"CF explanation failed: {e}")
            
            # Get CB prediction and explanation
            if self.cb_model is not None:
                try:
                    cb_prediction = self.cb_model.predict(user_id, [item_id])
                    if len(cb_prediction) > 0:
                        cb_score = cb_prediction[0]
                        explanation['cb_score'] = float(cb_score)
                        
                        # Get feature importance if requested
                        feature_importance = None
                        if include_feature_importance:
                            try:
                                feature_importance = self.cb_model.get_feature_importance(item_id)
                            except Exception as e:
                                self.logger.warning(f"Feature importance extraction failed: {e}")
                        
                        explanation['explanations'].append({
                            'type': 'content_based',
                            'score': float(cb_score),
                            'weight': self.cb_weight if use_cf else 1.0,
                            'contribution': float(cb_score * (self.cb_weight if use_cf else 1.0)),
                            'description': f"Property features match your preferences with {cb_score:.2f}/1.0 similarity",
                            'feature_importance': feature_importance
                        })
                except Exception as e:
                    self.logger.warning(f"CB explanation failed: {e}")
            
            # Calculate final hybrid score
            if 'cf_score' in explanation and 'cb_score' in explanation:
                hybrid_score = (explanation['cf_score'] * self.cf_weight + 
                               explanation['cb_score'] * self.cb_weight)
                explanation['hybrid_score'] = float(hybrid_score)
                explanation['hybrid_method'] = 'weighted_average'
            elif 'cb_score' in explanation:
                explanation['hybrid_score'] = explanation['cb_score']
                explanation['hybrid_method'] = 'content_based_only'
            elif 'cf_score' in explanation:
                explanation['hybrid_score'] = explanation['cf_score']
                explanation['hybrid_method'] = 'collaborative_filtering_only'
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {'error': str(e)}
    
    def create_ab_experiment(self, 
                           experiment_id: str,
                           name: str,
                           description: str,
                           control_method: FusionMethod,
                           treatment_method: FusionMethod,
                           traffic_split: float = 0.5,
                           min_sample_size: int = 100,
                           max_duration_days: int = 30,
                           success_metrics: List[str] = None) -> bool:
        """Create a new A/B testing experiment"""
        try:
            if experiment_id in self.experiments:
                self.logger.warning(f"Experiment {experiment_id} already exists")
                return False
            
            experiment = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                control_method=control_method,
                treatment_method=treatment_method,
                traffic_split=traffic_split,
                min_sample_size=min_sample_size,
                max_duration_days=max_duration_days,
                success_metrics=success_metrics or ['precision', 'recall', 'diversity']
            )
            
            self.experiments[experiment_id] = experiment
            self.experiment_results[experiment_id] = {
                'control': {'users': [], 'metrics': {}},
                'treatment': {'users': [], 'metrics': {}}
            }
            
            self.logger.info(f"Created A/B experiment {experiment_id}: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            return False
    
    def _get_experiment_method(self, experiment_id: str, user_id: int) -> FusionMethod:
        """Get the fusion method for a user in an experiment"""
        try:
            if experiment_id not in self.experiments:
                return self.fusion_method
            
            experiment = self.experiments[experiment_id]
            
            # Simple hash-based assignment
            user_hash = hash(f"{experiment_id}_{user_id}") % 100
            
            if user_hash < experiment.traffic_split * 100:
                # Add user to treatment group
                self.experiment_results[experiment_id]['treatment']['users'].append(user_id)
                return experiment.treatment_method
            else:
                # Add user to control group
                self.experiment_results[experiment_id]['control']['users'].append(user_id)
                return experiment.control_method
                
        except Exception as e:
            self.logger.warning(f"Error getting experiment method: {e}")
            return self.fusion_method
    
    def record_user_feedback(self, 
                           user_id: int, 
                           item_id: int, 
                           feedback_type: str, 
                           feedback_value: float,
                           experiment_id: Optional[str] = None) -> None:
        """Record user feedback for adaptive learning"""
        try:
            feedback = {
                'item_id': item_id,
                'feedback_type': feedback_type,
                'feedback_value': feedback_value,
                'timestamp': datetime.now(),
                'experiment_id': experiment_id
            }
            
            self.user_feedback[user_id].append(feedback)
            
            # Update experiment results if applicable
            if experiment_id and experiment_id in self.experiment_results:
                # Determine which group the user is in
                control_users = self.experiment_results[experiment_id]['control']['users']
                treatment_users = self.experiment_results[experiment_id]['treatment']['users']
                
                if user_id in control_users:
                    group = 'control'
                elif user_id in treatment_users:
                    group = 'treatment'
                else:
                    group = None
                
                if group:
                    if feedback_type not in self.experiment_results[experiment_id][group]['metrics']:
                        self.experiment_results[experiment_id][group]['metrics'][feedback_type] = []
                    
                    self.experiment_results[experiment_id][group]['metrics'][feedback_type].append(feedback_value)
            
            # Trigger adaptive learning if enabled
            if self.enable_adaptive_learning:
                self._update_weights_from_feedback(user_id, feedback)
                
        except Exception as e:
            self.logger.error(f"Failed to record user feedback: {e}")
    
    def _update_weights_from_feedback(self, user_id: int, feedback: Dict[str, Any]) -> None:
        """Update model weights based on user feedback"""
        try:
            # Simple adaptive learning implementation
            # In a production system, this would be more sophisticated
            
            feedback_value = feedback['feedback_value']
            
            # If feedback is positive, slightly increase weights
            if feedback_value > 0.7:
                # Determine which model contributed more to this recommendation
                # This is simplified - in practice, you'd track which model contributed what
                user_context = self.get_user_context(user_id)
                
                if user_context in [UserContext.NEW_USER, UserContext.RETURNING_USER]:
                    # CB likely contributed more
                    self.cb_weight = min(0.8, self.cb_weight * 1.01)
                    self.cf_weight = 1.0 - self.cb_weight
                else:
                    # CF likely contributed more
                    self.cf_weight = min(0.8, self.cf_weight * 1.01)
                    self.cb_weight = 1.0 - self.cf_weight
            
            # Record weight changes
            self.weight_history.append({
                'timestamp': datetime.now(),
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'trigger': 'user_feedback',
                'user_id': user_id,
                'feedback_value': feedback_value
            })
            
            # Keep only recent history
            if len(self.weight_history) > 1000:
                self.weight_history = self.weight_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Failed to update weights from feedback: {e}")
    
    def evaluate_model_performance(self, 
                                 test_user_item_matrix: np.ndarray,
                                 test_property_data: List[Dict],
                                 k_values: List[int] = None) -> ModelPerformance:
        """Evaluate model performance with comprehensive metrics"""
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before evaluation")
            
            k_values = k_values or [5, 10, 20]
            start_time = time.time()
            
            # Generate test predictions
            test_predictions = []
            test_actual = []
            
            for user_id in range(test_user_item_matrix.shape[0]):
                # Get non-zero items for this user
                actual_items = np.where(test_user_item_matrix[user_id] > 0)[0]
                if len(actual_items) == 0:
                    continue
                
                # Get predictions
                try:
                    predictions = self.recommend(
                        user_id=user_id,
                        num_recommendations=max(k_values),
                        exclude_seen=False
                    )
                    
                    pred_items = [rec.item_id for rec in predictions]
                    pred_scores = [rec.predicted_rating for rec in predictions]
                    
                    test_predictions.append(pred_items)
                    test_actual.append(actual_items.tolist())
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed for user {user_id}: {e}")
                    continue
            
            # Calculate metrics
            rmse = self._calculate_rmse(test_predictions, test_actual, test_user_item_matrix)
            mae = self._calculate_mae(test_predictions, test_actual, test_user_item_matrix)
            
            precision_at_k = {}
            recall_at_k = {}
            ndcg_at_k = {}
            
            for k in k_values:
                precision_at_k[k] = self._calculate_precision_at_k(test_predictions, test_actual, k)
                recall_at_k[k] = self._calculate_recall_at_k(test_predictions, test_actual, k)
                ndcg_at_k[k] = self._calculate_ndcg_at_k(test_predictions, test_actual, k)
            
            coverage = self._calculate_coverage(test_predictions, test_user_item_matrix.shape[1])
            diversity = self._calculate_diversity(test_predictions)
            novelty = self._calculate_novelty(test_predictions)
            serendipity = self._calculate_serendipity(test_predictions, test_actual)
            
            response_time = (time.time() - start_time) * 1000 / len(test_predictions)
            
            performance = ModelPerformance(
                model_name=f"hybrid_{self.fusion_method.value}",
                rmse=rmse,
                mae=mae,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                ndcg_at_k=ndcg_at_k,
                coverage=coverage,
                diversity=diversity,
                novelty=novelty,
                serendipity=serendipity,
                response_time_ms=response_time
            )
            
            # Store performance history
            self.performance_history.append(performance)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _calculate_rmse(self, predictions: List[List[int]], actual: List[List[int]], matrix: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        try:
            mse_sum = 0.0
            count = 0
            
            for pred, act in zip(predictions, actual):
                for item in pred:
                    if item in act:
                        # For simplicity, assume predicted rating is 1.0 if in recommendations
                        predicted_rating = 1.0
                        actual_rating = 1.0  # Binary rating
                        mse_sum += (predicted_rating - actual_rating) ** 2
                        count += 1
            
            return np.sqrt(mse_sum / count) if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"RMSE calculation failed: {e}")
            return 0.0
    
    def _calculate_mae(self, predictions: List[List[int]], actual: List[List[int]], matrix: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        try:
            mae_sum = 0.0
            count = 0
            
            for pred, act in zip(predictions, actual):
                for item in pred:
                    if item in act:
                        predicted_rating = 1.0
                        actual_rating = 1.0
                        mae_sum += abs(predicted_rating - actual_rating)
                        count += 1
            
            return mae_sum / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"MAE calculation failed: {e}")
            return 0.0
    
    def _calculate_precision_at_k(self, predictions: List[List[int]], actual: List[List[int]], k: int) -> float:
        """Calculate Precision@K"""
        try:
            precision_sum = 0.0
            count = 0
            
            for pred, act in zip(predictions, actual):
                if len(pred) == 0:
                    continue
                
                pred_k = pred[:k]
                relevant_items = set(act)
                
                hits = len([item for item in pred_k if item in relevant_items])
                precision = hits / len(pred_k)
                precision_sum += precision
                count += 1
            
            return precision_sum / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Precision@K calculation failed: {e}")
            return 0.0
    
    def _calculate_recall_at_k(self, predictions: List[List[int]], actual: List[List[int]], k: int) -> float:
        """Calculate Recall@K"""
        try:
            recall_sum = 0.0
            count = 0
            
            for pred, act in zip(predictions, actual):
                if len(act) == 0:
                    continue
                
                pred_k = pred[:k]
                relevant_items = set(act)
                
                hits = len([item for item in pred_k if item in relevant_items])
                recall = hits / len(relevant_items)
                recall_sum += recall
                count += 1
            
            return recall_sum / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Recall@K calculation failed: {e}")
            return 0.0
    
    def _calculate_ndcg_at_k(self, predictions: List[List[int]], actual: List[List[int]], k: int) -> float:
        """Calculate NDCG@K"""
        try:
            ndcg_sum = 0.0
            count = 0
            
            for pred, act in zip(predictions, actual):
                if len(pred) == 0 or len(act) == 0:
                    continue
                
                pred_k = pred[:k]
                relevant_items = set(act)
                
                # Calculate DCG
                dcg = 0.0
                for i, item in enumerate(pred_k):
                    if item in relevant_items:
                        dcg += 1.0 / np.log2(i + 2)
                
                # Calculate IDCG
                idcg = 0.0
                for i in range(min(len(relevant_items), k)):
                    idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_sum += ndcg
                count += 1
            
            return ndcg_sum / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"NDCG@K calculation failed: {e}")
            return 0.0
    
    def _calculate_coverage(self, predictions: List[List[int]], total_items: int) -> float:
        """Calculate catalog coverage"""
        try:
            recommended_items = set()
            for pred in predictions:
                recommended_items.update(pred)
            
            return len(recommended_items) / total_items if total_items > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Coverage calculation failed: {e}")
            return 0.0
    
    def _calculate_diversity(self, predictions: List[List[int]]) -> float:
        """Calculate intra-list diversity"""
        try:
            if not predictions:
                return 0.0
            
            diversity_sum = 0.0
            count = 0
            
            for pred in predictions:
                if len(pred) <= 1:
                    continue
                
                # Calculate pairwise diversity
                pair_diversity = 0.0
                pair_count = 0
                
                for i in range(len(pred)):
                    for j in range(i + 1, len(pred)):
                        item1, item2 = pred[i], pred[j]
                        
                        # Simple diversity calculation (different items are diverse)
                        if item1 != item2:
                            if item1 in self.item_features_cache and item2 in self.item_features_cache:
                                features1 = self.item_features_cache[item1]
                                features2 = self.item_features_cache[item2]
                                
                                # Calculate cosine similarity
                                similarity = np.dot(features1, features2) / (
                                    np.linalg.norm(features1) * np.linalg.norm(features2)
                                )
                                diversity = 1 - similarity
                            else:
                                diversity = 0.5  # Default diversity
                            
                            pair_diversity += diversity
                            pair_count += 1
                
                if pair_count > 0:
                    diversity_sum += pair_diversity / pair_count
                    count += 1
            
            return diversity_sum / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Diversity calculation failed: {e}")
            return 0.0
    
    def _calculate_novelty(self, predictions: List[List[int]]) -> float:
        """Calculate novelty score"""
        try:
            # Simple novelty calculation - more sophisticated versions would use item popularity
            if not predictions:
                return 0.0
            
            total_items = sum(len(pred) for pred in predictions)
            unique_items = len(set(item for pred in predictions for item in pred))
            
            return unique_items / total_items if total_items > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Novelty calculation failed: {e}")
            return 0.0
    
    def _calculate_serendipity(self, predictions: List[List[int]], actual: List[List[int]]) -> float:
        """Calculate serendipity score"""
        try:
            # Serendipity = unexpected relevant items
            serendipity_sum = 0.0
            count = 0
            
            for pred, act in zip(predictions, actual):
                if len(pred) == 0 or len(act) == 0:
                    continue
                
                relevant_items = set(act)
                unexpected_relevant = 0
                
                for item in pred:
                    if item in relevant_items:
                        # Simple unexpectedness calculation
                        # In practice, this would consider user's typical preferences
                        unexpectedness = 0.5  # Placeholder
                        unexpected_relevant += unexpectedness
                
                serendipity = unexpected_relevant / len(pred)
                serendipity_sum += serendipity
                count += 1
            
            return serendipity_sum / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Serendipity calculation failed: {e}")
            return 0.0
    
    def _generate_cache_key(self, 
                          user_id: int, 
                          num_recommendations: int, 
                          context: Optional[RecommendationContext]) -> str:
        """Generate cache key for recommendations"""
        try:
            key_parts = [str(user_id), str(num_recommendations)]
            
            if context:
                if context.session_id:
                    key_parts.append(context.session_id)
                if context.device_type:
                    key_parts.append(context.device_type)
                if context.location:
                    key_parts.append(context.location)
            
            key_string = "_".join(key_parts)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Cache key generation failed: {e}")
            return f"user_{user_id}_num_{num_recommendations}"
    
    def _get_cached_recommendations(self, cache_key: str) -> Optional[List[HybridRecommendationResult]]:
        """Get cached recommendations if available and valid"""
        try:
            with self.cache_lock:
                if cache_key in self.recommendation_cache:
                    recommendations, timestamp = self.recommendation_cache[cache_key]
                    
                    # Check if cache is still valid (5 minutes)
                    if datetime.now() - timestamp < timedelta(minutes=5):
                        return recommendations
                    else:
                        # Remove expired cache
                        del self.recommendation_cache[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def _cache_recommendations(self, 
                             cache_key: str, 
                             recommendations: List[HybridRecommendationResult]) -> None:
        """Cache recommendations for future use"""
        try:
            with self.cache_lock:
                # Clean old cache entries if cache is full
                if len(self.recommendation_cache) >= self.cache_size:
                    oldest_key = min(self.recommendation_cache.keys(), 
                                   key=lambda k: self.recommendation_cache[k][1])
                    del self.recommendation_cache[oldest_key]
                
                # Add new cache entry
                self.recommendation_cache[cache_key] = (recommendations, datetime.now())
                
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results from an A/B testing experiment"""
        try:
            if experiment_id not in self.experiments:
                return {'error': f'Experiment {experiment_id} not found'}
            
            experiment = self.experiments[experiment_id]
            results = self.experiment_results[experiment_id]
            
            # Calculate summary statistics
            summary = {
                'experiment_id': experiment_id,
                'name': experiment.name,
                'description': experiment.description,
                'control_method': experiment.control_method.value,
                'treatment_method': experiment.treatment_method.value,
                'traffic_split': experiment.traffic_split,
                'control_users': len(results['control']['users']),
                'treatment_users': len(results['treatment']['users']),
                'metrics': {}
            }
            
            # Calculate metric comparisons
            for metric in experiment.success_metrics:
                if metric in results['control']['metrics'] and metric in results['treatment']['metrics']:
                    control_values = results['control']['metrics'][metric]
                    treatment_values = results['treatment']['metrics'][metric]
                    
                    if control_values and treatment_values:
                        summary['metrics'][metric] = {
                            'control_mean': np.mean(control_values),
                            'treatment_mean': np.mean(treatment_values),
                            'control_std': np.std(control_values),
                            'treatment_std': np.std(treatment_values),
                            'sample_size_control': len(control_values),
                            'sample_size_treatment': len(treatment_values),
                            'lift': (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values) * 100
                        }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment results: {e}")
            return {'error': str(e)}
    
    def save_models(self, cf_model_path: str, cb_model_path: str, metadata_path: str = None):
        """Save both trained models and hybrid system metadata"""
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before saving")
            
            # Save individual models
            if self.cf_model is not None:
                self.cf_model.save_model(cf_model_path)
            
            if self.cb_model is not None:
                self.cb_model.save_model(cb_model_path)
            
            # Save hybrid system metadata
            if metadata_path:
                hybrid_metadata = {
                    'cf_weight': self.cf_weight,
                    'cb_weight': self.cb_weight,
                    'fusion_method': self.fusion_method.value,
                    'cold_start_strategy': self.cold_start_strategy.value,
                    'diversity_config': {
                        'intra_list_diversity': self.diversity_config.intra_list_diversity,
                        'novelty_weight': self.diversity_config.novelty_weight,
                        'serendipity_weight': self.diversity_config.serendipity_weight,
                        'coverage_weight': self.diversity_config.coverage_weight,
                        'diversity_decay': self.diversity_config.diversity_decay,
                        'min_diversity_threshold': self.diversity_config.min_diversity_threshold,
                        'max_similarity_threshold': self.diversity_config.max_similarity_threshold,
                        'use_clustering': self.diversity_config.use_clustering,
                        'clustering_method': self.diversity_config.clustering_method,
                        'num_clusters': self.diversity_config.num_clusters
                    },
                    'training_metadata': self.training_metadata,
                    'performance_history': [
                        {
                            'model_name': perf.model_name,
                            'rmse': perf.rmse,
                            'mae': perf.mae,
                            'precision_at_k': perf.precision_at_k,
                            'recall_at_k': perf.recall_at_k,
                            'ndcg_at_k': perf.ndcg_at_k,
                            'coverage': perf.coverage,
                            'diversity': perf.diversity,
                            'novelty': perf.novelty,
                            'serendipity': perf.serendipity,
                            'response_time_ms': perf.response_time_ms,
                            'timestamp': perf.timestamp.isoformat()
                        } for perf in self.performance_history
                    ],
                    'weight_history': [
                        {
                            'timestamp': wh['timestamp'].isoformat(),
                            'cf_weight': wh['cf_weight'],
                            'cb_weight': wh['cb_weight'],
                            'trigger': wh['trigger'],
                            'user_id': wh.get('user_id'),
                            'feedback_value': wh.get('feedback_value')
                        } for wh in self.weight_history
                    ]
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(hybrid_metadata, f, indent=2)
            
            self.logger.info(f"Hybrid system models saved to {cf_model_path}, {cb_model_path}, and {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise
    
    def load_models(self, cf_model_path: str, cb_model_path: str, metadata_path: str = None):
        """Load both trained models and hybrid system metadata"""
        try:
            # Load individual models
            if self.cf_model is not None:
                self.cf_model.load_model(cf_model_path)
            
            if self.cb_model is not None:
                self.cb_model.load_model(cb_model_path)
            
            # Load hybrid system metadata
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    hybrid_metadata = json.load(f)
                
                # Restore configuration
                self.cf_weight = hybrid_metadata.get('cf_weight', self.cf_weight)
                self.cb_weight = hybrid_metadata.get('cb_weight', self.cb_weight)
                
                if 'fusion_method' in hybrid_metadata:
                    self.fusion_method = FusionMethod(hybrid_metadata['fusion_method'])
                
                if 'cold_start_strategy' in hybrid_metadata:
                    self.cold_start_strategy = ColdStartStrategy(hybrid_metadata['cold_start_strategy'])
                
                if 'diversity_config' in hybrid_metadata:
                    dc = hybrid_metadata['diversity_config']
                    self.diversity_config = DiversityConfig(
                        intra_list_diversity=dc.get('intra_list_diversity', 0.8),
                        novelty_weight=dc.get('novelty_weight', 0.2),
                        serendipity_weight=dc.get('serendipity_weight', 0.1),
                        coverage_weight=dc.get('coverage_weight', 0.1),
                        diversity_decay=dc.get('diversity_decay', 0.9),
                        min_diversity_threshold=dc.get('min_diversity_threshold', 0.3),
                        max_similarity_threshold=dc.get('max_similarity_threshold', 0.7),
                        use_clustering=dc.get('use_clustering', True),
                        clustering_method=dc.get('clustering_method', 'kmeans'),
                        num_clusters=dc.get('num_clusters', 20)
                    )
                
                # Restore training metadata
                self.training_metadata = hybrid_metadata.get('training_metadata', {})
                
                # Restore performance history
                if 'performance_history' in hybrid_metadata:
                    self.performance_history = []
                    for perf_data in hybrid_metadata['performance_history']:
                        perf = ModelPerformance(
                            model_name=perf_data['model_name'],
                            rmse=perf_data['rmse'],
                            mae=perf_data['mae'],
                            precision_at_k=perf_data['precision_at_k'],
                            recall_at_k=perf_data['recall_at_k'],
                            ndcg_at_k=perf_data['ndcg_at_k'],
                            coverage=perf_data['coverage'],
                            diversity=perf_data['diversity'],
                            novelty=perf_data['novelty'],
                            serendipity=perf_data['serendipity'],
                            response_time_ms=perf_data['response_time_ms'],
                            timestamp=datetime.fromisoformat(perf_data['timestamp'])
                        )
                        self.performance_history.append(perf)
                
                # Restore weight history
                if 'weight_history' in hybrid_metadata:
                    self.weight_history = []
                    for wh_data in hybrid_metadata['weight_history']:
                        wh = {
                            'timestamp': datetime.fromisoformat(wh_data['timestamp']),
                            'cf_weight': wh_data['cf_weight'],
                            'cb_weight': wh_data['cb_weight'],
                            'trigger': wh_data['trigger'],
                            'user_id': wh_data.get('user_id'),
                            'feedback_value': wh_data.get('feedback_value')
                        }
                        self.weight_history.append(wh)
            
            self.is_trained = True
            self.logger.info(f"Hybrid system models loaded from {cf_model_path}, {cb_model_path}, and {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'system_type': 'comprehensive_hybrid_recommendation_system',
            'fusion_method': self.fusion_method.value,
            'cold_start_strategy': self.cold_start_strategy.value,
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'min_cf_interactions': self.min_cf_interactions,
            'fallback_to_content': self.fallback_to_content,
            'explanation_detail_level': self.explanation_detail_level,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'diversity_config': {
                'intra_list_diversity': self.diversity_config.intra_list_diversity,
                'novelty_weight': self.diversity_config.novelty_weight,
                'serendipity_weight': self.diversity_config.serendipity_weight,
                'coverage_weight': self.diversity_config.coverage_weight,
                'diversity_decay': self.diversity_config.diversity_decay,
                'min_diversity_threshold': self.diversity_config.min_diversity_threshold,
                'max_similarity_threshold': self.diversity_config.max_similarity_threshold,
                'use_clustering': self.diversity_config.use_clustering,
                'clustering_method': self.diversity_config.clustering_method,
                'num_clusters': self.diversity_config.num_clusters
            },
            'advanced_features': {
                'enable_ab_testing': self.enable_ab_testing,
                'enable_adaptive_learning': self.enable_adaptive_learning,
                'enable_real_time_serving': self.enable_real_time_serving,
                'cache_size': self.cache_size,
                'model_update_frequency': self.model_update_frequency
            },
            'performance_stats': {
                'num_performance_records': len(self.performance_history),
                'num_weight_changes': len(self.weight_history),
                'cache_utilization': len(self.recommendation_cache) / self.cache_size,
                'active_experiments': len([exp for exp in self.experiments.values() if exp.is_active])
            }
        }
        
        if self.cf_model is not None:
            info['cf_model_info'] = self.cf_model.get_model_info()
        
        if self.cb_model is not None:
            info['cb_model_info'] = self.cb_model.get_model_info()
        
        return info
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the hybrid system"""
        metrics = {
            'hybrid_system': {
                'fusion_method': self.fusion_method.value,
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'is_trained': self.is_trained,
                'last_model_update': self.last_model_update.isoformat()
            }
        }
        
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            metrics['latest_performance'] = {
                'rmse': latest_performance.rmse,
                'mae': latest_performance.mae,
                'precision_at_k': latest_performance.precision_at_k,
                'recall_at_k': latest_performance.recall_at_k,
                'ndcg_at_k': latest_performance.ndcg_at_k,
                'coverage': latest_performance.coverage,
                'diversity': latest_performance.diversity,
                'novelty': latest_performance.novelty,
                'serendipity': latest_performance.serendipity,
                'response_time_ms': latest_performance.response_time_ms,
                'timestamp': latest_performance.timestamp.isoformat()
            }
        
        if self.training_metadata and 'training_results' in self.training_metadata:
            training_results = self.training_metadata['training_results']
            
            if 'cf_results' in training_results:
                metrics['collaborative_filtering'] = training_results['cf_results']
            
            if 'cb_results' in training_results:
                metrics['content_based'] = training_results['cb_results']
        
        return metrics
    
    def update_weights(self, cf_weight: float, cb_weight: float, trigger: str = "manual"):
        """Update the weights for hybrid recommendations"""
        # Validate weights
        if not (0.0 <= cf_weight <= 1.0) or not (0.0 <= cb_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        
        # Normalize weights
        total_weight = cf_weight + cb_weight
        if total_weight > 0:
            old_cf_weight = self.cf_weight
            old_cb_weight = self.cb_weight
            
            self.cf_weight = cf_weight / total_weight
            self.cb_weight = cb_weight / total_weight
            
            # Record weight change
            self.weight_history.append({
                'timestamp': datetime.now(),
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'trigger': trigger,
                'old_cf_weight': old_cf_weight,
                'old_cb_weight': old_cb_weight
            })
            
            self.logger.info(f"Updated weights - CF: {self.cf_weight:.2f}, CB: {self.cb_weight:.2f} (trigger: {trigger})")
        else:
            raise ValueError("At least one weight must be positive")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if self.serving_pool:
                self.serving_pool.shutdown(wait=True)
        except:
            pass