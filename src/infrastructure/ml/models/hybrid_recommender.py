import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .collaborative_filter import CollaborativeFilteringModel, RecommendationResult
from .content_recommender import ContentBasedRecommender


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


class HybridRecommendationSystem:
    """
    Hybrid Recommendation System that combines collaborative filtering and content-based approaches.
    
    This system implements a weighted hybrid approach that:
    1. Combines predictions from both collaborative filtering and content-based models
    2. Handles cold start problems by falling back to content-based recommendations
    3. Provides configurable weighting between different recommendation strategies
    4. Includes comprehensive explanation generation for recommendations
    5. Supports both individual and batch prediction modes
    
    The system follows the masterplan architecture and provides production-ready
    recommendation capabilities for rental property systems.
    """
    
    def __init__(self, 
                 cf_weight: float = 0.6,
                 cb_weight: float = 0.4,
                 min_cf_interactions: int = 5,
                 fallback_to_content: bool = True,
                 explanation_detail_level: str = "detailed"):
        """
        Initialize the HybridRecommendationSystem.
        
        Args:
            cf_weight: Weight for collaborative filtering recommendations (0.0 to 1.0)
            cb_weight: Weight for content-based recommendations (0.0 to 1.0)
            min_cf_interactions: Minimum user interactions required for CF recommendations
            fallback_to_content: Whether to fall back to content-based for new users
            explanation_detail_level: Level of detail for explanations ("simple", "detailed", "technical")
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
        
        self.min_cf_interactions = min_cf_interactions
        self.fallback_to_content = fallback_to_content
        self.explanation_detail_level = explanation_detail_level
        
        # Model instances
        self.cf_model: Optional[CollaborativeFilteringModel] = None
        self.cb_model: Optional[ContentBasedRecommender] = None
        
        # Training state
        self.is_trained = False
        self.training_metadata: Dict[str, Any] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized HybridRecommendationSystem with CF weight: {self.cf_weight:.2f}, "
                        f"CB weight: {self.cb_weight:.2f}")
    
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
            
            # Store training metadata
            self.training_metadata = {
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'num_users': user_item_matrix.shape[0],
                'num_items': user_item_matrix.shape[1],
                'num_properties': len(property_data),
                'cf_epochs': cf_epochs,
                'cb_epochs': cb_epochs,
                'training_results': training_results
            }
            
            self.is_trained = True
            
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
    
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        Generate hybrid predictions for user-item pairs.
        
        Args:
            user_id: User identifier
            item_ids: List of item identifiers
            
        Returns:
            Array of hybrid prediction scores
        """
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before making predictions")
            
            if not item_ids:
                return np.array([])
            
            # Check if user has sufficient interactions for CF
            use_cf = self._should_use_collaborative_filtering(user_id)
            
            predictions = np.zeros(len(item_ids))
            
            if use_cf and self.cf_model is not None:
                # Get collaborative filtering predictions
                try:
                    cf_predictions = self.cf_model.predict(user_id, item_ids)
                    if len(cf_predictions) > 0:
                        predictions += self.cf_weight * cf_predictions
                    else:
                        use_cf = False
                except Exception as e:
                    self.logger.warning(f"CF prediction failed for user {user_id}: {e}")
                    use_cf = False
            
            # Get content-based predictions
            if self.cb_model is not None:
                try:
                    cb_predictions = self.cb_model.predict(user_id, item_ids)
                    if len(cb_predictions) > 0:
                        if use_cf:
                            predictions += self.cb_weight * cb_predictions
                        else:
                            # If CF not available, use only content-based
                            predictions = cb_predictions
                except Exception as e:
                    self.logger.warning(f"CB prediction failed for user {user_id}: {e}")
                    if not use_cf:
                        # If both fail, return zeros
                        predictions = np.zeros(len(item_ids))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.array([])
    
    def recommend(self, 
                  user_id: int, 
                  num_recommendations: int = 10,
                  exclude_seen: bool = True,
                  diversity_threshold: float = 0.1,
                  include_explanations: bool = True) -> List[HybridRecommendationResult]:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to generate
            exclude_seen: Whether to exclude previously seen items
            diversity_threshold: Minimum diversity score for recommendations
            include_explanations: Whether to include detailed explanations
            
        Returns:
            List of HybridRecommendationResult objects
        """
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before making recommendations")
            
            # Check if user has sufficient interactions for CF
            use_cf = self._should_use_collaborative_filtering(user_id)
            
            recommendations = []
            
            if use_cf and self.cf_model is not None:
                # Get collaborative filtering recommendations
                try:
                    cf_recommendations = self.cf_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations * 2,  # Get more for diversity
                        exclude_seen=exclude_seen
                    )
                    
                    # Get content-based recommendations
                    cb_recommendations = []
                    if self.cb_model is not None:
                        try:
                            cb_recommendations = self.cb_model.recommend(
                                user_id=user_id,
                                num_recommendations=num_recommendations * 2,
                                exclude_seen=exclude_seen
                            )
                        except Exception as e:
                            self.logger.warning(f"CB recommendations failed: {e}")
                    
                    # Combine recommendations
                    recommendations = self._combine_recommendations(
                        cf_recommendations, cb_recommendations, num_recommendations
                    )
                    
                except Exception as e:
                    self.logger.warning(f"CF recommendations failed: {e}")
                    use_cf = False
            
            # Fall back to content-based if CF not available
            if not use_cf and self.fallback_to_content and self.cb_model is not None:
                try:
                    cb_recommendations = self.cb_model.recommend(
                        user_id=user_id,
                        num_recommendations=num_recommendations,
                        exclude_seen=exclude_seen
                    )
                    
                    # Convert to hybrid results
                    recommendations = self._convert_to_hybrid_results(
                        cb_recommendations, method="content_based_only"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Content-based fallback failed: {e}")
                    return []
            
            # Add explanations if requested
            if include_explanations:
                recommendations = self._add_detailed_explanations(user_id, recommendations)
            
            # Apply diversity filtering
            if diversity_threshold > 0:
                recommendations = self._apply_diversity_filter(recommendations, diversity_threshold)
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def explain_recommendation(self, 
                             user_id: int, 
                             item_id: int,
                             include_feature_importance: bool = True) -> Dict[str, Any]:
        """
        Generate detailed explanation for a specific recommendation.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            include_feature_importance: Whether to include feature importance
            
        Returns:
            Dictionary containing detailed explanation
        """
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before generating explanations")
            
            explanation = {
                'user_id': user_id,
                'item_id': item_id,
                'hybrid_method': 'weighted_average',
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
    
    def _should_use_collaborative_filtering(self, user_id: int) -> bool:
        """
        Determine if collaborative filtering should be used for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Boolean indicating if CF should be used
        """
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
    
    def _combine_recommendations(self, 
                               cf_recommendations: List[RecommendationResult],
                               cb_recommendations: List[RecommendationResult],
                               num_recommendations: int) -> List[HybridRecommendationResult]:
        """
        Combine collaborative filtering and content-based recommendations.
        
        Args:
            cf_recommendations: CF recommendations
            cb_recommendations: CB recommendations
            num_recommendations: Target number of recommendations
            
        Returns:
            List of hybrid recommendations
        """
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
                    hybrid_score += self.cf_weight * cf_score
                
                if cb_rec is not None:
                    cb_score = cb_rec.predicted_rating
                    hybrid_score += self.cb_weight * cb_score
                
                # Calculate confidence
                confidence = self._calculate_hybrid_confidence(cf_rec, cb_rec)
                
                # Generate explanation
                explanation = self._generate_hybrid_explanation(cf_rec, cb_rec)
                
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
            
            return hybrid_recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation combination failed: {e}")
            return []
    
    def _convert_to_hybrid_results(self, 
                                 recommendations: List[RecommendationResult],
                                 method: str) -> List[HybridRecommendationResult]:
        """
        Convert single-model recommendations to hybrid results.
        
        Args:
            recommendations: Single-model recommendations
            method: Method used for recommendation
            
        Returns:
            List of hybrid recommendation results
        """
        hybrid_results = []
        
        for rec in recommendations:
            hybrid_rec = HybridRecommendationResult(
                item_id=rec.item_id,
                predicted_rating=rec.predicted_rating,
                confidence_score=rec.confidence_score,
                explanation=rec.explanation,
                cf_score=rec.predicted_rating if method == "collaborative_filtering_only" else None,
                cb_score=rec.predicted_rating if method == "content_based_only" else None,
                hybrid_method=method
            )
            hybrid_results.append(hybrid_rec)
        
        return hybrid_results
    
    def _calculate_hybrid_confidence(self, 
                                   cf_rec: Optional[RecommendationResult],
                                   cb_rec: Optional[RecommendationResult]) -> float:
        """
        Calculate confidence score for hybrid recommendation.
        
        Args:
            cf_rec: CF recommendation result
            cb_rec: CB recommendation result
            
        Returns:
            Confidence score
        """
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
                # Only CF available
                return cf_rec.confidence_score * 0.9  # Slight penalty for single model
            
            elif cb_rec is not None:
                # Only CB available
                return cb_rec.confidence_score * 0.9  # Slight penalty for single model
            
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_hybrid_explanation(self, 
                                   cf_rec: Optional[RecommendationResult],
                                   cb_rec: Optional[RecommendationResult]) -> str:
        """
        Generate explanation for hybrid recommendation.
        
        Args:
            cf_rec: CF recommendation result
            cb_rec: CB recommendation result
            
        Returns:
            Explanation string
        """
        try:
            if cf_rec is not None and cb_rec is not None:
                if self.explanation_detail_level == "simple":
                    return "Recommended based on similar users and property features"
                elif self.explanation_detail_level == "detailed":
                    return (f"Recommended by similar users (score: {cf_rec.predicted_rating:.2f}) "
                           f"and property features (score: {cb_rec.predicted_rating:.2f})")
                else:  # technical
                    return (f"Hybrid recommendation: CF({cf_rec.predicted_rating:.3f}) × {self.cf_weight:.2f} + "
                           f"CB({cb_rec.predicted_rating:.3f}) × {self.cb_weight:.2f}")
            
            elif cf_rec is not None:
                return cf_rec.explanation
            
            elif cb_rec is not None:
                return cb_rec.explanation
            
            else:
                return "No explanation available"
                
        except Exception as e:
            self.logger.warning(f"Explanation generation failed: {e}")
            return "Recommendation explanation unavailable"
    
    def _add_detailed_explanations(self, 
                                 user_id: int, 
                                 recommendations: List[HybridRecommendationResult]) -> List[HybridRecommendationResult]:
        """
        Add detailed explanations to recommendations.
        
        Args:
            user_id: User identifier
            recommendations: List of recommendations
            
        Returns:
            Recommendations with enhanced explanations
        """
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
                self.logger.warning(f"Detailed explanation failed for item {rec.item_id}: {e}")
        
        return recommendations
    
    def _apply_diversity_filter(self, 
                              recommendations: List[HybridRecommendationResult],
                              diversity_threshold: float) -> List[HybridRecommendationResult]:
        """
        Apply diversity filtering to recommendations.
        
        Args:
            recommendations: List of recommendations
            diversity_threshold: Minimum diversity score
            
        Returns:
            Filtered recommendations with diversity
        """
        try:
            if len(recommendations) <= 1:
                return recommendations
            
            # This is a simplified diversity filter
            # In a real implementation, you would use property features
            # to calculate actual diversity scores
            
            diverse_recommendations = []
            used_scores = set()
            
            for rec in recommendations:
                # Simple diversity check based on score ranges
                score_range = int(rec.predicted_rating * 10) / 10
                
                if score_range not in used_scores or len(diverse_recommendations) == 0:
                    diverse_recommendations.append(rec)
                    used_scores.add(score_range)
                elif len(diverse_recommendations) < len(recommendations) * 0.8:
                    # Allow some similar items but not too many
                    diverse_recommendations.append(rec)
            
            return diverse_recommendations
            
        except Exception as e:
            self.logger.warning(f"Diversity filtering failed: {e}")
            return recommendations
    
    def save_models(self, cf_model_path: str, cb_model_path: str):
        """
        Save both trained models.
        
        Args:
            cf_model_path: Path to save CF model
            cb_model_path: Path to save CB model
        """
        try:
            if not self.is_trained:
                raise ValueError("System must be trained before saving")
            
            if self.cf_model is not None:
                self.cf_model.save_model(cf_model_path)
            
            if self.cb_model is not None:
                self.cb_model.save_model(cb_model_path)
            
            self.logger.info(f"Hybrid system models saved to {cf_model_path} and {cb_model_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise
    
    def load_models(self, cf_model_path: str, cb_model_path: str):
        """
        Load both trained models.
        
        Args:
            cf_model_path: Path to load CF model
            cb_model_path: Path to load CB model
        """
        try:
            if self.cf_model is not None:
                self.cf_model.load_model(cf_model_path)
            
            if self.cb_model is not None:
                self.cb_model.load_model(cb_model_path)
            
            self.is_trained = True
            self.logger.info(f"Hybrid system models loaded from {cf_model_path} and {cb_model_path}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary containing system information
        """
        info = {
            'system_type': 'hybrid_recommendation_system',
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'min_cf_interactions': self.min_cf_interactions,
            'fallback_to_content': self.fallback_to_content,
            'explanation_detail_level': self.explanation_detail_level,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }
        
        if self.cf_model is not None:
            info['cf_model_info'] = self.cf_model.get_model_info()
        
        if self.cb_model is not None:
            info['cb_model_info'] = self.cb_model.get_model_info()
        
        return info
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for both models.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            'hybrid_system': {
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'is_trained': self.is_trained
            }
        }
        
        if self.training_metadata and 'training_results' in self.training_metadata:
            training_results = self.training_metadata['training_results']
            
            if 'cf_results' in training_results:
                metrics['collaborative_filtering'] = training_results['cf_results']
            
            if 'cb_results' in training_results:
                metrics['content_based'] = training_results['cb_results']
        
        return metrics
    
    def update_weights(self, cf_weight: float, cb_weight: float):
        """
        Update the weights for hybrid recommendations.
        
        Args:
            cf_weight: New weight for collaborative filtering
            cb_weight: New weight for content-based
        """
        # Validate weights
        if not (0.0 <= cf_weight <= 1.0) or not (0.0 <= cb_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        
        # Normalize weights
        total_weight = cf_weight + cb_weight
        if total_weight > 0:
            self.cf_weight = cf_weight / total_weight
            self.cb_weight = cb_weight / total_weight
        else:
            raise ValueError("At least one weight must be positive")
        
        self.logger.info(f"Updated weights - CF: {self.cf_weight:.2f}, CB: {self.cb_weight:.2f}")