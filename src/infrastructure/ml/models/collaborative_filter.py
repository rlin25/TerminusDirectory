import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import json
import pickle
import os
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RecommendationResult:
    item_id: int
    predicted_rating: float
    confidence_score: float
    explanation: str
    

@dataclass
class EvaluationMetrics:
    rmse: float
    mae: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    coverage: float
    diversity: float
    novelty: float
    

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    min_lr: float = 1e-6
    embedding_dim: int = 128
    hidden_layers: List[int] = None
    dropout_rate: float = 0.2
    regularization: float = 1e-6
    negative_sampling_ratio: float = 0.5
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]
            

class DataPreprocessor:
    """Comprehensive data preprocessing for collaborative filtering"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.rating_scaler = StandardScaler()
        self.user_features_scaler = StandardScaler()
        self.item_features_scaler = StandardScaler()
        self.is_fitted = False
        self.user_stats = {}
        self.item_stats = {}
        
    def fit_transform_interactions(self, 
                                 interactions_df: pd.DataFrame,
                                 user_col: str = 'user_id',
                                 item_col: str = 'item_id', 
                                 rating_col: str = 'rating',
                                 timestamp_col: str = 'timestamp') -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit and transform user-item interactions"""
        try:
            self.logger.info(f"Preprocessing {len(interactions_df)} interactions")
            
            # Encode user and item IDs
            interactions_df['user_encoded'] = self.user_encoder.fit_transform(interactions_df[user_col])
            interactions_df['item_encoded'] = self.item_encoder.fit_transform(interactions_df[item_col])
            
            # Handle ratings
            ratings = interactions_df[rating_col].values.reshape(-1, 1)
            normalized_ratings = self.rating_scaler.fit_transform(ratings).flatten()
            
            # Calculate user and item statistics
            self._calculate_statistics(interactions_df, user_col, item_col, rating_col)
            
            # Create user-item matrix
            num_users = len(self.user_encoder.classes_)
            num_items = len(self.item_encoder.classes_)
            
            user_item_matrix = np.zeros((num_users, num_items))
            
            for _, row in interactions_df.iterrows():
                user_idx = int(row['user_encoded'])
                item_idx = int(row['item_encoded'])
                rating = row[rating_col]
                user_item_matrix[user_idx, item_idx] = rating
                
            # Prepare training data
            training_data = {
                'users': interactions_df['user_encoded'].values,
                'items': interactions_df['item_encoded'].values,
                'ratings': normalized_ratings,
                'original_ratings': interactions_df[rating_col].values,
                'user_item_matrix': user_item_matrix,
                'num_users': num_users,
                'num_items': num_items
            }
            
            # Add timestamp features if available
            if timestamp_col in interactions_df.columns:
                training_data['timestamps'] = pd.to_datetime(interactions_df[timestamp_col])
                
            self.is_fitted = True
            self.logger.info(f"Preprocessed data: {num_users} users, {num_items} items")
            
            return user_item_matrix, training_data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _calculate_statistics(self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str):
        """Calculate user and item statistics"""
        # User statistics
        user_stats = df.groupby(user_col).agg({
            rating_col: ['count', 'mean', 'std'],
            item_col: 'nunique'
        }).reset_index()
        user_stats.columns = [user_col, 'rating_count', 'rating_mean', 'rating_std', 'item_count']
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        
        # Item statistics
        item_stats = df.groupby(item_col).agg({
            rating_col: ['count', 'mean', 'std'],
            user_col: 'nunique'
        }).reset_index()
        item_stats.columns = [item_col, 'rating_count', 'rating_mean', 'rating_std', 'user_count']
        item_stats['rating_std'] = item_stats['rating_std'].fillna(0)
        
        # Store statistics
        self.user_stats = user_stats.set_index(user_col).to_dict('index')
        self.item_stats = item_stats.set_index(item_col).to_dict('index')
    
    def add_negative_samples(self, training_data: Dict[str, Any], 
                           negative_ratio: float = 0.5) -> Dict[str, Any]:
        """Add negative samples for implicit feedback"""
        try:
            users = training_data['users']
            items = training_data['items']
            ratings = training_data['ratings']
            user_item_matrix = training_data['user_item_matrix']
            
            # Generate negative samples
            num_negative = int(len(users) * negative_ratio)
            negative_users = []
            negative_items = []
            negative_ratings = []
            
            for _ in range(num_negative):
                user_id = np.random.randint(0, training_data['num_users'])
                item_id = np.random.randint(0, training_data['num_items'])
                
                # Ensure it's truly a negative sample
                if user_item_matrix[user_id, item_id] == 0:
                    negative_users.append(user_id)
                    negative_items.append(item_id)
                    negative_ratings.append(0.0)
            
            # Combine positive and negative samples
            all_users = np.concatenate([users, negative_users])
            all_items = np.concatenate([items, negative_items])
            all_ratings = np.concatenate([ratings, negative_ratings])
            
            training_data.update({
                'users': all_users,
                'items': all_items,
                'ratings': all_ratings,
                'negative_samples': len(negative_users)
            })
            
            self.logger.info(f"Added {len(negative_users)} negative samples")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Negative sampling failed: {e}")
            raise
    
    def transform_user_id(self, user_id: Any) -> int:
        """Transform external user ID to internal encoding"""
        try:
            return self.user_encoder.transform([user_id])[0]
        except ValueError:
            return -1  # Unknown user
    
    def transform_item_id(self, item_id: Any) -> int:
        """Transform external item ID to internal encoding"""
        try:
            return self.item_encoder.transform([item_id])[0]
        except ValueError:
            return -1  # Unknown item
    
    def inverse_transform_user_id(self, user_encoded: int) -> Any:
        """Transform internal encoding back to external user ID"""
        return self.user_encoder.inverse_transform([user_encoded])[0]
    
    def inverse_transform_item_id(self, item_encoded: int) -> Any:
        """Transform internal encoding back to external item ID"""
        return self.item_encoder.inverse_transform([item_encoded])[0]
    
    def save_preprocessor(self, path: str):
        """Save preprocessor state"""
        try:
            preprocessor_data = {
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'rating_scaler': self.rating_scaler,
                'user_stats': self.user_stats,
                'item_stats': self.item_stats,
                'is_fitted': self.is_fitted
            }
            
            with open(path, 'wb') as f:
                pickle.dump(preprocessor_data, f)
                
            self.logger.info(f"Preprocessor saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save preprocessor: {e}")
            raise
    
    def load_preprocessor(self, path: str):
        """Load preprocessor state"""
        try:
            with open(path, 'rb') as f:
                preprocessor_data = pickle.load(f)
                
            self.user_encoder = preprocessor_data['user_encoder']
            self.item_encoder = preprocessor_data['item_encoder']
            self.rating_scaler = preprocessor_data['rating_scaler']
            self.user_stats = preprocessor_data['user_stats']
            self.item_stats = preprocessor_data['item_stats']
            self.is_fitted = preprocessor_data['is_fitted']
            
            self.logger.info(f"Preprocessor loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load preprocessor: {e}")
            raise
            
            
class ModelEvaluator:
    """Comprehensive model evaluation metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def evaluate_model(self, 
                      model,
                      test_data: Dict[str, Any],
                      user_item_matrix: np.ndarray,
                      k_values: List[int] = [5, 10, 20]) -> EvaluationMetrics:
        """Comprehensive model evaluation"""
        try:
            # Prediction accuracy metrics
            rmse = self._calculate_rmse(model, test_data)
            mae = self._calculate_mae(model, test_data)
            
            # Ranking metrics
            precision_at_k = self._calculate_precision_at_k(model, test_data, user_item_matrix, k_values)
            recall_at_k = self._calculate_recall_at_k(model, test_data, user_item_matrix, k_values)
            ndcg_at_k = self._calculate_ndcg_at_k(model, test_data, user_item_matrix, k_values)
            
            # Diversity and coverage metrics
            coverage = self._calculate_coverage(model, test_data, user_item_matrix)
            diversity = self._calculate_diversity(model, test_data, user_item_matrix)
            novelty = self._calculate_novelty(model, test_data, user_item_matrix)
            
            metrics = EvaluationMetrics(
                rmse=rmse,
                mae=mae,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                ndcg_at_k=ndcg_at_k,
                coverage=coverage,
                diversity=diversity,
                novelty=novelty
            )
            
            self.logger.info(f"Evaluation completed - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _calculate_rmse(self, model, test_data: Dict[str, Any]) -> float:
        """Calculate Root Mean Square Error"""
        predictions = model.predict_batch(test_data['users'], test_data['items'])
        return np.sqrt(mean_squared_error(test_data['ratings'], predictions))
    
    def _calculate_mae(self, model, test_data: Dict[str, Any]) -> float:
        """Calculate Mean Absolute Error"""
        predictions = model.predict_batch(test_data['users'], test_data['items'])
        return mean_absolute_error(test_data['ratings'], predictions)
    
    def _calculate_precision_at_k(self, model, test_data: Dict[str, Any], 
                                 user_item_matrix: np.ndarray, k_values: List[int]) -> Dict[int, float]:
        """Calculate Precision@K for each K"""
        precision_scores = {}
        
        for k in k_values:
            total_precision = 0
            num_users = 0
            
            # For each user in test set
            unique_users = np.unique(test_data['users'])
            
            for user_id in unique_users:
                # Get user's test items
                user_test_items = test_data['items'][test_data['users'] == user_id]
                
                if len(user_test_items) == 0:
                    continue
                    
                # Get top-k recommendations
                try:
                    recommendations = model.recommend(user_id, num_recommendations=k, exclude_seen=True)
                    recommended_items = [rec.item_id for rec in recommendations]
                    
                    # Calculate precision
                    relevant_items = set(user_test_items)
                    recommended_relevant = len(set(recommended_items) & relevant_items)
                    
                    precision = recommended_relevant / min(k, len(recommended_items)) if recommended_items else 0
                    total_precision += precision
                    num_users += 1
                    
                except Exception as e:
                    self.logger.warning(f"Precision calculation failed for user {user_id}: {e}")
                    continue
            
            precision_scores[k] = total_precision / num_users if num_users > 0 else 0
        
        return precision_scores
    
    def _calculate_recall_at_k(self, model, test_data: Dict[str, Any], 
                              user_item_matrix: np.ndarray, k_values: List[int]) -> Dict[int, float]:
        """Calculate Recall@K for each K"""
        recall_scores = {}
        
        for k in k_values:
            total_recall = 0
            num_users = 0
            
            # For each user in test set
            unique_users = np.unique(test_data['users'])
            
            for user_id in unique_users:
                # Get user's test items
                user_test_items = test_data['items'][test_data['users'] == user_id]
                
                if len(user_test_items) == 0:
                    continue
                    
                # Get top-k recommendations
                try:
                    recommendations = model.recommend(user_id, num_recommendations=k, exclude_seen=True)
                    recommended_items = [rec.item_id for rec in recommendations]
                    
                    # Calculate recall
                    relevant_items = set(user_test_items)
                    recommended_relevant = len(set(recommended_items) & relevant_items)
                    
                    recall = recommended_relevant / len(relevant_items) if relevant_items else 0
                    total_recall += recall
                    num_users += 1
                    
                except Exception as e:
                    self.logger.warning(f"Recall calculation failed for user {user_id}: {e}")
                    continue
            
            recall_scores[k] = total_recall / num_users if num_users > 0 else 0
        
        return recall_scores
    
    def _calculate_ndcg_at_k(self, model, test_data: Dict[str, Any], 
                            user_item_matrix: np.ndarray, k_values: List[int]) -> Dict[int, float]:
        """Calculate NDCG@K for each K"""
        ndcg_scores = {}
        
        for k in k_values:
            total_ndcg = 0
            num_users = 0
            
            # For each user in test set
            unique_users = np.unique(test_data['users'])
            
            for user_id in unique_users:
                # Get user's test items with ratings
                user_mask = test_data['users'] == user_id
                user_test_items = test_data['items'][user_mask]
                user_test_ratings = test_data['ratings'][user_mask]
                
                if len(user_test_items) == 0:
                    continue
                    
                # Get top-k recommendations
                try:
                    recommendations = model.recommend(user_id, num_recommendations=k, exclude_seen=True)
                    
                    # Calculate NDCG
                    ndcg = self._calculate_ndcg(recommendations, user_test_items, user_test_ratings, k)
                    total_ndcg += ndcg
                    num_users += 1
                    
                except Exception as e:
                    self.logger.warning(f"NDCG calculation failed for user {user_id}: {e}")
                    continue
            
            ndcg_scores[k] = total_ndcg / num_users if num_users > 0 else 0
        
        return ndcg_scores
    
    def _calculate_ndcg(self, recommendations: List[RecommendationResult], 
                       true_items: np.ndarray, true_ratings: np.ndarray, k: int) -> float:
        """Calculate NDCG for a single user"""
        if not recommendations:
            return 0.0
            
        # Create mapping from item to rating
        item_to_rating = dict(zip(true_items, true_ratings))
        
        # Calculate DCG
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k]):
            relevance = item_to_rating.get(rec.item_id, 0.0)
            dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        sorted_ratings = sorted(true_ratings, reverse=True)
        idcg = 0.0
        for i, rating in enumerate(sorted_ratings[:k]):
            idcg += rating / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_coverage(self, model, test_data: Dict[str, Any], 
                          user_item_matrix: np.ndarray) -> float:
        """Calculate catalog coverage"""
        try:
            recommended_items = set()
            unique_users = np.unique(test_data['users'])
            
            for user_id in unique_users[:min(100, len(unique_users))]:
                try:
                    recommendations = model.recommend(user_id, num_recommendations=20, exclude_seen=True)
                    recommended_items.update([rec.item_id for rec in recommendations])
                except Exception:
                    continue
            
            total_items = user_item_matrix.shape[1]
            coverage = len(recommended_items) / total_items if total_items > 0 else 0
            
            return coverage
            
        except Exception as e:
            self.logger.warning(f"Coverage calculation failed: {e}")
            return 0.0
    
    def _calculate_diversity(self, model, test_data: Dict[str, Any], 
                           user_item_matrix: np.ndarray) -> float:
        """Calculate intra-list diversity"""
        try:
            total_diversity = 0
            num_users = 0
            unique_users = np.unique(test_data['users'])
            
            for user_id in unique_users[:min(50, len(unique_users))]:
                try:
                    recommendations = model.recommend(user_id, num_recommendations=10, exclude_seen=True)
                    
                    if len(recommendations) < 2:
                        continue
                        
                    # Calculate pairwise diversity based on item embeddings
                    item_embeddings = model.get_item_embeddings()
                    
                    diversity_sum = 0
                    pairs = 0
                    
                    for i in range(len(recommendations)):
                        for j in range(i + 1, len(recommendations)):
                            item1_id = recommendations[i].item_id
                            item2_id = recommendations[j].item_id
                            
                            if item1_id < len(item_embeddings) and item2_id < len(item_embeddings):
                                # Cosine distance
                                emb1 = item_embeddings[item1_id]
                                emb2 = item_embeddings[item2_id]
                                
                                cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                                diversity_sum += 1 - cosine_sim
                                pairs += 1
                    
                    if pairs > 0:
                        user_diversity = diversity_sum / pairs
                        total_diversity += user_diversity
                        num_users += 1
                        
                except Exception:
                    continue
            
            return total_diversity / num_users if num_users > 0 else 0
            
        except Exception as e:
            self.logger.warning(f"Diversity calculation failed: {e}")
            return 0.0
    
    def _calculate_novelty(self, model, test_data: Dict[str, Any], 
                          user_item_matrix: np.ndarray) -> float:
        """Calculate novelty score"""
        try:
            # Calculate item popularity
            item_popularity = np.sum(user_item_matrix > 0, axis=0)
            total_users = user_item_matrix.shape[0]
            
            total_novelty = 0
            num_recommendations = 0
            unique_users = np.unique(test_data['users'])
            
            for user_id in unique_users[:min(50, len(unique_users))]:
                try:
                    recommendations = model.recommend(user_id, num_recommendations=10, exclude_seen=True)
                    
                    for rec in recommendations:
                        if rec.item_id < len(item_popularity):
                            popularity = item_popularity[rec.item_id] / total_users
                            novelty = -np.log2(popularity) if popularity > 0 else 0
                            total_novelty += novelty
                            num_recommendations += 1
                            
                except Exception:
                    continue
            
            return total_novelty / num_recommendations if num_recommendations > 0 else 0
            
        except Exception as e:
            self.logger.warning(f"Novelty calculation failed: {e}")
            return 0.0
            
            
class ColdStartHandler:
    """Handler for cold start problems"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.user_features = None
        self.item_features = None
        self.content_model = None
        self.popularity_model = None
        
    def handle_cold_user(self, user_features: Dict[str, Any], 
                        num_recommendations: int = 10) -> List[RecommendationResult]:
        """Handle recommendations for new users"""
        try:
            # Use popularity-based recommendations for cold users
            if self.popularity_model:
                return self.popularity_model.recommend(num_recommendations)
            
            # Fallback to random recommendations
            recommendations = []
            for i in range(num_recommendations):
                result = RecommendationResult(
                    item_id=i,
                    predicted_rating=0.5,
                    confidence_score=0.1,
                    explanation="Popular item recommendation for new user"
                )
                recommendations.append(result)
                
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Cold user handling failed: {e}")
            return []
    
    def handle_cold_item(self, item_features: Dict[str, Any], 
                        user_id: int) -> float:
        """Handle rating prediction for new items"""
        try:
            # Use content-based features if available
            if self.content_model and item_features:
                return self.content_model.predict_rating(user_id, item_features)
            
            # Fallback to global average
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Cold item handling failed: {e}")
            return 0.5
    
    def set_popularity_model(self, user_item_matrix: np.ndarray):
        """Set up popularity-based model for cold start"""
        item_popularity = np.sum(user_item_matrix > 0, axis=0)
        item_avg_rating = np.mean(user_item_matrix, axis=0)
        
        # Combine popularity and average rating
        self.popularity_scores = item_popularity * item_avg_rating
        self.popular_items = np.argsort(self.popularity_scores)[::-1]


class BaseRecommender(ABC):
    """Base class for recommendation models"""
    
    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame = None, user_item_matrix: np.ndarray = None, **kwargs):
        """Train the recommendation model"""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Predict ratings for user-item pairs"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, num_recommendations: int = 10, **kwargs) -> List[RecommendationResult]:
        """Get top-N recommendations for a user"""
        pass


class CollaborativeFilteringModel(BaseRecommender):
    """Neural Collaborative Filtering using TensorFlow"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50, 
                 reg_lambda: float = 1e-6):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.reg_lambda = reg_lambda
        self.model = None
        self.is_trained = False
        self.user_item_matrix = None
        self.logger = logging.getLogger(__name__)
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build neural collaborative filtering model"""
        try:
            # Input layers
            user_input = tf.keras.layers.Input(shape=(), name='user_id')
            item_input = tf.keras.layers.Input(shape=(), name='item_id')
            
            # Embedding layers
            user_embedding = tf.keras.layers.Embedding(
                self.num_users, 
                self.embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='user_embedding'
            )(user_input)
            
            item_embedding = tf.keras.layers.Embedding(
                self.num_items, 
                self.embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='item_embedding'
            )(item_input)
            
            # Flatten embeddings
            user_vec = tf.keras.layers.Flatten(name='user_flatten')(user_embedding)
            item_vec = tf.keras.layers.Flatten(name='item_flatten')(item_embedding)
            
            # Neural MF layers
            concat = tf.keras.layers.Concatenate(name='concat')([user_vec, item_vec])
            
            # Deep layers
            hidden = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(concat)
            hidden = tf.keras.layers.Dropout(0.2)(hidden)
            hidden = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(hidden)
            hidden = tf.keras.layers.Dropout(0.2)(hidden)
            hidden = tf.keras.layers.Dense(32, activation='relu', name='dense_3')(hidden)
            
            # Output layer
            output = tf.keras.layers.Dense(1, activation='sigmoid', name='rating_output')(hidden)
            
            # Create model
            model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['mae', 'mse']
            )
            
            self.logger.info("Successfully built collaborative filtering model")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to build collaborative filtering model: {e}")
            raise
    
    def fit(self, user_item_matrix: np.ndarray, epochs: int = 100, 
            batch_size: int = 256, validation_split: float = 0.2, **kwargs):
        """Train the collaborative filtering model"""
        try:
            self.user_item_matrix = user_item_matrix
            
            # Prepare training data
            users, items, ratings = [], [], []
            
            for user_id in range(user_item_matrix.shape[0]):
                for item_id in range(user_item_matrix.shape[1]):
                    rating = user_item_matrix[user_id, item_id]
                    if rating > 0:  # Only include observed ratings
                        users.append(user_id)
                        items.append(item_id)
                        ratings.append(rating)
            
            # Add negative samples (implicit feedback)
            negative_samples = len(ratings) // 2  # 50% negative samples
            for _ in range(negative_samples):
                user_id = np.random.randint(0, self.num_users)
                item_id = np.random.randint(0, self.num_items)
                if user_item_matrix[user_id, item_id] == 0:  # Unobserved interaction
                    users.append(user_id)
                    items.append(item_id)
                    ratings.append(0.0)  # Negative sample
            
            # Convert to numpy arrays
            users = np.array(users)
            items = np.array(items)
            ratings = np.array(ratings, dtype=np.float32)
            
            # Train model
            history = self.model.fit(
                [users, items], 
                ratings,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
                shuffle=True
            )
            
            self.is_trained = True
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_mae = history.history['mae'][-1]
            self.logger.info(f"Training completed - Loss: {final_loss:.4f}, MAE: {final_mae:.4f}")
            
            return {
                'final_loss': float(final_loss),
                'final_mae': float(final_mae),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(users)
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Predict ratings for user-item pairs"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if user_id >= self.num_users or user_id < 0:
                raise ValueError(f"User ID {user_id} is out of range")
            
            # Validate item IDs
            valid_item_ids = [item_id for item_id in item_ids if 0 <= item_id < self.num_items]
            if len(valid_item_ids) != len(item_ids):
                self.logger.warning(f"Some item IDs are out of range. Using {len(valid_item_ids)}/{len(item_ids)} items")
            
            if not valid_item_ids:
                return np.array([])
            
            # Prepare input
            user_ids = np.array([user_id] * len(valid_item_ids))
            item_ids_array = np.array(valid_item_ids)
            
            # Make predictions
            predictions = self.model.predict([user_ids, item_ids_array], verbose=0)
            
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.array([])
    
    def recommend(self, user_id: int, num_recommendations: int = 10, 
                  exclude_seen: bool = True) -> List[RecommendationResult]:
        """Get top-N recommendations for a user"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making recommendations")
            
            if user_id >= self.num_users or user_id < 0:
                raise ValueError(f"User ID {user_id} is out of range")
            
            # Get all item IDs
            all_item_ids = list(range(self.num_items))
            
            # Exclude items the user has already interacted with
            if exclude_seen and self.user_item_matrix is not None:
                seen_items = set(np.where(self.user_item_matrix[user_id] > 0)[0])
                all_item_ids = [item_id for item_id in all_item_ids if item_id not in seen_items]
            
            if not all_item_ids:
                return []
            
            # Predict ratings for all candidate items
            predictions = self.predict(user_id, all_item_ids)
            
            if len(predictions) == 0:
                return []
            
            # Create recommendation results
            recommendations = []
            for item_id, prediction in zip(all_item_ids, predictions):
                # Calculate confidence based on prediction value and user history
                confidence = self._calculate_confidence(user_id, item_id, prediction)
                
                # Generate explanation
                explanation = self._generate_explanation(user_id, item_id, prediction)
                
                result = RecommendationResult(
                    item_id=item_id,
                    predicted_rating=float(prediction),
                    confidence_score=confidence,
                    explanation=explanation
                )
                recommendations.append(result)
            
            # Sort by predicted rating (descending) and return top-N
            recommendations.sort(key=lambda x: x.predicted_rating, reverse=True)
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            self.logger.error(f"Recommendation failed: {e}")
            return []
    
    def _calculate_confidence(self, user_id: int, item_id: int, prediction: float) -> float:
        """Calculate confidence score for a recommendation"""
        try:
            # Base confidence from prediction value
            base_confidence = prediction
            
            # Adjust based on user activity (more active users -> higher confidence)
            if self.user_item_matrix is not None:
                user_activity = np.sum(self.user_item_matrix[user_id] > 0)
                activity_boost = min(user_activity / 10.0, 1.0)  # Max boost of 1.0
                
                # Adjust based on item popularity
                item_popularity = np.sum(self.user_item_matrix[:, item_id] > 0)
                popularity_boost = min(item_popularity / 50.0, 0.5)  # Max boost of 0.5
                
                confidence = base_confidence * (1.0 + activity_boost + popularity_boost)
            else:
                confidence = base_confidence
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return float(prediction)
    
    def _generate_explanation(self, user_id: int, item_id: int, prediction: float) -> str:
        """Generate explanation for a recommendation"""
        try:
            if prediction > 0.8:
                return "Highly recommended based on similar user preferences"
            elif prediction > 0.6:
                return "Recommended based on collaborative filtering patterns"
            elif prediction > 0.4:
                return "Moderately recommended by similar users"
            else:
                return "Suggested based on general user patterns"
                
        except Exception as e:
            self.logger.warning(f"Explanation generation failed: {e}")
            return "Recommended by collaborative filtering"
    
    def get_user_embeddings(self) -> np.ndarray:
        """Get learned user embeddings"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get embeddings")
        
        # Extract user embeddings from the model
        user_embedding_layer = self.model.get_layer('user_embedding')
        return user_embedding_layer.get_weights()[0]
    
    def get_item_embeddings(self) -> np.ndarray:
        """Get learned item embeddings"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get embeddings")
        
        # Extract item embeddings from the model
        item_embedding_layer = self.model.get_layer('item_embedding')
        return item_embedding_layer.get_weights()[0]
    
    def get_similar_users(self, user_id: int, num_similar: int = 10) -> List[Tuple[int, float]]:
        """Find similar users based on embedding similarity"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained to find similar users")
            
            user_embeddings = self.get_user_embeddings()
            target_user_embedding = user_embeddings[user_id]
            
            # Calculate cosine similarity with all users
            similarities = np.dot(user_embeddings, target_user_embedding) / (
                np.linalg.norm(user_embeddings, axis=1) * np.linalg.norm(target_user_embedding)
            )
            
            # Get top similar users (excluding the target user)
            similar_user_indices = np.argsort(similarities)[::-1]
            similar_users = []
            
            for idx in similar_user_indices:
                if idx != user_id and len(similar_users) < num_similar:
                    similar_users.append((int(idx), float(similarities[idx])))
            
            return similar_users
            
        except Exception as e:
            self.logger.error(f"Finding similar users failed: {e}")
            return []
    
    def get_similar_items(self, item_id: int, num_similar: int = 10) -> List[Tuple[int, float]]:
        """Find similar items based on embedding similarity"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained to find similar items")
            
            item_embeddings = self.get_item_embeddings()
            target_item_embedding = item_embeddings[item_id]
            
            # Calculate cosine similarity with all items
            similarities = np.dot(item_embeddings, target_item_embedding) / (
                np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(target_item_embedding)
            )
            
            # Get top similar items (excluding the target item)
            similar_item_indices = np.argsort(similarities)[::-1]
            similar_items = []
            
            for idx in similar_item_indices:
                if idx != item_id and len(similar_items) < num_similar:
                    similar_items.append((int(idx), float(similarities[idx])))
            
            return similar_items
            
        except Exception as e:
            self.logger.error(f"Finding similar items failed: {e}")
            return []
    
    def monitor_performance(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor model performance on test data"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before monitoring performance")
            
            # Calculate metrics
            metrics = self.evaluator.evaluate_model(self, test_data, self.user_item_matrix)
            
            # System metrics
            import psutil
            import time
            
            # Measure prediction latency
            start_time = time.time()
            sample_users = test_data['users'][:100]
            sample_items = test_data['items'][:100]
            self.predict_batch(sample_users, sample_items)
            latency = (time.time() - start_time) / 100  # Per prediction
            
            performance_info = {
                'evaluation_metrics': metrics,
                'system_metrics': {
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent(),
                    'prediction_latency_ms': latency * 1000
                },
                'model_health': {
                    'is_trained': self.is_trained,
                    'has_preprocessor': self.preprocessor.is_fitted,
                    'model_size_mb': self.model.count_params() * 4 / 1024 / 1024,  # Rough estimate
                    'training_epochs': len(self.training_history.get('loss', [])) if self.training_history else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Performance monitoring completed - RMSE: {metrics.rmse:.4f}, Latency: {latency*1000:.2f}ms")
            return performance_info
            
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            return {}
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance in embeddings"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before analyzing feature importance")
            
            # Get embeddings
            user_embeddings = self.get_user_embeddings()
            item_embeddings = self.get_item_embeddings()
            
            # Calculate embedding statistics
            user_stats = {
                'mean_norm': float(np.mean(np.linalg.norm(user_embeddings, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(user_embeddings, axis=1))),
                'dimension_variance': np.var(user_embeddings, axis=0).tolist()
            }
            
            item_stats = {
                'mean_norm': float(np.mean(np.linalg.norm(item_embeddings, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(item_embeddings, axis=1))),
                'dimension_variance': np.var(item_embeddings, axis=0).tolist()
            }
            
            # Most important embedding dimensions
            user_importance = np.argsort(user_stats['dimension_variance'])[::-1][:10]
            item_importance = np.argsort(item_stats['dimension_variance'])[::-1][:10]
            
            return {
                'user_embedding_stats': user_stats,
                'item_embedding_stats': item_stats,
                'most_important_user_dims': user_importance.tolist(),
                'most_important_item_dims': item_importance.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            return {}
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """Provide detailed explanation for a specific recommendation"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before explaining recommendations")
            
            # Get prediction
            prediction = self.predict(user_id, [item_id])[0]
            
            # Get user and item embeddings
            user_embedding = self.get_user_embeddings()[user_id]
            item_embedding = self.get_item_embeddings()[item_id]
            
            # Calculate similarity with user's interacted items
            user_interactions = np.where(self.user_item_matrix[user_id] > 0)[0]
            similar_items = []
            
            for interacted_item in user_interactions[:5]:  # Top 5 interactions
                interacted_embedding = self.get_item_embeddings()[interacted_item]
                similarity = np.dot(item_embedding, interacted_embedding) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(interacted_embedding)
                )
                similar_items.append({
                    'item_id': int(interacted_item),
                    'similarity': float(similarity),
                    'user_rating': float(self.user_item_matrix[user_id, interacted_item])
                })
            
            # Sort by similarity
            similar_items.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Get similar users
            similar_users = self.get_similar_users(user_id, num_similar=3)
            
            explanation = {
                'user_id': user_id,
                'item_id': item_id,
                'predicted_rating': float(prediction),
                'confidence': self._calculate_confidence(user_id, item_id, prediction),
                'similar_items': similar_items,
                'similar_users': [
                    {'user_id': uid, 'similarity': sim} for uid, sim in similar_users
                ],
                'user_embedding_norm': float(np.linalg.norm(user_embedding)),
                'item_embedding_norm': float(np.linalg.norm(item_embedding)),
                'user_item_dot_product': float(np.dot(user_embedding, item_embedding))
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {}
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            self.model.save(model_path)
            self.logger.info(f"Collaborative filtering model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.is_trained = True
            self.logger.info(f"Collaborative filtering model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'collaborative_filtering',
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
            'is_trained': self.is_trained,
            'regularization': self.reg_lambda
        }