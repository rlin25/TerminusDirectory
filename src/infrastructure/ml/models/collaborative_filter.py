import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass


@dataclass
class RecommendationResult:
    item_id: int
    predicted_rating: float
    confidence_score: float
    explanation: str


class BaseRecommender(ABC):
    """Base class for recommendation models"""
    
    @abstractmethod
    def fit(self, user_item_matrix: np.ndarray, **kwargs):
        """Train the recommendation model"""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Predict ratings for user-item pairs"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, num_recommendations: int = 10) -> List[RecommendationResult]:
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