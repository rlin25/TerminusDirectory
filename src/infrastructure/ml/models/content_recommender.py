import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from .collaborative_filter import BaseRecommender, RecommendationResult


@dataclass
class PropertyFeatures:
    """Structured container for property features"""
    location_features: np.ndarray
    price_features: np.ndarray
    bedroom_features: np.ndarray
    bathroom_features: np.ndarray
    amenity_features: np.ndarray
    combined_features: np.ndarray
    feature_names: List[str]


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommendation system using TensorFlow for rental properties.
    
    This recommender builds user preferences from property features including:
    - Location (neighborhood, city, region)
    - Price range and price per square foot
    - Property specifications (bedrooms, bathrooms, square footage)
    - Amenities and features
    - Property type and style
    
    The model uses deep neural networks to learn complex feature interactions
    and generate personalized recommendations based on content similarity.
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 location_vocab_size: int = 1000,
                 amenity_vocab_size: int = 500,
                 reg_lambda: float = 1e-5,
                 learning_rate: float = 0.001):
        """
        Initialize the ContentBasedRecommender.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            location_vocab_size: Maximum number of unique locations
            amenity_vocab_size: Maximum number of unique amenities
            reg_lambda: L2 regularization strength
            learning_rate: Learning rate for optimizer
        """
        self.embedding_dim = embedding_dim
        self.location_vocab_size = location_vocab_size
        self.amenity_vocab_size = amenity_vocab_size
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        
        # Model components
        self.model = None
        self.feature_processor = None
        self.location_encoder = None
        self.amenity_vectorizer = None
        self.price_scaler = None
        
        # Training data storage
        self.property_features = None
        self.user_item_matrix = None
        self.property_similarity_matrix = None
        
        # State tracking
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature processors
        self._initialize_feature_processors()
        
        # Build the neural network model
        self.model = self._build_model()
    
    def _initialize_feature_processors(self):
        """Initialize feature preprocessing components"""
        try:
            # Location encoder for categorical location features
            self.location_encoder = LabelEncoder()
            
            # TF-IDF vectorizer for amenities text processing
            self.amenity_vectorizer = TfidfVectorizer(
                max_features=self.amenity_vocab_size,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                lowercase=True
            )
            
            # Scaler for numerical features (price, bedrooms, bathrooms)
            self.price_scaler = StandardScaler()
            
            self.logger.info("Feature processors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature processors: {e}")
            raise
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the content-based recommendation neural network.
        
        The model processes different types of property features:
        1. Location embeddings
        2. Numerical features (price, bedrooms, bathrooms)
        3. Amenity text features
        4. Combined feature interactions
        
        Returns:
            Compiled TensorFlow model
        """
        try:
            # Input layers for different feature types
            location_input = tf.keras.layers.Input(shape=(), name='location_input')
            price_input = tf.keras.layers.Input(shape=(3,), name='price_input')  # price, bedrooms, bathrooms
            amenity_input = tf.keras.layers.Input(shape=(self.amenity_vocab_size,), name='amenity_input')
            
            # Location embedding layer
            location_embedding = tf.keras.layers.Embedding(
                input_dim=self.location_vocab_size,
                output_dim=self.embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='location_embedding'
            )(location_input)
            location_flat = tf.keras.layers.Flatten(name='location_flatten')(location_embedding)
            
            # Price and property spec processing
            price_dense = tf.keras.layers.Dense(
                64, 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='price_dense'
            )(price_input)
            price_dropout = tf.keras.layers.Dropout(0.2)(price_dense)
            
            # Amenity features processing
            amenity_dense = tf.keras.layers.Dense(
                64, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='amenity_dense'
            )(amenity_input)
            amenity_dropout = tf.keras.layers.Dropout(0.2)(amenity_dense)
            
            # Combine all features
            combined = tf.keras.layers.Concatenate(name='feature_concat')([
                location_flat,
                price_dropout,
                amenity_dropout
            ])
            
            # Deep neural network layers for feature interaction learning
            hidden1 = tf.keras.layers.Dense(
                256, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='hidden_1'
            )(combined)
            hidden1_dropout = tf.keras.layers.Dropout(0.3)(hidden1)
            
            hidden2 = tf.keras.layers.Dense(
                128, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='hidden_2'
            )(hidden1_dropout)
            hidden2_dropout = tf.keras.layers.Dropout(0.3)(hidden2)
            
            hidden3 = tf.keras.layers.Dense(
                64, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='hidden_3'
            )(hidden2_dropout)
            hidden3_dropout = tf.keras.layers.Dropout(0.2)(hidden3)
            
            # Output layer for preference prediction
            output = tf.keras.layers.Dense(
                1, 
                activation='sigmoid', 
                name='preference_output'
            )(hidden3_dropout)
            
            # Create and compile model
            model = tf.keras.Model(
                inputs=[location_input, price_input, amenity_input],
                outputs=output,
                name='content_based_recommender'
            )
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall', 'mae']
            )
            
            self.logger.info("Content-based recommendation model built successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to build content-based model: {e}")
            raise
    
    def extract_property_features(self, properties: List[Dict]) -> PropertyFeatures:
        """
        Extract and process features from property data.
        
        Args:
            properties: List of property dictionaries with features
            
        Returns:
            PropertyFeatures object containing processed features
        """
        try:
            if not properties:
                raise ValueError("Properties list cannot be empty")
            
            # Extract location features
            locations = []
            for prop in properties:
                location_parts = []
                if prop.get('neighborhood'):
                    location_parts.append(prop['neighborhood'])
                if prop.get('city'):
                    location_parts.append(prop['city'])
                if prop.get('state'):
                    location_parts.append(prop['state'])
                
                location_str = ' '.join(location_parts) if location_parts else 'unknown'
                locations.append(location_str)
            
            # Process location features
            if not hasattr(self.location_encoder, 'classes_'):
                # Fit the encoder on first use
                location_encoded = self.location_encoder.fit_transform(locations)
            else:
                # Handle new locations not seen during training
                location_encoded = []
                for location in locations:
                    try:
                        encoded = self.location_encoder.transform([location])[0]
                        location_encoded.append(encoded)
                    except ValueError:
                        # Unknown location, use a default value
                        location_encoded.append(0)
                location_encoded = np.array(location_encoded)
            
            # Extract price and property specification features
            price_features = []
            for prop in properties:
                price = float(prop.get('price', 0))
                bedrooms = float(prop.get('bedrooms', 0))
                bathrooms = float(prop.get('bathrooms', 0))
                price_features.append([price, bedrooms, bathrooms])
            
            price_features = np.array(price_features)
            
            # Scale price features
            if not hasattr(self.price_scaler, 'mean_'):
                price_features_scaled = self.price_scaler.fit_transform(price_features)
            else:
                price_features_scaled = self.price_scaler.transform(price_features)
            
            # Extract amenity features
            amenity_texts = []
            for prop in properties:
                amenities = prop.get('amenities', [])
                if isinstance(amenities, list):
                    amenity_text = ' '.join(amenities)
                else:
                    amenity_text = str(amenities)
                
                # Add property type and other textual features
                if prop.get('property_type'):
                    amenity_text += f" {prop['property_type']}"
                if prop.get('parking'):
                    amenity_text += " parking"
                if prop.get('pet_friendly'):
                    amenity_text += " pet_friendly"
                
                amenity_texts.append(amenity_text)
            
            # Process amenity features
            if not hasattr(self.amenity_vectorizer, 'vocabulary_'):
                amenity_features = self.amenity_vectorizer.fit_transform(amenity_texts)
            else:
                amenity_features = self.amenity_vectorizer.transform(amenity_texts)
            
            amenity_features = amenity_features.toarray()
            
            # Create bedroom and bathroom specific features
            bedroom_features = price_features[:, 1:2]  # Extract bedrooms
            bathroom_features = price_features[:, 2:3]  # Extract bathrooms
            
            # Combine all features for similarity computation
            combined_features = np.concatenate([
                location_encoded.reshape(-1, 1),
                price_features_scaled,
                amenity_features
            ], axis=1)
            
            # Generate feature names for interpretability
            feature_names = (['location'] + 
                           ['price', 'bedrooms', 'bathrooms'] + 
                           [f'amenity_{i}' for i in range(amenity_features.shape[1])])
            
            return PropertyFeatures(
                location_features=location_encoded,
                price_features=price_features_scaled,
                bedroom_features=bedroom_features,
                bathroom_features=bathroom_features,
                amenity_features=amenity_features,
                combined_features=combined_features,
                feature_names=feature_names
            )
            
        except Exception as e:
            self.logger.error(f"Property feature extraction failed: {e}")
            raise
    
    def fit(self, user_item_matrix: np.ndarray, property_data: List[Dict], 
            epochs: int = 100, batch_size: int = 128, validation_split: float = 0.2, **kwargs):
        """
        Train the content-based recommendation model.
        
        Args:
            user_item_matrix: User-item interaction matrix
            property_data: List of property dictionaries with features
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            **kwargs: Additional training parameters
        """
        try:
            self.user_item_matrix = user_item_matrix
            
            # Extract property features
            self.property_features = self.extract_property_features(property_data)
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(user_item_matrix)
            
            if len(X_train[0]) == 0:
                raise ValueError("No training data generated")
            
            # Train the model
            self.logger.info(f"Starting training with {len(y_train)} samples")
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=1e-6
                )
            ]
            
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            # Compute property similarity matrix for recommendation explanations
            self.property_similarity_matrix = cosine_similarity(
                self.property_features.combined_features
            )
            
            self.is_trained = True
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            final_mae = history.history['mae'][-1]
            
            self.logger.info(
                f"Training completed - Loss: {final_loss:.4f}, "
                f"Accuracy: {final_accuracy:.4f}, MAE: {final_mae:.4f}"
            )
            
            return {
                'final_loss': float(final_loss),
                'final_accuracy': float(final_accuracy),
                'final_mae': float(final_mae),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(y_train),
                'num_properties': len(property_data)
            }
            
        except Exception as e:
            self.logger.error(f"Content-based model training failed: {e}")
            raise
    
    def _prepare_training_data(self, user_item_matrix: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Prepare training data from user-item interactions and property features.
        
        Args:
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Tuple of (input_features, target_labels)
        """
        try:
            location_inputs = []
            price_inputs = []
            amenity_inputs = []
            labels = []
            
            num_users, num_items = user_item_matrix.shape
            
            # Generate positive samples from existing interactions
            for user_id in range(num_users):
                for item_id in range(num_items):
                    rating = user_item_matrix[user_id, item_id]
                    
                    if rating > 0:  # Positive interaction
                        location_inputs.append(self.property_features.location_features[item_id])
                        price_inputs.append(self.property_features.price_features[item_id])
                        amenity_inputs.append(self.property_features.amenity_features[item_id])
                        labels.append(1.0)
            
            # Generate negative samples from unobserved interactions
            num_positive = len(labels)
            num_negative = min(num_positive, num_users * num_items // 20)  # Limit negative samples
            
            for _ in range(num_negative):
                user_id = np.random.randint(0, num_users)
                item_id = np.random.randint(0, num_items)
                
                if user_item_matrix[user_id, item_id] == 0:  # Unobserved interaction
                    location_inputs.append(self.property_features.location_features[item_id])
                    price_inputs.append(self.property_features.price_features[item_id])
                    amenity_inputs.append(self.property_features.amenity_features[item_id])
                    labels.append(0.0)
            
            # Convert to numpy arrays
            X_train = [
                np.array(location_inputs),
                np.array(price_inputs),
                np.array(amenity_inputs)
            ]
            y_train = np.array(labels)
            
            self.logger.info(f"Prepared training data: {len(y_train)} samples "
                           f"({num_positive} positive, {len(y_train) - num_positive} negative)")
            
            return X_train, y_train
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            raise
    
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        Predict user preferences for given property items.
        
        Args:
            user_id: User identifier
            item_ids: List of property item IDs
            
        Returns:
            Array of predicted preference scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if self.property_features is None:
                raise ValueError("Property features are not available")
            
            # Validate item IDs
            valid_item_ids = [item_id for item_id in item_ids 
                            if 0 <= item_id < len(self.property_features.location_features)]
            
            if len(valid_item_ids) != len(item_ids):
                self.logger.warning(f"Some item IDs are out of range. "
                                  f"Using {len(valid_item_ids)}/{len(item_ids)} items")
            
            if not valid_item_ids:
                return np.array([])
            
            # Prepare input features
            location_inputs = self.property_features.location_features[valid_item_ids]
            price_inputs = self.property_features.price_features[valid_item_ids]
            amenity_inputs = self.property_features.amenity_features[valid_item_ids]
            
            # Make predictions
            predictions = self.model.predict([
                location_inputs,
                price_inputs,
                amenity_inputs
            ], verbose=0)
            
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.array([])
    
    def recommend(self, user_id: int, num_recommendations: int = 10, 
                  exclude_seen: bool = True) -> List[RecommendationResult]:
        """
        Generate content-based recommendations for a user.
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude previously seen items
            
        Returns:
            List of RecommendationResult objects
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making recommendations")
            
            if self.property_features is None:
                raise ValueError("Property features are not available")
            
            if user_id >= self.user_item_matrix.shape[0] or user_id < 0:
                raise ValueError(f"User ID {user_id} is out of range")
            
            # Get all property IDs
            all_item_ids = list(range(len(self.property_features.location_features)))
            
            # Exclude items the user has already interacted with
            if exclude_seen and self.user_item_matrix is not None:
                seen_items = set(np.where(self.user_item_matrix[user_id] > 0)[0])
                all_item_ids = [item_id for item_id in all_item_ids if item_id not in seen_items]
            
            if not all_item_ids:
                return []
            
            # Predict preferences for all candidate items
            predictions = self.predict(user_id, all_item_ids)
            
            if len(predictions) == 0:
                return []
            
            # Create recommendation results
            recommendations = []
            for item_id, prediction in zip(all_item_ids, predictions):
                # Calculate confidence score
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
            
            # Sort by predicted rating and return top-N
            recommendations.sort(key=lambda x: x.predicted_rating, reverse=True)
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            self.logger.error(f"Content-based recommendation failed: {e}")
            return []
    
    def _calculate_confidence(self, user_id: int, item_id: int, prediction: float) -> float:
        """
        Calculate confidence score for a recommendation.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            prediction: Model prediction score
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Base confidence from prediction value
            base_confidence = prediction
            
            # Boost confidence based on property similarity to user's history
            if self.property_similarity_matrix is not None and self.user_item_matrix is not None:
                # Find properties the user has liked
                user_likes = np.where(self.user_item_matrix[user_id] > 0)[0]
                
                if len(user_likes) > 0:
                    # Calculate average similarity to liked properties
                    similarities = self.property_similarity_matrix[item_id][user_likes]
                    avg_similarity = np.mean(similarities)
                    
                    # Boost confidence based on similarity
                    similarity_boost = min(avg_similarity * 0.5, 0.3)
                    confidence = base_confidence * (1.0 + similarity_boost)
                else:
                    confidence = base_confidence
            else:
                confidence = base_confidence
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return float(prediction)
    
    def _generate_explanation(self, user_id: int, item_id: int, prediction: float) -> str:
        """
        Generate explanation for a content-based recommendation.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            prediction: Model prediction score
            
        Returns:
            Human-readable explanation string
        """
        try:
            if prediction > 0.8:
                return "Strongly matches your property preferences based on location, price, and amenities"
            elif prediction > 0.6:
                return "Good match based on property features and your interaction history"
            elif prediction > 0.4:
                return "Moderately matches your preferences for similar properties"
            else:
                return "Recommended based on property characteristics"
                
        except Exception as e:
            self.logger.warning(f"Explanation generation failed: {e}")
            return "Recommended based on content-based filtering"
    
    def get_property_similarity(self, item_id: int, num_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar properties based on content features.
        
        Args:
            item_id: Property item ID
            num_similar: Number of similar properties to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        try:
            if self.property_similarity_matrix is None:
                raise ValueError("Property similarity matrix is not available")
            
            if item_id >= len(self.property_similarity_matrix):
                raise ValueError(f"Item ID {item_id} is out of range")
            
            # Get similarity scores for the property
            similarities = self.property_similarity_matrix[item_id]
            
            # Get top similar properties (excluding the property itself)
            similar_indices = np.argsort(similarities)[::-1]
            similar_properties = []
            
            for idx in similar_indices:
                if idx != item_id and len(similar_properties) < num_similar:
                    similar_properties.append((int(idx), float(similarities[idx])))
            
            return similar_properties
            
        except Exception as e:
            self.logger.error(f"Finding similar properties failed: {e}")
            return []
    
    def get_feature_importance(self, item_id: int) -> Dict[str, float]:
        """
        Get feature importance for a specific property recommendation.
        
        Args:
            item_id: Property item ID
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained to get feature importance")
            
            if self.property_features is None:
                raise ValueError("Property features are not available")
            
            # Simple feature importance based on feature magnitudes
            features = self.property_features.combined_features[item_id]
            feature_names = self.property_features.feature_names
            
            # Normalize feature values to get relative importance
            feature_importance = {}
            total_magnitude = np.sum(np.abs(features))
            
            for i, (name, value) in enumerate(zip(feature_names, features)):
                importance = abs(value) / total_magnitude if total_magnitude > 0 else 0
                feature_importance[name] = float(importance)
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def save_model(self, model_path: str):
        """Save the trained model and feature processors"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            self.model.save(model_path)
            self.logger.info(f"Content-based recommendation model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.is_trained = True
            self.logger.info(f"Content-based recommendation model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information and configuration"""
        return {
            'model_type': 'content_based_recommender',
            'embedding_dim': self.embedding_dim,
            'location_vocab_size': self.location_vocab_size,
            'amenity_vocab_size': self.amenity_vocab_size,
            'regularization': self.reg_lambda,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained,
            'num_properties': len(self.property_features.location_features) if self.property_features else 0
        }