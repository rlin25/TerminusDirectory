import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import logging
from dataclasses import dataclass, field
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRanker
import pickle
import json
import time
import threading
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import heapq
import math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


@dataclass
class QueryInfo:
    """Information about a search query"""
    original_query: str
    normalized_query: str
    tokens: List[str]
    intent: str
    query_type: str
    location_entities: List[str]
    price_range: Optional[Tuple[float, float]]
    property_type: Optional[str]
    amenities: List[str]
    expanded_terms: List[str]
    confidence: float

@dataclass
class RankingFeatures:
    """Features used for ranking"""
    text_similarity: float
    semantic_similarity: float
    location_match: float
    price_relevance: float
    amenity_match: float
    popularity_score: float
    recency_score: float
    user_preference_score: float
    query_intent_match: float
    title_match: float
    description_match: float
    exact_match_boost: float
    composite_score: float

@dataclass
class RankingResult:
    property_id: str
    property_data: Dict
    relevance_score: float
    ranking_features: RankingFeatures
    explanation: str
    ranking_position: int
    confidence: float

@dataclass
class UserProfile:
    """User profile for personalized ranking"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    search_history: List[str] = field(default_factory=list)
    interaction_history: List[Dict] = field(default_factory=list)
    demographic_info: Dict = field(default_factory=dict)
    location_preferences: List[str] = field(default_factory=list)
    price_sensitivity: float = 0.5
    amenity_preferences: Dict[str, float] = field(default_factory=dict)
    property_type_preferences: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class RankingMetrics:
    """Ranking evaluation metrics"""
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    mrr: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    click_through_rate: float = 0.0
    conversion_rate: float = 0.0
    average_position: float = 0.0
    query_coverage: float = 0.0


class NLPSearchRanker:
    """Advanced NLP-powered search ranking using TensorFlow and Transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.encoder = None
        self.ranking_model = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize tokenizer, encoder, and ranking model"""
        try:
            # Load pre-trained tokenizer and encoder
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.encoder = TFAutoModel.from_pretrained(self.model_name)
            
            # Build ranking model
            self.ranking_model = self._build_ranking_model()
            
            self.logger.info(f"Successfully initialized NLP Search Ranker with model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP Search Ranker: {e}")
            raise
    
    def _build_ranking_model(self) -> tf.keras.Model:
        """Build TensorFlow ranking model"""
        try:
            # Input layers
            query_input = tf.keras.layers.Input(shape=(384,), name="query_embedding")
            property_input = tf.keras.layers.Input(shape=(384,), name="property_embedding")
            
            # Feature interaction layers
            concat_features = tf.keras.layers.Concatenate()([query_input, property_input])
            
            # Deep ranking network
            hidden = tf.keras.layers.Dense(512, activation='relu', name="hidden_1")(concat_features)
            hidden = tf.keras.layers.Dropout(0.3)(hidden)
            hidden = tf.keras.layers.Dense(256, activation='relu', name="hidden_2")(hidden)
            hidden = tf.keras.layers.Dropout(0.3)(hidden)
            hidden = tf.keras.layers.Dense(128, activation='relu', name="hidden_3")(hidden)
            hidden = tf.keras.layers.Dropout(0.2)(hidden)
            
            # Ranking score output
            score = tf.keras.layers.Dense(1, activation='sigmoid', name="relevance_score")(hidden)
            
            # Create model
            model = tf.keras.Model(
                inputs=[query_input, property_input],
                outputs=score,
                name="search_ranking_model"
            )
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.logger.info("Successfully built ranking model")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to build ranking model: {e}")
            raise
    
    def encode_text(self, texts: List[str]) -> tf.Tensor:
        """Encode texts using transformer model"""
        try:
            if not texts:
                return tf.constant([], dtype=tf.float32)
            
            # Tokenize inputs
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='tf',
                max_length=512
            )
            
            # Generate embeddings
            outputs = self.encoder(**inputs)
            
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize embeddings
            embeddings = tf.nn.l2_normalize(embeddings, axis=1)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode text: {e}")
            raise
    
    def rank_properties(self, query: str, properties: List[Dict]) -> List[RankingResult]:
        """Rank properties based on search query using ML"""
        try:
            if not query.strip() or not properties:
                return []
            
            # Encode query
            query_embedding = self.encode_text([query])
            
            # Prepare property texts
            property_texts = []
            for prop in properties:
                # Combine relevant fields for ranking
                text_parts = []
                if prop.get('title'):
                    text_parts.append(prop['title'])
                if prop.get('description'):
                    text_parts.append(prop['description'])
                if prop.get('location'):
                    text_parts.append(prop['location'])
                if prop.get('amenities'):
                    if isinstance(prop['amenities'], list):
                        text_parts.extend(prop['amenities'])
                    else:
                        text_parts.append(str(prop['amenities']))
                
                combined_text = ' '.join(text_parts)
                property_texts.append(combined_text)
            
            # Encode property descriptions
            property_embeddings = self.encode_text(property_texts)
            
            # Get ranking scores
            if self.is_trained:
                # Use trained model for ranking
                scores = self.ranking_model.predict([
                    tf.repeat(query_embedding, len(properties), axis=0),
                    property_embeddings
                ], verbose=0)
            else:
                # Use cosine similarity as fallback
                scores = self._calculate_cosine_similarity(query_embedding, property_embeddings)
            
            # Create ranking results
            ranking_results = []
            for i, (prop, score) in enumerate(zip(properties, scores.flatten())):
                ranking_features = {
                    'text_similarity': float(score),
                    'has_title': bool(prop.get('title')),
                    'has_description': bool(prop.get('description')),
                    'has_amenities': bool(prop.get('amenities')),
                    'text_length': len(property_texts[i])
                }
                
                result = RankingResult(
                    property_id=prop.get('id', str(i)),
                    property_data=prop,
                    relevance_score=float(score),
                    ranking_features=ranking_features
                )
                ranking_results.append(result)
            
            # Sort by relevance score (descending)
            ranking_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return ranking_results
            
        except Exception as e:
            self.logger.error(f"Failed to rank properties: {e}")
            return []
    
    def _calculate_cosine_similarity(self, query_embedding: tf.Tensor, property_embeddings: tf.Tensor) -> tf.Tensor:
        """Calculate cosine similarity between query and property embeddings"""
        try:
            # Normalize embeddings
            query_norm = tf.nn.l2_normalize(query_embedding, axis=1)
            property_norm = tf.nn.l2_normalize(property_embeddings, axis=1)
            
            # Calculate cosine similarity
            similarity = tf.matmul(query_norm, property_norm, transpose_b=True)
            
            # Return similarity scores
            return tf.transpose(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cosine similarity: {e}")
            return tf.zeros((len(property_embeddings), 1))
    
    def train(self, training_data: List[Tuple[str, Dict, float]], 
              validation_data: Optional[List[Tuple[str, Dict, float]]] = None,
              epochs: int = 10, batch_size: int = 32) -> Dict:
        """Train the ranking model on labeled data"""
        try:
            if not training_data:
                raise ValueError("Training data cannot be empty")
            
            # Prepare training data
            queries, properties, labels = zip(*training_data)
            
            # Encode queries and properties
            query_embeddings = self.encode_text(list(queries))
            
            property_texts = []
            for prop in properties:
                text_parts = [
                    prop.get('title', ''),
                    prop.get('description', ''),
                    prop.get('location', '')
                ]
                combined_text = ' '.join(filter(None, text_parts))
                property_texts.append(combined_text)
            
            property_embeddings = self.encode_text(property_texts)
            
            # Convert labels to numpy array
            labels_array = np.array(labels, dtype=np.float32)
            
            # Prepare validation data if provided
            validation_data_processed = None
            if validation_data:
                val_queries, val_properties, val_labels = zip(*validation_data)
                val_query_embeddings = self.encode_text(list(val_queries))
                
                val_property_texts = []
                for prop in val_properties:
                    text_parts = [
                        prop.get('title', ''),
                        prop.get('description', ''),
                        prop.get('location', '')
                    ]
                    combined_text = ' '.join(filter(None, text_parts))
                    val_property_texts.append(combined_text)
                
                val_property_embeddings = self.encode_text(val_property_texts)
                val_labels_array = np.array(val_labels, dtype=np.float32)
                
                validation_data_processed = (
                    [val_query_embeddings, val_property_embeddings],
                    val_labels_array
                )
            
            # Train model
            history = self.ranking_model.fit(
                [query_embeddings, property_embeddings],
                labels_array,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data_processed,
                verbose=1
            )
            
            self.is_trained = True
            
            # Return training metrics
            training_metrics = {
                'final_loss': float(history.history['loss'][-1]),
                'final_accuracy': float(history.history['accuracy'][-1]),
                'epochs_trained': len(history.history['loss']),
                'is_trained': self.is_trained
            }
            
            if validation_data_processed:
                training_metrics.update({
                    'final_val_loss': float(history.history['val_loss'][-1]),
                    'final_val_accuracy': float(history.history['val_accuracy'][-1])
                })
            
            self.logger.info(f"Training completed successfully: {training_metrics}")
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, model_path: str):
        """Save the trained model to disk"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            self.ranking_model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load a trained model from disk"""
        try:
            self.ranking_model = tf.keras.models.load_model(model_path)
            self.is_trained = True
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'embedding_dim': 384,
            'architecture': 'transformer_encoder + neural_ranker',
            'components': {
                'tokenizer': self.tokenizer is not None,
                'encoder': self.encoder is not None,
                'ranking_model': self.ranking_model is not None
            }
        }
    
    def evaluate(self, test_data: List[Tuple[str, Dict, float]]) -> Dict:
        """Evaluate model performance on test data"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            if not test_data:
                raise ValueError("Test data cannot be empty")
            
            # Prepare test data
            queries, properties, labels = zip(*test_data)
            
            # Encode test data
            query_embeddings = self.encode_text(list(queries))
            
            property_texts = []
            for prop in properties:
                text_parts = [
                    prop.get('title', ''),
                    prop.get('description', ''),
                    prop.get('location', '')
                ]
                combined_text = ' '.join(filter(None, text_parts))
                property_texts.append(combined_text)
            
            property_embeddings = self.encode_text(property_texts)
            labels_array = np.array(labels, dtype=np.float32)
            
            # Evaluate model
            evaluation_results = self.ranking_model.evaluate(
                [query_embeddings, property_embeddings],
                labels_array,
                verbose=0
            )
            
            # Format results
            metrics = {
                'test_loss': float(evaluation_results[0]),
                'test_accuracy': float(evaluation_results[1]),
                'test_precision': float(evaluation_results[2]),
                'test_recall': float(evaluation_results[3]),
                'test_samples': len(test_data)
            }
            
            self.logger.info(f"Evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise


class RankingModelManager:
    """Manager for different ranking models and ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.logger = logging.getLogger(__name__)
    
    def add_model(self, name: str, model: NLPSearchRanker, weight: float = 1.0):
        """Add a ranking model to the ensemble"""
        self.models[name] = model
        self.ensemble_weights[name] = weight
        self.logger.info(f"Added model '{name}' with weight {weight}")
    
    def rank_with_ensemble(self, query: str, properties: List[Dict], 
                          user_profile: Optional[UserProfile] = None) -> List[RankingResult]:
        """Rank properties using ensemble of models"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")
            
            # Get rankings from all models
            all_rankings = {}
            for name, model in self.models.items():
                try:
                    rankings = model.rank_properties(query, properties, user_profile)
                    all_rankings[name] = rankings
                except Exception as e:
                    self.logger.warning(f"Model {name} failed: {e}")
                    continue
            
            if not all_rankings:
                return []
            
            # Combine rankings using weighted average
            combined_scores = defaultdict(float)
            combined_features = defaultdict(dict)
            
            for name, rankings in all_rankings.items():
                weight = self.ensemble_weights.get(name, 1.0)
                for ranking in rankings:
                    prop_id = ranking.property_id
                    combined_scores[prop_id] += ranking.relevance_score * weight
                    combined_features[prop_id] = ranking.ranking_features
            
            # Normalize scores
            total_weight = sum(self.ensemble_weights.values())
            for prop_id in combined_scores:
                combined_scores[prop_id] /= total_weight
            
            # Create final ranking results
            final_results = []
            for prop in properties:
                prop_id = prop.get('id', str(hash(str(prop))))
                if prop_id in combined_scores:
                    result = RankingResult(
                        property_id=prop_id,
                        property_data=prop,
                        relevance_score=combined_scores[prop_id],
                        ranking_features=combined_features[prop_id],
                        explanation=f"Ensemble score from {len(all_rankings)} models",
                        ranking_position=0,
                        confidence=min(combined_scores[prop_id], 1.0)
                    )
                    final_results.append(result)
            
            # Sort by score
            final_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Update positions
            for i, result in enumerate(final_results):
                result.ranking_position = i + 1
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Ensemble ranking failed: {e}")
            return []
    
    def optimize_ensemble_weights(self, validation_data: List[Tuple[str, Dict, float]]):
        """Optimize ensemble weights based on validation data"""
        try:
            if len(self.models) < 2:
                return
            
            self.logger.info("Optimizing ensemble weights...")
            
            # Grid search for optimal weights
            best_weights = self.ensemble_weights.copy()
            best_score = 0.0
            
            # Generate weight combinations
            weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            model_names = list(self.models.keys())
            
            for weights in self._generate_weight_combinations(weight_options, len(model_names)):
                test_weights = {name: weight for name, weight in zip(model_names, weights)}
                
                # Test weights
                self.ensemble_weights = test_weights
                
                # Evaluate on validation data
                score = self._evaluate_ensemble(validation_data)
                
                if score > best_score:
                    best_score = score
                    best_weights = test_weights.copy()
            
            # Apply best weights
            self.ensemble_weights = best_weights
            
            self.logger.info(f"Optimized ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize ensemble weights: {e}")
    
    def _generate_weight_combinations(self, options: List[float], num_models: int):
        """Generate weight combinations that sum to 1"""
        from itertools import combinations_with_replacement
        
        for combo in combinations_with_replacement(options, num_models):
            if abs(sum(combo) - 1.0) < 0.1:  # Allow small deviation
                normalized = [w / sum(combo) for w in combo]
                yield normalized
    
    def _evaluate_ensemble(self, validation_data: List[Tuple[str, Dict, float]]) -> float:
        """Evaluate ensemble performance"""
        try:
            total_score = 0.0
            count = 0
            
            for query, prop, true_label in validation_data:
                results = self.rank_with_ensemble(query, [prop])
                if results:
                    predicted_score = results[0].relevance_score
                    total_score += 1.0 - abs(predicted_score - true_label)
                    count += 1
            
            return total_score / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate ensemble: {e}")
            return 0.0


# Factory function for creating search ranker instances
def create_search_ranker(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                        config: Optional[Dict] = None) -> NLPSearchRanker:
    """Factory function to create configured search ranker"""
    ranker = NLPSearchRanker(model_name=model_name)
    
    if config:
        # Apply configuration
        if 'feature_weights' in config:
            ranker.feature_weights.update(config['feature_weights'])
        
        if 'cache_size' in config:
            # Implement cache size limits if needed
            pass
    
    return ranker


# Example usage and testing
if __name__ == "__main__":
    # Initialize ranker
    ranker = create_search_ranker()
    
    # Example properties
    properties = [
        {
            'id': '1',
            'title': 'Modern 2BR Apartment Downtown',
            'description': 'Beautiful modern apartment with city views',
            'location': 'Downtown Toronto',
            'price': 2500,
            'amenities': ['gym', 'pool', 'concierge'],
            'type': 'apartment',
            'created_date': '2024-01-15'
        },
        {
            'id': '2',
            'title': 'Cozy Studio Near University',
            'description': 'Perfect for students, close to campus',
            'location': 'University District',
            'price': 1200,
            'amenities': ['wifi', 'laundry'],
            'type': 'studio',
            'created_date': '2024-01-20'
        }
    ]
    
    # Test ranking
    results = ranker.rank_properties("downtown apartment with gym", properties)
    
    for result in results:
        print(f"Property: {result.property_data['title']}")
        print(f"Score: {result.relevance_score:.4f}")
        print(f"Explanation: {result.explanation}")
        print("---")