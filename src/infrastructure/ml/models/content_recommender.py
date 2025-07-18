import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, jaccard_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cluster import KMeans
import logging
from dataclasses import dataclass, field
from .collaborative_filter import BaseRecommender, RecommendationResult, EvaluationMetrics
import re
import time
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import joblib
from functools import lru_cache
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager
warnings.filterwarnings('ignore')


@dataclass
class PropertyFeatures:
    """Structured container for property features"""
    location_features: np.ndarray
    price_features: np.ndarray
    bedroom_features: np.ndarray
    bathroom_features: np.ndarray
    amenity_features: np.ndarray
    text_features: np.ndarray
    categorical_features: np.ndarray
    numerical_features: np.ndarray
    combined_features: np.ndarray
    feature_names: List[str]
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    
    def get_feature_vector(self, feature_type: str = 'all') -> np.ndarray:
        """Get specific feature vector type"""
        if feature_type == 'location':
            return self.location_features
        elif feature_type == 'price':
            return self.price_features
        elif feature_type == 'amenity':
            return self.amenity_features
        elif feature_type == 'text':
            return self.text_features
        elif feature_type == 'categorical':
            return self.categorical_features
        elif feature_type == 'numerical':
            return self.numerical_features
        else:
            return self.combined_features
            
    def get_weighted_features(self) -> np.ndarray:
        """Get features weighted by importance"""
        if not self.feature_weights:
            return self.combined_features
        
        weights = np.array([self.feature_weights.get(name, 1.0) for name in self.feature_names])
        return self.combined_features * weights


@dataclass
class SimilarityConfig:
    """Configuration for similarity computation"""
    cosine_weight: float = 0.4
    euclidean_weight: float = 0.3
    jaccard_weight: float = 0.2
    manhattan_weight: float = 0.1
    use_cache: bool = True
    cache_size: int = 10000
    similarity_threshold: float = 0.1
    
    
@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    text_max_features: int = 5000
    text_ngram_range: Tuple[int, int] = (1, 3)
    text_min_df: int = 2
    text_max_df: float = 0.8
    location_embedding_dim: int = 50
    amenity_embedding_dim: int = 50
    use_feature_selection: bool = True
    feature_selection_k: int = 1000
    use_dimensionality_reduction: bool = False
    pca_components: int = 100
    
    
@dataclass
class UserProfile:
    """User preference profile"""
    user_id: int
    preferred_locations: List[str] = field(default_factory=list)
    preferred_amenities: List[str] = field(default_factory=list)
    price_range: Tuple[float, float] = (0.0, float('inf'))
    preferred_bedrooms: int = 1
    preferred_bathrooms: int = 1
    feature_preferences: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    learned_preferences: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_from_interaction(self, property_data: Dict, rating: float):
        """Update user profile based on interaction"""
        self.interaction_history.append({
            'property_id': property_data.get('id'),
            'rating': rating,
            'timestamp': datetime.now(),
            'property_data': property_data
        })
        
        # Update preferences based on positive interactions
        if rating > 0.6:
            if 'location' in property_data:
                self.preferred_locations.append(property_data['location'])
            if 'amenities' in property_data:
                self.preferred_amenities.extend(property_data['amenities'])
        
        self.last_updated = datetime.now()
        
        
class AdvancedFeatureProcessor(BaseEstimator, TransformerMixin):
    """Advanced feature processing for property data"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.text_vectorizer = None
        self.location_encoder = None
        self.amenity_encoder = None
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.feature_selector = None
        self.dimensionality_reducer = None
        self.is_fitted = False
        
    def fit(self, X: List[Dict], y: np.ndarray = None):
        """Fit the feature processor"""
        self._initialize_processors()
        
        # Extract different types of features
        text_data = self._extract_text_features(X)
        location_data = self._extract_location_features(X)
        amenity_data = self._extract_amenity_features(X)
        categorical_data = self._extract_categorical_features(X)
        numerical_data = self._extract_numerical_features(X)
        
        # Fit text vectorizer
        if text_data:
            self.text_vectorizer.fit(text_data)
        
        # Fit location encoder
        if location_data:
            self.location_encoder.fit(location_data)
        
        # Fit amenity encoder
        if amenity_data:
            self.amenity_encoder.fit(amenity_data)
        
        # Fit categorical encoders
        for feature_name, values in categorical_data.items():
            if values:
                self.categorical_encoders[feature_name].fit(values)
        
        # Fit numerical scalers
        for feature_name, values in numerical_data.items():
            if values:
                values_array = np.array(values).reshape(-1, 1)
                self.numerical_scalers[feature_name].fit(values_array)
        
        # Fit feature selector if enabled
        if self.config.use_feature_selection:
            combined_features = self._transform_features(X)
            if y is not None and combined_features.shape[1] > self.config.feature_selection_k:
                self.feature_selector.fit(combined_features, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: List[Dict]) -> np.ndarray:
        """Transform property data into feature vectors"""
        check_is_fitted(self, 'is_fitted')
        return self._transform_features(X)
    
    def _initialize_processors(self):
        """Initialize feature processors"""
        # Text vectorizer with advanced preprocessing
        self.text_vectorizer = TfidfVectorizer(
            max_features=self.config.text_max_features,
            ngram_range=self.config.text_ngram_range,
            min_df=self.config.text_min_df,
            max_df=self.config.text_max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            preprocessor=self._preprocess_text
        )
        
        # Location encoder
        self.location_encoder = LabelEncoder()
        
        # Amenity encoder (for multi-label)
        self.amenity_encoder = TfidfVectorizer(
            max_features=self.config.amenity_embedding_dim,
            ngram_range=(1, 2),
            binary=True,
            lowercase=True
        )
        
        # Feature selector
        if self.config.use_feature_selection:
            self.feature_selector = SelectKBest(
                score_func=chi2,
                k=self.config.feature_selection_k
            )
        
        # Dimensionality reducer
        if self.config.use_dimensionality_reduction:
            self.dimensionality_reducer = TruncatedSVD(
                n_components=self.config.pca_components,
                random_state=42
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Common real estate term normalization
        text = re.sub(r'\bbath\b', 'bathroom', text)
        text = re.sub(r'\bbed\b', 'bedroom', text)
        text = re.sub(r'\bsq\s*ft\b', 'square_feet', text)
        text = re.sub(r'\bac\b', 'air_conditioning', text)
        text = re.sub(r'\bpkg\b', 'parking', text)
        
        return text
    
    def _extract_text_features(self, X: List[Dict]) -> List[str]:
        """Extract text features from property data"""
        text_features = []
        
        for prop in X:
            text_parts = []
            
            # Description
            if prop.get('description'):
                text_parts.append(str(prop['description']))
            
            # Property type
            if prop.get('property_type'):
                text_parts.append(str(prop['property_type']))
            
            # Amenities as text
            if prop.get('amenities'):
                if isinstance(prop['amenities'], list):
                    text_parts.extend(prop['amenities'])
                else:
                    text_parts.append(str(prop['amenities']))
            
            # Additional features
            for key in ['features', 'highlights', 'lease_terms']:
                if prop.get(key):
                    if isinstance(prop[key], list):
                        text_parts.extend(prop[key])
                    else:
                        text_parts.append(str(prop[key]))
            
            combined_text = ' '.join(text_parts)
            text_features.append(combined_text)
        
        return text_features
    
    def _extract_location_features(self, X: List[Dict]) -> List[str]:
        """Extract location features"""
        location_features = []
        
        for prop in X:
            location_parts = []
            
            for key in ['neighborhood', 'city', 'state', 'zip_code', 'region']:
                if prop.get(key):
                    location_parts.append(str(prop[key]))
            
            location_str = ' '.join(location_parts) if location_parts else 'unknown'
            location_features.append(location_str)
        
        return location_features
    
    def _extract_amenity_features(self, X: List[Dict]) -> List[str]:
        """Extract amenity features"""
        amenity_features = []
        
        for prop in X:
            amenities = prop.get('amenities', [])
            if isinstance(amenities, list):
                amenity_text = ' '.join(amenities)
            else:
                amenity_text = str(amenities)
            
            amenity_features.append(amenity_text)
        
        return amenity_features
    
    def _extract_categorical_features(self, X: List[Dict]) -> Dict[str, List[str]]:
        """Extract categorical features"""
        categorical_features = defaultdict(list)
        
        categorical_keys = ['property_type', 'building_type', 'parking_type', 'pet_policy']
        
        for prop in X:
            for key in categorical_keys:
                value = prop.get(key, 'unknown')
                categorical_features[key].append(str(value))
        
        return categorical_features
    
    def _extract_numerical_features(self, X: List[Dict]) -> Dict[str, List[float]]:
        """Extract numerical features"""
        numerical_features = defaultdict(list)
        
        numerical_keys = ['price', 'bedrooms', 'bathrooms', 'square_feet', 'lot_size']
        
        for prop in X:
            for key in numerical_keys:
                value = prop.get(key, 0)
                try:
                    numerical_features[key].append(float(value))
                except (ValueError, TypeError):
                    numerical_features[key].append(0.0)
        
        return numerical_features
    
    def _transform_features(self, X: List[Dict]) -> np.ndarray:
        """Transform features into final feature matrix"""
        feature_matrices = []
        
        # Text features
        text_data = self._extract_text_features(X)
        if text_data and self.text_vectorizer:
            text_features = self.text_vectorizer.transform(text_data)
            feature_matrices.append(text_features)
        
        # Location features
        location_data = self._extract_location_features(X)
        if location_data and self.location_encoder:
            try:
                location_encoded = self.location_encoder.transform(location_data)
                location_features = np.eye(len(self.location_encoder.classes_))[location_encoded]
                feature_matrices.append(csr_matrix(location_features))
            except ValueError:
                # Handle unseen locations
                location_features = np.zeros((len(X), len(self.location_encoder.classes_)))
                feature_matrices.append(csr_matrix(location_features))
        
        # Amenity features
        amenity_data = self._extract_amenity_features(X)
        if amenity_data and self.amenity_encoder:
            amenity_features = self.amenity_encoder.transform(amenity_data)
            feature_matrices.append(amenity_features)
        
        # Categorical features
        categorical_data = self._extract_categorical_features(X)
        for feature_name, values in categorical_data.items():
            if feature_name in self.categorical_encoders:
                encoder = self.categorical_encoders[feature_name]
                try:
                    encoded = encoder.transform(values)
                    one_hot = np.eye(len(encoder.classes_))[encoded]
                    feature_matrices.append(csr_matrix(one_hot))
                except ValueError:
                    # Handle unseen categories
                    one_hot = np.zeros((len(X), len(encoder.classes_)))
                    feature_matrices.append(csr_matrix(one_hot))
        
        # Numerical features
        numerical_data = self._extract_numerical_features(X)
        for feature_name, values in numerical_data.items():
            if feature_name in self.numerical_scalers:
                scaler = self.numerical_scalers[feature_name]
                values_array = np.array(values).reshape(-1, 1)
                scaled = scaler.transform(values_array)
                feature_matrices.append(csr_matrix(scaled))
        
        # Combine all features
        if feature_matrices:
            combined_features = hstack(feature_matrices)
            
            # Apply feature selection if enabled
            if self.config.use_feature_selection and self.feature_selector:
                combined_features = self.feature_selector.transform(combined_features)
            
            # Apply dimensionality reduction if enabled
            if self.config.use_dimensionality_reduction and self.dimensionality_reducer:
                combined_features = self.dimensionality_reducer.transform(combined_features)
            
            return combined_features.toarray()
        
        return np.array([])
        
        
class SimilarityCalculator:
    """Advanced similarity calculation with multiple methods"""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.similarity_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                           method: str = 'combined') -> float:
        """Calculate similarity between two feature vectors"""
        if method == 'combined':
            return self._calculate_combined_similarity(features1, features2)
        elif method == 'cosine':
            return self._calculate_cosine_similarity(features1, features2)
        elif method == 'euclidean':
            return self._calculate_euclidean_similarity(features1, features2)
        elif method == 'jaccard':
            return self._calculate_jaccard_similarity(features1, features2)
        elif method == 'manhattan':
            return self._calculate_manhattan_similarity(features1, features2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def calculate_pairwise_similarity(self, features: np.ndarray, 
                                    method: str = 'combined') -> np.ndarray:
        """Calculate pairwise similarity matrix"""
        if self.config.use_cache:
            cache_key = self._get_cache_key(features, method)
            if cache_key in self.similarity_cache:
                self.cache_hits += 1
                return self.similarity_cache[cache_key]
        
        if method == 'combined':
            similarity_matrix = self._calculate_combined_similarity_matrix(features)
        elif method == 'cosine':
            similarity_matrix = cosine_similarity(features)
        elif method == 'euclidean':
            distances = euclidean_distances(features)
            similarity_matrix = 1 / (1 + distances)
        elif method == 'jaccard':
            similarity_matrix = self._calculate_jaccard_similarity_matrix(features)
        elif method == 'manhattan':
            distances = pdist(features, metric='manhattan')
            distance_matrix = squareform(distances)
            similarity_matrix = 1 / (1 + distance_matrix)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Cache the result
        if self.config.use_cache:
            self._cache_similarity_matrix(cache_key, similarity_matrix)
        
        self.cache_misses += 1
        return similarity_matrix
    
    def _calculate_combined_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate combined similarity score"""
        cosine_sim = self._calculate_cosine_similarity(features1, features2)
        euclidean_sim = self._calculate_euclidean_similarity(features1, features2)
        jaccard_sim = self._calculate_jaccard_similarity(features1, features2)
        manhattan_sim = self._calculate_manhattan_similarity(features1, features2)
        
        combined_score = (
            self.config.cosine_weight * cosine_sim +
            self.config.euclidean_weight * euclidean_sim +
            self.config.jaccard_weight * jaccard_sim +
            self.config.manhattan_weight * manhattan_sim
        )
        
        return combined_score
    
    def _calculate_combined_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Calculate combined similarity matrix"""
        cosine_matrix = cosine_similarity(features)
        
        euclidean_distances = euclidean_distances(features)
        euclidean_matrix = 1 / (1 + euclidean_distances)
        
        jaccard_matrix = self._calculate_jaccard_similarity_matrix(features)
        
        manhattan_distances = pdist(features, metric='manhattan')
        manhattan_matrix = 1 / (1 + squareform(manhattan_distances))
        
        combined_matrix = (
            self.config.cosine_weight * cosine_matrix +
            self.config.euclidean_weight * euclidean_matrix +
            self.config.jaccard_weight * jaccard_matrix +
            self.config.manhattan_weight * manhattan_matrix
        )
        
        return combined_matrix
    
    def _calculate_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_euclidean_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate euclidean similarity"""
        distance = np.linalg.norm(features1 - features2)
        return 1 / (1 + distance)
    
    def _calculate_jaccard_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate Jaccard similarity for binary features"""
        # Convert to binary
        binary1 = (features1 > 0).astype(int)
        binary2 = (features2 > 0).astype(int)
        
        intersection = np.sum(binary1 & binary2)
        union = np.sum(binary1 | binary2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_manhattan_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate Manhattan similarity"""
        distance = np.sum(np.abs(features1 - features2))
        return 1 / (1 + distance)
    
    def _calculate_jaccard_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Calculate Jaccard similarity matrix"""
        binary_features = (features > 0).astype(int)
        n_samples = binary_features.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                sim = self._calculate_jaccard_similarity(binary_features[i], binary_features[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix
    
    def _get_cache_key(self, features: np.ndarray, method: str) -> str:
        """Generate cache key for similarity matrix"""
        features_hash = hashlib.md5(features.tobytes()).hexdigest()
        return f"{method}_{features_hash}_{features.shape}"
    
    def _cache_similarity_matrix(self, cache_key: str, matrix: np.ndarray):
        """Cache similarity matrix with size limit"""
        if len(self.similarity_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.similarity_cache))
            del self.similarity_cache[oldest_key]
        
        self.similarity_cache[cache_key] = matrix
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.similarity_cache),
            'max_cache_size': self.config.cache_size
        }
    
    def clear_cache(self):
        """Clear similarity cache"""
        self.similarity_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        
class UserPreferenceModeler:
    """Model user preferences from interaction history"""
    
    def __init__(self, feature_processor: AdvancedFeatureProcessor):
        self.feature_processor = feature_processor
        self.user_profiles = {}
        self.global_preferences = {}
        self.preference_weights = {}
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
    def update_user_profile(self, user_id: int, property_data: Dict, 
                           rating: float, timestamp: datetime = None):
        """Update user profile based on interaction"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize user profile if not exists
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        profile.update_from_interaction(property_data, rating)
        
        # Learn feature preferences
        self._learn_feature_preferences(user_id, property_data, rating)
        
        # Update global preferences
        self._update_global_preferences(property_data, rating)
    
    def get_user_preferences(self, user_id: int) -> Dict[str, float]:
        """Get user preference vector"""
        if user_id not in self.user_profiles:
            return self.global_preferences.copy()
        
        profile = self.user_profiles[user_id]
        return profile.learned_preferences
    
    def predict_user_preference(self, user_id: int, property_data: Dict) -> float:
        """Predict user preference for a property"""
        preferences = self.get_user_preferences(user_id)
        
        if not preferences:
            return 0.5  # Default neutral preference
        
        # Extract property features
        property_features = self.feature_processor.transform([property_data])[0]
        
        # Calculate preference score
        preference_score = 0.0
        total_weight = 0.0
        
        for feature_name, weight in preferences.items():
            if feature_name in self.feature_processor.feature_names:
                feature_idx = self.feature_processor.feature_names.index(feature_name)
                if feature_idx < len(property_features):
                    feature_value = property_features[feature_idx]
                    preference_score += weight * feature_value
                    total_weight += abs(weight)
        
        if total_weight > 0:
            preference_score /= total_weight
        
        # Normalize to [0, 1]
        return max(0, min(1, (preference_score + 1) / 2))
    
    def _learn_feature_preferences(self, user_id: int, property_data: Dict, rating: float):
        """Learn feature preferences from interaction"""
        profile = self.user_profiles[user_id]
        
        # Extract property features
        property_features = self.feature_processor.transform([property_data])[0]
        
        # Update learned preferences
        for i, feature_value in enumerate(property_features):
            if i < len(self.feature_processor.feature_names):
                feature_name = self.feature_processor.feature_names[i]
                
                # Current preference
                current_pref = profile.learned_preferences.get(feature_name, 0.0)
                
                # Update with exponential moving average
                target_pref = (rating - 0.5) * feature_value  # Center around 0.5
                new_pref = current_pref * self.decay_factor + target_pref * self.learning_rate
                
                profile.learned_preferences[feature_name] = new_pref
    
    def _update_global_preferences(self, property_data: Dict, rating: float):
        """Update global preference statistics"""
        # Extract property features
        property_features = self.feature_processor.transform([property_data])[0]
        
        # Update global preferences
        for i, feature_value in enumerate(property_features):
            if i < len(self.feature_processor.feature_names):
                feature_name = self.feature_processor.feature_names[i]
                
                # Current global preference
                current_pref = self.global_preferences.get(feature_name, 0.0)
                
                # Update with exponential moving average
                target_pref = (rating - 0.5) * feature_value
                new_pref = current_pref * self.decay_factor + target_pref * (self.learning_rate * 0.1)
                
                self.global_preferences[feature_name] = new_pref
    
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """Get user profile"""
        return self.user_profiles.get(user_id)
    
    def get_similar_users(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find similar users based on preferences"""
        if user_id not in self.user_profiles:
            return []
        
        target_preferences = self.user_profiles[user_id].learned_preferences
        similar_users = []
        
        for other_user_id, other_profile in self.user_profiles.items():
            if other_user_id == user_id:
                continue
            
            # Calculate preference similarity
            similarity = self._calculate_preference_similarity(
                target_preferences, other_profile.learned_preferences
            )
            
            similar_users.append((other_user_id, similarity))
        
        # Sort by similarity and return top-k
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[:top_k]
    
    def _calculate_preference_similarity(self, prefs1: Dict[str, float], 
                                       prefs2: Dict[str, float]) -> float:
        """Calculate similarity between two preference vectors"""
        common_features = set(prefs1.keys()) & set(prefs2.keys())
        
        if not common_features:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(prefs1[f] * prefs2[f] for f in common_features)
        norm1 = math.sqrt(sum(prefs1[f]**2 for f in common_features))
        norm2 = math.sqrt(sum(prefs2[f]**2 for f in common_features))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def save_user_profiles(self, file_path: str):
        """Save user profiles to file"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'user_profiles': self.user_profiles,
                'global_preferences': self.global_preferences,
                'preference_weights': self.preference_weights
            }, f)
    
    def load_user_profiles(self, file_path: str):
        """Load user profiles from file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.user_profiles = data['user_profiles']
            self.global_preferences = data['global_preferences']
            self.preference_weights = data['preference_weights']
            
            
class ContentBasedRecommender(BaseRecommender):
    """
    Advanced Content-based recommendation system for rental properties.
    
    This recommender builds comprehensive user preferences from property features including:
    - Location (neighborhood, city, region) with embedding representations
    - Price range and price per square foot with advanced scaling
    - Property specifications (bedrooms, bathrooms, square footage)
    - Amenities and features with TF-IDF and semantic processing
    - Property type and style with categorical encoding
    - Text descriptions with advanced NLP preprocessing
    
    The model uses multiple similarity methods, advanced feature engineering,
    user preference learning, and comprehensive evaluation metrics.
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 location_vocab_size: int = 1000,
                 amenity_vocab_size: int = 500,
                 reg_lambda: float = 1e-5,
                 learning_rate: float = 0.001,
                 feature_config: FeatureConfig = None,
                 similarity_config: SimilarityConfig = None,
                 use_neural_model: bool = True,
                 enable_user_modeling: bool = True,
                 enable_caching: bool = True):
        """
        Initialize the Advanced ContentBasedRecommender.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            location_vocab_size: Maximum number of unique locations
            amenity_vocab_size: Maximum number of unique amenities
            reg_lambda: L2 regularization strength
            learning_rate: Learning rate for optimizer
            feature_config: Configuration for feature engineering
            similarity_config: Configuration for similarity computation
            use_neural_model: Whether to use neural network model
            enable_user_modeling: Whether to enable user preference modeling
            enable_caching: Whether to enable similarity caching
        """
        # Configuration
        self.embedding_dim = embedding_dim
        self.location_vocab_size = location_vocab_size
        self.amenity_vocab_size = amenity_vocab_size
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.use_neural_model = use_neural_model
        self.enable_user_modeling = enable_user_modeling
        self.enable_caching = enable_caching
        
        # Feature and similarity configurations
        self.feature_config = feature_config or FeatureConfig()
        self.similarity_config = similarity_config or SimilarityConfig()
        
        # Core components
        self.model = None
        self.feature_processor = AdvancedFeatureProcessor(self.feature_config)
        self.similarity_calculator = SimilarityCalculator(self.similarity_config)
        self.user_preference_modeler = None
        
        # Legacy components for backward compatibility
        self.location_encoder = None
        self.amenity_vectorizer = None
        self.price_scaler = None
        
        # Training data storage
        self.property_features = None
        self.user_item_matrix = None
        self.property_similarity_matrix = None
        self.feature_importance_scores = {}
        self.feature_weights = {}
        
        # Performance tracking
        self.training_metrics = {}
        self.evaluation_metrics = {}
        self.performance_history = []
        
        # State tracking
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Build the neural network model if enabled
        if self.use_neural_model:
            self.model = self._build_neural_model()
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize user preference modeler if enabled
            if self.enable_user_modeling:
                self.user_preference_modeler = UserPreferenceModeler(self.feature_processor)
            
            # Initialize legacy components for backward compatibility
            self.location_encoder = LabelEncoder()
            self.amenity_vectorizer = TfidfVectorizer(
                max_features=self.amenity_vocab_size,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                lowercase=True
            )
            self.price_scaler = StandardScaler()
            
            # Initialize feature importance tracking
            self.feature_importance_scores = {}
            self.feature_weights = {}
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _build_neural_model(self) -> tf.keras.Model:
        """
        Build the advanced content-based recommendation neural network.
        
        The model processes multiple types of property features:
        1. Location embeddings with advanced encoding
        2. Numerical features (price, bedrooms, bathrooms, square feet)
        3. Amenity text features with TF-IDF
        4. Text description features with advanced NLP
        5. Categorical features with one-hot encoding
        6. Combined feature interactions with attention mechanisms
        
        Returns:
            Compiled TensorFlow model
        """
        try:
            # Input layers for different feature types
            location_input = tf.keras.layers.Input(shape=(), name='location_input')
            numerical_input = tf.keras.layers.Input(shape=(5,), name='numerical_input')  # price, bedrooms, bathrooms, sqft, lot_size
            amenity_input = tf.keras.layers.Input(shape=(self.amenity_vocab_size,), name='amenity_input')
            text_input = tf.keras.layers.Input(shape=(self.feature_config.text_max_features,), name='text_input')
            categorical_input = tf.keras.layers.Input(shape=(10,), name='categorical_input')  # Various categorical features
            
            # Location embedding layer
            location_embedding = tf.keras.layers.Embedding(
                input_dim=self.location_vocab_size,
                output_dim=self.embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='location_embedding'
            )(location_input)
            location_flat = tf.keras.layers.Flatten(name='location_flatten')(location_embedding)
            
            # Numerical features processing
            numerical_dense = tf.keras.layers.Dense(
                64, 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='numerical_dense'
            )(numerical_input)
            numerical_dropout = tf.keras.layers.Dropout(0.2)(numerical_dense)
            
            # Text features processing
            text_dense = tf.keras.layers.Dense(
                128, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='text_dense'
            )(text_input)
            text_dropout = tf.keras.layers.Dropout(0.3)(text_dense)
            
            # Categorical features processing
            categorical_dense = tf.keras.layers.Dense(
                32, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='categorical_dense'
            )(categorical_input)
            categorical_dropout = tf.keras.layers.Dropout(0.2)(categorical_dense)
            
            # Amenity features processing
            amenity_dense = tf.keras.layers.Dense(
                64, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='amenity_dense'
            )(amenity_input)
            amenity_dropout = tf.keras.layers.Dropout(0.2)(amenity_dense)
            
            # Combine all features with attention mechanism
            combined = tf.keras.layers.Concatenate(name='feature_concat')([
                location_flat,
                numerical_dropout,
                text_dropout,
                amenity_dropout,
                categorical_dropout
            ])
            
            # Add attention mechanism for feature importance
            attention = tf.keras.layers.Dense(
                combined.shape[-1],
                activation='softmax',
                name='attention_weights'
            )(combined)
            
            # Apply attention weights
            attended_features = tf.keras.layers.Multiply(name='attended_features')([combined, attention])
            
            # Deep neural network layers for feature interaction learning
            hidden1 = tf.keras.layers.Dense(
                512, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='hidden_1'
            )(attended_features)
            hidden1_bn = tf.keras.layers.BatchNormalization(name='bn_1')(hidden1)
            hidden1_dropout = tf.keras.layers.Dropout(0.3)(hidden1_bn)
            
            hidden2 = tf.keras.layers.Dense(
                256, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='hidden_2'
            )(hidden1_dropout)
            hidden2_bn = tf.keras.layers.BatchNormalization(name='bn_2')(hidden2)
            hidden2_dropout = tf.keras.layers.Dropout(0.3)(hidden2_bn)
            
            hidden3 = tf.keras.layers.Dense(
                128, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='hidden_3'
            )(hidden2_dropout)
            hidden3_bn = tf.keras.layers.BatchNormalization(name='bn_3')(hidden3)
            hidden3_dropout = tf.keras.layers.Dropout(0.2)(hidden3_bn)
            
            hidden4 = tf.keras.layers.Dense(
                64, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_lambda),
                name='hidden_4'
            )(hidden3_dropout)
            hidden4_dropout = tf.keras.layers.Dropout(0.2)(hidden4)
            
            # Output layer for preference prediction
            output = tf.keras.layers.Dense(
                1, 
                activation='sigmoid', 
                name='preference_output'
            )(hidden4_dropout)
            
            # Create and compile model
            model = tf.keras.Model(
                inputs=[location_input, numerical_input, amenity_input, text_input, categorical_input],
                outputs=output,
                name='advanced_content_based_recommender'
            )
            
            # Use adaptive learning rate
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall', 'mae', 'auc']
            )
            
            self.logger.info(f"Advanced content-based recommendation model built successfully with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to build advanced content-based model: {e}")
            raise
    
    def extract_property_features(self, properties: List[Dict], 
                                 use_advanced_processing: bool = True) -> PropertyFeatures:
        """
        Extract and process comprehensive features from property data.
        
        Args:
            properties: List of property dictionaries with features
            use_advanced_processing: Whether to use advanced feature processing
            
        Returns:
            PropertyFeatures object containing processed features
        """
        try:
            if not properties:
                raise ValueError("Properties list cannot be empty")
            
            start_time = time.time()
            
            if use_advanced_processing:
                # Use advanced feature processor
                combined_features = self.feature_processor.transform(properties)
                
                # Extract specific feature types for backward compatibility
                location_features = self._extract_location_features_legacy(properties)
                price_features = self._extract_price_features_legacy(properties)
                amenity_features = self._extract_amenity_features_legacy(properties)
                
                # Additional feature types
                text_features = self._extract_text_features_enhanced(properties)
                categorical_features = self._extract_categorical_features_enhanced(properties)
                numerical_features = self._extract_numerical_features_enhanced(properties)
                
                # Create bedroom and bathroom specific features
                bedroom_features = price_features[:, 1:2] if price_features.shape[1] > 1 else np.zeros((len(properties), 1))
                bathroom_features = price_features[:, 2:3] if price_features.shape[1] > 2 else np.zeros((len(properties), 1))
                
                # Calculate feature importance
                feature_importance = self._calculate_feature_importance(combined_features)
                
                # Calculate feature weights
                feature_weights = self._calculate_feature_weights(properties)
                
                # Generate comprehensive feature names
                feature_names = self._generate_feature_names(properties)
                
                processing_time = time.time() - start_time
                self.logger.info(f"Advanced feature extraction completed in {processing_time:.2f} seconds")
                
            else:
                # Use legacy feature extraction for backward compatibility
                location_features = self._extract_location_features_legacy(properties)
                price_features = self._extract_price_features_legacy(properties)
                amenity_features = self._extract_amenity_features_legacy(properties)
                
                # Default values for missing features
                text_features = np.zeros((len(properties), 100))
                categorical_features = np.zeros((len(properties), 10))
                numerical_features = price_features
                
                # Create bedroom and bathroom specific features
                bedroom_features = price_features[:, 1:2] if price_features.shape[1] > 1 else np.zeros((len(properties), 1))
                bathroom_features = price_features[:, 2:3] if price_features.shape[1] > 2 else np.zeros((len(properties), 1))
                
                # Combine all features for similarity computation
                combined_features = np.concatenate([
                    location_features.reshape(-1, 1),
                    price_features,
                    amenity_features
                ], axis=1)
                
                # Basic feature importance and weights
                feature_importance = {}
                feature_weights = {}
                
                # Generate basic feature names
                feature_names = (['location'] + 
                               ['price', 'bedrooms', 'bathrooms'] + 
                               [f'amenity_{i}' for i in range(amenity_features.shape[1])])
                
                processing_time = time.time() - start_time
                self.logger.info(f"Legacy feature extraction completed in {processing_time:.2f} seconds")
            
            return PropertyFeatures(
                location_features=location_features,
                price_features=price_features,
                bedroom_features=bedroom_features,
                bathroom_features=bathroom_features,
                amenity_features=amenity_features,
                text_features=text_features,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                combined_features=combined_features,
                feature_names=feature_names,
                feature_importance=feature_importance,
                feature_weights=feature_weights
            )
            
        except Exception as e:
            self.logger.error(f"Property feature extraction failed: {e}")
            raise
    
    def _extract_location_features_legacy(self, properties: List[Dict]) -> np.ndarray:
        """Extract location features using legacy method"""
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
            location_encoded = self.location_encoder.fit_transform(locations)
        else:
            location_encoded = []
            for location in locations:
                try:
                    encoded = self.location_encoder.transform([location])[0]
                    location_encoded.append(encoded)
                except ValueError:
                    location_encoded.append(0)
            location_encoded = np.array(location_encoded)
        
        return location_encoded
    
    def _extract_price_features_legacy(self, properties: List[Dict]) -> np.ndarray:
        """Extract price features using legacy method"""
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
        
        return price_features_scaled
    
    def _extract_amenity_features_legacy(self, properties: List[Dict]) -> np.ndarray:
        """Extract amenity features using legacy method"""
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
        
        return amenity_features.toarray()
    
    def _extract_text_features_enhanced(self, properties: List[Dict]) -> np.ndarray:
        """Extract enhanced text features"""
        if not hasattr(self.feature_processor, 'text_vectorizer') or not self.feature_processor.text_vectorizer:
            return np.zeros((len(properties), 100))
        
        text_data = self.feature_processor._extract_text_features(properties)
        if not text_data:
            return np.zeros((len(properties), 100))
        
        try:
            text_features = self.feature_processor.text_vectorizer.transform(text_data)
            return text_features.toarray()
        except Exception as e:
            self.logger.warning(f"Text feature extraction failed: {e}")
            return np.zeros((len(properties), 100))
    
    def _extract_categorical_features_enhanced(self, properties: List[Dict]) -> np.ndarray:
        """Extract enhanced categorical features"""
        categorical_data = self.feature_processor._extract_categorical_features(properties)
        feature_matrices = []
        
        for feature_name, values in categorical_data.items():
            if feature_name in self.feature_processor.categorical_encoders:
                encoder = self.feature_processor.categorical_encoders[feature_name]
                try:
                    encoded = encoder.transform(values)
                    one_hot = np.eye(len(encoder.classes_))[encoded]
                    feature_matrices.append(one_hot)
                except (ValueError, AttributeError):
                    # Handle unseen categories or unfitted encoder
                    feature_matrices.append(np.zeros((len(properties), 5)))
        
        if feature_matrices:
            return np.concatenate(feature_matrices, axis=1)
        else:
            return np.zeros((len(properties), 10))
    
    def _extract_numerical_features_enhanced(self, properties: List[Dict]) -> np.ndarray:
        """Extract enhanced numerical features"""
        numerical_data = self.feature_processor._extract_numerical_features(properties)
        feature_matrices = []
        
        for feature_name, values in numerical_data.items():
            if feature_name in self.feature_processor.numerical_scalers:
                scaler = self.feature_processor.numerical_scalers[feature_name]
                try:
                    values_array = np.array(values).reshape(-1, 1)
                    scaled = scaler.transform(values_array)
                    feature_matrices.append(scaled)
                except (ValueError, AttributeError):
                    # Handle unfitted scaler
                    feature_matrices.append(np.zeros((len(properties), 1)))
        
        if feature_matrices:
            return np.concatenate(feature_matrices, axis=1)
        else:
            return np.zeros((len(properties), 5))
    
    def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance scores"""
        try:
            if features.size == 0:
                return {}
            
            # Calculate variance-based importance
            variances = np.var(features, axis=0)
            
            # Normalize to sum to 1
            total_variance = np.sum(variances)
            if total_variance > 0:
                importance_scores = variances / total_variance
            else:
                importance_scores = np.ones(features.shape[1]) / features.shape[1]
            
            # Create importance dictionary
            importance_dict = {}
            for i, score in enumerate(importance_scores):
                importance_dict[f'feature_{i}'] = float(score)
            
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_feature_weights(self, properties: List[Dict]) -> Dict[str, float]:
        """Calculate feature weights based on data characteristics"""
        try:
            weights = {
                'location': 0.3,
                'price': 0.25,
                'amenities': 0.2,
                'text': 0.15,
                'categorical': 0.1
            }
            
            # Adjust weights based on data quality
            location_completeness = sum(1 for p in properties if p.get('neighborhood') or p.get('city')) / len(properties)
            price_completeness = sum(1 for p in properties if p.get('price')) / len(properties)
            amenity_completeness = sum(1 for p in properties if p.get('amenities')) / len(properties)
            
            weights['location'] *= location_completeness
            weights['price'] *= price_completeness
            weights['amenities'] *= amenity_completeness
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.warning(f"Feature weight calculation failed: {e}")
            return {'location': 0.3, 'price': 0.25, 'amenities': 0.2, 'text': 0.15, 'categorical': 0.1}
    
    def _generate_feature_names(self, properties: List[Dict]) -> List[str]:
        """Generate comprehensive feature names"""
        feature_names = []
        
        # Location features
        feature_names.append('location')
        
        # Numerical features
        feature_names.extend(['price', 'bedrooms', 'bathrooms', 'square_feet', 'lot_size'])
        
        # Categorical features
        feature_names.extend(['property_type', 'building_type', 'parking_type', 'pet_policy'])
        
        # Text features (first 100 most important)
        feature_names.extend([f'text_feature_{i}' for i in range(100)])
        
        # Amenity features
        feature_names.extend([f'amenity_{i}' for i in range(50)])
        
        return feature_names
    
    def fit(self, user_item_matrix: np.ndarray, property_data: List[Dict], 
            epochs: int = 100, batch_size: int = 128, validation_split: float = 0.2, 
            use_hyperparameter_optimization: bool = False, **kwargs):
        """
        Train the advanced content-based recommendation model.
        
        Args:
            user_item_matrix: User-item interaction matrix
            property_data: List of property dictionaries with features
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            use_hyperparameter_optimization: Whether to use hyperparameter optimization
            **kwargs: Additional training parameters
        """
        try:
            training_start_time = time.time()
            self.user_item_matrix = user_item_matrix
            
            # Extract property features with advanced processing
            self.logger.info("Extracting property features...")
            self.property_features = self.extract_property_features(property_data, use_advanced_processing=True)
            
            # Fit the advanced feature processor
            self.logger.info("Fitting feature processor...")
            self.feature_processor.fit(property_data)
            
            # Initialize or update user preference modeler
            if self.enable_user_modeling and self.user_preference_modeler:
                self.logger.info("Initializing user preference modeling...")
                self._initialize_user_preferences(user_item_matrix, property_data)
            
            # Hyperparameter optimization if requested
            if use_hyperparameter_optimization:
                self.logger.info("Performing hyperparameter optimization...")
                best_params = self._optimize_hyperparameters(user_item_matrix, property_data)
                self._update_model_with_best_params(best_params)
            
            # Prepare training data
            self.logger.info("Preparing training data...")
            X_train, y_train = self._prepare_advanced_training_data(user_item_matrix, property_data)
            
            if len(X_train[0]) == 0:
                raise ValueError("No training data generated")
            
            # Train the model
            self.logger.info(f"Starting training with {len(y_train)} samples")
            
            if self.use_neural_model and self.model:
                # Train neural network model
                training_metrics = self._train_neural_model(X_train, y_train, epochs, batch_size, validation_split)
            else:
                # Train using similarity-based approach
                training_metrics = self._train_similarity_model(user_item_matrix, property_data)
            
            # Compute comprehensive similarity matrices
            self.logger.info("Computing similarity matrices...")
            self._compute_similarity_matrices()
            
            # Calculate feature importance
            self.logger.info("Calculating feature importance...")
            self._calculate_global_feature_importance()
            
            # Evaluate model performance
            self.logger.info("Evaluating model performance...")
            evaluation_metrics = self._evaluate_model_performance(user_item_matrix, property_data)
            
            self.is_trained = True
            training_time = time.time() - training_start_time
            
            # Store comprehensive training metrics
            self.training_metrics = {
                **training_metrics,
                'training_time': training_time,
                'num_properties': len(property_data),
                'num_users': user_item_matrix.shape[0],
                'num_items': user_item_matrix.shape[1],
                'feature_dimensionality': self.property_features.combined_features.shape[1],
                'use_neural_model': self.use_neural_model,
                'use_advanced_processing': True
            }
            
            self.evaluation_metrics = evaluation_metrics
            
            # Log comprehensive training results
            self.logger.info(
                f"Advanced training completed in {training_time:.2f}s - "
                f"Final metrics: {training_metrics}"
            )
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"Advanced content-based model training failed: {e}")
            raise
    
    def _initialize_user_preferences(self, user_item_matrix: np.ndarray, property_data: List[Dict]):
        """Initialize user preferences from interaction history"""
        try:
            for user_id in range(user_item_matrix.shape[0]):
                for item_id in range(user_item_matrix.shape[1]):
                    rating = user_item_matrix[user_id, item_id]
                    if rating > 0 and item_id < len(property_data):
                        self.user_preference_modeler.update_user_profile(
                            user_id, property_data[item_id], rating
                        )
            
            self.logger.info(f"Initialized preferences for {len(self.user_preference_modeler.user_profiles)} users")
            
        except Exception as e:
            self.logger.warning(f"User preference initialization failed: {e}")
    
    def _optimize_hyperparameters(self, user_item_matrix: np.ndarray, property_data: List[Dict]) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search"""
        try:
            # Define hyperparameter search space
            param_grid = {
                'learning_rate': [0.001, 0.01, 0.1],
                'embedding_dim': [64, 128, 256],
                'reg_lambda': [1e-6, 1e-5, 1e-4],
                'batch_size': [64, 128, 256]
            }
            
            best_score = -np.inf
            best_params = {}
            
            # Simple grid search (can be replaced with more sophisticated methods)
            for lr in param_grid['learning_rate']:
                for embed_dim in param_grid['embedding_dim']:
                    for reg in param_grid['reg_lambda']:
                        for batch_size in param_grid['batch_size']:
                            try:
                                # Test this parameter combination
                                test_params = {
                                    'learning_rate': lr,
                                    'embedding_dim': embed_dim,
                                    'reg_lambda': reg,
                                    'batch_size': batch_size
                                }
                                
                                score = self._evaluate_hyperparameters(test_params, user_item_matrix, property_data)
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = test_params
                                    
                            except Exception as e:
                                self.logger.warning(f"Hyperparameter evaluation failed for {test_params}: {e}")
                                continue
            
            self.logger.info(f"Best hyperparameters found: {best_params} with score: {best_score}")
            return best_params
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return {}
    
    def _evaluate_hyperparameters(self, params: Dict[str, Any], user_item_matrix: np.ndarray, 
                                 property_data: List[Dict]) -> float:
        """Evaluate hyperparameters using cross-validation"""
        try:
            # Create a temporary model with these parameters
            temp_model = ContentBasedRecommender(
                embedding_dim=params.get('embedding_dim', self.embedding_dim),
                reg_lambda=params.get('reg_lambda', self.reg_lambda),
                learning_rate=params.get('learning_rate', self.learning_rate),
                feature_config=self.feature_config,
                similarity_config=self.similarity_config,
                use_neural_model=self.use_neural_model,
                enable_user_modeling=False,  # Disable for speed
                enable_caching=False  # Disable for speed
            )
            
            # Train with reduced epochs for speed
            temp_model.fit(
                user_item_matrix, 
                property_data, 
                epochs=10, 
                batch_size=params.get('batch_size', 128),
                validation_split=0.2,
                use_hyperparameter_optimization=False
            )
            
            # Evaluate performance
            evaluation_results = temp_model._evaluate_model_performance(user_item_matrix, property_data)
            
            # Return a combined score (can be customized)
            score = evaluation_results.get('accuracy', 0.0) * 0.4 + \
                   (1 - evaluation_results.get('mae', 1.0)) * 0.3 + \
                   evaluation_results.get('precision', 0.0) * 0.3
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter evaluation failed: {e}")
            return 0.0
    
    def _update_model_with_best_params(self, best_params: Dict[str, Any]):
        """Update model with best hyperparameters"""
        try:
            if not best_params:
                return
            
            # Update parameters
            self.learning_rate = best_params.get('learning_rate', self.learning_rate)
            self.embedding_dim = best_params.get('embedding_dim', self.embedding_dim)
            self.reg_lambda = best_params.get('reg_lambda', self.reg_lambda)
            
            # Rebuild model with new parameters
            if self.use_neural_model:
                self.model = self._build_neural_model()
            
            self.logger.info(f"Updated model with optimized hyperparameters: {best_params}")
            
        except Exception as e:
            self.logger.error(f"Model update with best params failed: {e}")
    
    def _prepare_advanced_training_data(self, user_item_matrix: np.ndarray, property_data: List[Dict]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Prepare advanced training data with multiple input types"""
        try:
            if not self.use_neural_model:
                # For non-neural models, use legacy preparation
                return self._prepare_training_data_legacy(user_item_matrix)
            
            # Prepare inputs for neural network
            location_inputs = []
            numerical_inputs = []
            amenity_inputs = []
            text_inputs = []
            categorical_inputs = []
            labels = []
            
            num_users, num_items = user_item_matrix.shape
            
            # Generate positive samples from existing interactions
            for user_id in range(num_users):
                for item_id in range(num_items):
                    rating = user_item_matrix[user_id, item_id]
                    
                    if rating > 0:  # Positive interaction
                        location_inputs.append(self.property_features.location_features[item_id])
                        
                        # Prepare numerical features (5 features)
                        numerical_feat = np.zeros(5)
                        if self.property_features.numerical_features.shape[1] >= 5:
                            numerical_feat = self.property_features.numerical_features[item_id][:5]
                        else:
                            numerical_feat[:self.property_features.numerical_features.shape[1]] = \
                                self.property_features.numerical_features[item_id]
                        numerical_inputs.append(numerical_feat)
                        
                        # Prepare amenity features
                        amenity_feat = np.zeros(self.amenity_vocab_size)
                        if self.property_features.amenity_features.shape[1] >= self.amenity_vocab_size:
                            amenity_feat = self.property_features.amenity_features[item_id][:self.amenity_vocab_size]
                        else:
                            amenity_feat[:self.property_features.amenity_features.shape[1]] = \
                                self.property_features.amenity_features[item_id]
                        amenity_inputs.append(amenity_feat)
                        
                        # Prepare text features
                        text_feat = np.zeros(self.feature_config.text_max_features)
                        if self.property_features.text_features.shape[1] >= self.feature_config.text_max_features:
                            text_feat = self.property_features.text_features[item_id][:self.feature_config.text_max_features]
                        else:
                            text_feat[:self.property_features.text_features.shape[1]] = \
                                self.property_features.text_features[item_id]
                        text_inputs.append(text_feat)
                        
                        # Prepare categorical features
                        categorical_feat = np.zeros(10)
                        if self.property_features.categorical_features.shape[1] >= 10:
                            categorical_feat = self.property_features.categorical_features[item_id][:10]
                        else:
                            categorical_feat[:self.property_features.categorical_features.shape[1]] = \
                                self.property_features.categorical_features[item_id]
                        categorical_inputs.append(categorical_feat)
                        
                        labels.append(1.0)
            
            # Generate negative samples
            num_positive = len(labels)
            num_negative = min(num_positive, num_users * num_items // 20)
            
            for _ in range(num_negative):
                user_id = np.random.randint(0, num_users)
                item_id = np.random.randint(0, num_items)
                
                if user_item_matrix[user_id, item_id] == 0:  # Unobserved interaction
                    location_inputs.append(self.property_features.location_features[item_id])
                    
                    # Prepare numerical features (5 features)
                    numerical_feat = np.zeros(5)
                    if self.property_features.numerical_features.shape[1] >= 5:
                        numerical_feat = self.property_features.numerical_features[item_id][:5]
                    else:
                        numerical_feat[:self.property_features.numerical_features.shape[1]] = \
                            self.property_features.numerical_features[item_id]
                    numerical_inputs.append(numerical_feat)
                    
                    # Prepare amenity features
                    amenity_feat = np.zeros(self.amenity_vocab_size)
                    if self.property_features.amenity_features.shape[1] >= self.amenity_vocab_size:
                        amenity_feat = self.property_features.amenity_features[item_id][:self.amenity_vocab_size]
                    else:
                        amenity_feat[:self.property_features.amenity_features.shape[1]] = \
                            self.property_features.amenity_features[item_id]
                    amenity_inputs.append(amenity_feat)
                    
                    # Prepare text features
                    text_feat = np.zeros(self.feature_config.text_max_features)
                    if self.property_features.text_features.shape[1] >= self.feature_config.text_max_features:
                        text_feat = self.property_features.text_features[item_id][:self.feature_config.text_max_features]
                    else:
                        text_feat[:self.property_features.text_features.shape[1]] = \
                            self.property_features.text_features[item_id]
                    text_inputs.append(text_feat)
                    
                    # Prepare categorical features
                    categorical_feat = np.zeros(10)
                    if self.property_features.categorical_features.shape[1] >= 10:
                        categorical_feat = self.property_features.categorical_features[item_id][:10]
                    else:
                        categorical_feat[:self.property_features.categorical_features.shape[1]] = \
                            self.property_features.categorical_features[item_id]
                    categorical_inputs.append(categorical_feat)
                    
                    labels.append(0.0)
            
            # Convert to numpy arrays
            X_train = [
                np.array(location_inputs),
                np.array(numerical_inputs),
                np.array(amenity_inputs),
                np.array(text_inputs),
                np.array(categorical_inputs)
            ]
            y_train = np.array(labels)
            
            self.logger.info(f"Prepared advanced training data: {len(y_train)} samples "
                           f"({num_positive} positive, {len(y_train) - num_positive} negative)")
            
            return X_train, y_train
            
        except Exception as e:
            self.logger.error(f"Advanced training data preparation failed: {e}")
            raise
    
    def _prepare_training_data_legacy(self, user_item_matrix: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Prepare legacy training data from user-item interactions and property features.
        
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
            
            self.logger.info(f"Prepared legacy training data: {len(y_train)} samples "
                           f"({num_positive} positive, {len(y_train) - num_positive} negative)")
            
            return X_train, y_train
            
        except Exception as e:
            self.logger.error(f"Legacy training data preparation failed: {e}")
            raise
    
    def _train_neural_model(self, X_train: List[np.ndarray], y_train: np.ndarray, 
                          epochs: int, batch_size: int, validation_split: float) -> Dict[str, Any]:
        """Train the neural network model"""
        try:
            # Define comprehensive callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=15, 
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=7, 
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_content_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train the model
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
            
            # Extract training metrics
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            final_mae = history.history['mae'][-1]
            final_auc = history.history['auc'][-1]
            
            val_loss = history.history['val_loss'][-1]
            val_accuracy = history.history['val_accuracy'][-1]
            val_mae = history.history['val_mae'][-1]
            val_auc = history.history['val_auc'][-1]
            
            return {
                'final_loss': float(final_loss),
                'final_accuracy': float(final_accuracy),
                'final_mae': float(final_mae),
                'final_auc': float(final_auc),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy),
                'val_mae': float(val_mae),
                'val_auc': float(val_auc),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(y_train),
                'convergence_epoch': len(history.history['loss'])
            }
            
        except Exception as e:
            self.logger.error(f"Neural model training failed: {e}")
            raise
    
    def _train_similarity_model(self, user_item_matrix: np.ndarray, property_data: List[Dict]) -> Dict[str, Any]:
        """Train using similarity-based approach"""
        try:
            # This is a non-neural approach that just computes similarities
            start_time = time.time()
            
            # Compute property similarities
            self.property_similarity_matrix = self.similarity_calculator.calculate_pairwise_similarity(
                self.property_features.combined_features, method='combined'
            )
            
            # Simple evaluation metrics
            num_interactions = np.sum(user_item_matrix > 0)
            sparsity = 1 - (num_interactions / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
            
            training_time = time.time() - start_time
            
            return {
                'training_time': training_time,
                'num_interactions': int(num_interactions),
                'sparsity': float(sparsity),
                'similarity_method': 'combined',
                'cache_stats': self.similarity_calculator.get_cache_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Similarity model training failed: {e}")
            raise
    
    def _compute_similarity_matrices(self):
        """Compute comprehensive similarity matrices"""
        try:
            # Compute different types of similarity matrices
            self.property_similarity_matrix = self.similarity_calculator.calculate_pairwise_similarity(
                self.property_features.combined_features, method='combined'
            )
            
            # Store individual similarity matrices for analysis
            self.cosine_similarity_matrix = self.similarity_calculator.calculate_pairwise_similarity(
                self.property_features.combined_features, method='cosine'
            )
            
            self.euclidean_similarity_matrix = self.similarity_calculator.calculate_pairwise_similarity(
                self.property_features.combined_features, method='euclidean'
            )
            
            self.logger.info("Computed all similarity matrices")
            
        except Exception as e:
            self.logger.error(f"Similarity matrix computation failed: {e}")
            raise
    
    def _calculate_global_feature_importance(self):
        """Calculate global feature importance scores"""
        try:
            if self.property_features.combined_features.size == 0:
                return
            
            # Calculate variance-based importance
            feature_variances = np.var(self.property_features.combined_features, axis=0)
            
            # Calculate mutual information if we have labels
            if hasattr(self, 'user_item_matrix') and self.user_item_matrix is not None:
                # Create pseudo-labels based on popularity
                item_popularity = np.sum(self.user_item_matrix > 0, axis=0)
                
                # Calculate correlation between features and popularity
                feature_correlations = []
                for i in range(self.property_features.combined_features.shape[1]):
                    if len(item_popularity) == self.property_features.combined_features.shape[0]:
                        corr = np.corrcoef(self.property_features.combined_features[:, i], item_popularity)[0, 1]
                        feature_correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                    else:
                        feature_correlations.append(0.0)
                
                # Combine variance and correlation for final importance
                normalized_variances = feature_variances / np.sum(feature_variances) if np.sum(feature_variances) > 0 else feature_variances
                normalized_correlations = np.array(feature_correlations) / np.sum(feature_correlations) if np.sum(feature_correlations) > 0 else np.array(feature_correlations)
                
                combined_importance = 0.6 * normalized_variances + 0.4 * normalized_correlations
            else:
                combined_importance = feature_variances / np.sum(feature_variances) if np.sum(feature_variances) > 0 else feature_variances
            
            # Store feature importance
            self.feature_importance_scores = {}
            for i, importance in enumerate(combined_importance):
                if i < len(self.property_features.feature_names):
                    feature_name = self.property_features.feature_names[i]
                    self.feature_importance_scores[feature_name] = float(importance)
            
            self.logger.info(f"Calculated importance for {len(self.feature_importance_scores)} features")
            
        except Exception as e:
            self.logger.error(f"Global feature importance calculation failed: {e}")
    
    def _evaluate_model_performance(self, user_item_matrix: np.ndarray, property_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate comprehensive model performance"""
        try:
            evaluation_results = {}
            
            # Basic statistics
            evaluation_results['num_users'] = user_item_matrix.shape[0]
            evaluation_results['num_items'] = user_item_matrix.shape[1]
            evaluation_results['num_interactions'] = int(np.sum(user_item_matrix > 0))
            evaluation_results['sparsity'] = 1 - (evaluation_results['num_interactions'] / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
            
            # Feature statistics
            evaluation_results['feature_dimensionality'] = self.property_features.combined_features.shape[1]
            evaluation_results['feature_density'] = float(np.mean(self.property_features.combined_features != 0))
            
            # Similarity statistics
            if hasattr(self, 'property_similarity_matrix') and self.property_similarity_matrix is not None:
                evaluation_results['avg_similarity'] = float(np.mean(self.property_similarity_matrix))
                evaluation_results['similarity_std'] = float(np.std(self.property_similarity_matrix))
            
            # Model-specific metrics
            if self.use_neural_model and self.model and self.is_trained:
                # Neural model evaluation
                test_sample_size = min(1000, evaluation_results['num_interactions'])
                test_users = np.random.choice(user_item_matrix.shape[0], test_sample_size, replace=True)
                test_items = np.random.choice(user_item_matrix.shape[1], test_sample_size, replace=True)
                
                try:
                    test_predictions = self.predict(test_users[0], test_items[:10].tolist())
                    if len(test_predictions) > 0:
                        evaluation_results['prediction_mean'] = float(np.mean(test_predictions))
                        evaluation_results['prediction_std'] = float(np.std(test_predictions))
                except Exception as e:
                    self.logger.warning(f"Neural model evaluation failed: {e}")
            
            # Cache statistics
            if hasattr(self, 'similarity_calculator'):
                evaluation_results['cache_stats'] = self.similarity_calculator.get_cache_stats()
            
            # User preference statistics
            if self.enable_user_modeling and self.user_preference_modeler:
                evaluation_results['num_user_profiles'] = len(self.user_preference_modeler.user_profiles)
                evaluation_results['avg_user_interactions'] = float(np.mean([len(profile.interaction_history) for profile in self.user_preference_modeler.user_profiles.values()])) if self.user_preference_modeler.user_profiles else 0
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Model performance evaluation failed: {e}")
            return {}
    
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        Predict user preferences for given property items using advanced methods.
        
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
            
            # Use user preference modeling if available
            if self.enable_user_modeling and self.user_preference_modeler:
                preference_predictions = []
                for item_id in valid_item_ids:
                    # Get user preference prediction
                    user_pref = self.user_preference_modeler.predict_user_preference(user_id, {})
                    preference_predictions.append(user_pref)
                
                # Combine with model predictions if neural model is used
                if self.use_neural_model and self.model:
                    model_predictions = self._predict_neural_model(valid_item_ids)
                    
                    # Combine predictions with weighted average
                    combined_predictions = []
                    for i, item_id in enumerate(valid_item_ids):
                        user_weight = 0.6 if user_id in self.user_preference_modeler.user_profiles else 0.2
                        model_weight = 1.0 - user_weight
                        
                        if i < len(model_predictions) and i < len(preference_predictions):
                            combined_pred = user_weight * preference_predictions[i] + model_weight * model_predictions[i]
                        else:
                            combined_pred = preference_predictions[i] if i < len(preference_predictions) else 0.5
                        
                        combined_predictions.append(combined_pred)
                    
                    return np.array(combined_predictions)
                else:
                    return np.array(preference_predictions)
            
            # Fall back to neural model predictions
            if self.use_neural_model and self.model:
                return self._predict_neural_model(valid_item_ids)
            
            # Fall back to similarity-based predictions
            return self._predict_similarity_based(user_id, valid_item_ids)
            
        except Exception as e:
            self.logger.error(f"Advanced prediction failed: {e}")
            return np.array([])
    
    def _predict_neural_model(self, item_ids: List[int]) -> np.ndarray:
        """Make predictions using neural network model"""
        try:
            # Prepare input features for neural model
            location_inputs = self.property_features.location_features[item_ids]
            
            # Prepare numerical features (5 features)
            numerical_inputs = []
            for item_id in item_ids:
                numerical_feat = np.zeros(5)
                if self.property_features.numerical_features.shape[1] >= 5:
                    numerical_feat = self.property_features.numerical_features[item_id][:5]
                else:
                    numerical_feat[:self.property_features.numerical_features.shape[1]] = \
                        self.property_features.numerical_features[item_id]
                numerical_inputs.append(numerical_feat)
            numerical_inputs = np.array(numerical_inputs)
            
            # Prepare amenity features
            amenity_inputs = []
            for item_id in item_ids:
                amenity_feat = np.zeros(self.amenity_vocab_size)
                if self.property_features.amenity_features.shape[1] >= self.amenity_vocab_size:
                    amenity_feat = self.property_features.amenity_features[item_id][:self.amenity_vocab_size]
                else:
                    amenity_feat[:self.property_features.amenity_features.shape[1]] = \
                        self.property_features.amenity_features[item_id]
                amenity_inputs.append(amenity_feat)
            amenity_inputs = np.array(amenity_inputs)
            
            # Prepare text features
            text_inputs = []
            for item_id in item_ids:
                text_feat = np.zeros(self.feature_config.text_max_features)
                if self.property_features.text_features.shape[1] >= self.feature_config.text_max_features:
                    text_feat = self.property_features.text_features[item_id][:self.feature_config.text_max_features]
                else:
                    text_feat[:self.property_features.text_features.shape[1]] = \
                        self.property_features.text_features[item_id]
                text_inputs.append(text_feat)
            text_inputs = np.array(text_inputs)
            
            # Prepare categorical features
            categorical_inputs = []
            for item_id in item_ids:
                categorical_feat = np.zeros(10)
                if self.property_features.categorical_features.shape[1] >= 10:
                    categorical_feat = self.property_features.categorical_features[item_id][:10]
                else:
                    categorical_feat[:self.property_features.categorical_features.shape[1]] = \
                        self.property_features.categorical_features[item_id]
                categorical_inputs.append(categorical_feat)
            categorical_inputs = np.array(categorical_inputs)
            
            # Make predictions
            predictions = self.model.predict([
                location_inputs,
                numerical_inputs,
                amenity_inputs,
                text_inputs,
                categorical_inputs
            ], verbose=0)
            
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"Neural model prediction failed: {e}")
            return np.array([0.5] * len(item_ids))
    
    def _predict_similarity_based(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Make predictions using similarity-based approach"""
        try:
            if not hasattr(self, 'property_similarity_matrix') or self.property_similarity_matrix is None:
                return np.array([0.5] * len(item_ids))
            
            predictions = []
            
            # Get user's interaction history
            if self.user_item_matrix is not None and user_id < self.user_item_matrix.shape[0]:
                user_interactions = np.where(self.user_item_matrix[user_id] > 0)[0]
                
                for item_id in item_ids:
                    if len(user_interactions) > 0:
                        # Calculate weighted average based on similarity to user's liked items
                        similarities = self.property_similarity_matrix[item_id][user_interactions]
                        ratings = self.user_item_matrix[user_id][user_interactions]
                        
                        # Weighted average
                        if np.sum(similarities) > 0:
                            prediction = np.sum(similarities * ratings) / np.sum(similarities)
                        else:
                            prediction = 0.5
                    else:
                        prediction = 0.5
                    
                    predictions.append(prediction)
            else:
                predictions = [0.5] * len(item_ids)
            
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"Similarity-based prediction failed: {e}")
            return np.array([0.5] * len(item_ids))
    
    def recommend(self, user_id: int, num_recommendations: int = 10, 
                  exclude_seen: bool = True, use_diversity: bool = True, 
                  diversity_weight: float = 0.1) -> List[RecommendationResult]:
        """
        Generate advanced content-based recommendations for a user.
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude previously seen items
            use_diversity: Whether to promote diversity in recommendations
            diversity_weight: Weight for diversity in final ranking
            
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
                # Calculate advanced confidence score
                confidence = self._calculate_advanced_confidence(user_id, item_id, prediction)
                
                # Generate detailed explanation
                explanation = self._generate_advanced_explanation(user_id, item_id, prediction)
                
                result = RecommendationResult(
                    item_id=item_id,
                    predicted_rating=float(prediction),
                    confidence_score=confidence,
                    explanation=explanation
                )
                recommendations.append(result)
            
            # Apply diversity if requested
            if use_diversity:
                recommendations = self._apply_diversity_reranking(recommendations, diversity_weight)
            
            # Sort by predicted rating and return top-N
            recommendations.sort(key=lambda x: x.predicted_rating, reverse=True)
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            self.logger.error(f"Advanced content-based recommendation failed: {e}")
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