"""
Advanced feature engineering pipeline for rental ML system.

This module provides comprehensive feature engineering capabilities including:
- Property feature extraction and transformation
- User behavior analysis and feature creation
- Temporal feature engineering
- Geographical feature processing
- Text feature extraction and NLP
- Feature selection and dimensionality reduction
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, TargetEncoder
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.cluster import KMeans
import geopy.distance
from geopy.geocoders import Nominatim

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Property features
    include_price_features: bool = True
    include_location_features: bool = True
    include_amenity_features: bool = True
    include_temporal_features: bool = True
    
    # User features
    include_user_behavior: bool = True
    include_user_preferences: bool = True
    include_interaction_history: bool = True
    
    # Advanced features
    include_similarity_features: bool = True
    include_market_features: bool = True
    include_nlp_features: bool = True
    
    # Feature selection
    max_features: Optional[int] = None
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_test", "pca"
    
    # Text processing
    max_text_features: int = 500
    min_df: int = 2
    max_df: float = 0.8
    
    # Scaling
    scaling_method: str = "standard"  # "standard", "minmax", "robust"


@dataclass
class ProcessedFeatures:
    """Container for processed features"""
    property_features: np.ndarray
    user_features: np.ndarray
    interaction_features: Optional[np.ndarray]
    feature_names: List[str]
    feature_metadata: Dict[str, Any]
    scalers: Dict[str, Any]
    encoders: Dict[str, Any]


class FeatureEngineer:
    """
    Advanced feature engineering pipeline.
    
    This class handles:
    - Property feature extraction and engineering
    - User behavior analysis and feature creation
    - Temporal and geographical feature processing
    - Text feature extraction using NLP
    - Feature selection and dimensionality reduction
    - Feature transformation and scaling
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self._init_processors()
        
        # Feature storage
        self.feature_metadata = {}
        self.is_fitted = False
        
    def _init_processors(self):
        """Initialize feature processors"""
        # Scalers
        if self.config.scaling_method == "standard":
            self.property_scaler = StandardScaler()
            self.user_scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.property_scaler = MinMaxScaler()
            self.user_scaler = MinMaxScaler()
        elif self.config.scaling_method == "robust":
            self.property_scaler = RobustScaler()
            self.user_scaler = RobustScaler()
        
        # Encoders
        self.location_encoder = LabelEncoder()
        self.property_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Text processors
        self.amenity_vectorizer = TfidfVectorizer(
            max_features=self.config.max_text_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.description_vectorizer = TfidfVectorizer(
            max_features=self.config.max_text_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Feature selection
        if self.config.feature_selection_method == "mutual_info":
            self.feature_selector = SelectKBest(
                score_func=mutual_info_regression,
                k=self.config.max_features or 'all'
            )
        elif self.config.feature_selection_method == "f_test":
            self.feature_selector = SelectKBest(
                score_func=f_regression,
                k=self.config.max_features or 'all'
            )
        elif self.config.feature_selection_method == "pca":
            self.feature_selector = PCA(
                n_components=self.config.max_features or 0.95
            )
        
        # Clustering for similarity features
        self.property_clusterer = KMeans(n_clusters=50, random_state=42)
        self.user_clusterer = KMeans(n_clusters=20, random_state=42)
        
        # NLP components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            self.logger.warning(f"NLTK setup failed: {e}")
            self.lemmatizer = None
            self.stop_words = set()
    
    def fit_transform(self, 
                     properties_data: List[Dict],
                     users_data: List[Dict],
                     interactions_data: List[Dict],
                     user_item_matrix: np.ndarray) -> ProcessedFeatures:
        """
        Fit feature processors and transform data.
        
        Args:
            properties_data: Raw property data
            users_data: Raw user data
            interactions_data: Raw interaction data
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Processed features
        """
        try:
            self.logger.info("Starting feature engineering...")
            
            # Convert to DataFrames for easier processing
            properties_df = pd.DataFrame(properties_data)
            users_df = pd.DataFrame(users_data)
            interactions_df = pd.DataFrame(interactions_data)
            
            # Engineer property features
            property_features = self._engineer_property_features(
                properties_df, interactions_df, user_item_matrix
            )
            
            # Engineer user features
            user_features = self._engineer_user_features(
                users_df, interactions_df, user_item_matrix
            )
            
            # Engineer interaction features if requested
            interaction_features = None
            if self.config.include_interaction_history:
                interaction_features = self._engineer_interaction_features(
                    interactions_df, user_item_matrix
                )
            
            # Combine and select features
            all_feature_names = (
                self._get_property_feature_names() +
                self._get_user_feature_names() +
                (self._get_interaction_feature_names() if interaction_features is not None else [])
            )
            
            # Store metadata
            self.feature_metadata = {
                'num_properties': len(properties_data),
                'num_users': len(users_data),
                'num_interactions': len(interactions_data),
                'property_feature_dim': property_features.shape[1],
                'user_feature_dim': user_features.shape[1],
                'interaction_feature_dim': interaction_features.shape[1] if interaction_features is not None else 0,
                'total_features': len(all_feature_names),
                'config': self.config
            }
            
            self.is_fitted = True
            
            # Create result
            result = ProcessedFeatures(
                property_features=property_features,
                user_features=user_features,
                interaction_features=interaction_features,
                feature_names=all_feature_names,
                feature_metadata=self.feature_metadata,
                scalers={
                    'property_scaler': self.property_scaler,
                    'user_scaler': self.user_scaler
                },
                encoders={
                    'location_encoder': self.location_encoder,
                    'property_type_encoder': self.property_type_encoder,
                    'amenity_vectorizer': self.amenity_vectorizer,
                    'description_vectorizer': self.description_vectorizer
                }
            )
            
            self.logger.info(
                f"Feature engineering completed: "
                f"{property_features.shape[1]} property features, "
                f"{user_features.shape[1]} user features"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _engineer_property_features(self, 
                                  properties_df: pd.DataFrame,
                                  interactions_df: pd.DataFrame,
                                  user_item_matrix: np.ndarray) -> np.ndarray:
        """Engineer property features"""
        try:
            features_list = []
            
            # Basic numerical features
            if self.config.include_price_features:
                price_features = self._extract_price_features(properties_df)
                features_list.append(price_features)
            
            # Location features
            if self.config.include_location_features:
                location_features = self._extract_location_features(properties_df)
                features_list.append(location_features)
            
            # Amenity features
            if self.config.include_amenity_features:
                amenity_features = self._extract_amenity_features(properties_df)
                features_list.append(amenity_features)
            
            # Temporal features
            if self.config.include_temporal_features:
                temporal_features = self._extract_temporal_features(properties_df)
                features_list.append(temporal_features)
            
            # Market features
            if self.config.include_market_features:
                market_features = self._extract_market_features(properties_df, interactions_df)
                features_list.append(market_features)
            
            # NLP features
            if self.config.include_nlp_features:
                nlp_features = self._extract_nlp_features(properties_df)
                features_list.append(nlp_features)
            
            # Similarity features
            if self.config.include_similarity_features:
                similarity_features = self._extract_property_similarity_features(
                    properties_df, user_item_matrix
                )
                features_list.append(similarity_features)
            
            # Combine all features
            property_features = np.concatenate(features_list, axis=1)
            
            # Scale features
            property_features = self.property_scaler.fit_transform(property_features)
            
            return property_features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Property feature engineering failed: {e}")
            raise
    
    def _extract_price_features(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Extract price-related features"""
        features = []
        
        # Basic price features
        features.append(properties_df['price'].fillna(0).values.reshape(-1, 1))
        
        # Price per square foot
        price_per_sqft = properties_df['price'] / properties_df['square_feet'].replace(0, np.nan)
        features.append(price_per_sqft.fillna(0).values.reshape(-1, 1))
        
        # Price per bedroom
        price_per_bedroom = properties_df['price'] / properties_df['bedrooms'].replace(0, np.nan)
        features.append(price_per_bedroom.fillna(0).values.reshape(-1, 1))
        
        # Price percentiles within location
        location_price_percentiles = []
        for location in properties_df['location'].unique():
            location_properties = properties_df[properties_df['location'] == location]['price']
            percentiles = location_properties.rank(pct=True)
            location_price_percentiles.extend(percentiles.values)
        
        features.append(np.array(location_price_percentiles).reshape(-1, 1))
        
        # Log price (for skewed distributions)
        log_price = np.log1p(properties_df['price'].fillna(0))
        features.append(log_price.values.reshape(-1, 1))
        
        return np.concatenate(features, axis=1)
    
    def _extract_location_features(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Extract location-based features"""
        features = []
        
        # Encoded location
        locations = properties_df['location'].fillna('unknown')
        location_encoded = self.location_encoder.fit_transform(locations)
        features.append(location_encoded.reshape(-1, 1))
        
        # Location frequency (popularity)
        location_counts = locations.value_counts()
        location_popularity = locations.map(location_counts).values
        features.append(location_popularity.reshape(-1, 1))
        
        # Extract city, state if available
        if 'city' in properties_df.columns:
            cities = properties_df['city'].fillna('unknown')
            city_encoder = LabelEncoder()
            city_encoded = city_encoder.fit_transform(cities)
            features.append(city_encoded.reshape(-1, 1))
        
        if 'state' in properties_df.columns:
            states = properties_df['state'].fillna('unknown')
            state_encoder = LabelEncoder()
            state_encoded = state_encoder.fit_transform(states)
            features.append(state_encoded.reshape(-1, 1))
        
        # Geographic clustering (if coordinates available)
        if 'latitude' in properties_df.columns and 'longitude' in properties_df.columns:
            coords = properties_df[['latitude', 'longitude']].fillna(0)
            geo_clusters = KMeans(n_clusters=20, random_state=42).fit_predict(coords)
            features.append(geo_clusters.reshape(-1, 1))
        
        return np.concatenate(features, axis=1)
    
    def _extract_amenity_features(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Extract amenity-based features"""
        # Process amenities text
        amenity_texts = []
        for amenities in properties_df['amenities']:
            if isinstance(amenities, list):
                amenity_text = ' '.join(amenities)
            else:
                amenity_text = str(amenities) if amenities else ''
            amenity_texts.append(amenity_text)
        
        # TF-IDF vectorization
        amenity_features = self.amenity_vectorizer.fit_transform(amenity_texts)
        
        # Additional amenity features
        amenity_count = np.array([len(a) if isinstance(a, list) else 0 
                                for a in properties_df['amenities']]).reshape(-1, 1)
        
        # Specific amenity indicators
        popular_amenities = ['parking', 'gym', 'pool', 'laundry', 'pet_friendly', 'balcony']
        amenity_indicators = []
        
        for amenity in popular_amenities:
            has_amenity = np.array([
                1 if isinstance(amenities, list) and amenity in amenities else 0
                for amenities in properties_df['amenities']
            ]).reshape(-1, 1)
            amenity_indicators.append(has_amenity)
        
        # Combine features
        combined_features = [
            amenity_features.toarray(),
            amenity_count
        ]
        combined_features.extend(amenity_indicators)
        
        return np.concatenate(combined_features, axis=1)
    
    def _extract_temporal_features(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Extract temporal features"""
        features = []
        
        # Days since scraped
        if 'scraped_at' in properties_df.columns:
            scraped_dates = pd.to_datetime(properties_df['scraped_at'])
            days_since_scraped = (datetime.now() - scraped_dates).dt.days
            features.append(days_since_scraped.fillna(0).values.reshape(-1, 1))
            
            # Scraped day of week
            day_of_week = scraped_dates.dt.dayofweek
            features.append(day_of_week.fillna(0).values.reshape(-1, 1))
            
            # Scraped month
            month = scraped_dates.dt.month
            features.append(month.fillna(0).values.reshape(-1, 1))
        
        # Property age (if available)
        if 'year_built' in properties_df.columns:
            property_age = datetime.now().year - properties_df['year_built'].fillna(0)
            features.append(property_age.values.reshape(-1, 1))
        
        if not features:
            # Return dummy feature if no temporal data
            features.append(np.zeros((len(properties_df), 1)))
        
        return np.concatenate(features, axis=1)
    
    def _extract_market_features(self, 
                               properties_df: pd.DataFrame,
                               interactions_df: pd.DataFrame) -> np.ndarray:
        """Extract market-based features"""
        features = []
        
        # Property popularity (interaction count)
        property_interactions = interactions_df.groupby('property_id').size()
        popularity = properties_df['id'].map(property_interactions).fillna(0)
        features.append(popularity.values.reshape(-1, 1))
        
        # Average rating (if available)
        if 'rating' in interactions_df.columns:
            avg_ratings = interactions_df.groupby('property_id')['rating'].mean()
            property_ratings = properties_df['id'].map(avg_ratings).fillna(0)
            features.append(property_ratings.values.reshape(-1, 1))
        
        # Market competition (similar properties in same location)
        location_competition = properties_df.groupby('location').size()
        competition = properties_df['location'].map(location_competition)
        features.append(competition.values.reshape(-1, 1))
        
        # Price rank within location
        properties_df['location_price_rank'] = properties_df.groupby('location')['price'].rank()
        features.append(properties_df['location_price_rank'].fillna(0).values.reshape(-1, 1))
        
        return np.concatenate(features, axis=1)
    
    def _extract_nlp_features(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Extract NLP-based features from text fields"""
        features = []
        
        # Process descriptions
        if 'description' in properties_df.columns:
            descriptions = properties_df['description'].fillna('')
            
            # TF-IDF features
            description_features = self.description_vectorizer.fit_transform(descriptions)
            features.append(description_features.toarray())
            
            # Text statistics
            text_stats = []
            for desc in descriptions:
                stats = [
                    len(desc),  # Character count
                    len(desc.split()),  # Word count
                    desc.count('!'),  # Exclamation marks
                    desc.count('?'),  # Question marks
                    len([w for w in desc.split() if w.isupper()]),  # Uppercase words
                ]
                text_stats.append(stats)
            
            features.append(np.array(text_stats))
        
        # Process titles
        if 'title' in properties_df.columns:
            titles = properties_df['title'].fillna('')
            
            # Title length
            title_lengths = np.array([len(title) for title in titles]).reshape(-1, 1)
            features.append(title_lengths)
            
            # Title word count
            title_word_counts = np.array([len(title.split()) for title in titles]).reshape(-1, 1)
            features.append(title_word_counts)
        
        if not features:
            # Return dummy feature if no text data
            features.append(np.zeros((len(properties_df), 1)))
        
        return np.concatenate(features, axis=1)
    
    def _extract_property_similarity_features(self, 
                                            properties_df: pd.DataFrame,
                                            user_item_matrix: np.ndarray) -> np.ndarray:
        """Extract property similarity features"""
        # Basic features for clustering
        basic_features = []
        
        # Numerical features
        numerical_cols = ['price', 'bedrooms', 'bathrooms', 'square_feet']
        for col in numerical_cols:
            if col in properties_df.columns:
                basic_features.append(properties_df[col].fillna(0).values.reshape(-1, 1))
        
        if basic_features:
            cluster_input = np.concatenate(basic_features, axis=1)
            
            # Fit clustering model
            property_clusters = self.property_clusterer.fit_predict(cluster_input)
            
            # Distance to cluster center
            cluster_distances = self.property_clusterer.transform(cluster_input)
            min_distances = np.min(cluster_distances, axis=1).reshape(-1, 1)
            
            return np.concatenate([
                property_clusters.reshape(-1, 1),
                min_distances
            ], axis=1)
        else:
            return np.zeros((len(properties_df), 2))
    
    def _engineer_user_features(self, 
                              users_df: pd.DataFrame,
                              interactions_df: pd.DataFrame,
                              user_item_matrix: np.ndarray) -> np.ndarray:
        """Engineer user features"""
        try:
            features_list = []
            
            # User preferences
            if self.config.include_user_preferences:
                preference_features = self._extract_user_preferences(users_df)
                features_list.append(preference_features)
            
            # User behavior
            if self.config.include_user_behavior:
                behavior_features = self._extract_user_behavior(users_df, interactions_df)
                features_list.append(behavior_features)
            
            # Interaction history features
            if self.config.include_interaction_history:
                history_features = self._extract_interaction_history_features(
                    users_df, interactions_df, user_item_matrix
                )
                features_list.append(history_features)
            
            # User similarity features
            if self.config.include_similarity_features:
                similarity_features = self._extract_user_similarity_features(
                    users_df, user_item_matrix
                )
                features_list.append(similarity_features)
            
            # Combine all features
            user_features = np.concatenate(features_list, axis=1)
            
            # Scale features
            user_features = self.user_scaler.fit_transform(user_features)
            
            return user_features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"User feature engineering failed: {e}")
            raise
    
    def _extract_user_preferences(self, users_df: pd.DataFrame) -> np.ndarray:
        """Extract user preference features"""
        features = []
        
        # Price preferences
        preference_cols = ['min_price', 'max_price', 'min_bedrooms', 'max_bedrooms',
                          'min_bathrooms', 'max_bathrooms']
        
        for col in preference_cols:
            if col in users_df.columns:
                values = users_df[col].fillna(0).values.reshape(-1, 1)
                features.append(values)
        
        # Price range
        if 'min_price' in users_df.columns and 'max_price' in users_df.columns:
            price_range = (users_df['max_price'] - users_df['min_price']).fillna(0)
            features.append(price_range.values.reshape(-1, 1))
        
        # Preferred locations count
        if 'preferred_locations' in users_df.columns:
            location_counts = np.array([
                len(locs) if isinstance(locs, list) else 0
                for locs in users_df['preferred_locations']
            ]).reshape(-1, 1)
            features.append(location_counts)
        
        # Required amenities count
        if 'required_amenities' in users_df.columns:
            amenity_counts = np.array([
                len(amenities) if isinstance(amenities, list) else 0
                for amenities in users_df['required_amenities']
            ]).reshape(-1, 1)
            features.append(amenity_counts)
        
        if not features:
            features.append(np.zeros((len(users_df), 1)))
        
        return np.concatenate(features, axis=1)
    
    def _extract_user_behavior(self, 
                             users_df: pd.DataFrame,
                             interactions_df: pd.DataFrame) -> np.ndarray:
        """Extract user behavior features"""
        features = []
        
        # Interaction statistics
        user_interactions = interactions_df.groupby('user_id').agg({
            'property_id': 'count',  # Total interactions
            'timestamp': ['min', 'max'],  # First and last interaction
            'interaction_type': lambda x: x.nunique(),  # Variety of interactions
            'duration_seconds': 'mean'  # Average interaction duration
        }).fillna(0)
        
        user_interactions.columns = ['interaction_count', 'first_interaction', 
                                   'last_interaction', 'interaction_variety', 
                                   'avg_duration']
        
        # Map to users
        for col in user_interactions.columns:
            if col in ['first_interaction', 'last_interaction']:
                # Convert to days since
                if not user_interactions[col].empty:
                    days_since = (datetime.now() - pd.to_datetime(user_interactions[col])).dt.days
                    values = users_df['id'].map(days_since).fillna(0).values.reshape(-1, 1)
                else:
                    values = np.zeros((len(users_df), 1))
            else:
                values = users_df['id'].map(user_interactions[col]).fillna(0).values.reshape(-1, 1)
            features.append(values)
        
        # Interaction type distribution
        interaction_types = ['view', 'like', 'inquiry', 'bookmark']
        for int_type in interaction_types:
            type_counts = interactions_df[interactions_df['interaction_type'] == int_type].groupby('user_id').size()
            values = users_df['id'].map(type_counts).fillna(0).values.reshape(-1, 1)
            features.append(values)
        
        return np.concatenate(features, axis=1)
    
    def _extract_interaction_history_features(self, 
                                            users_df: pd.DataFrame,
                                            interactions_df: pd.DataFrame,
                                            user_item_matrix: np.ndarray) -> np.ndarray:
        """Extract interaction history features"""
        features = []
        
        # User activity level
        user_activity = np.sum(user_item_matrix > 0, axis=1).reshape(-1, 1)
        features.append(user_activity)
        
        # Average interaction strength
        avg_interaction = np.mean(user_item_matrix, axis=1).reshape(-1, 1)
        features.append(avg_interaction)
        
        # Interaction variance (consistency)
        interaction_var = np.var(user_item_matrix, axis=1).reshape(-1, 1)
        features.append(interaction_var)
        
        # Recency of interactions
        if 'timestamp' in interactions_df.columns:
            recent_interactions = interactions_df.groupby('user_id')['timestamp'].max()
            days_since_last = (datetime.now() - pd.to_datetime(recent_interactions)).dt.days
            values = users_df['id'].map(days_since_last).fillna(999).values.reshape(-1, 1)
            features.append(values)
        
        return np.concatenate(features, axis=1)
    
    def _extract_user_similarity_features(self, 
                                        users_df: pd.DataFrame,
                                        user_item_matrix: np.ndarray) -> np.ndarray:
        """Extract user similarity features"""
        # Use interaction patterns for clustering
        if user_item_matrix.size > 0:
            # Reduce dimensionality for clustering
            svd = TruncatedSVD(n_components=min(50, user_item_matrix.shape[1]), random_state=42)
            reduced_interactions = svd.fit_transform(user_item_matrix)
            
            # Cluster users
            user_clusters = self.user_clusterer.fit_predict(reduced_interactions)
            
            # Distance to cluster center
            cluster_distances = self.user_clusterer.transform(reduced_interactions)
            min_distances = np.min(cluster_distances, axis=1).reshape(-1, 1)
            
            return np.concatenate([
                user_clusters.reshape(-1, 1),
                min_distances
            ], axis=1)
        else:
            return np.zeros((len(users_df), 2))
    
    def _engineer_interaction_features(self, 
                                     interactions_df: pd.DataFrame,
                                     user_item_matrix: np.ndarray) -> np.ndarray:
        """Engineer interaction-based features"""
        # This would create features for specific user-item pairs
        # For now, return summary statistics
        features = []
        
        # Global interaction statistics
        interaction_stats = [
            len(interactions_df),  # Total interactions
            interactions_df['user_id'].nunique(),  # Unique users
            interactions_df['property_id'].nunique(),  # Unique properties
            np.sum(user_item_matrix > 0),  # Non-zero interactions
            np.mean(user_item_matrix[user_item_matrix > 0])  # Average interaction strength
        ]
        
        return np.array(interaction_stats).reshape(1, -1)
    
    def _get_property_feature_names(self) -> List[str]:
        """Get property feature names"""
        names = []
        
        if self.config.include_price_features:
            names.extend(['price', 'price_per_sqft', 'price_per_bedroom', 
                         'location_price_percentile', 'log_price'])
        
        if self.config.include_location_features:
            names.extend(['location_encoded', 'location_popularity'])
        
        if self.config.include_amenity_features:
            names.extend([f'amenity_tfidf_{i}' for i in range(self.config.max_text_features)])
            names.extend(['amenity_count'])
            names.extend(['has_parking', 'has_gym', 'has_pool', 'has_laundry', 
                         'has_pet_friendly', 'has_balcony'])
        
        return names
    
    def _get_user_feature_names(self) -> List[str]:
        """Get user feature names"""
        names = []
        
        if self.config.include_user_preferences:
            names.extend(['min_price', 'max_price', 'min_bedrooms', 'max_bedrooms',
                         'min_bathrooms', 'max_bathrooms', 'price_range',
                         'preferred_locations_count', 'required_amenities_count'])
        
        if self.config.include_user_behavior:
            names.extend(['interaction_count', 'days_since_first', 'days_since_last',
                         'interaction_variety', 'avg_duration'])
            names.extend(['view_count', 'like_count', 'inquiry_count', 'bookmark_count'])
        
        return names
    
    def _get_interaction_feature_names(self) -> List[str]:
        """Get interaction feature names"""
        return ['total_interactions', 'unique_users', 'unique_properties',
                'nonzero_interactions', 'avg_interaction_strength']
    
    def transform(self, 
                 properties_data: List[Dict],
                 users_data: List[Dict],
                 interactions_data: List[Dict],
                 user_item_matrix: np.ndarray) -> ProcessedFeatures:
        """Transform new data using fitted processors"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # Similar to fit_transform but using existing processors
        # Implementation would reuse the extraction methods with transform instead of fit_transform
        pass
    
    def save_processors(self, save_path: str):
        """Save fitted processors"""
        save_data = {
            'config': self.config,
            'property_scaler': self.property_scaler,
            'user_scaler': self.user_scaler,
            'location_encoder': self.location_encoder,
            'property_type_encoder': self.property_type_encoder,
            'amenity_vectorizer': self.amenity_vectorizer,
            'description_vectorizer': self.description_vectorizer,
            'property_clusterer': self.property_clusterer,
            'user_clusterer': self.user_clusterer,
            'feature_metadata': self.feature_metadata,
            'is_fitted': self.is_fitted
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Feature processors saved to {save_path}")
    
    def load_processors(self, load_path: str):
        """Load fitted processors"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore processors
        for key, value in save_data.items():
            setattr(self, key, value)
        
        self.logger.info(f"Feature processors loaded from {load_path}")


def create_feature_engineering_pipeline(config: FeatureConfig) -> FeatureEngineer:
    """Create and return configured feature engineering pipeline"""
    return FeatureEngineer(config)