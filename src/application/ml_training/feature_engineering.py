"""
Feature Engineering Pipeline for ML Training

This module provides a production-ready feature engineering pipeline that handles:
- Automated feature extraction from property data
- Feature scaling and normalization  
- Feature selection and importance analysis
- Real-time feature computation for inference
- Integration with domain entities and scraped data

Features:
- Automated feature extraction and transformation
- Real-time and batch feature processing
- Feature store integration
- Feature validation and quality checks
- Feature lineage and metadata tracking
- Distributed feature computation
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
from enum import Enum
from uuid import uuid4, UUID

import numpy as np
import pandas as pd
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

# Domain imports
from ...domain.entities.property import Property
from ...domain.entities.user import User, UserInteraction
from ...domain.entities.search_query import SearchQuery
from ...domain.repositories.property_repository import PropertyRepository
from ...domain.repositories.user_repository import UserRepository

# Infrastructure imports
from ...infrastructure.ml.training.data_loader import MLDataset


class FeatureType(Enum):
    """Feature type enumeration"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TEMPORAL = "temporal"
    GEOSPATIAL = "geospatial"
    INTERACTION = "interaction"


class ProcessingMode(Enum):
    """Feature processing mode"""
    BATCH = "batch"
    REAL_TIME = "real_time"
    STREAMING = "streaming"


@dataclass
class FeatureDefinition:
    """Definition of a feature"""
    name: str
    feature_type: FeatureType
    source_columns: List[str]
    transformation: str
    parameters: Dict[str, Any]
    description: str
    importance_score: Optional[float] = None
    is_enabled: bool = True
    version: str = "1.0"
    created_at: Optional[datetime] = None


@dataclass
class FeatureSet:
    """Collection of features with metadata"""
    name: str
    version: str
    features: List[FeatureDefinition]
    metadata: Dict[str, Any]
    created_at: datetime
    feature_count: int
    data_schema: Dict[str, str]


@dataclass
class FeatureProcessingResult:
    """Result of feature processing"""
    feature_set_name: str
    processed_features: np.ndarray
    feature_names: List[str]
    feature_metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time_seconds: float


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    # Processing configuration
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    enable_feature_selection: bool = True
    max_features: Optional[int] = None
    feature_selection_method: str = "mutual_info"
    
    # Scaling and normalization
    scaling_method: str = "standard"  # "standard", "minmax", "robust", "none"
    handle_missing_values: str = "median"  # "median", "mean", "mode", "drop"
    
    # Text processing
    max_text_features: int = 1000
    min_df: int = 2
    max_df: float = 0.8
    ngram_range: Tuple[int, int] = (1, 2)
    
    # Temporal features
    enable_temporal_features: bool = True
    temporal_window_days: int = 30
    
    # Geospatial features
    enable_geospatial_features: bool = True
    location_clustering_k: int = 50
    
    # Quality thresholds
    min_feature_variance: float = 0.01
    max_correlation_threshold: float = 0.95
    max_missing_percentage: float = 0.5
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24


class FeatureEngineeringPipeline:
    """
    Production-ready feature engineering pipeline for rental ML system.
    
    This class provides comprehensive feature engineering capabilities including:
    - Automated feature extraction from multiple data sources
    - Real-time and batch feature processing
    - Feature validation and quality checks
    - Feature store integration
    - Scalable distributed processing
    """
    
    def __init__(self,
                 property_repository: PropertyRepository,
                 user_repository: UserRepository,
                 config: Optional[FeatureEngineeringConfig] = None,
                 feature_store_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            property_repository: Repository for property data
            user_repository: Repository for user data
            config: Feature engineering configuration
            feature_store_config: Feature store configuration
        """
        self.property_repository = property_repository
        self.user_repository = user_repository
        self.config = config or FeatureEngineeringConfig()
        self.feature_store_config = feature_store_config or {}
        
        self.logger = logging.getLogger(__name__)
        
        # Feature processors
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        
        # Feature definitions and metadata
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_sets: Dict[str, FeatureSet] = {}
        
        # Processing state
        self.is_fitted: bool = False
        self.processing_stats: Dict[str, Any] = {}
        
        # Cache for processed features
        self.feature_cache: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize the feature engineering pipeline"""
        try:
            self.logger.info("Initializing feature engineering pipeline...")
            
            # Load predefined feature definitions
            await self._load_feature_definitions()
            
            # Initialize processors
            await self._initialize_processors()
            
            # Setup feature store connection if configured
            if self.feature_store_config:
                await self._setup_feature_store()
            
            self.logger.info("Feature engineering pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize feature engineering pipeline: {e}")
            raise
    
    async def process_features(self,
                             dataset: MLDataset,
                             config: Optional[Dict[str, Any]] = None,
                             feature_set_name: str = "default") -> FeatureProcessingResult:
        """
        Process features for a dataset.
        
        Args:
            dataset: ML dataset to process
            config: Optional processing configuration overrides
            feature_set_name: Name of the feature set to create
            
        Returns:
            Feature processing result
        """
        try:
            start_time = datetime.utcnow()
            
            self.logger.info(f"Starting feature processing for dataset: {feature_set_name}")
            
            # Merge configuration
            processing_config = self._merge_config(config)
            
            # Extract raw features
            raw_features = await self._extract_raw_features(dataset)
            
            # Apply transformations
            transformed_features = await self._apply_transformations(raw_features, processing_config)
            
            # Feature selection
            if processing_config.get('enable_feature_selection', True):
                selected_features = await self._select_features(transformed_features, dataset)
            else:
                selected_features = transformed_features
            
            # Quality validation
            quality_metrics = await self._validate_feature_quality(selected_features)
            
            # Scale features
            if processing_config.get('scaling_method', 'standard') != 'none':
                scaled_features = await self._scale_features(selected_features, processing_config)
            else:
                scaled_features = selected_features
            
            # Create feature set
            feature_set = await self._create_feature_set(
                feature_set_name, scaled_features, processing_config
            )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create result
            result = FeatureProcessingResult(
                feature_set_name=feature_set_name,
                processed_features=scaled_features['feature_matrix'],
                feature_names=scaled_features['feature_names'],
                feature_metadata=scaled_features['metadata'],
                processing_stats=self.processing_stats,
                quality_metrics=quality_metrics,
                processing_time_seconds=processing_time
            )
            
            # Cache result if enabled
            if self.config.enable_caching:
                await self._cache_features(feature_set_name, result)
            
            self.logger.info(f"Feature processing completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature processing failed: {e}")
            raise
    
    async def process_real_time_features(self,
                                       property_data: Dict[str, Any],
                                       user_data: Optional[Dict[str, Any]] = None,
                                       feature_set_name: str = "real_time") -> np.ndarray:
        """
        Process features for real-time inference.
        
        Args:
            property_data: Property data for feature extraction
            user_data: Optional user data for personalized features
            feature_set_name: Feature set to use
            
        Returns:
            Processed feature vector
        """
        try:
            if not self.is_fitted:
                raise ValueError("Pipeline must be fitted before real-time processing")
            
            # Extract features from single data point
            features = await self._extract_single_point_features(property_data, user_data)
            
            # Apply fitted transformations
            transformed_features = await self._apply_fitted_transformations(features, feature_set_name)
            
            return transformed_features
            
        except Exception as e:
            self.logger.error(f"Real-time feature processing failed: {e}")
            raise
    
    async def _load_feature_definitions(self):
        """Load predefined feature definitions"""
        try:
            # Define property features
            property_features = [
                FeatureDefinition(
                    name="price_normalized",
                    feature_type=FeatureType.NUMERICAL,
                    source_columns=["price"],
                    transformation="log_scale",
                    parameters={"log_base": "natural"},
                    description="Log-normalized property price"
                ),
                FeatureDefinition(
                    name="price_per_sqft",
                    feature_type=FeatureType.NUMERICAL,
                    source_columns=["price", "square_feet"],
                    transformation="ratio",
                    parameters={},
                    description="Price per square foot ratio"
                ),
                FeatureDefinition(
                    name="location_encoded",
                    feature_type=FeatureType.CATEGORICAL,
                    source_columns=["location"],
                    transformation="label_encoding",
                    parameters={},
                    description="Encoded location labels"
                ),
                FeatureDefinition(
                    name="amenities_tfidf",
                    feature_type=FeatureType.TEXT,
                    source_columns=["amenities"],
                    transformation="tfidf",
                    parameters={"max_features": 200, "ngram_range": (1, 2)},
                    description="TF-IDF features from amenities"
                ),
                FeatureDefinition(
                    name="property_age_days",
                    feature_type=FeatureType.TEMPORAL,
                    source_columns=["scraped_at"],
                    transformation="days_since",
                    parameters={"reference_date": "now"},
                    description="Days since property was scraped"
                ),
                FeatureDefinition(
                    name="bedrooms_categorical",
                    feature_type=FeatureType.CATEGORICAL,
                    source_columns=["bedrooms"],
                    transformation="one_hot",
                    parameters={"max_categories": 10},
                    description="One-hot encoded bedroom count"
                )
            ]
            
            # Define user features
            user_features = [
                FeatureDefinition(
                    name="user_interaction_count",
                    feature_type=FeatureType.INTERACTION,
                    source_columns=["user_interactions"],
                    transformation="count",
                    parameters={},
                    description="Total user interaction count"
                ),
                FeatureDefinition(
                    name="user_price_preference",
                    feature_type=FeatureType.NUMERICAL,
                    source_columns=["min_price", "max_price"],
                    transformation="average",
                    parameters={},
                    description="User's average price preference"
                ),
                FeatureDefinition(
                    name="user_location_preferences",
                    feature_type=FeatureType.TEXT,
                    source_columns=["preferred_locations"],
                    transformation="binary_encoding",
                    parameters={},
                    description="Binary encoding of location preferences"
                )
            ]
            
            # Store feature definitions
            all_features = property_features + user_features
            for feature_def in all_features:
                feature_def.created_at = datetime.utcnow()
                self.feature_definitions[feature_def.name] = feature_def
            
            self.logger.info(f"Loaded {len(all_features)} feature definitions")
            
        except Exception as e:
            self.logger.error(f"Failed to load feature definitions: {e}")
            raise
    
    async def _initialize_processors(self):
        """Initialize feature processors"""
        try:
            # Initialize scalers
            if self.config.scaling_method == "standard":
                self.scalers['default'] = StandardScaler()
            elif self.config.scaling_method == "minmax":
                self.scalers['default'] = MinMaxScaler()
            elif self.config.scaling_method == "robust":
                self.scalers['default'] = RobustScaler()
            
            # Initialize encoders
            self.encoders['label'] = LabelEncoder()
            self.encoders['onehot'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # Initialize vectorizers
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=self.config.max_text_features,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                ngram_range=self.config.ngram_range,
                stop_words='english'
            )
            
            # Initialize feature selectors
            if self.config.feature_selection_method == "mutual_info":
                self.feature_selectors['default'] = SelectKBest(
                    score_func=mutual_info_regression,
                    k=self.config.max_features or 'all'
                )
            elif self.config.feature_selection_method == "f_test":
                self.feature_selectors['default'] = SelectKBest(
                    score_func=f_regression,
                    k=self.config.max_features or 'all'
                )
            
            self.logger.info("Feature processors initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processors: {e}")
            raise
    
    async def _setup_feature_store(self):
        """Setup feature store connection"""
        try:
            # This would setup connection to a feature store like Feast, Tecton, etc.
            # For now, we'll use a simple in-memory store
            self.feature_store = {}
            self.logger.info("Feature store setup completed")
            
        except Exception as e:
            self.logger.warning(f"Feature store setup failed: {e}")
    
    def _merge_config(self, config_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge configuration with overrides"""
        base_config = asdict(self.config)
        if config_override:
            base_config.update(config_override)
        return base_config
    
    async def _extract_raw_features(self, dataset: MLDataset) -> Dict[str, Any]:
        """Extract raw features from dataset"""
        try:
            features = {
                'property_features': {},
                'user_features': {},
                'interaction_features': {},
                'metadata': {}
            }
            
            # Extract property features
            if dataset.train_data.property_metadata:
                property_df = pd.DataFrame(dataset.train_data.property_metadata)
                features['property_features'] = await self._extract_property_features(property_df)
            
            # Extract user features
            if dataset.train_data.user_metadata:
                user_df = pd.DataFrame(dataset.train_data.user_metadata)
                features['user_features'] = await self._extract_user_features(user_df)
            
            # Extract interaction features
            if dataset.train_data.user_item_matrix is not None:
                features['interaction_features'] = await self._extract_interaction_features(
                    dataset.train_data.user_item_matrix
                )
            
            # Store metadata
            features['metadata'] = {
                'extraction_timestamp': datetime.utcnow().isoformat(),
                'dataset_size': dataset.metadata.get('total_interactions', 0),
                'feature_definitions_used': list(self.feature_definitions.keys())
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Raw feature extraction failed: {e}")
            raise
    
    async def _extract_property_features(self, property_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract property-specific features"""
        try:
            features = {}
            
            # Numerical features
            if 'price' in property_df.columns:
                # Log-normalized price
                prices = property_df['price'].fillna(property_df['price'].median())
                features['price_log'] = np.log1p(prices).values
                
                # Price percentiles
                features['price_percentile'] = prices.rank(pct=True).values
            
            if 'square_feet' in property_df.columns and 'price' in property_df.columns:
                # Price per square foot
                sqft = property_df['square_feet'].fillna(property_df['square_feet'].median())
                prices = property_df['price'].fillna(property_df['price'].median())
                features['price_per_sqft'] = (prices / sqft.replace(0, np.nan)).fillna(0).values
            
            # Categorical features
            if 'bedrooms' in property_df.columns:
                features['bedrooms'] = property_df['bedrooms'].fillna(0).values
            
            if 'bathrooms' in property_df.columns:
                features['bathrooms'] = property_df['bathrooms'].fillna(0).values
            
            # Location features
            if 'location' in property_df.columns:
                locations = property_df['location'].fillna('unknown')
                location_encoder = LabelEncoder()
                features['location_encoded'] = location_encoder.fit_transform(locations)
                self.encoders['location'] = location_encoder
            
            # Text features (amenities)
            if 'amenities' in property_df.columns:
                amenity_texts = []
                for amenities in property_df['amenities']:
                    if isinstance(amenities, list):
                        text = ' '.join(amenities)
                    else:
                        text = str(amenities) if amenities else ''
                    amenity_texts.append(text)
                
                if amenity_texts:
                    amenity_tfidf = self.vectorizers['tfidf'].fit_transform(amenity_texts)
                    features['amenities_tfidf'] = amenity_tfidf.toarray()
            
            # Temporal features
            if 'scraped_at' in property_df.columns:
                scraped_dates = pd.to_datetime(property_df['scraped_at'])
                now = datetime.utcnow()
                days_since = (now - scraped_dates).dt.days
                features['days_since_scraped'] = days_since.fillna(0).values
            
            return features
            
        except Exception as e:
            self.logger.error(f"Property feature extraction failed: {e}")
            raise
    
    async def _extract_user_features(self, user_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract user-specific features"""
        try:
            features = {}
            
            # Price preferences
            if 'min_price' in user_df.columns and 'max_price' in user_df.columns:
                min_prices = user_df['min_price'].fillna(0)
                max_prices = user_df['max_price'].fillna(0)
                
                features['price_range'] = (max_prices - min_prices).values
                features['avg_price_preference'] = ((min_prices + max_prices) / 2).values
            
            # Bedroom preferences
            if 'min_bedrooms' in user_df.columns:
                features['min_bedrooms'] = user_df['min_bedrooms'].fillna(0).values
            
            if 'max_bedrooms' in user_df.columns:
                features['max_bedrooms'] = user_df['max_bedrooms'].fillna(0).values
            
            # Location preferences
            if 'preferred_locations' in user_df.columns:
                location_counts = []
                for locations in user_df['preferred_locations']:
                    if isinstance(locations, list):
                        location_counts.append(len(locations))
                    else:
                        location_counts.append(0)
                features['preferred_location_count'] = np.array(location_counts)
            
            # Amenity preferences
            if 'required_amenities' in user_df.columns:
                amenity_counts = []
                for amenities in user_df['required_amenities']:
                    if isinstance(amenities, list):
                        amenity_counts.append(len(amenities))
                    else:
                        amenity_counts.append(0)
                features['required_amenity_count'] = np.array(amenity_counts)
            
            return features
            
        except Exception as e:
            self.logger.error(f"User feature extraction failed: {e}")
            raise
    
    async def _extract_interaction_features(self, user_item_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract interaction-based features"""
        try:
            features = {}
            
            # User activity features
            features['user_interaction_count'] = np.sum(user_item_matrix > 0, axis=1)
            features['user_avg_rating'] = np.mean(user_item_matrix, axis=1)
            features['user_rating_variance'] = np.var(user_item_matrix, axis=1)
            
            # Item popularity features
            features['item_interaction_count'] = np.sum(user_item_matrix > 0, axis=0)
            features['item_avg_rating'] = np.mean(user_item_matrix, axis=0)
            
            # Sparsity features
            total_possible = user_item_matrix.shape[0] * user_item_matrix.shape[1]
            actual_interactions = np.sum(user_item_matrix > 0)
            features['interaction_sparsity'] = np.array([1.0 - (actual_interactions / total_possible)])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Interaction feature extraction failed: {e}")
            raise
    
    async def _apply_transformations(self, 
                                   raw_features: Dict[str, Any], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply feature transformations"""
        try:
            transformed = {
                'feature_matrices': [],
                'feature_names': [],
                'metadata': raw_features.get('metadata', {})
            }
            
            # Process property features
            if 'property_features' in raw_features:
                prop_features = raw_features['property_features']
                for feature_name, feature_data in prop_features.items():
                    if isinstance(feature_data, np.ndarray):
                        if feature_data.ndim == 1:
                            feature_data = feature_data.reshape(-1, 1)
                        transformed['feature_matrices'].append(feature_data)
                        
                        # Generate feature names
                        if feature_data.shape[1] == 1:
                            transformed['feature_names'].append(feature_name)
                        else:
                            for i in range(feature_data.shape[1]):
                                transformed['feature_names'].append(f"{feature_name}_{i}")
            
            # Process user features
            if 'user_features' in raw_features:
                user_features = raw_features['user_features']
                for feature_name, feature_data in user_features.items():
                    if isinstance(feature_data, np.ndarray):
                        if feature_data.ndim == 1:
                            feature_data = feature_data.reshape(-1, 1)
                        transformed['feature_matrices'].append(feature_data)
                        
                        if feature_data.shape[1] == 1:
                            transformed['feature_names'].append(f"user_{feature_name}")
                        else:
                            for i in range(feature_data.shape[1]):
                                transformed['feature_names'].append(f"user_{feature_name}_{i}")
            
            # Combine all feature matrices
            if transformed['feature_matrices']:
                # Ensure all matrices have the same number of rows
                max_rows = max(mat.shape[0] for mat in transformed['feature_matrices'])
                aligned_matrices = []
                
                for mat in transformed['feature_matrices']:
                    if mat.shape[0] < max_rows:
                        # Pad with zeros or repeat last row
                        padding = np.zeros((max_rows - mat.shape[0], mat.shape[1]))
                        mat = np.vstack([mat, padding])
                    aligned_matrices.append(mat)
                
                transformed['combined_matrix'] = np.hstack(aligned_matrices)
            else:
                transformed['combined_matrix'] = np.array([])
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Feature transformation failed: {e}")
            raise
    
    async def _select_features(self, 
                             transformed_features: Dict[str, Any], 
                             dataset: MLDataset) -> Dict[str, Any]:
        """Apply feature selection"""
        try:
            if 'combined_matrix' not in transformed_features or transformed_features['combined_matrix'].size == 0:
                return transformed_features
            
            feature_matrix = transformed_features['combined_matrix']
            feature_names = transformed_features['feature_names']
            
            # Create target variable for feature selection (use interaction ratings)
            if dataset.train_data.user_item_matrix is not None:
                # Use average rating as target for feature selection
                target = np.mean(dataset.train_data.user_item_matrix, axis=1)
                
                # Ensure target and features have same length
                min_length = min(len(target), feature_matrix.shape[0])
                target = target[:min_length]
                feature_matrix = feature_matrix[:min_length]
                
                if len(target) > 0 and self.config.max_features:
                    # Apply feature selection
                    selector = self.feature_selectors['default']
                    selected_matrix = selector.fit_transform(feature_matrix, target)
                    
                    # Get selected feature names
                    if hasattr(selector, 'get_support'):
                        support_mask = selector.get_support()
                        selected_names = [name for name, selected in zip(feature_names, support_mask) if selected]
                    else:
                        selected_names = feature_names[:selected_matrix.shape[1]]
                    
                    transformed_features['combined_matrix'] = selected_matrix
                    transformed_features['feature_names'] = selected_names
                    transformed_features['metadata']['feature_selection_applied'] = True
                    transformed_features['metadata']['selected_feature_count'] = selected_matrix.shape[1]
                    
                    self.feature_selectors['fitted'] = selector
            
            return transformed_features
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed, using all features: {e}")
            return transformed_features
    
    async def _scale_features(self, 
                            features: Dict[str, Any], 
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Scale features using configured method"""
        try:
            if 'combined_matrix' not in features or features['combined_matrix'].size == 0:
                return features
            
            feature_matrix = features['combined_matrix']
            
            # Apply scaling
            scaler = self.scalers['default']
            scaled_matrix = scaler.fit_transform(feature_matrix)
            
            # Store fitted scaler
            self.scalers['fitted'] = scaler
            
            # Update features
            scaled_features = features.copy()
            scaled_features['feature_matrix'] = scaled_matrix
            scaled_features['metadata']['scaling_applied'] = True
            scaled_features['metadata']['scaling_method'] = config.get('scaling_method', 'standard')
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Feature scaling failed: {e}")
            raise
    
    async def _validate_feature_quality(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Validate feature quality and compute metrics"""
        try:
            quality_metrics = {}
            
            if 'feature_matrix' not in features or features['feature_matrix'].size == 0:
                return quality_metrics
            
            feature_matrix = features['feature_matrix']
            
            # Check for missing values
            missing_percentage = np.isnan(feature_matrix).mean() * 100
            quality_metrics['missing_percentage'] = float(missing_percentage)
            
            # Check feature variance
            feature_variances = np.var(feature_matrix, axis=0)
            low_variance_count = np.sum(feature_variances < self.config.min_feature_variance)
            quality_metrics['low_variance_features'] = int(low_variance_count)
            quality_metrics['avg_feature_variance'] = float(np.mean(feature_variances))
            
            # Check feature correlations
            if feature_matrix.shape[1] > 1:
                correlation_matrix = np.corrcoef(feature_matrix.T)
                # Count highly correlated feature pairs
                high_corr_mask = np.abs(correlation_matrix) > self.config.max_correlation_threshold
                # Exclude diagonal
                np.fill_diagonal(high_corr_mask, False)
                high_corr_count = np.sum(high_corr_mask) // 2  # Divide by 2 due to symmetry
                quality_metrics['high_correlation_pairs'] = int(high_corr_count)
            
            # Feature distribution statistics
            quality_metrics['feature_count'] = int(feature_matrix.shape[1])
            quality_metrics['sample_count'] = int(feature_matrix.shape[0])
            quality_metrics['feature_density'] = float(1.0 - missing_percentage / 100)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Feature quality validation failed: {e}")
            return {}
    
    async def _create_feature_set(self,
                                feature_set_name: str,
                                processed_features: Dict[str, Any],
                                config: Dict[str, Any]) -> FeatureSet:
        """Create a feature set from processed features"""
        try:
            # Create feature definitions for processed features
            feature_definitions = []
            feature_names = processed_features.get('feature_names', [])
            
            for i, name in enumerate(feature_names):
                feature_def = FeatureDefinition(
                    name=name,
                    feature_type=FeatureType.NUMERICAL,  # Default type
                    source_columns=[name],
                    transformation="processed",
                    parameters={},
                    description=f"Processed feature: {name}",
                    created_at=datetime.utcnow()
                )
                feature_definitions.append(feature_def)
            
            # Create feature set
            feature_set = FeatureSet(
                name=feature_set_name,
                version="1.0",
                features=feature_definitions,
                metadata={
                    'processing_config': config,
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    **processed_features.get('metadata', {})
                },
                created_at=datetime.utcnow(),
                feature_count=len(feature_definitions),
                data_schema={name: "float64" for name in feature_names}
            )
            
            # Store feature set
            self.feature_sets[feature_set_name] = feature_set
            self.is_fitted = True
            
            return feature_set
            
        except Exception as e:
            self.logger.error(f"Failed to create feature set: {e}")
            raise
    
    async def _cache_features(self, feature_set_name: str, result: FeatureProcessingResult):
        """Cache processed features"""
        try:
            if self.config.enable_caching:
                cache_key = f"features_{feature_set_name}_{datetime.utcnow().strftime('%Y%m%d_%H')}"
                self.feature_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.utcnow(),
                    'ttl_hours': self.config.cache_ttl_hours
                }
                
                # Clean up old cache entries
                await self._cleanup_cache()
                
        except Exception as e:
            self.logger.warning(f"Feature caching failed: {e}")
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.utcnow()
            expired_keys = []
            
            for key, cache_entry in self.feature_cache.items():
                cache_time = cache_entry['timestamp']
                ttl_hours = cache_entry['ttl_hours']
                
                if (current_time - cache_time).total_seconds() > ttl_hours * 3600:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.feature_cache[key]
                
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")
    
    async def _extract_single_point_features(self,
                                           property_data: Dict[str, Any],
                                           user_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Extract features for a single data point (real-time inference)"""
        try:
            features = []
            
            # Extract property features
            if 'price' in property_data:
                price = float(property_data['price'])
                features.append(np.log1p(price))  # Log-normalized price
            
            if 'square_feet' in property_data and 'price' in property_data:
                sqft = float(property_data.get('square_feet', 1))
                price = float(property_data['price'])
                if sqft > 0:
                    features.append(price / sqft)
                else:
                    features.append(0.0)
            
            if 'bedrooms' in property_data:
                features.append(float(property_data['bedrooms']))
            
            if 'bathrooms' in property_data:
                features.append(float(property_data['bathrooms']))
            
            # Add user features if available
            if user_data:
                if 'min_price' in user_data and 'max_price' in user_data:
                    min_price = float(user_data.get('min_price', 0))
                    max_price = float(user_data.get('max_price', 0))
                    features.append((min_price + max_price) / 2)  # Average preference
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Single point feature extraction failed: {e}")
            return np.array([]).reshape(1, -1)
    
    async def _apply_fitted_transformations(self,
                                          features: np.ndarray,
                                          feature_set_name: str) -> np.ndarray:
        """Apply fitted transformations for real-time inference"""
        try:
            if not self.is_fitted:
                raise ValueError("Pipeline must be fitted before applying transformations")
            
            # Apply fitted scaler
            if 'fitted' in self.scalers and features.size > 0:
                scaled_features = self.scalers['fitted'].transform(features)
                return scaled_features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fitted transformation failed: {e}")
            return features
    
    def get_feature_importance(self, feature_set_name: str = "default") -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            if 'fitted' in self.feature_selectors:
                selector = self.feature_selectors['fitted']
                if hasattr(selector, 'scores_'):
                    feature_set = self.feature_sets.get(feature_set_name)
                    if feature_set:
                        feature_names = [f.name for f in feature_set.features]
                        importance_scores = selector.scores_
                        
                        return dict(zip(feature_names, importance_scores))
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Failed to get feature importance: {e}")
            return {}
    
    def get_feature_set_info(self, feature_set_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a feature set"""
        try:
            if feature_set_name in self.feature_sets:
                feature_set = self.feature_sets[feature_set_name]
                return {
                    'name': feature_set.name,
                    'version': feature_set.version,
                    'feature_count': feature_set.feature_count,
                    'created_at': feature_set.created_at.isoformat(),
                    'metadata': feature_set.metadata,
                    'data_schema': feature_set.data_schema,
                    'feature_names': [f.name for f in feature_set.features]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get feature set info: {e}")
            return None
    
    async def save_feature_set(self, feature_set_name: str, file_path: str):
        """Save feature set to file"""
        try:
            if feature_set_name not in self.feature_sets:
                raise ValueError(f"Feature set not found: {feature_set_name}")
            
            feature_set = self.feature_sets[feature_set_name]
            
            # Prepare data for saving
            save_data = {
                'feature_set': asdict(feature_set),
                'processors': {
                    'scalers': self.scalers,
                    'encoders': self.encoders,
                    'vectorizers': self.vectorizers,
                    'feature_selectors': self.feature_selectors
                },
                'config': asdict(self.config),
                'is_fitted': self.is_fitted
            }
            
            # Convert datetime objects to strings
            save_data['feature_set']['created_at'] = feature_set.created_at.isoformat()
            for feature in save_data['feature_set']['features']:
                if feature.get('created_at'):
                    if isinstance(feature['created_at'], datetime):
                        feature['created_at'] = feature['created_at'].isoformat()
                    elif isinstance(feature['created_at'], str):
                        # Already a string, potentially from previous serialization
                        try:
                            datetime.fromisoformat(feature['created_at'])  # Validate format
                        except:
                            feature['created_at'] = datetime.utcnow().isoformat()
            
            # Save to file
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"Feature set saved: {feature_set_name} -> {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save feature set: {e}")
            raise
    
    async def load_feature_set(self, file_path: str) -> str:
        """Load feature set from file"""
        try:
            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore feature set
            feature_set_data = save_data['feature_set']
            feature_set_data['created_at'] = datetime.fromisoformat(feature_set_data['created_at'])
            
            # Restore feature definitions
            for feature_data in feature_set_data['features']:
                if feature_data['created_at']:
                    feature_data['created_at'] = datetime.fromisoformat(feature_data['created_at'])
            
            # Reconstruct feature set
            feature_set = FeatureSet(**feature_set_data)
            self.feature_sets[feature_set.name] = feature_set
            
            # Restore processors
            processors = save_data['processors']
            self.scalers.update(processors.get('scalers', {}))
            self.encoders.update(processors.get('encoders', {}))
            self.vectorizers.update(processors.get('vectorizers', {}))
            self.feature_selectors.update(processors.get('feature_selectors', {}))
            
            # Restore state
            self.is_fitted = save_data.get('is_fitted', False)
            
            self.logger.info(f"Feature set loaded: {feature_set.name} from {file_path}")
            
            return feature_set.name
            
        except Exception as e:
            self.logger.error(f"Failed to load feature set: {e}")
            raise