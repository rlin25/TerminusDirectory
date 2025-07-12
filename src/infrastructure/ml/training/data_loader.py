"""
Production data loaders for ML training pipelines.

This module provides optimized data loading capabilities that connect to PostgreSQL
to fetch training data for ML models. It includes data preprocessing, feature
engineering, and batch loading for efficient training.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from uuid import UUID

import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from ...data.config import get_database_config
from ....domain.entities.property import Property
from ....domain.entities.user import User, UserInteraction


@dataclass
class TrainingDataBatch:
    """Container for a batch of training data"""
    user_item_matrix: np.ndarray
    property_features: np.ndarray
    user_features: np.ndarray
    property_metadata: List[Dict]
    user_metadata: List[Dict]
    feature_names: List[str]
    batch_info: Dict[str, Any]


@dataclass
class MLDataset:
    """Complete ML dataset with train/validation splits"""
    train_data: TrainingDataBatch
    validation_data: TrainingDataBatch
    test_data: Optional[TrainingDataBatch]
    metadata: Dict[str, Any]


class ProductionDataLoader:
    """
    Production data loader for ML training pipelines.
    
    This class handles:
    - Efficient data fetching from PostgreSQL
    - Feature engineering and preprocessing
    - Train/validation/test splits
    - Batch generation for training
    - Data quality validation
    - Real-time feature computation
    """
    
    def __init__(self, database_url: str, cache_size: int = 10000):
        self.database_url = database_url
        self.cache_size = cache_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize database connections
        self.engine = None
        self.connection_pool = None
        
        # Feature processors
        self.property_scaler = StandardScaler()
        self.user_scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        self.amenity_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Data caches
        self._property_cache = {}
        self._user_cache = {}
        
        # Feature mappings
        self._feature_mappings = {}
        
    async def initialize(self):
        """Initialize database connections and feature processors"""
        try:
            # Create async engine for SQLAlchemy queries
            self.engine = create_async_engine(
                self.database_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create asyncpg connection pool for high-performance queries
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=10,
                max_size=20,
                command_timeout=60
            )
            
            self.logger.info("Database connections initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data loader: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
            if self.connection_pool:
                await self.connection_pool.close()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
    
    async def load_training_dataset(self, 
                                  train_split: float = 0.7,
                                  validation_split: float = 0.2,
                                  test_split: float = 0.1,
                                  min_interactions: int = 5,
                                  max_users: Optional[int] = None,
                                  max_properties: Optional[int] = None) -> MLDataset:
        """
        Load complete training dataset with train/validation/test splits.
        
        Args:
            train_split: Fraction of data for training
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            min_interactions: Minimum interactions per user to include
            max_users: Maximum number of users to include (for sampling)
            max_properties: Maximum number of properties to include
            
        Returns:
            MLDataset containing all splits and metadata
        """
        try:
            # Validate splits
            if abs(train_split + validation_split + test_split - 1.0) > 1e-6:
                raise ValueError("Data splits must sum to 1.0")
            
            self.logger.info("Loading production training dataset...")
            
            # Load raw data
            users_data = await self._load_users_data(
                min_interactions=min_interactions, 
                max_users=max_users
            )
            properties_data = await self._load_properties_data(max_properties=max_properties)
            interactions_data = await self._load_interactions_data()
            
            # Build user-item interaction matrix
            user_item_matrix, user_id_mapping, property_id_mapping = await self._build_interaction_matrix(
                users_data, properties_data, interactions_data
            )
            
            # Extract and process features
            property_features = await self._extract_property_features(properties_data)
            user_features = await self._extract_user_features(users_data)
            
            # Create metadata
            property_metadata = [
                {
                    'id': str(prop['id']),
                    'title': prop['title'],
                    'location': prop['location'],
                    'price': prop['price'],
                    'bedrooms': prop['bedrooms'],
                    'bathrooms': prop['bathrooms']
                }
                for prop in properties_data
            ]
            
            user_metadata = [
                {
                    'id': str(user['id']),
                    'email': user['email'],
                    'interaction_count': len([i for i in interactions_data if i['user_id'] == user['id']])
                }
                for user in users_data
            ]
            
            # Create feature names
            feature_names = self._get_feature_names()
            
            # Split data
            num_users = len(users_data)
            indices = np.arange(num_users)
            
            # First split: train vs temp (test + validation)
            train_indices, temp_indices = train_test_split(
                indices, 
                test_size=(validation_split + test_split),
                random_state=42,
                stratify=None  # Could stratify by user activity level
            )
            
            # Second split: validation vs test
            if test_split > 0:
                val_size = validation_split / (validation_split + test_split)
                val_indices, test_indices = train_test_split(
                    temp_indices,
                    test_size=(1 - val_size),
                    random_state=42
                )
            else:
                val_indices = temp_indices
                test_indices = []
            
            # Create data batches
            train_data = TrainingDataBatch(
                user_item_matrix=user_item_matrix[train_indices],
                property_features=property_features,
                user_features=user_features[train_indices],
                property_metadata=property_metadata,
                user_metadata=[user_metadata[i] for i in train_indices],
                feature_names=feature_names,
                batch_info={
                    'split': 'train',
                    'num_users': len(train_indices),
                    'num_properties': len(properties_data),
                    'num_interactions': np.sum(user_item_matrix[train_indices] > 0)
                }
            )
            
            validation_data = TrainingDataBatch(
                user_item_matrix=user_item_matrix[val_indices],
                property_features=property_features,
                user_features=user_features[val_indices],
                property_metadata=property_metadata,
                user_metadata=[user_metadata[i] for i in val_indices],
                feature_names=feature_names,
                batch_info={
                    'split': 'validation',
                    'num_users': len(val_indices),
                    'num_properties': len(properties_data),
                    'num_interactions': np.sum(user_item_matrix[val_indices] > 0)
                }
            )
            
            test_data = None
            if test_indices:
                test_data = TrainingDataBatch(
                    user_item_matrix=user_item_matrix[test_indices],
                    property_features=property_features,
                    user_features=user_features[test_indices],
                    property_metadata=property_metadata,
                    user_metadata=[user_metadata[i] for i in test_indices],
                    feature_names=feature_names,
                    batch_info={
                        'split': 'test',
                        'num_users': len(test_indices),
                        'num_properties': len(properties_data),
                        'num_interactions': np.sum(user_item_matrix[test_indices] > 0)
                    }
                )
            
            # Create dataset metadata
            dataset_metadata = {
                'created_at': datetime.utcnow().isoformat(),
                'total_users': len(users_data),
                'total_properties': len(properties_data),
                'total_interactions': len(interactions_data),
                'train_users': len(train_indices),
                'validation_users': len(val_indices),
                'test_users': len(test_indices) if test_indices else 0,
                'feature_dimensions': {
                    'property_features': property_features.shape[1],
                    'user_features': user_features.shape[1]
                },
                'data_quality': await self._assess_data_quality(
                    user_item_matrix, properties_data, users_data
                ),
                'splits': {
                    'train': train_split,
                    'validation': validation_split,
                    'test': test_split
                }
            }
            
            dataset = MLDataset(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                metadata=dataset_metadata
            )
            
            self.logger.info(
                f"Loaded dataset: {dataset_metadata['total_users']} users, "
                f"{dataset_metadata['total_properties']} properties, "
                f"{dataset_metadata['total_interactions']} interactions"
            )
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load training dataset: {e}")
            raise
    
    async def _load_users_data(self, 
                             min_interactions: int = 5, 
                             max_users: Optional[int] = None) -> List[Dict]:
        """Load users data from database"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Build query with interaction count filter
                query = """
                    SELECT DISTINCT
                        u.id,
                        u.email,
                        u.created_at,
                        u.is_active,
                        u.min_price,
                        u.max_price,
                        u.min_bedrooms,
                        u.max_bedrooms,
                        u.min_bathrooms,
                        u.max_bathrooms,
                        u.preferred_locations,
                        u.required_amenities,
                        u.property_types,
                        COUNT(ui.user_id) as interaction_count
                    FROM users u
                    LEFT JOIN user_interactions ui ON u.id = ui.user_id
                    WHERE u.is_active = true
                    GROUP BY u.id, u.email, u.created_at, u.is_active,
                             u.min_price, u.max_price, u.min_bedrooms, u.max_bedrooms,
                             u.min_bathrooms, u.max_bathrooms, u.preferred_locations,
                             u.required_amenities, u.property_types
                    HAVING COUNT(ui.user_id) >= $1
                    ORDER BY interaction_count DESC
                """
                
                params = [min_interactions]
                if max_users:
                    query += " LIMIT $2"
                    params.append(max_users)
                
                rows = await conn.fetch(query, *params)
                
                users_data = []
                for row in rows:
                    user_data = dict(row)
                    # Handle JSON arrays
                    user_data['preferred_locations'] = user_data['preferred_locations'] or []
                    user_data['required_amenities'] = user_data['required_amenities'] or []
                    user_data['property_types'] = user_data['property_types'] or ['apartment']
                    users_data.append(user_data)
                
                self.logger.info(f"Loaded {len(users_data)} users with >= {min_interactions} interactions")
                return users_data
                
        except Exception as e:
            self.logger.error(f"Failed to load users data: {e}")
            raise
    
    async def _load_properties_data(self, max_properties: Optional[int] = None) -> List[Dict]:
        """Load properties data from database"""
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                    SELECT 
                        id, title, description, price, location,
                        bedrooms, bathrooms, square_feet, amenities,
                        property_type, scraped_at, is_active,
                        price_per_sqft
                    FROM properties
                    WHERE is_active = true
                    ORDER BY scraped_at DESC
                """
                
                params = []
                if max_properties:
                    query += " LIMIT $1"
                    params.append(max_properties)
                
                rows = await conn.fetch(query, *params)
                
                properties_data = []
                for row in rows:
                    prop_data = dict(row)
                    # Handle JSON arrays
                    prop_data['amenities'] = prop_data['amenities'] or []
                    properties_data.append(prop_data)
                
                self.logger.info(f"Loaded {len(properties_data)} active properties")
                return properties_data
                
        except Exception as e:
            self.logger.error(f"Failed to load properties data: {e}")
            raise
    
    async def _load_interactions_data(self) -> List[Dict]:
        """Load user-property interactions from database"""
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        user_id, property_id, interaction_type, 
                        timestamp, duration_seconds
                    FROM user_interactions
                    ORDER BY timestamp DESC
                """)
                
                interactions_data = [dict(row) for row in rows]
                
                self.logger.info(f"Loaded {len(interactions_data)} user interactions")
                return interactions_data
                
        except Exception as e:
            self.logger.error(f"Failed to load interactions data: {e}")
            raise
    
    async def _build_interaction_matrix(self, 
                                      users_data: List[Dict],
                                      properties_data: List[Dict],
                                      interactions_data: List[Dict]) -> Tuple[np.ndarray, Dict, Dict]:
        """Build user-item interaction matrix"""
        try:
            # Create ID mappings
            user_ids = [user['id'] for user in users_data]
            property_ids = [prop['id'] for prop in properties_data]
            
            user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
            property_id_mapping = {prop_id: idx for idx, prop_id in enumerate(property_ids)}
            
            # Initialize matrix
            num_users = len(user_ids)
            num_properties = len(property_ids)
            interaction_matrix = np.zeros((num_users, num_properties), dtype=np.float32)
            
            # Fill matrix with interaction scores
            interaction_weights = {
                'view': 1.0,
                'like': 3.0,
                'inquiry': 5.0,
                'bookmark': 2.0
            }
            
            for interaction in interactions_data:
                user_id = interaction['user_id']
                property_id = interaction['property_id']
                interaction_type = interaction['interaction_type']
                
                if user_id in user_id_mapping and property_id in property_id_mapping:
                    user_idx = user_id_mapping[user_id]
                    prop_idx = property_id_mapping[property_id]
                    
                    weight = interaction_weights.get(interaction_type, 1.0)
                    interaction_matrix[user_idx, prop_idx] += weight
            
            # Normalize interaction scores
            max_score = np.max(interaction_matrix)
            if max_score > 0:
                interaction_matrix = interaction_matrix / max_score
            
            self.logger.info(
                f"Built interaction matrix: {num_users}x{num_properties}, "
                f"sparsity: {(np.sum(interaction_matrix > 0) / interaction_matrix.size):.3f}"
            )
            
            return interaction_matrix, user_id_mapping, property_id_mapping
            
        except Exception as e:
            self.logger.error(f"Failed to build interaction matrix: {e}")
            raise
    
    async def _extract_property_features(self, properties_data: List[Dict]) -> np.ndarray:
        """Extract and process property features"""
        try:
            # Prepare data for feature extraction
            df = pd.DataFrame(properties_data)
            
            # Numerical features
            numerical_features = ['price', 'bedrooms', 'bathrooms', 'square_feet', 'price_per_sqft']
            numerical_data = df[numerical_features].fillna(0).values
            
            # Scale numerical features
            numerical_scaled = self.property_scaler.fit_transform(numerical_data)
            
            # Location features (categorical)
            locations = df['location'].fillna('unknown').values
            location_encoded = self.location_encoder.fit_transform(locations).reshape(-1, 1)
            
            # Amenity features (text)
            amenity_texts = []
            for amenities in df['amenities']:
                if isinstance(amenities, list):
                    amenity_text = ' '.join(amenities)
                else:
                    amenity_text = str(amenities) if amenities else ''
                amenity_texts.append(amenity_text)
            
            amenity_features = self.amenity_vectorizer.fit_transform(amenity_texts).toarray()
            
            # Property type features (one-hot)
            property_types = df['property_type'].fillna('apartment').values
            unique_types = np.unique(property_types)
            type_features = np.zeros((len(properties_data), len(unique_types)))
            
            for i, prop_type in enumerate(property_types):
                type_idx = np.where(unique_types == prop_type)[0][0]
                type_features[i, type_idx] = 1
            
            # Combine all features
            property_features = np.concatenate([
                numerical_scaled,
                location_encoded,
                amenity_features,
                type_features
            ], axis=1)
            
            self.logger.info(f"Extracted property features: {property_features.shape}")
            return property_features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to extract property features: {e}")
            raise
    
    async def _extract_user_features(self, users_data: List[Dict]) -> np.ndarray:
        """Extract and process user features"""
        try:
            # Prepare data for feature extraction
            df = pd.DataFrame(users_data)
            
            # Numerical preference features
            numerical_features = [
                'min_price', 'max_price', 'min_bedrooms', 'max_bedrooms',
                'min_bathrooms', 'max_bathrooms', 'interaction_count'
            ]
            numerical_data = df[numerical_features].fillna(0).values
            
            # Scale numerical features
            numerical_scaled = self.user_scaler.fit_transform(numerical_data)
            
            # Location preferences (binary features for common locations)
            all_locations = set()
            for locations in df['preferred_locations']:
                if locations:
                    all_locations.update(locations)
            
            common_locations = list(all_locations)[:50]  # Top 50 locations
            location_features = np.zeros((len(users_data), len(common_locations)))
            
            for i, locations in enumerate(df['preferred_locations']):
                if locations:
                    for loc in locations:
                        if loc in common_locations:
                            loc_idx = common_locations.index(loc)
                            location_features[i, loc_idx] = 1
            
            # Amenity preferences (binary features)
            all_amenities = set()
            for amenities in df['required_amenities']:
                if amenities:
                    all_amenities.update(amenities)
            
            common_amenities = list(all_amenities)[:30]  # Top 30 amenities
            amenity_features = np.zeros((len(users_data), len(common_amenities)))
            
            for i, amenities in enumerate(df['required_amenities']):
                if amenities:
                    for amenity in amenities:
                        if amenity in common_amenities:
                            amenity_idx = common_amenities.index(amenity)
                            amenity_features[i, amenity_idx] = 1
            
            # Property type preferences
            all_prop_types = set()
            for prop_types in df['property_types']:
                if prop_types:
                    all_prop_types.update(prop_types)
            
            prop_type_list = list(all_prop_types)
            prop_type_features = np.zeros((len(users_data), len(prop_type_list)))
            
            for i, prop_types in enumerate(df['property_types']):
                if prop_types:
                    for prop_type in prop_types:
                        if prop_type in prop_type_list:
                            type_idx = prop_type_list.index(prop_type)
                            prop_type_features[i, type_idx] = 1
            
            # Combine all features
            user_features = np.concatenate([
                numerical_scaled,
                location_features,
                amenity_features,
                prop_type_features
            ], axis=1)
            
            self.logger.info(f"Extracted user features: {user_features.shape}")
            return user_features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to extract user features: {e}")
            raise
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        feature_names = []
        
        # Property feature names
        feature_names.extend(['price', 'bedrooms', 'bathrooms', 'square_feet', 'price_per_sqft'])
        feature_names.append('location_encoded')
        
        # Amenity feature names
        if hasattr(self.amenity_vectorizer, 'get_feature_names_out'):
            amenity_names = self.amenity_vectorizer.get_feature_names_out()
            feature_names.extend([f'amenity_{name}' for name in amenity_names])
        
        # Add property type names (would need to be stored during feature extraction)
        feature_names.extend(['type_apartment', 'type_house', 'type_condo'])
        
        return feature_names
    
    async def _assess_data_quality(self, 
                                 user_item_matrix: np.ndarray,
                                 properties_data: List[Dict],
                                 users_data: List[Dict]) -> Dict[str, Any]:
        """Assess data quality metrics"""
        try:
            # Interaction matrix quality
            total_interactions = np.sum(user_item_matrix > 0)
            sparsity = total_interactions / user_item_matrix.size
            
            # User distribution
            user_interactions = np.sum(user_item_matrix > 0, axis=1)
            avg_user_interactions = np.mean(user_interactions)
            
            # Property distribution
            property_interactions = np.sum(user_item_matrix > 0, axis=0)
            avg_property_interactions = np.mean(property_interactions)
            
            # Data completeness
            properties_df = pd.DataFrame(properties_data)
            completeness = {
                'price': properties_df['price'].notna().mean(),
                'location': properties_df['location'].notna().mean(),
                'bedrooms': properties_df['bedrooms'].notna().mean(),
                'bathrooms': properties_df['bathrooms'].notna().mean(),
                'amenities': properties_df['amenities'].apply(lambda x: len(x) > 0 if x else False).mean()
            }
            
            quality_metrics = {
                'interaction_sparsity': float(sparsity),
                'total_interactions': int(total_interactions),
                'avg_user_interactions': float(avg_user_interactions),
                'avg_property_interactions': float(avg_property_interactions),
                'data_completeness': completeness,
                'cold_start_users': int(np.sum(user_interactions == 0)),
                'cold_start_properties': int(np.sum(property_interactions == 0))
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to assess data quality: {e}")
            return {}
    
    async def get_batch_generator(self, 
                                dataset: MLDataset,
                                split: str = 'train',
                                batch_size: int = 1024,
                                shuffle: bool = True) -> AsyncGenerator[Dict[str, np.ndarray], None]:
        """
        Generate batches for training.
        
        Args:
            dataset: ML dataset
            split: Which split to use ('train', 'validation', 'test')
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Yields:
            Dictionary containing batch data
        """
        try:
            # Select data split
            if split == 'train':
                data = dataset.train_data
            elif split == 'validation':
                data = dataset.validation_data
            elif split == 'test':
                data = dataset.test_data
                if data is None:
                    raise ValueError("Test data not available")
            else:
                raise ValueError(f"Invalid split: {split}")
            
            num_samples = data.user_item_matrix.shape[0]
            indices = np.arange(num_samples)
            
            if shuffle:
                np.random.shuffle(indices)
            
            # Generate batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch = {
                    'user_item_matrix': data.user_item_matrix[batch_indices],
                    'user_features': data.user_features[batch_indices],
                    'property_features': data.property_features,
                    'batch_indices': batch_indices,
                    'batch_size': len(batch_indices)
                }
                
                yield batch
                
        except Exception as e:
            self.logger.error(f"Failed to generate batches: {e}")
            raise
    
    async def get_real_time_features(self, 
                                   user_id: UUID, 
                                   property_ids: List[UUID]) -> Dict[str, np.ndarray]:
        """
        Get real-time features for inference.
        
        Args:
            user_id: User identifier
            property_ids: List of property identifiers
            
        Returns:
            Dictionary containing features for inference
        """
        try:
            async with self.connection_pool.acquire() as conn:
                # Get user data
                user_row = await conn.fetchrow("""
                    SELECT id, min_price, max_price, min_bedrooms, max_bedrooms,
                           min_bathrooms, max_bathrooms, preferred_locations,
                           required_amenities, property_types
                    FROM users WHERE id = $1
                """, user_id)
                
                if not user_row:
                    raise ValueError(f"User {user_id} not found")
                
                # Get properties data
                property_rows = await conn.fetch("""
                    SELECT id, price, bedrooms, bathrooms, square_feet,
                           location, amenities, property_type, price_per_sqft
                    FROM properties 
                    WHERE id = ANY($1) AND is_active = true
                """, property_ids)
                
                if not property_rows:
                    raise ValueError("No valid properties found")
                
                # Process user features (simplified for real-time)
                user_features = np.array([[
                    user_row['min_price'] or 0,
                    user_row['max_price'] or 0,
                    user_row['min_bedrooms'] or 0,
                    user_row['max_bedrooms'] or 0,
                    user_row['min_bathrooms'] or 0,
                    user_row['max_bathrooms'] or 0
                ]], dtype=np.float32)
                
                # Process property features
                property_features_list = []
                for prop in property_rows:
                    features = [
                        prop['price'] or 0,
                        prop['bedrooms'] or 0,
                        prop['bathrooms'] or 0,
                        prop['square_feet'] or 0,
                        prop['price_per_sqft'] or 0
                    ]
                    property_features_list.append(features)
                
                property_features = np.array(property_features_list, dtype=np.float32)
                
                return {
                    'user_features': user_features,
                    'property_features': property_features,
                    'user_id': user_id,
                    'property_ids': [row['id'] for row in property_rows]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get real-time features: {e}")
            raise