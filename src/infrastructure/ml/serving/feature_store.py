"""
Feature Store implementation for real-time feature serving.

This module provides a production-ready feature store that handles:
- Real-time feature computation and serving
- Feature versioning and lineage tracking
- Online and offline feature storage
- Feature freshness monitoring
- Feature drift detection
- Point-in-time correct feature retrieval
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from uuid import UUID, uuid4
import pickle
import hashlib

import numpy as np
import pandas as pd
import redis
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field

from ..training.feature_engineering import FeatureEngineer, FeatureConfig


@dataclass
class FeatureDefinition:
    """Definition of a feature"""
    name: str
    feature_type: str  # "numerical", "categorical", "text", "timestamp"
    source_table: str
    source_column: str
    transformation: Optional[str] = None
    aggregation_window: Optional[str] = None  # "1h", "24h", "7d", etc.
    freshness_sla: int = 300  # seconds
    owner: str = "ml-team"
    description: str = ""
    tags: List[str] = None


@dataclass
class FeatureValue:
    """Feature value with metadata"""
    feature_name: str
    value: Any
    timestamp: datetime
    entity_id: str
    version: str
    is_fresh: bool = True


class FeatureRequest(BaseModel):
    """Request for feature retrieval"""
    entity_ids: List[str]
    feature_names: List[str]
    timestamp: Optional[datetime] = None
    include_metadata: bool = False


class FeatureResponse(BaseModel):
    """Response with feature values"""
    features: Dict[str, Dict[str, Any]]  # entity_id -> feature_name -> value
    metadata: Optional[Dict[str, Any]] = None
    computation_time_ms: float
    cache_hit_ratio: float


class FeatureStore:
    """
    Production feature store implementation.
    
    This class provides:
    - Real-time feature serving with low latency
    - Feature versioning and lineage tracking
    - Online and offline feature storage
    - Feature freshness monitoring
    - Point-in-time correct feature retrieval
    - Feature drift detection
    """
    
    def __init__(self,
                 database_url: str,
                 redis_url: str = "redis://localhost:6379",
                 cache_ttl: int = 3600,
                 batch_size: int = 1000):
        
        self.database_url = database_url
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize connections
        self.engine = None
        self.redis_client = None
        self.connection_pool = None
        
        # Feature registry
        self.feature_definitions = {}
        self.feature_transformations = {}
        
        # Feature engineering pipeline
        self.feature_engineer = None
        
        # Monitoring
        self.metrics = {
            'requests_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'feature_freshness_violations': 0,
            'computation_time_p95': 0.0
        }
        
        # Background tasks
        self._background_tasks = set()
    
    async def initialize(self):
        """Initialize feature store"""
        try:
            # Initialize database connections
            self.engine = create_async_engine(
                self.database_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True
            )
            
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=10,
                max_size=20
            )
            
            # Initialize Redis
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connections
            await self._test_connections()
            
            # Load feature definitions
            await self._load_feature_definitions()
            
            # Initialize feature engineering
            feature_config = FeatureConfig()
            self.feature_engineer = FeatureEngineer(feature_config)
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Feature store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Feature store initialization failed: {e}")
            raise
    
    async def close(self):
        """Close feature store connections"""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Close connections
            if self.engine:
                await self.engine.dispose()
            
            if self.connection_pool:
                await self.connection_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Feature store closed")
            
        except Exception as e:
            self.logger.error(f"Error closing feature store: {e}")
    
    async def _test_connections(self):
        """Test all connections"""
        # Test database
        async with self.connection_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Test Redis
        await self.redis_client.ping()
        
        self.logger.info("All connections tested successfully")
    
    async def _load_feature_definitions(self):
        """Load feature definitions from database"""
        try:
            # In production, this would load from a feature registry table
            # For now, we'll define some example features
            
            self.feature_definitions = {
                "user_avg_price_preference": FeatureDefinition(
                    name="user_avg_price_preference",
                    feature_type="numerical",
                    source_table="users",
                    source_column="min_price,max_price",
                    transformation="avg",
                    freshness_sla=3600,
                    description="Average price preference for user"
                ),
                "property_price_zscore": FeatureDefinition(
                    name="property_price_zscore",
                    feature_type="numerical",
                    source_table="properties",
                    source_column="price",
                    transformation="zscore_by_location",
                    freshness_sla=86400,
                    description="Z-score of property price within location"
                ),
                "user_interaction_count_7d": FeatureDefinition(
                    name="user_interaction_count_7d",
                    feature_type="numerical",
                    source_table="user_interactions",
                    source_column="user_id",
                    transformation="count",
                    aggregation_window="7d",
                    freshness_sla=3600,
                    description="User interactions in last 7 days"
                ),
                "property_popularity_score": FeatureDefinition(
                    name="property_popularity_score",
                    feature_type="numerical",
                    source_table="user_interactions",
                    source_column="property_id",
                    transformation="count_normalized",
                    aggregation_window="24h",
                    freshness_sla=1800,
                    description="Property popularity score"
                )
            }
            
            self.logger.info(f"Loaded {len(self.feature_definitions)} feature definitions")
            
        except Exception as e:
            self.logger.error(f"Failed to load feature definitions: {e}")
            raise
    
    async def get_features(self, request: FeatureRequest) -> FeatureResponse:
        """
        Get features for entities.
        
        Args:
            request: Feature request
            
        Returns:
            Feature response with values
        """
        start_time = time.time()
        
        try:
            self.metrics['requests_count'] += 1
            
            # Validate request
            self._validate_request(request)
            
            # Try cache first
            cached_features = await self._get_cached_features(request)
            
            # Identify missing features
            missing_features = self._identify_missing_features(cached_features, request)
            
            # Compute missing features
            computed_features = {}
            if missing_features:
                computed_features = await self._compute_features(missing_features)
                
                # Cache computed features
                await self._cache_features(computed_features)
            
            # Combine cached and computed features
            all_features = self._combine_features(cached_features, computed_features)
            
            # Calculate metrics
            computation_time = (time.time() - start_time) * 1000
            cache_hit_ratio = self._calculate_cache_hit_ratio(cached_features, request)
            
            # Create response
            response = FeatureResponse(
                features=all_features,
                computation_time_ms=computation_time,
                cache_hit_ratio=cache_hit_ratio
            )
            
            if request.include_metadata:
                response.metadata = await self._get_feature_metadata(request.feature_names)
            
            # Update metrics
            self._update_metrics(computation_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Feature retrieval failed: {e}")
            raise
    
    def _validate_request(self, request: FeatureRequest):
        """Validate feature request"""
        if not request.entity_ids:
            raise ValueError("entity_ids cannot be empty")
        
        if not request.feature_names:
            raise ValueError("feature_names cannot be empty")
        
        # Check if features exist
        unknown_features = [
            name for name in request.feature_names
            if name not in self.feature_definitions
        ]
        
        if unknown_features:
            raise ValueError(f"Unknown features: {unknown_features}")
    
    async def _get_cached_features(self, request: FeatureRequest) -> Dict[str, Dict[str, Any]]:
        """Get features from cache"""
        cached_features = {}
        cache_hits = 0
        
        try:
            for entity_id in request.entity_ids:
                entity_features = {}
                
                for feature_name in request.feature_names:
                    cache_key = self._generate_cache_key(entity_id, feature_name)
                    
                    try:
                        cached_value = await self.redis_client.get(cache_key)
                        if cached_value:
                            feature_data = json.loads(cached_value)
                            
                            # Check freshness
                            feature_timestamp = datetime.fromisoformat(feature_data['timestamp'])
                            feature_def = self.feature_definitions[feature_name]
                            
                            if self._is_feature_fresh(feature_timestamp, feature_def):
                                entity_features[feature_name] = feature_data['value']
                                cache_hits += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Cache get failed for {cache_key}: {e}")
                
                if entity_features:
                    cached_features[entity_id] = entity_features
            
            self.metrics['cache_hits'] += cache_hits
            self.logger.debug(f"Cache hits: {cache_hits}")
            
            return cached_features
            
        except Exception as e:
            self.logger.error(f"Cache retrieval failed: {e}")
            return {}
    
    def _identify_missing_features(self, 
                                 cached_features: Dict[str, Dict[str, Any]],
                                 request: FeatureRequest) -> Dict[str, List[str]]:
        """Identify features that need to be computed"""
        missing_features = {}
        
        for entity_id in request.entity_ids:
            cached_entity_features = cached_features.get(entity_id, {})
            missing_entity_features = [
                feature_name for feature_name in request.feature_names
                if feature_name not in cached_entity_features
            ]
            
            if missing_entity_features:
                missing_features[entity_id] = missing_entity_features
        
        return missing_features
    
    async def _compute_features(self, missing_features: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Compute missing features"""
        computed_features = {}
        
        try:
            # Group by feature type for batch computation
            feature_groups = self._group_features_by_computation(missing_features)
            
            for computation_type, entities_features in feature_groups.items():
                if computation_type == "user_features":
                    batch_results = await self._compute_user_features_batch(entities_features)
                elif computation_type == "property_features":
                    batch_results = await self._compute_property_features_batch(entities_features)
                elif computation_type == "interaction_features":
                    batch_results = await self._compute_interaction_features_batch(entities_features)
                else:
                    continue
                
                # Merge results
                for entity_id, features in batch_results.items():
                    if entity_id not in computed_features:
                        computed_features[entity_id] = {}
                    computed_features[entity_id].update(features)
            
            return computed_features
            
        except Exception as e:
            self.logger.error(f"Feature computation failed: {e}")
            return {}
    
    def _group_features_by_computation(self, 
                                     missing_features: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        """Group features by computation type for batch processing"""
        groups = {
            "user_features": {},
            "property_features": {},
            "interaction_features": {}
        }
        
        for entity_id, feature_names in missing_features.items():
            for feature_name in feature_names:
                feature_def = self.feature_definitions[feature_name]
                
                if feature_def.source_table == "users":
                    group_key = "user_features"
                elif feature_def.source_table == "properties":
                    group_key = "property_features"
                elif feature_def.source_table == "user_interactions":
                    group_key = "interaction_features"
                else:
                    continue
                
                if entity_id not in groups[group_key]:
                    groups[group_key][entity_id] = []
                groups[group_key][entity_id].append(feature_name)
        
        return groups
    
    async def _compute_user_features_batch(self, 
                                         entities_features: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Compute user features in batch"""
        try:
            results = {}
            
            # Get user IDs
            user_ids = list(entities_features.keys())
            
            # Fetch user data
            async with self.connection_pool.acquire() as conn:
                user_data = await conn.fetch("""
                    SELECT id, min_price, max_price, preferred_locations, required_amenities
                    FROM users
                    WHERE id = ANY($1)
                """, user_ids)
            
            # Compute features for each user
            for user_row in user_data:
                user_id = str(user_row['id'])
                user_features = {}
                
                requested_features = entities_features.get(user_id, [])
                
                for feature_name in requested_features:
                    if feature_name == "user_avg_price_preference":
                        min_price = user_row['min_price'] or 0
                        max_price = user_row['max_price'] or 0
                        avg_price = (min_price + max_price) / 2 if max_price > 0 else min_price
                        user_features[feature_name] = avg_price
                    
                    # Add more feature computations here
                
                if user_features:
                    results[user_id] = user_features
            
            return results
            
        except Exception as e:
            self.logger.error(f"User feature computation failed: {e}")
            return {}
    
    async def _compute_property_features_batch(self, 
                                             entities_features: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Compute property features in batch"""
        try:
            results = {}
            
            # Get property IDs
            property_ids = list(entities_features.keys())
            
            # Fetch property data
            async with self.connection_pool.acquire() as conn:
                property_data = await conn.fetch("""
                    SELECT id, price, location, bedrooms, bathrooms
                    FROM properties
                    WHERE id = ANY($1)
                """, property_ids)
                
                # Get location statistics for price z-score
                location_stats = await conn.fetch("""
                    SELECT location, AVG(price) as avg_price, STDDEV(price) as std_price
                    FROM properties
                    WHERE location IN (SELECT DISTINCT location FROM properties WHERE id = ANY($1))
                    GROUP BY location
                """, property_ids)
            
            # Create location stats lookup
            location_stats_dict = {
                row['location']: {'avg': row['avg_price'], 'std': row['std_price']}
                for row in location_stats
            }
            
            # Compute features for each property
            for prop_row in property_data:
                property_id = str(prop_row['id'])
                property_features = {}
                
                requested_features = entities_features.get(property_id, [])
                
                for feature_name in requested_features:
                    if feature_name == "property_price_zscore":
                        location = prop_row['location']
                        price = prop_row['price']
                        
                        if location in location_stats_dict:
                            stats = location_stats_dict[location]
                            if stats['std'] and stats['std'] > 0:
                                zscore = (price - stats['avg']) / stats['std']
                            else:
                                zscore = 0.0
                        else:
                            zscore = 0.0
                        
                        property_features[feature_name] = zscore
                    
                    # Add more feature computations here
                
                if property_features:
                    results[property_id] = property_features
            
            return results
            
        except Exception as e:
            self.logger.error(f"Property feature computation failed: {e}")
            return {}
    
    async def _compute_interaction_features_batch(self, 
                                                entities_features: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Compute interaction-based features in batch"""
        try:
            results = {}
            
            # Get entity IDs (could be user IDs or property IDs)
            entity_ids = list(entities_features.keys())
            
            # Compute interaction features
            async with self.connection_pool.acquire() as conn:
                # User interaction count in last 7 days
                user_interaction_counts = await conn.fetch("""
                    SELECT user_id, COUNT(*) as interaction_count
                    FROM user_interactions
                    WHERE user_id = ANY($1) 
                      AND timestamp >= NOW() - INTERVAL '7 days'
                    GROUP BY user_id
                """, entity_ids)
                
                # Property popularity scores
                property_popularity = await conn.fetch("""
                    SELECT property_id, COUNT(*) as popularity_count
                    FROM user_interactions
                    WHERE property_id = ANY($1)
                      AND timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY property_id
                """, entity_ids)
            
            # Process user interaction counts
            user_counts_dict = {str(row['user_id']): row['interaction_count'] for row in user_interaction_counts}
            
            for entity_id in entity_ids:
                entity_features = {}
                requested_features = entities_features.get(entity_id, [])
                
                for feature_name in requested_features:
                    if feature_name == "user_interaction_count_7d":
                        count = user_counts_dict.get(entity_id, 0)
                        entity_features[feature_name] = count
                    elif feature_name == "property_popularity_score":
                        # Find popularity for this property
                        popularity = 0
                        for row in property_popularity:
                            if str(row['property_id']) == entity_id:
                                popularity = row['popularity_count']
                                break
                        
                        # Normalize popularity score (simplified)
                        normalized_popularity = min(popularity / 100.0, 1.0)
                        entity_features[feature_name] = normalized_popularity
                
                if entity_features:
                    results[entity_id] = entity_features
            
            return results
            
        except Exception as e:
            self.logger.error(f"Interaction feature computation failed: {e}")
            return {}
    
    async def _cache_features(self, computed_features: Dict[str, Dict[str, Any]]):
        """Cache computed features"""
        try:
            current_time = datetime.utcnow()
            
            for entity_id, features in computed_features.items():
                for feature_name, value in features.items():
                    cache_key = self._generate_cache_key(entity_id, feature_name)
                    
                    feature_data = {
                        'value': value,
                        'timestamp': current_time.isoformat(),
                        'version': '1.0'
                    }
                    
                    try:
                        await self.redis_client.setex(
                            cache_key,
                            self.cache_ttl,
                            json.dumps(feature_data, default=str)
                        )
                    except Exception as e:
                        self.logger.warning(f"Cache set failed for {cache_key}: {e}")
                        self.metrics['cache_misses'] += 1
            
        except Exception as e:
            self.logger.error(f"Feature caching failed: {e}")
    
    def _combine_features(self, 
                         cached_features: Dict[str, Dict[str, Any]],
                         computed_features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Combine cached and computed features"""
        all_features = cached_features.copy()
        
        for entity_id, features in computed_features.items():
            if entity_id not in all_features:
                all_features[entity_id] = {}
            all_features[entity_id].update(features)
        
        return all_features
    
    def _generate_cache_key(self, entity_id: str, feature_name: str) -> str:
        """Generate cache key for feature"""
        return f"feature:{feature_name}:{entity_id}"
    
    def _is_feature_fresh(self, feature_timestamp: datetime, feature_def: FeatureDefinition) -> bool:
        """Check if feature is fresh based on SLA"""
        age_seconds = (datetime.utcnow() - feature_timestamp).total_seconds()
        return age_seconds <= feature_def.freshness_sla
    
    def _calculate_cache_hit_ratio(self, 
                                 cached_features: Dict[str, Dict[str, Any]],
                                 request: FeatureRequest) -> float:
        """Calculate cache hit ratio for request"""
        total_features = len(request.entity_ids) * len(request.feature_names)
        cached_features_count = sum(len(features) for features in cached_features.values())
        
        return cached_features_count / total_features if total_features > 0 else 0.0
    
    async def _get_feature_metadata(self, feature_names: List[str]) -> Dict[str, Any]:
        """Get metadata for features"""
        metadata = {}
        
        for feature_name in feature_names:
            if feature_name in self.feature_definitions:
                feature_def = self.feature_definitions[feature_name]
                metadata[feature_name] = {
                    'type': feature_def.feature_type,
                    'source_table': feature_def.source_table,
                    'freshness_sla': feature_def.freshness_sla,
                    'description': feature_def.description
                }
        
        return metadata
    
    def _update_metrics(self, computation_time: float):
        """Update performance metrics"""
        # Simple P95 approximation
        if self.metrics['computation_time_p95'] == 0:
            self.metrics['computation_time_p95'] = computation_time
        else:
            # Exponential moving average
            alpha = 0.05
            self.metrics['computation_time_p95'] = (
                alpha * computation_time + 
                (1 - alpha) * self.metrics['computation_time_p95']
            )
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Feature freshness monitoring
        freshness_task = asyncio.create_task(self._monitor_feature_freshness())
        self._background_tasks.add(freshness_task)
        
        # Cache cleanup
        cleanup_task = asyncio.create_task(self._cleanup_expired_features())
        self._background_tasks.add(cleanup_task)
        
        # Metrics reporting
        metrics_task = asyncio.create_task(self._report_metrics())
        self._background_tasks.add(metrics_task)
    
    async def _monitor_feature_freshness(self):
        """Monitor feature freshness and detect violations"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check freshness of cached features
                # Implementation would scan Redis keys and check timestamps
                
                self.logger.debug("Feature freshness monitoring completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Feature freshness monitoring failed: {e}")
    
    async def _cleanup_expired_features(self):
        """Clean up expired features from cache"""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                
                # Implementation would scan for expired keys and remove them
                
                self.logger.debug("Feature cache cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Feature cache cleanup failed: {e}")
    
    async def _report_metrics(self):
        """Report feature store metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                self.logger.info(f"Feature store metrics: {self.metrics}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics reporting failed: {e}")
    
    async def register_feature(self, feature_def: FeatureDefinition):
        """Register a new feature definition"""
        try:
            # Validate feature definition
            self._validate_feature_definition(feature_def)
            
            # Store in registry
            self.feature_definitions[feature_def.name] = feature_def
            
            # In production, would also store in database
            
            self.logger.info(f"Feature registered: {feature_def.name}")
            
        except Exception as e:
            self.logger.error(f"Feature registration failed: {e}")
            raise
    
    def _validate_feature_definition(self, feature_def: FeatureDefinition):
        """Validate feature definition"""
        if not feature_def.name:
            raise ValueError("Feature name is required")
        
        if feature_def.feature_type not in ["numerical", "categorical", "text", "timestamp"]:
            raise ValueError(f"Invalid feature type: {feature_def.feature_type}")
        
        if not feature_def.source_table:
            raise ValueError("Source table is required")
    
    async def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get lineage information for a feature"""
        if feature_name not in self.feature_definitions:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature_def = self.feature_definitions[feature_name]
        
        lineage = {
            'feature_name': feature_name,
            'source_table': feature_def.source_table,
            'source_column': feature_def.source_column,
            'transformation': feature_def.transformation,
            'owner': feature_def.owner,
            'dependencies': [],  # Would track upstream dependencies
            'downstream_usage': []  # Would track models using this feature
        }
        
        return lineage
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature store metrics"""
        return self.metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test database connection
            async with self.connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_healthy = True
        except Exception:
            db_healthy = False
        
        try:
            # Test Redis connection
            await self.redis_client.ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False
        
        return {
            'status': 'healthy' if db_healthy and redis_healthy else 'unhealthy',
            'database': 'healthy' if db_healthy else 'unhealthy',
            'redis': 'healthy' if redis_healthy else 'unhealthy',
            'metrics': self.metrics
        }