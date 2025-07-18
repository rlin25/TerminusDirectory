"""
Inference Cache for Production ML Serving Infrastructure.

This module provides high-performance caching for ML inference results including:
- Redis-based distributed caching
- TTL-based cache expiration
- Cache hit rate monitoring
- Intelligent cache warming
- Cache invalidation strategies
- Memory-efficient serialization
"""

import asyncio
import logging
import json
import time
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pickle
import hashlib

import aioredis
import redis
from prometheus_client import Counter, Histogram, Gauge
import msgpack


# Prometheus metrics
CACHE_HITS = Counter('ml_cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('ml_cache_misses_total', 'Total cache misses')
CACHE_SETS = Counter('ml_cache_sets_total', 'Total cache sets')
CACHE_OPERATIONS_TIME = Histogram('ml_cache_operation_duration_seconds', 'Cache operation time', ['operation'])
CACHE_SIZE = Gauge('ml_cache_size_bytes', 'Total cache size in bytes')


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    compression: str = "none"  # "none", "gzip", "lz4"
    size_bytes: int = 0


class SerializationManager:
    """Manages serialization and compression for cache entries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def serialize(self, data: Any, compression: str = "gzip") -> bytes:
        """Serialize data with optional compression"""
        try:
            # Use msgpack for efficient serialization
            serialized = msgpack.packb(data, default=self._msgpack_encoder)
            
            if compression == "gzip":
                serialized = zlib.compress(serialized)
            elif compression == "lz4":
                try:
                    import lz4.frame
                    serialized = lz4.frame.compress(serialized)
                except ImportError:
                    self.logger.warning("lz4 not available, using gzip")
                    serialized = zlib.compress(serialized)
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            # Fallback to pickle
            return pickle.dumps(data)
    
    def deserialize(self, data: bytes, compression: str = "gzip") -> Any:
        """Deserialize data with decompression"""
        try:
            # Decompress if needed
            if compression == "gzip":
                data = zlib.decompress(data)
            elif compression == "lz4":
                try:
                    import lz4.frame
                    data = lz4.frame.decompress(data)
                except ImportError:
                    data = zlib.decompress(data)
            
            # Deserialize
            return msgpack.unpackb(data, raw=False, object_hook=self._msgpack_decoder)
            
        except Exception as e:
            self.logger.error(f"Deserialization failed: {e}")
            # Fallback to pickle
            return pickle.loads(data)
    
    def _msgpack_encoder(self, obj):
        """Custom encoder for msgpack"""
        if isinstance(obj, datetime):
            return {'__datetime__': obj.isoformat()}
        elif hasattr(obj, '__dict__'):
            return {'__object__': obj.__dict__, '__class__': obj.__class__.__name__}
        raise TypeError(f"Object of type {type(obj)} is not serializable")
    
    def _msgpack_decoder(self, obj):
        """Custom decoder for msgpack"""
        if '__datetime__' in obj:
            return datetime.fromisoformat(obj['__datetime__'])
        elif '__object__' in obj:
            # Simple object reconstruction - extend as needed
            return obj['__object__']
        return obj


class CacheStrategy:
    """Cache strategy interface"""
    
    def should_cache(self, key: str, data: Any, metadata: Dict[str, Any]) -> bool:
        """Determine if data should be cached"""
        raise NotImplementedError
    
    def get_ttl(self, key: str, data: Any, metadata: Dict[str, Any]) -> int:
        """Get TTL for cache entry"""
        raise NotImplementedError
    
    def get_compression(self, key: str, data: Any, metadata: Dict[str, Any]) -> str:
        """Get compression type for cache entry"""
        raise NotImplementedError


class AdaptiveCacheStrategy(CacheStrategy):
    """Adaptive caching strategy based on data characteristics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def should_cache(self, key: str, data: Any, metadata: Dict[str, Any]) -> bool:
        """Cache if inference time is significant or data is frequently accessed"""
        inference_time = metadata.get('inference_time_ms', 0)
        
        # Cache if inference took > 100ms
        if inference_time > 100:
            return True
        
        # Cache recommendations for users (expensive to compute)
        if 'user_id' in key and 'recommendations' in str(data):
            return True
        
        # Cache search results
        if 'search' in key.lower():
            return True
        
        return False
    
    def get_ttl(self, key: str, data: Any, metadata: Dict[str, Any]) -> int:
        """Get TTL based on data type and characteristics"""
        model_type = metadata.get('model_type', 'unknown')
        
        # Longer TTL for expensive computations
        if model_type == 'hybrid':
            return 600  # 10 minutes
        elif model_type in ['collaborative', 'content']:
            return 300  # 5 minutes
        elif model_type == 'search_ranker':
            return 180  # 3 minutes
        
        return 300  # Default 5 minutes
    
    def get_compression(self, key: str, data: Any, metadata: Dict[str, Any]) -> str:
        """Get compression based on data size"""
        try:
            # Estimate data size
            data_str = str(data)
            if len(data_str) > 10000:  # Large data
                return "gzip"
            elif len(data_str) > 1000:  # Medium data
                return "lz4"
            else:
                return "none"
        except Exception:
            return "gzip"


class InferenceCache:
    """
    High-performance inference cache for ML serving.
    
    Features:
    - Redis-based distributed caching
    - Intelligent cache warming and eviction
    - Compression and serialization optimization
    - Cache hit rate monitoring
    - TTL-based expiration
    - Adaptive caching strategies
    """
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 default_ttl: int = 300,
                 key_prefix: str = "ml_cache:",
                 max_key_length: int = 250,
                 enable_compression: bool = True):
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.max_key_length = max_key_length
        self.enable_compression = enable_compression
        
        self.logger = logging.getLogger(__name__)
        
        # Redis clients
        self.redis_client = None
        self.redis_pool = None
        
        # Components
        self.serializer = SerializationManager()
        self.strategy = AdaptiveCacheStrategy()
        
        # Metrics
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0
        
        # Cache warming
        self.warming_tasks = {}
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            self.redis_client = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            self.logger.info("Inference cache initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            # Use in-memory fallback
            self.redis_client = None
            raise
    
    async def close(self):
        """Close Redis connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.redis_pool:
                await self.redis_pool.disconnect()
                
            self.logger.info("Inference cache closed")
            
        except Exception as e:
            self.logger.error(f"Error closing cache: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        start_time = time.time()
        
        try:
            # Normalize key
            cache_key = self._normalize_key(key)
            
            # Get from Redis
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                # Deserialize
                entry = self._deserialize_entry(cached_data)
                
                # Update access metadata
                await self._update_access_metadata(cache_key, entry)
                
                # Update metrics
                self.hit_count += 1
                CACHE_HITS.inc()
                
                operation_time = time.time() - start_time
                CACHE_OPERATIONS_TIME.labels(operation='get').observe(operation_time)
                
                return entry.data
            else:
                # Cache miss
                self.miss_count += 1
                CACHE_MISSES.inc()
                
                return None
                
        except Exception as e:
            self.logger.error(f"Cache get failed for key {key}: {e}")
            return None
        
        finally:
            self.total_requests += 1
    
    async def set(self, 
                  key: str, 
                  data: Any, 
                  ttl: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set value in cache"""
        if not self.redis_client:
            return False
        
        start_time = time.time()
        
        try:
            metadata = metadata or {}
            
            # Check if should cache
            if not self.strategy.should_cache(key, data, metadata):
                return False
            
            # Determine TTL
            if ttl is None:
                ttl = self.strategy.get_ttl(key, data, metadata)
            
            # Determine compression
            compression = "none"
            if self.enable_compression:
                compression = self.strategy.get_compression(key, data, metadata)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl,
                compression=compression
            )
            
            # Serialize entry
            serialized_data = self._serialize_entry(entry)
            entry.size_bytes = len(serialized_data)
            
            # Update size metric
            CACHE_SIZE.inc(entry.size_bytes)
            
            # Store in Redis
            cache_key = self._normalize_key(key)
            await self.redis_client.setex(cache_key, ttl, serialized_data)
            
            # Update metrics
            CACHE_SETS.inc()
            
            operation_time = time.time() - start_time
            CACHE_OPERATIONS_TIME.labels(operation='set').observe(operation_time)
            
            self.logger.debug(f"Cached {cache_key} (TTL: {ttl}s, Size: {entry.size_bytes}B)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._normalize_key(key)
            result = await self.redis_client.delete(cache_key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._normalize_key(key)
            result = await self.redis_client.exists(cache_key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hit_count / self.total_requests
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = {
                'hit_rate': await self.get_hit_rate(),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'total_requests': self.total_requests,
                'redis_connected': self.redis_client is not None
            }
            
            if self.redis_client:
                # Get Redis info
                redis_info = await self.redis_client.info()
                stats.update({
                    'redis_memory_used': redis_info.get('used_memory', 0),
                    'redis_keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'redis_keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'redis_connected_clients': redis_info.get('connected_clients', 0)
                })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    async def warm_cache(self, 
                        warm_keys: List[str],
                        warm_function,
                        batch_size: int = 10) -> Dict[str, Any]:
        """Warm cache with frequently accessed data"""
        try:
            warmed_count = 0
            failed_count = 0
            
            # Process in batches
            for i in range(0, len(warm_keys), batch_size):
                batch = warm_keys[i:i + batch_size]
                
                # Process batch concurrently
                tasks = []
                for key in batch:
                    task = self._warm_single_key(key, warm_function)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        failed_count += 1
                        self.logger.error(f"Cache warming failed: {result}")
                    else:
                        warmed_count += 1
            
            return {
                'warmed_count': warmed_count,
                'failed_count': failed_count,
                'total_keys': len(warm_keys)
            }
            
        except Exception as e:
            self.logger.error(f"Cache warming failed: {e}")
            return {'error': str(e)}
    
    async def _warm_single_key(self, key: str, warm_function) -> bool:
        """Warm a single cache key"""
        try:
            # Check if already cached
            if await self.exists(key):
                return True
            
            # Generate data using warm function
            data = await warm_function(key)
            
            if data is not None:
                await self.set(key, data)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Single key warming failed for {key}: {e}")
            raise
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            # Find matching keys
            search_pattern = self._normalize_key(pattern)
            keys = await self.redis_client.keys(search_pattern)
            
            if keys:
                # Delete all matching keys
                deleted = await self.redis_client.delete(*keys)
                self.logger.info(f"Invalidated {deleted} cache entries matching pattern {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Cache invalidation failed for pattern {pattern}: {e}")
            return 0
    
    async def clear_expired(self) -> int:
        """Clear expired cache entries (Redis handles this automatically, but useful for monitoring)"""
        try:
            # Get info about expired keys
            redis_info = await self.redis_client.info()
            expired_keys = redis_info.get('expired_keys', 0)
            
            return expired_keys
            
        except Exception as e:
            self.logger.error(f"Failed to get expired keys count: {e}")
            return 0
    
    def _normalize_key(self, key: str) -> str:
        """Normalize cache key"""
        # Add prefix
        cache_key = f"{self.key_prefix}{key}"
        
        # Handle long keys
        if len(cache_key) > self.max_key_length:
            # Hash long keys
            key_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
            cache_key = f"{self.key_prefix}hash_{key_hash}"
        
        return cache_key
    
    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry"""
        try:
            # Convert to dict
            entry_dict = asdict(entry)
            
            # Serialize with compression if enabled
            return self.serializer.serialize(entry_dict, entry.compression)
            
        except Exception as e:
            self.logger.error(f"Entry serialization failed: {e}")
            raise
    
    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry"""
        try:
            # Deserialize
            entry_dict = self.serializer.deserialize(data, "gzip")  # Default to gzip for compatibility
            
            # Convert back to CacheEntry
            entry = CacheEntry(**entry_dict)
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Entry deserialization failed: {e}")
            raise
    
    async def _update_access_metadata(self, cache_key: str, entry: CacheEntry):
        """Update access metadata for cache entry"""
        try:
            # Update access count and timestamp
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            # Optionally update in Redis with extended TTL for frequently accessed items
            if entry.access_count % 10 == 0:  # Every 10th access
                # Extend TTL for popular items
                extended_ttl = min(entry.ttl_seconds * 2, 3600)  # Max 1 hour
                await self.redis_client.expire(cache_key, extended_ttl)
                
        except Exception as e:
            self.logger.warning(f"Failed to update access metadata: {e}")
    
    async def monitor_performance(self) -> Dict[str, Any]:
        """Monitor cache performance metrics"""
        try:
            stats = await self.get_stats()
            
            # Add performance indicators
            hit_rate = stats.get('hit_rate', 0)
            
            performance = {
                'status': 'healthy',
                'hit_rate': hit_rate,
                'performance_grade': 'A'  # A, B, C, D, F
            }
            
            # Determine performance grade
            if hit_rate >= 0.8:
                performance['performance_grade'] = 'A'
            elif hit_rate >= 0.6:
                performance['performance_grade'] = 'B'
            elif hit_rate >= 0.4:
                performance['performance_grade'] = 'C'
            elif hit_rate >= 0.2:
                performance['performance_grade'] = 'D'
            else:
                performance['performance_grade'] = 'F'
            
            # Check for issues
            if hit_rate < 0.3:
                performance['status'] = 'degraded'
                performance['recommendations'] = [
                    'Consider tuning cache TTL values',
                    'Review caching strategy',
                    'Check for cache invalidation patterns'
                ]
            
            if not stats.get('redis_connected', False):
                performance['status'] = 'unhealthy'
                performance['recommendations'] = ['Redis connection failed']
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }