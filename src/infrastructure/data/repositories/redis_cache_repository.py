import asyncio
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import Redis
from uuid import UUID

from ....domain.repositories.model_repository import ModelRepository


class RedisCacheRepository(ModelRepository):
    """Redis-based caching repository for ML predictions and general data"""
    
    def __init__(self, redis_client: Redis, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)
        
        # Cache key prefixes
        self.prefixes = {
            'predictions': 'pred:',
            'recommendations': 'rec:',
            'search_results': 'search:',
            'user_data': 'user:',
            'property_data': 'prop:',
            'model_cache': 'model:',
            'similarity': 'sim:',
            'analytics': 'analytics:'
        }
    
    async def get_cached_predictions(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached predictions/recommendations"""
        try:
            full_key = self.prefixes['predictions'] + cache_key
            cached_data = await self.redis.get(full_key)
            
            if cached_data:
                data = json.loads(cached_data)
                
                # Check if cache has expired based on timestamp
                if self._is_cache_valid(data):
                    self.logger.debug(f"Cache hit for key: {cache_key}")
                    return data.get('payload')
                else:
                    # Remove expired cache
                    await self.redis.delete(full_key)
                    self.logger.debug(f"Cache expired for key: {cache_key}")
            
            self.logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cached predictions for {cache_key}: {e}")
            return None
    
    async def cache_predictions(self, cache_key: str, predictions: Any, 
                              ttl_seconds: Optional[int] = None) -> bool:
        """Cache predictions/recommendations with TTL"""
        try:
            ttl = ttl_seconds or self.default_ttl
            full_key = self.prefixes['predictions'] + cache_key
            
            # Prepare cache data with metadata
            cache_data = {
                'payload': predictions,
                'cached_at': datetime.utcnow().isoformat(),
                'ttl': ttl,
                'key': cache_key
            }
            
            # Serialize and cache
            serialized_data = json.dumps(cache_data, default=self._json_serializer)
            
            success = await self.redis.setex(
                full_key,
                ttl,
                serialized_data
            )
            
            if success:
                self.logger.debug(f"Cached predictions for key: {cache_key}, TTL: {ttl}s")
                return True
            else:
                self.logger.warning(f"Failed to cache predictions for key: {cache_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cache predictions for {cache_key}: {e}")
            return False
    
    async def clear_cache(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        try:
            # Get all keys matching pattern
            keys = await self.redis.keys(pattern)
            
            if keys:
                deleted_count = await self.redis.delete(*keys)
                self.logger.info(f"Cleared {deleted_count} cache entries matching pattern: {pattern}")
                return deleted_count
            else:
                self.logger.debug(f"No cache entries found for pattern: {pattern}")
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to clear cache with pattern {pattern}: {e}")
            return 0
    
    async def cache_search_results(self, query_hash: str, results: List[Dict], 
                                 ttl_seconds: int = 300) -> bool:
        """Cache search results with shorter TTL"""
        try:
            full_key = self.prefixes['search_results'] + query_hash
            
            cache_data = {
                'results': results,
                'result_count': len(results),
                'cached_at': datetime.utcnow().isoformat(),
                'ttl': ttl_seconds
            }
            
            serialized_data = json.dumps(cache_data, default=self._json_serializer)
            
            success = await self.redis.setex(full_key, ttl_seconds, serialized_data)
            
            if success:
                self.logger.debug(f"Cached {len(results)} search results, TTL: {ttl_seconds}s")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cache search results: {e}")
            return False
    
    async def get_cached_search_results(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached search results"""
        try:
            full_key = self.prefixes['search_results'] + query_hash
            cached_data = await self.redis.get(full_key)
            
            if cached_data:
                data = json.loads(cached_data)
                if self._is_cache_valid(data):
                    self.logger.debug(f"Search cache hit for query hash: {query_hash}")
                    return data.get('results', [])
                else:
                    await self.redis.delete(full_key)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cached search results: {e}")
            return None
    
    async def cache_user_data(self, user_id: UUID, data: Dict, 
                            ttl_seconds: int = 1800) -> bool:
        """Cache user-specific data (30 min default TTL)"""
        try:
            full_key = self.prefixes['user_data'] + str(user_id)
            
            cache_data = {
                'user_id': str(user_id),
                'data': data,
                'cached_at': datetime.utcnow().isoformat(),
                'ttl': ttl_seconds
            }
            
            serialized_data = json.dumps(cache_data, default=self._json_serializer)
            
            success = await self.redis.setex(full_key, ttl_seconds, serialized_data)
            
            if success:
                self.logger.debug(f"Cached user data for {user_id}, TTL: {ttl_seconds}s")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cache user data for {user_id}: {e}")
            return False
    
    async def get_cached_user_data(self, user_id: UUID) -> Optional[Dict]:
        """Get cached user data"""
        try:
            full_key = self.prefixes['user_data'] + str(user_id)
            cached_data = await self.redis.get(full_key)
            
            if cached_data:
                data = json.loads(cached_data)
                if self._is_cache_valid(data):
                    return data.get('data')
                else:
                    await self.redis.delete(full_key)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cached user data for {user_id}: {e}")
            return None
    
    async def cache_similarity_matrix(self, matrix_type: str, matrix_data: Any,
                                    ttl_seconds: int = 7200) -> bool:
        """Cache similarity matrices (2 hour TTL)"""
        try:
            full_key = self.prefixes['similarity'] + matrix_type
            
            # Use pickle for complex numpy arrays
            cache_data = {
                'matrix_type': matrix_type,
                'matrix_data': matrix_data,
                'cached_at': datetime.utcnow().isoformat(),
                'ttl': ttl_seconds
            }
            
            # Serialize with pickle for better numpy support
            serialized_data = pickle.dumps(cache_data)
            
            success = await self.redis.setex(full_key, ttl_seconds, serialized_data)
            
            if success:
                self.logger.debug(f"Cached similarity matrix: {matrix_type}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cache similarity matrix {matrix_type}: {e}")
            return False
    
    async def get_cached_similarity_matrix(self, matrix_type: str) -> Optional[Any]:
        """Get cached similarity matrix"""
        try:
            full_key = self.prefixes['similarity'] + matrix_type
            cached_data = await self.redis.get(full_key)
            
            if cached_data:
                data = pickle.loads(cached_data)
                
                # Check cache validity
                cached_at = datetime.fromisoformat(data['cached_at'])
                ttl_seconds = data.get('ttl', self.default_ttl)
                
                if datetime.utcnow() - cached_at < timedelta(seconds=ttl_seconds):
                    return data.get('matrix_data')
                else:
                    await self.redis.delete(full_key)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cached similarity matrix {matrix_type}: {e}")
            return None
    
    async def increment_counter(self, counter_name: str, increment: int = 1,
                              ttl_seconds: Optional[int] = None) -> int:
        """Increment a counter with optional TTL"""
        try:
            full_key = self.prefixes['analytics'] + counter_name
            
            # Use pipeline for atomic operations
            async with self.redis.pipeline() as pipe:
                pipe.multi()
                pipe.incr(full_key, increment)
                
                if ttl_seconds:
                    pipe.expire(full_key, ttl_seconds)
                
                results = await pipe.execute()
                
                new_value = results[0]
                self.logger.debug(f"Incremented counter {counter_name} to {new_value}")
                
                return new_value
                
        except Exception as e:
            self.logger.error(f"Failed to increment counter {counter_name}: {e}")
            return 0
    
    async def get_counter(self, counter_name: str) -> int:
        """Get current counter value"""
        try:
            full_key = self.prefixes['analytics'] + counter_name
            value = await self.redis.get(full_key)
            
            return int(value) if value else 0
            
        except Exception as e:
            self.logger.error(f"Failed to get counter {counter_name}: {e}")
            return 0
    
    async def cache_batch_data(self, batch_data: Dict[str, Any], 
                             prefix: str = 'batch:', ttl_seconds: int = 3600) -> bool:
        """Cache multiple items in a batch operation"""
        try:
            async with self.redis.pipeline() as pipe:
                pipe.multi()
                
                for key, value in batch_data.items():
                    full_key = prefix + key
                    cache_data = {
                        'payload': value,
                        'cached_at': datetime.utcnow().isoformat(),
                        'ttl': ttl_seconds
                    }
                    
                    serialized_data = json.dumps(cache_data, default=self._json_serializer)
                    pipe.setex(full_key, ttl_seconds, serialized_data)
                
                results = await pipe.execute()
                
                success_count = sum(1 for result in results if result)
                
                self.logger.debug(f"Batch cached {success_count}/{len(batch_data)} items")
                
                return success_count == len(batch_data)
                
        except Exception as e:
            self.logger.error(f"Failed to batch cache data: {e}")
            return False
    
    async def get_batch_data(self, keys: List[str], 
                           prefix: str = 'batch:') -> Dict[str, Any]:
        """Get multiple cached items in a batch operation"""
        try:
            full_keys = [prefix + key for key in keys]
            cached_values = await self.redis.mget(*full_keys)
            
            result = {}
            
            for key, cached_value in zip(keys, cached_values):
                if cached_value:
                    try:
                        data = json.loads(cached_value)
                        if self._is_cache_valid(data):
                            result[key] = data.get('payload')
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            self.logger.debug(f"Batch retrieved {len(result)}/{len(keys)} items")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to batch get data: {e}")
            return {}
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = await self.redis.info()
            
            # Get key counts by prefix
            key_counts = {}
            for prefix_name, prefix in self.prefixes.items():
                keys = await self.redis.keys(f"{prefix}*")
                key_counts[prefix_name] = len(keys)
            
            stats = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'key_counts_by_type': key_counts
            }
            
            # Calculate hit rate
            hits = stats['keyspace_hits']
            misses = stats['keyspace_misses']
            total_requests = hits + misses
            
            if total_requests > 0:
                stats['hit_rate'] = hits / total_requests
            else:
                stats['hit_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    async def warm_cache(self, warm_data: Dict[str, Any]) -> bool:
        """Warm up cache with frequently accessed data"""
        try:
            success_count = 0
            
            for cache_type, data_items in warm_data.items():
                if cache_type in self.prefixes:
                    prefix = self.prefixes[cache_type]
                    
                    for key, value in data_items.items():
                        success = await self.cache_predictions(
                            key, value, ttl_seconds=self.default_ttl * 2
                        )
                        if success:
                            success_count += 1
            
            self.logger.info(f"Cache warmed with {success_count} items")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to warm cache: {e}")
            return False
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired keys (Redis handles this automatically, but useful for monitoring)"""
        try:
            deleted_count = 0
            
            # Check each prefix for expired keys
            for prefix_name, prefix in self.prefixes.items():
                pattern = f"{prefix}*"
                keys = await self.redis.keys(pattern)
                
                # Check TTL for each key
                for key in keys:
                    ttl = await self.redis.ttl(key)
                    if ttl == -1:  # No expiration set
                        # Set default expiration
                        await self.redis.expire(key, self.default_ttl)
                    elif ttl == -2:  # Key doesn't exist (race condition)
                        deleted_count += 1
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} expired keys")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired keys: {e}")
            return 0
    
    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """Check if cached data is still valid based on TTL"""
        try:
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            ttl_seconds = cache_data.get('ttl', self.default_ttl)
            
            return datetime.utcnow() - cached_at < timedelta(seconds=ttl_seconds)
            
        except (KeyError, ValueError):
            return False
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return False