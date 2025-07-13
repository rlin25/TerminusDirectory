"""
Edge cache manager for personalized content caching and edge-level data processing.

This module provides intelligent caching strategies for personalized content,
API responses, and frequently accessed data at edge locations.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum

from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheLevel(str, Enum):
    """Cache levels for different data types"""
    L1_MEMORY = "l1_memory"      # In-memory cache (fastest)
    L2_REDIS = "l2_redis"        # Redis cache (fast)
    L3_DISK = "l3_disk"          # Disk cache (slower but persistent)
    L4_ORIGIN = "l4_origin"      # Origin server (fallback)


class CacheStrategy(str, Enum):
    """Cache strategies"""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    TTL = "ttl"                  # Time To Live
    ADAPTIVE = "adaptive"        # Adaptive based on access patterns
    PERSONALIZED = "personalized"  # User-specific caching


class CacheScope(str, Enum):
    """Cache scope for different data types"""
    GLOBAL = "global"            # Shared across all users
    USER_SPECIFIC = "user"       # User-specific data
    REGION_SPECIFIC = "region"   # Region-specific data
    SESSION_SPECIFIC = "session" # Session-specific data


class CacheEntry(BaseModel):
    """Cache entry metadata"""
    key: str = Field(..., description="Cache key")
    data: Any = Field(..., description="Cached data")
    data_type: str = Field(..., description="Data type")
    cache_level: CacheLevel = Field(..., description="Cache level")
    cache_scope: CacheScope = Field(..., description="Cache scope")
    
    # Metadata
    size_bytes: int = Field(..., description="Data size in bytes")
    compressed: bool = Field(default=False, description="Data is compressed")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")
    
    # Access tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    hit_count: int = Field(default=0)
    miss_count: int = Field(default=0)
    
    # Personalization
    user_id: Optional[str] = Field(None, description="Associated user ID")
    region: Optional[str] = Field(None, description="Associated region")
    tags: List[str] = Field(default=[], description="Cache tags")


class CacheConfig(BaseModel):
    """Cache configuration"""
    # Memory cache settings
    l1_max_size_mb: int = Field(default=512, description="L1 cache max size in MB")
    l1_max_entries: int = Field(default=10000, description="L1 cache max entries")
    
    # Redis cache settings
    l2_max_size_mb: int = Field(default=2048, description="L2 cache max size in MB")
    l2_default_ttl: int = Field(default=3600, description="L2 default TTL in seconds")
    
    # Disk cache settings
    l3_max_size_gb: int = Field(default=10, description="L3 cache max size in GB")
    l3_base_path: str = Field(default="/tmp/edge_cache", description="L3 cache base path")
    
    # Strategy settings
    default_strategy: CacheStrategy = Field(default=CacheStrategy.ADAPTIVE)
    compression_enabled: bool = Field(default=True)
    compression_threshold: int = Field(default=1024, description="Compression threshold in bytes")
    
    # Personalization settings
    enable_personalized_cache: bool = Field(default=True)
    personalization_weight: float = Field(default=0.3, description="Weight for personalization in cache decisions")
    
    # Performance settings
    prefetch_enabled: bool = Field(default=True)
    background_eviction: bool = Field(default=True)
    batch_operations: bool = Field(default=True)


class CacheStats(BaseModel):
    """Cache statistics"""
    period_start: datetime = Field(..., description="Statistics period start")
    period_end: datetime = Field(..., description="Statistics period end")
    
    # Hit/miss statistics
    total_requests: int = Field(default=0)
    cache_hits: int = Field(default=0)
    cache_misses: int = Field(default=0)
    hit_ratio: float = Field(default=0.0)
    
    # Performance statistics
    avg_response_time_ms: float = Field(default=0.0)
    p95_response_time_ms: float = Field(default=0.0)
    evictions: int = Field(default=0)
    
    # Memory statistics
    memory_usage_mb: float = Field(default=0.0)
    memory_utilization: float = Field(default=0.0)
    
    # Level-specific statistics
    l1_hits: int = Field(default=0)
    l2_hits: int = Field(default=0)
    l3_hits: int = Field(default=0)
    origin_requests: int = Field(default=0)
    
    # Top cached items
    top_keys: List[Dict[str, Any]] = Field(default=[])
    popular_tags: List[Dict[str, Any]] = Field(default=[])


class EdgeCacheManager:
    """Edge cache manager for personalized content caching"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        
        # Multi-level cache storage
        self.l1_cache: Dict[str, CacheEntry] = {}  # In-memory cache
        self.l1_access_order: List[str] = []  # LRU tracking
        self.l1_size_bytes = 0
        
        # Cache statistics
        self.stats = CacheStats(
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Background tasks
        self._eviction_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._prefetch_task: Optional[asyncio.Task] = None
        
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the edge cache manager"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            
            # Create disk cache directory
            import os
            os.makedirs(self.config.l3_base_path, exist_ok=True)
            
            # Start background tasks
            if self.config.background_eviction:
                self._eviction_task = asyncio.create_task(self.background_eviction())
            
            self._stats_task = asyncio.create_task(self.update_stats_periodically())
            
            if self.config.prefetch_enabled:
                self._prefetch_task = asyncio.create_task(self.background_prefetch())
            
            logger.info("Edge cache manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize edge cache manager: {e}")
            raise
    
    async def get(
        self,
        key: str,
        user_id: Optional[str] = None,
        region: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """Get value from cache with multi-level fallback"""
        start_time = datetime.utcnow()
        
        try:
            # Generate personalized cache key if needed
            cache_key = self.generate_cache_key(key, user_id, region)
            
            # Level 1: Memory cache
            l1_result = await self.get_from_l1(cache_key)
            if l1_result is not None:
                self.stats.l1_hits += 1
                self.stats.cache_hits += 1
                await self.track_access(cache_key, CacheLevel.L1_MEMORY, start_time)
                return l1_result
            
            # Level 2: Redis cache
            l2_result = await self.get_from_l2(cache_key)
            if l2_result is not None:
                self.stats.l2_hits += 1
                self.stats.cache_hits += 1
                
                # Promote to L1 cache
                await self.set_to_l1(cache_key, l2_result, user_id, region)
                await self.track_access(cache_key, CacheLevel.L2_REDIS, start_time)
                return l2_result
            
            # Level 3: Disk cache
            l3_result = await self.get_from_l3(cache_key)
            if l3_result is not None:
                self.stats.l3_hits += 1
                self.stats.cache_hits += 1
                
                # Promote to L2 and L1 caches
                await self.set_to_l2(cache_key, l3_result, user_id, region)
                await self.set_to_l1(cache_key, l3_result, user_id, region)
                await self.track_access(cache_key, CacheLevel.L3_DISK, start_time)
                return l3_result
            
            # Cache miss
            self.stats.cache_misses += 1
            self.stats.origin_requests += 1
            await self.track_access(cache_key, CacheLevel.L4_ORIGIN, start_time)
            
            return default
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return default
        finally:
            self.stats.total_requests += 1
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        user_id: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[List[str]] = None,
        cache_levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """Set value in cache across multiple levels"""
        try:
            # Generate cache key
            cache_key = self.generate_cache_key(key, user_id, region)
            
            # Default to all cache levels
            if cache_levels is None:
                cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]
            
            # Set to specified cache levels
            success = True
            
            if CacheLevel.L1_MEMORY in cache_levels:
                success &= await self.set_to_l1(cache_key, value, user_id, region, ttl, tags)
            
            if CacheLevel.L2_REDIS in cache_levels:
                success &= await self.set_to_l2(cache_key, value, user_id, region, ttl, tags)
            
            if CacheLevel.L3_DISK in cache_levels:
                success &= await self.set_to_l3(cache_key, value, user_id, region, ttl, tags)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(
        self,
        key: str,
        user_id: Optional[str] = None,
        region: Optional[str] = None
    ) -> bool:
        """Delete value from all cache levels"""
        try:
            cache_key = self.generate_cache_key(key, user_id, region)
            
            # Delete from all levels
            l1_success = await self.delete_from_l1(cache_key)
            l2_success = await self.delete_from_l2(cache_key)
            l3_success = await self.delete_from_l3(cache_key)
            
            return l1_success or l2_success or l3_success
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        try:
            invalidated_count = 0
            
            # Invalidate from L1 cache
            keys_to_remove = []
            for cache_key, entry in self.l1_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                await self.delete_from_l1(cache_key)
                invalidated_count += 1
            
            # Invalidate from L2 cache
            for tag in tags:
                tag_key = f"cache_tag:{tag}"
                tagged_keys = await self.redis_client.smembers(tag_key)
                
                for cache_key in tagged_keys:
                    await self.delete_from_l2(cache_key.decode())
                    invalidated_count += 1
            
            logger.info(f"Invalidated {invalidated_count} cache entries by tags: {tags}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Tag-based invalidation failed: {e}")
            return 0
    
    def generate_cache_key(
        self,
        base_key: str,
        user_id: Optional[str] = None,
        region: Optional[str] = None
    ) -> str:
        """Generate cache key with personalization"""
        key_parts = [base_key]
        
        if user_id and self.config.enable_personalized_cache:
            key_parts.append(f"user:{user_id}")
        
        if region:
            key_parts.append(f"region:{region}")
        
        return ":".join(key_parts)
    
    # Level 1 (Memory) Cache Operations
    
    async def get_from_l1(self, cache_key: str) -> Any:
        """Get value from L1 memory cache"""
        try:
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                
                # Check TTL
                if entry.ttl_seconds:
                    age = (datetime.utcnow() - entry.created_at).total_seconds()
                    if age > entry.ttl_seconds:
                        await self.delete_from_l1(cache_key)
                        return None
                
                # Update access tracking
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                entry.hit_count += 1
                
                # Move to front for LRU
                if cache_key in self.l1_access_order:
                    self.l1_access_order.remove(cache_key)
                self.l1_access_order.append(cache_key)
                
                return entry.data
            
            return None
            
        except Exception as e:
            logger.error(f"L1 cache get failed: {e}")
            return None
    
    async def set_to_l1(
        self,
        cache_key: str,
        value: Any,
        user_id: Optional[str] = None,
        region: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value to L1 memory cache"""
        try:
            # Serialize and compress data
            data, size_bytes, compressed = self.serialize_data(value)
            
            # Check if we need to evict entries
            await self.ensure_l1_capacity(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                data=data,
                data_type=type(value).__name__,
                cache_level=CacheLevel.L1_MEMORY,
                cache_scope=self.determine_cache_scope(user_id, region),
                size_bytes=size_bytes,
                compressed=compressed,
                ttl_seconds=ttl,
                user_id=user_id,
                region=region,
                tags=tags or []
            )
            
            # Store in cache
            self.l1_cache[cache_key] = entry
            self.l1_access_order.append(cache_key)
            self.l1_size_bytes += size_bytes
            
            return True
            
        except Exception as e:
            logger.error(f"L1 cache set failed: {e}")
            return False
    
    async def delete_from_l1(self, cache_key: str) -> bool:
        """Delete value from L1 memory cache"""
        try:
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                self.l1_size_bytes -= entry.size_bytes
                
                del self.l1_cache[cache_key]
                
                if cache_key in self.l1_access_order:
                    self.l1_access_order.remove(cache_key)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"L1 cache delete failed: {e}")
            return False
    
    # Level 2 (Redis) Cache Operations
    
    async def get_from_l2(self, cache_key: str) -> Any:
        """Get value from L2 Redis cache"""
        try:
            data = await self.redis_client.get(f"cache:{cache_key}")
            if data:
                # Deserialize data
                entry_data = json.loads(data)
                value = self.deserialize_data(
                    entry_data["data"], 
                    entry_data["compressed"]
                )
                
                # Update access tracking
                await self.redis_client.hincrby(f"cache_meta:{cache_key}", "access_count", 1)
                await self.redis_client.hset(
                    f"cache_meta:{cache_key}", 
                    "last_accessed", 
                    datetime.utcnow().isoformat()
                )
                
                return value
            
            return None
            
        except Exception as e:
            logger.error(f"L2 cache get failed: {e}")
            return None
    
    async def set_to_l2(
        self,
        cache_key: str,
        value: Any,
        user_id: Optional[str] = None,
        region: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value to L2 Redis cache"""
        try:
            # Serialize and compress data
            data, size_bytes, compressed = self.serialize_data(value)
            
            # Create cache entry data
            entry_data = {
                "data": data,
                "data_type": type(value).__name__,
                "compressed": compressed,
                "size_bytes": size_bytes,
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "region": region,
                "tags": tags or []
            }
            
            # Set cache value
            cache_ttl = ttl or self.config.l2_default_ttl
            await self.redis_client.setex(
                f"cache:{cache_key}",
                cache_ttl,
                json.dumps(entry_data)
            )
            
            # Set metadata
            await self.redis_client.hmset(f"cache_meta:{cache_key}", {
                "created_at": datetime.utcnow().isoformat(),
                "access_count": 0,
                "hit_count": 0,
                "size_bytes": size_bytes
            })
            await self.redis_client.expire(f"cache_meta:{cache_key}", cache_ttl)
            
            # Add to tag sets
            if tags:
                for tag in tags:
                    await self.redis_client.sadd(f"cache_tag:{tag}", cache_key)
                    await self.redis_client.expire(f"cache_tag:{tag}", cache_ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"L2 cache set failed: {e}")
            return False
    
    async def delete_from_l2(self, cache_key: str) -> bool:
        """Delete value from L2 Redis cache"""
        try:
            # Get tags before deletion
            data = await self.redis_client.get(f"cache:{cache_key}")
            if data:
                entry_data = json.loads(data)
                tags = entry_data.get("tags", [])
                
                # Remove from tag sets
                for tag in tags:
                    await self.redis_client.srem(f"cache_tag:{tag}", cache_key)
            
            # Delete cache entry and metadata
            deleted_count = await self.redis_client.delete(
                f"cache:{cache_key}",
                f"cache_meta:{cache_key}"
            )
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"L2 cache delete failed: {e}")
            return False
    
    # Level 3 (Disk) Cache Operations
    
    async def get_from_l3(self, cache_key: str) -> Any:
        """Get value from L3 disk cache"""
        try:
            cache_file = self.get_l3_file_path(cache_key)
            
            if cache_file.exists():
                # Check if file is expired
                file_age = datetime.utcnow() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                
                # Default TTL for disk cache (24 hours)
                if file_age.total_seconds() > 86400:
                    cache_file.unlink(missing_ok=True)
                    return None
                
                # Read and deserialize data
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"L3 cache get failed: {e}")
            return None
    
    async def set_to_l3(
        self,
        cache_key: str,
        value: Any,
        user_id: Optional[str] = None,
        region: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value to L3 disk cache"""
        try:
            cache_file = self.get_l3_file_path(cache_key)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize data to disk
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            return True
            
        except Exception as e:
            logger.error(f"L3 cache set failed: {e}")
            return False
    
    async def delete_from_l3(self, cache_key: str) -> bool:
        """Delete value from L3 disk cache"""
        try:
            cache_file = self.get_l3_file_path(cache_key)
            
            if cache_file.exists():
                cache_file.unlink()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"L3 cache delete failed: {e}")
            return False
    
    def get_l3_file_path(self, cache_key: str) -> "Path":
        """Get file path for L3 disk cache"""
        from pathlib import Path
        
        # Create subdirectories based on key hash for better distribution
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        subdir = key_hash[:2]
        
        return Path(self.config.l3_base_path) / subdir / f"{key_hash}.cache"
    
    # Cache Management and Optimization
    
    async def ensure_l1_capacity(self, required_bytes: int) -> None:
        """Ensure L1 cache has capacity for new entry"""
        max_size_bytes = self.config.l1_max_size_mb * 1024 * 1024
        max_entries = self.config.l1_max_entries
        
        # Check size limit
        while (self.l1_size_bytes + required_bytes > max_size_bytes or 
               len(self.l1_cache) >= max_entries):
            
            if not self.l1_access_order:
                break
            
            # Evict least recently used entry
            lru_key = self.l1_access_order.pop(0)
            await self.delete_from_l1(lru_key)
            self.stats.evictions += 1
    
    async def background_eviction(self) -> None:
        """Background task for cache eviction"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Evict expired entries from L1
                current_time = datetime.utcnow()
                expired_keys = []
                
                for cache_key, entry in self.l1_cache.items():
                    if entry.ttl_seconds:
                        age = (current_time - entry.created_at).total_seconds()
                        if age > entry.ttl_seconds:
                            expired_keys.append(cache_key)
                
                for cache_key in expired_keys:
                    await self.delete_from_l1(cache_key)
                    self.stats.evictions += 1
                
                # Clean up disk cache
                await self.cleanup_l3_cache()
                
            except Exception as e:
                logger.error(f"Background eviction failed: {e}")
    
    async def cleanup_l3_cache(self) -> None:
        """Clean up expired entries from L3 disk cache"""
        try:
            from pathlib import Path
            
            cache_dir = Path(self.config.l3_base_path)
            if not cache_dir.exists():
                return
            
            current_time = datetime.utcnow()
            max_age = timedelta(days=7)  # Keep disk cache for 7 days
            
            for cache_file in cache_dir.rglob("*.cache"):
                file_age = current_time - datetime.fromtimestamp(cache_file.stat().st_mtime)
                
                if file_age > max_age:
                    cache_file.unlink(missing_ok=True)
                    
        except Exception as e:
            logger.error(f"L3 cache cleanup failed: {e}")
    
    async def background_prefetch(self) -> None:
        """Background task for predictive prefetching"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze access patterns and prefetch popular content
                await self.analyze_and_prefetch()
                
            except Exception as e:
                logger.error(f"Background prefetch failed: {e}")
    
    async def analyze_and_prefetch(self) -> None:
        """Analyze access patterns and prefetch content"""
        try:
            # Get popular cache keys from L2
            pattern = "cache_meta:*"
            keys = await self.redis_client.keys(pattern)
            
            popular_keys = []
            
            for key in keys[:100]:  # Limit analysis
                meta = await self.redis_client.hgetall(key)
                
                if meta:
                    access_count = int(meta.get(b"access_count", 0))
                    if access_count > 10:  # Popular content threshold
                        cache_key = key.decode().replace("cache_meta:", "")
                        popular_keys.append((cache_key, access_count))
            
            # Sort by popularity
            popular_keys.sort(key=lambda x: x[1], reverse=True)
            
            # Prefetch top items to L1 if not already there
            for cache_key, _ in popular_keys[:20]:
                if cache_key not in self.l1_cache:
                    value = await self.get_from_l2(cache_key)
                    if value is not None:
                        await self.set_to_l1(cache_key, value)
                        
        except Exception as e:
            logger.error(f"Prefetch analysis failed: {e}")
    
    # Utility Methods
    
    def serialize_data(self, value: Any) -> Tuple[Any, int, bool]:
        """Serialize and optionally compress data"""
        try:
            # Serialize to JSON if possible, otherwise use pickle
            try:
                serialized = json.dumps(value)
                data_bytes = serialized.encode('utf-8')
            except (TypeError, ValueError):
                data_bytes = pickle.dumps(value)
                serialized = data_bytes
            
            size_bytes = len(data_bytes)
            compressed = False
            
            # Compress if enabled and data is large enough
            if (self.config.compression_enabled and 
                size_bytes > self.config.compression_threshold):
                
                compressed_data = zlib.compress(data_bytes)
                if len(compressed_data) < size_bytes:
                    serialized = compressed_data
                    size_bytes = len(compressed_data)
                    compressed = True
            
            return serialized, size_bytes, compressed
            
        except Exception as e:
            logger.error(f"Data serialization failed: {e}")
            return value, 0, False
    
    def deserialize_data(self, data: Any, compressed: bool) -> Any:
        """Deserialize and decompress data"""
        try:
            if compressed:
                data = zlib.decompress(data)
            
            # Try JSON first, then pickle
            try:
                if isinstance(data, bytes):
                    return json.loads(data.decode('utf-8'))
                else:
                    return json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if isinstance(data, bytes):
                    return pickle.loads(data)
                else:
                    return pickle.loads(data.encode('utf-8'))
                    
        except Exception as e:
            logger.error(f"Data deserialization failed: {e}")
            return None
    
    def determine_cache_scope(
        self,
        user_id: Optional[str],
        region: Optional[str]
    ) -> CacheScope:
        """Determine appropriate cache scope"""
        if user_id:
            return CacheScope.USER_SPECIFIC
        elif region:
            return CacheScope.REGION_SPECIFIC
        else:
            return CacheScope.GLOBAL
    
    async def track_access(
        self,
        cache_key: str,
        cache_level: CacheLevel,
        start_time: datetime
    ) -> None:
        """Track cache access for analytics"""
        try:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update statistics
            self.stats.avg_response_time_ms = (
                (self.stats.avg_response_time_ms * (self.stats.total_requests - 1) + response_time) /
                max(self.stats.total_requests, 1)
            )
            
        except Exception as e:
            logger.error(f"Access tracking failed: {e}")
    
    async def update_stats_periodically(self) -> None:
        """Update cache statistics periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                
                # Calculate hit ratio
                total_requests = self.stats.cache_hits + self.stats.cache_misses
                if total_requests > 0:
                    self.stats.hit_ratio = self.stats.cache_hits / total_requests
                
                # Update memory usage
                self.stats.memory_usage_mb = self.l1_size_bytes / (1024 * 1024)
                max_memory_mb = self.config.l1_max_size_mb
                self.stats.memory_utilization = self.stats.memory_usage_mb / max_memory_mb
                
                # Store stats in Redis
                stats_key = f"cache_stats:{int(datetime.utcnow().timestamp())}"
                await self.redis_client.setex(
                    stats_key,
                    86400 * 7,  # Keep for 7 days
                    self.stats.model_dump_json()
                )
                
                # Reset hourly stats
                self.stats = CacheStats(
                    period_start=datetime.utcnow(),
                    period_end=datetime.utcnow() + timedelta(hours=1)
                )
                
            except Exception as e:
                logger.error(f"Stats update failed: {e}")
    
    async def get_cache_statistics(self) -> CacheStats:
        """Get current cache statistics"""
        return self.stats
    
    async def warm_cache(self, keys_and_values: List[Tuple[str, Any]]) -> int:
        """Warm cache with predefined data"""
        try:
            warmed_count = 0
            
            for key, value in keys_and_values:
                success = await self.set(key, value)
                if success:
                    warmed_count += 1
            
            logger.info(f"Cache warmed with {warmed_count} entries")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            return 0
    
    async def close(self) -> None:
        """Clean up resources"""
        try:
            # Cancel background tasks
            if self._eviction_task:
                self._eviction_task.cancel()
            if self._stats_task:
                self._stats_task.cancel()
            if self._prefetch_task:
                self._prefetch_task.cancel()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
                
        except Exception as e:
            logger.error(f"Cache manager cleanup failed: {e}")
    
    # High-level cache operations for specific use cases
    
    async def cache_property_data(
        self,
        property_id: str,
        property_data: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """Cache property data with specific optimizations"""
        return await self.set(
            f"property:{property_id}",
            property_data,
            ttl=ttl,
            tags=["property", f"property:{property_id}"],
            cache_levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        )
    
    async def cache_user_recommendations(
        self,
        user_id: str,
        recommendations: List[Dict[str, Any]],
        ttl: int = 1800  # 30 minutes
    ) -> bool:
        """Cache user-specific recommendations"""
        return await self.set(
            f"recommendations",
            recommendations,
            ttl=ttl,
            user_id=user_id,
            tags=["recommendations", f"user:{user_id}"]
        )
    
    async def cache_search_results(
        self,
        search_query_hash: str,
        results: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        ttl: int = 900  # 15 minutes
    ) -> bool:
        """Cache search results"""
        return await self.set(
            f"search:{search_query_hash}",
            results,
            ttl=ttl,
            user_id=user_id,
            tags=["search", f"search:{search_query_hash}"]
        )