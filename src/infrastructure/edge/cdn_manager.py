"""
CDN (Content Delivery Network) Manager for global content delivery.

This module manages CDN integration for property images, static assets,
and API response caching with edge locations worldwide.
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urljoin, urlparse
from enum import Enum

import aiohttp
import aiofiles
from PIL import Image
from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CDNProvider(str, Enum):
    """Supported CDN providers"""
    CLOUDFLARE = "cloudflare"
    AWS_CLOUDFRONT = "aws_cloudfront"
    AZURE_CDN = "azure_cdn"
    GOOGLE_CLOUD_CDN = "google_cloud_cdn"
    FASTLY = "fastly"


class CachePolicy(str, Enum):
    """CDN cache policies"""
    NO_CACHE = "no-cache"
    SHORT_TERM = "short-term"      # 5 minutes
    MEDIUM_TERM = "medium-term"    # 1 hour
    LONG_TERM = "long-term"        # 24 hours
    IMMUTABLE = "immutable"        # 1 year


class ContentType(str, Enum):
    """Content types for CDN"""
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    API_RESPONSE = "api_response"
    STATIC_ASSET = "static_asset"
    PROPERTY_DATA = "property_data"


class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    AVIF = "avif"


class CDNConfig(BaseModel):
    """CDN configuration"""
    provider: CDNProvider = Field(..., description="CDN provider")
    base_url: str = Field(..., description="CDN base URL")
    api_key: Optional[str] = Field(None, description="API key for CDN management")
    zone_id: Optional[str] = Field(None, description="CDN zone/distribution ID")
    
    # Image optimization settings
    enable_image_optimization: bool = Field(default=True)
    supported_formats: List[ImageFormat] = Field(default=[ImageFormat.WEBP, ImageFormat.JPEG])
    max_image_width: int = Field(default=2048)
    max_image_height: int = Field(default=2048)
    quality_levels: Dict[str, int] = Field(default={
        "low": 60,
        "medium": 80,
        "high": 95
    })
    
    # Cache settings
    default_cache_policy: CachePolicy = Field(default=CachePolicy.MEDIUM_TERM)
    cache_policies: Dict[ContentType, CachePolicy] = Field(default={
        ContentType.IMAGE: CachePolicy.LONG_TERM,
        ContentType.STATIC_ASSET: CachePolicy.IMMUTABLE,
        ContentType.API_RESPONSE: CachePolicy.SHORT_TERM,
        ContentType.PROPERTY_DATA: CachePolicy.MEDIUM_TERM
    })
    
    # Edge locations
    edge_locations: List[str] = Field(default=[], description="Preferred edge locations")


class CDNAsset(BaseModel):
    """CDN asset information"""
    asset_id: str = Field(..., description="Unique asset identifier")
    original_url: str = Field(..., description="Original asset URL")
    cdn_url: str = Field(..., description="CDN URL")
    content_type: ContentType = Field(..., description="Content type")
    mime_type: str = Field(..., description="MIME type")
    
    # Metadata
    file_size: int = Field(..., description="File size in bytes")
    checksum: str = Field(..., description="Asset checksum")
    cache_policy: CachePolicy = Field(..., description="Cache policy")
    
    # Image-specific metadata
    width: Optional[int] = Field(None, description="Image width")
    height: Optional[int] = Field(None, description="Image height")
    format: Optional[ImageFormat] = Field(None, description="Image format")
    optimized_variants: Dict[str, str] = Field(default={}, description="Optimized image variants")
    
    # Status and metrics
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = Field(None)
    access_count: int = Field(default=0)
    cache_hit_ratio: float = Field(default=0.0)


class CDNPurgeRequest(BaseModel):
    """CDN cache purge request"""
    urls: List[str] = Field(..., description="URLs to purge")
    tags: List[str] = Field(default=[], description="Cache tags to purge")
    purge_everything: bool = Field(default=False, description="Purge entire cache")


class CDNAnalytics(BaseModel):
    """CDN analytics data"""
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")
    
    # Traffic metrics
    total_requests: int = Field(default=0)
    cached_requests: int = Field(default=0)
    origin_requests: int = Field(default=0)
    cache_hit_ratio: float = Field(default=0.0)
    
    # Bandwidth metrics
    total_bandwidth_gb: float = Field(default=0.0)
    cached_bandwidth_gb: float = Field(default=0.0)
    origin_bandwidth_gb: float = Field(default=0.0)
    
    # Geographic distribution
    requests_by_region: Dict[str, int] = Field(default={})
    top_assets: List[Dict[str, Any]] = Field(default=[])
    
    # Performance metrics
    avg_response_time_ms: float = Field(default=0.0)
    p95_response_time_ms: float = Field(default=0.0)
    error_rate: float = Field(default=0.0)


class CDNManager:
    """CDN manager for global content delivery"""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.assets_registry: Dict[str, CDNAsset] = {}
        
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the CDN manager"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Load existing assets from registry
            await self.load_assets_registry()
            
            # Start background tasks
            asyncio.create_task(self.update_cache_analytics())
            asyncio.create_task(self.cleanup_unused_assets())
            
            logger.info(f"CDN manager initialized with {self.config.provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CDN manager: {e}")
            raise
    
    async def upload_asset(
        self,
        file_path: str,
        content_type: ContentType,
        cache_policy: Optional[CachePolicy] = None,
        optimize: bool = True
    ) -> CDNAsset:
        """Upload asset to CDN"""
        try:
            cache_policy = cache_policy or self.config.cache_policies.get(
                content_type, 
                self.config.default_cache_policy
            )
            
            # Generate asset ID
            with open(file_path, 'rb') as f:
                content = f.read()
                checksum = hashlib.sha256(content).hexdigest()
                asset_id = f"{content_type.value}_{checksum[:16]}"
            
            # Check if asset already exists
            existing_asset = await self.get_asset(asset_id)
            if existing_asset:
                logger.info(f"Asset already exists: {asset_id}")
                return existing_asset
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # Optimize asset if applicable
            optimized_variants = {}
            width, height, format_type = None, None, None
            
            if content_type == ContentType.IMAGE and optimize:
                optimized_variants, width, height, format_type = await self.optimize_image(
                    file_path, asset_id
                )
            
            # Upload to CDN
            cdn_url = await self.upload_to_cdn(file_path, asset_id, mime_type, cache_policy)
            
            # Create asset record
            asset = CDNAsset(
                asset_id=asset_id,
                original_url=file_path,
                cdn_url=cdn_url,
                content_type=content_type,
                mime_type=mime_type,
                file_size=len(content),
                checksum=checksum,
                cache_policy=cache_policy,
                width=width,
                height=height,
                format=format_type,
                optimized_variants=optimized_variants
            )
            
            # Store asset in registry
            await self.store_asset(asset)
            self.assets_registry[asset_id] = asset
            
            logger.info(f"Asset uploaded successfully: {asset_id}")
            return asset
            
        except Exception as e:
            logger.error(f"Asset upload failed: {e}")
            raise
    
    async def optimize_image(
        self,
        image_path: str,
        asset_id: str
    ) -> Tuple[Dict[str, str], int, int, ImageFormat]:
        """Optimize image and create variants"""
        try:
            optimized_variants = {}
            
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                format_type = ImageFormat.WEBP  # Default to WebP for optimization
                
                # Create different quality variants
                for quality_name, quality_value in self.config.quality_levels.items():
                    # Resize if necessary
                    max_width = self.config.max_image_width
                    max_height = self.config.max_image_height
                    
                    if original_width > max_width or original_height > max_height:
                        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    
                    # Create optimized variant
                    variant_path = f"/tmp/{asset_id}_{quality_name}.webp"
                    
                    # Convert to RGB if necessary (for WebP)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    img.save(variant_path, 'WEBP', quality=quality_value, optimize=True)
                    
                    # Upload variant to CDN
                    variant_cdn_url = await self.upload_to_cdn(
                        variant_path,
                        f"{asset_id}_{quality_name}",
                        "image/webp",
                        CachePolicy.LONG_TERM
                    )
                    
                    optimized_variants[quality_name] = variant_cdn_url
                    
                    # Clean up temporary file
                    Path(variant_path).unlink(missing_ok=True)
                
                # Create responsive variants (different sizes)
                responsive_sizes = [
                    ("thumbnail", 150, 150),
                    ("small", 300, 300),
                    ("medium", 600, 600),
                    ("large", 1200, 1200)
                ]
                
                for size_name, max_w, max_h in responsive_sizes:
                    if original_width <= max_w and original_height <= max_h:
                        continue
                    
                    # Create resized variant
                    resized_img = img.copy()
                    resized_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                    
                    variant_path = f"/tmp/{asset_id}_{size_name}.webp"
                    resized_img.save(
                        variant_path, 
                        'WEBP', 
                        quality=self.config.quality_levels["medium"], 
                        optimize=True
                    )
                    
                    # Upload variant
                    variant_cdn_url = await self.upload_to_cdn(
                        variant_path,
                        f"{asset_id}_{size_name}",
                        "image/webp",
                        CachePolicy.LONG_TERM
                    )
                    
                    optimized_variants[size_name] = variant_cdn_url
                    Path(variant_path).unlink(missing_ok=True)
                
                return optimized_variants, original_width, original_height, format_type
                
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            return {}, 0, 0, ImageFormat.JPEG
    
    async def upload_to_cdn(
        self,
        file_path: str,
        asset_id: str,
        mime_type: str,
        cache_policy: CachePolicy
    ) -> str:
        """Upload file to CDN provider"""
        try:
            if self.config.provider == CDNProvider.CLOUDFLARE:
                return await self.upload_to_cloudflare(file_path, asset_id, mime_type, cache_policy)
            elif self.config.provider == CDNProvider.AWS_CLOUDFRONT:
                return await self.upload_to_cloudfront(file_path, asset_id, mime_type, cache_policy)
            else:
                # Generic upload simulation
                cdn_url = f"{self.config.base_url}/{asset_id}"
                logger.info(f"Simulated upload to {self.config.provider}: {cdn_url}")
                return cdn_url
                
        except Exception as e:
            logger.error(f"CDN upload failed: {e}")
            raise
    
    async def upload_to_cloudflare(
        self,
        file_path: str,
        asset_id: str,
        mime_type: str,
        cache_policy: CachePolicy
    ) -> str:
        """Upload to Cloudflare CDN"""
        try:
            # Cloudflare R2 or Images API integration
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": mime_type
            }
            
            # Set cache headers based on policy
            cache_headers = self.get_cache_headers(cache_policy)
            headers.update(cache_headers)
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload to Cloudflare
            upload_url = f"https://api.cloudflare.com/client/v4/accounts/{self.config.zone_id}/storage/kv/namespaces/assets/values/{asset_id}"
            
            async with self.session.put(
                upload_url,
                headers=headers,
                data=file_data
            ) as response:
                
                if response.status == 200:
                    cdn_url = f"{self.config.base_url}/{asset_id}"
                    logger.info(f"Uploaded to Cloudflare: {cdn_url}")
                    return cdn_url
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Cloudflare upload failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Cloudflare upload failed: {e}")
            raise
    
    async def upload_to_cloudfront(
        self,
        file_path: str,
        asset_id: str,
        mime_type: str,
        cache_policy: CachePolicy
    ) -> str:
        """Upload to AWS CloudFront (via S3)"""
        try:
            # This would use boto3 in a real implementation
            # For now, simulate the upload
            
            cdn_url = f"{self.config.base_url}/{asset_id}"
            logger.info(f"Simulated CloudFront upload: {cdn_url}")
            return cdn_url
            
        except Exception as e:
            logger.error(f"CloudFront upload failed: {e}")
            raise
    
    def get_cache_headers(self, cache_policy: CachePolicy) -> Dict[str, str]:
        """Get cache headers for CDN policy"""
        cache_headers = {}
        
        if cache_policy == CachePolicy.NO_CACHE:
            cache_headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        elif cache_policy == CachePolicy.SHORT_TERM:
            cache_headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
        elif cache_policy == CachePolicy.MEDIUM_TERM:
            cache_headers["Cache-Control"] = "public, max-age=3600"  # 1 hour
        elif cache_policy == CachePolicy.LONG_TERM:
            cache_headers["Cache-Control"] = "public, max-age=86400"  # 24 hours
        elif cache_policy == CachePolicy.IMMUTABLE:
            cache_headers["Cache-Control"] = "public, max-age=31536000, immutable"  # 1 year
        
        return cache_headers
    
    async def get_asset(self, asset_id: str) -> Optional[CDNAsset]:
        """Get asset information"""
        try:
            # Check local registry first
            if asset_id in self.assets_registry:
                return self.assets_registry[asset_id]
            
            # Check Redis
            asset_key = f"cdn_asset:{asset_id}"
            asset_data = await self.redis_client.get(asset_key)
            
            if asset_data:
                asset = CDNAsset.model_validate_json(asset_data)
                self.assets_registry[asset_id] = asset
                return asset
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get asset: {e}")
            return None
    
    async def store_asset(self, asset: CDNAsset) -> None:
        """Store asset in registry"""
        try:
            asset_key = f"cdn_asset:{asset.asset_id}"
            await self.redis_client.setex(
                asset_key,
                86400 * 365,  # 1 year
                asset.model_dump_json()
            )
            
            # Add to assets index
            await self.redis_client.sadd("cdn_assets", asset.asset_id)
            
        except Exception as e:
            logger.error(f"Failed to store asset: {e}")
    
    async def delete_asset(self, asset_id: str) -> bool:
        """Delete asset from CDN"""
        try:
            asset = await self.get_asset(asset_id)
            if not asset:
                return False
            
            # Delete from CDN provider
            success = await self.delete_from_cdn(asset_id)
            
            if success:
                # Remove from registry
                asset_key = f"cdn_asset:{asset_id}"
                await self.redis_client.delete(asset_key)
                await self.redis_client.srem("cdn_assets", asset_id)
                
                if asset_id in self.assets_registry:
                    del self.assets_registry[asset_id]
                
                # Delete optimized variants
                for variant_url in asset.optimized_variants.values():
                    variant_id = variant_url.split('/')[-1]
                    await self.delete_from_cdn(variant_id)
                
                logger.info(f"Asset deleted: {asset_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Asset deletion failed: {e}")
            return False
    
    async def delete_from_cdn(self, asset_id: str) -> bool:
        """Delete asset from CDN provider"""
        try:
            if self.config.provider == CDNProvider.CLOUDFLARE:
                return await self.delete_from_cloudflare(asset_id)
            elif self.config.provider == CDNProvider.AWS_CLOUDFRONT:
                return await self.delete_from_cloudfront(asset_id)
            else:
                # Simulate deletion
                logger.info(f"Simulated deletion from {self.config.provider}: {asset_id}")
                return True
                
        except Exception as e:
            logger.error(f"CDN deletion failed: {e}")
            return False
    
    async def delete_from_cloudflare(self, asset_id: str) -> bool:
        """Delete from Cloudflare CDN"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}"
            }
            
            delete_url = f"https://api.cloudflare.com/client/v4/accounts/{self.config.zone_id}/storage/kv/namespaces/assets/values/{asset_id}"
            
            async with self.session.delete(delete_url, headers=headers) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Cloudflare deletion failed: {e}")
            return False
    
    async def delete_from_cloudfront(self, asset_id: str) -> bool:
        """Delete from AWS CloudFront"""
        try:
            # Would use boto3 in real implementation
            logger.info(f"Simulated CloudFront deletion: {asset_id}")
            return True
            
        except Exception as e:
            logger.error(f"CloudFront deletion failed: {e}")
            return False
    
    async def purge_cache(self, purge_request: CDNPurgeRequest) -> bool:
        """Purge CDN cache"""
        try:
            if self.config.provider == CDNProvider.CLOUDFLARE:
                return await self.purge_cloudflare_cache(purge_request)
            elif self.config.provider == CDNProvider.AWS_CLOUDFRONT:
                return await self.purge_cloudfront_cache(purge_request)
            else:
                # Simulate cache purge
                logger.info(f"Simulated cache purge for {len(purge_request.urls)} URLs")
                return True
                
        except Exception as e:
            logger.error(f"Cache purge failed: {e}")
            return False
    
    async def purge_cloudflare_cache(self, purge_request: CDNPurgeRequest) -> bool:
        """Purge Cloudflare cache"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            purge_data = {}
            
            if purge_request.purge_everything:
                purge_data["purge_everything"] = True
            else:
                if purge_request.urls:
                    purge_data["files"] = purge_request.urls
                if purge_request.tags:
                    purge_data["tags"] = purge_request.tags
            
            purge_url = f"https://api.cloudflare.com/client/v4/zones/{self.config.zone_id}/purge_cache"
            
            async with self.session.post(
                purge_url,
                headers=headers,
                json=purge_data
            ) as response:
                
                if response.status == 200:
                    logger.info("Cloudflare cache purged successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Cloudflare cache purge failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Cloudflare cache purge failed: {e}")
            return False
    
    async def purge_cloudfront_cache(self, purge_request: CDNPurgeRequest) -> bool:
        """Purge CloudFront cache"""
        try:
            # Would use boto3 invalidation API in real implementation
            logger.info(f"Simulated CloudFront cache purge for {len(purge_request.urls)} URLs")
            return True
            
        except Exception as e:
            logger.error(f"CloudFront cache purge failed: {e}")
            return False
    
    async def get_optimized_url(
        self,
        asset_id: str,
        quality: str = "medium",
        size: Optional[str] = None,
        format_type: Optional[ImageFormat] = None
    ) -> Optional[str]:
        """Get optimized asset URL"""
        try:
            asset = await self.get_asset(asset_id)
            if not asset:
                return None
            
            # If no optimization requested, return original URL
            if not quality and not size and not format_type:
                return asset.cdn_url
            
            # Look for optimized variant
            variant_key = quality
            if size:
                variant_key = size
            
            if variant_key in asset.optimized_variants:
                return asset.optimized_variants[variant_key]
            
            # If specific variant not found, return original
            return asset.cdn_url
            
        except Exception as e:
            logger.error(f"Failed to get optimized URL: {e}")
            return None
    
    async def get_analytics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> CDNAnalytics:
        """Get CDN analytics data"""
        try:
            if self.config.provider == CDNProvider.CLOUDFLARE:
                return await self.get_cloudflare_analytics(start_date, end_date)
            elif self.config.provider == CDNProvider.AWS_CLOUDFRONT:
                return await self.get_cloudfront_analytics(start_date, end_date)
            else:
                # Return mock analytics
                return CDNAnalytics(
                    period_start=start_date,
                    period_end=end_date,
                    total_requests=10000,
                    cached_requests=8500,
                    origin_requests=1500,
                    cache_hit_ratio=0.85,
                    total_bandwidth_gb=150.5,
                    cached_bandwidth_gb=127.9,
                    origin_bandwidth_gb=22.6
                )
                
        except Exception as e:
            logger.error(f"Analytics retrieval failed: {e}")
            return CDNAnalytics(period_start=start_date, period_end=end_date)
    
    async def get_cloudflare_analytics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> CDNAnalytics:
        """Get Cloudflare analytics"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Format dates for Cloudflare API
            since = start_date.isoformat() + "Z"
            until = end_date.isoformat() + "Z"
            
            analytics_url = (
                f"https://api.cloudflare.com/client/v4/zones/{self.config.zone_id}/analytics/dashboard"
                f"?since={since}&until={until}"
            )
            
            async with self.session.get(analytics_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    
                    # Extract metrics from Cloudflare response
                    return CDNAnalytics(
                        period_start=start_date,
                        period_end=end_date,
                        total_requests=result.get("requests", {}).get("all", 0),
                        cached_requests=result.get("requests", {}).get("cached", 0),
                        origin_requests=result.get("requests", {}).get("uncached", 0),
                        cache_hit_ratio=result.get("requests", {}).get("ssl", {}).get("encrypted", 0) / 100,
                        total_bandwidth_gb=result.get("bandwidth", {}).get("all", 0) / (1024**3),
                        avg_response_time_ms=result.get("totals", {}).get("responseTimeAvg", 0)
                    )
                else:
                    logger.error(f"Cloudflare analytics API error: {response.status}")
                    return CDNAnalytics(period_start=start_date, period_end=end_date)
                    
        except Exception as e:
            logger.error(f"Cloudflare analytics failed: {e}")
            return CDNAnalytics(period_start=start_date, period_end=end_date)
    
    async def get_cloudfront_analytics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> CDNAnalytics:
        """Get CloudFront analytics"""
        try:
            # Would use boto3 CloudWatch API in real implementation
            return CDNAnalytics(
                period_start=start_date,
                period_end=end_date,
                total_requests=5000,
                cached_requests=4200,
                origin_requests=800,
                cache_hit_ratio=0.84,
                total_bandwidth_gb=89.3,
                cached_bandwidth_gb=75.1,
                origin_bandwidth_gb=14.2
            )
            
        except Exception as e:
            logger.error(f"CloudFront analytics failed: {e}")
            return CDNAnalytics(period_start=start_date, period_end=end_date)
    
    async def update_cache_analytics(self) -> None:
        """Background task to update cache analytics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                
                # Get analytics for the last hour
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)
                
                analytics = await self.get_analytics(start_time, end_time)
                
                # Store analytics in Redis
                analytics_key = f"cdn_analytics:{int(end_time.timestamp())}"
                await self.redis_client.setex(
                    analytics_key,
                    86400 * 7,  # Keep for 7 days
                    analytics.model_dump_json()
                )
                
                # Update asset access metrics
                await self.update_asset_metrics(analytics)
                
            except Exception as e:
                logger.error(f"Cache analytics update failed: {e}")
    
    async def update_asset_metrics(self, analytics: CDNAnalytics) -> None:
        """Update individual asset metrics"""
        try:
            # This would require detailed per-asset analytics from CDN provider
            # For now, we'll update overall cache hit ratio for all assets
            
            asset_ids = await self.redis_client.smembers("cdn_assets")
            
            for asset_id in asset_ids:
                asset = await self.get_asset(asset_id.decode())
                if asset:
                    # Update cache hit ratio (simplified)
                    alpha = 0.1  # Smoothing factor
                    asset.cache_hit_ratio = (
                        alpha * analytics.cache_hit_ratio +
                        (1 - alpha) * asset.cache_hit_ratio
                    )
                    
                    await self.store_asset(asset)
                    
        except Exception as e:
            logger.error(f"Asset metrics update failed: {e}")
    
    async def cleanup_unused_assets(self) -> None:
        """Background task to cleanup unused assets"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                
                asset_ids = await self.redis_client.smembers("cdn_assets")
                
                for asset_id in asset_ids:
                    asset = await self.get_asset(asset_id.decode())
                    
                    if asset and asset.last_accessed and asset.last_accessed < cutoff_time:
                        # Asset hasn't been accessed in 30 days
                        if asset.access_count < 5:  # And has low usage
                            logger.info(f"Cleaning up unused asset: {asset.asset_id}")
                            await self.delete_asset(asset.asset_id)
                            
            except Exception as e:
                logger.error(f"Asset cleanup failed: {e}")
    
    async def load_assets_registry(self) -> None:
        """Load assets registry from Redis"""
        try:
            asset_ids = await self.redis_client.smembers("cdn_assets")
            
            for asset_id in asset_ids:
                asset = await self.get_asset(asset_id.decode())
                if asset:
                    self.assets_registry[asset.asset_id] = asset
                    
            logger.info(f"Loaded {len(self.assets_registry)} assets from registry")
            
        except Exception as e:
            logger.error(f"Failed to load assets registry: {e}")
    
    async def close(self) -> None:
        """Clean up resources"""
        try:
            if self.session:
                await self.session.close()
            
            if self.redis_client:
                await self.redis_client.close()
                
        except Exception as e:
            logger.error(f"CDN manager cleanup failed: {e}")
    
    # Utility methods for property images
    
    async def upload_property_images(
        self,
        property_id: str,
        image_paths: List[str]
    ) -> Dict[str, CDNAsset]:
        """Upload multiple property images"""
        try:
            assets = {}
            
            upload_tasks = []
            for i, image_path in enumerate(image_paths):
                task = self.upload_asset(
                    image_path,
                    ContentType.IMAGE,
                    CachePolicy.LONG_TERM,
                    optimize=True
                )
                upload_tasks.append((f"image_{i}", task))
            
            # Upload all images concurrently
            results = await asyncio.gather(
                *[task for _, task in upload_tasks],
                return_exceptions=True
            )
            
            # Process results
            for i, result in enumerate(results):
                image_key = upload_tasks[i][0]
                
                if isinstance(result, CDNAsset):
                    assets[image_key] = result
                    
                    # Tag image with property ID
                    await self.tag_asset_with_property(result.asset_id, property_id)
                else:
                    logger.error(f"Failed to upload {image_paths[i]}: {result}")
            
            return assets
            
        except Exception as e:
            logger.error(f"Property images upload failed: {e}")
            return {}
    
    async def tag_asset_with_property(self, asset_id: str, property_id: str) -> None:
        """Tag asset with property ID for organization"""
        try:
            # Add asset to property's asset set
            property_assets_key = f"property_assets:{property_id}"
            await self.redis_client.sadd(property_assets_key, asset_id)
            await self.redis_client.expire(property_assets_key, 86400 * 365)
            
            # Add property reference to asset
            asset_property_key = f"asset_property:{asset_id}"
            await self.redis_client.setex(asset_property_key, 86400 * 365, property_id)
            
        except Exception as e:
            logger.error(f"Asset tagging failed: {e}")
    
    async def get_property_assets(self, property_id: str) -> List[CDNAsset]:
        """Get all assets for a property"""
        try:
            property_assets_key = f"property_assets:{property_id}"
            asset_ids = await self.redis_client.smembers(property_assets_key)
            
            assets = []
            for asset_id in asset_ids:
                asset = await self.get_asset(asset_id.decode())
                if asset:
                    assets.append(asset)
            
            return assets
            
        except Exception as e:
            logger.error(f"Failed to get property assets: {e}")
            return []