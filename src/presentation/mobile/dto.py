"""
Mobile-optimized Data Transfer Objects (DTOs) for the rental ML system.

This module provides mobile-specific DTOs with reduced payload sizes,
optimized for mobile network conditions and offline-first architecture.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

from .sync import SyncRecord, SyncConflict


class ImageQuality(str, Enum):
    """Image quality levels for mobile optimization"""
    LOW = "low"          # 480p, high compression
    MEDIUM = "medium"    # 720p, medium compression
    HIGH = "high"        # 1080p, low compression


class NetworkType(str, Enum):
    """Network connection types"""
    UNKNOWN = "unknown"
    WIFI = "wifi"
    ETHERNET = "ethernet"
    CELLULAR_2G = "2g"
    CELLULAR_3G = "3g"
    CELLULAR_4G = "4g"
    CELLULAR_5G = "5g"


class MobileImageDTO(BaseModel):
    """Mobile-optimized image information"""
    url: str = Field(..., description="Image URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    width: Optional[int] = Field(None, description="Image width")
    height: Optional[int] = Field(None, description="Image height")
    size_bytes: Optional[int] = Field(None, description="Image size in bytes")
    format: Optional[str] = Field(None, description="Image format (jpg, webp, etc.)")
    
    @classmethod
    def from_url(
        cls,
        url: str,
        quality: ImageQuality = ImageQuality.MEDIUM,
        network_type: NetworkType = NetworkType.UNKNOWN
    ) -> "MobileImageDTO":
        """Create mobile image DTO from URL with quality optimization"""
        
        # Optimize based on network conditions
        if network_type in [NetworkType.CELLULAR_2G, NetworkType.CELLULAR_3G]:
            quality = ImageQuality.LOW
        elif network_type == NetworkType.CELLULAR_4G and quality == ImageQuality.HIGH:
            quality = ImageQuality.MEDIUM
        
        # Generate optimized URLs (would integrate with CDN/image service)
        quality_suffix = f"?quality={quality.value}&format=webp&auto=compress"
        optimized_url = f"{url}{quality_suffix}"
        
        # Generate thumbnail
        thumbnail_url = f"{url}?quality=low&width=150&height=150&format=webp"
        
        return cls(
            url=optimized_url,
            thumbnail_url=thumbnail_url,
            format="webp"
        )


class MobileLocationDTO(BaseModel):
    """Mobile-optimized location information"""
    latitude: Optional[float] = Field(None, description="Latitude")
    longitude: Optional[float] = Field(None, description="Longitude")
    address: Optional[str] = Field(None, description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State/Province")
    zip_code: Optional[str] = Field(None, description="ZIP/Postal code")
    country: str = Field(default="US", description="Country code")
    
    # Simplified for mobile
    display_address: str = Field(..., description="Formatted display address")
    distance_km: Optional[float] = Field(None, description="Distance from user in km")


class MobilePropertyDTO(BaseModel):
    """Mobile-optimized property information"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Property ID")
    title: str = Field(..., description="Property title")
    price: float = Field(..., description="Monthly rent price")
    bedrooms: int = Field(..., description="Number of bedrooms")
    bathrooms: float = Field(..., description="Number of bathrooms")
    area_sqft: Optional[int] = Field(None, description="Area in square feet")
    
    # Mobile-optimized location
    location: MobileLocationDTO = Field(..., description="Property location")
    
    # Mobile-optimized images
    images: List[MobileImageDTO] = Field(default=[], description="Property images")
    primary_image: Optional[MobileImageDTO] = Field(None, description="Primary image")
    
    # Essential amenities only (reduced payload)
    key_amenities: List[str] = Field(default=[], description="Key amenities")
    
    # Mobile-specific fields
    is_favorited: bool = Field(default=False, description="User has favorited this property")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    availability_status: str = Field(default="available", description="Availability status")
    
    # Optional fields for detailed view
    description: Optional[str] = Field(None, description="Property description")
    contact_info: Optional[Dict[str, Any]] = Field(None, description="Contact information")
    
    @classmethod
    def from_property(
        cls,
        property_obj: Any,
        image_quality: ImageQuality = ImageQuality.MEDIUM,
        network_type: NetworkType = NetworkType.UNKNOWN,
        include_details: bool = False,
        user_favorites: Optional[List[str]] = None
    ) -> "MobilePropertyDTO":
        """Create mobile DTO from property object"""
        
        # Convert location
        location = MobileLocationDTO(
            latitude=getattr(property_obj, 'latitude', None),
            longitude=getattr(property_obj, 'longitude', None),
            address=getattr(property_obj, 'address', None),
            city=getattr(property_obj, 'city', ''),
            state=getattr(property_obj, 'state', ''),
            zip_code=getattr(property_obj, 'zip_code', None),
            display_address=f"{getattr(property_obj, 'address', '')}, {getattr(property_obj, 'city', '')}, {getattr(property_obj, 'state', '')}"
        )
        
        # Convert images
        images = []
        primary_image = None
        
        property_images = getattr(property_obj, 'images', [])
        for i, img_url in enumerate(property_images[:5]):  # Limit to 5 images for mobile
            mobile_img = MobileImageDTO.from_url(img_url, image_quality, network_type)
            images.append(mobile_img)
            
            if i == 0:  # First image as primary
                primary_image = mobile_img
        
        # Extract key amenities (limit to most important ones)
        all_amenities = getattr(property_obj, 'amenities', [])
        key_amenities = [a for a in all_amenities if a.lower() in [
            'parking', 'laundry', 'dishwasher', 'air conditioning', 'heating',
            'pet friendly', 'gym', 'pool', 'balcony', 'wifi'
        ]][:5]  # Limit to 5 key amenities
        
        # Check if favorited
        is_favorited = False
        if user_favorites and hasattr(property_obj, 'id'):
            is_favorited = str(property_obj.id) in user_favorites
        
        return cls(
            id=str(property_obj.id),
            title=getattr(property_obj, 'title', ''),
            price=float(getattr(property_obj, 'price', 0)),
            bedrooms=int(getattr(property_obj, 'bedrooms', 0)),
            bathrooms=float(getattr(property_obj, 'bathrooms', 0)),
            area_sqft=getattr(property_obj, 'area_sqft', None),
            location=location,
            images=images,
            primary_image=primary_image,
            key_amenities=key_amenities,
            is_favorited=is_favorited,
            last_updated=getattr(property_obj, 'updated_at', datetime.utcnow()),
            description=getattr(property_obj, 'description', None) if include_details else None,
            contact_info=getattr(property_obj, 'contact_info', None) if include_details else None
        )


class MobileSearchResultDTO(BaseModel):
    """Mobile-optimized search results"""
    properties: List[MobilePropertyDTO] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    page: int = Field(..., description="Current page")
    limit: int = Field(..., description="Results per page")
    has_more: bool = Field(..., description="More results available")
    search_time: float = Field(..., description="Search time in seconds")
    
    # Mobile-specific optimizations
    network_optimized: bool = Field(default=False, description="Results optimized for network")
    cache_hit: bool = Field(default=False, description="Results served from cache")
    
    # Search metadata
    filters_applied: Dict[str, Any] = Field(default={}, description="Applied search filters")
    suggested_filters: List[Dict[str, Any]] = Field(default=[], description="Suggested filters")


class MobileRecommendationDTO(BaseModel):
    """Mobile-optimized recommendation"""
    property: MobilePropertyDTO = Field(..., description="Recommended property")
    score: float = Field(..., description="Recommendation score")
    reasons: List[str] = Field(default=[], description="Recommendation reasons")
    
    # Mobile-specific fields
    recommendation_type: str = Field(..., description="Type of recommendation")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_recommendation(
        cls,
        recommendation_obj: Any,
        include_images: bool = True,
        network_type: NetworkType = NetworkType.UNKNOWN
    ) -> "MobileRecommendationDTO":
        """Create mobile recommendation DTO"""
        
        # Convert property
        property_dto = MobilePropertyDTO.from_property(
            recommendation_obj.property,
            image_quality=ImageQuality.MEDIUM if include_images else ImageQuality.LOW,
            network_type=network_type
        )
        
        # Extract recommendation reasons (limit for mobile)
        reasons = getattr(recommendation_obj, 'reasons', [])[:3]
        
        return cls(
            property=property_dto,
            score=float(getattr(recommendation_obj, 'score', 0)),
            reasons=reasons,
            recommendation_type=getattr(recommendation_obj, 'type', 'general')
        )


class MobileUserProfileDTO(BaseModel):
    """Mobile-optimized user profile"""
    user_id: str = Field(..., description="User ID")
    display_name: Optional[str] = Field(None, description="Display name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    
    # Preferences (essential only)
    preferred_locations: List[str] = Field(default=[], description="Preferred locations")
    price_range: Optional[Dict[str, float]] = Field(None, description="Price range preferences")
    bedroom_preference: Optional[int] = Field(None, description="Preferred bedrooms")
    
    # Mobile-specific settings
    notification_preferences: Dict[str, bool] = Field(default={}, description="Notification settings")
    offline_sync_enabled: bool = Field(default=True, description="Offline sync enabled")
    
    # Stats
    favorites_count: int = Field(default=0, description="Number of favorites")
    searches_count: int = Field(default=0, description="Number of searches")
    last_active: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_user(cls, user_obj: Any) -> "MobileUserProfileDTO":
        """Create mobile user profile DTO"""
        return cls(
            user_id=str(user_obj.id),
            display_name=getattr(user_obj, 'display_name', None),
            email=getattr(user_obj, 'email', None),
            phone=getattr(user_obj, 'phone', None),
            preferred_locations=getattr(user_obj, 'preferred_locations', []),
            price_range=getattr(user_obj, 'price_range', None),
            bedroom_preference=getattr(user_obj, 'bedroom_preference', None),
            notification_preferences=getattr(user_obj, 'notification_preferences', {}),
            favorites_count=getattr(user_obj, 'favorites_count', 0),
            searches_count=getattr(user_obj, 'searches_count', 0),
            last_active=getattr(user_obj, 'last_active', datetime.utcnow())
        )


class SyncRequestDTO(BaseModel):
    """Sync request from mobile client"""
    device_id: str = Field(..., description="Device identifier")
    last_sync_timestamp: Optional[datetime] = Field(None, description="Last successful sync")
    client_records: List[SyncRecord] = Field(default=[], description="Client-side changes")
    requested_entities: List[str] = Field(default=[], description="Entities to sync")
    batch_size: int = Field(default=50, description="Maximum records per batch")
    include_deleted: bool = Field(default=False, description="Include deleted records")
    
    # Mobile-specific sync options
    wifi_only: bool = Field(default=False, description="Sync only on WiFi")
    compress_data: bool = Field(default=True, description="Compress sync data")
    priority_entities: List[str] = Field(default=[], description="High priority entities")


class SyncResponseDTO(BaseModel):
    """Sync response to mobile client"""
    sync_id: str = Field(..., description="Sync session ID")
    server_records: List[SyncRecord] = Field(default=[], description="Server-side changes")
    conflicts: List[SyncConflict] = Field(default=[], description="Sync conflicts")
    last_sync_timestamp: datetime = Field(default_factory=datetime.utcnow)
    has_more: bool = Field(default=False, description="More records available")
    next_batch_token: Optional[str] = Field(None, description="Token for next batch")
    has_conflicts: bool = Field(default=False, description="Has unresolved conflicts")
    sync_status: str = Field(default="completed", description="Sync status")
    
    # Mobile-specific response data
    data_size_bytes: int = Field(default=0, description="Response size in bytes")
    compression_ratio: Optional[float] = Field(None, description="Compression ratio achieved")
    estimated_battery_impact: str = Field(default="low", description="Estimated battery impact")


class MobileSearchFilterDTO(BaseModel):
    """Mobile-optimized search filters"""
    location: Optional[str] = Field(None, description="Location filter")
    price_min: Optional[float] = Field(None, description="Minimum price")
    price_max: Optional[float] = Field(None, description="Maximum price")
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    amenities: List[str] = Field(default=[], description="Required amenities")
    property_type: Optional[str] = Field(None, description="Property type")
    
    # Mobile-specific filters
    distance_km: Optional[float] = Field(None, description="Maximum distance in km")
    sort_by: str = Field(default="relevance", description="Sort order")
    include_images: bool = Field(default=True, description="Include images in results")


class MobileFavoriteDTO(BaseModel):
    """Mobile-optimized favorite property"""
    property_id: str = Field(..., description="Property ID")
    property_title: str = Field(..., description="Property title")
    price: float = Field(..., description="Current price")
    location: str = Field(..., description="Property location")
    primary_image: Optional[MobileImageDTO] = Field(None, description="Primary image")
    added_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Mobile-specific fields
    price_change: Optional[float] = Field(None, description="Recent price change")
    availability_changed: bool = Field(default=False, description="Availability changed")
    last_checked: datetime = Field(default_factory=datetime.utcnow)


class MobileSearchHistoryDTO(BaseModel):
    """Mobile-optimized search history"""
    search_id: str = Field(..., description="Search ID")
    query: str = Field(..., description="Search query")
    filters: MobileSearchFilterDTO = Field(..., description="Search filters used")
    results_count: int = Field(..., description="Number of results")
    searched_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Mobile-specific fields
    location_used: Optional[str] = Field(None, description="Location when searched")
    saved: bool = Field(default=False, description="Search saved by user")


class MobileNotificationDTO(BaseModel):
    """Mobile-optimized notification"""
    notification_id: str = Field(..., description="Notification ID")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    type: str = Field(..., description="Notification type")
    
    # Mobile-specific fields
    action_url: Optional[str] = Field(None, description="Deep link URL")
    image_url: Optional[str] = Field(None, description="Notification image")
    read: bool = Field(default=False, description="Notification read status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Notification expiry")


class MobileAnalyticsEventDTO(BaseModel):
    """Mobile analytics event"""
    event_type: str = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(default={}, description="Event data")
    user_id: Optional[str] = Field(None, description="User ID")
    device_id: str = Field(..., description="Device ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Mobile-specific fields
    app_version: str = Field(..., description="App version")
    os_version: str = Field(..., description="OS version")
    network_type: NetworkType = Field(default=NetworkType.UNKNOWN)
    battery_level: Optional[int] = Field(None, description="Battery level percentage")
    location: Optional[Dict[str, float]] = Field(None, description="User location")


class MobileErrorDTO(BaseModel):
    """Mobile-optimized error response"""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Mobile-specific fields
    retry_after: Optional[int] = Field(None, description="Retry after seconds")
    offline_fallback: bool = Field(default=False, description="Offline fallback available")
    support_reference: Optional[str] = Field(None, description="Support reference ID")


class MobileBatchResponseDTO(BaseModel):
    """Mobile batch operation response"""
    batch_id: str = Field(..., description="Batch operation ID")
    total_items: int = Field(..., description="Total items in batch")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    errors: List[MobileErrorDTO] = Field(default=[], description="Batch errors")
    
    # Mobile-specific fields
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    data_transferred_bytes: int = Field(..., description="Data transferred in bytes")
    battery_impact: str = Field(default="low", description="Estimated battery impact")


# Response wrapper for mobile APIs
class MobileAPIResponse(BaseModel):
    """Standard mobile API response wrapper"""
    success: bool = Field(..., description="Operation success status")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[MobileErrorDTO] = Field(None, description="Error information")
    meta: Dict[str, Any] = Field(default={}, description="Response metadata")
    
    # Mobile-specific response fields
    network_optimized: bool = Field(default=False, description="Response optimized for network")
    cached: bool = Field(default=False, description="Response served from cache")
    server_time: datetime = Field(default_factory=datetime.utcnow)
    response_size_bytes: Optional[int] = Field(None, description="Response size")


# Utility functions for mobile optimization

def optimize_for_network(data: Any, network_type: NetworkType) -> Any:
    """Optimize data based on network conditions"""
    if network_type in [NetworkType.CELLULAR_2G, NetworkType.CELLULAR_3G]:
        # Aggressive optimization for slow networks
        if isinstance(data, list):
            return data[:10]  # Limit list size
        elif isinstance(data, dict):
            # Remove non-essential fields
            essential_fields = ['id', 'title', 'price', 'location']
            return {k: v for k, v in data.items() if k in essential_fields}
    
    return data


def calculate_data_size(obj: BaseModel) -> int:
    """Calculate approximate data size of a Pydantic model"""
    try:
        json_str = obj.model_dump_json()
        return len(json_str.encode('utf-8'))
    except Exception:
        return 0