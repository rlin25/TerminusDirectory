"""
Mobile-optimized FastAPI application for the rental ML system.

This module provides mobile-specific endpoints with reduced payload sizes,
optimized for mobile network conditions and offline-first architecture.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import MobileAuthHandler, get_current_mobile_user
from .notifications import PushNotificationService
from .sync import OfflineSyncManager
from .dto import (
    MobilePropertyDTO, MobileSearchResultDTO, MobileRecommendationDTO,
    MobileUserProfileDTO, SyncRequestDTO, SyncResponseDTO
)
from ...application.dto.property_dto import PropertySearchDTO
from ...application.dto.recommendation_dto import RecommendationRequestDTO
from ...infrastructure.data import get_repository_factory, close_repository_factory, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileAPIConfig(BaseModel):
    """Configuration for mobile API"""
    max_properties_per_page: int = Field(default=20, description="Maximum properties per page")
    max_recommendations: int = Field(default=10, description="Maximum recommendations")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    image_quality: str = Field(default="medium", description="Image quality: low, medium, high")
    enable_push_notifications: bool = Field(default=True)
    enable_offline_sync: bool = Field(default=True)
    sync_batch_size: int = Field(default=50, description="Sync batch size")


@asynccontextmanager
async def mobile_lifespan(app: FastAPI):
    """Mobile API lifespan context manager"""
    # Startup
    logger.info("🚀 Starting Mobile API...")
    
    try:
        # Initialize data layer
        config = DataConfig()
        repository_factory = await get_repository_factory(config)
        
        # Initialize mobile services
        auth_handler = MobileAuthHandler()
        push_service = PushNotificationService()
        sync_manager = OfflineSyncManager(repository_factory)
        
        # Store in app state
        app.state.repository_factory = repository_factory
        app.state.config = config
        app.state.mobile_config = MobileAPIConfig()
        app.state.auth_handler = auth_handler
        app.state.push_service = push_service
        app.state.sync_manager = sync_manager
        
        logger.info("✅ Mobile API initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Mobile API startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Mobile API...")
        await close_repository_factory()
        logger.info("✅ Mobile API shutdown complete")


def create_mobile_app() -> FastAPI:
    """Create and configure the mobile FastAPI application"""
    
    app = FastAPI(
        title="Rental ML System - Mobile API",
        description="""
        Mobile-optimized API for the rental property recommendation system.
        
        ## Features
        
        * **Optimized Payloads**: Reduced data transfer for mobile networks
        * **Offline Support**: Data synchronization and offline-first architecture
        * **Push Notifications**: Real-time property alerts and updates
        * **Mobile Authentication**: Biometric and secure mobile auth
        * **Image Optimization**: Compressed images for mobile devices
        * **Location Services**: Geofencing and location-based search
        
        ## Mobile-Specific Optimizations
        
        * Compressed responses for bandwidth efficiency
        * Paginated results with mobile-friendly page sizes
        * Image resizing and compression
        * Reduced JSON payload sizes
        * Battery-efficient caching strategies
        """,
        version="1.0.0",
        docs_url="/mobile/docs",
        redoc_url="/mobile/redoc",
        openapi_url="/mobile/openapi.json",
        lifespan=mobile_lifespan
    )
    
    # Add mobile-optimized middleware
    app.add_middleware(GZipMiddleware, minimum_size=500)  # Lower threshold for mobile
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for mobile apps
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def mobile_optimization_middleware(request: Request, call_next):
        """Mobile-specific optimizations"""
        start_time = time.time()
        
        # Add mobile headers
        response = await call_next(request)
        
        # Add mobile-specific headers
        response.headers["X-Mobile-Optimized"] = "true"
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 min cache
        
        return response
    
    @app.middleware("http")
    async def network_condition_middleware(request: Request, call_next):
        """Adapt to network conditions"""
        network_type = request.headers.get("X-Network-Type", "unknown")
        
        # Store network info for response optimization
        request.state.network_type = network_type
        
        response = await call_next(request)
        
        # Add network-aware cache headers
        if network_type in ["2g", "slow-2g"]:
            response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour for slow networks
        elif network_type in ["3g"]:
            response.headers["Cache-Control"] = "public, max-age=900"   # 15 min
        else:
            response.headers["Cache-Control"] = "public, max-age=300"   # 5 min
            
        return response
    
    # Root endpoint
    @app.get("/mobile")
    async def mobile_root():
        """Mobile API root endpoint"""
        return {
            "name": "Rental ML System - Mobile API",
            "version": "1.0.0",
            "description": "Mobile-optimized rental property API",
            "features": {
                "offline_support": True,
                "push_notifications": True,
                "image_optimization": True,
                "location_services": True,
                "biometric_auth": True
            },
            "endpoints": {
                "search": "/mobile/api/v1/search",
                "recommendations": "/mobile/api/v1/recommendations",
                "properties": "/mobile/api/v1/properties",
                "user": "/mobile/api/v1/user",
                "sync": "/mobile/api/v1/sync",
                "notifications": "/mobile/api/v1/notifications"
            }
        }
    
    # Mobile-optimized search endpoint
    @app.get("/mobile/api/v1/search", response_model=MobileSearchResultDTO)
    async def mobile_search(
        query: str,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        bedrooms: Optional[int] = None,
        bathrooms: Optional[int] = None,
        page: int = 1,
        limit: int = 20,
        image_quality: Optional[str] = None,
        request: Request = None,
        user_id: str = Depends(get_current_mobile_user)
    ):
        """Mobile-optimized property search with reduced payloads"""
        try:
            mobile_config = request.app.state.mobile_config
            repository_factory = request.app.state.repository_factory
            
            # Limit results for mobile
            limit = min(limit, mobile_config.max_properties_per_page)
            
            # Build search request
            search_request = PropertySearchDTO(
                query=query,
                location=location,
                latitude=latitude,
                longitude=longitude,
                max_price=max_price,
                min_price=min_price,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                limit=limit,
                offset=(page - 1) * limit
            )
            
            # Get search service
            search_service = repository_factory.get_search_service()
            
            # Perform search
            search_results = await search_service.search_properties(search_request)
            
            # Convert to mobile format with optimized payloads
            mobile_properties = []
            quality = image_quality or mobile_config.image_quality
            network_type = getattr(request.state, 'network_type', 'unknown')
            
            for prop in search_results.properties:
                mobile_prop = MobilePropertyDTO.from_property(
                    prop, 
                    image_quality=quality,
                    network_type=network_type
                )
                mobile_properties.append(mobile_prop)
            
            return MobileSearchResultDTO(
                properties=mobile_properties,
                total=search_results.total,
                page=page,
                limit=limit,
                has_more=search_results.total > page * limit,
                search_time=search_results.search_time,
                network_optimized=True
            )
            
        except Exception as e:
            logger.error(f"Mobile search error: {e}")
            raise HTTPException(status_code=500, detail="Search failed")
    
    # Mobile recommendations endpoint
    @app.get("/mobile/api/v1/recommendations", response_model=List[MobileRecommendationDTO])
    async def mobile_recommendations(
        request: Request,
        limit: int = 10,
        include_images: bool = True,
        user_id: str = Depends(get_current_mobile_user)
    ):
        """Get mobile-optimized property recommendations"""
        try:
            mobile_config = request.app.state.mobile_config
            repository_factory = request.app.state.repository_factory
            
            # Limit recommendations for mobile
            limit = min(limit, mobile_config.max_recommendations)
            
            # Build recommendation request
            rec_request = RecommendationRequestDTO(
                user_id=user_id,
                limit=limit,
                include_explanations=False  # Reduce payload
            )
            
            # Get recommendation service
            rec_service = repository_factory.get_recommendation_service()
            
            # Get recommendations
            recommendations = await rec_service.get_recommendations(rec_request)
            
            # Convert to mobile format
            mobile_recs = []
            network_type = getattr(request.state, 'network_type', 'unknown')
            
            for rec in recommendations:
                mobile_rec = MobileRecommendationDTO.from_recommendation(
                    rec,
                    include_images=include_images,
                    network_type=network_type
                )
                mobile_recs.append(mobile_rec)
            
            return mobile_recs
            
        except Exception as e:
            logger.error(f"Mobile recommendations error: {e}")
            raise HTTPException(status_code=500, detail="Recommendations failed")
    
    # User profile endpoint
    @app.get("/mobile/api/v1/user/profile", response_model=MobileUserProfileDTO)
    async def get_mobile_user_profile(
        request: Request,
        user_id: str = Depends(get_current_mobile_user)
    ):
        """Get mobile-optimized user profile"""
        try:
            repository_factory = request.app.state.repository_factory
            user_repository = repository_factory.get_user_repository()
            
            user = await user_repository.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            return MobileUserProfileDTO.from_user(user)
            
        except Exception as e:
            logger.error(f"Mobile user profile error: {e}")
            raise HTTPException(status_code=500, detail="Profile fetch failed")
    
    # Offline sync endpoint
    @app.post("/mobile/api/v1/sync", response_model=SyncResponseDTO)
    async def sync_offline_data(
        sync_request: SyncRequestDTO,
        background_tasks: BackgroundTasks,
        request: Request,
        user_id: str = Depends(get_current_mobile_user)
    ):
        """Synchronize offline data with server"""
        try:
            sync_manager = request.app.state.sync_manager
            
            # Process sync request
            sync_response = await sync_manager.process_sync(
                user_id=user_id,
                sync_request=sync_request
            )
            
            # Schedule background sync if needed
            if sync_response.has_conflicts:
                background_tasks.add_task(
                    sync_manager.resolve_conflicts,
                    user_id,
                    sync_response.conflicts
                )
            
            return sync_response
            
        except Exception as e:
            logger.error(f"Sync error: {e}")
            raise HTTPException(status_code=500, detail="Sync failed")
    
    # Push notification registration
    @app.post("/mobile/api/v1/notifications/register")
    async def register_push_token(
        device_token: str,
        device_type: str,  # ios, android
        request: Request,
        user_id: str = Depends(get_current_mobile_user)
    ):
        """Register device for push notifications"""
        try:
            push_service = request.app.state.push_service
            
            await push_service.register_device(
                user_id=user_id,
                device_token=device_token,
                device_type=device_type
            )
            
            return {"status": "registered", "message": "Device registered for notifications"}
            
        except Exception as e:
            logger.error(f"Push registration error: {e}")
            raise HTTPException(status_code=500, detail="Registration failed")
    
    # Location-based search
    @app.get("/mobile/api/v1/properties/nearby")
    async def get_nearby_properties(
        latitude: float,
        longitude: float,
        radius: float = 5.0,  # km
        limit: int = 20,
        request: Request = None,
        user_id: str = Depends(get_current_mobile_user)
    ):
        """Get properties near user location"""
        try:
            mobile_config = request.app.state.mobile_config
            repository_factory = request.app.state.repository_factory
            
            limit = min(limit, mobile_config.max_properties_per_page)
            
            # Get property repository
            property_repository = repository_factory.get_property_repository()
            
            # Search nearby properties
            nearby_properties = await property_repository.get_properties_by_location(
                latitude=latitude,
                longitude=longitude,
                radius=radius,
                limit=limit
            )
            
            # Convert to mobile format
            mobile_properties = []
            network_type = getattr(request.state, 'network_type', 'unknown')
            
            for prop in nearby_properties:
                mobile_prop = MobilePropertyDTO.from_property(
                    prop,
                    image_quality=mobile_config.image_quality,
                    network_type=network_type
                )
                mobile_properties.append(mobile_prop)
            
            return {
                "properties": mobile_properties,
                "location": {"latitude": latitude, "longitude": longitude},
                "radius": radius,
                "count": len(mobile_properties)
            }
            
        except Exception as e:
            logger.error(f"Nearby properties error: {e}")
            raise HTTPException(status_code=500, detail="Location search failed")
    
    # Health check for mobile
    @app.get("/mobile/health")
    async def mobile_health_check(request: Request):
        """Mobile API health check"""
        try:
            repository_factory = request.app.state.repository_factory
            health_status = await repository_factory.health_check()
            
            return {
                "status": "healthy" if health_status.get("overall") else "unhealthy",
                "mobile_api": True,
                "services": {
                    "database": health_status.get("database", False),
                    "redis": health_status.get("redis", False),
                    "push_notifications": True,
                    "offline_sync": True
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Mobile health check error: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    mobile_app = create_mobile_app()
    uvicorn.run(
        mobile_app,
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )