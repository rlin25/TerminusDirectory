"""
Main FastAPI application for the Rental ML System.

This module sets up the FastAPI application with all routes, middleware,
and configuration for the rental property recommendation system.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .routers import search_router, recommendation_router, property_router, user_router, health_router, scraping_router
from ...infrastructure.data import get_repository_factory, close_repository_factory, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("🚀 Starting Rental ML System API...")
    
    try:
        # Initialize data layer
        config = DataConfig()
        repository_factory = await get_repository_factory(config)
        
        # Perform health check
        health_status = await repository_factory.health_check()
        if not health_status.get("overall"):
            logger.error("❌ Health check failed during startup")
            raise RuntimeError("System health check failed")
        
        logger.info("✅ Data layer initialized successfully")
        
        # Store in app state for access in routes
        app.state.repository_factory = repository_factory
        app.state.config = config
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Rental ML System API...")
        await close_repository_factory()
        logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Rental ML System API",
    description="""
    An intelligent rental property recommendation and search system powered by machine learning.
    
    ## Features
    
    * **Advanced Search**: NLP-powered property search with intelligent ranking
    * **Personalized Recommendations**: Hybrid ML-based property recommendations
    * **User Interactions**: Track and learn from user behavior
    * **Similar Properties**: Find properties similar to user preferences
    * **Real-time Analytics**: Performance metrics and insights
    
    ## Authentication
    
    Currently, the API uses simple user ID-based authentication. In production,
    this should be replaced with proper OAuth2 or JWT authentication.
    
    ## Rate Limiting
    
    API endpoints are rate-limited to ensure fair usage and system stability.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
    
    try:
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info(
            f"📤 {request.method} {request.url.path} - "
            f"{response.status_code} - {duration:.3f}s"
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"❌ {request.method} {request.url.path} - "
            f"ERROR: {str(e)} - {duration:.3f}s"
        )
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "status_code": 500,
            "path": request.url.path,
            "method": request.method,
            "timestamp": time.time()
        }
    )


# Include routers
app.include_router(
    health_router.router,
    prefix="/health",
    tags=["Health Check"]
)

app.include_router(
    search_router.router,
    prefix="/api/v1/search",
    tags=["Search"]
)

app.include_router(
    recommendation_router.router,
    prefix="/api/v1/recommendations",
    tags=["Recommendations"]
)

app.include_router(
    property_router.router,
    prefix="/api/v1/properties",
    tags=["Properties"]
)

app.include_router(
    user_router.router,
    prefix="/api/v1/users",
    tags=["Users"]
)

app.include_router(
    scraping_router.router,
    prefix="/api/v1/scraping",
    tags=["Scraping"]
)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Rental ML System API",
        "version": "1.0.0",
        "description": "Intelligent rental property recommendation and search system",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "search": "/api/v1/search",
            "recommendations": "/api/v1/recommendations",
            "properties": "/api/v1/properties",
            "users": "/api/v1/users",
            "scraping": "/api/v1/scraping"
        },
        "status": "active"
    }


@app.get("/info")
async def get_api_info():
    """Get detailed API information and system status"""
    try:
        repository_factory = app.state.repository_factory
        health_status = await repository_factory.health_check()
        
        return {
            "api": {
                "name": "Rental ML System API",
                "version": "1.0.0",
                "environment": "development",  # Should come from config
                "uptime": time.time() - app.state.get("start_time", time.time())
            },
            "system": {
                "database": health_status.get("database", False),
                "redis": health_status.get("redis", False),
                "repositories": health_status.get("repositories", False),
                "overall_health": health_status.get("overall", False)
            },
            "features": {
                "search": True,
                "recommendations": True,
                "user_tracking": True,
                "analytics": True,
                "caching": True
            },
            "ml_models": {
                "collaborative_filtering": True,
                "content_based": True,
                "hybrid_recommender": True,
                "search_ranking": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get API info: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service temporarily unavailable",
                "message": "Unable to retrieve system information"
            }
        )


def custom_openapi():
    """Custom OpenAPI schema with additional metadata"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Rental ML System API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "Development server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "UserID": {
            "type": "apiKey",
            "in": "header",
            "name": "X-User-ID",
            "description": "User ID for personalized responses"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    # Set start time for uptime calculation
    app.state.start_time = time.time()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )