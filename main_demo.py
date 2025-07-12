"""
Demo FastAPI application for the Rental ML System.

This is a simplified version of the main FastAPI application that can run 
without database dependencies for demonstration purposes.
"""

import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load demo environment variables
load_dotenv(Path(__file__).parent / ".env.demo")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockRepositoryFactory:
    """Mock repository factory for demo purposes"""
    
    def __init__(self):
        self._initialized = True
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    async def health_check(self) -> Dict[str, Any]:
        """Mock health check that always returns healthy status"""
        return {
            "database": True,
            "redis": True,
            "repositories": True,
            "overall": True
        }
    
    def get_user_repository(self):
        return MockUserRepository()
    
    def get_property_repository(self):
        return MockPropertyRepository()
    
    def get_model_repository(self):
        return MockModelRepository()
    
    def get_cache_repository(self):
        return MockCacheRepository()


class MockUserRepository:
    """Mock user repository"""
    
    async def get_active_users_count(self) -> int:
        return 42


class MockPropertyRepository:
    """Mock property repository"""
    
    async def get_count(self) -> int:
        return 150
    
    async def get_active_count(self) -> int:
        return 142
    
    async def close(self):
        pass


class MockModelRepository:
    """Mock model repository"""
    
    async def close(self):
        pass


class MockCacheRepository:
    """Mock cache repository"""
    
    async def health_check(self) -> bool:
        return True
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "hit_rate": 0.85,
            "used_memory": 1024000,
            "key_counts_by_type": {"properties": 100, "users": 42, "models": 5},
            "instantaneous_ops_per_sec": 25
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("🚀 Starting Rental ML System API (Demo Mode)...")
    
    try:
        # Initialize mock repository factory
        repository_factory = MockRepositoryFactory()
        
        # Perform mock health check
        health_status = await repository_factory.health_check()
        if not health_status.get("overall"):
            logger.error("❌ Health check failed during startup")
            raise RuntimeError("System health check failed")
        
        logger.info("✅ Demo data layer initialized successfully")
        
        # Store in app state for access in routes
        app.state.repository_factory = repository_factory
        app.state.start_time = time.time()
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Rental ML System API (Demo Mode)...")
        logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Rental ML System API (Demo)",
    description="""
    A demo version of the intelligent rental property recommendation and search system.
    
    ## Features (Demo Mode)
    
    * **Mock Data**: Uses simulated data instead of real database
    * **Health Checks**: System monitoring endpoints
    * **API Documentation**: Interactive OpenAPI documentation
    * **CORS Support**: Cross-origin resource sharing enabled
    
    ## Demo Limitations
    
    * No real database connectivity
    * Mock data responses
    * Limited functionality for demonstration purposes
    
    ## Available Endpoints
    
    * `/health` - Health check endpoints
    * `/` - API information
    * `/docs` - Interactive API documentation
    * `/redoc` - Alternative API documentation
    """,
    version="1.0.0-demo",
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
    response.headers["X-Demo-Mode"] = "true"
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
            "timestamp": time.time(),
            "demo_mode": True
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
            "timestamp": time.time(),
            "demo_mode": True
        }
    )


# Basic Health Check Routes
@app.get("/health", tags=["Health Check"])
async def health_check(request: Request) -> Dict[str, Any]:
    """Basic health check endpoint for demo mode"""
    try:
        start_time = time.time()
        repository_factory = request.app.state.repository_factory
        
        # Perform health check on mock components
        health_status = await repository_factory.health_check()
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "demo_mode": True,
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "components": {
                "database": "up (mock)",
                "redis": "up (mock)",
                "repositories": "up (mock)"
            },
            "version": "1.0.0-demo",
            "environment": "demo"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "demo_mode": True,
                "error": str(e),
                "timestamp": time.time(),
                "version": "1.0.0-demo"
            },
            status_code=503
        )


@app.get("/health/detailed", tags=["Health Check"])
async def detailed_health_check(request: Request) -> Dict[str, Any]:
    """Detailed health check with mock data"""
    try:
        start_time = time.time()
        repository_factory = request.app.state.repository_factory
        
        # Get mock health status
        health_status = await repository_factory.health_check()
        
        # Get mock statistics
        cache_repo = repository_factory.get_cache_repository()
        cache_stats = await cache_repo.get_cache_stats()
        
        property_repo = repository_factory.get_property_repository()
        total_properties = await property_repo.get_count()
        active_properties = await property_repo.get_active_count()
        
        user_repo = repository_factory.get_user_repository()
        active_users = await user_repo.get_active_users_count()
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "demo_mode": True,
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "version": "1.0.0-demo",
            "environment": "demo",
            
            "components": {
                "database": {
                    "status": "up (mock)",
                    "type": "PostgreSQL (Simulated)",
                    "metrics": {
                        "total_properties": total_properties,
                        "active_properties": active_properties,
                        "active_users": active_users
                    }
                },
                "cache": {
                    "status": "up (mock)",
                    "type": "Redis (Simulated)",
                    "metrics": cache_stats
                },
                "repositories": {
                    "status": "up (mock)",
                    "available": {
                        "user_repository": True,
                        "property_repository": True,
                        "model_repository": True,
                        "cache_repository": True
                    }
                },
                "ml_models": {
                    "status": "available (mock)",
                    "models": {
                        "hybrid_recommender": "simulated",
                        "content_based_recommender": "simulated",
                        "collaborative_filter": "simulated",
                        "search_ranker": "simulated"
                    }
                }
            },
            
            "system_info": {
                "uptime_seconds": time.time() - request.app.state.start_time,
                "memory_usage": "Simulated",
                "cpu_usage": "Simulated",
                "disk_usage": "Simulated"
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "demo_mode": True,
                "error": str(e),
                "timestamp": time.time(),
                "version": "1.0.0-demo"
            },
            status_code=503
        )


@app.get("/", response_model=Dict[str, Any], tags=["Information"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Rental ML System API (Demo)",
        "version": "1.0.0-demo",
        "description": "Intelligent rental property recommendation and search system - Demo Mode",
        "demo_mode": True,
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "health": "/health",
            "detailed_health": "/health/detailed",
            "info": "/info"
        },
        "status": "active",
        "note": "This is a demo version with mock data and limited functionality"
    }


@app.get("/info", tags=["Information"])
async def get_api_info(request: Request):
    """Get detailed API information and system status"""
    try:
        repository_factory = request.app.state.repository_factory
        health_status = await repository_factory.health_check()
        
        return {
            "api": {
                "name": "Rental ML System API (Demo)",
                "version": "1.0.0-demo",
                "environment": "demo",
                "uptime": time.time() - request.app.state.start_time
            },
            "system": {
                "database": health_status.get("database", False),
                "redis": health_status.get("redis", False),
                "repositories": health_status.get("repositories", False),
                "overall_health": health_status.get("overall", False),
                "demo_mode": True
            },
            "features": {
                "search": "mock",
                "recommendations": "mock",
                "user_tracking": "mock",
                "analytics": "mock",
                "caching": "mock"
            },
            "ml_models": {
                "collaborative_filtering": "simulated",
                "content_based": "simulated",
                "hybrid_recommender": "simulated",
                "search_ranking": "simulated"
            },
            "demo_features": {
                "health_checks": True,
                "api_documentation": True,
                "cors_support": True,
                "request_logging": True,
                "mock_data": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get API info: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service temporarily unavailable",
                "message": "Unable to retrieve system information",
                "demo_mode": True
            }
        )


@app.get("/demo", tags=["Demo"])
async def demo_endpoints():
    """List available demo endpoints and their functionality"""
    return {
        "demo_mode": True,
        "message": "Welcome to the Rental ML System API Demo",
        "available_endpoints": {
            "/": "API root information",
            "/health": "Basic health check",
            "/health/detailed": "Detailed health check with metrics",
            "/info": "Detailed API and system information",
            "/demo": "This demo information endpoint",
            "/docs": "Interactive OpenAPI documentation",
            "/redoc": "Alternative API documentation",
            "/openapi.json": "OpenAPI schema"
        },
        "demo_features": [
            "Mock data responses",
            "Health monitoring endpoints",
            "Request/response logging",
            "CORS support",
            "Interactive API documentation",
            "Error handling examples"
        ],
        "limitations": [
            "No real database connectivity",
            "Simulated ML models",
            "Mock repository responses",
            "Limited business logic"
        ],
        "next_steps": [
            "Explore the /docs endpoint for interactive API testing",
            "Check /health endpoints for monitoring examples",
            "View /info for system status information"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("APP_LOG_LEVEL", "info").lower()
    
    logger.info(f"🚀 Starting Rental ML System API Demo on {host}:{port}")
    
    uvicorn.run(
        "main_demo:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        workers=1 if reload else workers
    )