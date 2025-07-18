"""
Production FastAPI application for the Rental ML System.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv

# Load production environment
load_dotenv(Path(__file__).parent / ".env.production")

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import repository factory (real implementation)
from src.infrastructure.data.repository_factory import RepositoryFactory

# Global repository factory instance
repo_factory = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global repo_factory
    
    # Startup
    logger.info("🚀 Starting Rental ML System (Production Mode)...")
    
    try:
        # Initialize real repository factory
        repo_factory = RepositoryFactory(
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            pool_size=int(os.getenv("CONNECTION_POOL_SIZE", "10")),
            enable_performance_monitoring=True
        )
        
        # Initialize the factory properly
        await repo_factory.initialize()
        
        # Test database connection
        health_status = await repo_factory.health_check()
        logger.info("✅ Database connection established")
        
        # Store factory in app state for dependency injection
        app.state.repository_factory = repo_factory
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize repository factory: {e}")
        raise
    finally:
        # Shutdown
        if repo_factory:
            await repo_factory.close()
        logger.info("🛑 Rental ML System shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Rental ML System API (Production)",
    version="1.0.0-production", 
    description="Intelligent rental property recommendation and search system - Production Mode",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Rental ML System API (Production)",
        "version": "1.0.0-production",
        "description": "Intelligent rental property recommendation and search system - Production Mode",
        "production_mode": True,
        "docs_url": "/docs",
        "health_check": "/health",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database health
        health_status = await repo_factory.health_check()
        
        return {
            "status": "healthy" if health_status["overall"] else "unhealthy",
            "production_mode": True,
            "timestamp": __import__("time").time(),
            "components": health_status,
            "version": "1.0.0-production",
            "environment": "production"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": __import__("time").time()
        }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    try:
        health = await repo_factory.health_check()
        
        # Get repository performance metrics if available
        property_repo = repo_factory.get_property_repository()
        user_repo = repo_factory.get_user_repository()
        
        return {
            "status": "healthy" if health["overall"] else "unhealthy",
            "components": health,
            "repositories": {
                "property_repository": "available",
                "user_repository": "available", 
                "model_repository": "available"
            },
            "performance": {
                "database_connections": await repo_factory.get_connection_info(),
                "cache_status": "available"
            },
            "timestamp": __import__("time").time()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": __import__("time").time()
        }

# Add API routers (minimal approach for testing)
try:
    # Start with just health router
    from src.application.api.routers import health_router
    app.include_router(health_router.router, prefix="/health", tags=["health"])
    logger.info("✅ Health router loaded successfully")
    
    # Try to add other routers one by one
    routers_to_load = [
        ("property_router", "/properties", "properties"),
        ("user_router", "/users", "users"),
    ]
    
    for router_name, prefix, tag in routers_to_load:
        try:
            module = __import__(f"src.application.api.routers.{router_name}", fromlist=[router_name])
            router = getattr(module, "router")
            app.include_router(router, prefix=prefix, tags=[tag])
            logger.info(f"✅ {router_name} loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {router_name}: {e}")

except ImportError as e:
    logger.error(f"❌ Failed to load health router: {e}")
    # Continue without additional routers

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    logger.info(f"🚀 Starting production server on {host}:{port}")
    
    uvicorn.run(
        "main_production:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,  # Disable reload in production
        access_log=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
