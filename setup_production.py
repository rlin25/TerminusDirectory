#!/usr/bin/env python3
"""
Quick Production Setup Script for Rental ML System

This script helps transition from demo to production by:
1. Setting up real database connections
2. Running database migrations  
3. Configuring production environment
4. Replacing mock services with real implementations
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[0;34m",    # Blue
        "success": "\033[0;32m", # Green
        "warning": "\033[0;33m", # Yellow
        "error": "\033[0;31m",   # Red
        "reset": "\033[0m"       # Reset
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️", 
        "error": "❌"
    }
    
    color = colors.get(status, colors["info"])
    icon = icons.get(status, "•")
    reset = colors["reset"]
    
    print(f"{color}{icon} {message}{reset}")

def run_command(command, description=""):
    """Run a shell command and return success status"""
    try:
        print_status(f"Running: {command}", "info")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print_status(f"✓ {description}" if description else "✓ Command completed", "success")
            return True
        else:
            print_status(f"✗ {description} failed: {result.stderr}", "error")
            return False
    except Exception as e:
        print_status(f"✗ Error running command: {e}", "error")
        return False

def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()

def create_production_env():
    """Create production environment file"""
    env_content = """# Rental ML System - Production Environment Configuration

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rental_ml
REDIS_URL=redis://localhost:6379

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
JWT_SECRET_KEY=change-this-super-secret-jwt-key-in-production
ENCRYPTION_KEY=change-this-encryption-key-too
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8501"]

# ML Configuration
ENABLE_ML_TRAINING=true
ML_MODEL_PATH=./models
ML_CACHE_SIZE=1000

# Scraping Configuration (optional - for data ingestion)
ENABLE_SCRAPING=false
SCRAPING_SCHEDULE="0 2 * * *"  # Daily at 2 AM

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090

# Performance
CONNECTION_POOL_SIZE=10
QUERY_TIMEOUT=30
CACHE_TTL=3600
"""
    
    with open('.env.production', 'w') as f:
        f.write(env_content)
    
    print_status("Created .env.production file", "success")

def create_production_main():
    """Create production version of main application"""
    
    production_main = '''"""
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
        
        # Test database connection
        await repo_factory.health_check()
        logger.info("✅ Database connection established")
        
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

# TODO: Add your API routers here
# from src.application.api.routers import property_router, user_router, search_router
# app.include_router(property_router.router, prefix="/properties", tags=["properties"])
# app.include_router(user_router.router, prefix="/users", tags=["users"])
# app.include_router(search_router.router, prefix="/search", tags=["search"])

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
'''
    
    with open('main_production.py', 'w') as f:
        f.write(production_main)
    
    print_status("Created main_production.py", "success")

def main():
    """Main setup function"""
    print_status("🏠 Rental ML System - Production Setup", "info")
    print_status("=" * 50, "info")
    
    # Check if we're in the right directory
    if not check_file_exists("src"):
        print_status("Please run this script from the project root directory", "error")
        sys.exit(1)
    
    print_status("1. Creating production environment configuration...", "info")
    create_production_env()
    
    print_status("2. Creating production application...", "info") 
    create_production_main()
    
    print_status("3. Checking Docker Compose availability...", "info")
    if shutil.which("docker-compose"):
        print_status("Docker Compose found", "success")
        
        # Start databases
        print_status("4. Starting PostgreSQL and Redis...", "info")
        if run_command("docker-compose up -d postgres redis", "Database startup"):
            print_status("Waiting for databases to initialize...", "info")
            run_command("sleep 10", "Database initialization wait")
        
        # Run migrations
        print_status("5. Running database migrations...", "info")
        if check_file_exists("migrations/run_migrations.py"):
            run_command("python3 migrations/run_migrations.py", "Database migrations")
        else:
            print_status("Migration script not found, skipping...", "warning")
    
    else:
        print_status("Docker Compose not found, skipping database setup", "warning")
        print_status("Please install Docker Compose and run: docker-compose up -d postgres redis", "info")
    
    print_status("6. Testing Python dependencies...", "info")
    try:
        # Test critical imports
        sys.path.insert(0, 'src')
        from src.domain.entities.property import Property
        from src.domain.entities.user import User
        print_status("Core entities available", "success")
    except ImportError as e:
        print_status(f"Import error: {e}", "error")
    
    print("")
    print_status("🎉 Production setup complete!", "success")
    print("")
    print_status("Next steps:", "info")
    print_status("1. Review and update .env.production with your settings", "info")
    print_status("2. Start production server: python3 main_production.py", "info")
    print_status("3. Test health endpoint: curl http://localhost:8000/health", "info")
    print_status("4. Add real property data using the scraping system", "info")
    print("")
    print_status("📚 See PRODUCTION_TRANSITION_GUIDE.md for detailed instructions", "info")

if __name__ == "__main__":
    main()