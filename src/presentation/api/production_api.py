"""
Production-ready FastAPI application for the Rental ML System.

This module provides a comprehensive production-ready API server with:
- Advanced authentication and authorization
- Rate limiting and request throttling
- API versioning and backward compatibility
- Comprehensive error handling and logging
- Request/response validation with Pydantic
- Health checks and monitoring
- Security middleware and CORS
- API documentation and OpenAPI schema
"""

import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, Request, HTTPException, Depends, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from jose import jwt, JWTError

from ...infrastructure.data import get_repository_factory, close_repository_factory, DataConfig
from .ml_endpoints import ml_router
from .property_endpoints import property_router
from .user_endpoints import user_router
from .admin_endpoints import admin_router
from .monitoring_endpoints import monitoring_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds


class SecurityConfig(BaseModel):
    """Security configuration model"""
    secret_key: str = Field(..., description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration")
    api_key: Optional[str] = Field(None, description="API key for service authentication")


class APIMetrics(BaseModel):
    """API metrics model"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)


class RateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit"""
        try:
            current = await self.redis.get(key)
            if current is None:
                await self.redis.setex(key, window, 1)
                return True
            
            current_count = int(current)
            if current_count >= limit:
                return False
            
            await self.redis.incr(key)
            return True
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow request if rate limiter fails


class SecurityManager:
    """Centralized security management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blacklisted_tokens = set()
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        if token in self.blacklisted_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token)
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        return api_key == self.config.api_key if self.config.api_key else False


# Global variables
rate_limiter: Optional[RateLimiter] = None
security_manager: Optional[SecurityManager] = None
api_metrics = APIMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    global rate_limiter, security_manager
    
    # Startup
    logger.info("🚀 Starting Production Rental ML System API...")
    
    try:
        # Initialize Redis for rate limiting
        redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        rate_limiter = RateLimiter(redis_client)
        
        # Initialize security manager
        security_config = SecurityConfig(
            secret_key=SECRET_KEY,
            api_key=os.getenv("API_KEY")
        )
        security_manager = SecurityManager(security_config)
        
        # Initialize data layer
        config = DataConfig()
        repository_factory = await get_repository_factory(config)
        
        # Perform health check
        health_status = await repository_factory.health_check()
        if not health_status.get("overall"):
            logger.error("❌ Health check failed during startup")
            raise RuntimeError("System health check failed")
        
        logger.info("✅ All systems initialized successfully")
        
        # Store in app state
        app.state.repository_factory = repository_factory
        app.state.config = config
        app.state.redis_client = redis_client
        app.state.rate_limiter = rate_limiter
        app.state.security_manager = security_manager
        app.state.start_time = time.time()
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Production API...")
        await close_repository_factory()
        if 'redis_client' in locals():
            await redis_client.close()
        logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Rental ML System - Production API",
    description="""
    ## Production-Ready Rental Property Recommendation System
    
    A comprehensive machine learning-powered platform for rental property recommendations and search.
    
    ### Features
    
    * **🔐 Enterprise Security**: JWT authentication, API keys, rate limiting
    * **🤖 ML Inference**: Real-time recommendations and search ranking
    * **📊 Analytics**: Comprehensive monitoring and metrics
    * **🏠 Property Management**: Full CRUD operations with bulk processing
    * **👥 User Management**: Profile management and interaction tracking
    * **⚡ High Performance**: Async operations with Redis caching
    * **📈 Monitoring**: Real-time system health and performance metrics
    
    ### Authentication
    
    This API supports multiple authentication methods:
    - **JWT Tokens**: For user authentication (Bearer token)
    - **API Keys**: For service-to-service communication (X-API-Key header)
    
    ### Rate Limiting
    
    All endpoints are rate-limited to ensure system stability:
    - **Default**: 100 requests per minute per IP
    - **Authenticated users**: Higher limits based on subscription tier
    
    ### Monitoring
    
    Comprehensive monitoring includes:
    - Request/response metrics
    - ML model performance
    - System health indicators
    - Business KPIs
    """,
    version="2.0.0",
    docs_url=None,  # Custom docs
    redoc_url=None,  # Custom redoc
    openapi_url="/api/v2/openapi.json",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for request validation"""
    # Add security headers
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    if rate_limiter is None:
        return await call_next(request)
    
    # Get client IP
    client_ip = request.client.host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    # Check rate limit
    rate_limit_key = f"rate_limit:{client_ip}"
    is_allowed = await rate_limiter.is_allowed(
        rate_limit_key, 
        RATE_LIMIT_REQUESTS, 
        RATE_LIMIT_WINDOW
    )
    
    if not is_allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds",
                "retry_after": RATE_LIMIT_WINDOW
            }
        )
    
    return await call_next(request)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Metrics collection middleware"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
    
    try:
        response = await call_next(request)
        
        # Update metrics
        duration = time.time() - start_time
        api_metrics.total_requests += 1
        
        if response.status_code < 400:
            api_metrics.successful_requests += 1
        else:
            api_metrics.failed_requests += 1
        
        # Update average response time
        api_metrics.average_response_time = (
            (api_metrics.average_response_time * (api_metrics.total_requests - 1) + duration) /
            api_metrics.total_requests
        )
        api_metrics.last_updated = datetime.now()
        
        # Add timing headers
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-Request-ID"] = str(id(request))
        
        logger.info(
            f"📤 {request.method} {request.url.path} - "
            f"{response.status_code} - {duration:.3f}s"
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        api_metrics.total_requests += 1
        api_metrics.failed_requests += 1
        
        logger.error(
            f"❌ {request.method} {request.url.path} - "
            f"ERROR: {str(e)} - {duration:.3f}s"
        )
        raise


# Authentication dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current authenticated user from JWT token"""
    if not security_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Security manager not initialized"
        )
    
    try:
        payload = security_manager.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        return {"user_id": user_id, "payload": payload}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for service authentication"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if not security_manager or not security_manager.verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return {"api_key": api_key, "authenticated": True}


# Exception handlers
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
            "timestamp": datetime.now().isoformat(),
            "request_id": str(id(request))
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
            "timestamp": datetime.now().isoformat(),
            "request_id": str(id(request))
        }
    )


# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Interactive API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc documentation"""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Reference",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.2/bundles/redoc.standalone.js",
    )


# Include routers with API versioning
app.include_router(
    ml_router,
    prefix="/api/v2/ml",
    tags=["Machine Learning"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    property_router,
    prefix="/api/v2/properties",
    tags=["Properties"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    user_router,
    prefix="/api/v2/users",
    tags=["Users"]
)

app.include_router(
    admin_router,
    prefix="/api/v2/admin",
    tags=["Administration"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    monitoring_router,
    prefix="/api/v2/monitoring",
    tags=["Monitoring"]
)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Rental ML System - Production API",
        "version": "2.0.0",
        "description": "Production-ready rental property recommendation system",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "status": "active",
        "documentation": {
            "interactive": "/docs",
            "reference": "/redoc",
            "openapi": "/api/v2/openapi.json"
        },
        "endpoints": {
            "ml": "/api/v2/ml",
            "properties": "/api/v2/properties",
            "users": "/api/v2/users",
            "admin": "/api/v2/admin",
            "monitoring": "/api/v2/monitoring"
        },
        "authentication": {
            "jwt": "Bearer token in Authorization header",
            "api_key": "X-API-Key header for service authentication"
        },
        "rate_limiting": {
            "requests_per_minute": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW
        }
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        repository_factory = app.state.repository_factory
        health_status = await repository_factory.health_check()
        
        return {
            "status": "healthy" if health_status.get("overall") else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - app.state.start_time,
            "version": "2.0.0",
            "components": {
                "database": {
                    "status": "healthy" if health_status.get("database") else "unhealthy",
                    "message": "PostgreSQL connection active"
                },
                "cache": {
                    "status": "healthy" if health_status.get("redis") else "unhealthy",
                    "message": "Redis connection active"
                },
                "repositories": {
                    "status": "healthy" if health_status.get("repositories") else "unhealthy",
                    "message": "All repositories functional"
                }
            },
            "metrics": {
                "total_requests": api_metrics.total_requests,
                "successful_requests": api_metrics.successful_requests,
                "failed_requests": api_metrics.failed_requests,
                "success_rate": (
                    api_metrics.successful_requests / api_metrics.total_requests * 100
                    if api_metrics.total_requests > 0 else 0
                ),
                "average_response_time": api_metrics.average_response_time
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Health check failed",
                "message": str(e)
            }
        )


def custom_openapi():
    """Enhanced OpenAPI schema with security and metadata"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for user authentication"
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service authentication"
        }
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "Production server"
        }
    ]
    
    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info"),
        access_log=True,
        server_header=False,
        date_header=False
    )