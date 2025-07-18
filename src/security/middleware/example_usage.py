"""
Security Middleware Usage Examples

This file demonstrates how to integrate and use the security middleware
components in a FastAPI application.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import os

from .middleware_integration import (
    SecurityMiddlewareStack,
    create_production_security_config,
    create_development_security_config
)
from ..auth.jwt_manager import JWTManager
from ..auth.authentication import AuthenticationManager
from ..auth.authorization import AuthorizationManager


def create_app_with_security() -> FastAPI:
    """Create FastAPI app with complete security middleware stack"""
    
    app = FastAPI(
        title="Rental ML System",
        description="Rental property ML system with comprehensive security",
        version="1.0.0"
    )
    
    # Initialize authentication components
    jwt_manager = JWTManager()
    auth_manager = AuthenticationManager()
    authorization_manager = AuthorizationManager()
    
    # Choose configuration based on environment
    if os.getenv("ENVIRONMENT") == "production":
        config = create_production_security_config()
    else:
        config = create_development_security_config()
    
    # Create and setup security middleware stack
    security_stack = SecurityMiddlewareStack(
        app=app,
        jwt_manager=jwt_manager,
        auth_manager=auth_manager,
        authorization_manager=authorization_manager,
        config=config
    )
    
    # Setup middleware stack
    security_stack.setup_middleware_stack()
    
    # Store security stack reference for later use
    app.state.security_stack = security_stack
    
    return app


def create_minimal_security_app() -> FastAPI:
    """Create FastAPI app with minimal security setup"""
    
    app = FastAPI(title="Minimal Security Example")
    
    # Add just the essential middleware
    from .input_validation_middleware import InputValidationMiddleware
    from .rate_limit_middleware import RateLimitMiddleware
    from .security_headers_middleware import SecurityHeadersMiddleware
    
    app.add_middleware(InputValidationMiddleware)
    app.add_middleware(RateLimitMiddleware, redis_url=None)  # Use in-memory storage
    app.add_middleware(SecurityHeadersMiddleware)
    
    return app


# Example usage in main application
app = create_app_with_security()


@app.on_event("startup")
async def startup_event():
    """Setup integrations on startup"""
    if hasattr(app.state, 'security_stack'):
        await app.state.security_stack.setup_integrations()
        logging.info("Security middleware integrations configured")


@app.get("/")
async def root():
    """Public endpoint"""
    return {"message": "Welcome to Rental ML System"}


@app.get("/health")
async def health_check():
    """Health check endpoint with security status"""
    if hasattr(app.state, 'security_stack'):
        security_health = await app.state.security_stack.security_health_check()
        return {
            "status": "healthy",
            "security": security_health
        }
    else:
        return {"status": "healthy", "security": "not_configured"}


@app.get("/api/v1/security/statistics")
async def get_security_statistics(request: Request):
    """Get security middleware statistics (admin only)"""
    # Check if user is admin (this would be handled by security middleware)
    security_context = getattr(request.state, 'security_context', None)
    if not security_context or not security_context.has_role("admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if hasattr(app.state, 'security_stack'):
        stats = app.state.security_stack.get_middleware_statistics()
        return stats
    else:
        return {"error": "Security stack not configured"}


@app.post("/api/v1/security/emergency-lockdown")
async def emergency_lockdown(request: Request):
    """Emergency lockdown endpoint (super admin only)"""
    security_context = getattr(request.state, 'security_context', None)
    if not security_context or not security_context.has_role("super_admin"):
        raise HTTPException(status_code=403, detail="Super admin access required")
    
    if hasattr(app.state, 'security_stack'):
        await app.state.security_stack.emergency_lockdown(duration_minutes=30)
        return {"message": "Emergency lockdown activated"}
    else:
        return {"error": "Security stack not configured"}


@app.get("/api/v1/properties/search")
async def search_properties():
    """Public property search endpoint"""
    # This endpoint is accessible without authentication
    return {
        "properties": [
            {"id": 1, "title": "Modern Apartment", "price": 2000},
            {"id": 2, "title": "Cozy House", "price": 1500}
        ]
    }


@app.get("/api/v1/users/profile")
async def get_user_profile(request: Request):
    """Protected endpoint requiring authentication"""
    security_context = getattr(request.state, 'security_context', None)
    if not security_context:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return {
        "user_id": security_context.user_id,
        "username": security_context.username,
        "roles": [role.value for role in security_context.roles]
    }


@app.get("/api/v1/admin/users")
async def list_users(request: Request):
    """Admin endpoint requiring admin role"""
    security_context = getattr(request.state, 'security_context', None)
    if not security_context or not security_context.has_role("admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "users": [
            {"id": 1, "username": "admin", "role": "admin"},
            {"id": 2, "username": "user1", "role": "tenant"}
        ]
    }


# Error handlers for security-related errors
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: HTTPException):
    """Custom handler for rate limiting errors"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please slow down.",
            "retry_after": "60"
        },
        headers={"Retry-After": "60"}
    )


@app.exception_handler(413)
async def payload_too_large_handler(request: Request, exc: HTTPException):
    """Custom handler for payload too large errors"""
    return JSONResponse(
        status_code=413,
        content={
            "error": "Payload too large",
            "message": "Request size exceeds maximum allowed limit"
        }
    )


@app.exception_handler(400)
async def validation_error_handler(request: Request, exc: HTTPException):
    """Custom handler for validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation failed",
            "message": "Request contains invalid or suspicious content"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )