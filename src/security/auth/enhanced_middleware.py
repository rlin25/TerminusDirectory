"""
Enhanced Security Middleware

Production-ready middleware that integrates with the unified security manager
for comprehensive authentication, authorization, and security monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from functools import wraps

from fastapi import HTTPException, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .models import Permission, SecurityContext, ThreatLevel, SecurityEventType
from .security_integration import SecurityManager


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware that provides:
    - Request authentication and authorization
    - Rate limiting
    - Suspicious activity detection
    - Security event logging
    - Request/response monitoring
    """
    
    def __init__(self, app, security_manager: SecurityManager, config: Dict[str, Any] = None):
        super().__init__(app)
        self.security_manager = security_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Middleware configuration
        self.enabled = self.config.get("enabled", True)
        self.log_requests = self.config.get("log_requests", True)
        self.check_rate_limits = self.config.get("check_rate_limits", True)
        self.detect_suspicious_activity = self.config.get("detect_suspicious_activity", True)
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # Exempt paths (no authentication required)
        self.exempt_paths = set(self.config.get("exempt_paths", [
            "/health",
            "/docs",
            "/openapi.json",
            "/auth/login",
            "/auth/register",
            "/auth/oauth2/authorize",
            "/auth/oauth2/callback"
        ]))
        
        # Path-specific rate limits
        self.path_rate_limits = self.config.get("path_rate_limits", {
            "/auth/login": {"limit": 5, "window": 300},  # 5 attempts per 5 minutes
            "/auth/register": {"limit": 3, "window": 3600},  # 3 attempts per hour
            "default": {"limit": 100, "window": 60}  # 100 requests per minute
        })
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        if not self.enabled:
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            # Extract request information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            path = request.url.path
            method = request.method
            
            # Add security context to request state
            request.state.security_context = None
            request.state.client_ip = client_ip
            request.state.user_agent = user_agent
            
            # Skip security checks for exempt paths
            if path in self.exempt_paths:
                response = await call_next(request)
                self._add_security_headers(response)
                return response
            
            # Rate limiting
            if self.check_rate_limits:
                rate_limit_result = await self._check_rate_limits(request, client_ip, path)
                if not rate_limit_result["allowed"]:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "retry_after": rate_limit_result.get("retry_after", 60)
                        },
                        headers={"Retry-After": str(rate_limit_result.get("retry_after", 60))}
                    )
            
            # Authentication
            auth_result = await self._authenticate_request(request)
            if not auth_result["success"]:
                await self._log_failed_authentication(request, auth_result["error"])
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication required", "details": auth_result["error"]},
                    headers=self._get_auth_headers()
                )
            
            # Set security context
            request.state.security_context = auth_result["security_context"]
            
            # Suspicious activity detection
            if self.detect_suspicious_activity:
                await self._detect_suspicious_activity(request)
            
            # Process request
            response = await call_next(request)
            
            # Log successful request
            if self.log_requests:
                processing_time = time.time() - start_time
                await self._log_request(request, response, processing_time)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            await self._log_http_exception(request, e)
            raise
            
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Security middleware error: {e}")
            await self._log_security_error(request, str(e))
            
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
                headers=self.security_headers
            )
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate request using various methods"""
        try:
            # Try JWT token authentication first
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]  # Remove "Bearer " prefix
                
                auth_result = await self.security_manager.authenticate_jwt_token(token)
                if auth_result.success:
                    return {
                        "success": True,
                        "security_context": auth_result.security_context,
                        "method": "jwt"
                    }
            
            # Try API key authentication
            api_key = request.headers.get("X-API-Key")
            if api_key:
                auth_result = await self.security_manager.authenticate_api_key(
                    api_key,
                    request.state.client_ip
                )
                if auth_result.success:
                    return {
                        "success": True,
                        "security_context": auth_result.security_context,
                        "method": "api_key"
                    }
            
            # Try session authentication
            session_token = request.cookies.get("session_token")
            if session_token:
                session = await self.security_manager.validate_session(session_token)
                if session:
                    # Create security context from session
                    # (In production, you'd fetch user data from the session)
                    security_context = SecurityContext(
                        user_id=session.user_id,
                        username="session_user",
                        email="",
                        roles=[],
                        permissions=set(),
                        session_id=session_token,
                        ip_address=request.state.client_ip,
                        user_agent=request.state.user_agent
                    )
                    
                    return {
                        "success": True,
                        "security_context": security_context,
                        "method": "session"
                    }
            
            return {
                "success": False,
                "error": "No valid authentication credentials provided"
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return {
                "success": False,
                "error": "Authentication service error"
            }
    
    async def _check_rate_limits(self, request: Request, client_ip: str, path: str) -> Dict[str, Any]:
        """Check rate limits for the request"""
        try:
            # Get rate limit configuration for path
            rate_config = self.path_rate_limits.get(path, self.path_rate_limits["default"])
            limit = rate_config["limit"]
            window = rate_config["window"]
            
            # Check rate limit
            return await self.security_manager.check_rate_limit(
                client_ip,
                limit,
                window,
                f"path_{path.replace('/', '_')}"
            )
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return {"allowed": True}  # Allow request if rate limiting fails
    
    async def _detect_suspicious_activity(self, request: Request):
        """Detect suspicious activity patterns"""
        try:
            security_context = request.state.security_context
            if security_context:
                metadata = {
                    "path": request.url.path,
                    "method": request.method,
                    "headers": dict(request.headers),
                    "query_params": dict(request.query_params)
                }
                
                await self.security_manager.detect_suspicious_activity(
                    security_context.user_id,
                    request.state.client_ip,
                    "api_request",
                    metadata
                )
                
        except Exception as e:
            self.logger.error(f"Suspicious activity detection error: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded IP headers
        forwarded_ips = [
            request.headers.get("X-Forwarded-For"),
            request.headers.get("X-Real-IP"),
            request.headers.get("CF-Connecting-IP"),  # Cloudflare
            request.headers.get("X-Client-IP")
        ]
        
        for ip in forwarded_ips:
            if ip:
                # Handle comma-separated IPs (take the first one)
                return ip.split(",")[0].strip()
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication-related headers"""
        return {
            "WWW-Authenticate": 'Bearer realm="API", charset="UTF-8"',
            **self.security_headers
        }
    
    async def _log_request(self, request: Request, response: Response, processing_time: float):
        """Log successful request"""
        try:
            security_context = request.state.security_context
            username = security_context.username if security_context else "anonymous"
            
            self.logger.info(
                f"{request.method} {request.url.path} - "
                f"User: {username} - "
                f"IP: {request.state.client_ip} - "
                f"Status: {response.status_code} - "
                f"Time: {processing_time:.3f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Request logging error: {e}")
    
    async def _log_failed_authentication(self, request: Request, error: str):
        """Log failed authentication attempt"""
        try:
            await self.security_manager._log_security_event(
                SecurityEventType.LOGIN_FAILURE,
                None,
                request.state.client_ip,
                request.state.user_agent,
                f"Failed authentication: {error}",
                ThreatLevel.MEDIUM
            )
            
        except Exception as e:
            self.logger.error(f"Failed authentication logging error: {e}")
    
    async def _log_http_exception(self, request: Request, exception: HTTPException):
        """Log HTTP exceptions"""
        try:
            security_context = request.state.security_context
            username = security_context.username if security_context else "anonymous"
            
            self.logger.warning(
                f"HTTP Exception - {request.method} {request.url.path} - "
                f"User: {username} - "
                f"IP: {request.state.client_ip} - "
                f"Status: {exception.status_code} - "
                f"Detail: {exception.detail}"
            )
            
        except Exception as e:
            self.logger.error(f"HTTP exception logging error: {e}")
    
    async def _log_security_error(self, request: Request, error: str):
        """Log security-related errors"""
        try:
            await self.security_manager._log_security_event(
                SecurityEventType.SECURITY_VIOLATION,
                None,
                request.state.client_ip,
                request.state.user_agent,
                f"Security middleware error: {error}",
                ThreatLevel.HIGH
            )
            
        except Exception as e:
            self.logger.error(f"Security error logging error: {e}")


class SecurityDependency:
    """FastAPI dependency for security context injection"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.bearer_scheme = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request) -> SecurityContext:
        """Extract security context from request state"""
        security_context = getattr(request.state, "security_context", None)
        
        if not security_context:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return security_context


def require_permissions(*permissions: Permission):
    """Decorator to require specific permissions for an endpoint"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context from request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(status_code=500, detail="Request object not found")
            
            security_context = getattr(request.state, "security_context", None)
            if not security_context:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check permissions
            missing_permissions = [
                perm for perm in permissions
                if not security_context.has_permission(perm)
            ]
            
            if missing_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required permissions: {[p.value for p in missing_permissions]}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_roles(*roles):
    """Decorator to require specific roles for an endpoint"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context from request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(status_code=500, detail="Request object not found")
            
            security_context = getattr(request.state, "security_context", None)
            if not security_context:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check roles
            has_required_role = any(security_context.has_role(role) for role in roles)
            
            if not has_required_role:
                raise HTTPException(
                    status_code=403,
                    detail=f"Required role not found. Required: {[r.value for r in roles]}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


async def get_current_user(request: Request) -> SecurityContext:
    """FastAPI dependency to get current user's security context"""
    security_context = getattr(request.state, "security_context", None)
    
    if not security_context:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return security_context


async def get_optional_user(request: Request) -> Optional[SecurityContext]:
    """FastAPI dependency to get current user's security context (optional)"""
    return getattr(request.state, "security_context", None)


class APIRateLimiter:
    """Rate limiter for specific API endpoints"""
    
    def __init__(self, security_manager: SecurityManager, limit: int, window: int = 60):
        self.security_manager = security_manager
        self.limit = limit
        self.window = window
    
    async def __call__(self, request: Request):
        """Check rate limit for specific endpoint"""
        client_ip = getattr(request.state, "client_ip", "unknown")
        
        result = await self.security_manager.check_rate_limit(
            client_ip,
            self.limit,
            self.window,
            f"endpoint_{request.url.path.replace('/', '_')}"
        )
        
        if not result["allowed"]:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(result.get("retry_after", self.window))}
            )


# Factory functions for common rate limiters
def create_login_rate_limiter(security_manager: SecurityManager):
    """Create rate limiter for login endpoints (5 attempts per 5 minutes)"""
    return APIRateLimiter(security_manager, limit=5, window=300)


def create_registration_rate_limiter(security_manager: SecurityManager):
    """Create rate limiter for registration endpoints (3 attempts per hour)"""
    return APIRateLimiter(security_manager, limit=3, window=3600)


def create_api_rate_limiter(security_manager: SecurityManager):
    """Create rate limiter for general API endpoints (100 requests per minute)"""
    return APIRateLimiter(security_manager, limit=100, window=60)