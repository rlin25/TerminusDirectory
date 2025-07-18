"""
Security Middleware

Main security middleware that orchestrates all security components including
authentication, authorization, rate limiting, and security monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..auth.models import SecurityContext, SecurityEvent, SecurityEventType, ThreatLevel
from ..auth.authentication import AuthenticationManager
from ..auth.authorization import AuthorizationManager
from ..auth.jwt_manager import JWTManager


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Main Security Middleware that provides:
    - JWT token validation
    - Session management
    - Security context creation
    - Request/response security logging
    - Performance monitoring
    - Error handling
    """
    
    def __init__(
        self,
        app,
        jwt_manager: JWTManager,
        auth_manager: AuthenticationManager,
        authorization_manager: AuthorizationManager,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.jwt_manager = jwt_manager
        self.auth_manager = auth_manager
        self.authorization_manager = authorization_manager
        self.config = config or {}
        
        # Security configuration
        self.require_auth_paths = self.config.get("require_auth_paths", [
            "/api/v1/users/profile",
            "/api/v1/recommendations",
            "/api/v1/properties/favorites"
        ])
        self.public_paths = self.config.get("public_paths", [
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/search/properties"
        ])
        self.admin_paths = self.config.get("admin_paths", [
            "/api/v1/admin",
            "/api/v1/system"
        ])
        
        # Performance monitoring
        self.enable_performance_monitoring = self.config.get("enable_performance_monitoring", True)
        self.slow_request_threshold_ms = self.config.get("slow_request_threshold_ms", 1000)
        
        # Security statistics
        self._security_stats = {
            "total_requests": 0,
            "authenticated_requests": 0,
            "unauthenticated_requests": 0,
            "authorization_failures": 0,
            "token_validation_failures": 0,
            "slow_requests": 0,
            "error_responses": 0
        }
        
        # HTTP Bearer for token extraction
        self.bearer_scheme = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        
        try:
            self._security_stats["total_requests"] += 1
            
            # Extract client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            request_id = self._generate_request_id()
            
            # Add request ID to request state
            request.state.request_id = request_id
            request.state.client_ip = client_ip
            request.state.start_time = start_time
            
            # Log incoming request
            self.logger.info(
                f"Request {request_id}: {request.method} {request.url.path} from {client_ip}"
            )
            
            # Check if path requires authentication
            requires_auth = self._requires_authentication(request.url.path)
            
            # Process authentication if required
            security_context = None
            if requires_auth:
                auth_result = await self._process_authentication(request)
                if not auth_result["success"]:
                    return JSONResponse(
                        status_code=401,
                        content={"error": auth_result["error"], "request_id": request_id}
                    )
                security_context = auth_result["security_context"]
                self._security_stats["authenticated_requests"] += 1
            else:
                # Try to extract security context even for public endpoints
                auth_result = await self._process_authentication(request, required=False)
                if auth_result["success"]:
                    security_context = auth_result["security_context"]
                    self._security_stats["authenticated_requests"] += 1
                else:
                    self._security_stats["unauthenticated_requests"] += 1
            
            # Add security context to request state
            request.state.security_context = security_context
            
            # Check authorization for protected paths
            if requires_auth and security_context:
                auth_check = await self._check_authorization(request, security_context)
                if not auth_check["allowed"]:
                    self._security_stats["authorization_failures"] += 1
                    return JSONResponse(
                        status_code=403,
                        content={"error": auth_check["reason"], "request_id": request_id}
                    )
            
            # Process the request
            response = await call_next(request)
            
            # Add security headers to response
            self._add_security_headers(response)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Calculate request duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log slow requests
            if self.enable_performance_monitoring and duration_ms > self.slow_request_threshold_ms:
                self._security_stats["slow_requests"] += 1
                self.logger.warning(
                    f"Slow request {request_id}: {request.method} {request.url.path} "
                    f"took {duration_ms:.2f}ms"
                )
            
            # Log response
            self.logger.info(
                f"Response {request_id}: {response.status_code} "
                f"({duration_ms:.2f}ms)"
            )
            
            # Track error responses
            if response.status_code >= 400:
                self._security_stats["error_responses"] += 1
            
            return response
            
        except HTTPException as e:
            # Handle HTTP exceptions
            duration_ms = (time.time() - start_time) * 1000
            self._security_stats["error_responses"] += 1
            
            self.logger.warning(
                f"HTTP exception {request_id}: {e.status_code} - {e.detail} "
                f"({duration_ms:.2f}ms)"
            )
            
            response = JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "request_id": request_id}
            )
            self._add_security_headers(response)
            return response
            
        except Exception as e:
            # Handle unexpected exceptions
            duration_ms = (time.time() - start_time) * 1000
            self._security_stats["error_responses"] += 1
            
            self.logger.error(
                f"Unexpected error {request_id}: {str(e)} ({duration_ms:.2f}ms)",
                exc_info=True
            )
            
            # Log security event for unexpected errors
            await self._log_security_event(
                SecurityEventType.SECURITY_VIOLATION,
                getattr(request.state, 'security_context', None),
                client_ip,
                user_agent,
                f"Unexpected middleware error: {str(e)}",
                ThreatLevel.MEDIUM
            )
            
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id
                }
            )
            self._add_security_headers(response)
            return response
    
    async def _process_authentication(
        self,
        request: Request,
        required: bool = True
    ) -> Dict[str, Any]:
        """Process authentication for the request"""
        try:
            # Try to get token from Authorization header
            token = None
            authorization = request.headers.get("authorization")
            
            if authorization and authorization.startswith("Bearer "):
                token = authorization[7:]  # Remove "Bearer " prefix
            
            # Try to get token from query parameter (for websockets, etc.)
            if not token:
                token = request.query_params.get("token")
            
            # Try to get API key
            api_key = request.headers.get("x-api-key")
            
            if not token and not api_key:
                if required:
                    return {
                        "success": False,
                        "error": "Authentication required"
                    }
                else:
                    return {"success": False}
            
            # Verify token
            if token:
                token_payload = self.jwt_manager.verify_token(token)
                if not token_payload:
                    self._security_stats["token_validation_failures"] += 1
                    return {
                        "success": False,
                        "error": "Invalid or expired token"
                    }
                
                # Create security context from token
                security_context = self.jwt_manager.create_security_context(token_payload)
                
                return {
                    "success": True,
                    "security_context": security_context
                }
            
            # Verify API key
            if api_key:
                client_ip = self._get_client_ip(request)
                api_result = await self.auth_manager.authenticate_with_api_key(
                    api_key, client_ip, request.headers.get("user-agent", "")
                )
                
                if not api_result.success:
                    return {
                        "success": False,
                        "error": api_result.error_message
                    }
                
                return {
                    "success": True,
                    "security_context": api_result.security_context
                }
            
            return {
                "success": False,
                "error": "No valid authentication method provided"
            }
            
        except Exception as e:
            self.logger.error(f"Authentication processing error: {e}")
            return {
                "success": False,
                "error": "Authentication service error"
            }
    
    async def _check_authorization(
        self,
        request: Request,
        security_context: SecurityContext
    ) -> Dict[str, Any]:
        """Check authorization for the request"""
        try:
            path = request.url.path
            method = request.method
            
            # Check admin paths
            if any(path.startswith(admin_path) for admin_path in self.admin_paths):
                from ..auth.models import UserRole
                if not security_context.has_role(UserRole.ADMIN) and not security_context.has_role(UserRole.SUPER_ADMIN):
                    return {
                        "allowed": False,
                        "reason": "Admin access required"
                    }
            
            # Additional authorization checks can be added here
            # For example, resource-specific permissions
            
            return {"allowed": True}
            
        except Exception as e:
            self.logger.error(f"Authorization check error: {e}")
            return {
                "allowed": False,
                "reason": "Authorization service error"
            }
    
    def _requires_authentication(self, path: str) -> bool:
        """Check if path requires authentication"""
        # Check if path is explicitly public
        for public_path in self.public_paths:
            if path.startswith(public_path):
                return False
        
        # Check if path explicitly requires auth
        for auth_path in self.require_auth_paths:
            if path.startswith(auth_path):
                return True
        
        # Default behavior - require auth for /api/ paths except public ones
        if path.startswith("/api/"):
            return True
        
        return False
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS (HTTPS Strict Transport Security)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # Server information hiding
        response.headers["Server"] = "Rental-ML-System"
    
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        security_context: Optional[SecurityContext],
        ip_address: str,
        user_agent: str,
        message: str,
        threat_level: ThreatLevel = ThreatLevel.LOW
    ):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            user_id=security_context.user_id if security_context else None,
            username=security_context.username if security_context else None,
            ip_address=ip_address,
            user_agent=user_agent,
            message=message,
            threat_level=threat_level
        )
        
        # In production, send to security monitoring system
        self.logger.info(f"Security event: {event.to_dict()}")
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security middleware statistics"""
        return {
            "statistics": self._security_stats.copy(),
            "configuration": {
                "require_auth_paths": self.require_auth_paths,
                "public_paths": self.public_paths,
                "admin_paths": self.admin_paths,
                "enable_performance_monitoring": self.enable_performance_monitoring,
                "slow_request_threshold_ms": self.slow_request_threshold_ms
            }
        }
    
    async def setup_middleware_integration(self, rate_limiter=None, ddos_protection=None, input_validator=None):
        """Setup integration with other middleware components"""
        self.rate_limiter = rate_limiter
        self.ddos_protection = ddos_protection
        self.input_validator = input_validator
        
        self.logger.info("Security middleware integration configured")
    
    def is_suspicious_request(self, request: Request) -> bool:
        """Check if request is flagged as suspicious by other middleware"""
        # Check if request has been flagged by other middleware
        return getattr(request.state, 'is_suspicious', False)