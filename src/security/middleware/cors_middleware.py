"""
CORS Middleware

Advanced Cross-Origin Resource Sharing (CORS) middleware with security-focused
configuration and origin validation for production environments.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Advanced CORS Middleware with:
    - Strict origin validation
    - Dynamic origin configuration
    - Preflight request handling
    - Credential support control
    - Request method filtering
    - Header validation
    - Security logging
    """
    
    def __init__(
        self,
        app,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # CORS configuration
        self.allowed_origins = self._parse_allowed_origins()
        self.allowed_methods = set(self.config.get("allowed_methods", [
            "GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"
        ]))
        self.allowed_headers = set(self.config.get("allowed_headers", [
            "Accept", "Accept-Language", "Content-Language", "Content-Type",
            "Authorization", "X-Requested-With", "X-API-Key", "X-Client-Version"
        ]))
        self.exposed_headers = set(self.config.get("exposed_headers", [
            "X-Total-Count", "X-Page-Count", "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset", "X-Request-ID"
        ]))
        
        # Security settings
        self.allow_credentials = self.config.get("allow_credentials", True)
        self.max_age = self.config.get("max_age", 600)  # 10 minutes
        self.strict_origin_validation = self.config.get("strict_origin_validation", True)
        self.log_cors_violations = self.config.get("log_cors_violations", True)
        
        # Development vs production settings
        self.environment = self.config.get("environment", "production")
        self.allow_all_origins = self.config.get("allow_all_origins", False)
        
        # Regex patterns for dynamic origin matching
        self.origin_patterns = self._compile_origin_patterns()
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "preflight_requests": 0,
            "cors_violations": 0,
            "allowed_origins": 0,
            "blocked_origins": 0,
            "credential_requests": 0
        }
    
    def _parse_allowed_origins(self) -> Set[str]:
        """Parse and validate allowed origins"""
        origins = set()
        
        # Get origins from config
        config_origins = self.config.get("allowed_origins", [])
        
        # Default origins for development
        if self.environment == "development":
            default_origins = [
                "http://localhost:3000",
                "http://localhost:3001",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000"
            ]
            config_origins.extend(default_origins)
        
        # Process each origin
        for origin in config_origins:
            if isinstance(origin, str):
                # Validate origin format
                if self._validate_origin_format(origin):
                    origins.add(origin.rstrip('/'))
                else:
                    self.logger.warning(f"Invalid origin format: {origin}")
        
        return origins
    
    def _validate_origin_format(self, origin: str) -> bool:
        """Validate origin format"""
        if origin == "*":
            return True
        
        try:
            parsed = urlparse(origin)
            # Must have scheme and netloc
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def _compile_origin_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for dynamic origin matching"""
        patterns = []
        
        pattern_strings = self.config.get("origin_patterns", [])
        
        # Add default patterns for subdomains if configured
        if self.config.get("allow_subdomains"):
            base_domains = self.config.get("base_domains", [])
            for domain in base_domains:
                # Allow any subdomain of the base domain
                pattern = f"^https?://([a-zA-Z0-9-]+\\.)*{re.escape(domain)}$"
                pattern_strings.append(pattern)
        
        # Compile patterns
        for pattern_str in pattern_strings:
            try:
                patterns.append(re.compile(pattern_str))
            except re.error as e:
                self.logger.error(f"Invalid origin pattern '{pattern_str}': {e}")
        
        return patterns
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method"""
        self._stats["total_requests"] += 1
        
        # Check if this is a preflight request
        if self._is_preflight_request(request):
            self._stats["preflight_requests"] += 1
            return await self._handle_preflight_request(request)
        
        # Process the actual request
        response = await call_next(request)
        
        # Add CORS headers to response
        self._add_cors_headers(request, response)
        
        return response
    
    def _is_preflight_request(self, request: Request) -> bool:
        """Check if request is a CORS preflight request"""
        return (
            request.method == "OPTIONS" and
            "access-control-request-method" in request.headers
        )
    
    async def _handle_preflight_request(self, request: Request) -> Response:
        """Handle CORS preflight request"""
        origin = request.headers.get("origin", "")
        
        # Validate origin
        if not self._is_origin_allowed(origin):
            self._stats["cors_violations"] += 1
            if self.log_cors_violations:
                self.logger.warning(f"CORS violation: Origin '{origin}' not allowed")
            
            return StarletteResponse(
                content="CORS origin not allowed",
                status_code=403
            )
        
        # Validate requested method
        requested_method = request.headers.get("access-control-request-method", "")
        if not self._is_method_allowed(requested_method):
            self._stats["cors_violations"] += 1
            if self.log_cors_violations:
                self.logger.warning(f"CORS violation: Method '{requested_method}' not allowed")
            
            return StarletteResponse(
                content="CORS method not allowed",
                status_code=405
            )
        
        # Validate requested headers
        requested_headers = self._parse_requested_headers(request)
        if not self._are_headers_allowed(requested_headers):
            self._stats["cors_violations"] += 1
            if self.log_cors_violations:
                self.logger.warning(f"CORS violation: Headers not allowed: {requested_headers}")
            
            return StarletteResponse(
                content="CORS headers not allowed",
                status_code=403
            )
        
        # Create preflight response
        response = StarletteResponse(status_code=200)
        
        # Add CORS headers
        self._add_preflight_headers(request, response, origin)
        
        self._stats["allowed_origins"] += 1
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if not origin:
            return False
        
        # Allow all origins if configured (development only)
        if self.allow_all_origins and self.environment == "development":
            return True
        
        # Check explicit allowed origins
        if origin in self.allowed_origins:
            return True
        
        # Check wildcard
        if "*" in self.allowed_origins:
            return True
        
        # Check regex patterns
        for pattern in self.origin_patterns:
            if pattern.match(origin):
                return True
        
        return False
    
    def _is_method_allowed(self, method: str) -> bool:
        """Check if HTTP method is allowed"""
        return method.upper() in self.allowed_methods
    
    def _parse_requested_headers(self, request: Request) -> Set[str]:
        """Parse requested headers from preflight request"""
        headers_str = request.headers.get("access-control-request-headers", "")
        if not headers_str:
            return set()
        
        # Split by comma and normalize
        headers = set()
        for header in headers_str.split(","):
            header = header.strip().lower()
            if header:
                headers.add(header)
        
        return headers
    
    def _are_headers_allowed(self, requested_headers: Set[str]) -> bool:
        """Check if all requested headers are allowed"""
        # Normalize allowed headers to lowercase
        allowed_headers_lower = {h.lower() for h in self.allowed_headers}
        
        # Check if all requested headers are allowed
        return requested_headers.issubset(allowed_headers_lower)
    
    def _add_cors_headers(self, request: Request, response: Response):
        """Add CORS headers to response"""
        origin = request.headers.get("origin", "")
        
        # Only add CORS headers if origin is allowed
        if not self._is_origin_allowed(origin):
            self._stats["blocked_origins"] += 1
            return
        
        # Set allowed origin
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            self._stats["allowed_origins"] += 1
        
        # Set credentials
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
            
            # Track credential requests
            if request.headers.get("authorization") or request.cookies:
                self._stats["credential_requests"] += 1
        
        # Set exposed headers
        if self.exposed_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.exposed_headers)
        
        # Add Vary header for proper caching
        vary_header = response.headers.get("Vary", "")
        if vary_header:
            if "Origin" not in vary_header:
                response.headers["Vary"] = f"{vary_header}, Origin"
        else:
            response.headers["Vary"] = "Origin"
    
    def _add_preflight_headers(self, request: Request, response: Response, origin: str):
        """Add headers for preflight response"""
        # Set allowed origin
        response.headers["Access-Control-Allow-Origin"] = origin
        
        # Set allowed methods
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        
        # Set allowed headers
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        
        # Set credentials
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Set max age
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
        
        # Add Vary header
        response.headers["Vary"] = "Origin, Access-Control-Request-Method, Access-Control-Request-Headers"
    
    def add_allowed_origin(self, origin: str):
        """Add an allowed origin dynamically"""
        if self._validate_origin_format(origin):
            self.allowed_origins.add(origin.rstrip('/'))
            self.logger.info(f"Added allowed origin: {origin}")
        else:
            self.logger.warning(f"Invalid origin format, not added: {origin}")
    
    def remove_allowed_origin(self, origin: str):
        """Remove an allowed origin"""
        self.allowed_origins.discard(origin.rstrip('/'))
        self.logger.info(f"Removed allowed origin: {origin}")
    
    def add_allowed_method(self, method: str):
        """Add an allowed HTTP method"""
        self.allowed_methods.add(method.upper())
        self.logger.info(f"Added allowed method: {method}")
    
    def remove_allowed_method(self, method: str):
        """Remove an allowed HTTP method"""
        self.allowed_methods.discard(method.upper())
        self.logger.info(f"Removed allowed method: {method}")
    
    def add_allowed_header(self, header: str):
        """Add an allowed header"""
        self.allowed_headers.add(header)
        self.logger.info(f"Added allowed header: {header}")
    
    def remove_allowed_header(self, header: str):
        """Remove an allowed header"""
        self.allowed_headers.discard(header)
        self.logger.info(f"Removed allowed header: {header}")
    
    def get_cors_statistics(self) -> Dict[str, Any]:
        """Get CORS statistics"""
        return {
            "statistics": dict(self._stats),
            "configuration": {
                "allowed_origins": len(self.allowed_origins),
                "allowed_methods": list(self.allowed_methods),
                "allowed_headers": list(self.allowed_headers),
                "exposed_headers": list(self.exposed_headers),
                "allow_credentials": self.allow_credentials,
                "max_age": self.max_age,
                "strict_origin_validation": self.strict_origin_validation,
                "environment": self.environment,
                "allow_all_origins": self.allow_all_origins
            },
            "origin_patterns": len(self.origin_patterns)
        }