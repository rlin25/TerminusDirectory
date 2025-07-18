"""
Security Headers Middleware

Comprehensive security headers middleware that adds various security-related
HTTP headers to protect against common web vulnerabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security Headers Middleware that adds:
    - Content Security Policy (CSP)
    - HTTP Strict Transport Security (HSTS)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    - X-Permitted-Cross-Domain-Policies
    - Clear-Site-Data (for logout endpoints)
    """
    
    def __init__(
        self,
        app,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Security headers configuration
        self.enable_csp = self.config.get("enable_csp", True)
        self.enable_hsts = self.config.get("enable_hsts", True)
        self.enable_frame_options = self.config.get("enable_frame_options", True)
        self.enable_content_type_options = self.config.get("enable_content_type_options", True)
        self.enable_xss_protection = self.config.get("enable_xss_protection", True)
        self.enable_referrer_policy = self.config.get("enable_referrer_policy", True)
        self.enable_permissions_policy = self.config.get("enable_permissions_policy", True)
        
        # Environment detection
        self.environment = self.config.get("environment", "production")
        self.is_https = self.config.get("is_https", True)
        
        # CSP configuration
        self.csp_config = self._build_csp_config()
        
        # HSTS configuration
        self.hsts_max_age = self.config.get("hsts_max_age", 31536000)  # 1 year
        self.hsts_include_subdomains = self.config.get("hsts_include_subdomains", True)
        self.hsts_preload = self.config.get("hsts_preload", True)
        
        # Frame options
        self.frame_options = self.config.get("frame_options", "DENY")
        
        # Referrer policy
        self.referrer_policy = self.config.get("referrer_policy", "strict-origin-when-cross-origin")
        
        # Permissions policy
        self.permissions_policy = self._build_permissions_policy()
        
        # Clear site data configuration
        self.clear_site_data_paths = set(self.config.get("clear_site_data_paths", [
            "/api/v1/auth/logout",
            "/logout"
        ]))
        
        # Server header
        self.server_header = self.config.get("server_header", "Rental-ML-System")
        self.hide_server_version = self.config.get("hide_server_version", True)
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "csp_headers_added": 0,
            "hsts_headers_added": 0,
            "clear_site_data_triggered": 0,
            "csp_violations_reported": 0
        }
    
    def _build_csp_config(self) -> Dict[str, Any]:
        """Build Content Security Policy configuration"""
        default_csp = {
            "default-src": ["'self'"],
            "script-src": ["'self'"],
            "style-src": ["'self'", "'unsafe-inline'"],
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'"],
            "connect-src": ["'self'"],
            "media-src": ["'self'"],
            "object-src": ["'none'"],
            "frame-src": ["'none'"],
            "worker-src": ["'self'"],
            "frame-ancestors": ["'none'"],
            "form-action": ["'self'"],
            "base-uri": ["'self'"],
            "manifest-src": ["'self'"],
            "upgrade-insecure-requests": []
        }
        
        # Override with custom CSP configuration
        custom_csp = self.config.get("csp", {})
        for directive, values in custom_csp.items():
            default_csp[directive] = values
        
        # Development environment adjustments
        if self.environment == "development":
            default_csp["script-src"].extend(["'unsafe-eval'", "'unsafe-inline'"])
            default_csp["connect-src"].extend([
                "ws://localhost:*",
                "wss://localhost:*",
                "http://localhost:*",
                "https://localhost:*"
            ])
        
        return default_csp
    
    def _build_permissions_policy(self) -> str:
        """Build Permissions Policy header"""
        default_policies = {
            "accelerometer": "(),",
            "autoplay": "(),",
            "camera": "(),",
            "cross-origin-isolated": "(),",
            "display-capture": "(),",
            "document-domain": "(),",
            "encrypted-media": "(),",
            "fullscreen": "(),",
            "geolocation": "(),",
            "gyroscope": "(),",
            "magnetometer": "(),",
            "microphone": "(),",
            "midi": "(),",
            "payment": "(),",
            "picture-in-picture": "(),",
            "publickey-credentials-get": "(),",
            "screen-wake-lock": "(),",
            "sync-xhr": "(),",
            "usb": "(),",
            "web-share": "(),",
            "xr-spatial-tracking": "()"
        }
        
        # Override with custom policies
        custom_policies = self.config.get("permissions_policy", {})
        default_policies.update(custom_policies)
        
        # Build policy string
        policy_parts = []
        for feature, allowlist in default_policies.items():
            if allowlist.endswith(','):
                policy_parts.append(f"{feature}={allowlist}")
            else:
                policy_parts.append(f"{feature}={allowlist}")
        
        return ", ".join(policy_parts)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method"""
        self._stats["total_requests"] += 1
        
        # Process the request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(request, response)
        
        return response
    
    def _add_security_headers(self, request: Request, response: Response):
        """Add all security headers to response"""
        
        # Content Security Policy
        if self.enable_csp:
            self._add_csp_header(response)
        
        # HTTP Strict Transport Security
        if self.enable_hsts and self.is_https:
            self._add_hsts_header(response)
        
        # X-Frame-Options
        if self.enable_frame_options:
            self._add_frame_options_header(response)
        
        # X-Content-Type-Options
        if self.enable_content_type_options:
            self._add_content_type_options_header(response)
        
        # X-XSS-Protection
        if self.enable_xss_protection:
            self._add_xss_protection_header(response)
        
        # Referrer-Policy
        if self.enable_referrer_policy:
            self._add_referrer_policy_header(response)
        
        # Permissions-Policy
        if self.enable_permissions_policy:
            self._add_permissions_policy_header(response)
        
        # Additional security headers
        self._add_additional_security_headers(response)
        
        # Clear-Site-Data for logout endpoints
        if self._should_clear_site_data(request):
            self._add_clear_site_data_header(response)
        
        # Server header
        self._add_server_header(response)
    
    def _add_csp_header(self, response: Response):
        """Add Content Security Policy header"""
        csp_parts = []
        
        for directive, values in self.csp_config.items():
            if values:
                csp_parts.append(f"{directive} {' '.join(values)}")
            else:
                csp_parts.append(directive)
        
        csp_header = "; ".join(csp_parts)
        response.headers["Content-Security-Policy"] = csp_header
        
        # Also add report-only header for testing
        if self.config.get("csp_report_only", False):
            response.headers["Content-Security-Policy-Report-Only"] = csp_header
        
        self._stats["csp_headers_added"] += 1
    
    def _add_hsts_header(self, response: Response):
        """Add HTTP Strict Transport Security header"""
        hsts_parts = [f"max-age={self.hsts_max_age}"]
        
        if self.hsts_include_subdomains:
            hsts_parts.append("includeSubDomains")
        
        if self.hsts_preload:
            hsts_parts.append("preload")
        
        response.headers["Strict-Transport-Security"] = "; ".join(hsts_parts)
        self._stats["hsts_headers_added"] += 1
    
    def _add_frame_options_header(self, response: Response):
        """Add X-Frame-Options header"""
        response.headers["X-Frame-Options"] = self.frame_options
    
    def _add_content_type_options_header(self, response: Response):
        """Add X-Content-Type-Options header"""
        response.headers["X-Content-Type-Options"] = "nosniff"
    
    def _add_xss_protection_header(self, response: Response):
        """Add X-XSS-Protection header"""
        response.headers["X-XSS-Protection"] = "1; mode=block"
    
    def _add_referrer_policy_header(self, response: Response):
        """Add Referrer-Policy header"""
        response.headers["Referrer-Policy"] = self.referrer_policy
    
    def _add_permissions_policy_header(self, response: Response):
        """Add Permissions-Policy header"""
        response.headers["Permissions-Policy"] = self.permissions_policy
    
    def _add_additional_security_headers(self, response: Response):
        """Add additional security headers"""
        # X-Permitted-Cross-Domain-Policies
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # X-DNS-Prefetch-Control
        response.headers["X-DNS-Prefetch-Control"] = "off"
        
        # X-Download-Options
        response.headers["X-Download-Options"] = "noopen"
        
        # Cross-Origin-Embedder-Policy
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        
        # Cross-Origin-Opener-Policy
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        
        # Cross-Origin-Resource-Policy
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    
    def _should_clear_site_data(self, request: Request) -> bool:
        """Check if Clear-Site-Data header should be added"""
        path = request.url.path
        return any(path.startswith(clear_path) for clear_path in self.clear_site_data_paths)
    
    def _add_clear_site_data_header(self, response: Response):
        """Add Clear-Site-Data header"""
        # Clear various types of data on logout
        clear_directive = '"cache", "cookies", "storage", "executionContexts"'
        response.headers["Clear-Site-Data"] = clear_directive
        self._stats["clear_site_data_triggered"] += 1
    
    def _add_server_header(self, response: Response):
        """Add or modify Server header"""
        if self.hide_server_version:
            response.headers["Server"] = self.server_header
        else:
            # Remove existing server header if present
            if "Server" in response.headers:
                del response.headers["Server"]
    
    def update_csp_directive(self, directive: str, values: List[str]):
        """Update CSP directive"""
        self.csp_config[directive] = values
        self.logger.info(f"Updated CSP directive '{directive}': {values}")
    
    def add_csp_source(self, directive: str, source: str):
        """Add source to CSP directive"""
        if directive not in self.csp_config:
            self.csp_config[directive] = []
        
        if source not in self.csp_config[directive]:
            self.csp_config[directive].append(source)
            self.logger.info(f"Added source '{source}' to CSP directive '{directive}'")
    
    def remove_csp_source(self, directive: str, source: str):
        """Remove source from CSP directive"""
        if directive in self.csp_config and source in self.csp_config[directive]:
            self.csp_config[directive].remove(source)
            self.logger.info(f"Removed source '{source}' from CSP directive '{directive}'")
    
    def add_clear_site_data_path(self, path: str):
        """Add path that should trigger Clear-Site-Data header"""
        self.clear_site_data_paths.add(path)
        self.logger.info(f"Added Clear-Site-Data path: {path}")
    
    def remove_clear_site_data_path(self, path: str):
        """Remove path from Clear-Site-Data triggers"""
        self.clear_site_data_paths.discard(path)
        self.logger.info(f"Removed Clear-Site-Data path: {path}")
    
    def get_security_headers_statistics(self) -> Dict[str, Any]:
        """Get security headers statistics"""
        return {
            "statistics": dict(self._stats),
            "configuration": {
                "enable_csp": self.enable_csp,
                "enable_hsts": self.enable_hsts,
                "enable_frame_options": self.enable_frame_options,
                "enable_content_type_options": self.enable_content_type_options,
                "enable_xss_protection": self.enable_xss_protection,
                "enable_referrer_policy": self.enable_referrer_policy,
                "enable_permissions_policy": self.enable_permissions_policy,
                "environment": self.environment,
                "is_https": self.is_https,
                "hsts_max_age": self.hsts_max_age,
                "frame_options": self.frame_options,
                "referrer_policy": self.referrer_policy
            },
            "csp_directives": len(self.csp_config),
            "clear_site_data_paths": len(self.clear_site_data_paths)
        }