"""
Security Middleware Module

Provides comprehensive security middleware for FastAPI applications including:
- Rate limiting with Redis backend
- DDoS protection and request filtering
- CORS configuration for production
- Input validation and sanitization
- SQL injection and XSS protection
- Content Security Policy (CSP) headers
- Security headers and HTTPS enforcement
"""

from .security_middleware import SecurityMiddleware
from .rate_limit_middleware import RateLimitMiddleware, RateLimitRule
from .ddos_protection_middleware import DDoSProtectionMiddleware, ThreatAlert
from .cors_middleware import CORSMiddleware
from .input_validation_middleware import InputValidationMiddleware, InputValidationRule
from .security_headers_middleware import SecurityHeadersMiddleware
from .request_logging_middleware import RequestLoggingMiddleware
from .middleware_integration import (
    SecurityMiddlewareStack,
    create_production_security_config,
    create_development_security_config
)

__all__ = [
    "SecurityMiddleware",
    "RateLimitMiddleware",
    "RateLimitRule", 
    "DDoSProtectionMiddleware",
    "ThreatAlert",
    "CORSMiddleware",
    "InputValidationMiddleware",
    "InputValidationRule",
    "SecurityHeadersMiddleware",
    "RequestLoggingMiddleware",
    "SecurityMiddlewareStack",
    "create_production_security_config",
    "create_development_security_config",
]