"""
Security Middleware Integration
 
Demonstrates how to set up and configure all security middleware components
to work together as a cohesive security layer.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI

from .security_middleware import SecurityMiddleware
from .rate_limit_middleware import RateLimitMiddleware
from .ddos_protection_middleware import DDoSProtectionMiddleware
from .input_validation_middleware import InputValidationMiddleware
from .security_headers_middleware import SecurityHeadersMiddleware
from .cors_middleware import CORSMiddleware
from .request_logging_middleware import RequestLoggingMiddleware

from ..auth.jwt_manager import JWTManager
from ..auth.authentication import AuthenticationManager
from ..auth.authorization import AuthorizationManager


class SecurityMiddlewareStack:
    """
    Complete security middleware stack that integrates all security components
    """
    
    def __init__(
        self,
        app: FastAPI,
        jwt_manager: JWTManager,
        auth_manager: AuthenticationManager,
        authorization_manager: AuthorizationManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self.app = app
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize middleware components
        self.input_validator = InputValidationMiddleware(
            app,
            config=self.config.get("input_validation", {})
        )
        
        self.rate_limiter = RateLimitMiddleware(
            app,
            redis_url=self.config.get("redis_url"),
            config=self.config.get("rate_limiting", {})
        )
        
        self.ddos_protection = DDoSProtectionMiddleware(
            app,
            config=self.config.get("ddos_protection", {})
        )
        
        self.security_middleware = SecurityMiddleware(
            app,
            jwt_manager,
            auth_manager,
            authorization_manager,
            config=self.config.get("security", {})
        )
        
        self.security_headers = SecurityHeadersMiddleware(
            app,
            config=self.config.get("security_headers", {})
        )
        
        self.cors_middleware = CORSMiddleware(
            app,
            config=self.config.get("cors", {})
        )
        
        self.request_logging = RequestLoggingMiddleware(
            app,
            config=self.config.get("request_logging", {})
        )
    
    def setup_middleware_stack(self):
        """
        Set up the complete middleware stack in the correct order.
        
        Order is important:
        1. Request Logging (first to capture all requests)
        2. CORS (handle preflight requests early)
        3. Security Headers (add headers before processing)
        4. DDoS Protection (block malicious traffic early)
        5. Rate Limiting (limit request rates)
        6. Input Validation (validate and sanitize input)
        7. Security Middleware (authentication and authorization)
        """
        
        # Add middleware in reverse order (FastAPI applies them in reverse)
        self.app.add_middleware(SecurityMiddleware, **self._get_security_middleware_kwargs())
        self.app.add_middleware(InputValidationMiddleware, config=self.config.get("input_validation", {}))
        self.app.add_middleware(
            RateLimitMiddleware,
            redis_url=self.config.get("redis_url"),
            config=self.config.get("rate_limiting", {})
        )
        self.app.add_middleware(DDoSProtectionMiddleware, config=self.config.get("ddos_protection", {}))
        self.app.add_middleware(SecurityHeadersMiddleware, config=self.config.get("security_headers", {}))
        self.app.add_middleware(CORSMiddleware, config=self.config.get("cors", {}))
        self.app.add_middleware(RequestLoggingMiddleware, config=self.config.get("request_logging", {}))
        
        self.logger.info("Security middleware stack configured successfully")
    
    def _get_security_middleware_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for security middleware"""
        return {
            "jwt_manager": self.security_middleware.jwt_manager,
            "auth_manager": self.security_middleware.auth_manager,
            "authorization_manager": self.security_middleware.authorization_manager,
            "config": self.config.get("security", {})
        }
    
    async def setup_integrations(self):
        """Set up cross-middleware integrations"""
        await self.security_middleware.setup_middleware_integration(
            rate_limiter=self.rate_limiter,
            ddos_protection=self.ddos_protection,
            input_validator=self.input_validator
        )
    
    def get_middleware_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all middleware components"""
        return {
            "security": self.security_middleware.get_security_statistics(),
            "rate_limiting": self.rate_limiter.get_rate_limit_statistics(),
            "ddos_protection": self.ddos_protection.get_ddos_statistics(),
            "input_validation": self.input_validator.get_validation_statistics(),
        }
    
    async def emergency_lockdown(self, duration_minutes: int = 30):
        """Emergency lockdown - temporarily restrict all non-essential access"""
        self.logger.critical(f"EMERGENCY LOCKDOWN ACTIVATED for {duration_minutes} minutes")
        
        # Implement emergency restrictions
        # This could include:
        # - Blocking all non-admin traffic
        # - Reducing rate limits significantly
        # - Enabling strict validation mode
        # - Alerting security team
        
        emergency_config = {
            "rate_limiting": {
                "global_emergency_limit": 10,  # 10 requests per minute
                "admin_only_paths": ["/api/v1/admin/", "/api/v1/emergency/"]
            },
            "ddos_protection": {
                "detection_sensitivity": "maximum",
                "auto_mitigation": True,
                "max_requests_per_second": 5
            },
            "input_validation": {
                "strict_mode": True,
                "max_request_size": 1024 * 1024  # 1MB
            }
        }
        
        # Apply emergency configuration
        # In a real implementation, this would update the middleware configs
        
    async def security_health_check(self) -> Dict[str, Any]:
        """Comprehensive security health check"""
        health_status = {
            "overall": "healthy",
            "components": {},
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # Check each middleware component
            stats = self.get_middleware_statistics()
            
            # Rate limiting health
            rate_limit_stats = stats.get("rate_limiting", {}).get("statistics", {})
            if rate_limit_stats.get("rate_limited_requests", 0) > 100:
                health_status["alerts"].append("High rate limiting activity detected")
            
            # DDoS protection health
            ddos_stats = stats.get("ddos_protection", {}).get("statistics", {})
            if ddos_stats.get("threats_detected", 0) > 0:
                health_status["alerts"].append("DDoS threats detected")
            
            # Input validation health
            validation_stats = stats.get("input_validation", {}).get("statistics", {})
            if validation_stats.get("validation_failures", 0) > 50:
                health_status["alerts"].append("High input validation failure rate")
            
            # Security middleware health
            security_stats = stats.get("security", {}).get("statistics", {})
            if security_stats.get("authorization_failures", 0) > 20:
                health_status["alerts"].append("High authorization failure rate")
            
            # Overall health assessment
            if len(health_status["alerts"]) == 0:
                health_status["overall"] = "healthy"
            elif len(health_status["alerts"]) <= 2:
                health_status["overall"] = "warning"
            else:
                health_status["overall"] = "critical"
            
            health_status["components"] = {
                "rate_limiting": "healthy" if rate_limit_stats else "unknown",
                "ddos_protection": "healthy" if ddos_stats else "unknown",
                "input_validation": "healthy" if validation_stats else "unknown",
                "security": "healthy" if security_stats else "unknown"
            }
            
        except Exception as e:
            health_status["overall"] = "error"
            health_status["alerts"].append(f"Health check error: {str(e)}")
        
        return health_status


def create_production_security_config() -> Dict[str, Any]:
    """Create production-ready security configuration"""
    return {
        "redis_url": "redis://localhost:6379/0",
        "input_validation": {
            "enable_validation": True,
            "strict_mode": False,
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "max_json_depth": 10,
            "allowed_content_types": [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data"
            ],
            "max_file_size": 50 * 1024 * 1024,  # 50MB
            "allowed_file_extensions": [
                ".jpg", ".jpeg", ".png", ".gif", ".pdf", 
                ".doc", ".docx", ".txt", ".csv"
            ]
        },
        "rate_limiting": {
            "enable_headers": True,
            "default_rate_limit": {"requests": 100, "window": 60},
            "custom_rules": [
                {
                    "name": "api_burst_protection",
                    "requests": 20,
                    "window_seconds": 1,
                    "paths": ["/api/v1/"]
                }
            ]
        },
        "ddos_protection": {
            "enable_protection": True,
            "detection_sensitivity": "medium",
            "auto_mitigation": True,
            "max_requests_per_second": 100,
            "max_requests_per_minute": 1000,
            "max_error_rate": 0.5,
            "max_requests_per_ip_per_minute": 60,
            "suspicious_score_threshold": 0.7,
            "monitoring_enabled": True
        },
        "security": {
            "require_auth_paths": [
                "/api/v1/users/profile",
                "/api/v1/recommendations",
                "/api/v1/properties/favorites"
            ],
            "public_paths": [
                "/",
                "/health",
                "/docs",
                "/openapi.json",
                "/api/v1/auth/login",
                "/api/v1/auth/register",
                "/api/v1/search/properties"
            ],
            "admin_paths": [
                "/api/v1/admin",
                "/api/v1/system"
            ],
            "enable_performance_monitoring": True,
            "slow_request_threshold_ms": 1000
        },
        "security_headers": {
            "enable_hsts": True,
            "enable_csp": True,
            "frame_options": "DENY",
            "content_type_options": "nosniff"
        },
        "cors": {
            "allow_origins": ["https://yourdomain.com"],
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["*"],
            "allow_credentials": True
        },
        "request_logging": {
            "log_level": "INFO",
            "log_requests": True,
            "log_responses": True,
            "log_headers": False,  # Set to True for debugging
            "exclude_paths": ["/health", "/metrics"]
        }
    }


def create_development_security_config() -> Dict[str, Any]:
    """Create development-friendly security configuration"""
    config = create_production_security_config()
    
    # Relax some restrictions for development
    config["rate_limiting"]["default_rate_limit"] = {"requests": 1000, "window": 60}
    config["ddos_protection"]["detection_sensitivity"] = "low"
    config["input_validation"]["strict_mode"] = False
    config["cors"]["allow_origins"] = ["*"]  # Allow all origins in dev
    config["request_logging"]["log_headers"] = True  # More verbose logging
    
    return config