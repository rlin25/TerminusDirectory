"""
Security Configuration Module

Centralized configuration for all security components including JWT, middleware,
rate limiting, and authentication settings.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import secrets


@dataclass
class SecuritySettings:
    """Main security configuration settings"""
    
    # JWT Configuration
    jwt_secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(64)))
    jwt_algorithm: str = "RS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    jwt_issuer: str = "rental-ml-system"
    jwt_audience: str = "rental-ml-api"
    
    # Password Policy
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digits: bool = True
    password_require_special: bool = True
    password_max_age_days: int = 90
    password_history_count: int = 12
    password_forbidden_patterns: List[str] = field(default_factory=lambda: [
        "password", "123456", "qwerty", "admin", "user", "rental", "system"
    ])
    
    # Session Configuration
    session_timeout_minutes: int = 480  # 8 hours
    session_max_concurrent: int = 5
    session_require_secure: bool = True
    
    # Rate Limiting
    enable_rate_limiting: bool = True
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    default_rate_limit_requests: int = 100
    default_rate_limit_window_seconds: int = 60
    login_rate_limit_requests: int = 5
    login_rate_limit_window_seconds: int = 900  # 15 minutes
    api_rate_limit_requests: int = 1000
    api_rate_limit_window_seconds: int = 3600  # 1 hour
    
    # Account Lockout
    max_failed_login_attempts: int = 5
    account_lockout_duration_minutes: int = 30
    progressive_lockout_enabled: bool = True
    progressive_delays: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    # Input Validation
    enable_input_validation: bool = True
    max_request_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_json_depth: int = 10
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".doc", ".docx", ".txt", ".csv"
    ])
    allowed_content_types: List[str] = field(default_factory=lambda: [
        "application/json", "application/x-www-form-urlencoded",
        "multipart/form-data", "text/plain", "application/xml", "text/xml"
    ])
    
    # CORS Configuration
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])  # Configure for production
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])
    cors_expose_headers: List[str] = field(default_factory=lambda: [
        "X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"
    ])
    
    # Security Headers
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000  # 1 year
    content_security_policy: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    
    # API Key Configuration
    api_key_default_expiry_days: int = 365
    api_key_max_per_user: int = 10
    api_key_require_ip_whitelist: bool = False
    
    # MFA Configuration
    mfa_totp_issuer: str = "Rental ML System"
    mfa_token_validity_minutes: int = 5
    mfa_max_attempts: int = 3
    mfa_enabled_by_default: bool = False
    
    # Monitoring and Logging
    enable_security_monitoring: bool = True
    enable_audit_logging: bool = True
    security_event_retention_days: int = 90
    suspicious_activity_threshold: int = 100  # events per hour
    
    # Database Configuration
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Email Configuration (for password reset, etc.)
    smtp_server: Optional[str] = os.getenv("SMTP_SERVER")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: Optional[str] = os.getenv("SMTP_USERNAME")
    smtp_password: Optional[str] = os.getenv("SMTP_PASSWORD")
    from_email: str = os.getenv("FROM_EMAIL", "noreply@rental-ml.com")
    
    # Environment Settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Public and Protected Paths
    public_paths: List[str] = field(default_factory=lambda: [
        "/", "/health", "/docs", "/redoc", "/openapi.json",
        "/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/refresh",
        "/api/v1/search/properties"
    ])
    
    protected_paths: List[str] = field(default_factory=lambda: [
        "/api/v1/users/profile", "/api/v1/recommendations",
        "/api/v1/properties/favorites", "/api/v1/users/interactions"
    ])
    
    admin_paths: List[str] = field(default_factory=lambda: [
        "/api/v1/admin", "/api/v1/system", "/api/v1/analytics/admin"
    ])
    
    # IP Whitelist/Blacklist
    ip_whitelist: List[str] = field(default_factory=list)
    ip_blacklist: List[str] = field(default_factory=list)


class SecurityConfigManager:
    """Security configuration manager with environment-specific settings"""
    
    def __init__(self, environment: Optional[str] = None):
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._settings: Optional[SecuritySettings] = None
    
    @property
    def settings(self) -> SecuritySettings:
        """Get security settings with environment-specific overrides"""
        if self._settings is None:
            self._settings = self._load_settings()
        return self._settings
    
    def _load_settings(self) -> SecuritySettings:
        """Load settings with environment-specific configurations"""
        base_settings = SecuritySettings()
        
        # Apply environment-specific overrides
        if self.environment == "production":
            return self._apply_production_settings(base_settings)
        elif self.environment == "staging":
            return self._apply_staging_settings(base_settings)
        elif self.environment == "testing":
            return self._apply_testing_settings(base_settings)
        else:
            return self._apply_development_settings(base_settings)
    
    def _apply_production_settings(self, settings: SecuritySettings) -> SecuritySettings:
        """Apply production-specific security settings"""
        # Stricter security in production
        settings.jwt_access_token_expire_minutes = 15  # Shorter token life
        settings.session_timeout_minutes = 240  # 4 hours
        settings.cors_allow_origins = [
            "https://rental-ml.com",
            "https://app.rental-ml.com",
            "https://api.rental-ml.com"
        ]
        settings.cors_allow_credentials = True
        settings.enable_security_headers = True
        settings.enable_rate_limiting = True
        settings.enable_input_validation = True
        settings.enable_security_monitoring = True
        settings.enable_audit_logging = True
        settings.mfa_enabled_by_default = True
        settings.api_key_require_ip_whitelist = True
        settings.session_require_secure = True
        settings.debug = False
        
        # More restrictive rate limits
        settings.login_rate_limit_requests = 3
        settings.default_rate_limit_requests = 50
        
        # Stricter password policy
        settings.password_min_length = 14
        settings.password_max_age_days = 60
        
        return settings
    
    def _apply_staging_settings(self, settings: SecuritySettings) -> SecuritySettings:
        """Apply staging-specific security settings"""
        # Similar to production but slightly relaxed
        settings.jwt_access_token_expire_minutes = 20
        settings.cors_allow_origins = [
            "https://staging.rental-ml.com",
            "https://staging-app.rental-ml.com"
        ]
        settings.enable_security_headers = True
        settings.enable_rate_limiting = True
        settings.enable_input_validation = True
        settings.enable_security_monitoring = True
        settings.mfa_enabled_by_default = False
        settings.debug = False
        
        return settings
    
    def _apply_testing_settings(self, settings: SecuritySettings) -> SecuritySettings:
        """Apply testing-specific security settings"""
        # Relaxed for testing
        settings.jwt_access_token_expire_minutes = 60
        settings.enable_rate_limiting = False  # Disable for tests
        settings.enable_input_validation = True  # Keep for security tests
        settings.enable_security_monitoring = False
        settings.enable_audit_logging = False
        settings.mfa_enabled_by_default = False
        settings.max_failed_login_attempts = 10  # More lenient for tests
        settings.debug = True
        
        # Use in-memory storage for tests
        settings.redis_url = None
        
        return settings
    
    def _apply_development_settings(self, settings: SecuritySettings) -> SecuritySettings:
        """Apply development-specific security settings"""
        # Relaxed for development
        settings.jwt_access_token_expire_minutes = 60
        settings.cors_allow_origins = ["*"]
        settings.enable_security_headers = True
        settings.enable_rate_limiting = True
        settings.enable_input_validation = True
        settings.enable_security_monitoring = True
        settings.enable_audit_logging = True
        settings.mfa_enabled_by_default = False
        settings.session_require_secure = False  # Allow HTTP in dev
        settings.debug = True
        
        # More lenient rate limits for development
        settings.login_rate_limit_requests = 10
        settings.default_rate_limit_requests = 200
        
        return settings
    
    def get_jwt_config(self) -> Dict[str, Any]:
        """Get JWT-specific configuration"""
        return {
            "secret_key": self.settings.jwt_secret_key,
            "algorithm": self.settings.jwt_algorithm,
            "access_token_expire_minutes": self.settings.jwt_access_token_expire_minutes,
            "refresh_token_expire_days": self.settings.jwt_refresh_token_expire_days,
            "issuer": self.settings.jwt_issuer,
            "audience": self.settings.jwt_audience
        }
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        return {
            "enabled": self.settings.enable_rate_limiting,
            "redis_url": self.settings.redis_url,
            "default_rate_limit": {
                "requests": self.settings.default_rate_limit_requests,
                "window": self.settings.default_rate_limit_window_seconds
            },
            "login_rate_limit": {
                "requests": self.settings.login_rate_limit_requests,
                "window": self.settings.login_rate_limit_window_seconds
            },
            "api_rate_limit": {
                "requests": self.settings.api_rate_limit_requests,
                "window": self.settings.api_rate_limit_window_seconds
            },
            "ip_whitelist": self.settings.ip_whitelist,
            "ip_blacklist": self.settings.ip_blacklist
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.settings.cors_allow_origins,
            "allow_credentials": self.settings.cors_allow_credentials,
            "allow_methods": self.settings.cors_allow_methods,
            "allow_headers": self.settings.cors_allow_headers,
            "expose_headers": self.settings.cors_expose_headers
        }
    
    def get_input_validation_config(self) -> Dict[str, Any]:
        """Get input validation configuration"""
        return {
            "enable_validation": self.settings.enable_input_validation,
            "max_request_size": self.settings.max_request_size_bytes,
            "max_json_depth": self.settings.max_json_depth,
            "max_file_size": self.settings.max_file_size_bytes,
            "allowed_file_extensions": self.settings.allowed_file_extensions,
            "allowed_content_types": self.settings.allowed_content_types
        }
    
    def get_security_headers_config(self) -> Dict[str, Any]:
        """Get security headers configuration"""
        return {
            "enable_headers": self.settings.enable_security_headers,
            "hsts_max_age": self.settings.hsts_max_age,
            "content_security_policy": self.settings.content_security_policy
        }
    
    def get_auth_config(self) -> Dict[str, Any]:
        """Get authentication configuration"""
        return {
            "password_policy": {
                "min_length": self.settings.password_min_length,
                "require_uppercase": self.settings.password_require_uppercase,
                "require_lowercase": self.settings.password_require_lowercase,
                "require_digits": self.settings.password_require_digits,
                "require_special": self.settings.password_require_special,
                "max_age_days": self.settings.password_max_age_days,
                "history_count": self.settings.password_history_count,
                "forbidden_patterns": self.settings.password_forbidden_patterns
            },
            "lockout_policy": {
                "max_attempts": self.settings.max_failed_login_attempts,
                "lockout_duration_minutes": self.settings.account_lockout_duration_minutes,
                "progressive_enabled": self.settings.progressive_lockout_enabled,
                "progressive_delays": self.settings.progressive_delays
            },
            "session_policy": {
                "timeout_minutes": self.settings.session_timeout_minutes,
                "max_concurrent": self.settings.session_max_concurrent,
                "require_secure": self.settings.session_require_secure
            },
            "mfa_policy": {
                "enabled_by_default": self.settings.mfa_enabled_by_default,
                "totp_issuer": self.settings.mfa_totp_issuer,
                "token_validity_minutes": self.settings.mfa_token_validity_minutes,
                "max_attempts": self.settings.mfa_max_attempts
            },
            "email_config": {
                "smtp_server": self.settings.smtp_server,
                "smtp_port": self.settings.smtp_port,
                "smtp_username": self.settings.smtp_username,
                "smtp_password": self.settings.smtp_password,
                "from_email": self.settings.from_email
            }
        }
    
    def get_path_config(self) -> Dict[str, List[str]]:
        """Get path configuration for authentication"""
        return {
            "public_paths": self.settings.public_paths,
            "protected_paths": self.settings.protected_paths,
            "admin_paths": self.settings.admin_paths
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == "testing"


# Global security configuration instance
security_config = SecurityConfigManager()


def get_security_config() -> SecurityConfigManager:
    """Get the global security configuration instance"""
    return security_config


def reload_security_config(environment: Optional[str] = None) -> SecurityConfigManager:
    """Reload security configuration with optional environment override"""
    global security_config
    security_config = SecurityConfigManager(environment)
    return security_config