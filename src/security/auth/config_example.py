"""
Security Configuration Examples

Complete configuration examples for all security components
including production-ready settings and integration examples.
"""

import os
from typing import Dict, Any


class SecurityConfig:
    """Comprehensive security configuration"""
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Production security configuration"""
        return {
            # Database configuration
            "database": {
                "enabled": True,
                "url": os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/rental_ml"),
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 3600
            },
            
            # Redis configuration
            "redis": {
                "enabled": True,
                "url": os.getenv("REDIS_URL", "redis://localhost:6379"),
                "socket_timeout": 5,
                "socket_connect_timeout": 5,
                "retry_on_timeout": True,
                "health_check_interval": 30
            },
            
            # JWT configuration
            "jwt": {
                "algorithm": "RS256",
                "issuer": "rental-ml-system",
                "audience": "rental-ml-api",
                "access_token_expire_minutes": 30,
                "refresh_token_expire_days": 7
            },
            
            # Session configuration
            "sessions": {
                "session_timeout_minutes": 480,  # 8 hours
                "max_concurrent_sessions": 5,
                "require_secure": True
            },
            
            # MFA configuration
            "mfa": {
                "totp_issuer": "Rental ML System",
                "token_validity_minutes": 5,
                "max_attempts": 3
            },
            
            # API key configuration
            "api_keys": {
                "default_expiry_days": 365,
                "max_keys_per_user": 10
            },
            
            # OAuth2 configuration
            "oauth2": {
                "google": {
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", "https://yourdomain.com/auth/oauth2/google/callback")
                },
                "facebook": {
                    "client_id": os.getenv("FACEBOOK_CLIENT_ID"),
                    "client_secret": os.getenv("FACEBOOK_CLIENT_SECRET"),
                    "redirect_uri": os.getenv("FACEBOOK_REDIRECT_URI", "https://yourdomain.com/auth/oauth2/facebook/callback")
                },
                "apple": {
                    "client_id": os.getenv("APPLE_CLIENT_ID"),
                    "client_secret": os.getenv("APPLE_CLIENT_SECRET"),
                    "redirect_uri": os.getenv("APPLE_REDIRECT_URI", "https://yourdomain.com/auth/oauth2/apple/callback")
                },
                "microsoft": {
                    "client_id": os.getenv("MICROSOFT_CLIENT_ID"),
                    "client_secret": os.getenv("MICROSOFT_CLIENT_SECRET"),
                    "tenant_id": os.getenv("MICROSOFT_TENANT_ID", "common"),
                    "redirect_uri": os.getenv("MICROSOFT_REDIRECT_URI", "https://yourdomain.com/auth/oauth2/microsoft/callback")
                }
            },
            
            # Email service configuration
            "email": {
                "enabled": True,
                "provider": os.getenv("EMAIL_PROVIDER", "smtp"),  # smtp, sendgrid, aws_ses, mailgun
                
                # SMTP configuration
                "smtp_host": os.getenv("SMTP_HOST", "localhost"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "smtp_username": os.getenv("SMTP_USERNAME"),
                "smtp_password": os.getenv("SMTP_PASSWORD"),
                "smtp_use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
                "smtp_timeout": 30,
                
                # SendGrid configuration
                "sendgrid_api_key": os.getenv("SENDGRID_API_KEY"),
                
                # AWS SES configuration
                "aws_region": os.getenv("AWS_REGION", "us-east-1"),
                "aws_access_key": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                
                # Mailgun configuration
                "mailgun_api_key": os.getenv("MAILGUN_API_KEY"),
                "mailgun_domain": os.getenv("MAILGUN_DOMAIN"),
                
                # Common email settings
                "from_email": os.getenv("FROM_EMAIL", "noreply@yourdomain.com"),
                "from_name": os.getenv("FROM_NAME", "Rental ML System")
            },
            
            # SMS service configuration
            "sms": {
                "enabled": True,
                "provider": os.getenv("SMS_PROVIDER", "twilio"),  # twilio, aws_sns, nexmo
                
                # Twilio configuration
                "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
                "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
                "twilio_from_number": os.getenv("TWILIO_FROM_NUMBER"),
                
                # AWS SNS configuration
                "aws_region": os.getenv("AWS_REGION", "us-east-1"),
                "aws_access_key": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                
                # Nexmo configuration
                "nexmo_api_key": os.getenv("NEXMO_API_KEY"),
                "nexmo_api_secret": os.getenv("NEXMO_API_SECRET"),
                "nexmo_from_number": os.getenv("NEXMO_FROM_NUMBER", "RentalML")
            }
        }
    
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Development security configuration with relaxed settings"""
        config = SecurityConfig.get_production_config()
        
        # Relaxed settings for development
        config.update({
            "database": {
                **config["database"],
                "echo": True,  # Enable SQL logging
                "url": "postgresql+asyncpg://postgres:password@localhost/rental_ml_dev"
            },
            
            "redis": {
                **config["redis"],
                "url": "redis://localhost:6379/1"  # Use different Redis DB
            },
            
            "jwt": {
                **config["jwt"],
                "access_token_expire_minutes": 1440,  # 24 hours for development
            },
            
            "sessions": {
                **config["sessions"],
                "require_secure": False,  # Allow HTTP in development
                "session_timeout_minutes": 1440  # 24 hours
            },
            
            # Mock email/SMS for development
            "email": {
                "enabled": True,
                "provider": "mock",
                "from_email": "dev@localhost",
                "from_name": "Rental ML Dev"
            },
            
            "sms": {
                "enabled": True,
                "provider": "mock"
            }
        })
        
        return config
    
    @staticmethod
    def get_testing_config() -> Dict[str, Any]:
        """Testing configuration with minimal external dependencies"""
        return {
            "database": {
                "enabled": False  # Use in-memory storage for tests
            },
            
            "redis": {
                "enabled": False  # Use in-memory storage for tests
            },
            
            "jwt": {
                "algorithm": "RS256",
                "issuer": "test-system",
                "audience": "test-api",
                "access_token_expire_minutes": 60,
                "refresh_token_expire_days": 1
            },
            
            "sessions": {
                "session_timeout_minutes": 60,
                "max_concurrent_sessions": 10,
                "require_secure": False
            },
            
            "mfa": {
                "totp_issuer": "Test System",
                "token_validity_minutes": 10,
                "max_attempts": 5
            },
            
            "api_keys": {
                "default_expiry_days": 30,
                "max_keys_per_user": 5
            },
            
            "email": {
                "enabled": True,
                "provider": "mock"
            },
            
            "sms": {
                "enabled": True,
                "provider": "mock"
            },
            
            "oauth2": {}  # Disable OAuth2 for tests
        }


class MiddlewareConfig:
    """Middleware configuration examples"""
    
    @staticmethod
    def get_production_middleware_config() -> Dict[str, Any]:
        """Production middleware configuration"""
        return {
            "enabled": True,
            "log_requests": True,
            "check_rate_limits": True,
            "detect_suspicious_activity": True,
            
            "exempt_paths": [
                "/health",
                "/metrics",
                "/docs",
                "/openapi.json",
                "/auth/login",
                "/auth/register",
                "/auth/oauth2/authorize",
                "/auth/oauth2/callback",
                "/auth/password-reset"
            ],
            
            "path_rate_limits": {
                "/auth/login": {"limit": 5, "window": 300},
                "/auth/register": {"limit": 3, "window": 3600},
                "/auth/password-reset": {"limit": 3, "window": 3600},
                "/api/search": {"limit": 1000, "window": 60},
                "/api/recommendations": {"limit": 100, "window": 60},
                "default": {"limit": 100, "window": 60}
            }
        }
    
    @staticmethod
    def get_development_middleware_config() -> Dict[str, Any]:
        """Development middleware configuration"""
        config = MiddlewareConfig.get_production_middleware_config()
        
        # Relaxed rate limits for development
        config["path_rate_limits"] = {
            path: {"limit": limit_config["limit"] * 10, "window": limit_config["window"]}
            for path, limit_config in config["path_rate_limits"].items()
        }
        
        return config


# Environment-specific configuration loader
def get_security_config(environment: str = None) -> Dict[str, Any]:
    """Get security configuration based on environment"""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return SecurityConfig.get_production_config()
    elif environment == "testing":
        return SecurityConfig.get_testing_config()
    else:  # development
        return SecurityConfig.get_development_config()


def get_middleware_config(environment: str = None) -> Dict[str, Any]:
    """Get middleware configuration based on environment"""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return MiddlewareConfig.get_production_middleware_config()
    else:
        return MiddlewareConfig.get_development_middleware_config()


# Example environment variables file (.env)
ENV_EXAMPLE = """
# Environment
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/rental_ml

# Redis
REDIS_URL=redis://localhost:6379

# OAuth2 Providers
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/oauth2/google/callback

FACEBOOK_CLIENT_ID=your_facebook_client_id
FACEBOOK_CLIENT_SECRET=your_facebook_client_secret
FACEBOOK_REDIRECT_URI=https://yourdomain.com/auth/oauth2/facebook/callback

APPLE_CLIENT_ID=your_apple_client_id
APPLE_CLIENT_SECRET=your_apple_client_secret
APPLE_REDIRECT_URI=https://yourdomain.com/auth/oauth2/apple/callback

MICROSOFT_CLIENT_ID=your_microsoft_client_id
MICROSOFT_CLIENT_SECRET=your_microsoft_client_secret
MICROSOFT_TENANT_ID=common
MICROSOFT_REDIRECT_URI=https://yourdomain.com/auth/oauth2/microsoft/callback

# Email Service (Choose one)
EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=your_sendgrid_api_key
FROM_EMAIL=noreply@yourdomain.com
FROM_NAME=Rental ML System

# Or SMTP
# EMAIL_PROVIDER=smtp
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your_email@gmail.com
# SMTP_PASSWORD=your_app_password
# SMTP_USE_TLS=true

# SMS Service (Choose one)
SMS_PROVIDER=twilio
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_FROM_NUMBER=+1234567890

# Or AWS SNS
# SMS_PROVIDER=aws_sns
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# Or Nexmo
# SMS_PROVIDER=nexmo
# NEXMO_API_KEY=your_nexmo_api_key
# NEXMO_API_SECRET=your_nexmo_api_secret
# NEXMO_FROM_NUMBER=RentalML
"""


# Docker Compose configuration example
DOCKER_COMPOSE_EXAMPLE = """
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/rental_ml
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    env_file:
      - .env

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: rental_ml
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
"""