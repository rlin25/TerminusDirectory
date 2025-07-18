"""
Authentication and Authorization Module

Provides comprehensive authentication and authorization services including:
- JWT-based authentication with refresh tokens
- OAuth2/OpenID Connect integration
- Multi-factor authentication (MFA)
- Role-based access control (RBAC)
- API key management
- Session management
"""

from .authentication import AuthenticationManager
from .authorization import AuthorizationManager, RBACManager
from .jwt_manager import JWTManager
from .oauth2_manager import OAuth2Manager
from .mfa_manager import MFAManager
from .api_key_manager import APIKeyManager
from .session_manager import SessionManager
from .models import *

__all__ = [
    "AuthenticationManager",
    "AuthorizationManager", 
    "RBACManager",
    "JWTManager",
    "OAuth2Manager",
    "MFAManager",
    "APIKeyManager",
    "SessionManager",
    "UserRole",
    "Permission",
    "SecurityContext",
    "AuthenticationResult",
    "AuthorizationResult",
]