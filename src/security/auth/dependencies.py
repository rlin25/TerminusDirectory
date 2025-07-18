"""
FastAPI Authentication Dependencies

Provides FastAPI dependencies for JWT authentication, authorization, and security context.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import SecurityContext, UserRole, Permission, AuthenticationResult
from .jwt_manager import JWTManager
from .authentication import AuthenticationManager
from .authorization import AuthorizationManager
from ..config import get_security_config

logger = logging.getLogger(__name__)

# Security schemes
security_scheme = HTTPBearer(auto_error=False)
api_key_scheme = HTTPBearer(scheme_name="API Key", auto_error=False)


class SecurityDependencies:
    """Security dependencies for FastAPI"""
    
    def __init__(
        self,
        jwt_manager: JWTManager,
        auth_manager: AuthenticationManager,
        authorization_manager: AuthorizationManager
    ):
        self.jwt_manager = jwt_manager
        self.auth_manager = auth_manager
        self.authorization_manager = authorization_manager
        self.config = get_security_config()
        
    async def get_current_user(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
    ) -> SecurityContext:
        """Get current authenticated user (required authentication)"""
        # Check if security context was set by middleware
        if hasattr(request.state, 'security_context') and request.state.security_context:
            return request.state.security_context
        
        # Fallback to manual authentication
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return await self._authenticate_token(credentials.credentials, request)
    
    async def get_current_user_optional(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
    ) -> Optional[SecurityContext]:
        """Get current authenticated user (optional authentication)"""
        # Check if security context was set by middleware
        if hasattr(request.state, 'security_context') and request.state.security_context:
            return request.state.security_context
        
        # Try to authenticate if token is provided
        if credentials:
            try:
                return await self._authenticate_token(credentials.credentials, request)
            except HTTPException:
                return None
        
        return None
    
    async def _authenticate_token(self, token: str, request: Request) -> SecurityContext:
        """Authenticate JWT token and return security context"""
        try:
            # Verify JWT token
            payload = self.jwt_manager.verify_token(token)
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Create security context
            security_context = self.jwt_manager.create_security_context(payload)
            
            # Check if token has expired
            if security_context.is_expired():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Store in request state for other middleware
            request.state.security_context = security_context
            
            return security_context
            
        except Exception as e:
            logger.error(f"Token authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def require_permissions(self, *permissions: Permission):
        """Dependency factory for permission-based authorization"""
        async def permission_checker(
            current_user: SecurityContext = Depends(self.get_current_user)
        ) -> SecurityContext:
            """Check if user has required permissions"""
            missing_permissions = []
            for permission in permissions:
                if not current_user.has_permission(permission):
                    missing_permissions.append(permission.value)
            
            if missing_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permissions: {', '.join(missing_permissions)}"
                )
            
            return current_user
        
        return permission_checker
    
    def require_roles(self, *roles: UserRole):
        """Dependency factory for role-based authorization"""
        async def role_checker(
            current_user: SecurityContext = Depends(self.get_current_user)
        ) -> SecurityContext:
            """Check if user has required roles"""
            if not any(current_user.has_role(role) for role in roles):
                role_names = [role.value for role in roles]
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required roles: {', '.join(role_names)}"
                )
            
            return current_user
        
        return role_checker
    
    def require_admin(self):
        """Dependency for admin-only endpoints"""
        return self.require_roles(UserRole.ADMIN, UserRole.SUPER_ADMIN)
    
    def require_property_manager(self):
        """Dependency for property manager endpoints"""
        return self.require_roles(
            UserRole.PROPERTY_MANAGER, UserRole.ADMIN, UserRole.SUPER_ADMIN
        )
    
    def require_mfa(self):
        """Dependency that requires MFA verification"""
        async def mfa_checker(
            current_user: SecurityContext = Depends(self.get_current_user)
        ) -> SecurityContext:
            """Check if user has completed MFA verification"""
            if not current_user.mfa_verified:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Multi-factor authentication required"
                )
            
            return current_user
        
        return mfa_checker
    
    def require_api_key(self):
        """Dependency for API key authentication"""
        async def api_key_checker(
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(api_key_scheme)
        ) -> SecurityContext:
            """Authenticate using API key"""
            api_key = None
            
            # Try to get API key from header
            if credentials:
                api_key = credentials.credentials
            else:
                # Try to get from X-API-Key header
                api_key = request.headers.get("x-api-key")
            
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Authenticate with API key
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            
            result = await self.auth_manager.authenticate_with_api_key(
                api_key, client_ip, user_agent
            )
            
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=result.error_message or "Invalid API key"
                )
            
            return result.security_context
        
        return api_key_checker
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"


# Global security dependencies instance (will be initialized in main app)
security_dependencies: Optional[SecurityDependencies] = None


def get_security_dependencies() -> SecurityDependencies:
    """Get global security dependencies instance"""
    if security_dependencies is None:
        raise RuntimeError("Security dependencies not initialized")
    return security_dependencies


def init_security_dependencies(
    jwt_manager: JWTManager,
    auth_manager: AuthenticationManager,
    authorization_manager: AuthorizationManager
):
    """Initialize global security dependencies"""
    global security_dependencies
    security_dependencies = SecurityDependencies(
        jwt_manager, auth_manager, authorization_manager
    )


# Convenience dependencies for common use cases
async def get_current_user(request: Request) -> SecurityContext:
    """Get current authenticated user"""
    deps = get_security_dependencies()
    return await deps.get_current_user(request)


async def get_current_user_optional(request: Request) -> Optional[SecurityContext]:
    """Get current authenticated user (optional)"""
    deps = get_security_dependencies()
    return await deps.get_current_user_optional(request)


def require_permissions(*permissions: Permission):
    """Require specific permissions"""
    deps = get_security_dependencies()
    return deps.require_permissions(*permissions)


def require_roles(*roles: UserRole):
    """Require specific roles"""
    deps = get_security_dependencies()
    return deps.require_roles(*roles)


def require_admin():
    """Require admin role"""
    deps = get_security_dependencies()
    return deps.require_admin()


def require_property_manager():
    """Require property manager role"""
    deps = get_security_dependencies()
    return deps.require_property_manager()


def require_mfa():
    """Require MFA verification"""
    deps = get_security_dependencies()
    return deps.require_mfa()


def require_api_key():
    """Require API key authentication"""
    deps = get_security_dependencies()
    return deps.require_api_key()