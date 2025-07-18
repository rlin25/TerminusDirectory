"""
Authentication Router

Provides authentication endpoints for login, registration, token refresh, and logout.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, validator

from ....security.auth.dependencies import (
    get_current_user, get_current_user_optional, require_permissions
)
from ....security.auth.models import (
    SecurityContext, UserRole, Permission, AuthenticationResult, MFAMethod
)
from ....security.auth.rate_limiter import RateLimitType
from ....security.config import get_security_config

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class LoginRequest(BaseModel):
    identifier: str  # username or email
    password: str
    remember_me: bool = False

    @validator('identifier')
    def validate_identifier(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Username or email is required")
        return v.strip()

    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class LoginResponse(BaseModel):
    success: bool
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    expires_at: Optional[datetime] = None
    mfa_required: bool = False
    mfa_methods: Optional[list] = None
    mfa_token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    agree_to_terms: bool = True

    @validator('username')
    def validate_username(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Username must be at least 3 characters long")
        return v.strip().lower()

    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v

    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError("Passwords do not match")
        return v

    @validator('agree_to_terms')
    def validate_terms(cls, v):
        if not v:
            raise ValueError("You must agree to the terms and conditions")
        return v


class RegisterResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    success: bool
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    expires_at: Optional[datetime] = None
    message: Optional[str] = None


class MFAVerificationRequest(BaseModel):
    mfa_token: str
    mfa_code: str
    method: MFAMethod = MFAMethod.TOTP


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str

    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError("Passwords do not match")
        return v


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str
    confirm_password: str

    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError("Passwords do not match")
        return v


class UserProfileResponse(BaseModel):
    user_id: str
    username: str
    email: str
    roles: list
    permissions: list
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    mfa_enabled: bool = False
    account_active: bool = True


# Router setup
router = APIRouter(prefix="/auth", tags=["Authentication"])
security_scheme = HTTPBearer(auto_error=False)

# Global variables for dependency injection (will be set by main app)
auth_manager = None
rate_limiter = None


def get_client_ip(request: Request) -> str:
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


async def check_rate_limit(request: Request, limit_type: RateLimitType):
    """Check rate limit for request"""
    if not rate_limiter:
        return  # Skip if rate limiter not configured
    
    client_ip = get_client_ip(request)
    identifier = f"{client_ip}:{request.url.path}"
    
    rate_limit_result = rate_limiter.check_rate_limit(identifier, limit_type, client_ip)
    
    if not rate_limit_result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=rate_limit_result.reason,
            headers={"Retry-After": str(rate_limit_result.retry_after or 60)}
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest
) -> LoginResponse:
    """Authenticate user and return access token"""
    try:
        # Rate limiting
        await check_rate_limit(request, RateLimitType.LOGIN)
        
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Authenticate user
        result = await auth_manager.authenticate_user(
            identifier=login_data.identifier,
            password=login_data.password,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not result.success:
            return LoginResponse(
                success=False,
                message=result.error_message or "Authentication failed"
            )
        
        # Handle MFA requirement
        if result.mfa_required:
            return LoginResponse(
                success=True,
                mfa_required=True,
                mfa_methods=[method.value for method in result.mfa_methods],
                mfa_token=result.access_token,
                message="Multi-factor authentication required"
            )
        
        # Successful authentication
        config = get_security_config()
        expires_in = config.settings.jwt_access_token_expire_minutes * 60
        
        user_info = {
            "user_id": result.user_id,
            "username": result.username,
            "roles": [role.value for role in result.security_context.roles] if result.security_context else [],
            "permissions": [perm.value for perm in result.security_context.permissions] if result.security_context else []
        }
        
        return LoginResponse(
            success=True,
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            expires_in=expires_in,
            expires_at=result.expires_at,
            user=user_info,
            message="Authentication successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@router.post("/register", response_model=RegisterResponse)
async def register(
    request: Request,
    register_data: RegisterRequest
) -> RegisterResponse:
    """Register new user account"""
    try:
        # Rate limiting
        await check_rate_limit(request, RateLimitType.REGISTRATION)
        
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        
        # Create user account
        result = await auth_manager.create_user_account(
            email=register_data.email,
            password=register_data.password,
            username=register_data.username,
            ip_address=client_ip
        )
        
        if not result["success"]:
            return RegisterResponse(
                success=False,
                message=result["error"]
            )
        
        return RegisterResponse(
            success=True,
            message="Account created successfully",
            user_id=result["user_id"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration service error"
        )


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    request: Request,
    refresh_data: RefreshTokenRequest
) -> RefreshTokenResponse:
    """Refresh access token using refresh token"""
    try:
        # Rate limiting
        await check_rate_limit(request, RateLimitType.TOKEN_REFRESH)
        
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Refresh token
        result = await auth_manager.refresh_token(
            refresh_token=refresh_data.refresh_token,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not result.success:
            return RefreshTokenResponse(
                success=False,
                message=result.error_message or "Token refresh failed"
            )
        
        # Calculate expires_in
        config = get_security_config()
        expires_in = config.settings.jwt_access_token_expire_minutes * 60
        
        return RefreshTokenResponse(
            success=True,
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            expires_in=expires_in,
            expires_at=result.expires_at,
            message="Token refreshed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh service error"
        )


@router.post("/mfa/verify", response_model=LoginResponse)
async def verify_mfa(
    request: Request,
    mfa_data: MFAVerificationRequest
) -> LoginResponse:
    """Verify MFA code and complete authentication"""
    try:
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Verify MFA
        result = await auth_manager.verify_mfa_and_complete_login(
            mfa_token=mfa_data.mfa_token,
            mfa_code=mfa_data.mfa_code,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not result.success:
            return LoginResponse(
                success=False,
                message=result.error_message or "MFA verification failed"
            )
        
        # Successful authentication
        config = get_security_config()
        expires_in = config.settings.jwt_access_token_expire_minutes * 60
        
        user_info = {
            "user_id": result.user_id,
            "username": result.username,
            "roles": [role.value for role in result.security_context.roles] if result.security_context else [],
            "permissions": [perm.value for perm in result.security_context.permissions] if result.security_context else []
        }
        
        return LoginResponse(
            success=True,
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            expires_in=expires_in,
            expires_at=result.expires_at,
            user=user_info,
            message="MFA verification successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification service error"
        )


@router.post("/logout")
async def logout(
    request: Request,
    current_user: SecurityContext = Depends(get_current_user)
) -> Dict[str, Any]:
    """Logout user and invalidate session"""
    try:
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Get access token from Authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required"
            )
        
        access_token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Logout user
        result = await auth_manager.logout_user(
            access_token=access_token,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not result["success"]:
            return {"success": False, "message": result["error"]}
        
        return {"success": True, "message": "Logged out successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service error"
        )


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: SecurityContext = Depends(get_current_user)
) -> UserProfileResponse:
    """Get current user profile"""
    try:
        return UserProfileResponse(
            user_id=str(current_user.user_id),
            username=current_user.username,
            email=current_user.email,
            roles=[role.value for role in current_user.roles],
            permissions=[perm.value for perm in current_user.permissions],
            created_at=current_user.created_at,
            mfa_enabled=current_user.mfa_verified,
            account_active=True
        )
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile service error"
        )


@router.post("/password/change")
async def change_password(
    request: Request,
    password_data: PasswordChangeRequest,
    current_user: SecurityContext = Depends(get_current_user)
) -> Dict[str, Any]:
    """Change user password"""
    try:
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        
        # Change password
        result = await auth_manager.change_password(
            user_id=str(current_user.user_id),
            current_password=password_data.current_password,
            new_password=password_data.new_password,
            ip_address=client_ip
        )
        
        if not result["success"]:
            return {"success": False, "message": result["error"]}
        
        return {"success": True, "message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change service error"
        )


@router.post("/password/reset")
async def request_password_reset(
    request: Request,
    reset_data: PasswordResetRequest
) -> Dict[str, Any]:
    """Request password reset"""
    try:
        # Rate limiting
        await check_rate_limit(request, RateLimitType.PASSWORD_RESET)
        
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Request password reset
        result = await auth_manager.initiate_password_reset(
            email=reset_data.email,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not result["success"]:
            return {"success": False, "message": result["error"]}
        
        return {"success": True, "message": result["message"]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset service error"
        )


@router.post("/password/reset/confirm")
async def confirm_password_reset(
    request: Request,
    reset_data: PasswordResetConfirmRequest
) -> Dict[str, Any]:
    """Confirm password reset with token"""
    try:
        if not auth_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        # Get client information
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Confirm password reset
        result = await auth_manager.reset_password(
            reset_token=reset_data.token,
            new_password=reset_data.new_password,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        if not result["success"]:
            return {"success": False, "message": result["error"]}
        
        return {"success": True, "message": result["message"]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset confirm error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset service error"
        )


@router.get("/validate")
async def validate_token(
    current_user: SecurityContext = Depends(get_current_user)
) -> Dict[str, Any]:
    """Validate current token"""
    try:
        return {
            "valid": True,
            "user_id": str(current_user.user_id),
            "username": current_user.username,
            "roles": [role.value for role in current_user.roles],
            "permissions": [perm.value for perm in current_user.permissions],
            "expires_at": current_user.expires_at.isoformat() if current_user.expires_at else None
        }
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token validation service error"
        )


# Initialize authentication manager (will be called by main app)
def init_auth_router(authentication_manager, rate_limiter_instance):
    """Initialize authentication router with dependencies"""
    global auth_manager, rate_limiter
    auth_manager = authentication_manager
    rate_limiter = rate_limiter_instance
    logger.info("Authentication router initialized")