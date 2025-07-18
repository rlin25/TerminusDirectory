"""
Security System Usage Examples

Complete examples showing how to integrate and use the security system
in a FastAPI application with all components working together.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer

from .security_integration import SecurityManager, security_manager_context
from .enhanced_middleware import (
    SecurityMiddleware, SecurityDependency, require_permissions, require_roles,
    get_current_user, get_optional_user, create_login_rate_limiter
)
from .models import Permission, UserRole, MFAMethod
from .config_example import get_security_config, get_middleware_config


# Initialize FastAPI app
app = FastAPI(title="Rental ML API", version="1.0.0")

# Global security manager (will be initialized in startup)
security_manager: Optional[SecurityManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize security system on startup"""
    global security_manager
    
    try:
        # Get configuration
        security_config = get_security_config()
        middleware_config = get_middleware_config()
        
        # Initialize security manager
        security_manager = SecurityManager(security_config)
        await security_manager.initialize()
        
        # Add security middleware
        app.add_middleware(SecurityMiddleware, security_manager=security_manager, config=middleware_config)
        
        logging.info("Security system initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize security system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup security system on shutdown"""
    global security_manager
    
    if security_manager:
        await security_manager.shutdown()
        logging.info("Security system shutdown complete")


# Security dependency
security_dependency = SecurityDependency(security_manager)


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/auth/login")
async def login(
    request: Request,
    login_data: Dict[str, str],
    rate_limiter=Depends(lambda: create_login_rate_limiter(security_manager))
):
    """User login endpoint"""
    try:
        username = login_data.get("username")
        password = login_data.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        # Authenticate user
        auth_result = await security_manager.authenticate_password(
            username=username,
            password=password,
            ip_address=request.state.client_ip,
            user_agent=request.state.user_agent
        )
        
        if not auth_result.success:
            raise HTTPException(status_code=401, detail=auth_result.error_message)
        
        # Handle MFA requirement
        if auth_result.mfa_required:
            return {
                "success": True,
                "mfa_required": True,
                "mfa_methods": [method.value for method in auth_result.mfa_methods],
                "user_id": str(auth_result.user_id),
                "message": "Multi-factor authentication required"
            }
        
        # Return tokens
        return {
            "success": True,
            "access_token": auth_result.access_token,
            "refresh_token": auth_result.refresh_token,
            "token_type": "bearer",
            "expires_at": auth_result.expires_at.isoformat(),
            "user": {
                "id": str(auth_result.user_id),
                "username": auth_result.username
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/auth/mfa/setup")
async def setup_mfa(
    request: Request,
    mfa_data: Dict[str, str],
    current_user=Depends(get_current_user)
):
    """Set up MFA for current user"""
    try:
        method = MFAMethod(mfa_data.get("method", "totp"))
        
        if method == MFAMethod.TOTP:
            result = await security_manager.setup_mfa(
                user_id=current_user.user_id,
                method=method,
                contact_info={
                    "username": current_user.username,
                    "email": current_user.email
                }
            )
        elif method == MFAMethod.SMS:
            phone = mfa_data.get("phone")
            if not phone:
                raise HTTPException(status_code=400, detail="Phone number required for SMS MFA")
            
            result = await security_manager.setup_mfa(
                user_id=current_user.user_id,
                method=method,
                contact_info={"phone": phone}
            )
        elif method == MFAMethod.EMAIL:
            result = await security_manager.setup_mfa(
                user_id=current_user.user_id,
                method=method,
                contact_info={"email": current_user.email}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported MFA method: {method.value}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"MFA setup error: {e}")
        raise HTTPException(status_code=500, detail="MFA setup failed")


@app.post("/auth/mfa/verify")
async def verify_mfa(
    request: Request,
    verification_data: Dict[str, str]
):
    """Verify MFA code"""
    try:
        token_id = verification_data.get("token_id")
        verification_code = verification_data.get("code")
        method = verification_data.get("method")
        
        if not token_id or not verification_code:
            raise HTTPException(status_code=400, detail="Token ID and verification code required")
        
        mfa_method = MFAMethod(method) if method else None
        
        result = await security_manager.verify_mfa(token_id, verification_code, mfa_method)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Complete authentication after successful MFA
        # In a real implementation, you'd retrieve the pending authentication session
        # and complete it with token generation
        
        return {
            "success": True,
            "message": "MFA verification successful",
            "user_id": result.get("user_id")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"MFA verification error: {e}")
        raise HTTPException(status_code=500, detail="MFA verification failed")


@app.post("/auth/refresh")
async def refresh_tokens(refresh_data: Dict[str, str]):
    """Refresh access token"""
    try:
        refresh_token = refresh_data.get("refresh_token")
        if not refresh_token:
            raise HTTPException(status_code=400, detail="Refresh token required")
        
        result = await security_manager.refresh_tokens(refresh_token)
        
        if not result:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        access_token, new_refresh_token, expiry = result
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_at": expiry.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@app.post("/auth/logout")
async def logout(
    request: Request,
    logout_data: Optional[Dict[str, str]] = None,
    current_user=Depends(get_current_user)
):
    """User logout"""
    try:
        # Get tokens to revoke
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            access_token = auth_header[7:]
            await security_manager.revoke_token(access_token, "logout")
        
        # Revoke refresh token if provided
        if logout_data and logout_data.get("refresh_token"):
            await security_manager.revoke_token(logout_data["refresh_token"], "logout")
        
        # Invalidate session
        if current_user.session_id:
            await security_manager.invalidate_session(current_user.session_id, current_user.user_id)
        
        return {"success": True, "message": "Logged out successfully"}
        
    except Exception as e:
        logging.error(f"Logout error: {e}")
        return {"success": True, "message": "Logged out successfully"}  # Always succeed


# ============================================================================
# OAuth2 Endpoints
# ============================================================================

@app.get("/auth/oauth2/{provider}/authorize")
async def oauth2_authorize(provider: str, state: Optional[str] = None):
    """Initiate OAuth2 authorization"""
    try:
        auth_url_data = security_manager.oauth2_manager.get_authorization_url(
            provider_name=provider,
            state=state
        )
        
        return RedirectResponse(url=auth_url_data["authorization_url"])
        
    except Exception as e:
        logging.error(f"OAuth2 authorization error: {e}")
        raise HTTPException(status_code=400, detail=f"OAuth2 authorization failed: {str(e)}")


@app.get("/auth/oauth2/{provider}/callback")
async def oauth2_callback(
    provider: str,
    code: str,
    state: str,
    request: Request
):
    """Handle OAuth2 callback"""
    try:
        auth_result = await security_manager.authenticate_oauth2_callback(
            provider=provider,
            code=code,
            state=state,
            ip_address=request.state.client_ip,
            user_agent=request.state.user_agent
        )
        
        if not auth_result.success:
            raise HTTPException(status_code=401, detail=auth_result.error_message)
        
        return {
            "success": True,
            "access_token": auth_result.access_token,
            "refresh_token": auth_result.refresh_token,
            "token_type": "bearer",
            "expires_at": auth_result.expires_at.isoformat(),
            "user": {
                "id": str(auth_result.user_id),
                "username": auth_result.username
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"OAuth2 callback error: {e}")
        raise HTTPException(status_code=500, detail="OAuth2 authentication failed")


# ============================================================================
# API Key Management Endpoints
# ============================================================================

@app.post("/auth/api-keys")
@require_permissions(Permission.MANAGE_SYSTEM)
async def create_api_key(
    key_data: Dict[str, str],
    current_user=Depends(get_current_user)
):
    """Create new API key"""
    try:
        result = await security_manager.api_key_manager.create_api_key(
            name=key_data["name"],
            user_id=current_user.user_id,
            permissions=[Permission(p) for p in key_data.get("permissions", [])],
            rate_limit=key_data.get("rate_limit", 100),
            ip_whitelist=key_data.get("ip_whitelist", []),
            expires_in_days=key_data.get("expires_in_days"),
            created_by=current_user.username
        )
        
        return result
        
    except Exception as e:
        logging.error(f"API key creation error: {e}")
        raise HTTPException(status_code=500, detail="API key creation failed")


@app.get("/auth/api-keys")
@require_permissions(Permission.MANAGE_SYSTEM)
async def list_api_keys(current_user=Depends(get_current_user)):
    """List user's API keys"""
    try:
        keys = await security_manager.api_key_manager.list_api_keys(
            user_id=current_user.user_id
        )
        return {"api_keys": keys}
        
    except Exception as e:
        logging.error(f"API key listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list API keys")


@app.delete("/auth/api-keys/{key_id}")
@require_permissions(Permission.MANAGE_SYSTEM)
async def revoke_api_key(
    key_id: str,
    current_user=Depends(get_current_user)
):
    """Revoke API key"""
    try:
        result = await security_manager.api_key_manager.revoke_api_key(
            key_id=key_id,
            revoked_by=current_user.username
        )
        
        return result
        
    except Exception as e:
        logging.error(f"API key revocation error: {e}")
        raise HTTPException(status_code=500, detail="API key revocation failed")


# ============================================================================
# Protected API Endpoints
# ============================================================================

@app.get("/api/properties")
@require_permissions(Permission.READ_PROPERTY)
async def get_properties(
    limit: int = 10,
    offset: int = 0,
    current_user=Depends(get_current_user)
):
    """Get properties (requires read permission)"""
    # Mock implementation
    return {
        "properties": [
            {"id": i, "title": f"Property {i}", "price": 1000 + i * 100}
            for i in range(offset, offset + limit)
        ],
        "total": 1000,
        "user": current_user.username
    }


@app.post("/api/properties")
@require_permissions(Permission.CREATE_PROPERTY)
async def create_property(
    property_data: Dict[str, str],
    current_user=Depends(get_current_user)
):
    """Create property (requires create permission)"""
    # Mock implementation
    return {
        "success": True,
        "property_id": "new_property_123",
        "created_by": current_user.username
    }


@app.get("/api/admin/users")
@require_roles(UserRole.ADMIN, UserRole.SUPER_ADMIN)
async def get_users(current_user=Depends(get_current_user)):
    """Get users (admin only)"""
    # Mock implementation
    return {
        "users": [
            {"id": "1", "username": "user1", "email": "user1@example.com"},
            {"id": "2", "username": "user2", "email": "user2@example.com"}
        ],
        "requested_by": current_user.username
    }


@app.get("/api/analytics")
@require_permissions(Permission.VIEW_ANALYTICS)
async def get_analytics(current_user=Depends(get_current_user)):
    """Get analytics (requires analytics permission)"""
    # Mock implementation
    return {
        "total_properties": 1000,
        "total_users": 500,
        "monthly_revenue": 50000,
        "requested_by": current_user.username
    }


# ============================================================================
# Public Endpoints (No Authentication Required)
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if security_manager:
            health_status = await security_manager.health_check()
            return health_status
        else:
            return {"status": "healthy", "note": "Security manager not initialized"}
            
    except Exception as e:
        logging.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/api/public/properties")
async def get_public_properties(
    limit: int = 10,
    offset: int = 0,
    current_user=Depends(get_optional_user)
):
    """Get public properties (no authentication required)"""
    # Mock implementation with optional user context
    user_info = {"username": current_user.username} if current_user else {"username": "anonymous"}
    
    return {
        "properties": [
            {"id": i, "title": f"Public Property {i}", "price": 1000 + i * 100}
            for i in range(offset, offset + limit)
        ],
        "total": 1000,
        "requested_by": user_info["username"]
    }


# ============================================================================
# Security Management Endpoints
# ============================================================================

@app.get("/admin/security/statistics")
@require_permissions(Permission.MANAGE_SECURITY)
async def get_security_statistics(current_user=Depends(get_current_user)):
    """Get security statistics"""
    try:
        stats = security_manager.get_security_statistics()
        return stats
        
    except Exception as e:
        logging.error(f"Security statistics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security statistics")


@app.get("/admin/security/sessions/{user_id}")
@require_permissions(Permission.MANAGE_SECURITY)
async def get_user_sessions(user_id: str, current_user=Depends(get_current_user)):
    """Get user sessions"""
    try:
        sessions = await security_manager.session_manager.get_user_sessions(UUID(user_id))
        return {"sessions": sessions}
        
    except Exception as e:
        logging.error(f"Get user sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user sessions")


@app.delete("/admin/security/sessions/{user_id}")
@require_permissions(Permission.MANAGE_SECURITY)
async def invalidate_user_sessions(
    user_id: str,
    current_user=Depends(get_current_user)
):
    """Invalidate all user sessions"""
    try:
        count = await security_manager.session_manager.invalidate_user_sessions(UUID(user_id))
        return {"success": True, "invalidated_sessions": count}
        
    except Exception as e:
        logging.error(f"Invalidate user sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate user sessions")


# ============================================================================
# Example Usage Functions
# ============================================================================

async def example_security_operations():
    """Example of using security operations programmatically"""
    
    # Initialize security manager
    config = get_security_config("development")
    
    async with security_manager_context(config) as sm:
        
        # 1. Create API key
        api_key_result = await sm.api_key_manager.create_api_key(
            name="Test API Key",
            service_name="test_service",
            permissions=[Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES],
            rate_limit=1000
        )
        print(f"Created API key: {api_key_result}")
        
        # 2. Authenticate with API key
        if api_key_result["success"]:
            auth_result = await sm.authenticate_api_key(
                api_key_result["api_key"],
                "192.168.1.1"
            )
            print(f"API key authentication: {auth_result.success}")
        
        # 3. Check authorization
        if auth_result.success:
            authz_result = await sm.authorize_request(
                auth_result.security_context,
                [Permission.READ_PROPERTY],
                "properties",
                "read"
            )
            print(f"Authorization result: {authz_result.allowed}")
        
        # 4. Rate limiting
        rate_limit_result = await sm.check_rate_limit(
            "192.168.1.1",
            limit=100,
            window_seconds=60
        )
        print(f"Rate limit check: {rate_limit_result}")
        
        # 5. Security statistics
        stats = sm.get_security_statistics()
        print(f"Security statistics: {stats}")


# Example of running the security operations
if __name__ == "__main__":
    asyncio.run(example_security_operations())