"""
Example Usage of the Comprehensive Authentication System

This file demonstrates how to use the enhanced authentication system
with all its security features and integrations.
"""

import asyncio
import logging
from typing import Dict, Any

from .authentication import AuthenticationManager
from .jwt_manager import JWTManager
from .mfa_manager import MFAManager
from .session_manager import SessionManager
from .rate_limiter import RateLimiter, RateLimitType
from .models import AuthenticationMethod, UserRole


async def example_authentication_flow():
    """Example of complete authentication flow with all security features"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=== Comprehensive Authentication System Example ===\n")
    
    # 1. Initialize the authentication system
    print("1. Initializing Authentication System...")
    
    # Configuration
    config = {
        # Password policy
        "password_min_length": 12,
        "password_require_uppercase": True,
        "password_require_lowercase": True,
        "password_require_digits": True,
        "password_require_special": True,
        
        # Account lockout
        "max_failed_attempts": 5,
        "lockout_duration_minutes": 30,
        
        # Email configuration (for password reset)
        "smtp_server": "localhost",
        "smtp_port": 587,
        "from_email": "noreply@rental-ml.com",
        
        # Rate limiting
        "rate_limits": {
            "login": {"requests": 5, "window_seconds": 60},
            "password_reset": {"requests": 3, "window_seconds": 900}
        }
    }
    
    # Initialize components
    jwt_manager = JWTManager(config)
    mfa_manager = MFAManager(config)
    session_manager = SessionManager(config)
    rate_limiter = RateLimiter(config)
    
    # Initialize authentication manager
    auth_manager = AuthenticationManager(
        jwt_manager=jwt_manager,
        mfa_manager=mfa_manager,
        session_manager=session_manager,
        user_repository=None,  # Could integrate with UserRepository
        database_manager=None,  # Could integrate with database
        config=config
    )
    
    print("✓ Authentication system initialized\n")
    
    # 2. Create a user account
    print("2. Creating User Account...")
    
    create_result = await auth_manager.create_user_account(
        email="john.doe@example.com",
        password="SecurePassword123!",
        username="johndoe",
        roles=["tenant"],
        ip_address="192.168.1.100"
    )
    
    if create_result["success"]:
        print(f"✓ User created: {create_result['username']} ({create_result['email']})")
        user_id = create_result["user_id"]
    else:
        print(f"✗ User creation failed: {create_result['error']}")
        return
    
    print()
    
    # 3. Test input validation with invalid attempts
    print("3. Testing Input Validation...")
    
    # Test invalid email
    invalid_auth = await auth_manager.authenticate_user(
        identifier="invalid-email",
        password="password",
        ip_address="192.168.1.100",
        user_agent="TestBrowser/1.0"
    )
    print(f"Invalid email test: {'✓' if not invalid_auth.success else '✗'}")
    
    # Test SQL injection attempt
    sql_injection_auth = await auth_manager.authenticate_user(
        identifier="'; DROP TABLE users; --",
        password="password",
        ip_address="192.168.1.100",
        user_agent="TestBrowser/1.0"
    )
    print(f"SQL injection protection: {'✓' if not sql_injection_auth.success else '✗'}")
    print()
    
    # 4. Test rate limiting
    print("4. Testing Rate Limiting...")
    
    # Make multiple failed login attempts
    for i in range(7):  # Exceed the limit of 5
        rate_check = rate_limiter.check_rate_limit(
            identifier="test@example.com",
            limit_type=RateLimitType.LOGIN,
            ip_address="192.168.1.100"
        )
        
        if rate_check.allowed:
            print(f"Attempt {i+1}: Allowed (remaining: {rate_check.remaining})")
        else:
            print(f"Attempt {i+1}: Blocked - {rate_check.reason}")
            break
    
    print()
    
    # 5. Successful authentication
    print("5. Successful Authentication...")
    
    auth_result = await auth_manager.authenticate_user(
        identifier="test@example.com",  # Use test user from mock data
        password="password123",
        ip_address="192.168.1.100",
        user_agent="TestBrowser/1.0",
        authentication_method=AuthenticationMethod.PASSWORD
    )
    
    if auth_result.success:
        print("✓ Authentication successful")
        print(f"  User: {auth_result.username}")
        print(f"  Access Token: {auth_result.access_token[:20]}...")
        print(f"  Expires: {auth_result.expires_at}")
        
        if auth_result.security_context:
            print(f"  Roles: {[role.value for role in auth_result.security_context.roles]}")
            print(f"  Permissions: {len(auth_result.security_context.permissions)} permissions")
    else:
        print(f"✗ Authentication failed: {auth_result.error_message}")
    
    print()
    
    # 6. Token verification and refresh
    print("6. Token Operations...")
    
    if auth_result.success and auth_result.access_token:
        # Verify token
        token_payload = jwt_manager.verify_token(auth_result.access_token)
        if token_payload:
            print("✓ Token verification successful")
            print(f"  Token subject: {token_payload.get('sub')}")
            print(f"  Token roles: {token_payload.get('roles')}")
        
        # Refresh token
        if auth_result.refresh_token:
            refresh_result = await auth_manager.refresh_token(
                refresh_token=auth_result.refresh_token,
                ip_address="192.168.1.100",
                user_agent="TestBrowser/1.0"
            )
            
            if refresh_result.success:
                print("✓ Token refresh successful")
                print(f"  New access token: {refresh_result.access_token[:20]}...")
            else:
                print(f"✗ Token refresh failed: {refresh_result.error_message}")
    
    print()
    
    # 7. Password reset flow
    print("7. Password Reset Flow...")
    
    # Initiate password reset
    reset_init = await auth_manager.initiate_password_reset(
        email="test@example.com",
        ip_address="192.168.1.100",
        user_agent="TestBrowser/1.0"
    )
    
    if reset_init["success"]:
        print("✓ Password reset initiated")
        print(f"  Message: {reset_init['message']}")
        
        # Note: In a real system, the reset token would be sent via email
        # For this example, we can't actually reset the password without the token
        
    else:
        print(f"✗ Password reset failed: {reset_init['error']}")
    
    print()
    
    # 8. User account management
    print("8. User Account Management...")
    
    # Get user security info
    security_info = await auth_manager.get_user_security_info("test-user-id")
    if security_info["success"]:
        print("✓ User security information retrieved")
        print(f"  Account active: {security_info['account_active']}")
        print(f"  MFA enabled: {security_info['mfa_enabled']}")
        print(f"  Roles: {security_info['roles']}")
        print(f"  Recent failed attempts: {security_info['recent_failed_attempts']}")
        print(f"  Is locked: {security_info['is_locked']}")
        print(f"  Active sessions: {security_info['active_sessions']}")
    
    print()
    
    # 9. Security statistics
    print("9. Security Statistics...")
    
    auth_stats = auth_manager.get_authentication_statistics()
    print("Authentication Statistics:")
    print(f"  Successful logins: {auth_stats['statistics']['successful_logins']}")
    print(f"  Failed logins: {auth_stats['statistics']['failed_logins']}")
    print(f"  Locked accounts: {auth_stats['statistics']['locked_accounts']}")
    print(f"  Password changes: {auth_stats['statistics']['password_changes']}")
    
    rate_stats = rate_limiter.get_statistics()
    print("\nRate Limiting Statistics:")
    print(f"  Total requests: {rate_stats['total_requests']}")
    print(f"  Blocked requests: {rate_stats['blocked_requests']}")
    print(f"  Block rate: {rate_stats['block_rate']:.2f}%")
    print(f"  Currently blocked: {rate_stats['currently_blocked']}")
    
    print()
    
    # 10. Cleanup
    print("10. Cleanup Operations...")
    
    # Logout user
    if auth_result.success and auth_result.access_token:
        logout_result = await auth_manager.logout_user(
            access_token=auth_result.access_token,
            ip_address="192.168.1.100",
            user_agent="TestBrowser/1.0"
        )
        
        if logout_result["success"]:
            print("✓ User logged out successfully")
        else:
            print(f"✗ Logout failed: {logout_result['error']}")
    
    # Clean up expired data
    auth_manager.cleanup_expired_data()
    rate_limiter.cleanup_expired()
    print("✓ Expired data cleaned up")
    
    print("\n=== Authentication System Example Complete ===")


async def example_api_integration():
    """Example of how to integrate with a web API framework"""
    
    print("\n=== API Integration Example ===\n")
    
    # This would typically be in your FastAPI/Flask app
    class MockRequest:
        def __init__(self, headers: Dict[str, str], client_ip: str):
            self.headers = headers
            self.client = {"host": client_ip}
    
    # Initialize rate limiter
    rate_limiter = RateLimiter()
    
    # Example API endpoint protection
    def protect_endpoint(limit_type: RateLimitType):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Mock request
                request = MockRequest(
                    headers={"Authorization": "Bearer mock_token"},
                    client_ip="192.168.1.100"
                )
                
                # Check rate limit
                from .rate_limiter import RateLimitMiddleware
                middleware = RateLimitMiddleware(rate_limiter)
                rate_result = middleware.check_request(request, limit_type)
                
                if not rate_result.allowed:
                    return {
                        "error": "Rate limit exceeded",
                        "retry_after": rate_result.retry_after,
                        "message": rate_result.reason
                    }
                
                # Call the actual function
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    # Example protected endpoints
    @protect_endpoint(RateLimitType.LOGIN)
    async def login_endpoint():
        return {"message": "Login successful"}
    
    @protect_endpoint(RateLimitType.API_REQUEST)
    async def api_endpoint():
        return {"data": "API response"}
    
    # Test the protected endpoints
    print("Testing protected endpoints...")
    
    # Test login endpoint
    for i in range(7):
        result = await login_endpoint()
        if "error" in result:
            print(f"Login attempt {i+1}: Rate limited - {result['message']}")
            break
        else:
            print(f"Login attempt {i+1}: Success")
    
    print()
    
    # Test API endpoint (higher limit)
    for i in range(3):
        result = await api_endpoint()
        print(f"API request {i+1}: {'Success' if 'data' in result else 'Rate limited'}")
    
    print("\n=== API Integration Example Complete ===")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(example_authentication_flow())
    asyncio.run(example_api_integration())