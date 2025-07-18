"""
Authentication Manager

Comprehensive authentication system supporting multiple authentication methods,
security policies, and integration with external identity providers.
"""

import asyncio
import hashlib
import secrets
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

import bcrypt
from email_validator import validate_email, EmailNotValidError

from .models import (
    AuthenticationResult, AuthenticationMethod, SecurityContext,
    UserRole, Permission, SecurityEvent, SecurityEventType, ThreatLevel,
    SecurityConfig, MFAMethod
)
from .jwt_manager import JWTManager
from .mfa_manager import MFAManager
from .session_manager import SessionManager


class AuthenticationManager:
    """
    Authentication Manager with comprehensive security features:
    - Multiple authentication methods (password, OAuth2, API key)
    - Password policy enforcement
    - Account lockout protection
    - Security event logging
    - Integration with MFA and session management
    """
    
    def __init__(
        self,
        jwt_manager: JWTManager,
        mfa_manager: MFAManager,
        session_manager: SessionManager,
        user_repository: Optional[Any] = None,
        database_manager=None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.jwt_manager = jwt_manager
        self.mfa_manager = mfa_manager
        self.session_manager = session_manager
        self.user_repository = user_repository
        self.database_manager = database_manager
        self.config = config or {}
        
        # Email configuration for password reset
        self.email_config = {
            "smtp_server": self.config.get("smtp_server", "localhost"),
            "smtp_port": self.config.get("smtp_port", 587),
            "smtp_username": self.config.get("smtp_username", ""),
            "smtp_password": self.config.get("smtp_password", ""),
            "from_email": self.config.get("from_email", "noreply@rental-ml.com")
        }
        
        # Authentication configuration
        self.password_policy = self._load_password_policy()
        self.lockout_policy = self._load_lockout_policy()
        
        # Failed login tracking (in production, use Redis)
        self._failed_login_attempts: Dict[str, List[datetime]] = {}
        self._locked_accounts: Dict[str, datetime] = {}
        
        # Rate limiting (in production, use Redis)
        self._rate_limiter: Dict[str, List[datetime]] = defaultdict(list)
        self._password_reset_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Input validation patterns
        self._email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self._username_pattern = re.compile(r'^[a-zA-Z0-9_-]{3,20}$')
        self._safe_string_pattern = re.compile(r'^[a-zA-Z0-9\s\._-]+$')
        
        # Authentication statistics
        self._auth_stats = {
            "successful_logins": 0,
            "failed_logins": 0,
            "locked_accounts": 0,
            "password_changes": 0,
            "mfa_activations": 0
        }
    
    # Input Validation and Sanitization Methods
    def _validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email format and content"""
        if not email or len(email) > 254:
            return {"valid": False, "reason": "Invalid email format"}
        
        try:
            # Basic regex check
            if not self._email_pattern.match(email):
                return {"valid": False, "reason": "Invalid email format"}
            
            # More comprehensive validation
            valid = validate_email(email)
            normalized_email = valid.email.lower()
            
            # Check for suspicious patterns
            suspicious_patterns = ['admin', 'root', 'test', 'noreply', 'bounce']
            local_part = normalized_email.split('@')[0]
            if any(pattern in local_part for pattern in suspicious_patterns):
                self.logger.warning(f"Suspicious email pattern detected: {email}")
            
            return {"valid": True, "email": normalized_email}
            
        except EmailNotValidError as e:
            return {"valid": False, "reason": str(e)}
    
    def _validate_username(self, username: str) -> Dict[str, Any]:
        """Validate username format and content"""
        if not username:
            return {"valid": False, "reason": "Username is required"}
        
        if len(username) < 3 or len(username) > 20:
            return {"valid": False, "reason": "Username must be between 3 and 20 characters"}
        
        if not self._username_pattern.match(username):
            return {"valid": False, "reason": "Username can only contain letters, numbers, underscore, and dash"}
        
        # Check for reserved usernames
        reserved_usernames = ['admin', 'root', 'system', 'api', 'service', 'test', 'user']
        if username.lower() in reserved_usernames:
            return {"valid": False, "reason": "Username is reserved"}
        
        return {"valid": True, "username": username.lower()}
    
    def _sanitize_string_input(self, input_str: str, max_length: int = 255) -> str:
        """Sanitize string input to prevent injection attacks"""
        if not input_str:
            return ""
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in ['\t', '\n'])
        
        # Limit length
        sanitized = sanitized[:max_length]
        
        # Remove potentially dangerous patterns
        dangerous_patterns = ['<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        return sanitized.strip()
    
    def _validate_ip_address(self, ip_address: str) -> bool:
        """Validate IP address format"""
        if not ip_address:
            return False
        
        try:
            import ipaddress
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return False
    
    def _check_rate_limit(self, identifier: str, limit_type: str = "default") -> Dict[str, Any]:
        """Check rate limiting for various operations"""
        limits = {
            "login": {"requests": 5, "window_minutes": 1},
            "password_reset": {"requests": 3, "window_minutes": 15},
            "default": {"requests": 10, "window_minutes": 1}
        }
        
        limit_config = limits.get(limit_type, limits["default"])
        key = f"{identifier}:{limit_type}"
        
        now = datetime.now()
        window_start = now - timedelta(minutes=limit_config["window_minutes"])
        
        # Clean old requests
        self._rate_limiter[key] = [
            req_time for req_time in self._rate_limiter[key] 
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(self._rate_limiter[key]) >= limit_config["requests"]:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded for {limit_type}",
                "retry_after": limit_config["window_minutes"] * 60
            }
        
        # Record this request
        self._rate_limiter[key].append(now)
        
        return {"allowed": True}
    
    def _load_password_policy(self) -> Dict[str, Any]:
        """Load password policy configuration"""
        return {
            "min_length": self.config.get("password_min_length", SecurityConfig.PASSWORD_MIN_LENGTH),
            "require_uppercase": self.config.get("password_require_uppercase", SecurityConfig.PASSWORD_REQUIRE_UPPERCASE),
            "require_lowercase": self.config.get("password_require_lowercase", SecurityConfig.PASSWORD_REQUIRE_LOWERCASE),
            "require_digits": self.config.get("password_require_digits", SecurityConfig.PASSWORD_REQUIRE_DIGITS),
            "require_special": self.config.get("password_require_special", SecurityConfig.PASSWORD_REQUIRE_SPECIAL),
            "max_age_days": self.config.get("password_max_age_days", SecurityConfig.PASSWORD_MAX_AGE_DAYS),
            "history_count": self.config.get("password_history_count", SecurityConfig.PASSWORD_HISTORY_COUNT),
            "min_unique_chars": self.config.get("password_min_unique_chars", 6),
            "forbidden_patterns": self.config.get("password_forbidden_patterns", [
                "password", "123456", "qwerty", "admin", "user", "rental"
            ])
        }
    
    def _load_lockout_policy(self) -> Dict[str, Any]:
        """Load account lockout policy configuration"""
        return {
            "max_attempts": self.config.get("max_failed_attempts", SecurityConfig.MAX_FAILED_LOGIN_ATTEMPTS),
            "lockout_duration_minutes": self.config.get("lockout_duration_minutes", SecurityConfig.ACCOUNT_LOCKOUT_DURATION_MINUTES),
            "attempt_window_minutes": self.config.get("attempt_window_minutes", 15),
            "progressive_delays": self.config.get("progressive_delays", [1, 2, 4, 8, 16])  # seconds
        }
    
    async def authenticate_user(
        self,
        identifier: str,  # username or email
        password: str,
        ip_address: str,
        user_agent: str,
        authentication_method: AuthenticationMethod = AuthenticationMethod.PASSWORD
    ) -> AuthenticationResult:
        """
        Authenticate user with comprehensive security checks
        
        Args:
            identifier: Username or email
            password: User password
            ip_address: Client IP address
            user_agent: Client user agent
            authentication_method: Authentication method used
        
        Returns:
            AuthenticationResult with success status and tokens
        """
        try:
            # Input validation and sanitization
            validation_errors = []
            
            # Validate and sanitize identifier
            if not identifier:
                validation_errors.append("Username or email is required")
            else:
                identifier = self._sanitize_string_input(identifier, 254)
                if '@' in identifier:
                    email_validation = self._validate_email(identifier)
                    if not email_validation["valid"]:
                        validation_errors.append(email_validation["reason"])
                    else:
                        identifier = email_validation["email"]
                else:
                    username_validation = self._validate_username(identifier)
                    if not username_validation["valid"]:
                        validation_errors.append(username_validation["reason"])
                    else:
                        identifier = username_validation["username"]
            
            # Validate password
            if not password:
                validation_errors.append("Password is required")
            elif len(password) > 128:  # Prevent DoS via extremely long passwords
                validation_errors.append("Password is too long")
            
            # Validate IP address
            if not self._validate_ip_address(ip_address):
                validation_errors.append("Invalid IP address")
            
            # Validate user agent
            user_agent = self._sanitize_string_input(user_agent, 512)
            
            if validation_errors:
                self.logger.warning(f"Input validation failed for {identifier}: {validation_errors}")
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    username=identifier,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message=f"Input validation failed: {'; '.join(validation_errors)}",
                    threat_level=ThreatLevel.MEDIUM
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid input parameters"
                )
            
            # Rate limiting check
            rate_limit_check = self._check_rate_limit(f"{identifier}:{ip_address}", "login")
            if not rate_limit_check["allowed"]:
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    username=identifier,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message=rate_limit_check["reason"],
                    threat_level=ThreatLevel.HIGH
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Too many login attempts. Please try again later."
                )
            
            self.logger.info(f"Authentication attempt for {identifier} from {ip_address}")
            
            # Pre-authentication security checks
            security_check = await self._pre_authentication_checks(identifier, ip_address)
            if not security_check["allowed"]:
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    username=identifier,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message=security_check["reason"],
                    threat_level=ThreatLevel.HIGH
                )
                return AuthenticationResult(
                    success=False,
                    error_message=security_check["reason"]
                )
            
            # Retrieve user data
            user_data = await self._get_user_by_identifier(identifier)
            if not user_data:
                await self._record_failed_attempt(identifier, ip_address)
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    username=identifier,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="User not found",
                    threat_level=ThreatLevel.MEDIUM
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Check account status
            account_check = await self._check_account_status(user_data)
            if not account_check["valid"]:
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    username=user_data["username"],
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message=account_check["reason"],
                    threat_level=ThreatLevel.HIGH
                )
                return AuthenticationResult(
                    success=False,
                    error_message=account_check["reason"]
                )
            
            # Verify password
            if not await self._verify_password(password, user_data["password_hash"], user_data.get("salt")):
                await self._record_failed_attempt(identifier, ip_address)
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    username=user_data["username"],
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Invalid password",
                    threat_level=ThreatLevel.MEDIUM
                )
                self._auth_stats["failed_logins"] += 1
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Check if MFA is required
            mfa_required = user_data.get("mfa_enabled", False)
            if mfa_required:
                # Create temporary context for MFA verification
                temp_token = await self._create_mfa_verification_token(user_data)
                return AuthenticationResult(
                    success=True,
                    mfa_required=True,
                    mfa_methods=user_data.get("mfa_methods", [MFAMethod.TOTP]),
                    access_token=temp_token
                )
            
            # Create successful authentication result
            result = await self._create_authentication_result(
                user_data, ip_address, user_agent, authentication_method
            )
            
            # Reset failed attempts
            await self._clear_failed_attempts(identifier)
            
            # Log successful authentication
            await self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                username=user_data["username"],
                ip_address=ip_address,
                user_agent=user_agent,
                message="Authentication successful"
            )
            
            self._auth_stats["successful_logins"] += 1
            self.logger.info(f"Authentication successful for {user_data['username']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Authentication error for {identifier}: {e}")
            await self._log_security_event(
                SecurityEventType.SECURITY_VIOLATION,
                username=identifier,
                ip_address=ip_address,
                user_agent=user_agent,
                message=f"Authentication error: {str(e)}",
                threat_level=ThreatLevel.HIGH
            )
            return AuthenticationResult(
                success=False,
                error_message="Authentication service temporarily unavailable"
            )
    
    async def verify_mfa_and_complete_login(
        self,
        mfa_token: str,
        mfa_code: str,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """Complete authentication after MFA verification"""
        try:
            # Verify MFA token and code
            mfa_result = await self.mfa_manager.verify_mfa_code(mfa_token, mfa_code)
            if not mfa_result["success"]:
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    username=mfa_result.get("username"),
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="MFA verification failed",
                    threat_level=ThreatLevel.HIGH
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid MFA code"
                )
            
            # Get user data
            user_data = await self._get_user_by_id(mfa_result["user_id"])
            if not user_data:
                return AuthenticationResult(
                    success=False,
                    error_message="User not found"
                )
            
            # Create authentication result with MFA verified
            result = await self._create_authentication_result(
                user_data, ip_address, user_agent, 
                AuthenticationMethod.PASSWORD, mfa_verified=True
            )
            
            await self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                username=user_data["username"],
                ip_address=ip_address,
                user_agent=user_agent,
                message="Authentication successful with MFA"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"MFA completion error: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Authentication service temporarily unavailable"
            )
    
    async def authenticate_with_api_key(
        self,
        api_key: str,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """Authenticate using API key"""
        try:
            # Verify API key
            api_key_data = await self._verify_api_key(api_key, ip_address)
            if not api_key_data:
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Invalid API key",
                    threat_level=ThreatLevel.HIGH
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid API key"
                )
            
            # Get user or service data
            if api_key_data["user_id"]:
                user_data = await self._get_user_by_id(api_key_data["user_id"])
                username = user_data["username"]
            else:
                # Service API key
                username = api_key_data["service_name"]
                user_data = {
                    "id": api_key_data["service_name"],
                    "username": username,
                    "email": f"{username}@service.local",
                    "roles": ["api_user"],
                    "permissions": api_key_data["permissions"]
                }
            
            # Create service token
            access_token, expiry = self.jwt_manager.create_access_token(
                user_id=str(user_data["id"]),
                username=username,
                email=user_data["email"],
                roles=[UserRole.API_USER],
                permissions=api_key_data["permissions"]
            )
            
            # Update API key usage
            await self._update_api_key_usage(api_key)
            
            await self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                message="API key authentication successful"
            )
            
            return AuthenticationResult(
                success=True,
                user_id=user_data["id"],
                username=username,
                access_token=access_token,
                expires_at=expiry
            )
            
        except Exception as e:
            self.logger.error(f"API key authentication error: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Authentication service temporarily unavailable"
            )
    
    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Change user password with security validation"""
        try:
            # Get user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {"success": False, "error": "User not found"}
            
            # Verify current password
            if not await self._verify_password(current_password, user_data["password_hash"], user_data.get("salt")):
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    username=user_data["username"],
                    ip_address=ip_address,
                    message="Invalid current password in password change",
                    threat_level=ThreatLevel.MEDIUM
                )
                return {"success": False, "error": "Invalid current password"}
            
            # Validate new password
            validation_result = await self._validate_password(new_password, user_data)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["reason"]}
            
            # Hash new password
            new_salt = secrets.token_hex(32)
            new_hash = await self._hash_password(new_password, new_salt)
            
            # Update password
            await self._update_user_password(user_id, new_hash, new_salt)
            
            # Invalidate all user sessions except current
            await self.session_manager.invalidate_user_sessions(user_id, exclude_current=True)
            
            await self._log_security_event(
                SecurityEventType.PASSWORD_CHANGE,
                username=user_data["username"],
                ip_address=ip_address,
                message="Password changed successfully"
            )
            
            self._auth_stats["password_changes"] += 1
            self.logger.info(f"Password changed for user {user_data['username']}")
            
            return {"success": True, "message": "Password changed successfully"}
            
        except Exception as e:
            self.logger.error(f"Password change error for user {user_id}: {e}")
            return {"success": False, "error": "Password change service temporarily unavailable"}
    
    async def initiate_password_reset(
        self,
        email: str,
        ip_address: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """Initiate password reset process"""
        try:
            # Validate email
            email_validation = self._validate_email(email)
            if not email_validation["valid"]:
                return {"success": False, "error": email_validation["reason"]}
            
            email = email_validation["email"]
            
            # Rate limiting for password reset requests
            rate_limit_check = self._check_rate_limit(f"{email}:{ip_address}", "password_reset")
            if not rate_limit_check["allowed"]:
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    username=email,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Password reset rate limit exceeded",
                    threat_level=ThreatLevel.MEDIUM
                )
                return {"success": False, "error": "Too many password reset requests. Please try again later."}
            
            # Check if user exists
            user_data = await self._get_user_by_identifier(email)
            if not user_data:
                # Don't reveal whether user exists, but log the attempt
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    username=email,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Password reset attempted for non-existent user",
                    threat_level=ThreatLevel.MEDIUM
                )
                # Return success to prevent user enumeration
                return {"success": True, "message": "If the email exists, a reset link has been sent"}
            
            # Generate secure reset token
            reset_token = secrets.token_urlsafe(32)
            reset_expires = datetime.now() + timedelta(hours=1)  # 1 hour expiry
            
            # Store reset token
            self._password_reset_tokens[reset_token] = {
                "user_id": user_data["id"],
                "email": email,
                "created_at": datetime.now(),
                "expires_at": reset_expires,
                "ip_address": ip_address,
                "used": False
            }
            
            # Send reset email
            email_sent = await self._send_password_reset_email(email, reset_token, user_data["username"])
            
            if email_sent:
                await self._log_security_event(
                    SecurityEventType.PASSWORD_CHANGE,
                    username=user_data["username"],
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Password reset initiated"
                )
                return {"success": True, "message": "Password reset link sent to your email"}
            else:
                return {"success": False, "error": "Failed to send reset email"}
                
        except Exception as e:
            self.logger.error(f"Password reset initiation error: {e}")
            return {"success": False, "error": "Password reset service temporarily unavailable"}
    
    async def reset_password(
        self,
        reset_token: str,
        new_password: str,
        ip_address: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """Reset password using reset token"""
        try:
            # Validate reset token
            if not reset_token or reset_token not in self._password_reset_tokens:
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Invalid password reset token used",
                    threat_level=ThreatLevel.HIGH
                )
                return {"success": False, "error": "Invalid or expired reset token"}
            
            token_data = self._password_reset_tokens[reset_token]
            
            # Check if token is expired
            if datetime.now() > token_data["expires_at"]:
                del self._password_reset_tokens[reset_token]
                return {"success": False, "error": "Reset token has expired"}
            
            # Check if token was already used
            if token_data["used"]:
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Attempt to reuse password reset token",
                    threat_level=ThreatLevel.HIGH
                )
                return {"success": False, "error": "Reset token has already been used"}
            
            # Get user data
            user_data = await self._get_user_by_id(token_data["user_id"])
            if not user_data:
                return {"success": False, "error": "User not found"}
            
            # Validate new password
            validation_result = await self._validate_password(new_password, user_data)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["reason"]}
            
            # Hash new password
            new_salt = secrets.token_hex(32)
            new_hash = await self._hash_password(new_password, new_salt)
            
            # Update password
            await self._update_user_password(token_data["user_id"], new_hash, new_salt)
            
            # Mark token as used
            token_data["used"] = True
            
            # Invalidate all user sessions
            await self.session_manager.invalidate_user_sessions(token_data["user_id"])
            
            # Clean up expired tokens
            self._cleanup_expired_reset_tokens()
            
            await self._log_security_event(
                SecurityEventType.PASSWORD_CHANGE,
                username=user_data["username"],
                ip_address=ip_address,
                user_agent=user_agent,
                message="Password reset completed"
            )
            
            self.logger.info(f"Password reset completed for user {user_data['username']}")
            return {"success": True, "message": "Password reset successfully"}
            
        except Exception as e:
            self.logger.error(f"Password reset error: {e}")
            return {"success": False, "error": "Password reset service temporarily unavailable"}
    
    async def _send_password_reset_email(self, email: str, reset_token: str, username: str) -> bool:
        """Send password reset email"""
        try:
            if not self.email_config.get("smtp_server"):
                self.logger.warning("Email not configured, skipping reset email")
                return False
            
            # Create reset URL (this would be your frontend URL in production)
            reset_url = f"https://your-app.com/reset-password?token={reset_token}"
            
            # Create email content
            subject = "Password Reset Request - Rental ML System"
            
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Password Reset</title>
            </head>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2>Password Reset Request</h2>
                <p>Hello {username},</p>
                <p>We received a request to reset your password for your Rental ML System account.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href="{reset_url}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                <p>Or copy and paste this URL into your browser:</p>
                <p>{reset_url}</p>
                <p><strong>This link will expire in 1 hour.</strong></p>
                <p>If you didn't request this password reset, please ignore this email.</p>
                <hr>
                <p style="color: #666; font-size: 12px;">This is an automated message from Rental ML System. Please do not reply to this email.</p>
            </body>
            </html>
            """
            
            text_body = f"""
            Password Reset Request
            
            Hello {username},
            
            We received a request to reset your password for your Rental ML System account.
            
            Please visit the following URL to reset your password:
            {reset_url}
            
            This link will expire in 1 hour.
            
            If you didn't request this password reset, please ignore this email.
            
            ---
            This is an automated message from Rental ML System.
            """
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config["from_email"]
            msg['To'] = email
            
            # Add text and HTML parts
            text_part = MIMEText(text_body, 'plain')
            html_part = MIMEText(html_body, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"]) as server:
                if self.email_config.get("smtp_username"):
                    server.starttls()
                    server.login(self.email_config["smtp_username"], self.email_config["smtp_password"])
                
                server.send_message(msg)
            
            self.logger.info(f"Password reset email sent to {email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send password reset email: {e}")
            return False
    
    def _cleanup_expired_reset_tokens(self):
        """Clean up expired password reset tokens"""
        now = datetime.now()
        expired_tokens = [
            token for token, data in self._password_reset_tokens.items()
            if now > data["expires_at"]
        ]
        
        for token in expired_tokens:
            del self._password_reset_tokens[token]
        
        if expired_tokens:
            self.logger.info(f"Cleaned up {len(expired_tokens)} expired reset tokens")
    
    async def logout_user(
        self,
        access_token: str,
        ip_address: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        try:
            # Verify token to get user info
            token_payload = self.jwt_manager.verify_token(access_token)
            if not token_payload:
                return {"success": False, "error": "Invalid token"}
            
            username = token_payload.get("username", "unknown")
            session_id = token_payload.get("session_id")
            
            # Blacklist the token
            self.jwt_manager.blacklist_token(access_token, "logout")
            
            # Invalidate session if session_id exists
            if session_id:
                await self.session_manager.invalidate_session(session_id)
            
            await self._log_security_event(
                SecurityEventType.LOGOUT,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                message="User logged out successfully"
            )
            
            return {"success": True, "message": "Logged out successfully"}
            
        except Exception as e:
            self.logger.error(f"Logout error: {e}")
            return {"success": False, "error": "Logout service temporarily unavailable"}
    
    async def refresh_token(
        self,
        refresh_token: str,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            payload = self.jwt_manager.verify_token(refresh_token)
            if not payload or payload.get("token_type") != "refresh":
                await self._log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    message="Invalid refresh token used",
                    threat_level=ThreatLevel.HIGH
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid refresh token"
                )
            
            user_id = payload["sub"]
            session_id = payload.get("session_id")
            
            # Get user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return AuthenticationResult(
                    success=False,
                    error_message="User not found"
                )
            
            # Check if session is still valid
            if session_id:
                session_valid = await self.session_manager.is_session_valid(session_id)
                if not session_valid:
                    return AuthenticationResult(
                        success=False,
                        error_message="Session expired"
                    )
            
            # Create new tokens
            roles = [UserRole(role) for role in user_data.get("roles", ["guest"])]
            permissions = list(self._get_permissions_for_roles(roles))
            
            new_access_token, new_refresh_token, expiry = self.jwt_manager.refresh_access_token(
                refresh_token, {
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "roles": roles,
                    "permissions": permissions,
                    "mfa_verified": user_data.get("mfa_verified", False)
                }
            )
            
            if not new_access_token:
                return AuthenticationResult(
                    success=False,
                    error_message="Token refresh failed"
                )
            
            await self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                username=user_data["username"],
                ip_address=ip_address,
                user_agent=user_agent,
                message="Token refreshed successfully"
            )
            
            return AuthenticationResult(
                success=True,
                user_id=user_data["id"],
                username=user_data["username"],
                access_token=new_access_token,
                refresh_token=new_refresh_token,
                expires_at=expiry
            )
            
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Token refresh service temporarily unavailable"
            )
    
    async def _pre_authentication_checks(self, identifier: str, ip_address: str) -> Dict[str, Any]:
        """Perform pre-authentication security checks"""
        # Check if account is locked
        if identifier in self._locked_accounts:
            lockout_expires = self._locked_accounts[identifier]
            if datetime.now() < lockout_expires:
                return {
                    "allowed": False,
                    "reason": f"Account locked until {lockout_expires}"
                }
            else:
                # Remove expired lockout
                del self._locked_accounts[identifier]
        
        # Check failed login attempts
        failed_attempts = self._failed_login_attempts.get(identifier, [])
        recent_attempts = [
            attempt for attempt in failed_attempts
            if datetime.now() - attempt < timedelta(minutes=self.lockout_policy["attempt_window_minutes"])
        ]
        
        if len(recent_attempts) >= self.lockout_policy["max_attempts"]:
            # Lock account
            lockout_duration = timedelta(minutes=self.lockout_policy["lockout_duration_minutes"])
            self._locked_accounts[identifier] = datetime.now() + lockout_duration
            self._auth_stats["locked_accounts"] += 1
            
            return {
                "allowed": False,
                "reason": "Account locked due to excessive failed login attempts"
            }
        
        # Check for progressive delays
        if recent_attempts:
            delay_index = min(len(recent_attempts) - 1, len(self.lockout_policy["progressive_delays"]) - 1)
            delay_seconds = self.lockout_policy["progressive_delays"][delay_index]
            
            last_attempt = max(recent_attempts)
            if datetime.now() - last_attempt < timedelta(seconds=delay_seconds):
                return {
                    "allowed": False,
                    "reason": f"Too many recent attempts. Try again in {delay_seconds} seconds"
                }
        
        return {"allowed": True}
    
    async def _get_user_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get user data by username or email"""
        try:
            # First try user repository if available
            if self.user_repository:
                # Try to get by email first
                if '@' in identifier:
                    user = await self.user_repository.get_by_email(identifier)
                else:
                    # For username, we need to extend the repository interface
                    # For now, check email field
                    user = await self.user_repository.get_by_email(identifier)
                
                if user:
                    return await self._convert_user_entity_to_auth_data(user)
            
            # Fallback to database manager
            if self.database_manager:
                async with self.database_manager.get_connection() as conn:
                    user_data = await conn.fetchrow("""
                        SELECT id, username, email, password_hash, salt, roles,
                               mfa_enabled, mfa_methods, account_active, account_expires,
                               password_expires, last_password_change
                        FROM security.user_accounts 
                        WHERE username = $1 OR email = $1
                    """, identifier)
                    
                    return dict(user_data) if user_data else None
            
            # Mock user data for testing
            if identifier in ["testuser", "test@example.com"]:
                return {
                    "id": "test-user-id",
                    "username": "testuser",
                    "email": "test@example.com",
                    "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj.KNBEF5K66",  # "password123"
                    "salt": "test_salt",
                    "roles": ["tenant"],
                    "mfa_enabled": False,
                    "account_active": True,
                    "account_expires": None,
                    "password_expires": None
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting user by identifier {identifier}: {e}")
            return None
    
    async def _convert_user_entity_to_auth_data(self, user: User) -> Dict[str, Any]:
        """Convert User entity to authentication data format"""
        # Since the User entity doesn't have authentication fields,
        # we need to get them from the database
        if self.database_manager:
            try:
                async with self.database_manager.get_connection() as conn:
                    auth_data = await conn.fetchrow("""
                        SELECT password_hash, salt, roles, mfa_enabled, mfa_methods,
                               account_active, account_expires, password_expires, 
                               last_password_change
                        FROM security.user_accounts 
                        WHERE id = $1
                    """, str(user.id))
                    
                    if auth_data:
                        return {
                            "id": str(user.id),
                            "username": user.email.split('@')[0],  # Extract username from email
                            "email": user.email,
                            "password_hash": auth_data["password_hash"],
                            "salt": auth_data["salt"],
                            "roles": auth_data.get("roles", ["tenant"]),
                            "mfa_enabled": auth_data.get("mfa_enabled", False),
                            "mfa_methods": auth_data.get("mfa_methods", []),
                            "account_active": auth_data.get("account_active", user.is_active),
                            "account_expires": auth_data.get("account_expires"),
                            "password_expires": auth_data.get("password_expires"),
                            "last_password_change": auth_data.get("last_password_change")
                        }
            except Exception as e:
                self.logger.error(f"Error converting user entity to auth data: {e}")
        
        # Fallback to basic data from User entity
        return {
            "id": str(user.id),
            "username": user.email.split('@')[0],
            "email": user.email,
            "password_hash": None,  # Will cause authentication to fail
            "salt": None,
            "roles": ["tenant"],
            "mfa_enabled": False,
            "account_active": user.is_active,
            "account_expires": None,
            "password_expires": None
        }
    
    async def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by ID"""
        try:
            # First try user repository if available
            if self.user_repository:
                try:
                    user_uuid = UUID(user_id)
                    user = await self.user_repository.get_by_id(user_uuid)
                    if user:
                        return await self._convert_user_entity_to_auth_data(user)
                except ValueError:
                    self.logger.warning(f"Invalid UUID format for user_id: {user_id}")
            
            # Fallback to database manager
            if self.database_manager:
                async with self.database_manager.get_connection() as conn:
                    user_data = await conn.fetchrow("""
                        SELECT id, username, email, password_hash, salt, roles,
                               mfa_enabled, mfa_methods, account_active, account_expires,
                               password_expires, last_password_change
                        FROM security.user_accounts 
                        WHERE id = $1
                    """, user_id)
                    
                    return dict(user_data) if user_data else None
            
            # Mock user data for testing
            if user_id == "test-user-id":
                return {
                    "id": "test-user-id",
                    "username": "testuser",
                    "email": "test@example.com",
                    "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj.KNBEF5K66",
                    "salt": "test_salt",
                    "roles": ["tenant"],
                    "mfa_enabled": False,
                    "account_active": True,
                    "account_expires": None,
                    "password_expires": None
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting user by ID {user_id}: {e}")
            return None
    
    async def _check_account_status(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if user account is valid and active"""
        if not user_data.get("account_active", True):
            return {"valid": False, "reason": "Account is deactivated"}
        
        account_expires = user_data.get("account_expires")
        if account_expires and datetime.now() > account_expires:
            return {"valid": False, "reason": "Account has expired"}
        
        password_expires = user_data.get("password_expires")
        if password_expires and datetime.now() > password_expires:
            return {"valid": False, "reason": "Password has expired"}
        
        return {"valid": True}
    
    async def _verify_password(self, password: str, stored_hash: str, salt: Optional[str] = None) -> bool:
        """Verify password against stored hash"""
        try:
            if salt:
                # Use salt-based verification
                combined = (password + salt).encode('utf-8')
                return bcrypt.checkpw(combined, stored_hash.encode('utf-8'))
            else:
                # Direct bcrypt verification
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False
    
    async def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt"""
        combined = (password + salt).encode('utf-8')
        hashed = bcrypt.hashpw(combined, bcrypt.gensalt())
        return hashed.decode('utf-8')
    
    async def _validate_password(self, password: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate password against security policy"""
        policy = self.password_policy
        
        # Length check
        if len(password) < policy["min_length"]:
            return {
                "valid": False,
                "reason": f"Password must be at least {policy['min_length']} characters long"
            }
        
        # Character requirements
        if policy["require_uppercase"] and not any(c.isupper() for c in password):
            return {"valid": False, "reason": "Password must contain uppercase letters"}
        
        if policy["require_lowercase"] and not any(c.islower() for c in password):
            return {"valid": False, "reason": "Password must contain lowercase letters"}
        
        if policy["require_digits"] and not any(c.isdigit() for c in password):
            return {"valid": False, "reason": "Password must contain digits"}
        
        if policy["require_special"]:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return {"valid": False, "reason": "Password must contain special characters"}
        
        # Unique characters check
        unique_chars = len(set(password))
        if unique_chars < policy["min_unique_chars"]:
            return {
                "valid": False,
                "reason": f"Password must contain at least {policy['min_unique_chars']} unique characters"
            }
        
        # Forbidden patterns check
        password_lower = password.lower()
        for pattern in policy["forbidden_patterns"]:
            if pattern.lower() in password_lower:
                return {"valid": False, "reason": "Password contains forbidden patterns"}
        
        # Username/email check
        if user_data:
            username = user_data.get("username", "").lower()
            email = user_data.get("email", "").lower()
            
            if username and username in password_lower:
                return {"valid": False, "reason": "Password cannot contain username"}
            
            if email and email.split("@")[0] in password_lower:
                return {"valid": False, "reason": "Password cannot contain email address"}
        
        return {"valid": True}
    
    async def _record_failed_attempt(self, identifier: str, ip_address: str):
        """Record failed login attempt"""
        if identifier not in self._failed_login_attempts:
            self._failed_login_attempts[identifier] = []
        
        self._failed_login_attempts[identifier].append(datetime.now())
        
        # Keep only recent attempts
        cutoff = datetime.now() - timedelta(hours=24)
        self._failed_login_attempts[identifier] = [
            attempt for attempt in self._failed_login_attempts[identifier]
            if attempt > cutoff
        ]
    
    async def _clear_failed_attempts(self, identifier: str):
        """Clear failed login attempts for user"""
        if identifier in self._failed_login_attempts:
            del self._failed_login_attempts[identifier]
    
    async def _create_authentication_result(
        self,
        user_data: Dict[str, Any],
        ip_address: str,
        user_agent: str,
        authentication_method: AuthenticationMethod,
        mfa_verified: bool = False
    ) -> AuthenticationResult:
        """Create authentication result with tokens and session"""
        # Parse user roles and permissions
        roles = [UserRole(role) for role in user_data.get("roles", ["guest"])]
        permissions = self._get_permissions_for_roles(roles)
        
        # Create session
        session = await self.session_manager.create_session(
            user_id=user_data["id"],
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Create tokens
        access_token, access_expiry = self.jwt_manager.create_access_token(
            user_id=str(user_data["id"]),
            username=user_data["username"],
            email=user_data["email"],
            roles=roles,
            permissions=list(permissions),
            session_id=session.session_token,
            mfa_verified=mfa_verified
        )
        
        refresh_token, _ = self.jwt_manager.create_refresh_token(
            user_id=str(user_data["id"]),
            session_id=session.session_token
        )
        
        # Create security context
        security_context = SecurityContext(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=roles,
            permissions=permissions,
            session_id=session.session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            authentication_method=authentication_method,
            mfa_verified=mfa_verified,
            expires_at=access_expiry
        )
        
        return AuthenticationResult(
            success=True,
            user_id=user_data["id"],
            username=user_data["username"],
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=access_expiry,
            security_context=security_context
        )
    
    def _get_permissions_for_roles(self, roles: List[UserRole]) -> set[Permission]:
        """Get all permissions for given roles"""
        from .models import RolePermissionMapping
        
        all_permissions = set()
        role_mappings = RolePermissionMapping.get_default_mappings()
        
        for role in roles:
            if role in role_mappings:
                all_permissions.update(role_mappings[role])
        
        return all_permissions
    
    async def _create_mfa_verification_token(self, user_data: Dict[str, Any]) -> str:
        """Create temporary token for MFA verification"""
        # Create a limited token that only allows MFA verification
        token, _ = self.jwt_manager.create_access_token(
            user_id=str(user_data["id"]),
            username=user_data["username"],
            email=user_data["email"],
            roles=[UserRole.GUEST],  # Limited permissions
            permissions=[],
            additional_claims={"mfa_pending": True}
        )
        return token
    
    async def _verify_api_key(self, api_key: str, ip_address: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data"""
        # In production, this would query the database
        # For now, return mock data
        return {
            "id": "api-key-1",
            "user_id": None,
            "service_name": "test-service",
            "permissions": [Permission.API_ACCESS, Permission.READ_PROPERTY],
            "rate_limit": 1000,
            "ip_whitelist": [],
            "expires_at": None
        }
    
    async def _update_api_key_usage(self, api_key: str):
        """Update API key last used timestamp"""
        # In production, update database
        pass
    
    async def _update_user_password(self, user_id: str, password_hash: str, salt: str):
        """Update user password in database"""
        if not self.database_manager:
            return
        
        async with self.database_manager.get_connection() as conn:
            await conn.execute("""
                UPDATE security.user_accounts 
                SET password_hash = $1, salt = $2, last_password_change = CURRENT_TIMESTAMP
                WHERE id = $3
            """, password_hash, salt, user_id)
    
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        message: str = "",
        threat_level: ThreatLevel = ThreatLevel.LOW,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            message=message,
            threat_level=threat_level,
            metadata=metadata or {}
        )
        
        # In production, send to security monitoring system
        self.logger.info(f"Security event: {event.to_dict()}")
    
    async def create_user_account(
        self,
        email: str,
        password: str,
        username: Optional[str] = None,
        roles: List[str] = None,
        ip_address: str = "127.0.0.1"
    ) -> Dict[str, Any]:
        """Create new user account with authentication data"""
        try:
            # Input validation
            email_validation = self._validate_email(email)
            if not email_validation["valid"]:
                return {"success": False, "error": email_validation["reason"]}
            
            email = email_validation["email"]
            
            if username:
                username_validation = self._validate_username(username)
                if not username_validation["valid"]:
                    return {"success": False, "error": username_validation["reason"]}
                username = username_validation["username"]
            else:
                username = email.split('@')[0]
            
            # Check if user already exists
            existing_user = await self._get_user_by_identifier(email)
            if existing_user:
                return {"success": False, "error": "User already exists"}
            
            # Validate password
            password_validation = await self._validate_password(password, {"email": email})
            if not password_validation["valid"]:
                return {"success": False, "error": password_validation["reason"]}
            
            # Generate password hash and salt
            salt = secrets.token_hex(32)
            password_hash = await self._hash_password(password, salt)
            
            # Set default roles
            if not roles:
                roles = ["tenant"]
            
            user_id = str(uuid4())
            
            # Create user in database
            if self.database_manager:
                async with self.database_manager.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO security.user_accounts 
                        (id, username, email, password_hash, salt, roles, 
                         account_active, created_at, last_password_change)
                        VALUES ($1, $2, $3, $4, $5, $6, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, user_id, username, email, password_hash, salt, roles)
            
            # Create user entity if repository is available
            if self.user_repository:
                from ...domain.entities.user import User, UserPreferences
                user_entity = User.create(email=email)
                await self.user_repository.create(user_entity)
            
            await self._log_security_event(
                SecurityEventType.CONFIGURATION_CHANGE,
                username=username,
                ip_address=ip_address,
                message="User account created",
                threat_level=ThreatLevel.LOW
            )
            
            self.logger.info(f"User account created: {username} ({email})")
            return {
                "success": True, 
                "user_id": user_id,
                "username": username,
                "email": email,
                "message": "User account created successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error creating user account: {e}")
            return {"success": False, "error": "Failed to create user account"}
    
    async def update_user_roles(
        self,
        user_id: str,
        roles: List[str],
        admin_ip_address: str,
        admin_username: str = "system"
    ) -> Dict[str, Any]:
        """Update user roles"""
        try:
            # Validate roles
            valid_roles = [role.value for role in UserRole]
            invalid_roles = [role for role in roles if role not in valid_roles]
            if invalid_roles:
                return {"success": False, "error": f"Invalid roles: {invalid_roles}"}
            
            # Get user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {"success": False, "error": "User not found"}
            
            # Update roles in database
            if self.database_manager:
                async with self.database_manager.get_connection() as conn:
                    await conn.execute("""
                        UPDATE security.user_accounts 
                        SET roles = $1, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $2
                    """, roles, user_id)
            
            await self._log_security_event(
                SecurityEventType.PERMISSION_GRANTED,
                username=user_data["username"],
                ip_address=admin_ip_address,
                message=f"User roles updated by {admin_username}: {roles}",
                threat_level=ThreatLevel.LOW
            )
            
            return {"success": True, "message": "User roles updated successfully"}
            
        except Exception as e:
            self.logger.error(f"Error updating user roles: {e}")
            return {"success": False, "error": "Failed to update user roles"}
    
    async def disable_user_account(
        self,
        user_id: str,
        admin_ip_address: str,
        admin_username: str = "system",
        reason: str = "Account disabled"
    ) -> Dict[str, Any]:
        """Disable user account"""
        try:
            # Get user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {"success": False, "error": "User not found"}
            
            # Update account status
            if self.database_manager:
                async with self.database_manager.get_connection() as conn:
                    await conn.execute("""
                        UPDATE security.user_accounts 
                        SET account_active = false, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                    """, user_id)
            
            # Invalidate all user sessions
            await self.session_manager.invalidate_user_sessions(user_id)
            
            await self._log_security_event(
                SecurityEventType.CONFIGURATION_CHANGE,
                username=user_data["username"],
                ip_address=admin_ip_address,
                message=f"Account disabled by {admin_username}: {reason}",
                threat_level=ThreatLevel.MEDIUM
            )
            
            return {"success": True, "message": "User account disabled successfully"}
            
        except Exception as e:
            self.logger.error(f"Error disabling user account: {e}")
            return {"success": False, "error": "Failed to disable user account"}
    
    async def enable_user_account(
        self,
        user_id: str,
        admin_ip_address: str,
        admin_username: str = "system"
    ) -> Dict[str, Any]:
        """Enable user account"""
        try:
            # Get user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {"success": False, "error": "User not found"}
            
            # Update account status
            if self.database_manager:
                async with self.database_manager.get_connection() as conn:
                    await conn.execute("""
                        UPDATE security.user_accounts 
                        SET account_active = true, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                    """, user_id)
            
            await self._log_security_event(
                SecurityEventType.CONFIGURATION_CHANGE,
                username=user_data["username"],
                ip_address=admin_ip_address,
                message=f"Account enabled by {admin_username}",
                threat_level=ThreatLevel.LOW
            )
            
            return {"success": True, "message": "User account enabled successfully"}
            
        except Exception as e:
            self.logger.error(f"Error enabling user account: {e}")
            return {"success": False, "error": "Failed to enable user account"}
    
    async def force_password_change(
        self,
        user_id: str,
        admin_ip_address: str,
        admin_username: str = "system"
    ) -> Dict[str, Any]:
        """Force user to change password on next login"""
        try:
            # Get user data
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {"success": False, "error": "User not found"}
            
            # Set password expiry to now
            if self.database_manager:
                async with self.database_manager.get_connection() as conn:
                    await conn.execute("""
                        UPDATE security.user_accounts 
                        SET password_expires = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                    """, user_id)
            
            # Invalidate all user sessions
            await self.session_manager.invalidate_user_sessions(user_id)
            
            await self._log_security_event(
                SecurityEventType.CONFIGURATION_CHANGE,
                username=user_data["username"],
                ip_address=admin_ip_address,
                message=f"Forced password change by {admin_username}",
                threat_level=ThreatLevel.LOW
            )
            
            return {"success": True, "message": "User will be required to change password on next login"}
            
        except Exception as e:
            self.logger.error(f"Error forcing password change: {e}")
            return {"success": False, "error": "Failed to force password change"}
    
    def get_authentication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive authentication statistics"""
        return {
            "statistics": self._auth_stats.copy(),
            "failed_attempts_tracking": len(self._failed_login_attempts),
            "locked_accounts": len(self._locked_accounts),
            "password_reset_tokens": len(self._password_reset_tokens),
            "rate_limiter_entries": len(self._rate_limiter),
            "password_policy": self.password_policy,
            "lockout_policy": self.lockout_policy,
            "security_config": {
                "jwt_algorithm": SecurityConfig.JWT_ALGORITHM,
                "access_token_expire_minutes": SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
                "refresh_token_expire_days": SecurityConfig.JWT_REFRESH_TOKEN_EXPIRE_DAYS,
                "max_failed_attempts": SecurityConfig.MAX_FAILED_LOGIN_ATTEMPTS,
                "lockout_duration_minutes": SecurityConfig.ACCOUNT_LOCKOUT_DURATION_MINUTES
            }
        }
    
    def cleanup_expired_data(self):
        """Clean up expired data from memory (call periodically)"""
        try:
            # Clean up expired reset tokens
            self._cleanup_expired_reset_tokens()
            
            # Clean up old failed login attempts
            cutoff = datetime.now() - timedelta(hours=24)
            for identifier in list(self._failed_login_attempts.keys()):
                self._failed_login_attempts[identifier] = [
                    attempt for attempt in self._failed_login_attempts[identifier]
                    if attempt > cutoff
                ]
                if not self._failed_login_attempts[identifier]:
                    del self._failed_login_attempts[identifier]
            
            # Clean up expired account lockouts
            now = datetime.now()
            expired_lockouts = [
                identifier for identifier, expiry in self._locked_accounts.items()
                if now >= expiry
            ]
            for identifier in expired_lockouts:
                del self._locked_accounts[identifier]
            
            # Clean up old rate limiter entries
            for key in list(self._rate_limiter.keys()):
                # Keep only last hour of rate limiting data
                cutoff = datetime.now() - timedelta(hours=1)
                self._rate_limiter[key] = [
                    req_time for req_time in self._rate_limiter[key]
                    if req_time > cutoff
                ]
                if not self._rate_limiter[key]:
                    del self._rate_limiter[key]
            
            # Clean up JWT token blacklist if available
            if hasattr(self.jwt_manager, 'cleanup_expired_blacklist'):
                self.jwt_manager.cleanup_expired_blacklist()
            
            self.logger.info("Expired authentication data cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired data: {e}")
    
    async def get_user_security_info(self, user_id: str) -> Dict[str, Any]:
        """Get user security information"""
        try:
            user_data = await self._get_user_by_id(user_id)
            if not user_data:
                return {"success": False, "error": "User not found"}
            
            # Get session information
            active_sessions = await self.session_manager.get_user_sessions(user_id)
            
            # Check for recent failed attempts
            identifier = user_data["email"]
            recent_failed = len(self._failed_login_attempts.get(identifier, []))
            
            # Check lockout status
            is_locked = identifier in self._locked_accounts
            lockout_expires = None
            if is_locked:
                lockout_expires = self._locked_accounts[identifier]
            
            return {
                "success": True,
                "user_id": user_id,
                "username": user_data["username"],
                "email": user_data["email"],
                "account_active": user_data.get("account_active", True),
                "mfa_enabled": user_data.get("mfa_enabled", False),
                "roles": user_data.get("roles", []),
                "last_password_change": user_data.get("last_password_change"),
                "password_expires": user_data.get("password_expires"),
                "account_expires": user_data.get("account_expires"),
                "recent_failed_attempts": recent_failed,
                "is_locked": is_locked,
                "lockout_expires": lockout_expires.isoformat() if lockout_expires else None,
                "active_sessions": len(active_sessions) if active_sessions else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user security info: {e}")
            return {"success": False, "error": "Failed to get user security information"}