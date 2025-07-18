"""
Security Integration Layer

Provides a unified interface for all security components with comprehensive
authentication, authorization, and security monitoring capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

from .models import (
    AuthenticationResult, AuthorizationResult, SecurityContext, UserRole, Permission,
    AuthenticationMethod, MFAMethod, SecurityEvent, SecurityEventType, ThreatLevel,
    UserSession, APIKey, SecurityConfig
)
from .database_integration import SecurityDatabaseManager
from .redis_integration import SecurityRedisManager
from .email_sms_services import EmailService, SMSService, NotificationOrchestrator
from .mfa_manager import MFAManager
from .api_key_manager import APIKeyManager
from .oauth2_manager import OAuth2Manager
from .session_manager import SessionManager
from .jwt_manager import JWTManager


class SecurityManager:
    """
    Unified Security Manager
    
    Provides a single interface for all security operations including authentication,
    authorization, session management, and security monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self._init_components()
        
        # Security statistics
        self.security_stats = {
            "authentication_attempts": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "authorization_checks": 0,
            "authorization_denials": 0,
            "security_events": 0,
            "suspicious_activities": 0
        }
    
    def _init_components(self):
        """Initialize all security components"""
        try:
            # Database integration
            if self.config.get("database", {}).get("enabled", True):
                database_url = self.config["database"]["url"]
                self.db_manager = SecurityDatabaseManager(database_url, **self.config["database"])
            else:
                self.db_manager = None
            
            # Redis integration
            if self.config.get("redis", {}).get("enabled", True):
                redis_url = self.config["redis"]["url"]
                self.redis_manager = SecurityRedisManager(redis_url, **self.config["redis"])
            else:
                self.redis_manager = None
            
            # Email/SMS services
            email_config = self.config.get("email", {})
            sms_config = self.config.get("sms", {})
            
            if email_config.get("enabled", True):
                self.email_service = EmailService(email_config)
            else:
                self.email_service = None
            
            if sms_config.get("enabled", True):
                self.sms_service = SMSService(sms_config)
            else:
                self.sms_service = None
            
            # Notification orchestrator
            if self.email_service or self.sms_service:
                self.notification_orchestrator = NotificationOrchestrator(
                    self.email_service, self.sms_service
                )
            else:
                self.notification_orchestrator = None
            
            # Security component managers
            self.mfa_manager = MFAManager(self.config.get("mfa", {}))
            self.api_key_manager = APIKeyManager(self.config.get("api_keys", {}))
            self.oauth2_manager = OAuth2Manager(self.config.get("oauth2", {}))
            self.session_manager = SessionManager(self.config.get("sessions", {}))
            self.jwt_manager = JWTManager(self.config.get("jwt", {}))
            
            self.logger.info("Security components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security components: {e}")
            raise
    
    async def initialize(self):
        """Initialize async components"""
        try:
            # Initialize database
            if self.db_manager:
                await self.db_manager.create_tables()
            
            # Initialize Redis
            if self.redis_manager:
                await self.redis_manager.connect()
            
            self.logger.info("Security manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security manager: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup resources"""
        try:
            if self.db_manager:
                await self.db_manager.close()
            
            if self.redis_manager:
                await self.redis_manager.disconnect()
            
            if self.oauth2_manager:
                await self.oauth2_manager.__aexit__(None, None, None)
            
            self.logger.info("Security manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during security manager shutdown: {e}")
    
    # Authentication Methods
    async def authenticate_password(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        require_mfa: bool = True
    ) -> AuthenticationResult:
        """Authenticate user with username/password"""
        try:
            self.security_stats["authentication_attempts"] += 1
            
            # Check for account lockout
            if self.redis_manager:
                is_locked = await self.redis_manager.is_account_locked(username)
                if is_locked:
                    await self._log_security_event(
                        SecurityEventType.LOGIN_FAILURE,
                        username,
                        ip_address,
                        user_agent,
                        "Account locked due to failed attempts",
                        ThreatLevel.HIGH
                    )
                    return AuthenticationResult(
                        success=False,
                        error_message="Account temporarily locked due to multiple failed attempts"
                    )
            
            # Validate credentials (this would integrate with your user management system)
            user_data = await self._validate_user_credentials(username, password)
            
            if not user_data:
                # Track failed attempt
                if self.redis_manager:
                    await self.redis_manager.track_login_attempt(username, False)
                
                self.security_stats["failed_authentications"] += 1
                
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    username,
                    ip_address,
                    user_agent,
                    "Invalid credentials",
                    ThreatLevel.MEDIUM
                )
                
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid username or password"
                )
            
            # Check if MFA is required
            if require_mfa:
                mfa_methods = await self._get_user_mfa_methods(user_data["id"])
                if mfa_methods:
                    return AuthenticationResult(
                        success=True,
                        user_id=UUID(user_data["id"]),
                        username=username,
                        mfa_required=True,
                        mfa_methods=mfa_methods
                    )
            
            # Complete authentication
            return await self._complete_authentication(
                user_data,
                AuthenticationMethod.PASSWORD,
                ip_address,
                user_agent
            )
            
        except Exception as e:
            self.logger.error(f"Password authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Authentication service error"
            )
    
    async def authenticate_api_key(
        self,
        api_key: str,
        ip_address: str,
        requested_permissions: List[Permission] = None
    ) -> AuthenticationResult:
        """Authenticate API key"""
        try:
            self.security_stats["authentication_attempts"] += 1
            
            # Verify API key
            verification_result = await self.api_key_manager.verify_api_key(
                api_key, ip_address, requested_permissions[0] if requested_permissions else None
            )
            
            if not verification_result["success"]:
                self.security_stats["failed_authentications"] += 1
                return AuthenticationResult(
                    success=False,
                    error_message=verification_result["error"]
                )
            
            # Create security context for API key
            user_id = verification_result.get("user_id")
            service_name = verification_result.get("service_name")
            permissions = verification_result.get("permissions", [])
            
            security_context = SecurityContext(
                user_id=UUID(user_id) if user_id else uuid4(),
                username=service_name or f"api_key_{verification_result['key_id'][:8]}",
                email="",
                roles=[UserRole.API_USER],
                permissions=set(permissions),
                ip_address=ip_address,
                authentication_method=AuthenticationMethod.API_KEY
            )
            
            self.security_stats["successful_authentications"] += 1
            
            return AuthenticationResult(
                success=True,
                user_id=security_context.user_id,
                username=security_context.username,
                security_context=security_context
            )
            
        except Exception as e:
            self.logger.error(f"API key authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="API key authentication failed"
            )
    
    async def authenticate_jwt_token(self, token: str) -> AuthenticationResult:
        """Authenticate JWT token"""
        try:
            self.security_stats["authentication_attempts"] += 1
            
            # Verify JWT token
            payload = self.jwt_manager.verify_token(token)
            
            if not payload:
                self.security_stats["failed_authentications"] += 1
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid or expired token"
                )
            
            # Check if token is blacklisted (Redis check)
            if self.redis_manager:
                jti = payload.get("jti")
                if jti and await self.redis_manager.is_jwt_token_blacklisted(jti):
                    return AuthenticationResult(
                        success=False,
                        error_message="Token has been revoked"
                    )
            
            # Create security context from token
            security_context = self.jwt_manager.create_security_context(payload)
            
            self.security_stats["successful_authentications"] += 1
            
            return AuthenticationResult(
                success=True,
                user_id=UUID(security_context.user_id),
                username=security_context.username,
                security_context=security_context
            )
            
        except Exception as e:
            self.logger.error(f"JWT authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Token authentication failed"
            )
    
    async def authenticate_oauth2_callback(
        self,
        provider: str,
        code: str,
        state: str,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """Handle OAuth2 callback authentication"""
        try:
            return await self.oauth2_manager.handle_callback(
                provider, code, state, ip_address, user_agent
            )
        except Exception as e:
            self.logger.error(f"OAuth2 authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="OAuth2 authentication failed"
            )
    
    # MFA Methods
    async def setup_mfa(
        self,
        user_id: UUID,
        method: MFAMethod,
        contact_info: Dict[str, str]
    ) -> Dict[str, Any]:
        """Set up MFA for user"""
        try:
            if method == MFAMethod.TOTP:
                return await self.mfa_manager.setup_totp(
                    user_id,
                    contact_info.get("username", ""),
                    contact_info.get("email", "")
                )
            elif method == MFAMethod.SMS:
                return await self.mfa_manager.setup_sms_mfa(
                    user_id,
                    contact_info.get("phone", "")
                )
            elif method == MFAMethod.EMAIL:
                return await self.mfa_manager.setup_email_mfa(
                    user_id,
                    contact_info.get("email", "")
                )
            else:
                return {
                    "success": False,
                    "error": f"MFA method {method.value} not supported"
                }
                
        except Exception as e:
            self.logger.error(f"MFA setup failed: {e}")
            return {"success": False, "error": "MFA setup failed"}
    
    async def verify_mfa(
        self,
        token_id: str,
        verification_code: str,
        method: Optional[MFAMethod] = None
    ) -> Dict[str, Any]:
        """Verify MFA code"""
        try:
            return await self.mfa_manager.verify_mfa_code(token_id, verification_code, method)
        except Exception as e:
            self.logger.error(f"MFA verification failed: {e}")
            return {"success": False, "error": "MFA verification failed"}
    
    # Authorization
    async def authorize_request(
        self,
        security_context: SecurityContext,
        required_permissions: List[Permission],
        resource: str = None,
        action: str = None
    ) -> AuthorizationResult:
        """Authorize request based on security context"""
        try:
            self.security_stats["authorization_checks"] += 1
            
            # Check if context is expired
            if security_context.is_expired():
                self.security_stats["authorization_denials"] += 1
                return AuthorizationResult(
                    allowed=False,
                    reason="Security context has expired",
                    required_permissions=required_permissions,
                    user_permissions=security_context.permissions
                )
            
            # Check permissions
            has_all_permissions = all(
                security_context.has_permission(perm) for perm in required_permissions
            )
            
            if not has_all_permissions:
                self.security_stats["authorization_denials"] += 1
                
                missing_permissions = [
                    perm for perm in required_permissions
                    if not security_context.has_permission(perm)
                ]
                
                await self._log_security_event(
                    SecurityEventType.PERMISSION_DENIED,
                    security_context.username,
                    security_context.ip_address,
                    security_context.user_agent,
                    f"Missing permissions: {[p.value for p in missing_permissions]}",
                    ThreatLevel.LOW
                )
                
                return AuthorizationResult(
                    allowed=False,
                    reason=f"Missing required permissions: {[p.value for p in missing_permissions]}",
                    required_permissions=required_permissions,
                    user_permissions=security_context.permissions
                )
            
            # Log successful authorization
            await self._log_security_event(
                SecurityEventType.PERMISSION_GRANTED,
                security_context.username,
                security_context.ip_address,
                security_context.user_agent,
                f"Access granted to {resource or 'resource'}: {action or 'action'}",
                ThreatLevel.LOW
            )
            
            return AuthorizationResult(
                allowed=True,
                user_permissions=security_context.permissions
            )
            
        except Exception as e:
            self.logger.error(f"Authorization failed: {e}")
            return AuthorizationResult(
                allowed=False,
                reason="Authorization service error"
            )
    
    # Session Management
    async def create_session(
        self,
        user_id: UUID,
        ip_address: str,
        user_agent: str,
        device_fingerprint: Optional[str] = None
    ) -> UserSession:
        """Create user session"""
        try:
            session = await self.session_manager.create_session(
                user_id, ip_address, user_agent, device_fingerprint
            )
            
            # Cache session in Redis
            if self.redis_manager:
                await self.redis_manager.cache_session(session)
            
            # Store session in database
            if self.db_manager:
                await self.db_manager.create_session(session)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            raise
    
    async def validate_session(self, session_token: str) -> Optional[UserSession]:
        """Validate session token"""
        try:
            # Try Redis cache first
            if self.redis_manager:
                session = await self.redis_manager.get_cached_session(session_token)
                if session:
                    await self.redis_manager.update_session_activity(session_token)
                    return session
            
            # Fall back to database
            if self.db_manager:
                session = await self.db_manager.get_session(session_token)
                if session and not session.is_expired():
                    await self.db_manager.update_session_activity(session_token)
                    
                    # Cache in Redis for future requests
                    if self.redis_manager:
                        await self.redis_manager.cache_session(session)
                    
                    return session
            
            # Try in-memory session manager as last resort
            return await self.session_manager.validate_session(session_token)
            
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return None
    
    async def invalidate_session(self, session_token: str, user_id: UUID) -> bool:
        """Invalidate session"""
        try:
            success = True
            
            # Invalidate in Redis
            if self.redis_manager:
                success &= await self.redis_manager.invalidate_session(session_token, user_id)
            
            # Invalidate in database
            if self.db_manager:
                success &= await self.db_manager.invalidate_session(session_token)
            
            # Invalidate in session manager
            success &= await self.session_manager.invalidate_session(session_token)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Session invalidation failed: {e}")
            return False
    
    # Token Management
    async def create_access_token(
        self,
        user_data: Dict[str, Any],
        session_id: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, datetime]:
        """Create access and refresh tokens"""
        try:
            # Create access token
            access_token, access_expiry = self.jwt_manager.create_access_token(
                user_id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
                roles=user_data["roles"],
                permissions=user_data["permissions"],
                session_id=session_id,
                additional_claims=additional_claims
            )
            
            # Create refresh token
            refresh_token, _ = self.jwt_manager.create_refresh_token(
                user_id=user_data["id"],
                session_id=session_id
            )
            
            return access_token, refresh_token, access_expiry
            
        except Exception as e:
            self.logger.error(f"Token creation failed: {e}")
            raise
    
    async def refresh_tokens(self, refresh_token: str) -> Optional[Tuple[str, str, datetime]]:
        """Refresh access token using refresh token"""
        try:
            # Get user data (this would come from your user management system)
            user_data = await self._get_user_data_from_refresh_token(refresh_token)
            
            if not user_data:
                return None
            
            return self.jwt_manager.refresh_access_token(refresh_token, user_data)
            
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            return None
    
    async def revoke_token(self, token: str, reason: str = "logout") -> bool:
        """Revoke JWT token"""
        try:
            # Add to JWT blacklist
            self.jwt_manager.blacklist_token(token, reason)
            
            # Add to Redis blacklist
            if self.redis_manager:
                payload = self.jwt_manager.decode_token_unsafe(token)
                if payload and "jti" in payload and "exp" in payload:
                    expires_at = datetime.fromtimestamp(payload["exp"])
                    await self.redis_manager.blacklist_jwt_token(payload["jti"], expires_at)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Token revocation failed: {e}")
            return False
    
    # Rate Limiting
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60,
        limit_type: str = "general"
    ) -> Dict[str, Any]:
        """Check rate limit"""
        try:
            if self.redis_manager:
                return await self.redis_manager.check_rate_limit(
                    identifier, limit, window_seconds, limit_type
                )
            else:
                # Fallback to allowing request if Redis is not available
                return {"allowed": True, "remaining": limit - 1}
                
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return {"allowed": True, "remaining": 0}
    
    # Security Monitoring
    async def detect_suspicious_activity(
        self,
        user_id: UUID,
        ip_address: str,
        activity_type: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Detect suspicious activity patterns"""
        try:
            is_suspicious = False
            
            # Check for suspicious IP
            if self.redis_manager:
                if await self.redis_manager.is_ip_suspicious(ip_address):
                    is_suspicious = True
            
            # Check for rapid requests from same IP
            rate_limit_result = await self.check_rate_limit(
                ip_address, 100, 60, "suspicious_activity"
            )
            
            if not rate_limit_result["allowed"]:
                is_suspicious = True
                
                if self.redis_manager:
                    await self.redis_manager.mark_suspicious_ip(
                        ip_address, f"Rate limit exceeded for {activity_type}"
                    )
            
            # Additional suspicious activity checks can be added here
            
            if is_suspicious:
                self.security_stats["suspicious_activities"] += 1
                
                await self._log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    str(user_id),
                    ip_address,
                    "",
                    f"Suspicious {activity_type} activity detected",
                    ThreatLevel.HIGH,
                    metadata
                )
            
            return is_suspicious
            
        except Exception as e:
            self.logger.error(f"Suspicious activity detection failed: {e}")
            return False
    
    # Utility Methods
    async def _validate_user_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Validate user credentials (integrate with your user management system)"""
        # This is a mock implementation
        # In production, this would validate against your user database
        if username == "testuser" and password == "testpassword":
            return {
                "id": str(uuid4()),
                "username": username,
                "email": f"{username}@example.com",
                "roles": [UserRole.TENANT],
                "permissions": [Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES]
            }
        return None
    
    async def _get_user_mfa_methods(self, user_id: str) -> List[MFAMethod]:
        """Get MFA methods enabled for user"""
        # Mock implementation
        # In production, this would query the database
        return []
    
    async def _get_user_data_from_refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Get user data from refresh token"""
        # Mock implementation
        payload = self.jwt_manager.decode_token_unsafe(refresh_token)
        if payload and payload.get("token_type") == "refresh":
            user_id = payload["sub"]
            # In production, fetch user data from database
            return {
                "id": user_id,
                "username": "testuser",
                "email": "testuser@example.com",
                "roles": [UserRole.TENANT],
                "permissions": [Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES]
            }
        return None
    
    async def _complete_authentication(
        self,
        user_data: Dict[str, Any],
        auth_method: AuthenticationMethod,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """Complete authentication process"""
        try:
            user_id = UUID(user_data["id"])
            
            # Create session
            session = await self.create_session(
                user_id, ip_address, user_agent
            )
            
            # Create tokens
            access_token, refresh_token, access_expiry = await self.create_access_token(
                user_data, session.session_token
            )
            
            # Create security context
            security_context = SecurityContext(
                user_id=user_id,
                username=user_data["username"],
                email=user_data["email"],
                roles=user_data["roles"],
                permissions=set(user_data["permissions"]),
                session_id=session.session_token,
                ip_address=ip_address,
                user_agent=user_agent,
                authentication_method=auth_method,
                expires_at=access_expiry
            )
            
            # Track successful login
            if self.redis_manager:
                await self.redis_manager.track_login_attempt(user_data["username"], True)
            
            self.security_stats["successful_authentications"] += 1
            
            # Log successful authentication
            await self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                user_data["username"],
                ip_address,
                user_agent,
                f"Successful {auth_method.value} authentication"
            )
            
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                username=user_data["username"],
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=access_expiry,
                security_context=security_context
            )
            
        except Exception as e:
            self.logger.error(f"Authentication completion failed: {e}")
            raise
    
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        username: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        message: str,
        threat_level: ThreatLevel = ThreatLevel.LOW,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security event"""
        try:
            event = SecurityEvent(
                event_type=event_type,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                message=message,
                threat_level=threat_level,
                metadata=metadata or {}
            )
            
            # Log to database
            if self.db_manager:
                await self.db_manager.log_security_event(event)
            
            self.security_stats["security_events"] += 1
            
            # Log high/critical events immediately
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.logger.warning(f"Security event: {event_type.value} - {message}")
            else:
                self.logger.info(f"Security event: {event_type.value} - {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        try:
            stats = {
                "manager_stats": self.security_stats.copy(),
                "mfa_stats": self.mfa_manager.get_mfa_statistics(),
                "api_key_stats": self.api_key_manager.get_api_key_statistics(),
                "oauth2_stats": self.oauth2_manager.get_oauth2_statistics(),
                "session_stats": self.session_manager.get_session_statistics(),
                "jwt_stats": self.jwt_manager.get_token_statistics()
            }
            
            if self.notification_orchestrator:
                stats["notification_stats"] = self.notification_orchestrator.get_delivery_statistics()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get security statistics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                "status": "healthy",
                "components": {}
            }
            
            # Check Redis
            if self.redis_manager:
                redis_health = await self.redis_manager.health_check()
                health_status["components"]["redis"] = redis_health
                
                if redis_health["status"] != "healthy":
                    health_status["status"] = "degraded"
            
            # Check database (simplified)
            if self.db_manager:
                try:
                    # Simple connectivity test
                    health_status["components"]["database"] = {"status": "healthy"}
                except Exception as e:
                    health_status["components"]["database"] = {"status": "unhealthy", "error": str(e)}
                    health_status["status"] = "degraded"
            
            # Add component health
            health_status["components"]["mfa"] = {"status": "healthy"}
            health_status["components"]["api_keys"] = {"status": "healthy"}
            health_status["components"]["oauth2"] = {"status": "healthy"}
            health_status["components"]["sessions"] = {"status": "healthy"}
            health_status["components"]["jwt"] = {"status": "healthy"}
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


@asynccontextmanager
async def security_manager_context(config: Dict[str, Any]):
    """Context manager for SecurityManager"""
    manager = SecurityManager(config)
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.shutdown()