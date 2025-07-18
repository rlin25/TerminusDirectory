"""
Security Models and Enums

Defines core security models, enums, and data structures used across the security system.
"""

import enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import secrets


class UserRole(enum.Enum):
    """User roles for RBAC"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    PROPERTY_MANAGER = "property_manager"
    TENANT = "tenant"
    AGENT = "agent"
    ANALYST = "analyst"
    ML_ENGINEER = "ml_engineer"
    API_USER = "api_user"
    READ_ONLY = "read_only"
    GUEST = "guest"


class Permission(enum.Enum):
    """System permissions"""
    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    
    # Property management
    CREATE_PROPERTY = "create_property"
    READ_PROPERTY = "read_property"
    UPDATE_PROPERTY = "update_property"
    DELETE_PROPERTY = "delete_property"
    
    # Search and recommendations
    SEARCH_PROPERTIES = "search_properties"
    GET_RECOMMENDATIONS = "get_recommendations"
    VIEW_ANALYTICS = "view_analytics"
    
    # ML model management
    CREATE_MODEL = "create_model"
    UPDATE_MODEL = "update_model"
    DELETE_MODEL = "delete_model"
    TRAIN_MODEL = "train_model"
    DEPLOY_MODEL = "deploy_model"
    
    # System administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_SECURITY = "manage_security"
    MANAGE_PERMISSIONS = "manage_permissions"
    
    # Data access
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"
    DELETE_DATA = "delete_data"
    
    # API access
    API_ACCESS = "api_access"
    WEBHOOK_ACCESS = "webhook_access"


class AuthenticationMethod(enum.Enum):
    """Authentication methods"""
    PASSWORD = "password"
    OAUTH2_GOOGLE = "oauth2_google"
    OAUTH2_FACEBOOK = "oauth2_facebook"
    OAUTH2_APPLE = "oauth2_apple"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    BIOMETRIC = "biometric"


class MFAMethod(enum.Enum):
    """Multi-factor authentication methods"""
    TOTP = "totp"  # Time-based OTP (Google Authenticator)
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    HARDWARE_KEY = "hardware_key"
    BIOMETRIC = "biometric"


class SecurityEventType(enum.Enum):
    """Security event types for monitoring"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"


class ThreatLevel(enum.Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: UUID
    username: str
    email: str
    roles: List[UserRole]
    permissions: Set[Permission]
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    authentication_method: Optional[AuthenticationMethod] = None
    mfa_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    def is_expired(self) -> bool:
        """Check if security context has expired"""
        return self.expires_at and datetime.now() > self.expires_at


@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    success: bool
    user_id: Optional[UUID] = None
    username: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    mfa_required: bool = False
    mfa_methods: List[MFAMethod] = field(default_factory=list)
    error_message: Optional[str] = None
    security_context: Optional[SecurityContext] = None
    expires_at: Optional[datetime] = None


@dataclass
class AuthorizationResult:
    """Result of authorization check"""
    allowed: bool
    reason: Optional[str] = None
    required_permissions: List[Permission] = field(default_factory=list)
    user_permissions: Set[Permission] = field(default_factory=set)


@dataclass
class JWTClaims:
    """JWT token claims"""
    sub: str  # Subject (user ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration time
    iat: int  # Issued at
    jti: str  # JWT ID
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: Optional[str] = None
    mfa_verified: bool = False


@dataclass
class APIKey:
    """API key model"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    user_id: Optional[UUID] = None
    service_name: Optional[str] = None
    permissions: List[Permission] = field(default_factory=list)
    rate_limit: Optional[int] = None  # Requests per minute
    ip_whitelist: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if API key has expired"""
        return self.expires_at and datetime.now() > self.expires_at
    
    def can_access_from_ip(self, ip_address: str) -> bool:
        """Check if API key can be used from given IP"""
        if not self.ip_whitelist:
            return True
        return ip_address in self.ip_whitelist


@dataclass
class UserSession:
    """User session model"""
    id: UUID = field(default_factory=uuid4)
    user_id: UUID = None
    session_token: str = field(default_factory=lambda: secrets.token_urlsafe(64))
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=8))
    last_activity_at: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() > self.expires_at
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity_at = datetime.now()


@dataclass
class MFAToken:
    """Multi-factor authentication token"""
    id: UUID = field(default_factory=uuid4)
    user_id: UUID = None
    method: MFAMethod = MFAMethod.TOTP
    token: str = ""
    secret: Optional[str] = None  # For TOTP
    phone_number: Optional[str] = None  # For SMS
    email: Optional[str] = None  # For email
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=5))
    verified: bool = False
    attempts: int = 0
    max_attempts: int = 3
    
    def is_expired(self) -> bool:
        """Check if MFA token has expired"""
        return datetime.now() > self.expires_at
    
    def increment_attempts(self) -> bool:
        """Increment verification attempts and return if still valid"""
        self.attempts += 1
        return self.attempts <= self.max_attempts


@dataclass
class SecurityEvent:
    """Security event for monitoring and auditing"""
    id: UUID = field(default_factory=uuid4)
    event_type: SecurityEventType = SecurityEventType.SUSPICIOUS_ACTIVITY
    user_id: Optional[UUID] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"
    threat_level: ThreatLevel = ThreatLevel.LOW
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "user_id": str(self.user_id) if self.user_id else None,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "threat_level": self.threat_level.value,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RolePermissionMapping:
    """Mapping of roles to permissions"""
    role: UserRole
    permissions: Set[Permission]
    
    @classmethod
    def get_default_mappings(cls) -> Dict[UserRole, Set[Permission]]:
        """Get default role-permission mappings"""
        return {
            UserRole.SUPER_ADMIN: set(Permission),  # All permissions
            
            UserRole.ADMIN: {
                Permission.CREATE_USER, Permission.READ_USER, Permission.UPDATE_USER,
                Permission.CREATE_PROPERTY, Permission.READ_PROPERTY, Permission.UPDATE_PROPERTY,
                Permission.SEARCH_PROPERTIES, Permission.GET_RECOMMENDATIONS, Permission.VIEW_ANALYTICS,
                Permission.MANAGE_SYSTEM, Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_PERMISSIONS,
                Permission.API_ACCESS
            },
            
            UserRole.PROPERTY_MANAGER: {
                Permission.CREATE_PROPERTY, Permission.READ_PROPERTY, Permission.UPDATE_PROPERTY,
                Permission.SEARCH_PROPERTIES, Permission.VIEW_ANALYTICS,
                Permission.API_ACCESS
            },
            
            UserRole.TENANT: {
                Permission.READ_USER, Permission.UPDATE_USER,
                Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES, Permission.GET_RECOMMENDATIONS,
                Permission.API_ACCESS
            },
            
            UserRole.AGENT: {
                Permission.READ_USER, Permission.UPDATE_USER,
                Permission.CREATE_PROPERTY, Permission.READ_PROPERTY, Permission.UPDATE_PROPERTY,
                Permission.SEARCH_PROPERTIES, Permission.GET_RECOMMENDATIONS, Permission.VIEW_ANALYTICS,
                Permission.API_ACCESS
            },
            
            UserRole.ANALYST: {
                Permission.READ_PROPERTY, Permission.VIEW_ANALYTICS,
                Permission.SEARCH_PROPERTIES, Permission.GET_RECOMMENDATIONS,
                Permission.API_ACCESS
            },
            
            UserRole.ML_ENGINEER: {
                Permission.READ_PROPERTY, Permission.VIEW_ANALYTICS,
                Permission.CREATE_MODEL, Permission.UPDATE_MODEL, Permission.TRAIN_MODEL,
                Permission.DEPLOY_MODEL, Permission.API_ACCESS
            },
            
            UserRole.API_USER: {
                Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES,
                Permission.GET_RECOMMENDATIONS, Permission.API_ACCESS
            },
            
            UserRole.READ_ONLY: {
                Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES,
                Permission.API_ACCESS
            },
            
            UserRole.GUEST: {
                Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES
            }
        }


# Security configuration constants
class SecurityConfig:
    """Security configuration constants"""
    
    # JWT settings
    JWT_ALGORITHM = "RS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Password policy
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_MAX_AGE_DAYS = 90
    PASSWORD_HISTORY_COUNT = 12
    
    # Session policy
    SESSION_TIMEOUT_MINUTES = 480  # 8 hours
    SESSION_MAX_CONCURRENT = 5
    SESSION_REQUIRE_SECURE = True
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = 100  # requests per minute
    LOGIN_RATE_LIMIT = 5  # login attempts per minute
    API_RATE_LIMIT = 1000  # API requests per minute
    
    # MFA settings
    MFA_TOTP_ISSUER = "Rental ML System"
    MFA_TOKEN_VALIDITY_MINUTES = 5
    MFA_MAX_ATTEMPTS = 3
    
    # Security monitoring
    MAX_FAILED_LOGIN_ATTEMPTS = 5
    ACCOUNT_LOCKOUT_DURATION_MINUTES = 30
    SUSPICIOUS_ACTIVITY_THRESHOLD = 100  # events per hour
    
    # API key settings
    API_KEY_DEFAULT_EXPIRY_DAYS = 365
    API_KEY_MAX_PER_USER = 10