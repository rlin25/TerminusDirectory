"""
Authorization and RBAC Manager

Comprehensive authorization system with role-based access control (RBAC),
permission management, hierarchical roles, permission inheritance,
resource-based authorization, and fine-grained access control policies.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from uuid import UUID
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import os

from .models import (
    SecurityContext, UserRole, Permission, AuthorizationResult,
    RolePermissionMapping, SecurityEvent, SecurityEventType, ThreatLevel
)


@dataclass
class PermissionDelegation:
    """Permission delegation record"""
    from_user_id: UUID
    to_user_id: UUID
    permissions: Set[Permission]
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if delegation has expired"""
        return self.expires_at and datetime.now() > self.expires_at


@dataclass
class ConditionalAccess:
    """Conditional access policy"""
    name: str
    conditions: Dict[str, Any]
    required_permissions: Set[Permission]
    allowed_roles: Set[UserRole]
    resource_types: Optional[Set[str]] = None
    time_restrictions: Optional[Dict[str, Any]] = None
    ip_restrictions: Optional[List[str]] = None
    is_active: bool = True


class PolicyAction(Enum):
    """Policy enforcement actions"""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_MFA = "require_mfa"
    REQUIRE_APPROVAL = "require_approval"
    AUDIT_ONLY = "audit_only"


@dataclass
class AccessPolicy:
    """Dynamic access policy"""
    id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    action: PolicyAction
    priority: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class AuthorizationManager:
    """
    Authorization Manager with comprehensive RBAC features:
    - Role-based access control with hierarchy
    - Permission checking and validation
    - Resource-level authorization
    - Permission delegation
    - Conditional access policies
    - Dynamic policy evaluation
    - Authorization caching
    - Comprehensive audit logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_configuration(config)
        
        # Load role-permission mappings
        self.role_permissions = RolePermissionMapping.get_default_mappings()
        
        # Custom permission policies
        self._custom_policies: Dict[str, Callable] = {}
        
        # Authorization cache (in production, use Redis)
        self._auth_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = self.config.get("cache_ttl_seconds", 300)  # 5 minutes
        
        # Permission delegation
        self._delegated_permissions: Dict[str, List[PermissionDelegation]] = {}
        
        # Conditional access policies
        self._conditional_policies: List[ConditionalAccess] = []
        
        # Dynamic access policies
        self._access_policies: List[AccessPolicy] = []
        
        # Resource-specific permissions
        self._resource_permissions: Dict[str, Dict[str, Set[Permission]]] = {}
        
        # Authorization statistics
        self._auth_stats = {
            "permission_checks": 0,
            "allowed": 0,
            "denied": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "delegations_used": 0,
            "policies_applied": 0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _load_configuration(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load authorization configuration from various sources"""
        default_config = {
            "cache_ttl_seconds": 300,
            "enable_delegation": True,
            "enable_conditional_access": True,
            "enable_resource_permissions": True,
            "log_all_decisions": False,
            "require_mfa_for_admin": True,
            "session_timeout_minutes": 480,
            "max_delegation_depth": 3,
            "enable_policy_evaluation": True,
            "audit_mode": False
        }
        
        # Override with provided config
        if config:
            default_config.update(config)
        
        # Override with environment variables
        env_overrides = {
            "cache_ttl_seconds": int(os.getenv("AUTH_CACHE_TTL", default_config["cache_ttl_seconds"])),
            "enable_delegation": os.getenv("AUTH_ENABLE_DELEGATION", str(default_config["enable_delegation"])).lower() == "true",
            "log_all_decisions": os.getenv("AUTH_LOG_ALL", str(default_config["log_all_decisions"])).lower() == "true",
            "require_mfa_for_admin": os.getenv("AUTH_REQUIRE_MFA_ADMIN", str(default_config["require_mfa_for_admin"])).lower() == "true",
            "audit_mode": os.getenv("AUTH_AUDIT_MODE", str(default_config["audit_mode"])).lower() == "true"
        }
        
        default_config.update(env_overrides)
        
        self.logger.info(f"Authorization configuration loaded: {json.dumps(default_config, indent=2)}")
        return default_config
    
    def _initialize_default_policies(self):
        """Initialize default access policies"""
        try:
            # High-security operations require MFA
            admin_mfa_policy = AccessPolicy(
                id="admin_mfa_required",
                name="Admin MFA Required",
                description="Require MFA for administrative operations",
                conditions={
                    "roles": [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value],
                    "permissions": [Permission.MANAGE_SYSTEM.value, Permission.MANAGE_SECURITY.value],
                    "mfa_required": True
                },
                action=PolicyAction.REQUIRE_MFA,
                priority=100
            )
            
            # Restrict sensitive operations to specific time windows
            time_restriction_policy = AccessPolicy(
                id="business_hours_only",
                name="Business Hours Restriction",
                description="Restrict sensitive operations to business hours",
                conditions={
                    "permissions": [Permission.DELETE_DATA.value, Permission.MANAGE_SYSTEM.value],
                    "time_window": {"start": "09:00", "end": "17:00"},
                    "weekdays_only": True
                },
                action=PolicyAction.REQUIRE_APPROVAL,
                priority=50
            )
            
            if self.config.get("enable_policy_evaluation", True):
                self._access_policies.extend([admin_mfa_policy, time_restriction_policy])
                self.logger.info("Default access policies initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize default policies: {e}")
    
    def authorize_permission(
        self,
        security_context: SecurityContext,
        permission: Permission,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuthorizationResult:
        """
        Check if user has permission for a specific action
        
        Args:
            security_context: User's security context
            permission: Required permission
            resource_id: Optional resource identifier
            resource_type: Optional resource type
            additional_context: Additional context for authorization
        
        Returns:
            AuthorizationResult with authorization decision
        """
        try:
            self._auth_stats["permission_checks"] += 1
            
            # Check cache first
            cache_key = self._get_cache_key(
                security_context.user_id, permission, resource_id, resource_type
            )
            cached_result = self._get_cached_authorization(cache_key)
            if cached_result:
                self._auth_stats["cache_hits"] += 1
                return cached_result
            
            self._auth_stats["cache_misses"] += 1
            
            # Get all effective permissions (including delegated)
            effective_permissions = self._get_effective_permissions(security_context)
            
            # Check if user has the permission directly or through delegation
            if permission in effective_permissions:
                # Evaluate conditional policies
                policy_result = self.evaluate_conditional_policies(
                    security_context, permission, resource_type, additional_context
                )
                
                if not policy_result.allowed:
                    result = policy_result
                else:
                    result = self._check_resource_access(
                        security_context, permission, resource_id, resource_type, additional_context
                    )
            else:
                result = AuthorizationResult(
                    allowed=False,
                    reason=f"User lacks required permission: {permission.value}",
                    required_permissions=[permission],
                    user_permissions=effective_permissions
                )
            
            # Cache the result
            self._cache_authorization(cache_key, result)
            
            # Update statistics and logging
            if result.allowed:
                self._auth_stats["allowed"] += 1
                if self.config.get("log_all_decisions", False):
                    self._log_authorization_event(
                        security_context, permission, resource_id, 
                        resource_type, True, "Access granted"
                    )
            else:
                self._auth_stats["denied"] += 1
                self._log_authorization_event(
                    security_context, permission, resource_id,
                    resource_type, False, result.reason
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Authorization check failed: {e}")
            return AuthorizationResult(
                allowed=False,
                reason="Authorization service error"
            )
    
    def _get_effective_permissions(self, security_context: SecurityContext) -> Set[Permission]:
        """Get all effective permissions including role-based and delegated"""
        try:
            # Start with role-based permissions
            effective_permissions = set(security_context.permissions)
            
            # Add delegated permissions if enabled
            if self.config.get("enable_delegation", True):
                delegated_perms = self.get_delegated_permissions(security_context.user_id)
                effective_permissions.update(delegated_perms)
                
                if delegated_perms:
                    self._auth_stats["delegations_used"] += 1
            
            return effective_permissions
            
        except Exception as e:
            self.logger.error(f"Failed to get effective permissions: {e}")
            return set(security_context.permissions)
    
    def delegate_permission(
        self,
        from_user_id: UUID,
        to_user_id: UUID,
        permissions: Set[Permission],
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Delegate permissions from one user to another"""
        try:
            if not self.config.get("enable_delegation", True):
                self.logger.warning("Permission delegation is disabled")
                return False
            
            delegation = PermissionDelegation(
                from_user_id=from_user_id,
                to_user_id=to_user_id,
                permissions=permissions,
                resource_type=resource_type,
                resource_id=resource_id,
                expires_at=expires_at
            )
            
            user_key = str(to_user_id)
            if user_key not in self._delegated_permissions:
                self._delegated_permissions[user_key] = []
            
            self._delegated_permissions[user_key].append(delegation)
            
            # Clear cache for affected user
            self.clear_user_cache(to_user_id)
            
            self.logger.info(
                f"Permission delegation created: {from_user_id} -> {to_user_id}, "
                f"permissions: {[p.value for p in permissions]}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delegate permissions: {e}")
            return False
    
    def revoke_delegation(
        self,
        from_user_id: UUID,
        to_user_id: UUID,
        permissions: Optional[Set[Permission]] = None
    ) -> bool:
        """Revoke delegated permissions"""
        try:
            user_key = str(to_user_id)
            if user_key not in self._delegated_permissions:
                return False
            
            delegations = self._delegated_permissions[user_key]
            
            # Remove matching delegations
            if permissions:
                # Remove specific permissions
                for delegation in delegations:
                    if delegation.from_user_id == from_user_id:
                        delegation.permissions -= permissions
                        if not delegation.permissions:
                            delegation.is_active = False
            else:
                # Remove all delegations from this user
                for delegation in delegations:
                    if delegation.from_user_id == from_user_id:
                        delegation.is_active = False
            
            # Clean up inactive delegations
            self._delegated_permissions[user_key] = [
                d for d in delegations if d.is_active and not d.is_expired()
            ]
            
            # Clear cache for affected user
            self.clear_user_cache(to_user_id)
            
            self.logger.info(f"Revoked delegation: {from_user_id} -> {to_user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to revoke delegation: {e}")
            return False
    
    def get_delegated_permissions(self, user_id: UUID) -> Set[Permission]:
        """Get all delegated permissions for a user"""
        try:
            user_key = str(user_id)
            if user_key not in self._delegated_permissions:
                return set()
            
            delegated_perms = set()
            current_time = datetime.now()
            
            # Clean up expired delegations
            active_delegations = []
            
            for delegation in self._delegated_permissions[user_key]:
                if delegation.is_active and not delegation.is_expired():
                    delegated_perms.update(delegation.permissions)
                    active_delegations.append(delegation)
            
            # Update the list to remove expired delegations
            self._delegated_permissions[user_key] = active_delegations
            
            return delegated_perms
            
        except Exception as e:
            self.logger.error(f"Failed to get delegated permissions for user {user_id}: {e}")
            return set()
    
    def evaluate_conditional_policies(
        self,
        security_context: SecurityContext,
        permission: Permission,
        resource_type: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuthorizationResult:
        """Evaluate conditional access policies"""
        try:
            for policy in self._conditional_policies:
                if not policy.is_active:
                    continue
                
                # Check if policy applies to this request
                if not self._policy_applies(policy, security_context, permission, resource_type, additional_context):
                    continue
                
                # Evaluate policy conditions
                if not self._evaluate_policy_conditions(policy, security_context, additional_context):
                    return AuthorizationResult(
                        allowed=False,
                        reason=f"Conditional access policy '{policy.name}' blocked access"
                    )
                
                self._auth_stats["policies_applied"] += 1
            
            return AuthorizationResult(allowed=True)
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate conditional policies: {e}")
            return AuthorizationResult(
                allowed=False,
                reason="Policy evaluation error"
            )
    
    def _policy_applies(
        self,
        policy: ConditionalAccess,
        security_context: SecurityContext,
        permission: Permission,
        resource_type: Optional[str],
        additional_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if a policy applies to the current request"""
        # Check permissions
        if policy.required_permissions and permission not in policy.required_permissions:
            return False
        
        # Check roles
        if policy.allowed_roles and not any(role in policy.allowed_roles for role in security_context.roles):
            return False
        
        # Check resource types
        if policy.resource_types and resource_type and resource_type not in policy.resource_types:
            return False
        
        return True
    
    def _evaluate_policy_conditions(
        self,
        policy: ConditionalAccess,
        security_context: SecurityContext,
        additional_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate policy conditions"""
        try:
            # Time-based restrictions
            if policy.time_restrictions:
                if not self._check_time_restrictions(policy.time_restrictions):
                    return False
            
            # IP-based restrictions
            if policy.ip_restrictions:
                if not self._check_ip_restrictions(policy.ip_restrictions, security_context.ip_address):
                    return False
            
            # Custom conditions
            for condition_name, condition_value in policy.conditions.items():
                if not self._evaluate_condition(condition_name, condition_value, security_context, additional_context):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate policy conditions: {e}")
            return False
    
    def _check_time_restrictions(self, time_restrictions: Dict[str, Any]) -> bool:
        """Check time-based access restrictions"""
        try:
            current_time = datetime.now()
            
            # Check time window
            if "start_time" in time_restrictions and "end_time" in time_restrictions:
                start_time = datetime.strptime(time_restrictions["start_time"], "%H:%M").time()
                end_time = datetime.strptime(time_restrictions["end_time"], "%H:%M").time()
                
                if not (start_time <= current_time.time() <= end_time):
                    return False
            
            # Check weekdays only
            if time_restrictions.get("weekdays_only", False):
                if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check time restrictions: {e}")
            return False
    
    def _check_ip_restrictions(self, ip_restrictions: List[str], user_ip: Optional[str]) -> bool:
        """Check IP-based access restrictions"""
        try:
            if not user_ip:
                return False
            
            # Simple IP matching (in production, use proper CIDR matching)
            return user_ip in ip_restrictions
            
        except Exception as e:
            self.logger.error(f"Failed to check IP restrictions: {e}")
            return False
    
    def _evaluate_condition(
        self,
        condition_name: str,
        condition_value: Any,
        security_context: SecurityContext,
        additional_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate a single condition"""
        try:
            if condition_name == "mfa_required":
                return security_context.mfa_verified if condition_value else True
            
            elif condition_name == "session_age_max_minutes":
                if not security_context.created_at:
                    return False
                session_age = (datetime.now() - security_context.created_at).total_seconds() / 60
                return session_age <= condition_value
            
            elif condition_name == "require_secure_connection":
                if condition_value and additional_context:
                    return additional_context.get("is_secure", False)
                return True
            
            # Add more condition types as needed
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate condition {condition_name}: {e}")
            return False
    
    def authorize_multiple_permissions(
        self,
        security_context: SecurityContext,
        permissions: List[Permission],
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        require_all: bool = True
    ) -> AuthorizationResult:
        """
        Check if user has multiple permissions
        
        Args:
            security_context: User's security context
            permissions: List of required permissions
            resource_id: Optional resource identifier
            resource_type: Optional resource type
            require_all: If True, user must have all permissions; if False, any permission is sufficient
        
        Returns:
            AuthorizationResult with authorization decision
        """
        try:
            results = []
            allowed_permissions = []
            denied_permissions = []
            
            for permission in permissions:
                result = self.authorize_permission(
                    security_context, permission, resource_id, resource_type
                )
                results.append(result)
                
                if result.allowed:
                    allowed_permissions.append(permission)
                else:
                    denied_permissions.append(permission)
            
            if require_all:
                # User must have all permissions
                allowed = len(denied_permissions) == 0
                reason = f"Missing permissions: {[p.value for p in denied_permissions]}" if not allowed else None
            else:
                # User needs at least one permission
                allowed = len(allowed_permissions) > 0
                reason = "User lacks any of the required permissions" if not allowed else None
            
            return AuthorizationResult(
                allowed=allowed,
                reason=reason,
                required_permissions=permissions,
                user_permissions=security_context.permissions
            )
            
        except Exception as e:
            self.logger.error(f"Multiple permission authorization failed: {e}")
            return AuthorizationResult(
                allowed=False,
                reason="Authorization service error"
            )
    
    def authorize_role(
        self,
        security_context: SecurityContext,
        required_role: UserRole
    ) -> AuthorizationResult:
        """
        Check if user has a specific role
        
        Args:
            security_context: User's security context
            required_role: Required role
        
        Returns:
            AuthorizationResult with authorization decision
        """
        try:
            allowed = required_role in security_context.roles
            
            return AuthorizationResult(
                allowed=allowed,
                reason=f"User lacks required role: {required_role.value}" if not allowed else None,
                user_permissions=security_context.permissions
            )
            
        except Exception as e:
            self.logger.error(f"Role authorization failed: {e}")
            return AuthorizationResult(
                allowed=False,
                reason="Authorization service error"
            )
    
    def authorize_resource_ownership(
        self,
        security_context: SecurityContext,
        resource_owner_id: str,
        allow_admin_override: bool = True
    ) -> AuthorizationResult:
        """
        Check if user owns a resource or has admin privileges
        
        Args:
            security_context: User's security context
            resource_owner_id: ID of the resource owner
            allow_admin_override: Allow admin roles to access any resource
        
        Returns:
            AuthorizationResult with authorization decision
        """
        try:
            # Check if user owns the resource
            if str(security_context.user_id) == str(resource_owner_id):
                return AuthorizationResult(
                    allowed=True,
                    user_permissions=security_context.permissions
                )
            
            # Check admin override
            if allow_admin_override:
                admin_roles = {UserRole.SUPER_ADMIN, UserRole.ADMIN}
                if any(role in admin_roles for role in security_context.roles):
                    return AuthorizationResult(
                        allowed=True,
                        reason="Admin override",
                        user_permissions=security_context.permissions
                    )
            
            return AuthorizationResult(
                allowed=False,
                reason="User does not own this resource",
                user_permissions=security_context.permissions
            )
            
        except Exception as e:
            self.logger.error(f"Resource ownership authorization failed: {e}")
            return AuthorizationResult(
                allowed=False,
                reason="Authorization service error"
            )
    
    def _check_resource_access(
        self,
        security_context: SecurityContext,
        permission: Permission,
        resource_id: Optional[str],
        resource_type: Optional[str],
        additional_context: Optional[Dict[str, Any]]
    ) -> AuthorizationResult:
        """
        Check resource-specific access rules
        
        This method can be extended to implement fine-grained resource access control
        based on resource type, ownership, organization membership, etc.
        """
        try:
            # Check resource-specific permissions first
            if resource_type and resource_id:
                resource_perms = self.get_resource_permissions(resource_type, resource_id)
                if resource_perms and permission not in resource_perms:
                    return AuthorizationResult(
                        allowed=False,
                        reason=f"Permission {permission.value} not allowed for this resource"
                    )
            
            # Default: if user has permission, they can access
            result = AuthorizationResult(
                allowed=True,
                user_permissions=security_context.permissions
            )
            
            # Apply resource-specific rules
            if resource_type and resource_id:
                # Example: Property access rules
                if resource_type == "property":
                    result = self._check_property_access(
                        security_context, permission, resource_id, additional_context
                    )
                
                # Example: User data access rules
                elif resource_type == "user":
                    result = self._check_user_data_access(
                        security_context, permission, resource_id, additional_context
                    )
                
                # Example: ML model access rules
                elif resource_type == "ml_model":
                    result = self._check_ml_model_access(
                        security_context, permission, resource_id, additional_context
                    )
            
            # Apply custom policies if defined
            if resource_type in self._custom_policies:
                policy_result = self._custom_policies[resource_type](
                    security_context, permission, resource_id, additional_context
                )
                if not policy_result.allowed:
                    result = policy_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Resource access check failed: {e}")
            return AuthorizationResult(
                allowed=False,
                reason="Resource access check error"
            )
    
    def _check_property_access(
        self,
        security_context: SecurityContext,
        permission: Permission,
        property_id: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> AuthorizationResult:
        """Check property-specific access rules"""
        # Example property access logic
        
        # Tenants can only read properties they're interested in
        if UserRole.TENANT in security_context.roles:
            if permission in [Permission.UPDATE_PROPERTY, Permission.DELETE_PROPERTY]:
                return AuthorizationResult(
                    allowed=False,
                    reason="Tenants cannot modify properties"
                )
        
        # Property managers can manage their own properties
        if UserRole.PROPERTY_MANAGER in security_context.roles:
            # In a real system, you'd check if the user manages this property
            pass
        
        return AuthorizationResult(allowed=True)
    
    def _check_user_data_access(
        self,
        security_context: SecurityContext,
        permission: Permission,
        user_id: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> AuthorizationResult:
        """Check user data access rules"""
        # Users can access their own data
        if str(security_context.user_id) == user_id:
            return AuthorizationResult(allowed=True)
        
        # Admins can access any user data
        admin_roles = {UserRole.SUPER_ADMIN, UserRole.ADMIN}
        if any(role in admin_roles for role in security_context.roles):
            return AuthorizationResult(allowed=True, reason="Admin access")
        
        # Analysts can read user data for analytics
        if (UserRole.ANALYST in security_context.roles and 
            permission == Permission.READ_USER):
            return AuthorizationResult(allowed=True, reason="Analytics access")
        
        return AuthorizationResult(
            allowed=False,
            reason="Cannot access other users' data"
        )
    
    def _check_ml_model_access(
        self,
        security_context: SecurityContext,
        permission: Permission,
        model_id: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> AuthorizationResult:
        """Check ML model access rules"""
        # ML engineers can manage models
        if UserRole.ML_ENGINEER in security_context.roles:
            return AuthorizationResult(allowed=True)
        
        # API users can only read/use models for inference
        if (UserRole.API_USER in security_context.roles and 
            permission in [Permission.READ_PROPERTY, Permission.GET_RECOMMENDATIONS]):
            return AuthorizationResult(allowed=True, reason="API inference access")
        
        return AuthorizationResult(
            allowed=False,
            reason="Insufficient privileges for ML model access"
        )
    
    def set_resource_permissions(
        self,
        resource_type: str,
        resource_id: str,
        permissions: Set[Permission]
    ) -> bool:
        """Set specific permissions for a resource"""
        try:
            if not self.config.get("enable_resource_permissions", True):
                return False
            
            if resource_type not in self._resource_permissions:
                self._resource_permissions[resource_type] = {}
            
            self._resource_permissions[resource_type][resource_id] = permissions
            
            self.logger.info(
                f"Set resource permissions: {resource_type}/{resource_id} -> "
                f"{[p.value for p in permissions]}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set resource permissions: {e}")
            return False
    
    def get_resource_permissions(
        self,
        resource_type: str,
        resource_id: str
    ) -> Optional[Set[Permission]]:
        """Get specific permissions for a resource"""
        try:
            if resource_type in self._resource_permissions:
                return self._resource_permissions[resource_type].get(resource_id)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get resource permissions: {e}")
            return None
    
    def register_custom_policy(
        self,
        resource_type: str,
        policy_function: Callable[[SecurityContext, Permission, str, Optional[Dict[str, Any]]], AuthorizationResult]
    ):
        """Register custom authorization policy for resource type"""
        self._custom_policies[resource_type] = policy_function
        self.logger.info(f"Registered custom policy for resource type: {resource_type}")
    
    def add_conditional_access_policy(self, policy: ConditionalAccess) -> bool:
        """Add a conditional access policy"""
        try:
            if not self.config.get("enable_conditional_access", True):
                self.logger.warning("Conditional access is disabled")
                return False
            
            self._conditional_policies.append(policy)
            self.logger.info(f"Added conditional access policy: {policy.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add conditional access policy: {e}")
            return False
    
    def _get_cache_key(
        self,
        user_id: UUID,
        permission: Permission,
        resource_id: Optional[str],
        resource_type: Optional[str]
    ) -> str:
        """Generate cache key for authorization result"""
        parts = [str(user_id), permission.value]
        if resource_type:
            parts.append(resource_type)
        if resource_id:
            parts.append(resource_id)
        return ":".join(parts)
    
    def _get_cached_authorization(self, cache_key: str) -> Optional[AuthorizationResult]:
        """Get cached authorization result"""
        if cache_key not in self._auth_cache:
            return None
        
        cached_data = self._auth_cache[cache_key]
        
        # Check if cache entry has expired
        if datetime.now() > cached_data["expires_at"]:
            del self._auth_cache[cache_key]
            return None
        
        return cached_data["result"]
    
    def _cache_authorization(self, cache_key: str, result: AuthorizationResult):
        """Cache authorization result"""
        expires_at = datetime.now() + timedelta(seconds=self._cache_ttl_seconds)
        self._auth_cache[cache_key] = {
            "result": result,
            "expires_at": expires_at
        }
    
    def _log_authorization_event(
        self,
        security_context: SecurityContext,
        permission: Permission,
        resource_id: Optional[str],
        resource_type: Optional[str],
        allowed: bool,
        reason: Optional[str]
    ):
        """Log authorization event for audit purposes"""
        event_type = SecurityEventType.PERMISSION_GRANTED if allowed else SecurityEventType.PERMISSION_DENIED
        threat_level = ThreatLevel.LOW if allowed else ThreatLevel.MEDIUM
        
        self.logger.info(
            f"Authorization {('granted' if allowed else 'denied')}: "
            f"user={security_context.username}, "
            f"permission={permission.value}, "
            f"resource_type={resource_type}, "
            f"resource_id={resource_id}, "
            f"reason={reason}"
        )
    
    def clear_user_cache(self, user_id: UUID):
        """Clear cached authorization results for a user"""
        user_id_str = str(user_id)
        keys_to_remove = [
            key for key in self._auth_cache.keys()
            if key.startswith(user_id_str)
        ]
        
        for key in keys_to_remove:
            del self._auth_cache[key]
        
        self.logger.info(f"Cleared authorization cache for user {user_id}")
    
    def clear_cache(self):
        """Clear all cached authorization results"""
        self._auth_cache.clear()
        self.logger.info("Cleared all authorization cache")
    
    def get_user_permissions(self, roles: List[UserRole]) -> Set[Permission]:
        """Get all permissions for given roles"""
        all_permissions = set()
        
        for role in roles:
            if role in self.role_permissions:
                all_permissions.update(self.role_permissions[role])
        
        return all_permissions
    
    def get_authorization_statistics(self) -> Dict[str, Any]:
        """Get authorization statistics"""
        return {
            "statistics": self._auth_stats.copy(),
            "cache_size": len(self._auth_cache),
            "custom_policies": len(self._custom_policies),
            "role_mappings": len(self.role_permissions),
            "conditional_policies": len(self._conditional_policies),
            "access_policies": len(self._access_policies),
            "delegations": sum(len(delegations) for delegations in self._delegated_permissions.values())
        }
    
    def export_authorization_config(self) -> Dict[str, Any]:
        """Export current authorization configuration"""
        try:
            return {
                "config": self.config,
                "role_permissions": {
                    role.value: [p.value for p in perms] 
                    for role, perms in self.role_permissions.items()
                },
                "conditional_policies": len(self._conditional_policies),
                "access_policies": len(self._access_policies),
                "delegations_count": sum(len(delegations) for delegations in self._delegated_permissions.values()),
                "resource_permissions_count": sum(
                    len(resources) for resources in self._resource_permissions.values()
                ),
                "statistics": self._auth_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export authorization config: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform authorization system health check"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "cache": "healthy",
                    "policies": "healthy",
                    "delegation": "healthy",
                    "resource_permissions": "healthy"
                },
                "metrics": {
                    "cache_size": len(self._auth_cache),
                    "active_policies": len([p for p in self._conditional_policies if p.is_active]),
                    "active_delegations": sum(
                        len([d for d in delegations if d.is_active and not d.is_expired()])
                        for delegations in self._delegated_permissions.values()
                    )
                }
            }
            
            # Check cache health
            if len(self._auth_cache) > 10000:  # Arbitrary threshold
                health_status["components"]["cache"] = "warning"
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class RBACManager:
    """
    Role-Based Access Control Manager with Hierarchical Roles
    
    Manages roles, permissions, hierarchical relationships, and inheritance
    """
    
    def __init__(self, authorization_manager: AuthorizationManager):
        self.logger = logging.getLogger(__name__)
        self.auth_manager = authorization_manager
        
        # Dynamic role assignments (in production, use database)
        self._user_roles: Dict[str, Set[UserRole]] = {}
        self._role_hierarchy: Dict[UserRole, Set[UserRole]] = {}
        
        # Role inheritance cache
        self._inheritance_cache: Dict[UserRole, Set[UserRole]] = {}
        
        self._setup_default_hierarchy()
    
    def _setup_default_hierarchy(self):
        """Set up default role hierarchy with inheritance"""
        # Super admin inherits all roles
        self._role_hierarchy[UserRole.SUPER_ADMIN] = set(UserRole)
        
        # Admin inherits from property manager, agent, analyst
        self._role_hierarchy[UserRole.ADMIN] = {
            UserRole.PROPERTY_MANAGER,
            UserRole.AGENT,
            UserRole.ANALYST,
            UserRole.TENANT,
            UserRole.API_USER,
            UserRole.READ_ONLY
        }
        
        # Property manager inherits from agent
        self._role_hierarchy[UserRole.PROPERTY_MANAGER] = {
            UserRole.AGENT,
            UserRole.READ_ONLY
        }
        
        # Agent inherits from read-only
        self._role_hierarchy[UserRole.AGENT] = {
            UserRole.READ_ONLY
        }
        
        # Analyst inherits from read-only
        self._role_hierarchy[UserRole.ANALYST] = {
            UserRole.READ_ONLY
        }
        
        # ML Engineer inherits from read-only
        self._role_hierarchy[UserRole.ML_ENGINEER] = {
            UserRole.READ_ONLY
        }
        
        self.logger.info("Role hierarchy initialized with inheritance")
    
    def assign_role_to_user(self, user_id: str, role: UserRole) -> bool:
        """Assign role to user"""
        try:
            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()
            
            self._user_roles[user_id].add(role)
            
            # Clear user's authorization cache
            self.auth_manager.clear_user_cache(UUID(user_id))
            
            self.logger.info(f"Assigned role {role.value} to user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to assign role {role.value} to user {user_id}: {e}")
            return False
    
    def remove_role_from_user(self, user_id: str, role: UserRole) -> bool:
        """Remove role from user"""
        try:
            if user_id in self._user_roles and role in self._user_roles[user_id]:
                self._user_roles[user_id].remove(role)
                
                # Clear user's authorization cache
                self.auth_manager.clear_user_cache(UUID(user_id))
                
                self.logger.info(f"Removed role {role.value} from user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove role {role.value} from user {user_id}: {e}")
            return False
    
    def get_user_roles(self, user_id: str, include_inherited: bool = True) -> Set[UserRole]:
        """Get all roles for user including inherited roles"""
        direct_roles = self._user_roles.get(user_id, set())
        
        if not include_inherited:
            return direct_roles
        
        # Include inherited roles
        all_roles = set(direct_roles)
        
        for role in direct_roles:
            inherited_roles = self._get_inherited_roles(role)
            all_roles.update(inherited_roles)
        
        return all_roles
    
    def _get_inherited_roles(self, role: UserRole) -> Set[UserRole]:
        """Get all roles that inherit from the given role"""
        if role in self._inheritance_cache:
            return self._inheritance_cache[role]
        
        inherited_roles = set()
        
        if role in self._role_hierarchy:
            inherited_roles.update(self._role_hierarchy[role])
            
            # Recursively get inherited roles
            for inherited_role in self._role_hierarchy[role]:
                inherited_roles.update(self._get_inherited_roles(inherited_role))
        
        self._inheritance_cache[role] = inherited_roles
        return inherited_roles
    
    def get_effective_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for user (including inherited)"""
        roles = self.get_user_roles(user_id, include_inherited=True)
        return self.auth_manager.get_user_permissions(list(roles))
    
    def has_role(self, user_id: str, role: UserRole, include_inherited: bool = True) -> bool:
        """Check if user has specific role"""
        user_roles = self.get_user_roles(user_id, include_inherited)
        return role in user_roles
    
    def add_role_inheritance(self, parent_role: UserRole, child_role: UserRole) -> bool:
        """Add role inheritance relationship"""
        try:
            if parent_role not in self._role_hierarchy:
                self._role_hierarchy[parent_role] = set()
            
            self._role_hierarchy[parent_role].add(child_role)
            
            # Clear inheritance cache
            self._inheritance_cache.clear()
            
            self.logger.info(f"Added inheritance: {parent_role.value} -> {child_role.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add role inheritance: {e}")
            return False
    
    def remove_role_inheritance(self, parent_role: UserRole, child_role: UserRole) -> bool:
        """Remove role inheritance relationship"""
        try:
            if parent_role in self._role_hierarchy and child_role in self._role_hierarchy[parent_role]:
                self._role_hierarchy[parent_role].remove(child_role)
                
                # Clear inheritance cache
                self._inheritance_cache.clear()
                
                self.logger.info(f"Removed inheritance: {parent_role.value} -> {child_role.value}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove role inheritance: {e}")
            return False
    
    async def get_rbac_statistics(self) -> Dict[str, Any]:
        """Get enhanced RBAC statistics"""
        total_delegations = sum(len(delegations) for delegations in self.auth_manager._delegated_permissions.values())
        
        return {
            "total_users_with_roles": len(self._user_roles),
            "role_hierarchy_size": len(self._role_hierarchy),
            "roles_defined": len(UserRole),
            "permissions_defined": len(Permission),
            "inheritance_enabled": True,
            "delegation_enabled": total_delegations > 0,
            "total_delegations": total_delegations,
            "conditional_policies": len(self.auth_manager._conditional_policies),
            "access_policies": len(self.auth_manager._access_policies)
        }


def require_permission(permission: Permission, resource_type: Optional[str] = None):
    """
    Decorator to require specific permission for endpoint access
    
    Usage:
        @require_permission(Permission.CREATE_PROPERTY)
        async def create_property(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object (assumes it's the first argument for FastAPI)
            request = args[0] if args and hasattr(args[0], 'state') else None
            
            if not request:
                raise ValueError("Request object not found in function arguments")
            
            # Get security context from request state
            security_context = getattr(request.state, 'security_context', None)
            if not security_context:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Get authorization manager from request app state
            auth_manager = getattr(request.app.state, 'authorization_manager', None)
            if not auth_manager:
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail="Authorization service unavailable")
            
            # Check permission
            result = auth_manager.authorize_permission(
                security_context, permission, resource_type=resource_type
            )
            
            if not result.allowed:
                from fastapi import HTTPException
                raise HTTPException(status_code=403, detail=result.reason or "Access denied")
            
            # Call the original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: UserRole):
    """
    Decorator to require specific role for endpoint access
    
    Usage:
        @require_role(UserRole.ADMIN)
        async def admin_function(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = args[0] if args and hasattr(args[0], 'state') else None
            
            if not request:
                raise ValueError("Request object not found in function arguments")
            
            # Get security context
            security_context = getattr(request.state, 'security_context', None)
            if not security_context:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check role
            if role not in security_context.roles:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403, 
                    detail=f"Required role: {role.value}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_any_role(*roles: UserRole):
    """
    Decorator to require any of the specified roles for endpoint access
    
    Usage:
        @require_any_role(UserRole.ADMIN, UserRole.PROPERTY_MANAGER)
        async def manager_function(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = args[0] if args and hasattr(args[0], 'state') else None
            
            if not request:
                raise ValueError("Request object not found in function arguments")
            
            # Get security context
            security_context = getattr(request.state, 'security_context', None)
            if not security_context:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check if user has any of the required roles
            if not any(role in security_context.roles for role in roles):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403, 
                    detail=f"Required roles: {[r.value for r in roles]}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_resource_ownership(allow_admin_override: bool = True):
    """
    Decorator to require resource ownership for endpoint access
    
    Usage:
        @require_resource_ownership()
        async def update_user_profile(request: Request, user_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = args[0] if args and hasattr(args[0], 'state') else None
            
            if not request:
                raise ValueError("Request object not found in function arguments")
            
            # Get security context
            security_context = getattr(request.state, 'security_context', None)
            if not security_context:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Get authorization manager
            auth_manager = getattr(request.app.state, 'authorization_manager', None)
            if not auth_manager:
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail="Authorization service unavailable")
            
            # Extract resource owner ID from kwargs (assumes parameter name includes 'user_id' or 'owner_id')
            resource_owner_id = None
            for key, value in kwargs.items():
                if 'user_id' in key.lower() or 'owner_id' in key.lower():
                    resource_owner_id = value
                    break
            
            if not resource_owner_id:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Resource owner ID not found")
            
            # Check ownership
            result = auth_manager.authorize_resource_ownership(
                security_context, resource_owner_id, allow_admin_override
            )
            
            if not result.allowed:
                from fastapi import HTTPException
                raise HTTPException(status_code=403, detail=result.reason or "Access denied")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator