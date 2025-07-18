"""
API Key Manager

Manages API keys for service-to-service authentication and external API access
with comprehensive security features and rate limiting.
"""

import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4

from .models import (
    APIKey, Permission, SecurityEvent, SecurityEventType, 
    ThreatLevel, SecurityConfig
)


class APIKeyManager:
    """
    API Key Manager with comprehensive security features:
    - Secure API key generation and storage
    - Permission-based access control
    - Rate limiting per API key
    - IP address whitelisting
    - Key rotation and expiration
    - Usage analytics and monitoring
    - Audit logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # API key storage (in production, use database)
        self._api_keys: Dict[str, APIKey] = {}
        self._key_hashes: Dict[str, str] = {}  # hash -> key_id mapping
        
        # Rate limiting (in production, use Redis)
        self._rate_limit_buckets: Dict[str, Dict[str, Any]] = {}
        
        # Usage tracking
        self._usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # API key configuration
        self.default_expiry_days = self.config.get(
            "default_expiry_days", 
            SecurityConfig.API_KEY_DEFAULT_EXPIRY_DAYS
        )
        self.max_keys_per_user = self.config.get(
            "max_keys_per_user",
            SecurityConfig.API_KEY_MAX_PER_USER
        )
        
        # Global API key statistics
        self._global_stats = {
            "keys_created": 0,
            "keys_revoked": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "rate_limit_violations": 0,
            "ip_violations": 0
        }
    
    async def create_api_key(
        self,
        name: str,
        user_id: Optional[UUID] = None,
        service_name: Optional[str] = None,
        permissions: List[Permission] = None,
        rate_limit: Optional[int] = None,
        ip_whitelist: List[str] = None,
        expires_in_days: Optional[int] = None,
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Create a new API key
        
        Args:
            name: Human-readable name for the API key
            user_id: User ID (for user-specific keys)
            service_name: Service name (for service-to-service keys)
            permissions: List of permissions granted to this key
            rate_limit: Requests per minute limit
            ip_whitelist: List of allowed IP addresses/ranges
            expires_in_days: Expiration time in days
            created_by: Who created this key
        
        Returns:
            Dictionary with API key data and the actual key
        """
        try:
            # Validate input
            if user_id and service_name:
                raise ValueError("API key cannot be both user-specific and service-specific")
            
            if not user_id and not service_name:
                raise ValueError("API key must specify either user_id or service_name")
            
            # Check user key limit
            if user_id and self._count_user_keys(user_id) >= self.max_keys_per_user:
                raise ValueError(f"User has reached maximum API key limit ({self.max_keys_per_user})")
            
            # Generate API key
            api_key = APIKey(
                name=name,
                user_id=user_id,
                service_name=service_name,
                permissions=permissions or [],
                rate_limit=rate_limit or SecurityConfig.DEFAULT_RATE_LIMIT,
                ip_whitelist=ip_whitelist or [],
                expires_at=None
            )
            
            # Set expiration
            if expires_in_days:
                api_key.expires_at = datetime.now() + timedelta(days=expires_in_days)
            elif self.default_expiry_days > 0:
                api_key.expires_at = datetime.now() + timedelta(days=self.default_expiry_days)
            
            # Store API key
            key_id = str(api_key.id)
            self._api_keys[key_id] = api_key
            
            # Store key hash for lookup
            key_hash = self._hash_api_key(api_key.key)
            self._key_hashes[key_hash] = key_id
            
            # Initialize usage stats
            self._usage_stats[key_id] = {
                "total_requests": 0,
                "last_used": None,
                "daily_requests": {},
                "error_count": 0,
                "rate_limit_violations": 0
            }
            
            self._global_stats["keys_created"] += 1
            
            self.logger.info(f"API key created: {name} for {user_id or service_name}")
            
            # Return key data (only return the actual key once)
            return {
                "success": True,
                "api_key": api_key.key,  # Return actual key
                "key_id": key_id,
                "name": name,
                "permissions": [p.value for p in api_key.permissions],
                "rate_limit": api_key.rate_limit,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "created_at": api_key.created_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create API key {name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def verify_api_key(
        self,
        api_key: str,
        ip_address: str,
        requested_permission: Optional[Permission] = None
    ) -> Dict[str, Any]:
        """
        Verify API key and check permissions
        
        Args:
            api_key: The API key to verify
            ip_address: Client IP address
            requested_permission: Permission being requested
        
        Returns:
            Dictionary with verification result and key data
        """
        try:
            # Hash the provided key
            key_hash = self._hash_api_key(api_key)
            
            # Find API key by hash
            key_id = self._key_hashes.get(key_hash)
            if not key_id:
                self._global_stats["failed_authentications"] += 1
                await self._log_api_event(
                    SecurityEventType.LOGIN_FAILURE,
                    None,
                    ip_address,
                    "Invalid API key",
                    ThreatLevel.HIGH
                )
                return {
                    "success": False,
                    "error": "Invalid API key"
                }
            
            api_key_obj = self._api_keys.get(key_id)
            if not api_key_obj:
                return {
                    "success": False,
                    "error": "API key not found"
                }
            
            # Check if key is active
            if not api_key_obj.is_active:
                return {
                    "success": False,
                    "error": "API key is deactivated"
                }
            
            # Check expiration
            if api_key_obj.is_expired():
                return {
                    "success": False,
                    "error": "API key has expired"
                }
            
            # Check IP whitelist
            if not api_key_obj.can_access_from_ip(ip_address):
                self._global_stats["ip_violations"] += 1
                await self._log_api_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    api_key_obj.user_id or api_key_obj.service_name,
                    ip_address,
                    f"IP address {ip_address} not in whitelist",
                    ThreatLevel.HIGH
                )
                return {
                    "success": False,
                    "error": "IP address not allowed"
                }
            
            # Check rate limiting
            rate_limit_result = await self._check_rate_limit(key_id, api_key_obj.rate_limit)
            if not rate_limit_result["allowed"]:
                self._global_stats["rate_limit_violations"] += 1
                self._usage_stats[key_id]["rate_limit_violations"] += 1
                
                await self._log_api_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    api_key_obj.user_id or api_key_obj.service_name,
                    ip_address,
                    "Rate limit exceeded",
                    ThreatLevel.MEDIUM
                )
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "retry_after": rate_limit_result.get("retry_after")
                }
            
            # Check specific permission if requested
            if requested_permission and requested_permission not in api_key_obj.permissions:
                return {
                    "success": False,
                    "error": f"API key lacks required permission: {requested_permission.value}"
                }
            
            # Update usage statistics
            await self._update_usage_stats(key_id)
            
            self._global_stats["successful_authentications"] += 1
            
            return {
                "success": True,
                "key_id": key_id,
                "user_id": api_key_obj.user_id,
                "service_name": api_key_obj.service_name,
                "permissions": api_key_obj.permissions,
                "name": api_key_obj.name
            }
            
        except Exception as e:
            self.logger.error(f"API key verification failed: {e}")
            return {
                "success": False,
                "error": "API key verification service error"
            }
    
    async def revoke_api_key(self, key_id: str, revoked_by: str = "system") -> Dict[str, Any]:
        """Revoke an API key"""
        try:
            api_key_obj = self._api_keys.get(key_id)
            if not api_key_obj:
                return {
                    "success": False,
                    "error": "API key not found"
                }
            
            # Deactivate the key
            api_key_obj.is_active = False
            
            # Remove from hash lookup
            key_hash = self._hash_api_key(api_key_obj.key)
            self._key_hashes.pop(key_hash, None)
            
            self._global_stats["keys_revoked"] += 1
            
            self.logger.info(f"API key revoked: {api_key_obj.name} by {revoked_by}")
            
            return {
                "success": True,
                "message": "API key revoked successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to revoke API key {key_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def rotate_api_key(self, key_id: str, rotated_by: str = "system") -> Dict[str, Any]:
        """Rotate an API key (generate new key, keep same permissions)"""
        try:
            api_key_obj = self._api_keys.get(key_id)
            if not api_key_obj:
                return {
                    "success": False,
                    "error": "API key not found"
                }
            
            # Remove old key hash
            old_key_hash = self._hash_api_key(api_key_obj.key)
            self._key_hashes.pop(old_key_hash, None)
            
            # Generate new key
            api_key_obj.key = secrets.token_urlsafe(32)
            
            # Store new key hash
            new_key_hash = self._hash_api_key(api_key_obj.key)
            self._key_hashes[new_key_hash] = key_id
            
            self.logger.info(f"API key rotated: {api_key_obj.name} by {rotated_by}")
            
            return {
                "success": True,
                "api_key": api_key_obj.key,  # Return new key
                "message": "API key rotated successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to rotate API key {key_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_api_keys(
        self,
        user_id: Optional[UUID] = None,
        service_name: Optional[str] = None,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """List API keys for user or service"""
        try:
            keys = []
            
            for key_id, api_key_obj in self._api_keys.items():
                # Filter by user or service
                if user_id and api_key_obj.user_id != user_id:
                    continue
                if service_name and api_key_obj.service_name != service_name:
                    continue
                
                # Filter inactive keys
                if not include_inactive and not api_key_obj.is_active:
                    continue
                
                # Get usage stats
                usage = self._usage_stats.get(key_id, {})
                
                keys.append({
                    "key_id": key_id,
                    "name": api_key_obj.name,
                    "permissions": [p.value for p in api_key_obj.permissions],
                    "rate_limit": api_key_obj.rate_limit,
                    "ip_whitelist": api_key_obj.ip_whitelist,
                    "created_at": api_key_obj.created_at.isoformat(),
                    "expires_at": api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None,
                    "last_used_at": api_key_obj.last_used_at.isoformat() if api_key_obj.last_used_at else None,
                    "is_active": api_key_obj.is_active,
                    "is_expired": api_key_obj.is_expired(),
                    "total_requests": usage.get("total_requests", 0),
                    "error_count": usage.get("error_count", 0)
                })
            
            return keys
            
        except Exception as e:
            self.logger.error(f"Failed to list API keys: {e}")
            return []
    
    async def update_api_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        permissions: Optional[List[Permission]] = None,
        rate_limit: Optional[int] = None,
        ip_whitelist: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Update API key properties"""
        try:
            api_key_obj = self._api_keys.get(key_id)
            if not api_key_obj:
                return {
                    "success": False,
                    "error": "API key not found"
                }
            
            # Update properties
            if name is not None:
                api_key_obj.name = name
            if permissions is not None:
                api_key_obj.permissions = permissions
            if rate_limit is not None:
                api_key_obj.rate_limit = rate_limit
            if ip_whitelist is not None:
                api_key_obj.ip_whitelist = ip_whitelist
            if expires_at is not None:
                api_key_obj.expires_at = expires_at
            
            self.logger.info(f"API key updated: {api_key_obj.name}")
            
            return {
                "success": True,
                "message": "API key updated successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update API key {key_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_rate_limit(self, key_id: str, rate_limit: int) -> Dict[str, Any]:
        """Check rate limit for API key"""
        now = datetime.now()
        minute_bucket = now.replace(second=0, microsecond=0)
        
        if key_id not in self._rate_limit_buckets:
            self._rate_limit_buckets[key_id] = {}
        
        bucket = self._rate_limit_buckets[key_id]
        
        # Clean old buckets (keep last 5 minutes)
        cutoff = minute_bucket - timedelta(minutes=5)
        bucket = {k: v for k, v in bucket.items() if k >= cutoff}
        self._rate_limit_buckets[key_id] = bucket
        
        # Check current minute
        current_count = bucket.get(minute_bucket, 0)
        
        if current_count >= rate_limit:
            return {
                "allowed": False,
                "retry_after": 60 - now.second  # Seconds until next minute
            }
        
        # Increment counter
        bucket[minute_bucket] = current_count + 1
        
        return {"allowed": True}
    
    async def _update_usage_stats(self, key_id: str):
        """Update usage statistics for API key"""
        now = datetime.now()
        today = now.date().isoformat()
        
        # Update API key last used
        if key_id in self._api_keys:
            self._api_keys[key_id].last_used_at = now
        
        # Update usage stats
        if key_id not in self._usage_stats:
            self._usage_stats[key_id] = {
                "total_requests": 0,
                "last_used": None,
                "daily_requests": {},
                "error_count": 0,
                "rate_limit_violations": 0
            }
        
        stats = self._usage_stats[key_id]
        stats["total_requests"] += 1
        stats["last_used"] = now.isoformat()
        
        # Update daily stats
        if "daily_requests" not in stats:
            stats["daily_requests"] = {}
        stats["daily_requests"][today] = stats["daily_requests"].get(today, 0) + 1
        
        # Keep only last 30 days
        cutoff_date = (now - timedelta(days=30)).date().isoformat()
        stats["daily_requests"] = {
            date: count for date, count in stats["daily_requests"].items()
            if date >= cutoff_date
        }
    
    def _count_user_keys(self, user_id: UUID) -> int:
        """Count active API keys for user"""
        return sum(
            1 for api_key_obj in self._api_keys.values()
            if api_key_obj.user_id == user_id and api_key_obj.is_active
        )
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def _log_api_event(
        self,
        event_type: SecurityEventType,
        user_identifier: Optional[str],
        ip_address: str,
        message: str,
        threat_level: ThreatLevel = ThreatLevel.LOW
    ):
        """Log API key security event"""
        self.logger.info(
            f"API key event: {event_type.value} - {user_identifier} - {message} - IP: {ip_address}"
        )
    
    def get_api_key_statistics(self) -> Dict[str, Any]:
        """Get API key usage statistics"""
        total_keys = len(self._api_keys)
        active_keys = sum(1 for key in self._api_keys.values() if key.is_active)
        expired_keys = sum(1 for key in self._api_keys.values() if key.is_expired())
        
        # Calculate total usage
        total_requests = sum(
            stats.get("total_requests", 0) 
            for stats in self._usage_stats.values()
        )
        
        return {
            "global_statistics": self._global_stats.copy(),
            "key_counts": {
                "total": total_keys,
                "active": active_keys,
                "inactive": total_keys - active_keys,
                "expired": expired_keys
            },
            "usage": {
                "total_requests": total_requests,
                "rate_limit_buckets": len(self._rate_limit_buckets)
            }
        }
    
    def cleanup_expired_data(self):
        """Clean up expired rate limit buckets and old usage data"""
        now = datetime.now()
        
        # Clean rate limit buckets
        for key_id in list(self._rate_limit_buckets.keys()):
            bucket = self._rate_limit_buckets[key_id]
            cutoff = now.replace(second=0, microsecond=0) - timedelta(minutes=5)
            
            cleaned_bucket = {k: v for k, v in bucket.items() if k >= cutoff}
            
            if cleaned_bucket:
                self._rate_limit_buckets[key_id] = cleaned_bucket
            else:
                del self._rate_limit_buckets[key_id]
        
        # Clean old daily usage data
        cutoff_date = (now - timedelta(days=30)).date().isoformat()
        
        for stats in self._usage_stats.values():
            if "daily_requests" in stats:
                stats["daily_requests"] = {
                    date: count for date, count in stats["daily_requests"].items()
                    if date >= cutoff_date
                }
        
        self.logger.debug("Cleaned up expired API key data")