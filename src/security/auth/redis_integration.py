"""
Redis Integration for Security Components

Provides Redis-based caching, rate limiting, and session management
for high-performance security operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from uuid import UUID

import redis.asyncio as aioredis
from redis.asyncio import Redis

from .models import UserSession, SecurityEvent, SecurityEventType, ThreatLevel


class SecurityRedisManager:
    """
    Redis manager for security components
    
    Provides high-performance caching, rate limiting, and temporary data storage
    for security operations with automatic expiration and cleanup.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", **kwargs):
        self.logger = logging.getLogger(__name__)
        self.redis_url = redis_url
        
        # Redis client configuration
        self.redis_config = {
            "encoding": "utf-8",
            "decode_responses": True,
            "socket_timeout": kwargs.get("socket_timeout", 5),
            "socket_connect_timeout": kwargs.get("socket_connect_timeout", 5),
            "retry_on_timeout": kwargs.get("retry_on_timeout", True),
            "health_check_interval": kwargs.get("health_check_interval", 30),
        }
        
        self.redis_client: Optional[Redis] = None
        
        # Key prefixes for different data types
        self.prefixes = {
            "session": "sec:session:",
            "mfa_token": "sec:mfa:",
            "rate_limit": "sec:rate:",
            "api_key_usage": "sec:api:",
            "blacklist": "sec:blacklist:",
            "oauth_state": "sec:oauth:",
            "pkce": "sec:pkce:",
            "login_attempts": "sec:login_attempts:",
            "security_events": "sec:events:",
            "user_sessions": "sec:user_sessions:",
            "suspicious_ips": "sec:suspicious:",
            "device_fingerprints": "sec:devices:"
        }
    
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                **self.redis_config
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Redis connection closed")
    
    # Session Management
    async def cache_session(self, session: UserSession, ttl_seconds: int = 28800) -> bool:
        """Cache session data with TTL (default 8 hours)"""
        try:
            session_key = f"{self.prefixes['session']}{session.session_token}"
            session_data = {
                "id": str(session.id),
                "user_id": str(session.user_id),
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "last_activity_at": session.last_activity_at.isoformat(),
                "ip_address": session.ip_address,
                "user_agent": session.user_agent,
                "is_active": session.is_active
            }
            
            await self.redis_client.setex(
                session_key,
                ttl_seconds,
                json.dumps(session_data)
            )
            
            # Track user sessions
            user_sessions_key = f"{self.prefixes['user_sessions']}{session.user_id}"
            await self.redis_client.sadd(user_sessions_key, session.session_token)
            await self.redis_client.expire(user_sessions_key, ttl_seconds)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache session: {e}")
            return False
    
    async def get_cached_session(self, session_token: str) -> Optional[UserSession]:
        """Get cached session data"""
        try:
            session_key = f"{self.prefixes['session']}{session_token}"
            session_data = await self.redis_client.get(session_key)
            
            if not session_data:
                return None
            
            data = json.loads(session_data)
            return UserSession(
                id=UUID(data["id"]),
                user_id=UUID(data["user_id"]),
                session_token=session_token,
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]),
                last_activity_at=datetime.fromisoformat(data["last_activity_at"]),
                ip_address=data["ip_address"],
                user_agent=data["user_agent"],
                is_active=data["is_active"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get cached session: {e}")
            return None
    
    async def invalidate_session(self, session_token: str, user_id: UUID) -> bool:
        """Invalidate cached session"""
        try:
            session_key = f"{self.prefixes['session']}{session_token}"
            user_sessions_key = f"{self.prefixes['user_sessions']}{user_id}"
            
            # Remove session data
            await self.redis_client.delete(session_key)
            
            # Remove from user sessions set
            await self.redis_client.srem(user_sessions_key, session_token)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate session: {e}")
            return False
    
    async def update_session_activity(self, session_token: str) -> bool:
        """Update session last activity timestamp"""
        try:
            session_key = f"{self.prefixes['session']}{session_token}"
            session_data = await self.redis_client.get(session_key)
            
            if not session_data:
                return False
            
            data = json.loads(session_data)
            data["last_activity_at"] = datetime.now().isoformat()
            
            # Get current TTL and preserve it
            ttl = await self.redis_client.ttl(session_key)
            if ttl > 0:
                await self.redis_client.setex(session_key, ttl, json.dumps(data))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update session activity: {e}")
            return False
    
    async def get_user_sessions(self, user_id: UUID) -> List[str]:
        """Get all active session tokens for user"""
        try:
            user_sessions_key = f"{self.prefixes['user_sessions']}{user_id}"
            return await self.redis_client.smembers(user_sessions_key)
            
        except Exception as e:
            self.logger.error(f"Failed to get user sessions: {e}")
            return []
    
    # Rate Limiting
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60,
        limit_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Check rate limit using sliding window algorithm
        
        Args:
            identifier: Unique identifier (IP, user ID, API key, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            limit_type: Type of rate limit (for different key prefixes)
        
        Returns:
            Dict with allowed status and remaining count
        """
        try:
            rate_key = f"{self.prefixes['rate_limit']}{limit_type}:{identifier}"
            now = datetime.now()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Use sorted set to track requests with timestamps
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(rate_key, 0, window_start.timestamp())
            
            # Count current requests in window
            pipe.zcard(rate_key)
            
            # Add current request
            pipe.zadd(rate_key, {str(now.timestamp()): now.timestamp()})
            
            # Set expiration for cleanup
            pipe.expire(rate_key, window_seconds + 1)
            
            results = await pipe.execute()
            current_count = results[1] + 1  # Include the request we just added
            
            remaining = max(0, limit - current_count)
            allowed = current_count <= limit
            
            if not allowed:
                # Remove the request we just added since it's not allowed
                await self.redis_client.zrem(rate_key, str(now.timestamp()))
            
            return {
                "allowed": allowed,
                "remaining": remaining,
                "current_count": current_count,
                "reset_time": (window_start + timedelta(seconds=window_seconds)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check rate limit: {e}")
            # Default to allowing the request if Redis is down
            return {"allowed": True, "remaining": limit - 1, "current_count": 1}
    
    async def get_rate_limit_info(self, identifier: str, limit_type: str = "general") -> Dict[str, Any]:
        """Get current rate limit status without incrementing"""
        try:
            rate_key = f"{self.prefixes['rate_limit']}{limit_type}:{identifier}"
            current_count = await self.redis_client.zcard(rate_key)
            
            return {
                "current_count": current_count,
                "window_start": (datetime.now() - timedelta(seconds=60)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get rate limit info: {e}")
            return {"current_count": 0}
    
    # MFA Token Caching
    async def cache_mfa_token(self, token_id: str, token_data: Dict[str, Any], ttl_seconds: int = 300) -> bool:
        """Cache MFA token with TTL (default 5 minutes)"""
        try:
            mfa_key = f"{self.prefixes['mfa_token']}{token_id}"
            await self.redis_client.setex(mfa_key, ttl_seconds, json.dumps(token_data))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache MFA token: {e}")
            return False
    
    async def get_mfa_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get cached MFA token"""
        try:
            mfa_key = f"{self.prefixes['mfa_token']}{token_id}"
            token_data = await self.redis_client.get(mfa_key)
            
            if token_data:
                return json.loads(token_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get MFA token: {e}")
            return None
    
    async def invalidate_mfa_token(self, token_id: str) -> bool:
        """Remove MFA token from cache"""
        try:
            mfa_key = f"{self.prefixes['mfa_token']}{token_id}"
            await self.redis_client.delete(mfa_key)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate MFA token: {e}")
            return False
    
    # JWT Token Blacklist
    async def blacklist_jwt_token(self, jti: str, expires_at: datetime) -> bool:
        """Add JWT token to blacklist"""
        try:
            blacklist_key = f"{self.prefixes['blacklist']}{jti}"
            ttl_seconds = int((expires_at - datetime.now()).total_seconds())
            
            if ttl_seconds > 0:
                await self.redis_client.setex(blacklist_key, ttl_seconds, "blacklisted")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to blacklist JWT token: {e}")
            return False
    
    async def is_jwt_token_blacklisted(self, jti: str) -> bool:
        """Check if JWT token is blacklisted"""
        try:
            blacklist_key = f"{self.prefixes['blacklist']}{jti}"
            return await self.redis_client.exists(blacklist_key)
            
        except Exception as e:
            self.logger.error(f"Failed to check JWT blacklist: {e}")
            return True  # Err on the side of caution
    
    # OAuth2 State Management
    async def store_oauth_state(self, state: str, state_data: Dict[str, Any], ttl_seconds: int = 600) -> bool:
        """Store OAuth2 state with TTL (default 10 minutes)"""
        try:
            oauth_key = f"{self.prefixes['oauth_state']}{state}"
            await self.redis_client.setex(oauth_key, ttl_seconds, json.dumps(state_data))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store OAuth2 state: {e}")
            return False
    
    async def get_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Get OAuth2 state data"""
        try:
            oauth_key = f"{self.prefixes['oauth_state']}{state}"
            state_data = await self.redis_client.get(oauth_key)
            
            if state_data:
                return json.loads(state_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get OAuth2 state: {e}")
            return None
    
    async def remove_oauth_state(self, state: str) -> bool:
        """Remove OAuth2 state after use"""
        try:
            oauth_key = f"{self.prefixes['oauth_state']}{state}"
            await self.redis_client.delete(oauth_key)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove OAuth2 state: {e}")
            return False
    
    # PKCE Code Verifiers
    async def store_pkce_verifier(self, state: str, code_verifier: str, ttl_seconds: int = 600) -> bool:
        """Store PKCE code verifier"""
        try:
            pkce_key = f"{self.prefixes['pkce']}{state}"
            await self.redis_client.setex(pkce_key, ttl_seconds, code_verifier)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store PKCE verifier: {e}")
            return False
    
    async def get_pkce_verifier(self, state: str) -> Optional[str]:
        """Get and remove PKCE code verifier"""
        try:
            pkce_key = f"{self.prefixes['pkce']}{state}"
            
            # Get and delete in one operation
            pipe = self.redis_client.pipeline()
            pipe.get(pkce_key)
            pipe.delete(pkce_key)
            results = await pipe.execute()
            
            return results[0]
            
        except Exception as e:
            self.logger.error(f"Failed to get PKCE verifier: {e}")
            return None
    
    # Login Attempt Tracking
    async def track_login_attempt(self, identifier: str, success: bool, ttl_seconds: int = 3600) -> Dict[str, Any]:
        """Track login attempts for brute force protection"""
        try:
            attempts_key = f"{self.prefixes['login_attempts']}{identifier}"
            
            # Get current attempts
            attempts_data = await self.redis_client.get(attempts_key)
            
            if attempts_data:
                attempts = json.loads(attempts_data)
            else:
                attempts = {"failed": 0, "total": 0, "last_attempt": None, "locked_until": None}
            
            attempts["total"] += 1
            attempts["last_attempt"] = datetime.now().isoformat()
            
            if not success:
                attempts["failed"] += 1
                
                # Lock account after 5 failed attempts
                if attempts["failed"] >= 5:
                    lockout_until = datetime.now() + timedelta(minutes=30)
                    attempts["locked_until"] = lockout_until.isoformat()
            else:
                # Reset failed attempts on successful login
                attempts["failed"] = 0
                attempts["locked_until"] = None
            
            await self.redis_client.setex(attempts_key, ttl_seconds, json.dumps(attempts))
            
            return attempts
            
        except Exception as e:
            self.logger.error(f"Failed to track login attempt: {e}")
            return {"failed": 0, "total": 1, "locked_until": None}
    
    async def is_account_locked(self, identifier: str) -> bool:
        """Check if account is locked due to failed attempts"""
        try:
            attempts_key = f"{self.prefixes['login_attempts']}{identifier}"
            attempts_data = await self.redis_client.get(attempts_key)
            
            if not attempts_data:
                return False
            
            attempts = json.loads(attempts_data)
            locked_until = attempts.get("locked_until")
            
            if locked_until:
                lockout_time = datetime.fromisoformat(locked_until)
                return datetime.now() < lockout_time
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check account lock: {e}")
            return False
    
    async def unlock_account(self, identifier: str) -> bool:
        """Manually unlock account"""
        try:
            attempts_key = f"{self.prefixes['login_attempts']}{identifier}"
            await self.redis_client.delete(attempts_key)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unlock account: {e}")
            return False
    
    # Suspicious Activity Tracking
    async def mark_suspicious_ip(self, ip_address: str, reason: str, ttl_seconds: int = 86400) -> bool:
        """Mark IP as suspicious for 24 hours"""
        try:
            suspicious_key = f"{self.prefixes['suspicious_ips']}{ip_address}"
            suspicious_data = {
                "reason": reason,
                "marked_at": datetime.now().isoformat(),
                "count": 1
            }
            
            # Check if already marked and increment count
            existing_data = await self.redis_client.get(suspicious_key)
            if existing_data:
                existing = json.loads(existing_data)
                suspicious_data["count"] = existing.get("count", 0) + 1
            
            await self.redis_client.setex(suspicious_key, ttl_seconds, json.dumps(suspicious_data))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to mark suspicious IP: {e}")
            return False
    
    async def is_ip_suspicious(self, ip_address: str) -> bool:
        """Check if IP is marked as suspicious"""
        try:
            suspicious_key = f"{self.prefixes['suspicious_ips']}{ip_address}"
            return await self.redis_client.exists(suspicious_key)
            
        except Exception as e:
            self.logger.error(f"Failed to check suspicious IP: {e}")
            return False
    
    # Device Fingerprinting
    async def store_device_fingerprint(self, user_id: UUID, fingerprint: str, device_info: Dict[str, Any], ttl_seconds: int = 2592000) -> bool:
        """Store device fingerprint for 30 days"""
        try:
            device_key = f"{self.prefixes['device_fingerprints']}{user_id}:{fingerprint}"
            device_data = {
                "fingerprint": fingerprint,
                "user_id": str(user_id),
                "device_info": device_info,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
            
            # Check if device already exists and update last_seen
            existing_data = await self.redis_client.get(device_key)
            if existing_data:
                existing = json.loads(existing_data)
                device_data["first_seen"] = existing.get("first_seen", device_data["first_seen"])
            
            await self.redis_client.setex(device_key, ttl_seconds, json.dumps(device_data))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store device fingerprint: {e}")
            return False
    
    async def is_device_known(self, user_id: UUID, fingerprint: str) -> bool:
        """Check if device fingerprint is known for user"""
        try:
            device_key = f"{self.prefixes['device_fingerprints']}{user_id}:{fingerprint}"
            return await self.redis_client.exists(device_key)
            
        except Exception as e:
            self.logger.error(f"Failed to check device fingerprint: {e}")
            return False
    
    # Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        try:
            start_time = datetime.now()
            await self.redis_client.ping()
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "redis_version": info.get("redis_version", "unknown")
            }
            
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Cleanup and Maintenance
    async def cleanup_expired_keys(self):
        """Clean up expired keys and optimize memory usage"""
        try:
            # Get memory usage before cleanup
            info_before = await self.redis_client.info("memory")
            
            # Force cleanup of expired keys
            await self.redis_client.flushdb(async_mode=True)
            
            # Get memory usage after cleanup
            info_after = await self.redis_client.info("memory")
            
            self.logger.info(
                f"Redis cleanup completed. Memory before: {info_before.get('used_memory_human')}, "
                f"after: {info_after.get('used_memory_human')}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup Redis: {e}")


# Context manager for Redis operations
class RedisSecurityContext:
    """Context manager for Redis security operations"""
    
    def __init__(self, redis_manager: SecurityRedisManager):
        self.redis_manager = redis_manager
    
    async def __aenter__(self):
        if not self.redis_manager.redis_client:
            await self.redis_manager.connect()
        return self.redis_manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't close connection here as it might be shared
        pass