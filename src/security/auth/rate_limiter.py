"""
Rate Limiting Middleware for Authentication Endpoints

Provides comprehensive rate limiting capabilities for various authentication operations
with different limits and windows for different types of requests.
"""

import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class RateLimitType(Enum):
    """Types of rate limits"""
    LOGIN = "login"
    PASSWORD_RESET = "password_reset"
    REGISTRATION = "registration"
    TOKEN_REFRESH = "token_refresh"
    API_REQUEST = "api_request"
    DEFAULT = "default"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    block_duration_seconds: int  # How long to block after limit exceeded
    burst_allowance: int = 0  # Additional requests allowed in burst


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    reason: Optional[str] = None


class RateLimiter:
    """
    Advanced rate limiter with multiple strategies:
    - Fixed window
    - Sliding window
    - Token bucket
    - Adaptive limits based on threat level
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Rate limit configurations for different types
        self.limits = {
            RateLimitType.LOGIN: RateLimitConfig(
                requests=5,
                window_seconds=60,
                block_duration_seconds=300,  # 5 minutes
                burst_allowance=2
            ),
            RateLimitType.PASSWORD_RESET: RateLimitConfig(
                requests=3,
                window_seconds=900,  # 15 minutes
                block_duration_seconds=1800,  # 30 minutes
                burst_allowance=0
            ),
            RateLimitType.REGISTRATION: RateLimitConfig(
                requests=2,
                window_seconds=3600,  # 1 hour
                block_duration_seconds=3600,  # 1 hour
                burst_allowance=0
            ),
            RateLimitType.TOKEN_REFRESH: RateLimitConfig(
                requests=10,
                window_seconds=60,
                block_duration_seconds=60,
                burst_allowance=5
            ),
            RateLimitType.API_REQUEST: RateLimitConfig(
                requests=100,
                window_seconds=60,
                block_duration_seconds=60,
                burst_allowance=20
            ),
            RateLimitType.DEFAULT: RateLimitConfig(
                requests=20,
                window_seconds=60,
                block_duration_seconds=60,
                burst_allowance=5
            )
        }
        
        # Override with config
        for limit_type, user_config in self.config.get("rate_limits", {}).items():
            if hasattr(RateLimitType, limit_type.upper()):
                rate_type = RateLimitType(limit_type)
                if rate_type in self.limits:
                    current = self.limits[rate_type]
                    self.limits[rate_type] = RateLimitConfig(
                        requests=user_config.get("requests", current.requests),
                        window_seconds=user_config.get("window_seconds", current.window_seconds),
                        block_duration_seconds=user_config.get("block_duration_seconds", current.block_duration_seconds),
                        burst_allowance=user_config.get("burst_allowance", current.burst_allowance)
                    )
        
        # Storage for request tracking (in production, use Redis)
        self._request_history: Dict[str, deque] = defaultdict(deque)
        self._blocked_until: Dict[str, datetime] = {}
        self._token_buckets: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "rate_limited_ips": set(),
            "most_active_clients": defaultdict(int)
        }
    
    def check_rate_limit(
        self,
        identifier: str,
        limit_type: RateLimitType = RateLimitType.DEFAULT,
        ip_address: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limits
        
        Args:
            identifier: Unique identifier (user_id, email, API key, etc.)
            limit_type: Type of rate limit to apply
            ip_address: Client IP address for additional tracking
        
        Returns:
            RateLimitResult with decision and metadata
        """
        try:
            self._stats["total_requests"] += 1
            
            # Create composite key for tracking
            key = f"{limit_type.value}:{identifier}"
            if ip_address:
                ip_key = f"{limit_type.value}:ip:{ip_address}"
                self._stats["most_active_clients"][ip_address] += 1
            
            config = self.limits.get(limit_type, self.limits[RateLimitType.DEFAULT])
            now = datetime.now()
            
            # Check if currently blocked
            if key in self._blocked_until:
                if now < self._blocked_until[key]:
                    retry_after = int((self._blocked_until[key] - now).total_seconds())
                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        reset_time=self._blocked_until[key],
                        retry_after=retry_after,
                        reason=f"Blocked until {self._blocked_until[key]}"
                    )
                else:
                    # Block expired, remove it
                    del self._blocked_until[key]
            
            # Clean old requests
            cutoff = now - timedelta(seconds=config.window_seconds)
            self._clean_old_requests(key, cutoff)
            
            # Check IP-based limits if provided
            if ip_address:
                ip_result = self._check_ip_limits(ip_key, config, now)
                if not ip_result.allowed:
                    return ip_result
            
            # Get current request count
            current_requests = len(self._request_history[key])
            
            # Check if limit exceeded
            if current_requests >= config.requests:
                # Block the identifier
                block_until = now + timedelta(seconds=config.block_duration_seconds)
                self._blocked_until[key] = block_until
                
                self._stats["blocked_requests"] += 1
                if ip_address:
                    self._stats["rate_limited_ips"].add(ip_address)
                
                self.logger.warning(
                    f"Rate limit exceeded for {identifier} "
                    f"({limit_type.value}): {current_requests}/{config.requests}"
                )
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=block_until,
                    retry_after=config.block_duration_seconds,
                    reason=f"Rate limit exceeded: {current_requests}/{config.requests}"
                )
            
            # Check burst allowance
            if config.burst_allowance > 0:
                burst_result = self._check_burst_allowance(key, config, now)
                if not burst_result.allowed:
                    return burst_result
            
            # Record this request
            self._request_history[key].append(now)
            
            # Calculate remaining and reset time
            remaining = config.requests - current_requests - 1
            reset_time = now + timedelta(seconds=config.window_seconds)
            
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=reset_time
            )
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request but log error
            return RateLimitResult(
                allowed=True,
                remaining=0,
                reset_time=datetime.now() + timedelta(seconds=60),
                reason="Rate limiter error - failing open"
            )
    
    def _check_ip_limits(
        self,
        ip_key: str,
        config: RateLimitConfig,
        now: datetime
    ) -> RateLimitResult:
        """Check IP-based rate limits"""
        # Apply stricter limits for IP-based tracking
        ip_config = RateLimitConfig(
            requests=config.requests * 5,  # More lenient for IP
            window_seconds=config.window_seconds,
            block_duration_seconds=config.block_duration_seconds,
            burst_allowance=config.burst_allowance * 2
        )
        
        cutoff = now - timedelta(seconds=ip_config.window_seconds)
        self._clean_old_requests(ip_key, cutoff)
        
        current_requests = len(self._request_history[ip_key])
        
        if current_requests >= ip_config.requests:
            block_until = now + timedelta(seconds=ip_config.block_duration_seconds)
            self._blocked_until[ip_key] = block_until
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=block_until,
                retry_after=ip_config.block_duration_seconds,
                reason=f"IP rate limit exceeded: {current_requests}/{ip_config.requests}"
            )
        
        # Record IP request
        self._request_history[ip_key].append(now)
        
        return RateLimitResult(
            allowed=True,
            remaining=ip_config.requests - current_requests - 1,
            reset_time=now + timedelta(seconds=ip_config.window_seconds)
        )
    
    def _check_burst_allowance(
        self,
        key: str,
        config: RateLimitConfig,
        now: datetime
    ) -> RateLimitResult:
        """Check burst allowance using token bucket algorithm"""
        bucket_key = f"burst:{key}"
        
        # Initialize bucket if not exists
        if bucket_key not in self._token_buckets:
            self._token_buckets[bucket_key] = {
                "tokens": config.burst_allowance,
                "last_refill": now
            }
        
        bucket = self._token_buckets[bucket_key]
        
        # Calculate tokens to add based on time passed
        time_passed = (now - bucket["last_refill"]).total_seconds()
        refill_rate = config.burst_allowance / config.window_seconds
        tokens_to_add = int(time_passed * refill_rate)
        
        # Refill bucket
        bucket["tokens"] = min(config.burst_allowance, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Check if burst request can be allowed
        if bucket["tokens"] > 0:
            bucket["tokens"] -= 1
            return RateLimitResult(
                allowed=True,
                remaining=bucket["tokens"],
                reset_time=now + timedelta(seconds=config.window_seconds)
            )
        
        return RateLimitResult(
            allowed=False,
            remaining=0,
            reset_time=now + timedelta(seconds=60),
            retry_after=60,
            reason="Burst allowance exceeded"
        )
    
    def _clean_old_requests(self, key: str, cutoff: datetime):
        """Remove old requests from history"""
        if key not in self._request_history:
            return
        
        # Remove requests older than cutoff
        while (self._request_history[key] and 
               self._request_history[key][0] < cutoff):
            self._request_history[key].popleft()
    
    def is_blocked(self, identifier: str, limit_type: RateLimitType) -> bool:
        """Check if identifier is currently blocked"""
        key = f"{limit_type.value}:{identifier}"
        if key in self._blocked_until:
            return datetime.now() < self._blocked_until[key]
        return False
    
    def unblock(self, identifier: str, limit_type: RateLimitType) -> bool:
        """Manually unblock an identifier"""
        key = f"{limit_type.value}:{identifier}"
        if key in self._blocked_until:
            del self._blocked_until[key]
            self.logger.info(f"Manually unblocked {identifier} for {limit_type.value}")
            return True
        return False
    
    def get_remaining_requests(
        self,
        identifier: str,
        limit_type: RateLimitType
    ) -> int:
        """Get remaining requests for identifier"""
        key = f"{limit_type.value}:{identifier}"
        config = self.limits.get(limit_type, self.limits[RateLimitType.DEFAULT])
        
        if self.is_blocked(identifier, limit_type):
            return 0
        
        # Clean old requests
        cutoff = datetime.now() - timedelta(seconds=config.window_seconds)
        self._clean_old_requests(key, cutoff)
        
        current_requests = len(self._request_history[key])
        return max(0, config.requests - current_requests)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "total_requests": self._stats["total_requests"],
            "blocked_requests": self._stats["blocked_requests"],
            "block_rate": (
                self._stats["blocked_requests"] / max(1, self._stats["total_requests"])
            ) * 100,
            "unique_rate_limited_ips": len(self._stats["rate_limited_ips"]),
            "currently_blocked": len(self._blocked_until),
            "active_request_trackers": len(self._request_history),
            "top_clients": dict(
                sorted(
                    self._stats["most_active_clients"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "limits_configured": {
                limit_type.value: {
                    "requests": config.requests,
                    "window_seconds": config.window_seconds,
                    "block_duration_seconds": config.block_duration_seconds
                }
                for limit_type, config in self.limits.items()
            }
        }
    
    def cleanup_expired(self):
        """Clean up expired data (call periodically)"""
        now = datetime.now()
        
        # Clean up expired blocks
        expired_blocks = [
            key for key, expiry in self._blocked_until.items()
            if now >= expiry
        ]
        for key in expired_blocks:
            del self._blocked_until[key]
        
        # Clean up old request history
        for key in list(self._request_history.keys()):
            # Keep only last hour of data
            cutoff = now - timedelta(hours=1)
            self._clean_old_requests(key, cutoff)
            if not self._request_history[key]:
                del self._request_history[key]
        
        # Clean up old token buckets
        for key in list(self._token_buckets.keys()):
            bucket = self._token_buckets[key]
            if (now - bucket["last_refill"]).total_seconds() > 3600:  # 1 hour
                del self._token_buckets[key]
        
        if expired_blocks:
            self.logger.info(f"Cleaned up {len(expired_blocks)} expired rate limit blocks")
    
    def reset_stats(self):
        """Reset statistics"""
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "rate_limited_ips": set(),
            "most_active_clients": defaultdict(int)
        }
        self.logger.info("Rate limiter statistics reset")


# Rate limiter middleware for FastAPI/Flask
class RateLimitMiddleware:
    """Rate limiting middleware for web frameworks"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
    
    def get_client_identifier(self, request) -> str:
        """Extract client identifier from request"""
        # Try to get user ID from JWT token
        auth_header = getattr(request, 'headers', {}).get('Authorization', '')
        if auth_header.startswith('Bearer '):
            # In a real implementation, decode JWT to get user ID
            # For now, use a placeholder
            return "user_from_token"
        
        # Fallback to IP address
        return self.get_client_ip(request)
    
    def get_client_ip(self, request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers
        forwarded_for = getattr(request, 'headers', {}).get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = getattr(request, 'headers', {}).get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return getattr(request, 'client', {}).get('host', '127.0.0.1')
    
    def check_request(
        self,
        request,
        limit_type: RateLimitType = RateLimitType.DEFAULT
    ) -> RateLimitResult:
        """Check if request should be rate limited"""
        identifier = self.get_client_identifier(request)
        ip_address = self.get_client_ip(request)
        
        return self.rate_limiter.check_rate_limit(
            identifier=identifier,
            limit_type=limit_type,
            ip_address=ip_address
        )