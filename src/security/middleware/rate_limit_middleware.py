"""
Rate Limiting Middleware

Advanced rate limiting middleware with Redis backend, sliding window algorithm,
and multiple rate limiting strategies for different endpoints and users.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RateLimitRule:
    """Rate limiting rule configuration"""
    
    def __init__(
        self,
        name: str,
        requests: int,
        window_seconds: int,
        key_func: Optional[callable] = None,
        paths: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        user_types: Optional[List[str]] = None,
        bypass_roles: Optional[List[str]] = None
    ):
        self.name = name
        self.requests = requests
        self.window_seconds = window_seconds
        self.key_func = key_func or self.default_key_func
        self.paths = paths or []
        self.methods = methods or ["GET", "POST", "PUT", "DELETE", "PATCH"]
        self.user_types = user_types or []
        self.bypass_roles = bypass_roles or []
    
    def default_key_func(self, request: Request) -> str:
        """Default key function using IP address"""
        return f"ip:{self._get_client_ip(request)}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def applies_to(self, request: Request) -> bool:
        """Check if rule applies to request"""
        # Check HTTP method
        if request.method not in self.methods:
            return False
        
        # Check path patterns
        if self.paths:
            path = request.url.path
            if not any(path.startswith(pattern) for pattern in self.paths):
                return False
        
        return True
    
    def should_bypass(self, request: Request) -> bool:
        """Check if request should bypass rate limiting"""
        security_context = getattr(request.state, 'security_context', None)
        if not security_context:
            return False
        
        # Check bypass roles
        if self.bypass_roles:
            user_roles = [role.value for role in security_context.roles]
            if any(role in user_roles for role in self.bypass_roles):
                return True
        
        return False


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_windows = defaultdict(list)  # Fallback for in-memory storage
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        current_time: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed using sliding window algorithm
        
        Returns:
            Tuple of (allowed, metadata) where metadata contains:
            - remaining: requests remaining in window
            - reset_time: when window resets
            - retry_after: seconds to wait if rate limited
        """
        if current_time is None:
            current_time = time.time()
        
        if self.redis_client:
            return await self._redis_sliding_window(key, limit, window_seconds, current_time)
        else:
            return await self._memory_sliding_window(key, limit, window_seconds, current_time)
    
    async def _redis_sliding_window(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        current_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based sliding window implementation"""
        try:
            # Use Redis sorted set for sliding window
            pipeline = self.redis_client.pipeline()
            
            # Remove expired entries
            pipeline.zremrangebyscore(key, 0, current_time - window_seconds)
            
            # Count current entries
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipeline.expire(key, window_seconds + 1)
            
            results = await pipeline.execute()
            current_count = results[1]
            
            # Check if under limit
            allowed = current_count < limit
            remaining = max(0, limit - current_count - 1)
            
            if not allowed:
                # Remove the request we just added since it's not allowed
                await self.redis_client.zrem(key, str(current_time))
            
            reset_time = current_time + window_seconds
            retry_after = window_seconds if not allowed else 0
            
            return allowed, {
                "remaining": remaining,
                "reset_time": reset_time,
                "retry_after": retry_after,
                "current_count": current_count
            }
            
        except Exception as e:
            logging.error(f"Redis rate limiting error: {e}")
            # Fallback to allowing the request
            return True, {"remaining": limit, "reset_time": current_time + window_seconds}
    
    async def _memory_sliding_window(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        current_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """In-memory sliding window implementation"""
        # Clean expired entries
        cutoff_time = current_time - window_seconds
        self.local_windows[key] = [
            timestamp for timestamp in self.local_windows[key]
            if timestamp > cutoff_time
        ]
        
        current_count = len(self.local_windows[key])
        allowed = current_count < limit
        
        if allowed:
            self.local_windows[key].append(current_time)
        
        remaining = max(0, limit - current_count - 1)
        reset_time = current_time + window_seconds
        retry_after = window_seconds if not allowed else 0
        
        return allowed, {
            "remaining": remaining,
            "reset_time": reset_time,
            "retry_after": retry_after,
            "current_count": current_count
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced Rate Limiting Middleware with:
    - Multiple rate limiting strategies
    - Redis backend for distributed rate limiting
    - Sliding window algorithm
    - Per-user, per-IP, and per-endpoint limits
    - Rate limit headers
    - Whitelist/blacklist support
    - Burst allowances
    """
    
    def __init__(
        self,
        app,
        redis_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize Redis connection
        self.redis_client = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                asyncio.create_task(self._test_redis_connection())
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        # Initialize rate limiter
        self.rate_limiter = SlidingWindowRateLimiter(self.redis_client)
        
        # Load rate limiting rules
        self.rules = self._load_rate_limit_rules()
        
        # Whitelist/blacklist
        self.ip_whitelist = set(self.config.get("ip_whitelist", []))
        self.ip_blacklist = set(self.config.get("ip_blacklist", []))
        
        # Global settings
        self.enable_rate_limit_headers = self.config.get("enable_headers", True)
        self.default_rate_limit = self.config.get("default_rate_limit", {"requests": 100, "window": 60})
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "whitelisted_requests": 0,
            "blacklisted_requests": 0,
            "rules_applied": defaultdict(int)
        }
    
    def _load_rate_limit_rules(self) -> List[RateLimitRule]:
        """Load rate limiting rules from configuration"""
        rules = []
        
        # Default rules
        rules.extend([
            # Global IP-based rate limit
            RateLimitRule(
                name="global_ip_limit",
                requests=1000,
                window_seconds=3600,  # 1000 requests per hour per IP
                paths=["/api/"]
            ),
            
            # Login endpoint protection
            RateLimitRule(
                name="login_limit",
                requests=5,
                window_seconds=900,  # 5 attempts per 15 minutes per IP
                paths=["/api/v1/auth/login"],
                methods=["POST"]
            ),
            
            # Registration endpoint protection
            RateLimitRule(
                name="registration_limit",
                requests=3,
                window_seconds=3600,  # 3 registrations per hour per IP
                paths=["/api/v1/auth/register"],
                methods=["POST"]
            ),
            
            # Search endpoint limit
            RateLimitRule(
                name="search_limit",
                requests=100,
                window_seconds=60,  # 100 searches per minute per IP
                paths=["/api/v1/search/"],
                methods=["GET", "POST"]
            ),
            
            # Authenticated user limit (more generous)
            RateLimitRule(
                name="authenticated_user_limit",
                requests=2000,
                window_seconds=3600,  # 2000 requests per hour per user
                key_func=self._user_key_func,
                bypass_roles=["admin", "super_admin"]
            ),
            
            # API key limit
            RateLimitRule(
                name="api_key_limit",
                requests=5000,
                window_seconds=3600,  # 5000 requests per hour per API key
                key_func=self._api_key_func
            ),
            
            # Admin endpoints (very restrictive)
            RateLimitRule(
                name="admin_limit",
                requests=100,
                window_seconds=3600,  # 100 requests per hour
                paths=["/api/v1/admin/"],
                key_func=self._user_key_func
            )
        ])
        
        # Load custom rules from config
        custom_rules = self.config.get("custom_rules", [])
        for rule_config in custom_rules:
            rules.append(RateLimitRule(**rule_config))
        
        return rules
    
    def _user_key_func(self, request: Request) -> str:
        """Key function for authenticated users"""
        security_context = getattr(request.state, 'security_context', None)
        if security_context:
            return f"user:{security_context.user_id}"
        return f"ip:{self._get_client_ip(request)}"
    
    def _api_key_func(self, request: Request) -> str:
        """Key function for API key authentication"""
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{api_key[:8]}"  # Use first 8 chars for privacy
        return f"ip:{self._get_client_ip(request)}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            self._stats["total_requests"] += 1
            
            # Check IP blacklist
            if client_ip in self.ip_blacklist:
                self._stats["blacklisted_requests"] += 1
                self.logger.warning(f"Blocked blacklisted IP: {client_ip}")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied"}
                )
            
            # Check IP whitelist (bypass rate limiting)
            if client_ip in self.ip_whitelist:
                self._stats["whitelisted_requests"] += 1
                response = await call_next(request)
                return response
            
            # Apply rate limiting rules
            rate_limit_result = await self._check_rate_limits(request)
            
            if not rate_limit_result["allowed"]:
                self._stats["rate_limited_requests"] += 1
                self.logger.warning(
                    f"Rate limited: {client_ip} - {rate_limit_result['rule']} - "
                    f"{request.method} {request.url.path}"
                )
                
                # Create rate limit response
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Try again in {rate_limit_result['retry_after']} seconds.",
                        "rule": rate_limit_result["rule"]
                    }
                )
                
                # Add rate limit headers
                if self.enable_rate_limit_headers:
                    self._add_rate_limit_headers(response, rate_limit_result)
                
                return response
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to successful responses
            if self.enable_rate_limit_headers and rate_limit_result.get("metadata"):
                self._add_rate_limit_headers(response, rate_limit_result)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Rate limiting middleware error: {e}", exc_info=True)
            # Allow request to proceed on middleware error
            return await call_next(request)
    
    async def _check_rate_limits(self, request: Request) -> Dict[str, Any]:
        """Check all applicable rate limiting rules"""
        current_time = time.time()
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.rules
            if rule.applies_to(request) and not rule.should_bypass(request)
        ]
        
        if not applicable_rules:
            return {"allowed": True}
        
        # Check each rule
        for rule in applicable_rules:
            key = f"rate_limit:{rule.name}:{rule.key_func(request)}"
            
            allowed, metadata = await self.rate_limiter.is_allowed(
                key, rule.requests, rule.window_seconds, current_time
            )
            
            self._stats["rules_applied"][rule.name] += 1
            
            if not allowed:
                return {
                    "allowed": False,
                    "rule": rule.name,
                    "retry_after": metadata.get("retry_after", rule.window_seconds),
                    "metadata": metadata
                }
        
        # If we get here, all rules passed
        # Return metadata from the most restrictive rule
        most_restrictive_rule = min(applicable_rules, key=lambda r: r.requests / r.window_seconds)
        key = f"rate_limit:{most_restrictive_rule.name}:{most_restrictive_rule.key_func(request)}"
        _, metadata = await self.rate_limiter.is_allowed(
            key, most_restrictive_rule.requests, most_restrictive_rule.window_seconds, current_time
        )
        
        return {
            "allowed": True,
            "rule": most_restrictive_rule.name,
            "metadata": metadata
        }
    
    def _add_rate_limit_headers(self, response: Response, rate_limit_result: Dict[str, Any]):
        """Add rate limit headers to response"""
        metadata = rate_limit_result.get("metadata", {})
        
        # Standard rate limit headers
        response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", "unknown"))
        response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(int(metadata.get("reset_time", time.time())))
        
        # Additional headers
        if "rule" in rate_limit_result:
            response.headers["X-RateLimit-Rule"] = rate_limit_result["rule"]
        
        if not rate_limit_result.get("allowed", True):
            response.headers["Retry-After"] = str(int(metadata.get("retry_after", 60)))
    
    async def add_to_whitelist(self, ip_address: str):
        """Add IP address to whitelist"""
        self.ip_whitelist.add(ip_address)
        self.logger.info(f"Added IP to whitelist: {ip_address}")
    
    async def add_to_blacklist(self, ip_address: str):
        """Add IP address to blacklist"""
        self.ip_blacklist.add(ip_address)
        self.logger.info(f"Added IP to blacklist: {ip_address}")
    
    async def remove_from_whitelist(self, ip_address: str):
        """Remove IP address from whitelist"""
        self.ip_whitelist.discard(ip_address)
        self.logger.info(f"Removed IP from whitelist: {ip_address}")
    
    async def remove_from_blacklist(self, ip_address: str):
        """Remove IP address from blacklist"""
        self.ip_blacklist.discard(ip_address)
        self.logger.info(f"Removed IP from blacklist: {ip_address}")
    
    async def reset_rate_limit(self, key: str):
        """Reset rate limit for specific key"""
        if self.redis_client:
            await self.redis_client.delete(f"rate_limit:*:{key}")
        else:
            # Clear from memory storage
            keys_to_remove = [
                k for k in self.rate_limiter.local_windows.keys()
                if key in k
            ]
            for k in keys_to_remove:
                del self.rate_limiter.local_windows[k]
        
        self.logger.info(f"Reset rate limit for key: {key}")
    
    def get_rate_limit_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        return {
            "statistics": dict(self._stats),
            "rules": [
                {
                    "name": rule.name,
                    "requests": rule.requests,
                    "window_seconds": rule.window_seconds,
                    "paths": rule.paths,
                    "methods": rule.methods
                }
                for rule in self.rules
            ],
            "whitelist_size": len(self.ip_whitelist),
            "blacklist_size": len(self.ip_blacklist),
            "redis_connected": self.redis_client is not None
        }
    
    async def cleanup_expired_data(self):
        """Clean up expired rate limit data (for memory storage)"""
        if not self.redis_client:
            current_time = time.time()
            
            for key in list(self.rate_limiter.local_windows.keys()):
                # Keep only recent entries (last hour)
                cutoff_time = current_time - 3600
                self.rate_limiter.local_windows[key] = [
                    timestamp for timestamp in self.rate_limiter.local_windows[key]
                    if timestamp > cutoff_time
                ]
                
                # Remove empty entries
                if not self.rate_limiter.local_windows[key]:
                    del self.rate_limiter.local_windows[key]
            
            self.logger.debug("Cleaned up expired rate limit data")
    
    async def __aenter__(self):
        return self
    
    async def _test_redis_connection(self):
        """Test Redis connection"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                self.logger.info("Redis connection established successfully")
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            self.redis_client = None
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.redis_client:
            await self.redis_client.close()