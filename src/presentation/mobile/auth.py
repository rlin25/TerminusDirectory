"""
Mobile authentication handler with biometric support and secure token management.

This module provides mobile-specific authentication including biometric authentication,
secure token storage, and mobile device management.
"""

import asyncio
import hashlib
import hmac
import jwt
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from enum import Enum

from fastapi import HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class BiometricType(str, Enum):
    """Supported biometric authentication types"""
    FINGERPRINT = "fingerprint"
    FACE_ID = "face_id"
    VOICE = "voice"
    IRIS = "iris"


class DeviceType(str, Enum):
    """Supported device types"""
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"


class MobileAuthConfig(BaseModel):
    """Mobile authentication configuration"""
    jwt_secret: str = Field(default="your-secret-key", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=30, description="Refresh token expiry")
    biometric_challenge_expire_minutes: int = Field(default=5, description="Biometric challenge expiry")
    max_failed_attempts: int = Field(default=5, description="Max failed auth attempts")
    lockout_duration_minutes: int = Field(default=15, description="Account lockout duration")


class DeviceInfo(BaseModel):
    """Mobile device information"""
    device_id: str = Field(..., description="Unique device identifier")
    device_type: DeviceType = Field(..., description="Device type")
    device_name: str = Field(..., description="Device name")
    app_version: str = Field(..., description="App version")
    os_version: str = Field(..., description="OS version")
    push_token: Optional[str] = Field(None, description="Push notification token")
    biometric_types: List[BiometricType] = Field(default=[], description="Available biometric types")
    last_seen: datetime = Field(default_factory=datetime.utcnow)


class BiometricChallenge(BaseModel):
    """Biometric authentication challenge"""
    challenge_id: str = Field(..., description="Challenge identifier")
    user_id: str = Field(..., description="User identifier")
    device_id: str = Field(..., description="Device identifier")
    biometric_type: BiometricType = Field(..., description="Biometric type")
    challenge_data: str = Field(..., description="Challenge data")
    expires_at: datetime = Field(..., description="Challenge expiry")


class MobileTokens(BaseModel):
    """Mobile authentication tokens"""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    biometric_enabled: bool = Field(default=False, description="Biometric auth available")


class BiometricAuthRequest(BaseModel):
    """Biometric authentication request"""
    challenge_id: str = Field(..., description="Challenge identifier")
    biometric_data: str = Field(..., description="Biometric authentication data")
    device_signature: str = Field(..., description="Device signature")


class MobileAuthHandler:
    """Mobile authentication handler with biometric support"""
    
    def __init__(self):
        self.config = MobileAuthConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.security = HTTPBearer()
        
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the auth handler"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Mobile auth handler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize mobile auth: {e}")
            raise
    
    def generate_tokens(self, user_id: str, device_id: str) -> MobileTokens:
        """Generate access and refresh tokens for mobile"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "user_id": user_id,
            "device_id": device_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=self.config.access_token_expire_minutes)
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user_id,
            "device_id": device_id,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(days=self.config.refresh_token_expire_days)
        }
        
        # Generate tokens
        access_token = jwt.encode(
            access_payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )
        
        return MobileTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60,
            biometric_enabled=True
        )
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is blacklisted
            if await self.is_token_blacklisted(token):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def register_device(self, user_id: str, device_info: DeviceInfo) -> str:
        """Register a mobile device for a user"""
        try:
            # Generate device secret for additional security
            device_secret = secrets.token_urlsafe(32)
            
            # Store device info in Redis
            device_key = f"mobile_device:{user_id}:{device_info.device_id}"
            device_data = {
                "device_info": device_info.model_dump_json(),
                "device_secret": device_secret,
                "registered_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            await self.redis_client.hmset(device_key, device_data)
            await self.redis_client.expire(device_key, 86400 * 365)  # 1 year
            
            logger.info(f"Device registered for user {user_id}: {device_info.device_id}")
            return device_secret
            
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            raise HTTPException(status_code=500, detail="Device registration failed")
    
    async def create_biometric_challenge(
        self,
        user_id: str,
        device_id: str,
        biometric_type: BiometricType
    ) -> BiometricChallenge:
        """Create a biometric authentication challenge"""
        try:
            challenge_id = secrets.token_urlsafe(16)
            challenge_data = secrets.token_urlsafe(32)
            
            challenge = BiometricChallenge(
                challenge_id=challenge_id,
                user_id=user_id,
                device_id=device_id,
                biometric_type=biometric_type,
                challenge_data=challenge_data,
                expires_at=datetime.utcnow() + timedelta(
                    minutes=self.config.biometric_challenge_expire_minutes
                )
            )
            
            # Store challenge in Redis
            challenge_key = f"biometric_challenge:{challenge_id}"
            await self.redis_client.setex(
                challenge_key,
                self.config.biometric_challenge_expire_minutes * 60,
                challenge.model_dump_json()
            )
            
            logger.info(f"Biometric challenge created: {challenge_id}")
            return challenge
            
        except Exception as e:
            logger.error(f"Failed to create biometric challenge: {e}")
            raise HTTPException(status_code=500, detail="Challenge creation failed")
    
    async def verify_biometric_auth(
        self,
        auth_request: BiometricAuthRequest
    ) -> MobileTokens:
        """Verify biometric authentication and generate tokens"""
        try:
            # Get challenge from Redis
            challenge_key = f"biometric_challenge:{auth_request.challenge_id}"
            challenge_data = await self.redis_client.get(challenge_key)
            
            if not challenge_data:
                raise HTTPException(status_code=401, detail="Invalid or expired challenge")
            
            challenge = BiometricChallenge.model_validate_json(challenge_data)
            
            # Verify device signature
            expected_signature = hmac.new(
                challenge.challenge_data.encode(),
                f"{challenge.device_id}:{auth_request.biometric_data}".encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(expected_signature, auth_request.device_signature):
                await self.increment_failed_attempts(challenge.user_id, challenge.device_id)
                raise HTTPException(status_code=401, detail="Biometric verification failed")
            
            # Clean up challenge
            await self.redis_client.delete(challenge_key)
            
            # Generate tokens
            tokens = self.generate_tokens(challenge.user_id, challenge.device_id)
            
            # Update device last seen
            await self.update_device_last_seen(challenge.user_id, challenge.device_id)
            
            logger.info(f"Biometric auth successful for user {challenge.user_id}")
            return tokens
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Biometric verification failed: {e}")
            raise HTTPException(status_code=500, detail="Biometric verification failed")
    
    async def refresh_token(self, refresh_token: str) -> MobileTokens:
        """Refresh access token using refresh token"""
        try:
            payload = await self.verify_token(refresh_token)
            
            if payload.get("type") != "refresh":
                raise HTTPException(status_code=401, detail="Invalid refresh token")
            
            user_id = payload["user_id"]
            device_id = payload["device_id"]
            
            # Check if device is still active
            if not await self.is_device_active(user_id, device_id):
                raise HTTPException(status_code=401, detail="Device no longer active")
            
            # Generate new tokens
            new_tokens = self.generate_tokens(user_id, device_id)
            
            # Blacklist old refresh token
            await self.blacklist_token(refresh_token)
            
            logger.info(f"Token refreshed for user {user_id}")
            return new_tokens
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise HTTPException(status_code=500, detail="Token refresh failed")
    
    async def logout_device(self, user_id: str, device_id: str) -> bool:
        """Logout from specific device"""
        try:
            # Get all active tokens for device
            pattern = f"active_token:{user_id}:{device_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            # Blacklist all tokens
            if keys:
                await self.redis_client.delete(*keys)
            
            # Update device status
            device_key = f"mobile_device:{user_id}:{device_id}"
            await self.redis_client.hset(device_key, "status", "logged_out")
            
            logger.info(f"Device logged out: {user_id}:{device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            blacklist_key = f"blacklisted_token:{token_hash}"
            return await self.redis_client.exists(blacklist_key)
        except Exception:
            return False
    
    async def blacklist_token(self, token: str) -> None:
        """Blacklist a token"""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            blacklist_key = f"blacklisted_token:{token_hash}"
            
            # Blacklist for the remaining token lifetime
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": False}
            )
            
            exp_time = payload.get("exp", 0)
            now = datetime.utcnow().timestamp()
            ttl = max(int(exp_time - now), 0)
            
            if ttl > 0:
                await self.redis_client.setex(blacklist_key, ttl, "1")
                
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
    
    async def increment_failed_attempts(self, user_id: str, device_id: str) -> None:
        """Increment failed authentication attempts"""
        try:
            failed_key = f"failed_attempts:{user_id}:{device_id}"
            attempts = await self.redis_client.incr(failed_key)
            await self.redis_client.expire(failed_key, 3600)  # 1 hour
            
            if attempts >= self.config.max_failed_attempts:
                # Lock device
                lock_key = f"locked_device:{user_id}:{device_id}"
                await self.redis_client.setex(
                    lock_key,
                    self.config.lockout_duration_minutes * 60,
                    "locked"
                )
                logger.warning(f"Device locked due to failed attempts: {user_id}:{device_id}")
                
        except Exception as e:
            logger.error(f"Failed to increment attempts: {e}")
    
    async def is_device_locked(self, user_id: str, device_id: str) -> bool:
        """Check if device is locked"""
        try:
            lock_key = f"locked_device:{user_id}:{device_id}"
            return await self.redis_client.exists(lock_key)
        except Exception:
            return False
    
    async def is_device_active(self, user_id: str, device_id: str) -> bool:
        """Check if device is active"""
        try:
            device_key = f"mobile_device:{user_id}:{device_id}"
            status = await self.redis_client.hget(device_key, "status")
            return status == "active"
        except Exception:
            return False
    
    async def update_device_last_seen(self, user_id: str, device_id: str) -> None:
        """Update device last seen timestamp"""
        try:
            device_key = f"mobile_device:{user_id}:{device_id}"
            await self.redis_client.hset(
                device_key,
                "last_seen",
                datetime.utcnow().isoformat()
            )
        except Exception as e:
            logger.error(f"Failed to update last seen: {e}")
    
    async def get_user_devices(self, user_id: str) -> List[DeviceInfo]:
        """Get all devices for a user"""
        try:
            pattern = f"mobile_device:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            devices = []
            for key in keys:
                device_data = await self.redis_client.hgetall(key)
                if device_data and device_data.get("device_info"):
                    device_info = DeviceInfo.model_validate_json(device_data["device_info"])
                    devices.append(device_info)
            
            return devices
            
        except Exception as e:
            logger.error(f"Failed to get user devices: {e}")
            return []


# Dependency to get current mobile user
async def get_current_mobile_user(
    request: Request,
    authorization: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    x_device_id: Optional[str] = Header(None)
) -> str:
    """Get current mobile user from token"""
    try:
        if not hasattr(request.app.state, 'auth_handler'):
            raise HTTPException(status_code=500, detail="Auth handler not initialized")
        
        auth_handler: MobileAuthHandler = request.app.state.auth_handler
        
        # Verify token
        payload = await auth_handler.verify_token(authorization.credentials)
        
        user_id = payload.get("user_id")
        device_id = payload.get("device_id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        # Verify device if device ID header is provided
        if x_device_id and device_id != x_device_id:
            raise HTTPException(status_code=401, detail="Device mismatch")
        
        # Check if device is locked
        if device_id and await auth_handler.is_device_locked(user_id, device_id):
            raise HTTPException(status_code=423, detail="Device is locked")
        
        return user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mobile auth verification failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


# Optional device verification dependency
async def verify_device(
    request: Request,
    x_device_id: str = Header(...),
    user_id: str = Depends(get_current_mobile_user)
) -> str:
    """Verify device is registered and active"""
    try:
        auth_handler: MobileAuthHandler = request.app.state.auth_handler
        
        if not await auth_handler.is_device_active(user_id, x_device_id):
            raise HTTPException(status_code=401, detail="Device not active")
        
        return x_device_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Device verification failed: {e}")
        raise HTTPException(status_code=401, detail="Device verification failed")