"""
JWT Token Manager

Manages JWT token creation, validation, and refresh functionality with comprehensive security features.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
from uuid import uuid4

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from .models import (
    JWTClaims, SecurityContext, UserRole, Permission, 
    SecurityConfig, AuthenticationResult, SecurityEvent, SecurityEventType
)


class JWTManager:
    """
    JWT Token Manager with advanced security features:
    - RSA256 asymmetric encryption
    - Token rotation and blacklisting
    - Secure key management
    - Audience and issuer validation
    - Token analytics and monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # JWT configuration
        self.algorithm = self.config.get("algorithm", SecurityConfig.JWT_ALGORITHM)
        self.issuer = self.config.get("issuer", "rental-ml-system")
        self.audience = self.config.get("audience", "rental-ml-api")
        
        # Token expiration times
        self.access_token_expire_minutes = self.config.get(
            "access_token_expire_minutes", 
            SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )
        self.refresh_token_expire_days = self.config.get(
            "refresh_token_expire_days",
            SecurityConfig.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        
        # Initialize key pair
        self._private_key = None
        self._public_key = None
        self._generate_key_pair()
        
        # Token blacklist (in production, use Redis)
        self._blacklisted_tokens: set = set()
        
        # Token analytics
        self._token_stats = {
            "issued": 0,
            "verified": 0,
            "expired": 0,
            "invalid": 0,
            "blacklisted": 0
        }
    
    def _generate_key_pair(self):
        """Generate RSA key pair for JWT signing"""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize keys
            self._private_key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            self._public_key = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            self.logger.info("JWT key pair generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate JWT key pair: {e}")
            raise
    
    def create_access_token(
        self, 
        user_id: str, 
        username: str, 
        email: str,
        roles: List[UserRole], 
        permissions: List[Permission],
        session_id: Optional[str] = None,
        mfa_verified: bool = False,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, datetime]:
        """
        Create JWT access token with comprehensive claims
        
        Returns:
            Tuple of (token, expiration_datetime)
        """
        try:
            now = datetime.utcnow()
            expiry = now + timedelta(minutes=self.access_token_expire_minutes)
            jti = str(uuid4())
            
            # Build claims
            claims = JWTClaims(
                sub=user_id,
                iss=self.issuer,
                aud=self.audience,
                exp=int(expiry.timestamp()),
                iat=int(now.timestamp()),
                jti=jti,
                username=username,
                email=email,
                roles=[role.value for role in roles],
                permissions=[perm.value for perm in permissions],
                session_id=session_id,
                mfa_verified=mfa_verified
            )
            
            # Convert to dict
            payload = {
                "sub": claims.sub,
                "iss": claims.iss,
                "aud": claims.aud,
                "exp": claims.exp,
                "iat": claims.iat,
                "jti": claims.jti,
                "username": claims.username,
                "email": claims.email,
                "roles": claims.roles,
                "permissions": claims.permissions,
                "session_id": claims.session_id,
                "mfa_verified": claims.mfa_verified,
                "token_type": "access"
            }
            
            # Add additional claims
            if additional_claims:
                payload.update(additional_claims)
            
            # Sign token
            token = jwt.encode(
                payload,
                self._private_key,
                algorithm=self.algorithm,
                headers={"kid": "rental-ml-key-1"}  # Key ID for rotation
            )
            
            self._token_stats["issued"] += 1
            self.logger.debug(f"Access token created for user {username}")
            
            return token, expiry
            
        except Exception as e:
            self.logger.error(f"Failed to create access token: {e}")
            raise
    
    def create_refresh_token(
        self,
        user_id: str,
        session_id: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, datetime]:
        """
        Create JWT refresh token
        
        Returns:
            Tuple of (token, expiration_datetime)
        """
        try:
            now = datetime.utcnow()
            expiry = now + timedelta(days=self.refresh_token_expire_days)
            jti = str(uuid4())
            
            payload = {
                "sub": user_id,
                "iss": self.issuer,
                "aud": self.audience,
                "exp": int(expiry.timestamp()),
                "iat": int(now.timestamp()),
                "jti": jti,
                "session_id": session_id,
                "token_type": "refresh"
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(
                payload,
                self._private_key,
                algorithm=self.algorithm,
                headers={"kid": "rental-ml-key-1"}
            )
            
            self.logger.debug(f"Refresh token created for user {user_id}")
            return token, expiry
            
        except Exception as e:
            self.logger.error(f"Failed to create refresh token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token
        
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            # Check if token is blacklisted
            if self._is_token_blacklisted(token):
                self._token_stats["blacklisted"] += 1
                self.logger.warning("Attempted to use blacklisted token")
                return None
            
            # Decode and verify token
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True
                }
            )
            
            self._token_stats["verified"] += 1
            return payload
            
        except jwt.ExpiredSignatureError:
            self._token_stats["expired"] += 1
            self.logger.debug("Token has expired")
            return None
            
        except jwt.InvalidTokenError as e:
            self._token_stats["invalid"] += 1
            self.logger.warning(f"Invalid token: {e}")
            return None
            
        except Exception as e:
            self.logger.error(f"Token verification failed: {e}")
            return None
    
    def decode_token_unsafe(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without verification (for debugging/analysis)
        
        Returns:
            Token payload without verification
        """
        try:
            return jwt.decode(
                token,
                options={"verify_signature": False}
            )
        except Exception as e:
            self.logger.error(f"Failed to decode token: {e}")
            return None
    
    def create_security_context(self, token_payload: Dict[str, Any]) -> SecurityContext:
        """Create security context from verified token payload"""
        try:
            roles = [UserRole(role) for role in token_payload.get("roles", [])]
            permissions = {Permission(perm) for perm in token_payload.get("permissions", [])}
            
            expires_at = datetime.fromtimestamp(token_payload["exp"])
            
            return SecurityContext(
                user_id=token_payload["sub"],
                username=token_payload.get("username", ""),
                email=token_payload.get("email", ""),
                roles=roles,
                permissions=permissions,
                session_id=token_payload.get("session_id"),
                mfa_verified=token_payload.get("mfa_verified", False),
                expires_at=expires_at
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create security context: {e}")
            raise
    
    def blacklist_token(self, token: str, reason: str = "logout"):
        """Add token to blacklist"""
        try:
            # Decode to get JTI
            payload = self.decode_token_unsafe(token)
            if payload and "jti" in payload:
                self._blacklisted_tokens.add(payload["jti"])
                self.logger.info(f"Token blacklisted: {reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to blacklist token: {e}")
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        try:
            payload = self.decode_token_unsafe(token)
            if payload and "jti" in payload:
                return payload["jti"] in self._blacklisted_tokens
            return False
            
        except Exception:
            return True  # Consider invalid tokens as blacklisted
    
    def refresh_access_token(
        self, 
        refresh_token: str,
        user_data: Dict[str, Any]
    ) -> Optional[Tuple[str, str, datetime]]:
        """
        Create new access token from refresh token
        
        Returns:
            Tuple of (new_access_token, new_refresh_token, expiry) or None
        """
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token)
            if not payload or payload.get("token_type") != "refresh":
                return None
            
            user_id = payload["sub"]
            session_id = payload.get("session_id")
            
            # Create new access token
            access_token, access_expiry = self.create_access_token(
                user_id=user_id,
                username=user_data["username"],
                email=user_data["email"],
                roles=user_data["roles"],
                permissions=user_data["permissions"],
                session_id=session_id,
                mfa_verified=user_data.get("mfa_verified", False)
            )
            
            # Create new refresh token
            new_refresh_token, _ = self.create_refresh_token(
                user_id=user_id,
                session_id=session_id
            )
            
            # Blacklist old refresh token
            self.blacklist_token(refresh_token, "token_refresh")
            
            self.logger.info(f"Tokens refreshed for user {user_id}")
            return access_token, new_refresh_token, access_expiry
            
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            return None
    
    def validate_token_claims(self, token_payload: Dict[str, Any]) -> bool:
        """Validate token claims for additional security checks"""
        try:
            # Check required claims
            required_claims = ["sub", "iss", "aud", "exp", "iat", "jti"]
            for claim in required_claims:
                if claim not in token_payload:
                    return False
            
            # Check token type
            token_type = token_payload.get("token_type")
            if token_type not in ["access", "refresh"]:
                return False
            
            # Additional custom validations can be added here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Token claims validation failed: {e}")
            return False
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return {
            "statistics": self._token_stats.copy(),
            "blacklisted_count": len(self._blacklisted_tokens),
            "algorithm": self.algorithm,
            "issuer": self.issuer,
            "audience": self.audience,
            "access_token_expiry_minutes": self.access_token_expire_minutes,
            "refresh_token_expiry_days": self.refresh_token_expire_days
        }
    
    def cleanup_expired_blacklist(self):
        """Clean up expired tokens from blacklist (should be called periodically)"""
        # In a real implementation, this would clean up expired JTIs
        # For now, we'll implement a simple size-based cleanup
        if len(self._blacklisted_tokens) > 10000:
            # Keep only the most recent 5000 tokens
            tokens_list = list(self._blacklisted_tokens)
            self._blacklisted_tokens = set(tokens_list[-5000:])
            self.logger.info("Cleaned up blacklisted tokens")
    
    def rotate_keys(self):
        """Rotate JWT signing keys (for scheduled key rotation)"""
        try:
            # In production, this would:
            # 1. Generate new key pair
            # 2. Update key ID
            # 3. Keep old key for verification during transition period
            # 4. Update key in secure storage (Vault, etc.)
            
            old_key_id = "rental-ml-key-1"
            self._generate_key_pair()
            
            self.logger.info(f"JWT keys rotated from {old_key_id}")
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            raise
    
    def export_public_key(self) -> str:
        """Export public key for external verification"""
        return self._public_key.decode('utf-8')
    
    def create_service_token(
        self,
        service_name: str,
        permissions: List[Permission],
        expires_in_hours: int = 24
    ) -> Tuple[str, datetime]:
        """Create service-to-service authentication token"""
        try:
            now = datetime.utcnow()
            expiry = now + timedelta(hours=expires_in_hours)
            jti = str(uuid4())
            
            payload = {
                "sub": f"service:{service_name}",
                "iss": self.issuer,
                "aud": self.audience,
                "exp": int(expiry.timestamp()),
                "iat": int(now.timestamp()),
                "jti": jti,
                "service_name": service_name,
                "permissions": [perm.value for perm in permissions],
                "token_type": "service"
            }
            
            token = jwt.encode(
                payload,
                self._private_key,
                algorithm=self.algorithm,
                headers={"kid": "rental-ml-key-1"}
            )
            
            self.logger.info(f"Service token created for {service_name}")
            return token, expiry
            
        except Exception as e:
            self.logger.error(f"Failed to create service token: {e}")
            raise