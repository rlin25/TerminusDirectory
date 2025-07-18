"""
Database Integration for Security Components

Provides database persistence for all security-related data including
sessions, API keys, MFA settings, and security events.
"""

import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, JSON, UUID as SQLUUID

from .models import (
    UserSession, APIKey, MFAToken, SecurityEvent, SecurityEventType,
    UserRole, Permission, MFAMethod, ThreatLevel
)


Base = declarative_base()


class UserSessionDB(Base):
    """Database model for user sessions"""
    __tablename__ = "user_sessions"
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(SQLUUID(as_uuid=True), nullable=False, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    last_activity_at = Column(DateTime, default=datetime.now, nullable=False)
    ip_address = Column(String(45))  # IPv6 support
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True, nullable=False)
    device_fingerprint = Column(String(255))
    location_data = Column(JSON)


class APIKeyDB(Base):
    """Database model for API keys"""
    __tablename__ = "api_keys"
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(SQLUUID(as_uuid=True), index=True)
    service_name = Column(String(255), index=True)
    permissions = Column(JSON, default=list)
    rate_limit = Column(Integer)
    ip_whitelist = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    expires_at = Column(DateTime, index=True)
    last_used_at = Column(DateTime)
    is_active = Column(Boolean, default=True, nullable=False)
    usage_count = Column(Integer, default=0)


class MFASettingsDB(Base):
    """Database model for MFA settings"""
    __tablename__ = "mfa_settings"
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(SQLUUID(as_uuid=True), unique=True, nullable=False, index=True)
    totp_secret = Column(String(255))  # Encrypted
    sms_phone = Column(String(20))
    email = Column(String(255))
    backup_codes = Column(JSON)  # Hashed backup codes
    enabled_methods = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class MFATokenDB(Base):
    """Database model for MFA tokens"""
    __tablename__ = "mfa_tokens"
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(SQLUUID(as_uuid=True), nullable=False, index=True)
    token_hash = Column(String(64), nullable=False)
    method = Column(String(20), nullable=False)
    phone_number = Column(String(20))
    email = Column(String(255))
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    verified = Column(Boolean, default=False)
    attempts = Column(Integer, default=0)


class SecurityEventDB(Base):
    """Database model for security events"""
    __tablename__ = "security_events"
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(SQLUUID(as_uuid=True), index=True)
    username = Column(String(255))
    ip_address = Column(String(45), index=True)
    user_agent = Column(Text)
    resource = Column(String(255))
    action = Column(String(255))
    result = Column(String(20), default="success")
    threat_level = Column(String(20), default="low", index=True)
    message = Column(Text)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)


class BlacklistedTokenDB(Base):
    """Database model for blacklisted JWT tokens"""
    __tablename__ = "blacklisted_tokens"
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    jti = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(SQLUUID(as_uuid=True), index=True)
    reason = Column(String(255))
    blacklisted_at = Column(DateTime, default=datetime.now, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)


class SecurityDatabaseManager:
    """
    Database manager for security components
    
    Provides async database operations for all security-related data
    with connection pooling, transaction management, and error handling.
    """
    
    def __init__(self, database_url: str, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.database_url = database_url
        
        # Create async engine
        self.engine = create_async_engine(
            database_url,
            echo=kwargs.get("echo", False),
            pool_size=kwargs.get("pool_size", 10),
            max_overflow=kwargs.get("max_overflow", 20),
            pool_timeout=kwargs.get("pool_timeout", 30),
            pool_recycle=kwargs.get("pool_recycle", 3600),
        )
        
        # Create session factory
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self):
        """Create all security tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Security database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create security tables: {e}")
            raise
    
    async def cleanup_expired_data(self):
        """Clean up expired tokens, sessions, and events"""
        try:
            async with self.async_session() as session:
                now = datetime.now()
                
                # Clean expired sessions
                await session.execute(
                    "DELETE FROM user_sessions WHERE expires_at < :now OR "
                    "(last_activity_at < :inactive_cutoff AND is_active = false)",
                    {"now": now, "inactive_cutoff": now - timedelta(days=7)}
                )
                
                # Clean expired MFA tokens
                await session.execute(
                    "DELETE FROM mfa_tokens WHERE expires_at < :now",
                    {"now": now}
                )
                
                # Clean expired blacklisted tokens
                await session.execute(
                    "DELETE FROM blacklisted_tokens WHERE expires_at < :now",
                    {"now": now}
                )
                
                # Clean old security events (keep 90 days)
                cutoff = now - timedelta(days=90)
                await session.execute(
                    "DELETE FROM security_events WHERE timestamp < :cutoff",
                    {"cutoff": cutoff}
                )
                
                await session.commit()
                self.logger.info("Cleaned up expired security data")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired data: {e}")
            raise
    
    # Session Management
    async def create_session(self, session: UserSession) -> bool:
        """Create a new user session"""
        try:
            async with self.async_session() as db_session:
                db_obj = UserSessionDB(
                    id=session.id,
                    user_id=session.user_id,
                    session_token=session.session_token,
                    created_at=session.created_at,
                    expires_at=session.expires_at,
                    last_activity_at=session.last_activity_at,
                    ip_address=session.ip_address,
                    user_agent=session.user_agent,
                    is_active=session.is_active
                )
                db_session.add(db_obj)
                await db_session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            return False
    
    async def get_session(self, session_token: str) -> Optional[UserSession]:
        """Get session by token"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    "SELECT * FROM user_sessions WHERE session_token = :token AND is_active = true",
                    {"token": session_token}
                )
                row = result.fetchone()
                if row:
                    return UserSession(
                        id=row.id,
                        user_id=row.user_id,
                        session_token=row.session_token,
                        created_at=row.created_at,
                        expires_at=row.expires_at,
                        last_activity_at=row.last_activity_at,
                        ip_address=row.ip_address,
                        user_agent=row.user_agent,
                        is_active=row.is_active
                    )
                return None
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            return None
    
    async def update_session_activity(self, session_token: str) -> bool:
        """Update session last activity"""
        try:
            async with self.async_session() as session:
                await session.execute(
                    "UPDATE user_sessions SET last_activity_at = :now WHERE session_token = :token",
                    {"now": datetime.now(), "token": session_token}
                )
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to update session activity: {e}")
            return False
    
    async def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session"""
        try:
            async with self.async_session() as session:
                await session.execute(
                    "UPDATE user_sessions SET is_active = false WHERE session_token = :token",
                    {"token": session_token}
                )
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate session: {e}")
            return False
    
    async def get_user_sessions(self, user_id: UUID) -> List[UserSession]:
        """Get all active sessions for user"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    "SELECT * FROM user_sessions WHERE user_id = :user_id AND is_active = true "
                    "ORDER BY last_activity_at DESC",
                    {"user_id": user_id}
                )
                sessions = []
                for row in result.fetchall():
                    sessions.append(UserSession(
                        id=row.id,
                        user_id=row.user_id,
                        session_token=row.session_token,
                        created_at=row.created_at,
                        expires_at=row.expires_at,
                        last_activity_at=row.last_activity_at,
                        ip_address=row.ip_address,
                        user_agent=row.user_agent,
                        is_active=row.is_active
                    ))
                return sessions
        except Exception as e:
            self.logger.error(f"Failed to get user sessions: {e}")
            return []
    
    # API Key Management
    async def create_api_key(self, api_key: APIKey) -> bool:
        """Create a new API key"""
        try:
            async with self.async_session() as session:
                key_hash = hashlib.sha256(api_key.key.encode()).hexdigest()
                
                db_obj = APIKeyDB(
                    id=api_key.id,
                    name=api_key.name,
                    key_hash=key_hash,
                    user_id=api_key.user_id,
                    service_name=api_key.service_name,
                    permissions=[p.value for p in api_key.permissions],
                    rate_limit=api_key.rate_limit,
                    ip_whitelist=api_key.ip_whitelist,
                    created_at=api_key.created_at,
                    expires_at=api_key.expires_at,
                    is_active=api_key.is_active
                )
                session.add(db_obj)
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to create API key: {e}")
            return False
    
    async def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    "SELECT * FROM api_keys WHERE key_hash = :hash AND is_active = true",
                    {"hash": key_hash}
                )
                row = result.fetchone()
                if row:
                    return APIKey(
                        id=row.id,
                        name=row.name,
                        key="[REDACTED]",  # Never return the actual key
                        user_id=row.user_id,
                        service_name=row.service_name,
                        permissions=[Permission(p) for p in row.permissions],
                        rate_limit=row.rate_limit,
                        ip_whitelist=row.ip_whitelist,
                        created_at=row.created_at,
                        expires_at=row.expires_at,
                        last_used_at=row.last_used_at,
                        is_active=row.is_active
                    )
                return None
        except Exception as e:
            self.logger.error(f"Failed to get API key: {e}")
            return None
    
    # MFA Management
    async def store_mfa_settings(self, user_id: UUID, settings: Dict[str, Any]) -> bool:
        """Store MFA settings for user"""
        try:
            async with self.async_session() as session:
                # Upsert MFA settings
                await session.execute("""
                    INSERT INTO mfa_settings (user_id, totp_secret, sms_phone, email, backup_codes, enabled_methods)
                    VALUES (:user_id, :totp_secret, :sms_phone, :email, :backup_codes, :enabled_methods)
                    ON CONFLICT (user_id) DO UPDATE SET
                        totp_secret = EXCLUDED.totp_secret,
                        sms_phone = EXCLUDED.sms_phone,
                        email = EXCLUDED.email,
                        backup_codes = EXCLUDED.backup_codes,
                        enabled_methods = EXCLUDED.enabled_methods,
                        updated_at = NOW()
                """, {
                    "user_id": user_id,
                    "totp_secret": settings.get("totp_secret"),
                    "sms_phone": settings.get("sms_phone"),
                    "email": settings.get("email"),
                    "backup_codes": settings.get("backup_codes", []),
                    "enabled_methods": settings.get("enabled_methods", [])
                })
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to store MFA settings: {e}")
            return False
    
    async def get_mfa_settings(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get MFA settings for user"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    "SELECT * FROM mfa_settings WHERE user_id = :user_id",
                    {"user_id": user_id}
                )
                row = result.fetchone()
                if row:
                    return {
                        "totp_secret": row.totp_secret,
                        "sms_phone": row.sms_phone,
                        "email": row.email,
                        "backup_codes": row.backup_codes,
                        "enabled_methods": row.enabled_methods
                    }
                return None
        except Exception as e:
            self.logger.error(f"Failed to get MFA settings: {e}")
            return None
    
    # Security Events
    async def log_security_event(self, event: SecurityEvent) -> bool:
        """Log a security event"""
        try:
            async with self.async_session() as session:
                db_obj = SecurityEventDB(
                    id=event.id,
                    event_type=event.event_type.value,
                    user_id=event.user_id,
                    username=event.username,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    resource=event.resource,
                    action=event.action,
                    result=event.result,
                    threat_level=event.threat_level.value,
                    message=event.message,
                    metadata=event.metadata,
                    timestamp=event.timestamp
                )
                session.add(db_obj)
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
            return False
    
    # JWT Token Blacklist
    async def blacklist_token(self, jti: str, user_id: UUID, expires_at: datetime, reason: str = "logout") -> bool:
        """Add JWT token to blacklist"""
        try:
            async with self.async_session() as session:
                db_obj = BlacklistedTokenDB(
                    jti=jti,
                    user_id=user_id,
                    reason=reason,
                    expires_at=expires_at
                )
                session.add(db_obj)
                await session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to blacklist token: {e}")
            return False
    
    async def is_token_blacklisted(self, jti: str) -> bool:
        """Check if JWT token is blacklisted"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    "SELECT 1 FROM blacklisted_tokens WHERE jti = :jti AND expires_at > NOW()",
                    {"jti": jti}
                )
                return result.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Failed to check token blacklist: {e}")
            return True  # Err on the side of caution
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()