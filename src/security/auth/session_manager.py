"""
Session Manager

Manages user sessions with comprehensive security features including
session tracking, concurrent session limits, and security monitoring.
"""

import asyncio
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4

from .models import (
    UserSession, SecurityEvent, SecurityEventType, 
    ThreatLevel, SecurityConfig
)


class SessionManager:
    """
    Session Manager with comprehensive security features:
    - Secure session token generation
    - Session expiration and cleanup
    - Concurrent session limits
    - Session hijacking detection
    - Device fingerprinting
    - Geographic anomaly detection
    - Session analytics and monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Session storage (in production, use Redis or database)
        self._sessions: Dict[str, UserSession] = {}
        self._user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_tokens
        
        # Session configuration
        self.session_timeout_minutes = self.config.get(
            "session_timeout_minutes",
            SecurityConfig.SESSION_TIMEOUT_MINUTES
        )
        self.max_concurrent_sessions = self.config.get(
            "max_concurrent_sessions",
            SecurityConfig.SESSION_MAX_CONCURRENT
        )
        self.require_secure = self.config.get(
            "require_secure",
            SecurityConfig.SESSION_REQUIRE_SECURE
        )
        
        # Security thresholds
        self.max_location_distance_km = self.config.get("max_location_distance_km", 1000)
        self.suspicious_activity_threshold = self.config.get("suspicious_activity_threshold", 50)
        
        # Session analytics
        self._session_stats = {
            "sessions_created": 0,
            "sessions_expired": 0,
            "sessions_invalidated": 0,
            "concurrent_sessions_exceeded": 0,
            "suspicious_sessions_detected": 0,
            "geographic_anomalies": 0
        }
        
        # Device fingerprints (simplified)
        self._device_fingerprints: Dict[str, Dict[str, Any]] = {}
        
        # IP geolocation cache (in production, use external service)
        self._ip_locations: Dict[str, Dict[str, Any]] = {}
    
    async def create_session(
        self,
        user_id: UUID,
        ip_address: str,
        user_agent: str,
        device_fingerprint: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None
    ) -> UserSession:
        """
        Create a new user session
        
        Args:
            user_id: User ID
            ip_address: Client IP address
            user_agent: Client user agent
            device_fingerprint: Optional device fingerprint
            location: Optional location data
        
        Returns:
            UserSession object
        """
        try:
            user_id_str = str(user_id)
            
            # Check concurrent session limits
            await self._enforce_concurrent_session_limits(user_id_str)
            
            # Detect suspicious activity
            await self._detect_suspicious_session_creation(
                user_id_str, ip_address, user_agent, location
            )
            
            # Create session
            session = UserSession(
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_at=datetime.now() + timedelta(minutes=self.session_timeout_minutes)
            )
            
            # Store session
            session_token = session.session_token
            self._sessions[session_token] = session
            
            # Track user sessions
            if user_id_str not in self._user_sessions:
                self._user_sessions[user_id_str] = set()
            self._user_sessions[user_id_str].add(session_token)
            
            # Store device fingerprint
            if device_fingerprint:
                self._device_fingerprints[session_token] = {
                    "fingerprint": device_fingerprint,
                    "user_id": user_id_str,
                    "created_at": datetime.now()
                }
            
            # Store location
            if location:
                self._ip_locations[ip_address] = location
            elif ip_address not in self._ip_locations:
                # Get location from IP (mock implementation)
                self._ip_locations[ip_address] = await self._get_ip_location(ip_address)
            
            self._session_stats["sessions_created"] += 1
            
            self.logger.info(f"Session created for user {user_id} from {ip_address}")
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to create session for user {user_id}: {e}")
            raise
    
    async def validate_session(self, session_token: str) -> Optional[UserSession]:
        """
        Validate session token and return session if valid
        
        Args:
            session_token: Session token to validate
        
        Returns:
            UserSession if valid, None otherwise
        """
        try:
            session = self._sessions.get(session_token)
            if not session:
                return None
            
            # Check if session is expired
            if session.is_expired():
                await self._expire_session(session_token)
                return None
            
            # Check if session is active
            if not session.is_active:
                return None
            
            # Update last activity
            session.update_activity()
            
            return session
            
        except Exception as e:
            self.logger.error(f"Session validation failed for token {session_token[:8]}...: {e}")
            return None
    
    async def extend_session(
        self,
        session_token: str,
        extension_minutes: Optional[int] = None
    ) -> bool:
        """
        Extend session expiration time
        
        Args:
            session_token: Session token to extend
            extension_minutes: Minutes to extend (defaults to session timeout)
        
        Returns:
            True if extended successfully, False otherwise
        """
        try:
            session = self._sessions.get(session_token)
            if not session or not session.is_active:
                return False
            
            if not extension_minutes:
                extension_minutes = self.session_timeout_minutes
            
            # Extend expiration
            session.expires_at = datetime.now() + timedelta(minutes=extension_minutes)
            session.update_activity()
            
            self.logger.debug(f"Session extended for {extension_minutes} minutes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extend session {session_token[:8]}...: {e}")
            return False
    
    async def invalidate_session(self, session_token: str, reason: str = "logout") -> bool:
        """
        Invalidate a specific session
        
        Args:
            session_token: Session token to invalidate
            reason: Reason for invalidation
        
        Returns:
            True if invalidated successfully, False otherwise
        """
        try:
            session = self._sessions.get(session_token)
            if not session:
                return False
            
            # Mark as inactive
            session.is_active = False
            
            # Remove from user sessions tracking
            user_id_str = str(session.user_id)
            if user_id_str in self._user_sessions:
                self._user_sessions[user_id_str].discard(session_token)
                if not self._user_sessions[user_id_str]:
                    del self._user_sessions[user_id_str]
            
            # Clean up device fingerprint
            self._device_fingerprints.pop(session_token, None)
            
            self._session_stats["sessions_invalidated"] += 1
            
            self.logger.info(f"Session invalidated: {reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate session {session_token[:8]}...: {e}")
            return False
    
    async def invalidate_user_sessions(
        self,
        user_id: UUID,
        exclude_current: bool = False,
        current_session_token: Optional[str] = None
    ) -> int:
        """
        Invalidate all sessions for a user
        
        Args:
            user_id: User ID
            exclude_current: Whether to exclude current session
            current_session_token: Current session token to exclude
        
        Returns:
            Number of sessions invalidated
        """
        try:
            user_id_str = str(user_id)
            user_session_tokens = self._user_sessions.get(user_id_str, set()).copy()
            
            invalidated_count = 0
            
            for session_token in user_session_tokens:
                # Skip current session if requested
                if exclude_current and session_token == current_session_token:
                    continue
                
                success = await self.invalidate_session(session_token, "user_logout_all")
                if success:
                    invalidated_count += 1
            
            self.logger.info(f"Invalidated {invalidated_count} sessions for user {user_id}")
            
            return invalidated_count
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate user sessions for {user_id}: {e}")
            return 0
    
    async def get_user_sessions(self, user_id: UUID) -> List[Dict[str, Any]]:
        """
        Get all active sessions for a user
        
        Args:
            user_id: User ID
        
        Returns:
            List of session information
        """
        try:
            user_id_str = str(user_id)
            user_session_tokens = self._user_sessions.get(user_id_str, set())
            
            sessions = []
            
            for session_token in user_session_tokens:
                session = self._sessions.get(session_token)
                if not session or not session.is_active:
                    continue
                
                # Get location info
                location = self._ip_locations.get(session.ip_address, {})
                
                # Get device fingerprint
                device_info = self._device_fingerprints.get(session_token, {})
                
                sessions.append({
                    "session_id": str(session.id),
                    "session_token": session_token[:8] + "...",  # Masked token
                    "created_at": session.created_at.isoformat(),
                    "expires_at": session.expires_at.isoformat(),
                    "last_activity_at": session.last_activity_at.isoformat(),
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                    "location": {
                        "country": location.get("country"),
                        "city": location.get("city"),
                        "region": location.get("region")
                    },
                    "device_fingerprint": device_info.get("fingerprint", "")[:16] + "..." if device_info.get("fingerprint") else None,
                    "is_current": False  # This would be determined by the calling code
                })
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []
    
    async def detect_session_anomalies(self, session_token: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies in session behavior
        
        Args:
            session_token: Session token to analyze
        
        Returns:
            List of detected anomalies
        """
        try:
            session = self._sessions.get(session_token)
            if not session:
                return []
            
            anomalies = []
            user_id_str = str(session.user_id)
            
            # Check for IP address changes
            user_session_tokens = self._user_sessions.get(user_id_str, set())
            user_ips = set()
            
            for token in user_session_tokens:
                other_session = self._sessions.get(token)
                if other_session and other_session.is_active:
                    user_ips.add(other_session.ip_address)
            
            if len(user_ips) > 1:
                anomalies.append({
                    "type": "multiple_ip_addresses",
                    "severity": "medium",
                    "description": f"User has active sessions from {len(user_ips)} different IP addresses",
                    "details": {"ip_addresses": list(user_ips)}
                })
            
            # Check for geographic anomalies
            current_location = self._ip_locations.get(session.ip_address)
            if current_location:
                for ip in user_ips:
                    if ip != session.ip_address:
                        other_location = self._ip_locations.get(ip)
                        if other_location:
                            distance = self._calculate_distance(current_location, other_location)
                            if distance > self.max_location_distance_km:
                                anomalies.append({
                                    "type": "geographic_anomaly",
                                    "severity": "high",
                                    "description": f"Sessions detected from locations {distance:.0f}km apart",
                                    "details": {
                                        "distance_km": distance,
                                        "locations": [current_location, other_location]
                                    }
                                })
                                self._session_stats["geographic_anomalies"] += 1
            
            # Check for user agent changes
            user_agents = set()
            for token in user_session_tokens:
                other_session = self._sessions.get(token)
                if other_session and other_session.is_active:
                    user_agents.add(other_session.user_agent)
            
            if len(user_agents) > 3:  # Allow some variation for browser updates
                anomalies.append({
                    "type": "multiple_user_agents",
                    "severity": "low",
                    "description": f"User has active sessions from {len(user_agents)} different user agents",
                    "details": {"user_agent_count": len(user_agents)}
                })
            
            if anomalies:
                self._session_stats["suspicious_sessions_detected"] += 1
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to detect session anomalies: {e}")
            return []
    
    async def _enforce_concurrent_session_limits(self, user_id: str):
        """Enforce concurrent session limits for user"""
        if user_id not in self._user_sessions:
            return
        
        user_session_tokens = self._user_sessions[user_id].copy()
        active_sessions = []
        
        # Check which sessions are still active
        for session_token in user_session_tokens:
            session = self._sessions.get(session_token)
            if session and session.is_active and not session.is_expired():
                active_sessions.append((session_token, session))
            else:
                # Clean up expired/inactive sessions
                self._user_sessions[user_id].discard(session_token)
        
        # If at limit, remove oldest sessions
        if len(active_sessions) >= self.max_concurrent_sessions:
            # Sort by creation time
            active_sessions.sort(key=lambda x: x[1].created_at)
            
            # Remove oldest sessions to make room
            sessions_to_remove = len(active_sessions) - self.max_concurrent_sessions + 1
            
            for i in range(sessions_to_remove):
                session_token, _ = active_sessions[i]
                await self.invalidate_session(session_token, "concurrent_limit_exceeded")
                self._session_stats["concurrent_sessions_exceeded"] += 1
    
    async def _detect_suspicious_session_creation(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        location: Optional[Dict[str, Any]]
    ):
        """Detect suspicious session creation patterns"""
        try:
            # Check recent session creation rate
            now = datetime.now()
            recent_sessions = 0
            
            for session_token in self._user_sessions.get(user_id, set()):
                session = self._sessions.get(session_token)
                if session and (now - session.created_at).total_seconds() < 300:  # 5 minutes
                    recent_sessions += 1
            
            if recent_sessions > 5:  # More than 5 sessions in 5 minutes
                await self._log_session_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    user_id,
                    ip_address,
                    "Rapid session creation detected",
                    ThreatLevel.HIGH
                )
            
            # Check for geographic anomalies
            if location:
                user_session_tokens = self._user_sessions.get(user_id, set())
                for session_token in user_session_tokens:
                    session = self._sessions.get(session_token)
                    if session and session.is_active:
                        other_location = self._ip_locations.get(session.ip_address)
                        if other_location:
                            distance = self._calculate_distance(location, other_location)
                            if distance > self.max_location_distance_km:
                                await self._log_session_event(
                                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                                    user_id,
                                    ip_address,
                                    f"Geographic anomaly: {distance:.0f}km distance",
                                    ThreatLevel.HIGH
                                )
                                break
            
        except Exception as e:
            self.logger.error(f"Error detecting suspicious session creation: {e}")
    
    async def _expire_session(self, session_token: str):
        """Mark session as expired and clean up"""
        session = self._sessions.get(session_token)
        if session:
            session.is_active = False
            
            # Remove from user sessions
            user_id_str = str(session.user_id)
            if user_id_str in self._user_sessions:
                self._user_sessions[user_id_str].discard(session_token)
            
            self._session_stats["sessions_expired"] += 1
    
    async def _get_ip_location(self, ip_address: str) -> Dict[str, Any]:
        """Get location information for IP address (mock implementation)"""
        # In production, use a real geolocation service
        return {
            "country": "Unknown",
            "region": "Unknown",
            "city": "Unknown",
            "latitude": 0.0,
            "longitude": 0.0
        }
    
    def _calculate_distance(self, loc1: Dict[str, Any], loc2: Dict[str, Any]) -> float:
        """Calculate distance between two locations in kilometers"""
        try:
            # Simplified distance calculation (use proper geospatial calculation in production)
            lat1, lon1 = loc1.get("latitude", 0), loc1.get("longitude", 0)
            lat2, lon2 = loc2.get("latitude", 0), loc2.get("longitude", 0)
            
            # Haversine formula (simplified)
            import math
            
            R = 6371  # Earth's radius in kilometers
            
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            
            a = (math.sin(delta_lat / 2) ** 2 +
                 math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            return R * c
            
        except Exception:
            return 0.0
    
    async def _log_session_event(
        self,
        event_type: SecurityEventType,
        user_id: str,
        ip_address: str,
        message: str,
        threat_level: ThreatLevel = ThreatLevel.LOW
    ):
        """Log session security event"""
        self.logger.info(
            f"Session event: {event_type.value} - User: {user_id} - IP: {ip_address} - {message}"
        )
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now()
        expired_tokens = []
        
        for session_token, session in self._sessions.items():
            if session.is_expired() or not session.is_active:
                expired_tokens.append(session_token)
        
        for session_token in expired_tokens:
            session = self._sessions[session_token]
            user_id_str = str(session.user_id)
            
            # Remove from sessions
            del self._sessions[session_token]
            
            # Remove from user sessions
            if user_id_str in self._user_sessions:
                self._user_sessions[user_id_str].discard(session_token)
                if not self._user_sessions[user_id_str]:
                    del self._user_sessions[user_id_str]
            
            # Clean up device fingerprint
            self._device_fingerprints.pop(session_token, None)
        
        if expired_tokens:
            self.logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_sessions = len(self._sessions)
        active_sessions = sum(1 for s in self._sessions.values() if s.is_active)
        
        return {
            "statistics": self._session_stats.copy(),
            "current_state": {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "users_with_sessions": len(self._user_sessions),
                "device_fingerprints": len(self._device_fingerprints),
                "ip_locations_cached": len(self._ip_locations)
            },
            "configuration": {
                "session_timeout_minutes": self.session_timeout_minutes,
                "max_concurrent_sessions": self.max_concurrent_sessions,
                "require_secure": self.require_secure
            }
        }