"""
OAuth2 and OpenID Connect Manager

Provides OAuth2 authentication with support for Google, Facebook, Apple,
and other identity providers with comprehensive security features.
"""

import asyncio
import base64
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlencode, parse_qs, urlparse
from uuid import UUID, uuid4

import httpx
from jose import jwt, JWTError

from .models import (
    AuthenticationResult, AuthenticationMethod, SecurityContext,
    UserRole, Permission, SecurityEvent, SecurityEventType, ThreatLevel
)


class OAuth2Provider:
    """OAuth2 provider configuration"""
    
    def __init__(
        self,
        name: str,
        client_id: str,
        client_secret: str,
        authorization_url: str,
        token_url: str,
        userinfo_url: str,
        scope: List[str],
        redirect_uri: str
    ):
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_url = authorization_url
        self.token_url = token_url
        self.userinfo_url = userinfo_url
        self.scope = scope
        self.redirect_uri = redirect_uri


class OAuth2Manager:
    """
    OAuth2 and OpenID Connect Manager with comprehensive security features:
    - Multiple provider support (Google, Facebook, Apple, etc.)
    - PKCE (Proof Key for Code Exchange) support
    - State parameter validation
    - ID token verification
    - User information retrieval and mapping
    - Security event logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # OAuth2 providers
        self.providers: Dict[str, OAuth2Provider] = {}
        self._setup_providers()
        
        # Active OAuth2 states (in production, use Redis)
        self._oauth_states: Dict[str, Dict[str, Any]] = {}
        
        # PKCE code verifiers (in production, use Redis)
        self._pkce_verifiers: Dict[str, str] = {}
        
        # HTTP client for OAuth2 requests
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # OAuth2 statistics
        self._oauth_stats = {
            "authorization_requests": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "token_exchanges": 0,
            "userinfo_requests": 0
        }
    
    def _setup_providers(self):
        """Set up OAuth2 providers from configuration"""
        
        # Google OAuth2
        google_config = self.config.get("google", {})
        if google_config.get("client_id"):
            self.providers["google"] = OAuth2Provider(
                name="google",
                client_id=google_config["client_id"],
                client_secret=google_config["client_secret"],
                authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
                token_url="https://oauth2.googleapis.com/token",
                userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
                scope=["openid", "email", "profile"],
                redirect_uri=google_config["redirect_uri"]
            )
        
        # Facebook OAuth2
        facebook_config = self.config.get("facebook", {})
        if facebook_config.get("client_id"):
            self.providers["facebook"] = OAuth2Provider(
                name="facebook",
                client_id=facebook_config["client_id"],
                client_secret=facebook_config["client_secret"],
                authorization_url="https://www.facebook.com/v18.0/dialog/oauth",
                token_url="https://graph.facebook.com/v18.0/oauth/access_token",
                userinfo_url="https://graph.facebook.com/v18.0/me",
                scope=["email", "public_profile"],
                redirect_uri=facebook_config["redirect_uri"]
            )
        
        # Apple OAuth2
        apple_config = self.config.get("apple", {})
        if apple_config.get("client_id"):
            self.providers["apple"] = OAuth2Provider(
                name="apple",
                client_id=apple_config["client_id"],
                client_secret=apple_config["client_secret"],
                authorization_url="https://appleid.apple.com/auth/authorize",
                token_url="https://appleid.apple.com/auth/token",
                userinfo_url="",  # Apple doesn't have a separate userinfo endpoint
                scope=["name", "email"],
                redirect_uri=apple_config["redirect_uri"]
            )
        
        # Microsoft OAuth2
        microsoft_config = self.config.get("microsoft", {})
        if microsoft_config.get("client_id"):
            tenant_id = microsoft_config.get("tenant_id", "common")
            self.providers["microsoft"] = OAuth2Provider(
                name="microsoft",
                client_id=microsoft_config["client_id"],
                client_secret=microsoft_config["client_secret"],
                authorization_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
                token_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                userinfo_url="https://graph.microsoft.com/v1.0/me",
                scope=["openid", "email", "profile"],
                redirect_uri=microsoft_config["redirect_uri"]
            )
    
    def get_authorization_url(
        self,
        provider_name: str,
        state: Optional[str] = None,
        use_pkce: bool = True
    ) -> Dict[str, Any]:
        """
        Generate OAuth2 authorization URL
        
        Args:
            provider_name: OAuth2 provider name
            state: Optional state parameter
            use_pkce: Use PKCE for enhanced security
        
        Returns:
            Dictionary with authorization URL and state information
        """
        try:
            provider = self.providers.get(provider_name)
            if not provider:
                raise ValueError(f"Unknown OAuth2 provider: {provider_name}")
            
            # Generate state parameter if not provided
            if not state:
                state = secrets.token_urlsafe(32)
            
            # Generate PKCE parameters if enabled
            code_verifier = None
            code_challenge = None
            if use_pkce:
                code_verifier = self._generate_code_verifier()
                code_challenge = self._generate_code_challenge(code_verifier)
                self._pkce_verifiers[state] = code_verifier
            
            # Store OAuth2 state
            oauth_state = {
                "provider": provider_name,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(minutes=10),
                "code_verifier": code_verifier,
                "use_pkce": use_pkce
            }
            self._oauth_states[state] = oauth_state
            
            # Build authorization parameters
            auth_params = {
                "client_id": provider.client_id,
                "redirect_uri": provider.redirect_uri,
                "scope": " ".join(provider.scope),
                "response_type": "code",
                "state": state
            }
            
            # Add PKCE parameters
            if use_pkce:
                auth_params.update({
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256"
                })
            
            # Provider-specific parameters
            if provider_name == "google":
                auth_params["access_type"] = "offline"
                auth_params["prompt"] = "consent"
            elif provider_name == "microsoft":
                auth_params["prompt"] = "consent"
            
            # Build authorization URL
            authorization_url = f"{provider.authorization_url}?{urlencode(auth_params)}"
            
            self._oauth_stats["authorization_requests"] += 1
            self.logger.info(f"Generated authorization URL for {provider_name}")
            
            return {
                "authorization_url": authorization_url,
                "state": state,
                "provider": provider_name,
                "expires_at": oauth_state["expires_at"].isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate authorization URL for {provider_name}: {e}")
            raise
    
    async def handle_callback(
        self,
        provider_name: str,
        code: str,
        state: str,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """
        Handle OAuth2 callback and complete authentication
        
        Args:
            provider_name: OAuth2 provider name
            code: Authorization code from provider
            state: State parameter from authorization request
            ip_address: Client IP address
            user_agent: Client user agent
        
        Returns:
            AuthenticationResult with authentication status and tokens
        """
        try:
            # Validate state parameter
            oauth_state = self._oauth_states.get(state)
            if not oauth_state:
                await self._log_oauth_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    provider_name,
                    ip_address,
                    user_agent,
                    "Invalid OAuth2 state parameter",
                    ThreatLevel.HIGH
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid state parameter"
                )
            
            # Check state expiration
            if datetime.now() > oauth_state["expires_at"]:
                del self._oauth_states[state]
                return AuthenticationResult(
                    success=False,
                    error_message="OAuth2 state has expired"
                )
            
            # Verify provider matches
            if oauth_state["provider"] != provider_name:
                return AuthenticationResult(
                    success=False,
                    error_message="Provider mismatch"
                )
            
            provider = self.providers[provider_name]
            
            # Exchange code for tokens
            token_data = await self._exchange_code_for_tokens(
                provider, code, oauth_state
            )
            
            if not token_data:
                self._oauth_stats["failed_authentications"] += 1
                return AuthenticationResult(
                    success=False,
                    error_message="Failed to exchange authorization code"
                )
            
            # Get user information
            user_info = await self._get_user_info(provider, token_data)
            if not user_info:
                self._oauth_stats["failed_authentications"] += 1
                return AuthenticationResult(
                    success=False,
                    error_message="Failed to retrieve user information"
                )
            
            # Create or update user account
            user_data = await self._process_oauth_user(
                provider_name, user_info, token_data
            )
            
            # Create authentication result
            result = await self._create_oauth_authentication_result(
                user_data, provider_name, ip_address, user_agent
            )
            
            # Clean up state
            del self._oauth_states[state]
            if oauth_state.get("code_verifier"):
                self._pkce_verifiers.pop(state, None)
            
            self._oauth_stats["successful_authentications"] += 1
            
            await self._log_oauth_event(
                SecurityEventType.LOGIN_SUCCESS,
                provider_name,
                ip_address,
                user_agent,
                f"OAuth2 authentication successful for {user_info.get('email', 'unknown')}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"OAuth2 callback handling failed: {e}")
            self._oauth_stats["failed_authentications"] += 1
            
            await self._log_oauth_event(
                SecurityEventType.SECURITY_VIOLATION,
                provider_name,
                ip_address,
                user_agent,
                f"OAuth2 callback error: {str(e)}",
                ThreatLevel.HIGH
            )
            
            return AuthenticationResult(
                success=False,
                error_message="OAuth2 authentication failed"
            )
    
    async def _exchange_code_for_tokens(
        self,
        provider: OAuth2Provider,
        code: str,
        oauth_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token"""
        try:
            # Prepare token request
            token_data = {
                "client_id": provider.client_id,
                "client_secret": provider.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": provider.redirect_uri
            }
            
            # Add PKCE code verifier if used
            if oauth_state.get("use_pkce") and oauth_state.get("code_verifier"):
                token_data["code_verifier"] = oauth_state["code_verifier"]
            
            # Make token request
            response = await self.http_client.post(
                provider.token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code != 200:
                self.logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return None
            
            token_response = response.json()
            self._oauth_stats["token_exchanges"] += 1
            
            return token_response
            
        except Exception as e:
            self.logger.error(f"Token exchange error: {e}")
            return None
    
    async def _get_user_info(
        self,
        provider: OAuth2Provider,
        token_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get user information from OAuth2 provider"""
        try:
            access_token = token_data.get("access_token")
            if not access_token:
                return None
            
            # Handle provider-specific user info retrieval
            if provider.name == "apple":
                # Apple provides user info in the ID token
                id_token = token_data.get("id_token")
                if id_token:
                    return self._decode_apple_id_token(id_token)
                return None
            
            # Standard userinfo endpoint
            if not provider.userinfo_url:
                return None
            
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # Provider-specific userinfo parameters
            params = {}
            if provider.name == "facebook":
                params["fields"] = "id,name,email,picture"
            
            response = await self.http_client.get(
                provider.userinfo_url,
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                self.logger.error(f"Userinfo request failed: {response.status_code} - {response.text}")
                return None
            
            user_info = response.json()
            self._oauth_stats["userinfo_requests"] += 1
            
            return user_info
            
        except Exception as e:
            self.logger.error(f"User info retrieval error: {e}")
            return None
    
    def _decode_apple_id_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """Decode Apple ID token (simplified - in production, verify signature)"""
        try:
            # In production, you should verify the JWT signature using Apple's public keys
            payload = jwt.get_unverified_claims(id_token)
            
            return {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "name": payload.get("name"),
                "email_verified": payload.get("email_verified", False)
            }
            
        except JWTError as e:
            self.logger.error(f"Apple ID token decode error: {e}")
            return None
    
    async def _process_oauth_user(
        self,
        provider_name: str,
        user_info: Dict[str, Any],
        token_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process OAuth2 user data and create/update user account"""
        # Normalize user data across providers
        normalized_user = self._normalize_user_data(provider_name, user_info)
        
        # In production, this would:
        # 1. Check if user exists by email or provider ID
        # 2. Create new user if doesn't exist
        # 3. Update existing user's OAuth2 data
        # 4. Link OAuth2 account to existing user
        
        # Mock user data for now
        user_data = {
            "id": str(uuid4()),
            "username": normalized_user["email"].split("@")[0],
            "email": normalized_user["email"],
            "full_name": normalized_user["name"],
            "roles": [UserRole.TENANT],  # Default role for OAuth2 users
            "oauth_provider": provider_name,
            "oauth_id": normalized_user["id"],
            "email_verified": normalized_user.get("email_verified", False),
            "profile_picture": normalized_user.get("picture"),
            "created_at": datetime.now(),
            "last_login": datetime.now()
        }
        
        self.logger.info(f"Processed OAuth2 user: {normalized_user['email']} from {provider_name}")
        
        return user_data
    
    def _normalize_user_data(
        self,
        provider_name: str,
        user_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize user data across different OAuth2 providers"""
        
        if provider_name == "google":
            return {
                "id": user_info.get("id"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "first_name": user_info.get("given_name"),
                "last_name": user_info.get("family_name"),
                "picture": user_info.get("picture"),
                "email_verified": user_info.get("verified_email", False)
            }
        
        elif provider_name == "facebook":
            return {
                "id": user_info.get("id"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "picture": user_info.get("picture", {}).get("data", {}).get("url"),
                "email_verified": True  # Facebook emails are considered verified
            }
        
        elif provider_name == "apple":
            return {
                "id": user_info.get("id"),
                "email": user_info.get("email"),
                "name": user_info.get("name", ""),
                "email_verified": user_info.get("email_verified", False)
            }
        
        elif provider_name == "microsoft":
            return {
                "id": user_info.get("id"),
                "email": user_info.get("mail") or user_info.get("userPrincipalName"),
                "name": user_info.get("displayName"),
                "first_name": user_info.get("givenName"),
                "last_name": user_info.get("surname"),
                "email_verified": True  # Microsoft emails are considered verified
            }
        
        # Default normalization
        return {
            "id": user_info.get("id"),
            "email": user_info.get("email"),
            "name": user_info.get("name", ""),
            "email_verified": False
        }
    
    async def _create_oauth_authentication_result(
        self,
        user_data: Dict[str, Any],
        provider_name: str,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """Create authentication result for OAuth2 user"""
        # This would integrate with the main authentication system
        # For now, return a mock result
        
        from .jwt_manager import JWTManager
        from .session_manager import SessionManager
        
        # In production, these would be injected dependencies
        jwt_manager = JWTManager()
        session_manager = SessionManager()
        
        # Get user permissions
        roles = user_data.get("roles", [UserRole.TENANT])
        permissions = self._get_permissions_for_roles(roles)
        
        # Create session
        session = await session_manager.create_session(
            user_id=user_data["id"],
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Create tokens
        access_token, access_expiry = jwt_manager.create_access_token(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=roles,
            permissions=list(permissions),
            session_id=session.session_token,
            additional_claims={
                "oauth_provider": provider_name,
                "oauth_id": user_data.get("oauth_id"),
                "email_verified": user_data.get("email_verified", False)
            }
        )
        
        refresh_token, _ = jwt_manager.create_refresh_token(
            user_id=user_data["id"],
            session_id=session.session_token
        )
        
        # Create security context
        security_context = SecurityContext(
            user_id=UUID(user_data["id"]),
            username=user_data["username"],
            email=user_data["email"],
            roles=roles,
            permissions=permissions,
            session_id=session.session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            authentication_method=getattr(AuthenticationMethod, f"OAUTH2_{provider_name.upper()}"),
            expires_at=access_expiry
        )
        
        return AuthenticationResult(
            success=True,
            user_id=UUID(user_data["id"]),
            username=user_data["username"],
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=access_expiry,
            security_context=security_context
        )
    
    def _get_permissions_for_roles(self, roles: List[UserRole]) -> set:
        """Get permissions for roles (would integrate with RBAC)"""
        from .models import RolePermissionMapping
        
        all_permissions = set()
        role_mappings = RolePermissionMapping.get_default_mappings()
        
        for role in roles:
            if role in role_mappings:
                all_permissions.update(role_mappings[role])
        
        return all_permissions
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, code_verifier: str) -> str:
        """Generate PKCE code challenge"""
        digest = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    async def _log_oauth_event(
        self,
        event_type: SecurityEventType,
        provider: str,
        ip_address: str,
        user_agent: str,
        message: str,
        threat_level: ThreatLevel = ThreatLevel.LOW
    ):
        """Log OAuth2 security event"""
        self.logger.info(
            f"OAuth2 event: {event_type.value} - {provider} - {message} - IP: {ip_address}"
        )
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported OAuth2 providers"""
        return list(self.providers.keys())
    
    def get_oauth2_statistics(self) -> Dict[str, Any]:
        """Get OAuth2 usage statistics"""
        return {
            "statistics": self._oauth_stats.copy(),
            "active_states": len(self._oauth_states),
            "supported_providers": len(self.providers),
            "pkce_verifiers": len(self._pkce_verifiers)
        }
    
    def cleanup_expired_states(self):
        """Clean up expired OAuth2 states"""
        now = datetime.now()
        expired_states = [
            state for state, data in self._oauth_states.items()
            if now > data["expires_at"]
        ]
        
        for state in expired_states:
            del self._oauth_states[state]
            self._pkce_verifiers.pop(state, None)
        
        if expired_states:
            self.logger.info(f"Cleaned up {len(expired_states)} expired OAuth2 states")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()