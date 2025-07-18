"""
Security Integration Manager

Coordinates all security components including authentication, authorization, rate limiting,
input validation, and security monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .auth.jwt_manager import JWTManager
from .auth.authentication import AuthenticationManager
from .auth.authorization import AuthorizationManager
from .auth.session_manager import SessionManager
from .auth.mfa_manager import MFAManager
from .auth.dependencies import init_security_dependencies
from .auth.rate_limiter import RateLimiter
from .middleware.security_middleware import SecurityMiddleware
from .middleware.input_validation_middleware import InputValidationMiddleware
from .middleware.rate_limit_middleware import RateLimitMiddleware
from .middleware.cors_middleware import CORSMiddleware as CustomCORSMiddleware
from .middleware.security_headers_middleware import SecurityHeadersMiddleware
from .middleware.ddos_protection_middleware import DDoSProtectionMiddleware
from .monitoring.security_monitor import SecurityMonitor
from .monitoring.audit_logger import AuditLogger
from .config import SecurityConfigManager, get_security_config

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Main security manager that coordinates all security components
    """
    
    def __init__(
        self,
        app: FastAPI,
        config_manager: Optional[SecurityConfigManager] = None,
        database_manager: Optional[Any] = None,
        user_repository: Optional[Any] = None
    ):
        self.app = app
        self.config_manager = config_manager or get_security_config()
        self.database_manager = database_manager
        self.user_repository = user_repository
        
        # Security components
        self.jwt_manager: Optional[JWTManager] = None
        self.session_manager: Optional[SessionManager] = None
        self.mfa_manager: Optional[MFAManager] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.authorization_manager: Optional[AuthorizationManager] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.security_monitor: Optional[SecurityMonitor] = None
        self.audit_logger: Optional[AuditLogger] = None
        
        # Middleware components
        self.security_middleware: Optional[SecurityMiddleware] = None
        self.input_validation_middleware: Optional[InputValidationMiddleware] = None
        self.rate_limit_middleware: Optional[RateLimitMiddleware] = None
        self.cors_middleware: Optional[CustomCORSMiddleware] = None
        self.security_headers_middleware: Optional[SecurityHeadersMiddleware] = None
        self.ddos_protection_middleware: Optional[DDoSProtectionMiddleware] = None
        
        # Initialization status
        self._initialized = False
        self._middleware_added = False
        
        logger.info("Security manager created")
    
    async def initialize(self):
        """Initialize all security components"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing security components...")
            
            # Initialize core security components
            await self._initialize_core_components()
            
            # Initialize middleware components
            await self._initialize_middleware_components()
            
            # Initialize monitoring components
            await self._initialize_monitoring_components()
            
            # Initialize security dependencies
            init_security_dependencies(
                self.jwt_manager,
                self.auth_manager,
                self.authorization_manager
            )
            
            self._initialized = True
            logger.info("Security components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize security components: {e}")
            raise
    
    async def _initialize_core_components(self):
        """Initialize core security components"""
        # JWT Manager
        jwt_config = self.config_manager.get_jwt_config()
        self.jwt_manager = JWTManager(jwt_config)
        
        # Session Manager
        session_config = self.config_manager.get_auth_config()["session_policy"]
        self.session_manager = SessionManager(
            config=session_config,
            database_manager=self.database_manager
        )
        
        # MFA Manager
        mfa_config = self.config_manager.get_auth_config()["mfa_policy"]
        self.mfa_manager = MFAManager(
            config=mfa_config,
            database_manager=self.database_manager
        )
        
        # Authorization Manager
        self.authorization_manager = AuthorizationManager(
            config=self.config_manager.get_auth_config()
        )
        
        # Authentication Manager
        auth_config = self.config_manager.get_auth_config()
        self.auth_manager = AuthenticationManager(
            jwt_manager=self.jwt_manager,
            mfa_manager=self.mfa_manager,
            session_manager=self.session_manager,
            user_repository=self.user_repository,
            database_manager=self.database_manager,
            config=auth_config
        )
        
        # Rate Limiter
        rate_limit_config = self.config_manager.get_rate_limit_config()
        self.rate_limiter = RateLimiter(rate_limit_config)
        
        logger.info("Core security components initialized")
    
    async def _initialize_middleware_components(self):
        """Initialize middleware components"""
        # Security Middleware
        self.security_middleware = SecurityMiddleware(
            app=self.app,
            jwt_manager=self.jwt_manager,
            auth_manager=self.auth_manager,
            authorization_manager=self.authorization_manager,
            config={
                "public_paths": self.config_manager.settings.public_paths,
                "protected_paths": self.config_manager.settings.protected_paths,
                "admin_paths": self.config_manager.settings.admin_paths,
                "enable_performance_monitoring": True,
                "slow_request_threshold_ms": 1000
            }
        )
        
        # Input Validation Middleware
        input_validation_config = self.config_manager.get_input_validation_config()
        self.input_validation_middleware = InputValidationMiddleware(
            app=self.app,
            config=input_validation_config
        )
        
        # Rate Limit Middleware
        self.rate_limit_middleware = RateLimitMiddleware(
            app=self.app,
            rate_limiter=self.rate_limiter,
            config=self.config_manager.get_rate_limit_config()
        )
        
        # CORS Middleware
        cors_config = self.config_manager.get_cors_config()
        self.cors_middleware = CustomCORSMiddleware(
            app=self.app,
            config=cors_config
        )
        
        # Security Headers Middleware
        headers_config = self.config_manager.get_security_headers_config()
        self.security_headers_middleware = SecurityHeadersMiddleware(
            app=self.app,
            config=headers_config
        )
        
        # DDoS Protection Middleware
        self.ddos_protection_middleware = DDoSProtectionMiddleware(
            app=self.app,
            rate_limiter=self.rate_limiter,
            config={
                "enable_protection": True,
                "request_threshold": 1000,
                "time_window_seconds": 60,
                "block_duration_seconds": 300
            }
        )
        
        logger.info("Middleware components initialized")
    
    async def _initialize_monitoring_components(self):
        """Initialize monitoring components"""
        # Security Monitor
        self.security_monitor = SecurityMonitor(
            config={
                "enable_monitoring": self.config_manager.settings.enable_security_monitoring,
                "threat_threshold": self.config_manager.settings.suspicious_activity_threshold,
                "retention_days": self.config_manager.settings.security_event_retention_days
            },
            database_manager=self.database_manager
        )
        
        # Audit Logger
        self.audit_logger = AuditLogger(
            config={
                "enable_logging": self.config_manager.settings.enable_audit_logging,
                "retention_days": self.config_manager.settings.security_event_retention_days
            },
            database_manager=self.database_manager
        )
        
        logger.info("Monitoring components initialized")
    
    def add_middleware(self):
        """Add all middleware to the FastAPI application"""
        if self._middleware_added:
            return
        
        if not self._initialized:
            raise RuntimeError("Security manager not initialized. Call initialize() first.")
        
        try:
            logger.info("Adding security middleware to FastAPI application...")
            
            # Add middleware in correct order (reverse order of execution)
            
            # 1. Security headers (last to execute, first to add)
            if self.security_headers_middleware:
                self.app.add_middleware(SecurityHeadersMiddleware, 
                                       config=self.config_manager.get_security_headers_config())
            
            # 2. CORS middleware
            cors_config = self.config_manager.get_cors_config()
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_config["allow_origins"],
                allow_credentials=cors_config["allow_credentials"],
                allow_methods=cors_config["allow_methods"],
                allow_headers=cors_config["allow_headers"],
                expose_headers=cors_config["expose_headers"]
            )
            
            # 3. Main security middleware
            if self.security_middleware:
                self.app.add_middleware(
                    SecurityMiddleware,
                    jwt_manager=self.jwt_manager,
                    auth_manager=self.auth_manager,
                    authorization_manager=self.authorization_manager,
                    config={
                        "public_paths": self.config_manager.settings.public_paths,
                        "protected_paths": self.config_manager.settings.protected_paths,
                        "admin_paths": self.config_manager.settings.admin_paths,
                        "enable_performance_monitoring": True,
                        "slow_request_threshold_ms": 1000
                    }
                )
            
            # 4. Rate limiting middleware
            if self.rate_limit_middleware:
                self.app.add_middleware(
                    RateLimitMiddleware,
                    rate_limiter=self.rate_limiter,
                    config=self.config_manager.get_rate_limit_config()
                )
            
            # 5. Input validation middleware
            if self.input_validation_middleware:
                self.app.add_middleware(
                    InputValidationMiddleware,
                    config=self.config_manager.get_input_validation_config()
                )
            
            # 6. DDoS protection middleware (first to execute)
            if self.ddos_protection_middleware:
                self.app.add_middleware(
                    DDoSProtectionMiddleware,
                    rate_limiter=self.rate_limiter,
                    config={
                        "enable_protection": True,
                        "request_threshold": 1000,
                        "time_window_seconds": 60,
                        "block_duration_seconds": 300
                    }
                )
            
            self._middleware_added = True
            logger.info("Security middleware added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add security middleware: {e}")
            raise
    
    def add_auth_router(self):
        """Add authentication router to the application"""
        from ..application.api.routers.auth_router import router as auth_router, init_auth_router
        
        # Initialize the router with dependencies
        init_auth_router(self.auth_manager, self.rate_limiter)
        
        # Add router to app
        self.app.include_router(auth_router, prefix="/api/v1")
        
        logger.info("Authentication router added")
    
    def add_security_endpoints(self):
        """Add security management endpoints"""
        from ..application.api.routers.security_router import router as security_router, init_security_router
        
        # Initialize the router with dependencies
        init_security_router(
            self.auth_manager,
            self.security_monitor,
            self.audit_logger,
            self.rate_limiter
        )
        
        # Add router to app
        self.app.include_router(security_router, prefix="/api/v1")
        
        logger.info("Security management endpoints added")
    
    async def setup_background_tasks(self):
        """Setup background tasks for security maintenance"""
        if not self._initialized:
            raise RuntimeError("Security manager not initialized")
        
        try:
            # Setup periodic cleanup tasks
            asyncio.create_task(self._periodic_cleanup())
            
            # Setup security monitoring tasks
            if self.security_monitor:
                asyncio.create_task(self._periodic_security_monitoring())
            
            # Setup audit log rotation
            if self.audit_logger:
                asyncio.create_task(self._periodic_audit_log_rotation())
            
            logger.info("Background security tasks started")
            
        except Exception as e:
            logger.error(f"Failed to setup background tasks: {e}")
            raise
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired tokens, sessions, etc."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup expired authentication data
                if self.auth_manager:
                    self.auth_manager.cleanup_expired_data()
                
                # Cleanup expired sessions
                if self.session_manager:
                    await self.session_manager.cleanup_expired_sessions()
                
                # Cleanup rate limiter data
                if self.rate_limiter:
                    self.rate_limiter.cleanup_expired()
                
                # Cleanup JWT blacklist
                if self.jwt_manager:
                    self.jwt_manager.cleanup_expired_blacklist()
                
                logger.info("Periodic security cleanup completed")
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _periodic_security_monitoring(self):
        """Periodic security monitoring and threat detection"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if self.security_monitor:
                    await self.security_monitor.analyze_security_events()
                
                logger.debug("Periodic security monitoring completed")
                
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
    
    async def _periodic_audit_log_rotation(self):
        """Periodic audit log rotation"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run every day
                
                if self.audit_logger:
                    await self.audit_logger.rotate_logs()
                
                logger.info("Periodic audit log rotation completed")
                
            except Exception as e:
                logger.error(f"Error in audit log rotation: {e}")
    
    async def shutdown(self):
        """Shutdown security components"""
        try:
            logger.info("Shutting down security components...")
            
            # Shutdown monitoring components
            if self.security_monitor:
                await self.security_monitor.shutdown()
            
            if self.audit_logger:
                await self.audit_logger.shutdown()
            
            # Shutdown core components
            if self.session_manager:
                await self.session_manager.shutdown()
            
            logger.info("Security components shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during security shutdown: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            "initialized": self._initialized,
            "middleware_added": self._middleware_added,
            "components": {
                "jwt_manager": self.jwt_manager is not None,
                "auth_manager": self.auth_manager is not None,
                "authorization_manager": self.authorization_manager is not None,
                "rate_limiter": self.rate_limiter is not None,
                "security_monitor": self.security_monitor is not None,
                "audit_logger": self.audit_logger is not None,
            },
            "middleware": {
                "security_middleware": self.security_middleware is not None,
                "input_validation": self.input_validation_middleware is not None,
                "rate_limiting": self.rate_limit_middleware is not None,
                "cors": self.cors_middleware is not None,
                "security_headers": self.security_headers_middleware is not None,
                "ddos_protection": self.ddos_protection_middleware is not None,
            },
            "configuration": {
                "environment": self.config_manager.environment,
                "jwt_algorithm": self.config_manager.settings.jwt_algorithm,
                "rate_limiting_enabled": self.config_manager.settings.enable_rate_limiting,
                "input_validation_enabled": self.config_manager.settings.enable_input_validation,
                "security_monitoring_enabled": self.config_manager.settings.enable_security_monitoring,
                "audit_logging_enabled": self.config_manager.settings.enable_audit_logging,
            }
        }
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics from all components"""
        stats = {
            "timestamp": logger.name,
            "environment": self.config_manager.environment,
            "components": {}
        }
        
        # JWT Manager statistics
        if self.jwt_manager:
            stats["components"]["jwt"] = self.jwt_manager.get_token_statistics()
        
        # Authentication Manager statistics
        if self.auth_manager:
            stats["components"]["authentication"] = self.auth_manager.get_authentication_statistics()
        
        # Rate Limiter statistics
        if self.rate_limiter:
            stats["components"]["rate_limiting"] = self.rate_limiter.get_statistics()
        
        # Security Middleware statistics
        if self.security_middleware:
            stats["components"]["security_middleware"] = self.security_middleware.get_security_statistics()
        
        # Input Validation statistics
        if self.input_validation_middleware:
            stats["components"]["input_validation"] = self.input_validation_middleware.get_validation_statistics()
        
        # Security Monitor statistics
        if self.security_monitor:
            stats["components"]["security_monitoring"] = self.security_monitor.get_monitoring_statistics()
        
        return stats


# Global security manager instance
security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    if security_manager is None:
        raise RuntimeError("Security manager not initialized")
    return security_manager


def init_security_manager(
    app: FastAPI,
    config_manager: Optional[SecurityConfigManager] = None,
    database_manager: Optional[Any] = None,
    user_repository: Optional[Any] = None
) -> SecurityManager:
    """Initialize global security manager"""
    global security_manager
    security_manager = SecurityManager(
        app=app,
        config_manager=config_manager,
        database_manager=database_manager,
        user_repository=user_repository
    )
    return security_manager