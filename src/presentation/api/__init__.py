"""
API package for the Rental ML System.

This package contains all production-ready API endpoints and server configuration
for the rental property recommendation system.
"""

from .production_api import app
from .ml_endpoints import ml_router
from .property_endpoints import property_router
from .user_endpoints import user_router
from .admin_endpoints import admin_router
from .monitoring_endpoints import monitoring_router

__all__ = [
    "app",
    "ml_router",
    "property_router", 
    "user_router",
    "admin_router",
    "monitoring_router"
]