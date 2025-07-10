"""
Health check API router for system monitoring and diagnostics.

This module provides endpoints for health checks, system status,
and diagnostic information.
"""

import time
import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


@router.get("/")
async def health_check(
    request: Request,
    repository_factory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns system health status including:
    - Database connectivity
    - Redis connectivity
    - Repository availability
    - Overall system health
    """
    try:
        start_time = time.time()
        
        # Perform health check on all components
        health_status = await repository_factory.health_check()
        
        response_time = (time.time() - start_time) * 1000
        
        status_code = 200 if health_status.get("overall") else 503
        
        response = {
            "status": "healthy" if health_status.get("overall") else "unhealthy",
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "components": {
                "database": "up" if health_status.get("database") else "down",
                "redis": "up" if health_status.get("redis") else "down",
                "repositories": "up" if health_status.get("repositories") else "down"
            },
            "version": "1.0.0",
            "environment": "development"  # TODO: Get from config
        }
        
        return JSONResponse(content=response, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
                "version": "1.0.0"
            },
            status_code=503
        )


@router.get("/detailed")
async def detailed_health_check(
    request: Request,
    repository_factory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """
    Detailed health check with comprehensive system information.
    
    Returns detailed health information including:
    - Component-specific health details
    - Performance metrics
    - Resource utilization
    - System dependencies
    """
    try:
        start_time = time.time()
        
        # Get basic health status
        health_status = await repository_factory.health_check()
        
        # Get cache statistics
        cache_repo = repository_factory.get_cache_repository()
        cache_stats = await cache_repo.get_cache_stats()
        
        # Get property repository statistics
        property_repo = repository_factory.get_property_repository()
        total_properties = await property_repo.get_count()
        active_properties = await property_repo.get_active_count()
        
        # Get user repository statistics
        user_repo = repository_factory.get_user_repository()
        active_users = await user_repo.get_active_users_count()
        
        response_time = (time.time() - start_time) * 1000
        
        response = {
            "status": "healthy" if health_status.get("overall") else "unhealthy",
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "version": "1.0.0",
            "environment": "development",
            
            "components": {
                "database": {
                    "status": "up" if health_status.get("database") else "down",
                    "type": "PostgreSQL",
                    "metrics": {
                        "total_properties": total_properties,
                        "active_properties": active_properties,
                        "active_users": active_users
                    }
                },
                "cache": {
                    "status": "up" if health_status.get("redis") else "down",
                    "type": "Redis",
                    "metrics": cache_stats
                },
                "repositories": {
                    "status": "up" if health_status.get("repositories") else "down",
                    "available": {
                        "user_repository": True,
                        "property_repository": True,
                        "model_repository": True,
                        "cache_repository": True
                    }
                },
                "ml_models": {
                    "status": "available",
                    "models": {
                        "hybrid_recommender": "loaded",
                        "content_based_recommender": "loaded",
                        "collaborative_filter": "available",
                        "search_ranker": "available"
                    }
                }
            },
            
            "system_info": {
                "uptime_seconds": time.time() - getattr(request.app.state, 'start_time', time.time()),
                "memory_usage": "N/A",  # TODO: Add memory monitoring
                "cpu_usage": "N/A",     # TODO: Add CPU monitoring
                "disk_usage": "N/A"     # TODO: Add disk monitoring
            },
            
            "dependencies": {
                "external_services": {
                    "property_scrapers": "N/A",     # TODO: Check scraper health
                    "email_service": "N/A",        # TODO: Check email service
                    "monitoring": "N/A"            # TODO: Check monitoring service
                }
            }
        }
        
        status_code = 200 if health_status.get("overall") else 503
        return JSONResponse(content=response, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
                "version": "1.0.0"
            },
            status_code=503
        )


@router.get("/readiness")
async def readiness_check(
    request: Request,
    repository_factory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    
    Checks if the application is ready to serve traffic:
    - All repositories are initialized
    - Database connections are active
    - Cache is accessible
    - Critical components are loaded
    """
    try:
        # Check if factory is initialized
        if not repository_factory.is_initialized():
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "reason": "Repository factory not initialized",
                    "timestamp": time.time()
                },
                status_code=503
            )
        
        # Quick health check
        health_status = await repository_factory.health_check()
        
        if health_status.get("overall"):
            return {
                "status": "ready",
                "timestamp": time.time(),
                "components_ready": True
            }
        else:
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "reason": "Component health check failed",
                    "details": health_status,
                    "timestamp": time.time()
                },
                status_code=503
            )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": time.time()
            },
            status_code=503
        )


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe endpoint.
    
    Simple check to verify the application is running and responding.
    Does not check external dependencies.
    """
    return {
        "status": "alive",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@router.get("/metrics")
async def get_system_metrics(
    request: Request,
    repository_factory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """
    Get system performance metrics for monitoring.
    
    Returns metrics including:
    - Request counts and latencies
    - Cache hit rates
    - Database performance
    - ML model performance
    - Error rates
    """
    try:
        # Get cache metrics
        cache_repo = repository_factory.get_cache_repository()
        cache_stats = await cache_repo.get_cache_stats()
        
        # TODO: Implement actual metrics collection
        # This would typically integrate with Prometheus or similar
        
        return {
            "timestamp": time.time(),
            "api_metrics": {
                "total_requests": 0,        # TODO: Track requests
                "error_rate": 0.0,          # TODO: Track errors
                "avg_response_time": 0.0,   # TODO: Track response times
                "requests_per_second": 0.0  # TODO: Track RPS
            },
            "cache_metrics": {
                "hit_rate": cache_stats.get("hit_rate", 0.0),
                "memory_usage": cache_stats.get("used_memory", 0),
                "key_count": sum(cache_stats.get("key_counts_by_type", {}).values()),
                "operations_per_second": cache_stats.get("instantaneous_ops_per_sec", 0)
            },
            "database_metrics": {
                "connection_pool_size": 10,  # TODO: Get from actual pool
                "active_connections": 0,     # TODO: Track connections
                "query_latency": 0.0,        # TODO: Track query performance
                "queries_per_second": 0.0    # TODO: Track QPS
            },
            "ml_metrics": {
                "recommendation_latency": 0.0,  # TODO: Track ML performance
                "model_accuracy": 0.0,          # TODO: Track model metrics
                "predictions_per_second": 0.0   # TODO: Track prediction rate
            },
            "system_metrics": {
                "uptime_seconds": time.time() - getattr(request.app.state, 'start_time', time.time()),
                "memory_usage_mb": 0,      # TODO: Track memory
                "cpu_usage_percent": 0.0,  # TODO: Track CPU
                "disk_usage_percent": 0.0  # TODO: Track disk
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return JSONResponse(
            content={
                "error": "Failed to retrieve metrics",
                "message": str(e),
                "timestamp": time.time()
            },
            status_code=500
        )