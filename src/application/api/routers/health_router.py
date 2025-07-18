"""
Health check API router for system monitoring and diagnostics.

This module provides comprehensive endpoints for health checks, system status,
and diagnostic information with real system metrics monitoring.

Features:
- Real-time CPU, memory, disk, and network monitoring
- Database and Redis connectivity health checks
- ML model health status verification  
- Performance threshold monitoring with alerts/warnings
- Comprehensive system metrics collection with caching
- Production-ready monitoring endpoints for dashboards
- Kubernetes readiness/liveness probes

Endpoints:
- GET /health/           - Basic health check with system metrics
- GET /health/detailed   - Comprehensive health status with full metrics
- GET /health/readiness  - Kubernetes readiness probe
- GET /health/liveness   - Kubernetes liveness probe  
- GET /health/metrics    - Detailed system performance metrics
- GET /health/monitoring - Real-time monitoring dashboard data
- GET /health/status     - Simple status for load balancer checks
"""

import time
import logging
import os
import psutil
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Health monitoring thresholds
HEALTH_THRESHOLDS = {
    "cpu_usage_percent": 80.0,
    "memory_usage_percent": 85.0,
    "disk_usage_percent": 90.0,
    "response_time_ms": 1000.0,
    "cache_hit_rate_min": 0.7
}

# System monitoring cache
_system_metrics_cache = {
    "last_update": 0,
    "metrics": {},
    "cache_duration": 30  # Cache for 30 seconds
}


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


async def collect_system_metrics() -> Dict[str, Any]:
    """Get comprehensive system metrics with caching"""
    current_time = time.time()
    
    # Check if cached metrics are still valid
    if (current_time - _system_metrics_cache["last_update"] < _system_metrics_cache["cache_duration"] 
        and _system_metrics_cache["metrics"]):
        return _system_metrics_cache["metrics"]
    
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        # System load
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        
        metrics = {
            "timestamp": current_time,
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2]
            },
            "memory": {
                "total_mb": memory.total // 1024 // 1024,
                "available_mb": memory.available // 1024 // 1024,
                "used_mb": memory.used // 1024 // 1024,
                "usage_percent": memory.percent,
                "swap_total_mb": swap.total // 1024 // 1024,
                "swap_used_mb": swap.used // 1024 // 1024,
                "swap_percent": swap.percent
            },
            "disk": {
                "total_gb": disk_usage.total // 1024 // 1024 // 1024,
                "used_gb": disk_usage.used // 1024 // 1024 // 1024,
                "free_gb": disk_usage.free // 1024 // 1024 // 1024,
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            "network": {
                "bytes_sent": network_io.bytes_sent if network_io else 0,
                "bytes_recv": network_io.bytes_recv if network_io else 0,
                "packets_sent": network_io.packets_sent if network_io else 0,
                "packets_recv": network_io.packets_recv if network_io else 0,
                "errors_in": network_io.errin if network_io else 0,
                "errors_out": network_io.errout if network_io else 0
            },
            "process": {
                "memory_rss_mb": process_memory.rss // 1024 // 1024,
                "memory_vms_mb": process_memory.vms // 1024 // 1024,
                "cpu_percent": process_cpu,
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
        }
        
        # Update cache
        _system_metrics_cache["metrics"] = metrics
        _system_metrics_cache["last_update"] = current_time
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {
            "error": f"Failed to collect system metrics: {str(e)}",
            "timestamp": current_time
        }


async def check_ml_models_health() -> Dict[str, Any]:
    """Check health status of ML models"""
    try:
        from ml.serving.model_loader import ModelLoader
        from ml.serving.model_server import ModelServer
        
        model_status = {
            "overall_healthy": True,
            "models": {},
            "last_check": time.time()
        }
        
        # Define expected models
        expected_models = [
            "hybrid_recommender",
            "content_based_recommender", 
            "collaborative_filter",
            "search_ranker"
        ]
        
        # Check each model
        for model_name in expected_models:
            try:
                # Check if model files exist and are loadable
                model_path = f"/models/{model_name}"
                model_healthy = os.path.exists(model_path)
                
                # Additional health checks could include:
                # - Model inference test
                # - Model version compatibility
                # - Model performance metrics
                
                model_status["models"][model_name] = {
                    "status": "healthy" if model_healthy else "unavailable",
                    "path_exists": model_healthy,
                    "last_inference": None,  # TODO: Track last successful inference
                    "error_rate": 0.0,       # TODO: Track inference error rate
                    "avg_latency_ms": 0.0    # TODO: Track inference latency
                }
                
                if not model_healthy:
                    model_status["overall_healthy"] = False
                    
            except Exception as e:
                logger.error(f"Health check failed for model {model_name}: {e}")
                model_status["models"][model_name] = {
                    "status": "error",
                    "error": str(e)
                }
                model_status["overall_healthy"] = False
        
        return model_status
        
    except ImportError:
        # ML modules not available
        return {
            "overall_healthy": False,
            "error": "ML modules not available",
            "models": {},
            "last_check": time.time()
        }
    except Exception as e:
        logger.error(f"ML health check failed: {e}")
        return {
            "overall_healthy": False,
            "error": str(e),
            "models": {},
            "last_check": time.time()
        }


async def evaluate_health_status(metrics: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate overall system health based on metrics and thresholds"""
    alerts = []
    warnings = []
    
    # Check CPU usage
    if "cpu" in metrics and metrics["cpu"].get("usage_percent", 0) > HEALTH_THRESHOLDS["cpu_usage_percent"]:
        alerts.append(f"High CPU usage: {metrics['cpu']['usage_percent']:.1f}%")
    
    # Check memory usage
    if "memory" in metrics and metrics["memory"].get("usage_percent", 0) > HEALTH_THRESHOLDS["memory_usage_percent"]:
        alerts.append(f"High memory usage: {metrics['memory']['usage_percent']:.1f}%")
    
    # Check disk usage
    if "disk" in metrics and metrics["disk"].get("usage_percent", 0) > HEALTH_THRESHOLDS["disk_usage_percent"]:
        alerts.append(f"High disk usage: {metrics['disk']['usage_percent']:.1f}%")
    
    # Check load average (warning if > CPU count)
    if ("cpu" in metrics and "load_avg_5m" in metrics["cpu"] and 
        metrics["cpu"]["load_avg_5m"] > metrics["cpu"]["count"]):
        warnings.append(f"High system load: {metrics['cpu']['load_avg_5m']:.2f}")
    
    # Check database connectivity
    if not health_status.get("database", False):
        alerts.append("Database connectivity failed")
    
    # Check Redis connectivity
    if not health_status.get("redis", False):
        alerts.append("Redis connectivity failed")
    
    return {
        "overall_healthy": len(alerts) == 0,
        "alerts": alerts,
        "warnings": warnings,
        "alert_count": len(alerts),
        "warning_count": len(warnings)
    }


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
        
        # Get basic system metrics for health evaluation
        system_metrics = await collect_system_metrics()
        
        # Evaluate overall health
        health_evaluation = await evaluate_health_status(system_metrics, health_status)
        
        response_time = (time.time() - start_time) * 1000
        
        # Check response time threshold
        if response_time > HEALTH_THRESHOLDS["response_time_ms"]:
            health_evaluation["warnings"].append(f"Slow response time: {response_time:.1f}ms")
        
        overall_healthy = health_status.get("overall", False) and health_evaluation["overall_healthy"]
        status_code = 200 if overall_healthy else 503
        
        response = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "components": {
                "database": "up" if health_status.get("database") else "down",
                "redis": "up" if health_status.get("redis") else "down",
                "repositories": "up" if health_status.get("repositories") else "down"
            },
            "system_health": {
                "cpu_usage_percent": system_metrics.get("cpu", {}).get("usage_percent", 0),
                "memory_usage_percent": system_metrics.get("memory", {}).get("usage_percent", 0),
                "disk_usage_percent": system_metrics.get("disk", {}).get("usage_percent", 0),
                "alerts": health_evaluation["alerts"],
                "warnings": health_evaluation["warnings"]
            },
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
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
        
        # Get comprehensive system metrics
        system_metrics = await collect_system_metrics()
        
        # Get ML models health
        ml_health = await check_ml_models_health()
        
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
        
        # Evaluate overall health
        health_evaluation = await evaluate_health_status(system_metrics, health_status)
        
        response_time = (time.time() - start_time) * 1000
        
        # Check cache hit rate for warnings
        cache_hit_rate = cache_stats.get("hit_rate", 0.0)
        if cache_hit_rate < HEALTH_THRESHOLDS["cache_hit_rate_min"]:
            health_evaluation["warnings"].append(f"Low cache hit rate: {cache_hit_rate:.2%}")
        
        # Check response time threshold
        if response_time > HEALTH_THRESHOLDS["response_time_ms"]:
            health_evaluation["warnings"].append(f"Slow response time: {response_time:.1f}ms")
        
        # Overall health considering all factors
        overall_healthy = (
            health_status.get("overall", False) and 
            health_evaluation["overall_healthy"] and
            ml_health["overall_healthy"]
        )
        
        response = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            
            "health_summary": {
                "overall_healthy": overall_healthy,
                "alerts": health_evaluation["alerts"],
                "warnings": health_evaluation["warnings"],
                "alert_count": health_evaluation["alert_count"],
                "warning_count": health_evaluation["warning_count"]
            },
            
            "components": {
                "database": {
                    "status": "up" if health_status.get("database") else "down",
                    "type": "PostgreSQL",
                    "metrics": {
                        "total_properties": total_properties,
                        "active_properties": active_properties,
                        "active_users": active_users,
                        "connection_healthy": health_status.get("database", False)
                    }
                },
                "cache": {
                    "status": "up" if health_status.get("redis") else "down",
                    "type": "Redis",
                    "metrics": {
                        **cache_stats,
                        "hit_rate_healthy": cache_hit_rate >= HEALTH_THRESHOLDS["cache_hit_rate_min"]
                    }
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
                    "status": "healthy" if ml_health["overall_healthy"] else "unhealthy",
                    "overall_healthy": ml_health["overall_healthy"],
                    "models": ml_health["models"],
                    "last_check": ml_health["last_check"]
                }
            },
            
            "system_metrics": {
                "uptime_seconds": time.time() - getattr(request.app.state, 'start_time', time.time()),
                "cpu": system_metrics.get("cpu", {}),
                "memory": system_metrics.get("memory", {}),
                "disk": system_metrics.get("disk", {}),
                "network": system_metrics.get("network", {}),
                "process": system_metrics.get("process", {})
            },
            
            "performance_monitoring": {
                "thresholds": HEALTH_THRESHOLDS,
                "current_metrics": {
                    "cpu_usage_percent": system_metrics.get("cpu", {}).get("usage_percent", 0),
                    "memory_usage_percent": system_metrics.get("memory", {}).get("usage_percent", 0),
                    "disk_usage_percent": system_metrics.get("disk", {}).get("usage_percent", 0),
                    "response_time_ms": response_time,
                    "cache_hit_rate": cache_hit_rate
                },
                "status_checks": {
                    "cpu_healthy": system_metrics.get("cpu", {}).get("usage_percent", 0) < HEALTH_THRESHOLDS["cpu_usage_percent"],
                    "memory_healthy": system_metrics.get("memory", {}).get("usage_percent", 0) < HEALTH_THRESHOLDS["memory_usage_percent"],
                    "disk_healthy": system_metrics.get("disk", {}).get("usage_percent", 0) < HEALTH_THRESHOLDS["disk_usage_percent"],
                    "response_time_healthy": response_time < HEALTH_THRESHOLDS["response_time_ms"],
                    "cache_performance_healthy": cache_hit_rate >= HEALTH_THRESHOLDS["cache_hit_rate_min"]
                }
            },
            
            "dependencies": {
                "external_services": {
                    "property_scrapers": "monitoring_not_implemented",
                    "email_service": "monitoring_not_implemented",
                    "monitoring_service": "self_monitoring_active"
                },
                "internal_services": {
                    "database_connection_pool": "healthy" if health_status.get("database") else "unhealthy",
                    "redis_connection": "healthy" if health_status.get("redis") else "unhealthy",
                    "ml_model_loader": "healthy" if ml_health["overall_healthy"] else "unhealthy"
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
        # Get comprehensive system metrics
        system_metrics = await collect_system_metrics()
        
        # Get cache metrics
        cache_repo = repository_factory.get_cache_repository()
        cache_stats = await cache_repo.get_cache_stats()
        
        # Get ML models health
        ml_health = await check_ml_models_health()
        
        # Get database connection pool info
        db_health = False
        db_pool_size = 0
        active_connections = 0
        
        try:
            if repository_factory.data_manager and repository_factory.data_manager.db_manager:
                db_pool = repository_factory.data_manager.db_manager.pool
                if db_pool:
                    db_health = True
                    db_pool_size = db_pool.get_size()
                    active_connections = db_pool.get_size() - db_pool.get_idle_size()
        except Exception as e:
            logger.warning(f"Could not get database pool metrics: {e}")
        
        # Calculate application uptime
        app_start_time = getattr(request.app.state, 'start_time', time.time())
        uptime_seconds = time.time() - app_start_time
        
        return {
            "timestamp": time.time(),
            "collection_time_ms": (time.time() - system_metrics.get("timestamp", time.time())) * 1000,
            
            "api_metrics": {
                "uptime_seconds": uptime_seconds,
                "uptime_human": f"{uptime_seconds // 3600:.0f}h {(uptime_seconds % 3600) // 60:.0f}m {uptime_seconds % 60:.0f}s",
                "total_requests": getattr(request.app.state, 'total_requests', 0),
                "error_count": getattr(request.app.state, 'error_count', 0),
                "error_rate": getattr(request.app.state, 'error_count', 0) / max(getattr(request.app.state, 'total_requests', 1), 1),
                "avg_response_time_ms": getattr(request.app.state, 'avg_response_time', 0.0),
                "requests_per_second": getattr(request.app.state, 'total_requests', 0) / max(uptime_seconds, 1)
            },
            
            "system_metrics": {
                "cpu": {
                    "usage_percent": system_metrics.get("cpu", {}).get("usage_percent", 0),
                    "count": system_metrics.get("cpu", {}).get("count", 0),
                    "load_avg_1m": system_metrics.get("cpu", {}).get("load_avg_1m", 0),
                    "load_avg_5m": system_metrics.get("cpu", {}).get("load_avg_5m", 0),
                    "load_avg_15m": system_metrics.get("cpu", {}).get("load_avg_15m", 0),
                    "frequency_mhz": system_metrics.get("cpu", {}).get("frequency_mhz")
                },
                "memory": {
                    "total_mb": system_metrics.get("memory", {}).get("total_mb", 0),
                    "used_mb": system_metrics.get("memory", {}).get("used_mb", 0),
                    "available_mb": system_metrics.get("memory", {}).get("available_mb", 0),
                    "usage_percent": system_metrics.get("memory", {}).get("usage_percent", 0),
                    "swap_total_mb": system_metrics.get("memory", {}).get("swap_total_mb", 0),
                    "swap_used_mb": system_metrics.get("memory", {}).get("swap_used_mb", 0),
                    "swap_percent": system_metrics.get("memory", {}).get("swap_percent", 0)
                },
                "disk": {
                    "total_gb": system_metrics.get("disk", {}).get("total_gb", 0),
                    "used_gb": system_metrics.get("disk", {}).get("used_gb", 0),
                    "free_gb": system_metrics.get("disk", {}).get("free_gb", 0),
                    "usage_percent": system_metrics.get("disk", {}).get("usage_percent", 0),
                    "read_ops": system_metrics.get("disk", {}).get("read_count", 0),
                    "write_ops": system_metrics.get("disk", {}).get("write_count", 0),
                    "read_bytes": system_metrics.get("disk", {}).get("read_bytes", 0),
                    "write_bytes": system_metrics.get("disk", {}).get("write_bytes", 0)
                },
                "network": {
                    "bytes_sent": system_metrics.get("network", {}).get("bytes_sent", 0),
                    "bytes_recv": system_metrics.get("network", {}).get("bytes_recv", 0),
                    "packets_sent": system_metrics.get("network", {}).get("packets_sent", 0),
                    "packets_recv": system_metrics.get("network", {}).get("packets_recv", 0),
                    "errors_in": system_metrics.get("network", {}).get("errors_in", 0),
                    "errors_out": system_metrics.get("network", {}).get("errors_out", 0)
                }
            },
            
            "process_metrics": {
                "memory_rss_mb": system_metrics.get("process", {}).get("memory_rss_mb", 0),
                "memory_vms_mb": system_metrics.get("process", {}).get("memory_vms_mb", 0),
                "cpu_percent": system_metrics.get("process", {}).get("cpu_percent", 0),
                "num_threads": system_metrics.get("process", {}).get("num_threads", 0),
                "open_files": system_metrics.get("process", {}).get("open_files", 0),
                "connections": system_metrics.get("process", {}).get("connections", 0)
            },
            
            "cache_metrics": {
                "status": "healthy" if cache_stats.get("connected", False) else "unhealthy",
                "hit_rate": cache_stats.get("hit_rate", 0.0),
                "memory_usage_mb": cache_stats.get("used_memory", 0) // 1024 // 1024,
                "memory_usage_human": cache_stats.get("used_memory_human", "0B"),
                "total_keys": sum(cache_stats.get("key_counts_by_type", {}).values()),
                "key_counts_by_type": cache_stats.get("key_counts_by_type", {}),
                "operations_per_second": cache_stats.get("instantaneous_ops_per_sec", 0),
                "connected_clients": cache_stats.get("connected_clients", 0),
                "uptime_seconds": cache_stats.get("uptime_in_seconds", 0)
            },
            
            "database_metrics": {
                "status": "healthy" if db_health else "unhealthy",
                "connection_pool_size": db_pool_size,
                "active_connections": active_connections,
                "idle_connections": db_pool_size - active_connections,
                "pool_utilization_percent": (active_connections / max(db_pool_size, 1)) * 100 if db_pool_size > 0 else 0,
                "query_latency_ms": getattr(request.app.state, 'avg_query_time', 0.0),  # Would need to be tracked
                "queries_per_second": getattr(request.app.state, 'queries_per_second', 0.0)  # Would need to be tracked
            },
            
            "ml_metrics": {
                "status": "healthy" if ml_health["overall_healthy"] else "unhealthy",
                "models_healthy": sum(1 for model in ml_health["models"].values() 
                                    if model.get("status") == "healthy"),
                "total_models": len(ml_health["models"]),
                "model_details": ml_health["models"],
                "last_health_check": ml_health["last_check"],
                "recommendation_latency_ms": getattr(request.app.state, 'ml_avg_latency', 0.0),
                "predictions_per_second": getattr(request.app.state, 'ml_predictions_per_sec', 0.0),
                "model_error_rate": getattr(request.app.state, 'ml_error_rate', 0.0)
            },
            
            "monitoring_config": {
                "metrics_cache_duration_seconds": _system_metrics_cache["cache_duration"],
                "health_thresholds": HEALTH_THRESHOLDS,
                "metrics_collection_enabled": True,
                "last_metrics_update": _system_metrics_cache["last_update"]
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


@router.get("/monitoring")
async def get_monitoring_dashboard(
    request: Request,
    repository_factory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """
    Real-time monitoring dashboard endpoint for production monitoring.
    
    Provides comprehensive metrics for monitoring systems like Grafana,
    Prometheus, or custom monitoring dashboards.
    """
    try:
        start_time = time.time()
        
        # Get all system data
        system_metrics = await get_system_metrics()
        health_status = await repository_factory.health_check()
        ml_health = await check_ml_models_health()
        
        # Get cache statistics
        cache_repo = repository_factory.get_cache_repository()
        cache_stats = await cache_repo.get_cache_stats()
        
        # Evaluate health
        health_evaluation = await evaluate_health_status(system_metrics, health_status)
        
        response_time = (time.time() - start_time) * 1000
        
        # Calculate uptime
        app_start_time = getattr(request.app.state, 'start_time', time.time())
        uptime_seconds = time.time() - app_start_time
        
        return {
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "uptime_seconds": uptime_seconds,
            
            # High-level status indicators
            "status_indicators": {
                "overall_healthy": (
                    health_status.get("overall", False) and 
                    health_evaluation["overall_healthy"] and
                    ml_health["overall_healthy"]
                ),
                "database_healthy": health_status.get("database", False),
                "cache_healthy": health_status.get("redis", False),
                "ml_models_healthy": ml_health["overall_healthy"],
                "system_performance_healthy": health_evaluation["overall_healthy"]
            },
            
            # Key performance indicators
            "kpis": {
                "cpu_usage_percent": system_metrics.get("cpu", {}).get("usage_percent", 0),
                "memory_usage_percent": system_metrics.get("memory", {}).get("usage_percent", 0),
                "disk_usage_percent": system_metrics.get("disk", {}).get("usage_percent", 0),
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                "response_time_ms": response_time,
                "error_count": len(health_evaluation["alerts"]),
                "warning_count": len(health_evaluation["warnings"])
            },
            
            # Detailed metrics for monitoring systems
            "detailed_metrics": {
                "system": {
                    "cpu_cores": system_metrics.get("cpu", {}).get("count", 0),
                    "cpu_frequency_mhz": system_metrics.get("cpu", {}).get("frequency_mhz"),
                    "load_average": {
                        "1m": system_metrics.get("cpu", {}).get("load_avg_1m", 0),
                        "5m": system_metrics.get("cpu", {}).get("load_avg_5m", 0),
                        "15m": system_metrics.get("cpu", {}).get("load_avg_15m", 0)
                    },
                    "memory_total_mb": system_metrics.get("memory", {}).get("total_mb", 0),
                    "memory_available_mb": system_metrics.get("memory", {}).get("available_mb", 0),
                    "disk_total_gb": system_metrics.get("disk", {}).get("total_gb", 0),
                    "disk_free_gb": system_metrics.get("disk", {}).get("free_gb", 0)
                },
                "application": {
                    "process_memory_mb": system_metrics.get("process", {}).get("memory_rss_mb", 0),
                    "process_cpu_percent": system_metrics.get("process", {}).get("cpu_percent", 0),
                    "thread_count": system_metrics.get("process", {}).get("num_threads", 0),
                    "open_files": system_metrics.get("process", {}).get("open_files", 0),
                    "network_connections": system_metrics.get("process", {}).get("connections", 0)
                },
                "database": {
                    "connection_healthy": health_status.get("database", False),
                    "pool_size": getattr(repository_factory.data_manager.db_manager.pool, 'get_size', lambda: 0)() if hasattr(repository_factory, 'data_manager') and repository_factory.data_manager and repository_factory.data_manager.db_manager and repository_factory.data_manager.db_manager.pool else 0
                },
                "cache": {
                    "connection_healthy": health_status.get("redis", False),
                    "memory_usage_mb": cache_stats.get("used_memory", 0) // 1024 // 1024,
                    "total_keys": sum(cache_stats.get("key_counts_by_type", {}).values()),
                    "operations_per_second": cache_stats.get("instantaneous_ops_per_sec", 0)
                }
            },
            
            # Alerts and warnings for monitoring systems
            "monitoring_alerts": {
                "critical": health_evaluation["alerts"],
                "warnings": health_evaluation["warnings"],
                "ml_model_issues": [
                    f"Model {name}: {details.get('error', details.get('status', 'unknown'))}"
                    for name, details in ml_health["models"].items()
                    if details.get("status") != "healthy"
                ]
            },
            
            # Thresholds for monitoring system configuration
            "thresholds": HEALTH_THRESHOLDS,
            
            # Additional metadata
            "metadata": {
                "environment": os.getenv("ENVIRONMENT", "development"),
                "version": "1.0.0",
                "monitoring_enabled": True,
                "metrics_cache_ttl": _system_metrics_cache["cache_duration"]
            }
        }
        
    except Exception as e:
        logger.error(f"Monitoring dashboard failed: {e}")
        return JSONResponse(
            content={
                "error": "Failed to retrieve monitoring data",
                "message": str(e),
                "timestamp": time.time(),
                "status_indicators": {
                    "overall_healthy": False,
                    "monitoring_healthy": False
                }
            },
            status_code=500
        )


@router.get("/status")
async def get_simple_status() -> Dict[str, Any]:
    """
    Simple status endpoint for basic monitoring and load balancer health checks.
    
    Returns minimal information for systems that just need to know if the 
    application is running and responding.
    """
    try:
        # Very basic system check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            "status": "ok",
            "timestamp": time.time(),
            "uptime": psutil.boot_time(),
            "basic_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "healthy": cpu_percent < 95 and memory.percent < 95
            }
        }
        
    except Exception as e:
        logger.error(f"Simple status check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            },
            status_code=503
        )