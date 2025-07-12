"""
Administrative Endpoints for the Rental ML System.

This module provides comprehensive administrative capabilities including:
- System monitoring and health checks
- Model management and deployment
- Data pipeline status and controls
- User management and moderation
- Analytics and reporting endpoints
- Configuration management
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import pandas as pd

from ...domain.entities.user import User
from ...domain.entities.property import Property
from ...domain.repositories.user_repository import UserRepository
from ...domain.repositories.property_repository import PropertyRepository
from ...infrastructure.ml.serving.model_server import ModelServer
from ...infrastructure.data import DataConfig

logger = logging.getLogger(__name__)

admin_router = APIRouter()


class SystemStatus(str, Enum):
    """System status options"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


class ModelStatus(str, Enum):
    """Model status options"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    UPDATING = "updating"
    ERROR = "error"


class UserRole(str, Enum):
    """User role options"""
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    AGENT = "agent"


class DataPipelineStatus(str, Enum):
    """Data pipeline status options"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SystemHealthResponse(BaseModel):
    """Response model for system health"""
    overall_status: SystemStatus
    components: Dict[str, Any]
    metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]
    uptime_seconds: float
    last_check: datetime
    recommendations: List[str]


class ModelManagementRequest(BaseModel):
    """Request model for model management"""
    model_name: str = Field(..., description="Model name")
    action: str = Field(..., description="Action to perform (deploy, stop, update, rollback)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Action parameters")
    force: bool = Field(default=False, description="Force action even if risky")


class ModelDeploymentResponse(BaseModel):
    """Response model for model deployment"""
    model_name: str
    action: str
    status: ModelStatus
    deployment_id: str
    timestamp: datetime
    message: str
    rollback_available: bool


class UserModerationRequest(BaseModel):
    """Request model for user moderation"""
    user_id: UUID = Field(..., description="User ID")
    action: str = Field(..., description="Moderation action")
    reason: str = Field(..., description="Reason for action")
    duration_hours: Optional[int] = Field(None, description="Duration for temporary actions")
    notify_user: bool = Field(default=True, description="Send notification to user")


class DataPipelineRequest(BaseModel):
    """Request model for data pipeline management"""
    pipeline_name: str = Field(..., description="Pipeline name")
    action: str = Field(..., description="Action to perform")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Pipeline parameters")


class ConfigurationRequest(BaseModel):
    """Request model for configuration updates"""
    config_key: str = Field(..., description="Configuration key")
    config_value: Any = Field(..., description="Configuration value")
    environment: str = Field(default="production", description="Environment")
    restart_required: bool = Field(default=False, description="Whether restart is required")


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics"""
    timestamp: datetime
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    ml_metrics: Dict[str, float]
    business_metrics: Dict[str, float]
    error_rates: Dict[str, float]
    trends: Dict[str, List[float]]


class UserManagementResponse(BaseModel):
    """Response model for user management"""
    total_users: int
    active_users: int
    new_users_today: int
    user_segments: Dict[str, int]
    moderation_actions: List[Dict[str, Any]]
    top_users: List[Dict[str, Any]]


class ReportGenerationRequest(BaseModel):
    """Request model for report generation"""
    report_type: str = Field(..., description="Type of report")
    start_date: datetime = Field(..., description="Report start date")
    end_date: datetime = Field(..., description="Report end date")
    filters: Optional[Dict[str, Any]] = Field(None, description="Report filters")
    format: str = Field(default="json", description="Report format (json, csv, pdf)")
    include_charts: bool = Field(default=False, description="Include charts in report")


class AlertConfigurationRequest(BaseModel):
    """Request model for alert configuration"""
    alert_name: str = Field(..., description="Alert name")
    metric: str = Field(..., description="Metric to monitor")
    threshold: float = Field(..., description="Alert threshold")
    severity: AlertSeverity = Field(..., description="Alert severity")
    notification_channels: List[str] = Field(..., description="Notification channels")
    enabled: bool = Field(default=True, description="Whether alert is enabled")


# Dependency injection
async def get_user_repository() -> UserRepository:
    """Get user repository instance"""
    from ...infrastructure.data.repositories.postgres_user_repository import PostgresUserRepository
    return PostgresUserRepository()


async def get_property_repository() -> PropertyRepository:
    """Get property repository instance"""
    from ...infrastructure.data.repositories.postgres_property_repository import PostgresPropertyRepository
    return PostgresPropertyRepository()


async def get_model_server() -> ModelServer:
    """Get model server instance"""
    return ModelServer()


@admin_router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    detailed: bool = Query(default=True, description="Include detailed component health"),
    user_repo: UserRepository = Depends(get_user_repository),
    property_repo: PropertyRepository = Depends(get_property_repository),
    model_server: ModelServer = Depends(get_model_server)
):
    """
    Get comprehensive system health status.
    
    Monitors all system components and provides actionable insights.
    """
    try:
        logger.info("Checking system health")
        
        components = {}
        metrics = {}
        alerts = []
        overall_status = SystemStatus.HEALTHY
        
        # Check database health
        try:
            db_start = time.time()
            # Mock database check - in reality would test actual connections
            db_health = True
            db_latency = (time.time() - db_start) * 1000
            
            components["database"] = {
                "status": "healthy" if db_health else "critical",
                "latency_ms": db_latency,
                "connections": 45,  # Mock data
                "max_connections": 100
            }
            metrics["db_latency_ms"] = db_latency
            
            if db_latency > 1000:
                alerts.append({
                    "type": "database_latency",
                    "severity": "warning",
                    "message": f"Database latency high: {db_latency:.2f}ms"
                })
            
        except Exception as e:
            components["database"] = {"status": "critical", "error": str(e)}
            overall_status = SystemStatus.CRITICAL
            alerts.append({
                "type": "database_connection",
                "severity": "critical",
                "message": f"Database connection failed: {str(e)}"
            })
        
        # Check Redis/Cache health
        try:
            cache_health = True  # Mock
            components["cache"] = {
                "status": "healthy" if cache_health else "warning",
                "memory_usage": "256MB",
                "hit_rate": 0.95,
                "connected_clients": 12
            }
            metrics["cache_hit_rate"] = 0.95
            
        except Exception as e:
            components["cache"] = {"status": "warning", "error": str(e)}
            if overall_status == SystemStatus.HEALTHY:
                overall_status = SystemStatus.WARNING
        
        # Check ML models health
        try:
            model_health = await model_server.check_model_health()
            components["ml_models"] = model_health["models"]
            
            # Check if any critical models are down
            critical_models = ["hybrid_recommender", "search_ranker"]
            for model in critical_models:
                if model in model_health["models"] and model_health["models"][model]["status"] != "active":
                    overall_status = SystemStatus.CRITICAL
                    alerts.append({
                        "type": "model_failure",
                        "severity": "critical",
                        "message": f"Critical model '{model}' is not active"
                    })
            
        except Exception as e:
            components["ml_models"] = {"status": "critical", "error": str(e)}
            overall_status = SystemStatus.CRITICAL
        
        # Check API performance
        api_metrics = {
            "requests_per_minute": 150,  # Mock data
            "average_response_time": 245,
            "error_rate": 0.02,
            "active_connections": 23
        }
        components["api"] = {
            "status": "healthy",
            "metrics": api_metrics
        }
        metrics.update(api_metrics)
        
        if api_metrics["error_rate"] > 0.05:
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"API error rate elevated: {api_metrics['error_rate']:.2%}"
            })
        
        # System resources
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        components["system_resources"] = {
            "status": "healthy" if max(cpu_percent, memory_percent, disk_percent) < 80 else "warning",
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent
        }
        
        metrics.update({
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent
        })
        
        if cpu_percent > 90:
            alerts.append({
                "type": "high_cpu",
                "severity": "critical",
                "message": f"CPU usage critical: {cpu_percent:.1f}%"
            })
        
        # Generate recommendations
        recommendations = []
        if memory_percent > 80:
            recommendations.append("Consider increasing server memory or optimizing memory usage")
        if len(alerts) > 5:
            recommendations.append("Multiple alerts detected - investigate system stability")
        if api_metrics["average_response_time"] > 500:
            recommendations.append("API response times elevated - check for performance bottlenecks")
        
        return SystemHealthResponse(
            overall_status=overall_status,
            components=components if detailed else {"summary": f"{len(components)} components checked"},
            metrics=metrics,
            alerts=alerts,
            uptime_seconds=time.time() - 1640995200,  # Mock start time
            last_check=datetime.now(),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return SystemHealthResponse(
            overall_status=SystemStatus.CRITICAL,
            components={"error": str(e)},
            metrics={},
            alerts=[{
                "type": "health_check_failure",
                "severity": "critical",
                "message": f"Health check failed: {str(e)}"
            }],
            uptime_seconds=0,
            last_check=datetime.now(),
            recommendations=["Investigate health check system failure"]
        )


@admin_router.post("/models/manage", response_model=ModelDeploymentResponse)
async def manage_model(
    request: ModelManagementRequest,
    background_tasks: BackgroundTasks,
    model_server: ModelServer = Depends(get_model_server)
):
    """
    Manage ML model deployment and lifecycle.
    
    Supports:
    - deploy: Deploy new model version
    - stop: Stop model serving
    - update: Update model configuration
    - rollback: Rollback to previous version
    """
    try:
        logger.info(f"Managing model '{request.model_name}' with action '{request.action}'")
        
        deployment_id = str(uuid4())
        
        if request.action == "deploy":
            # Deploy new model
            result = await model_server.deploy_model(
                model_name=request.model_name,
                parameters=request.parameters or {},
                force=request.force
            )
            
            status = ModelStatus.ACTIVE if result["success"] else ModelStatus.ERROR
            message = result.get("message", "Model deployed successfully")
            
        elif request.action == "stop":
            # Stop model
            result = await model_server.stop_model(request.model_name)
            status = ModelStatus.INACTIVE
            message = "Model stopped successfully"
            
        elif request.action == "update":
            # Update model configuration
            result = await model_server.update_model_config(
                model_name=request.model_name,
                config=request.parameters or {}
            )
            status = ModelStatus.ACTIVE
            message = "Model configuration updated"
            
        elif request.action == "rollback":
            # Rollback to previous version
            result = await model_server.rollback_model(request.model_name)
            status = ModelStatus.ACTIVE if result["success"] else ModelStatus.ERROR
            message = result.get("message", "Model rolled back successfully")
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown action: {request.action}"
            )
        
        # Schedule background monitoring
        background_tasks.add_task(
            monitor_model_deployment,
            model_name=request.model_name,
            deployment_id=deployment_id
        )
        
        return ModelDeploymentResponse(
            model_name=request.model_name,
            action=request.action,
            status=status,
            deployment_id=deployment_id,
            timestamp=datetime.now(),
            message=message,
            rollback_available=True
        )
        
    except Exception as e:
        logger.error(f"Model management failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model management failed: {str(e)}"
        )


@admin_router.get("/users/management", response_model=UserManagementResponse)
async def get_user_management_overview(
    include_segments: bool = Query(default=True, description="Include user segments"),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """Get comprehensive user management overview."""
    try:
        logger.info("Getting user management overview")
        
        # Get user statistics (mock data for now)
        stats = {
            "total_users": 15420,
            "active_users": 12350,
            "new_users_today": 47,
            "user_segments": {
                "free": 10500,
                "premium": 3200,
                "agent": 1200,
                "admin": 20
            },
            "moderation_actions": [
                {
                    "user_id": str(uuid4()),
                    "action": "warning",
                    "reason": "Spam behavior",
                    "timestamp": datetime.now() - timedelta(hours=2)
                }
            ],
            "top_users": [
                {
                    "user_id": str(uuid4()),
                    "email": "user@example.com",
                    "total_interactions": 245,
                    "properties_viewed": 89,
                    "join_date": datetime.now() - timedelta(days=30)
                }
            ]
        }
        
        return UserManagementResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get user management overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user overview: {str(e)}"
        )


@admin_router.post("/users/moderate")
async def moderate_user(
    request: UserModerationRequest,
    background_tasks: BackgroundTasks,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Perform user moderation actions.
    
    Supports:
    - warn: Send warning to user
    - suspend: Suspend user account
    - ban: Ban user account
    - restrict: Restrict user features
    """
    try:
        logger.info(f"Moderating user {request.user_id} with action '{request.action}'")
        
        # Get user
        user = await user_repo.get_by_id(request.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {request.user_id} not found"
            )
        
        # Perform moderation action
        moderation_record = {
            "user_id": request.user_id,
            "action": request.action,
            "reason": request.reason,
            "duration_hours": request.duration_hours,
            "moderator_id": "admin",  # In real implementation, get from auth
            "timestamp": datetime.now()
        }
        
        if request.action == "warn":
            # Send warning
            background_tasks.add_task(
                send_user_warning,
                user_id=request.user_id,
                reason=request.reason
            )
            
        elif request.action == "suspend":
            # Suspend user
            user.deactivate()
            await user_repo.update(user)
            
            if request.duration_hours:
                background_tasks.add_task(
                    schedule_user_reactivation,
                    user_id=request.user_id,
                    hours=request.duration_hours
                )
            
        elif request.action == "ban":
            # Ban user permanently
            user.deactivate()
            # Add to ban list (in real implementation)
            await user_repo.update(user)
            
        elif request.action == "restrict":
            # Restrict user features
            # In real implementation, update user permissions
            pass
        
        # Log moderation action
        background_tasks.add_task(
            log_moderation_action,
            moderation_record=moderation_record
        )
        
        # Notify user if requested
        if request.notify_user:
            background_tasks.add_task(
                notify_user_of_moderation,
                user_id=request.user_id,
                action=request.action,
                reason=request.reason
            )
        
        return {
            "message": f"User {request.user_id} {request.action} action completed",
            "moderation_id": str(uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User moderation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User moderation failed: {str(e)}"
        )


@admin_router.get("/pipelines/status")
async def get_data_pipeline_status():
    """Get status of all data pipelines."""
    try:
        logger.info("Getting data pipeline status")
        
        # Mock pipeline status
        pipelines = {
            "property_scraping": {
                "status": DataPipelineStatus.RUNNING,
                "last_run": datetime.now() - timedelta(minutes=30),
                "next_run": datetime.now() + timedelta(minutes=30),
                "success_rate": 0.95,
                "properties_scraped_today": 1250,
                "errors": []
            },
            "ml_training": {
                "status": DataPipelineStatus.RUNNING,
                "last_run": datetime.now() - timedelta(hours=2),
                "next_run": datetime.now() + timedelta(hours=22),
                "success_rate": 1.0,
                "models_updated": 3,
                "errors": []
            },
            "data_quality": {
                "status": DataPipelineStatus.RUNNING,
                "last_run": datetime.now() - timedelta(minutes=15),
                "next_run": datetime.now() + timedelta(minutes=45),
                "success_rate": 0.98,
                "issues_detected": 5,
                "errors": []
            },
            "analytics_processing": {
                "status": DataPipelineStatus.ERROR,
                "last_run": datetime.now() - timedelta(hours=1),
                "next_run": datetime.now() + timedelta(hours=1),
                "success_rate": 0.85,
                "errors": ["Database connection timeout"]
            }
        }
        
        return {
            "pipelines": pipelines,
            "summary": {
                "total_pipelines": len(pipelines),
                "running": sum(1 for p in pipelines.values() if p["status"] == DataPipelineStatus.RUNNING),
                "error": sum(1 for p in pipelines.values() if p["status"] == DataPipelineStatus.ERROR),
                "stopped": sum(1 for p in pipelines.values() if p["status"] == DataPipelineStatus.STOPPED)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline status: {str(e)}"
        )


@admin_router.post("/pipelines/control")
async def control_data_pipeline(
    request: DataPipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Control data pipeline operations.
    
    Supports:
    - start: Start pipeline
    - stop: Stop pipeline
    - restart: Restart pipeline
    - trigger: Trigger immediate run
    """
    try:
        logger.info(f"Controlling pipeline '{request.pipeline_name}' with action '{request.action}'")
        
        # Validate pipeline exists
        valid_pipelines = ["property_scraping", "ml_training", "data_quality", "analytics_processing"]
        if request.pipeline_name not in valid_pipelines:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline '{request.pipeline_name}' not found"
            )
        
        # Perform action
        if request.action == "start":
            background_tasks.add_task(
                start_pipeline,
                pipeline_name=request.pipeline_name,
                parameters=request.parameters
            )
            message = f"Pipeline '{request.pipeline_name}' start initiated"
            
        elif request.action == "stop":
            background_tasks.add_task(
                stop_pipeline,
                pipeline_name=request.pipeline_name
            )
            message = f"Pipeline '{request.pipeline_name}' stop initiated"
            
        elif request.action == "restart":
            background_tasks.add_task(
                restart_pipeline,
                pipeline_name=request.pipeline_name,
                parameters=request.parameters
            )
            message = f"Pipeline '{request.pipeline_name}' restart initiated"
            
        elif request.action == "trigger":
            background_tasks.add_task(
                trigger_pipeline,
                pipeline_name=request.pipeline_name,
                parameters=request.parameters
            )
            message = f"Pipeline '{request.pipeline_name}' triggered for immediate run"
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown action: {request.action}"
            )
        
        return {
            "message": message,
            "pipeline_name": request.pipeline_name,
            "action": request.action,
            "timestamp": datetime.now().isoformat(),
            "job_id": str(uuid4())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline control failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline control failed: {str(e)}"
        )


@admin_router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics"),
    granularity: str = Query(default="hour", description="Metrics granularity (minute, hour, day)")
):
    """Get comprehensive system metrics and performance data."""
    try:
        logger.info("Getting system metrics")
        
        # Mock metrics data
        metrics = {
            "performance_metrics": {
                "avg_response_time_ms": 245.5,
                "p95_response_time_ms": 580.2,
                "requests_per_second": 125.3,
                "concurrent_users": 1250,
                "cache_hit_rate": 0.94
            },
            "resource_usage": {
                "cpu_usage_percent": 45.2,
                "memory_usage_percent": 67.8,
                "disk_usage_percent": 23.1,
                "network_io_mbps": 12.5,
                "database_connections": 45
            },
            "ml_metrics": {
                "recommendation_accuracy": 0.87,
                "search_relevance_score": 0.92,
                "model_inference_time_ms": 125.3,
                "feature_engineering_time_ms": 45.7,
                "model_training_accuracy": 0.89
            },
            "business_metrics": {
                "daily_active_users": 2150,
                "properties_viewed_per_user": 15.3,
                "search_to_view_rate": 0.45,
                "user_engagement_score": 0.72,
                "recommendation_click_rate": 0.23
            },
            "error_rates": {
                "api_error_rate": 0.018,
                "ml_model_error_rate": 0.003,
                "database_error_rate": 0.001,
                "scraping_error_rate": 0.052
            },
            "trends": {
                "response_time_trend": [245, 250, 240, 255, 245],
                "user_growth_trend": [2100, 2120, 2135, 2140, 2150],
                "error_rate_trend": [0.02, 0.018, 0.019, 0.017, 0.018]
            }
        }
        
        return SystemMetricsResponse(
            timestamp=datetime.now(),
            **metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@admin_router.post("/reports/generate")
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate comprehensive system and business reports."""
    try:
        logger.info(f"Generating {request.report_type} report")
        
        report_id = str(uuid4())
        
        # Validate report type
        valid_reports = [
            "system_performance", "user_analytics", "property_insights",
            "ml_model_performance", "business_metrics", "security_audit"
        ]
        
        if request.report_type not in valid_reports:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid report type: {request.report_type}"
            )
        
        # Schedule report generation
        background_tasks.add_task(
            generate_system_report,
            report_id=report_id,
            report_type=request.report_type,
            start_date=request.start_date,
            end_date=request.end_date,
            filters=request.filters or {},
            format=request.format,
            include_charts=request.include_charts
        )
        
        return {
            "message": f"Report generation initiated",
            "report_id": report_id,
            "report_type": request.report_type,
            "estimated_completion": datetime.now() + timedelta(minutes=10),
            "download_url": f"/admin/reports/{report_id}/download",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )


@admin_router.post("/configuration/update")
async def update_configuration(
    request: ConfigurationRequest,
    background_tasks: BackgroundTasks
):
    """Update system configuration settings."""
    try:
        logger.info(f"Updating configuration: {request.config_key}")
        
        # Validate configuration key
        allowed_configs = [
            "rate_limit_per_minute", "max_recommendations", "cache_ttl_seconds",
            "ml_batch_size", "scraping_interval_minutes", "log_level"
        ]
        
        if request.config_key not in allowed_configs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Configuration key '{request.config_key}' not allowed"
            )
        
        # Update configuration (in real implementation, update in database/config store)
        config_update = {
            "key": request.config_key,
            "value": request.config_value,
            "environment": request.environment,
            "updated_by": "admin",  # Get from auth context
            "updated_at": datetime.now(),
            "previous_value": "previous_value_here"  # Get from current config
        }
        
        # Schedule configuration reload if needed
        if request.restart_required:
            background_tasks.add_task(
                schedule_service_restart,
                config_key=request.config_key,
                delay_minutes=5
            )
        
        # Log configuration change
        background_tasks.add_task(
            log_configuration_change,
            config_update=config_update
        )
        
        return {
            "message": f"Configuration '{request.config_key}' updated successfully",
            "config_key": request.config_key,
            "new_value": request.config_value,
            "restart_required": request.restart_required,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration update failed: {str(e)}"
        )


@admin_router.post("/alerts/configure")
async def configure_alert(
    request: AlertConfigurationRequest,
    background_tasks: BackgroundTasks
):
    """Configure system alerts and monitoring."""
    try:
        logger.info(f"Configuring alert: {request.alert_name}")
        
        # Create alert configuration
        alert_config = {
            "alert_id": str(uuid4()),
            "name": request.alert_name,
            "metric": request.metric,
            "threshold": request.threshold,
            "severity": request.severity.value,
            "notification_channels": request.notification_channels,
            "enabled": request.enabled,
            "created_at": datetime.now(),
            "created_by": "admin"  # Get from auth context
        }
        
        # Save alert configuration (in real implementation)
        background_tasks.add_task(
            save_alert_configuration,
            alert_config=alert_config
        )
        
        # Test alert if enabled
        if request.enabled:
            background_tasks.add_task(
                test_alert_configuration,
                alert_id=alert_config["alert_id"]
            )
        
        return {
            "message": f"Alert '{request.alert_name}' configured successfully",
            "alert_id": alert_config["alert_id"],
            "alert_name": request.alert_name,
            "enabled": request.enabled,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert configuration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert configuration failed: {str(e)}"
        )


# Background tasks
async def monitor_model_deployment(model_name: str, deployment_id: str):
    """Monitor model deployment progress"""
    logger.info(f"Monitoring deployment {deployment_id} for model {model_name}")


async def send_user_warning(user_id: UUID, reason: str):
    """Send warning notification to user"""
    logger.info(f"Sending warning to user {user_id}: {reason}")


async def schedule_user_reactivation(user_id: UUID, hours: int):
    """Schedule user reactivation after suspension"""
    logger.info(f"Scheduling reactivation for user {user_id} in {hours} hours")


async def log_moderation_action(moderation_record: Dict[str, Any]):
    """Log moderation action"""
    logger.info(f"Logging moderation action: {moderation_record}")


async def notify_user_of_moderation(user_id: UUID, action: str, reason: str):
    """Notify user of moderation action"""
    logger.info(f"Notifying user {user_id} of {action}: {reason}")


async def start_pipeline(pipeline_name: str, parameters: Optional[Dict[str, Any]]):
    """Start data pipeline"""
    logger.info(f"Starting pipeline {pipeline_name}")


async def stop_pipeline(pipeline_name: str):
    """Stop data pipeline"""
    logger.info(f"Stopping pipeline {pipeline_name}")


async def restart_pipeline(pipeline_name: str, parameters: Optional[Dict[str, Any]]):
    """Restart data pipeline"""
    logger.info(f"Restarting pipeline {pipeline_name}")


async def trigger_pipeline(pipeline_name: str, parameters: Optional[Dict[str, Any]]):
    """Trigger pipeline immediate run"""
    logger.info(f"Triggering pipeline {pipeline_name}")


async def generate_system_report(
    report_id: str,
    report_type: str,
    start_date: datetime,
    end_date: datetime,
    filters: Dict[str, Any],
    format: str,
    include_charts: bool
):
    """Generate system report"""
    logger.info(f"Generating {report_type} report {report_id}")


async def schedule_service_restart(config_key: str, delay_minutes: int):
    """Schedule service restart"""
    logger.info(f"Scheduling restart for config change: {config_key} in {delay_minutes} minutes")


async def log_configuration_change(config_update: Dict[str, Any]):
    """Log configuration change"""
    logger.info(f"Configuration change logged: {config_update}")


async def save_alert_configuration(alert_config: Dict[str, Any]):
    """Save alert configuration"""
    logger.info(f"Saving alert configuration: {alert_config}")


async def test_alert_configuration(alert_id: str):
    """Test alert configuration"""
    logger.info(f"Testing alert configuration: {alert_id}")