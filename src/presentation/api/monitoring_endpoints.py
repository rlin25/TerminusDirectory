"""
Monitoring and Metrics Endpoints for the Rental ML System.

This module provides comprehensive monitoring and metrics capabilities including:
- Real-time system metrics
- Performance monitoring
- Error rate and latency tracking
- ML model performance metrics
- Business metrics and KPIs
- Alert configuration and management
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import asyncio
import psutil

logger = logging.getLogger(__name__)

monitoring_router = APIRouter()


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertStatus(str, Enum):
    """Alert status options"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class TimeRange(str, Enum):
    """Time range options for metrics"""
    LAST_HOUR = "1h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"


class MetricQuery(BaseModel):
    """Request model for metric queries"""
    metric_name: str = Field(..., description="Name of the metric to query")
    start_time: Optional[datetime] = Field(None, description="Start time for query")
    end_time: Optional[datetime] = Field(None, description="End time for query")
    time_range: Optional[TimeRange] = Field(None, description="Predefined time range")
    aggregation: str = Field(default="avg", description="Aggregation method (avg, sum, max, min)")
    group_by: Optional[List[str]] = Field(None, description="Group by labels")
    filters: Optional[Dict[str, str]] = Field(None, description="Metric filters")


class AlertRule(BaseModel):
    """Model for alert rules"""
    name: str = Field(..., description="Alert rule name")
    metric: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., description="Alert condition (>, <, ==, !=)")
    threshold: float = Field(..., description="Alert threshold")
    duration: str = Field(default="5m", description="Duration before alert fires")
    severity: str = Field(default="warning", description="Alert severity")
    description: Optional[str] = Field(None, description="Alert description")
    runbook_url: Optional[str] = Field(None, description="Runbook URL")
    labels: Optional[Dict[str, str]] = Field(None, description="Additional labels")


class MetricData(BaseModel):
    """Model for metric data points"""
    timestamp: datetime
    value: float
    labels: Optional[Dict[str, str]] = None


class MetricResponse(BaseModel):
    """Response model for metrics"""
    metric_name: str
    data_points: List[MetricData]
    summary: Dict[str, float]
    query_info: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class SystemHealthMetrics(BaseModel):
    """Model for system health metrics"""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    load_average: List[float]
    network_io: Dict[str, float]
    database_connections: int
    cache_hit_rate: float
    api_response_time_ms: float
    error_rate_percent: float
    active_users: int


class MLModelMetrics(BaseModel):
    """Model for ML model metrics"""
    model_name: str
    inference_count: int
    average_latency_ms: float
    error_count: int
    accuracy_score: Optional[float] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    f1_score: Optional[float] = None
    last_training_date: Optional[datetime] = None
    model_size_mb: Optional[float] = None


class BusinessMetrics(BaseModel):
    """Model for business metrics"""
    daily_active_users: int
    total_properties: int
    total_searches: int
    total_recommendations: int
    user_engagement_rate: float
    conversion_rate: float
    average_session_duration_minutes: float
    revenue_metrics: Optional[Dict[str, float]] = None


class AlertInstance(BaseModel):
    """Model for alert instances"""
    alert_id: str
    rule_name: str
    status: AlertStatus
    severity: str
    message: str
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    labels: Dict[str, str]
    annotations: Optional[Dict[str, str]] = None


class DashboardMetrics(BaseModel):
    """Model for dashboard metrics summary"""
    system_health: SystemHealthMetrics
    ml_models: List[MLModelMetrics]
    business_metrics: BusinessMetrics
    active_alerts: List[AlertInstance]
    performance_summary: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# Global metrics storage (in production, use proper metrics store like Prometheus)
metrics_store = {}
alert_rules = {}
active_alerts = {}


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.metrics = {}
        self.last_collection_time = {}
    
    async def collect_system_metrics(self) -> SystemHealthMetrics:
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent_per_sec": getattr(network, 'bytes_sent', 0),
                "bytes_recv_per_sec": getattr(network, 'bytes_recv', 0)
            }
            
            # Mock additional metrics
            return SystemHealthMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                disk_usage_percent=disk_percent,
                load_average=list(load_avg),
                network_io=network_io,
                database_connections=45,  # Mock
                cache_hit_rate=0.94,      # Mock
                api_response_time_ms=245.5,  # Mock
                error_rate_percent=1.8,   # Mock
                active_users=1250         # Mock
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics on error
            return SystemHealthMetrics(
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                load_average=[0.0, 0.0, 0.0],
                network_io={"bytes_sent_per_sec": 0, "bytes_recv_per_sec": 0},
                database_connections=0,
                cache_hit_rate=0.0,
                api_response_time_ms=0.0,
                error_rate_percent=0.0,
                active_users=0
            )
    
    async def collect_ml_metrics(self) -> List[MLModelMetrics]:
        """Collect ML model performance metrics"""
        # Mock ML metrics - in production, integrate with actual ML monitoring
        models = [
            MLModelMetrics(
                model_name="hybrid_recommender",
                inference_count=15420,
                average_latency_ms=125.3,
                error_count=12,
                accuracy_score=0.87,
                precision_score=0.85,
                recall_score=0.89,
                f1_score=0.87,
                last_training_date=datetime.now() - timedelta(days=2),
                model_size_mb=245.7
            ),
            MLModelMetrics(
                model_name="search_ranker",
                inference_count=25680,
                average_latency_ms=95.2,
                error_count=8,
                accuracy_score=0.92,
                last_training_date=datetime.now() - timedelta(days=1),
                model_size_mb=156.3
            ),
            MLModelMetrics(
                model_name="content_recommender",
                inference_count=8750,
                average_latency_ms=75.8,
                error_count=3,
                accuracy_score=0.84,
                last_training_date=datetime.now() - timedelta(days=3),
                model_size_mb=89.1
            )
        ]
        return models
    
    async def collect_business_metrics(self) -> BusinessMetrics:
        """Collect business KPI metrics"""
        # Mock business metrics - in production, query from analytics database
        return BusinessMetrics(
            daily_active_users=2150,
            total_properties=125000,
            total_searches=8750,
            total_recommendations=15420,
            user_engagement_rate=0.72,
            conversion_rate=0.045,
            average_session_duration_minutes=12.5,
            revenue_metrics={
                "monthly_recurring_revenue": 125000,
                "average_revenue_per_user": 25.50
            }
        )


# Global metrics collector
metrics_collector = MetricsCollector()


@monitoring_router.get("/health")
async def get_health_status():
    """Quick health check endpoint"""
    try:
        system_metrics = await metrics_collector.collect_system_metrics()
        
        # Determine overall health
        health_score = 100
        issues = []
        
        if system_metrics.cpu_usage_percent > 80:
            health_score -= 20
            issues.append("High CPU usage")
        
        if system_metrics.memory_usage_percent > 85:
            health_score -= 25
            issues.append("High memory usage")
        
        if system_metrics.error_rate_percent > 5:
            health_score -= 30
            issues.append("High error rate")
        
        status_text = "healthy"
        if health_score < 70:
            status_text = "critical"
        elif health_score < 85:
            status_text = "warning"
        
        return {
            "status": status_text,
            "health_score": health_score,
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - 1640995200,  # Mock start time
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "critical",
                "health_score": 0,
                "issues": ["Health check system failure"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@monitoring_router.get("/metrics/system", response_model=SystemHealthMetrics)
async def get_system_metrics():
    """Get current system performance metrics"""
    try:
        return await metrics_collector.collect_system_metrics()
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect system metrics: {str(e)}"
        )


@monitoring_router.get("/metrics/ml", response_model=List[MLModelMetrics])
async def get_ml_metrics():
    """Get ML model performance metrics"""
    try:
        return await metrics_collector.collect_ml_metrics()
    except Exception as e:
        logger.error(f"Failed to get ML metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect ML metrics: {str(e)}"
        )


@monitoring_router.get("/metrics/business", response_model=BusinessMetrics)
async def get_business_metrics():
    """Get business KPI metrics"""
    try:
        return await metrics_collector.collect_business_metrics()
    except Exception as e:
        logger.error(f"Failed to get business metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect business metrics: {str(e)}"
        )


@monitoring_router.get("/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get comprehensive dashboard metrics"""
    try:
        logger.info("Collecting dashboard metrics")
        
        # Collect all metrics concurrently
        system_health, ml_models, business_metrics = await asyncio.gather(
            metrics_collector.collect_system_metrics(),
            metrics_collector.collect_ml_metrics(),
            metrics_collector.collect_business_metrics()
        )
        
        # Get active alerts (mock data)
        active_alerts_list = [
            AlertInstance(
                alert_id=str(uuid4()),
                rule_name="high_error_rate",
                status=AlertStatus.ACTIVE,
                severity="warning",
                message="API error rate above threshold",
                fired_at=datetime.now() - timedelta(minutes=15),
                labels={"service": "api", "environment": "production"}
            )
        ]
        
        # Performance summary
        performance_summary = {
            "overall_performance_score": 87.5,
            "availability_percentage": 99.9,
            "total_requests_today": 125000,
            "successful_requests_today": 122850,
            "failed_requests_today": 2150,
            "average_response_time_today": 245.5
        }
        
        return DashboardMetrics(
            system_health=system_health,
            ml_models=ml_models,
            business_metrics=business_metrics,
            active_alerts=active_alerts_list,
            performance_summary=performance_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect dashboard metrics: {str(e)}"
        )


@monitoring_router.post("/metrics/query", response_model=MetricResponse)
async def query_metrics(request: MetricQuery):
    """Query specific metrics with time range and filters"""
    try:
        logger.info(f"Querying metric: {request.metric_name}")
        
        # Determine time range
        if request.time_range:
            if request.time_range == TimeRange.LAST_HOUR:
                start_time = datetime.now() - timedelta(hours=1)
                end_time = datetime.now()
            elif request.time_range == TimeRange.LAST_DAY:
                start_time = datetime.now() - timedelta(days=1)
                end_time = datetime.now()
            elif request.time_range == TimeRange.LAST_WEEK:
                start_time = datetime.now() - timedelta(weeks=1)
                end_time = datetime.now()
            elif request.time_range == TimeRange.LAST_MONTH:
                start_time = datetime.now() - timedelta(days=30)
                end_time = datetime.now()
        else:
            start_time = request.start_time or (datetime.now() - timedelta(hours=1))
            end_time = request.end_time or datetime.now()
        
        # Generate mock time series data
        data_points = []
        current_time = start_time
        time_delta = (end_time - start_time) / 50  # 50 data points
        
        base_value = 100
        if "cpu" in request.metric_name.lower():
            base_value = 45
        elif "memory" in request.metric_name.lower():
            base_value = 67
        elif "response_time" in request.metric_name.lower():
            base_value = 245
        
        while current_time <= end_time:
            # Add some variance to make it realistic
            import random
            variance = random.uniform(-0.1, 0.1)
            value = base_value * (1 + variance)
            
            data_points.append(MetricData(
                timestamp=current_time,
                value=value,
                labels=request.filters
            ))
            current_time += time_delta
        
        # Calculate summary statistics
        values = [dp.value for dp in data_points]
        summary = {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": sum(values) / len(values) if values else 0,
            "count": len(values)
        }
        
        if request.aggregation == "sum":
            summary["aggregated_value"] = sum(values)
        elif request.aggregation == "max":
            summary["aggregated_value"] = max(values) if values else 0
        elif request.aggregation == "min":
            summary["aggregated_value"] = min(values) if values else 0
        else:  # avg
            summary["aggregated_value"] = summary["avg"]
        
        return MetricResponse(
            metric_name=request.metric_name,
            data_points=data_points,
            summary=summary,
            query_info={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "aggregation": request.aggregation,
                "filters": request.filters or {},
                "group_by": request.group_by or []
            }
        )
        
    except Exception as e:
        logger.error(f"Metric query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metric query failed: {str(e)}"
        )


@monitoring_router.get("/alerts")
async def get_alerts(
    status: Optional[AlertStatus] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(default=50, ge=1, le=500)
):
    """Get current alerts"""
    try:
        # Mock alerts data
        alerts = [
            AlertInstance(
                alert_id=str(uuid4()),
                rule_name="high_error_rate",
                status=AlertStatus.ACTIVE,
                severity="warning",
                message="API error rate above 2% for 5 minutes",
                fired_at=datetime.now() - timedelta(minutes=15),
                labels={"service": "api", "environment": "production"},
                annotations={"runbook": "https://docs.example.com/runbooks/high-error-rate"}
            ),
            AlertInstance(
                alert_id=str(uuid4()),
                rule_name="high_memory_usage",
                status=AlertStatus.ACKNOWLEDGED,
                severity="critical",
                message="Memory usage above 85% for 10 minutes",
                fired_at=datetime.now() - timedelta(hours=1),
                acknowledged_at=datetime.now() - timedelta(minutes=30),
                labels={"instance": "server-1", "environment": "production"}
            ),
            AlertInstance(
                alert_id=str(uuid4()),
                rule_name="ml_model_accuracy_drop",
                status=AlertStatus.RESOLVED,
                severity="warning",
                message="Recommendation model accuracy below 80%",
                fired_at=datetime.now() - timedelta(hours=3),
                resolved_at=datetime.now() - timedelta(hours=1),
                labels={"model": "hybrid_recommender", "environment": "production"}
            )
        ]
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Limit results
        alerts = alerts[:limit]
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "summary": {
                "active": len([a for a in alerts if a.status == AlertStatus.ACTIVE]),
                "acknowledged": len([a for a in alerts if a.status == AlertStatus.ACKNOWLEDGED]),
                "resolved": len([a for a in alerts if a.status == AlertStatus.RESOLVED])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alerts: {str(e)}"
        )


@monitoring_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    background_tasks: BackgroundTasks
):
    """Acknowledge an alert"""
    try:
        logger.info(f"Acknowledging alert: {alert_id}")
        
        # In production, update alert in database
        background_tasks.add_task(
            update_alert_status,
            alert_id=alert_id,
            status=AlertStatus.ACKNOWLEDGED,
            acknowledged_by="admin"  # Get from auth context
        )
        
        return {
            "message": f"Alert {alert_id} acknowledged successfully",
            "alert_id": alert_id,
            "acknowledged_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to acknowledge alert: {str(e)}"
        )


@monitoring_router.post("/alerts/rules", response_model=Dict[str, Any])
async def create_alert_rule(
    rule: AlertRule,
    background_tasks: BackgroundTasks
):
    """Create a new alert rule"""
    try:
        logger.info(f"Creating alert rule: {rule.name}")
        
        rule_id = str(uuid4())
        
        # Validate metric exists
        valid_metrics = [
            "cpu_usage_percent", "memory_usage_percent", "api_response_time_ms",
            "error_rate_percent", "ml_model_accuracy", "database_connections"
        ]
        
        if rule.metric not in valid_metrics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metric: {rule.metric}. Valid metrics: {valid_metrics}"
            )
        
        # Create rule configuration
        rule_config = {
            "rule_id": rule_id,
            "name": rule.name,
            "metric": rule.metric,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "duration": rule.duration,
            "severity": rule.severity,
            "description": rule.description,
            "runbook_url": rule.runbook_url,
            "labels": rule.labels or {},
            "created_at": datetime.now(),
            "enabled": True
        }
        
        # Save rule (in production, save to database)
        alert_rules[rule_id] = rule_config
        
        # Schedule rule evaluation
        background_tasks.add_task(
            schedule_alert_evaluation,
            rule_id=rule_id,
            rule_config=rule_config
        )
        
        return {
            "message": f"Alert rule '{rule.name}' created successfully",
            "rule_id": rule_id,
            "rule_name": rule.name,
            "enabled": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert rule: {str(e)}"
        )


@monitoring_router.get("/metrics/export/prometheus")
async def export_prometheus_metrics():
    """Export metrics in Prometheus format"""
    try:
        logger.info("Exporting metrics in Prometheus format")
        
        # Collect current metrics
        system_metrics = await metrics_collector.collect_system_metrics()
        ml_metrics = await metrics_collector.collect_ml_metrics()
        business_metrics = await metrics_collector.collect_business_metrics()
        
        # Generate Prometheus format
        prometheus_output = []
        
        # System metrics
        prometheus_output.extend([
            f"# HELP cpu_usage_percent CPU usage percentage",
            f"# TYPE cpu_usage_percent gauge",
            f"cpu_usage_percent {system_metrics.cpu_usage_percent}",
            "",
            f"# HELP memory_usage_percent Memory usage percentage",
            f"# TYPE memory_usage_percent gauge",
            f"memory_usage_percent {system_metrics.memory_usage_percent}",
            "",
            f"# HELP api_response_time_ms API response time in milliseconds",
            f"# TYPE api_response_time_ms gauge",
            f"api_response_time_ms {system_metrics.api_response_time_ms}",
            ""
        ])
        
        # ML model metrics
        for model in ml_metrics:
            prometheus_output.extend([
                f"# HELP ml_model_inference_count Total number of inferences",
                f"# TYPE ml_model_inference_count counter",
                f"ml_model_inference_count{{model=\"{model.model_name}\"}} {model.inference_count}",
                "",
                f"# HELP ml_model_latency_ms Average inference latency",
                f"# TYPE ml_model_latency_ms gauge",
                f"ml_model_latency_ms{{model=\"{model.model_name}\"}} {model.average_latency_ms}",
                ""
            ])
            
            if model.accuracy_score is not None:
                prometheus_output.extend([
                    f"# HELP ml_model_accuracy Model accuracy score",
                    f"# TYPE ml_model_accuracy gauge",
                    f"ml_model_accuracy{{model=\"{model.model_name}\"}} {model.accuracy_score}",
                    ""
                ])
        
        # Business metrics
        prometheus_output.extend([
            f"# HELP daily_active_users Number of daily active users",
            f"# TYPE daily_active_users gauge",
            f"daily_active_users {business_metrics.daily_active_users}",
            "",
            f"# HELP user_engagement_rate User engagement rate",
            f"# TYPE user_engagement_rate gauge",
            f"user_engagement_rate {business_metrics.user_engagement_rate}",
            ""
        ])
        
        return StreamingResponse(
            iter(["\n".join(prometheus_output)]),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=metrics.txt"}
        )
        
    except Exception as e:
        logger.error(f"Failed to export Prometheus metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export metrics: {str(e)}"
        )


@monitoring_router.websocket("/realtime")
async def realtime_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming"""
    await websocket.accept()
    
    try:
        logger.info("Client connected to real-time metrics stream")
        
        while True:
            # Collect current metrics
            system_metrics = await metrics_collector.collect_system_metrics()
            ml_metrics = await metrics_collector.collect_ml_metrics()
            business_metrics = await metrics_collector.collect_business_metrics()
            
            # Prepare real-time data
            realtime_data = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": system_metrics.cpu_usage_percent,
                    "memory_percent": system_metrics.memory_usage_percent,
                    "api_response_time": system_metrics.api_response_time_ms,
                    "error_rate": system_metrics.error_rate_percent,
                    "active_users": system_metrics.active_users
                },
                "ml_models": [
                    {
                        "name": model.model_name,
                        "latency": model.average_latency_ms,
                        "error_count": model.error_count,
                        "accuracy": model.accuracy_score
                    }
                    for model in ml_metrics
                ],
                "business": {
                    "daily_active_users": business_metrics.daily_active_users,
                    "engagement_rate": business_metrics.user_engagement_rate,
                    "conversion_rate": business_metrics.conversion_rate
                }
            }
            
            # Send data to client
            await websocket.send_text(json.dumps(realtime_data))
            
            # Wait before next update
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        logger.info("Client disconnected from real-time metrics stream")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")


@monitoring_router.get("/performance/report")
async def get_performance_report(
    start_date: Optional[datetime] = Query(None, description="Report start date"),
    end_date: Optional[datetime] = Query(None, description="Report end date"),
    include_trends: bool = Query(default=True, description="Include trend analysis")
):
    """Generate comprehensive performance report"""
    try:
        logger.info("Generating performance report")
        
        # Default to last 24 hours if no dates provided
        if not start_date:
            start_date = datetime.now() - timedelta(days=1)
        if not end_date:
            end_date = datetime.now()
        
        # Generate report data (mock)
        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_hours": (end_date - start_date).total_seconds() / 3600
            },
            "system_performance": {
                "average_cpu_usage": 45.2,
                "peak_cpu_usage": 78.5,
                "average_memory_usage": 67.8,
                "peak_memory_usage": 89.2,
                "average_response_time": 245.5,
                "p95_response_time": 580.2,
                "p99_response_time": 1250.0,
                "uptime_percentage": 99.95
            },
            "api_performance": {
                "total_requests": 125000,
                "successful_requests": 122850,
                "failed_requests": 2150,
                "error_rate": 1.72,
                "requests_per_second_avg": 34.7,
                "requests_per_second_peak": 125.3
            },
            "ml_performance": {
                "total_inferences": 45250,
                "average_inference_time": 125.3,
                "model_accuracy_avg": 0.875,
                "recommendation_ctr": 0.23,
                "search_relevance_score": 0.92
            },
            "user_metrics": {
                "unique_users": 15420,
                "new_users": 347,
                "session_duration_avg": 12.5,
                "bounce_rate": 0.32,
                "engagement_score": 0.72
            },
            "alerts_summary": {
                "total_alerts": 15,
                "critical_alerts": 2,
                "warning_alerts": 8,
                "info_alerts": 5,
                "resolved_alerts": 12,
                "mean_time_to_resolution": 25.5
            }
        }
        
        if include_trends:
            report["trends"] = {
                "response_time_trend": "improving",
                "error_rate_trend": "stable",
                "user_growth_trend": "increasing",
                "model_performance_trend": "stable",
                "resource_usage_trend": "increasing"
            }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


# Background tasks
async def update_alert_status(alert_id: str, status: AlertStatus, acknowledged_by: str):
    """Update alert status"""
    logger.info(f"Updating alert {alert_id} status to {status}")


async def schedule_alert_evaluation(rule_id: str, rule_config: Dict[str, Any]):
    """Schedule alert rule evaluation"""
    logger.info(f"Scheduling evaluation for alert rule {rule_id}")


# Startup task to begin metrics collection
@monitoring_router.on_event("startup")
async def start_metrics_collection():
    """Start background metrics collection"""
    logger.info("Starting background metrics collection")
    
    async def collect_metrics_periodically():
        while True:
            try:
                # Collect and store metrics
                timestamp = datetime.now()
                
                system_metrics = await metrics_collector.collect_system_metrics()
                ml_metrics = await metrics_collector.collect_ml_metrics()
                business_metrics = await metrics_collector.collect_business_metrics()
                
                # Store in metrics store (in production, use proper time series DB)
                metrics_store[timestamp] = {
                    "system": system_metrics.dict(),
                    "ml": [m.dict() for m in ml_metrics],
                    "business": business_metrics.dict()
                }
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.now() - timedelta(hours=24)
                metrics_store = {
                    k: v for k, v in metrics_store.items() 
                    if k > cutoff_time
                }
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    # Start background task
    asyncio.create_task(collect_metrics_periodically())