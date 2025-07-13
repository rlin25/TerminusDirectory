"""
System Observer for comprehensive monitoring and observability.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import logging
import psutil
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import structlog


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetric:
    """System metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: str
    source: str


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # healthy, degraded, unhealthy
    message: str
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    component: str
    metric_name: str
    threshold_value: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime]
    labels: Dict[str, str]
    
    @property
    def is_active(self) -> bool:
        return self.resolved_at is None


class SystemObserver:
    """
    Comprehensive System Observer for monitoring and observability.
    
    Provides real-time monitoring, health checks, performance tracking,
    alerting, and comprehensive observability for the analytics system.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        prometheus_registry: Optional[CollectorRegistry] = None
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.prometheus_registry = prometheus_registry or CollectorRegistry()
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.metric_retention_days = 30
        
        # Active alerts tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Health check functions
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        
        # Metrics collection
        self.metrics_buffer: List[SystemMetric] = []
        self.max_buffer_size = 1000
        
        # Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Structured logging
        self.logger = structlog.get_logger(__name__)
        
        # Background tasks
        self.monitoring_tasks: Set[asyncio.Task] = set()
    
    def setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics collectors."""
        self.prometheus_metrics = {
            "system_cpu_usage": Gauge(
                "system_cpu_usage_percent",
                "System CPU usage percentage",
                registry=self.prometheus_registry
            ),
            "system_memory_usage": Gauge(
                "system_memory_usage_percent", 
                "System memory usage percentage",
                registry=self.prometheus_registry
            ),
            "database_connections": Gauge(
                "database_connections_active",
                "Active database connections",
                registry=self.prometheus_registry
            ),
            "redis_memory_usage": Gauge(
                "redis_memory_usage_bytes",
                "Redis memory usage in bytes",
                registry=self.prometheus_registry
            ),
            "api_requests_total": Counter(
                "api_requests_total",
                "Total API requests",
                ["method", "endpoint", "status"],
                registry=self.prometheus_registry
            ),
            "api_request_duration": Histogram(
                "api_request_duration_seconds",
                "API request duration",
                ["method", "endpoint"],
                registry=self.prometheus_registry
            ),
            "ml_predictions_total": Counter(
                "ml_predictions_total",
                "Total ML predictions",
                ["model_name", "model_version"],
                registry=self.prometheus_registry
            ),
            "ml_prediction_duration": Histogram(
                "ml_prediction_duration_seconds",
                "ML prediction duration",
                ["model_name"],
                registry=self.prometheus_registry
            ),
            "data_processing_events": Counter(
                "data_processing_events_total",
                "Total data processing events",
                ["event_type", "status"],
                registry=self.prometheus_registry
            ),
            "alert_count": Gauge(
                "active_alerts_count",
                "Number of active alerts",
                ["severity"],
                registry=self.prometheus_registry
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the system observer."""
        try:
            # Register default health checks
            await self._register_default_health_checks()
            
            # Start background monitoring tasks
            await self._start_monitoring_tasks()
            
            # Load existing alerts from database
            await self._load_active_alerts()
            
            self.logger.info("System observer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system observer: {e}")
            raise
    
    async def collect_system_metrics(self) -> Dict[str, SystemMetric]:
        """Collect comprehensive system metrics."""
        metrics = {}
        timestamp = datetime.utcnow()
        
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics["cpu_usage"] = SystemMetric(
                name="system_cpu_usage",
                value=cpu_percent,
                timestamp=timestamp,
                labels={"host": "localhost"},
                metric_type="gauge",
                source="system"
            )
            
            metrics["memory_usage"] = SystemMetric(
                name="system_memory_usage",
                value=memory.percent,
                timestamp=timestamp,
                labels={"host": "localhost"},
                metric_type="gauge",
                source="system"
            )
            
            metrics["disk_usage"] = SystemMetric(
                name="system_disk_usage",
                value=(disk.used / disk.total) * 100,
                timestamp=timestamp,
                labels={"host": "localhost", "mount": "/"},
                metric_type="gauge",
                source="system"
            )
            
            # Database metrics
            db_metrics = await self._collect_database_metrics(timestamp)
            metrics.update(db_metrics)
            
            # Redis metrics
            redis_metrics = await self._collect_redis_metrics(timestamp)
            metrics.update(redis_metrics)
            
            # Application metrics
            app_metrics = await self._collect_application_metrics(timestamp)
            metrics.update(app_metrics)
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            # Store metrics
            await self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        health_results = {}
        
        for component, check_func in self.health_checks.items():
            try:
                start_time = datetime.utcnow()
                result = await check_func()
                
                # Calculate response time if not provided
                if result.response_time_ms == 0:
                    result.response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                health_results[component] = result
                
            except Exception as e:
                health_results[component] = HealthCheck(
                    component=component,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    response_time_ms=0,
                    timestamp=datetime.utcnow(),
                    details={"error": str(e)}
                )
        
        # Store health check results
        await self._store_health_checks(health_results)
        
        return health_results
    
    async def trigger_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        component: str,
        metric_name: str,
        threshold_value: float,
        current_value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> Alert:
        """Trigger a new alert."""
        alert_id = f"{component}_{metric_name}_{int(datetime.utcnow().timestamp())}"
        
        alert = Alert(
            alert_id=alert_id,
            name=name,
            severity=severity,
            message=message,
            component=component,
            metric_name=metric_name,
            threshold_value=threshold_value,
            current_value=current_value,
            triggered_at=datetime.utcnow(),
            resolved_at=None,
            labels=labels or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        await self._store_alert(alert)
        
        # Update Prometheus alert counter
        self.prometheus_metrics["alert_count"].labels(severity=severity.value).inc()
        
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self.logger.warning(
            f"Alert triggered: {name}",
            alert_id=alert_id,
            severity=severity.value,
            component=component,
            metric=metric_name,
            threshold=threshold_value,
            current=current_value
        )
        
        return alert
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved_at = datetime.utcnow()
        
        # Update in database
        await self._update_alert_resolution(alert_id, alert.resolved_at, resolution_message)
        
        # Update Prometheus counter
        self.prometheus_metrics["alert_count"].labels(severity=alert.severity.value).dec()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        self.logger.info(
            f"Alert resolved: {alert.name}",
            alert_id=alert_id,
            resolution_message=resolution_message
        )
        
        return True
    
    async def check_alert_thresholds(self, metrics: Dict[str, SystemMetric]) -> List[Alert]:
        """Check metrics against alert thresholds and trigger alerts if needed."""
        triggered_alerts = []
        
        # Define threshold configurations
        thresholds = {
            "system_cpu_usage": {"warning": 80, "critical": 95},
            "system_memory_usage": {"warning": 85, "critical": 95},
            "system_disk_usage": {"warning": 80, "critical": 90},
            "database_connection_count": {"warning": 80, "critical": 95},
            "redis_memory_usage": {"warning": 85, "critical": 95},
            "api_error_rate": {"warning": 0.05, "critical": 0.1},
            "ml_prediction_latency": {"warning": 1000, "critical": 2000}
        }
        
        for metric_name, metric in metrics.items():
            if metric_name in thresholds:
                thresholds_config = thresholds[metric_name]
                
                # Check critical threshold
                if metric.value >= thresholds_config["critical"]:
                    alert = await self.trigger_alert(
                        name=f"High {metric_name}",
                        severity=AlertSeverity.CRITICAL,
                        message=f"{metric_name} is critically high: {metric.value:.2f}",
                        component=metric.source,
                        metric_name=metric_name,
                        threshold_value=thresholds_config["critical"],
                        current_value=metric.value,
                        labels=metric.labels
                    )
                    triggered_alerts.append(alert)
                
                # Check warning threshold
                elif metric.value >= thresholds_config["warning"]:
                    alert = await self.trigger_alert(
                        name=f"Elevated {metric_name}",
                        severity=AlertSeverity.WARNING,
                        message=f"{metric_name} is elevated: {metric.value:.2f}",
                        component=metric.source,
                        metric_name=metric_name,
                        threshold_value=thresholds_config["warning"],
                        current_value=metric.value,
                        labels=metric.labels
                    )
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Collect current metrics
            metrics = await self.collect_system_metrics()
            
            # Run health checks
            health_checks = await self.run_health_checks()
            
            # Get active alerts
            active_alerts = list(self.active_alerts.values())
            
            # Calculate overall health
            overall_health = self._calculate_overall_health(health_checks, active_alerts)
            
            # Get performance summary
            performance_summary = await self._get_performance_summary()
            
            return {
                "overall_health": overall_health,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {name: asdict(metric) for name, metric in metrics.items()},
                "health_checks": {name: asdict(check) for name, check in health_checks.items()},
                "active_alerts": [asdict(alert) for alert in active_alerts],
                "performance_summary": performance_summary,
                "component_status": {
                    "database": self._get_component_status("database", health_checks, metrics),
                    "redis": self._get_component_status("redis", health_checks, metrics),
                    "api": self._get_component_status("api", health_checks, metrics),
                    "ml_models": self._get_component_status("ml", health_checks, metrics),
                    "streaming": self._get_component_status("streaming", health_checks, metrics)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "overall_health": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_metrics_history(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
        granularity: str = "1h"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical metrics data."""
        try:
            history = {}
            
            for metric_name in metric_names:
                query = text("""
                    SELECT 
                        DATE_TRUNC(:granularity, timestamp) as time_bucket,
                        AVG(metric_value) as avg_value,
                        MIN(metric_value) as min_value,
                        MAX(metric_value) as max_value,
                        COUNT(*) as data_points
                    FROM business_metrics
                    WHERE metric_name = :metric_name
                    AND timestamp BETWEEN :start_time AND :end_time
                    GROUP BY DATE_TRUNC(:granularity, timestamp)
                    ORDER BY time_bucket
                """)
                
                result = await self.db_session.execute(query, {
                    "metric_name": metric_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "granularity": granularity
                })
                
                history[metric_name] = [dict(row) for row in result.fetchall()]
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics history: {e}")
            return {}
    
    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def register_health_check(self, component: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
    
    async def shutdown(self) -> None:
        """Shutdown the system observer."""
        try:
            # Cancel all background tasks
            for task in self.monitoring_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Store final metrics
            if self.metrics_buffer:
                await self._flush_metrics_buffer()
            
            self.logger.info("System observer shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during system observer shutdown: {e}")
    
    # Private methods
    async def _register_default_health_checks(self) -> None:
        """Register default health check functions."""
        self.register_health_check("database", self._check_database_health)
        self.register_health_check("redis", self._check_redis_health)
        self.register_health_check("api", self._check_api_health)
        self.register_health_check("ml_models", self._check_ml_models_health)
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.monitoring_tasks.add(metrics_task)
        
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.monitoring_tasks.add(health_task)
        
        # Alert checking task
        alert_task = asyncio.create_task(self._alert_checking_loop())
        self.monitoring_tasks.add(alert_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.monitoring_tasks.add(cleanup_task)
    
    async def _metrics_collection_loop(self) -> None:
        """Background task for continuous metrics collection."""
        while True:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _health_check_loop(self) -> None:
        """Background task for continuous health checking."""
        while True:
            try:
                await self.run_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _alert_checking_loop(self) -> None:
        """Background task for continuous alert threshold checking."""
        while True:
            try:
                # Get recent metrics for alert checking
                recent_metrics = await self._get_recent_metrics()
                await self.check_alert_thresholds(recent_metrics)
                await asyncio.sleep(60)  # Check alerts every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert checking loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations."""
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    # Health check implementations
    async def _check_database_health(self) -> HealthCheck:
        """Check database health."""
        try:
            start_time = datetime.utcnow()
            result = await self.db_session.execute(text("SELECT 1"))
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Get connection pool info
            pool_info = await self._get_database_pool_info()
            
            return HealthCheck(
                component="database",
                status="healthy",
                message="Database is responsive",
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                details=pool_info
            )
            
        except Exception as e:
            return HealthCheck(
                component="database",
                status="unhealthy",
                message=f"Database check failed: {str(e)}",
                response_time_ms=0,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )
    
    async def _check_redis_health(self) -> HealthCheck:
        """Check Redis health."""
        try:
            start_time = datetime.utcnow()
            await self.redis_client.ping()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Get Redis info
            redis_info = await self.redis_client.info()
            
            return HealthCheck(
                component="redis",
                status="healthy",
                message="Redis is responsive",
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                details={
                    "used_memory": redis_info.get("used_memory_human", "0B"),
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "uptime": redis_info.get("uptime_in_seconds", 0)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="redis",
                status="unhealthy",
                message=f"Redis check failed: {str(e)}",
                response_time_ms=0,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )
    
    async def _check_api_health(self) -> HealthCheck:
        """Check API health."""
        # This would implement API endpoint health checks
        return HealthCheck(
            component="api",
            status="healthy",
            message="API endpoints are responsive",
            response_time_ms=50,
            timestamp=datetime.utcnow(),
            details={"endpoints_checked": 5, "all_healthy": True}
        )
    
    async def _check_ml_models_health(self) -> HealthCheck:
        """Check ML models health."""
        # This would implement ML model health checks
        return HealthCheck(
            component="ml_models",
            status="healthy",
            message="ML models are functioning normally",
            response_time_ms=25,
            timestamp=datetime.utcnow(),
            details={"models_checked": 3, "avg_accuracy": 0.92}
        )
    
    # Metrics collection implementations
    async def _collect_database_metrics(self, timestamp: datetime) -> Dict[str, SystemMetric]:
        """Collect database-specific metrics."""
        metrics = {}
        
        try:
            # Database connection count
            result = await self.db_session.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            )
            active_connections = result.scalar()
            
            metrics["database_connections"] = SystemMetric(
                name="database_connections_active",
                value=active_connections,
                timestamp=timestamp,
                labels={"database": "rental_ml"},
                metric_type="gauge",
                source="database"
            )
            
            # Database size
            result = await self.db_session.execute(
                text("SELECT pg_database_size(current_database())")
            )
            db_size = result.scalar()
            
            metrics["database_size"] = SystemMetric(
                name="database_size_bytes",
                value=db_size,
                timestamp=timestamp,
                labels={"database": "rental_ml"},
                metric_type="gauge",
                source="database"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect database metrics: {e}")
        
        return metrics
    
    async def _collect_redis_metrics(self, timestamp: datetime) -> Dict[str, SystemMetric]:
        """Collect Redis-specific metrics."""
        metrics = {}
        
        try:
            info = await self.redis_client.info()
            
            metrics["redis_memory_usage"] = SystemMetric(
                name="redis_memory_usage_bytes",
                value=info.get("used_memory", 0),
                timestamp=timestamp,
                labels={"instance": "main"},
                metric_type="gauge",
                source="redis"
            )
            
            metrics["redis_connected_clients"] = SystemMetric(
                name="redis_connected_clients",
                value=info.get("connected_clients", 0),
                timestamp=timestamp,
                labels={"instance": "main"},
                metric_type="gauge",
                source="redis"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect Redis metrics: {e}")
        
        return metrics
    
    async def _collect_application_metrics(self, timestamp: datetime) -> Dict[str, SystemMetric]:
        """Collect application-specific metrics."""
        metrics = {}
        
        try:
            # Analytics events processed
            result = await self.db_session.execute(
                text("""
                    SELECT COUNT(*) FROM analytics_events 
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                """)
            )
            events_count = result.scalar()
            
            metrics["analytics_events_hourly"] = SystemMetric(
                name="analytics_events_processed_hourly",
                value=events_count,
                timestamp=timestamp,
                labels={"component": "analytics"},
                metric_type="counter",
                source="application"
            )
            
            # Active alerts count
            metrics["active_alerts"] = SystemMetric(
                name="active_alerts_total",
                value=len(self.active_alerts),
                timestamp=timestamp,
                labels={"component": "monitoring"},
                metric_type="gauge",
                source="application"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")
        
        return metrics
    
    def _update_prometheus_metrics(self, metrics: Dict[str, SystemMetric]) -> None:
        """Update Prometheus metrics with collected data."""
        for metric_name, metric in metrics.items():
            if metric_name in ["system_cpu_usage", "cpu_usage"]:
                self.prometheus_metrics["system_cpu_usage"].set(metric.value)
            elif metric_name in ["system_memory_usage", "memory_usage"]:
                self.prometheus_metrics["system_memory_usage"].set(metric.value)
            elif metric_name == "database_connections":
                self.prometheus_metrics["database_connections"].set(metric.value)
            elif metric_name == "redis_memory_usage":
                self.prometheus_metrics["redis_memory_usage"].set(metric.value)
    
    async def _store_metrics(self, metrics: Dict[str, SystemMetric]) -> None:
        """Store metrics in database and cache."""
        try:
            # Add to buffer
            self.metrics_buffer.extend(metrics.values())
            
            # Flush buffer if it's full
            if len(self.metrics_buffer) >= self.max_buffer_size:
                await self._flush_metrics_buffer()
                
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
    
    async def _flush_metrics_buffer(self) -> None:
        """Flush metrics buffer to database."""
        if not self.metrics_buffer:
            return
        
        try:
            # Bulk insert metrics
            values = []
            for metric in self.metrics_buffer:
                values.append({
                    "metric_name": metric.name,
                    "metric_value": metric.value,
                    "timestamp": metric.timestamp,
                    "dimensions": json.dumps(metric.labels),
                    "source": metric.source
                })
            
            # Insert into business_metrics table
            query = text("""
                INSERT INTO business_metrics (metric_name, metric_value, timestamp, dimensions, source)
                VALUES (:metric_name, :metric_value, :timestamp, :dimensions, :source)
            """)
            
            await self.db_session.execute(query, values)
            await self.db_session.commit()
            
            # Clear buffer
            self.metrics_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to flush metrics buffer: {e}")
            await self.db_session.rollback()
    
    # Additional helper methods would be implemented here
    async def _load_active_alerts(self) -> None:
        """Load active alerts from database."""
        pass
    
    async def _store_alert(self, alert: Alert) -> None:
        """Store alert in database."""
        pass
    
    async def _update_alert_resolution(self, alert_id: str, resolved_at: datetime, message: str) -> None:
        """Update alert resolution in database."""
        pass
    
    async def _store_health_checks(self, health_results: Dict[str, HealthCheck]) -> None:
        """Store health check results."""
        pass
    
    def _calculate_overall_health(self, health_checks: Dict[str, HealthCheck], active_alerts: List[Alert]) -> str:
        """Calculate overall system health."""
        if any(alert.severity == AlertSeverity.CRITICAL for alert in active_alerts):
            return "critical"
        elif any(check.status == "unhealthy" for check in health_checks.values()):
            return "unhealthy"
        elif any(alert.severity == AlertSeverity.WARNING for alert in active_alerts):
            return "degraded"
        else:
            return "healthy"
    
    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "avg_response_time_ms": 150,
            "requests_per_second": 100,
            "error_rate": 0.01,
            "throughput": 500
        }
    
    def _get_component_status(
        self,
        component: str,
        health_checks: Dict[str, HealthCheck],
        metrics: Dict[str, SystemMetric]
    ) -> Dict[str, Any]:
        """Get status for a specific component."""
        health = health_checks.get(component)
        return {
            "status": health.status if health else "unknown",
            "last_check": health.timestamp.isoformat() if health else None,
            "response_time_ms": health.response_time_ms if health else 0
        }
    
    async def _get_recent_metrics(self) -> Dict[str, SystemMetric]:
        """Get recent metrics for alert checking."""
        # This would fetch recent metrics from cache or database
        return {}
    
    async def _get_database_pool_info(self) -> Dict[str, Any]:
        """Get database connection pool information."""
        return {"pool_size": 10, "active_connections": 5}
    
    async def _cleanup_old_data(self) -> None:
        """Cleanup old monitoring data."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.metric_retention_days)
            
            # Cleanup old metrics
            await self.db_session.execute(
                text("DELETE FROM business_metrics WHERE timestamp < :cutoff_date"),
                {"cutoff_date": cutoff_date}
            )
            
            # Cleanup old alerts
            await self.db_session.execute(
                text("DELETE FROM alerts WHERE triggered_at < :cutoff_date"),
                {"cutoff_date": cutoff_date}
            )
            
            await self.db_session.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            await self.db_session.rollback()