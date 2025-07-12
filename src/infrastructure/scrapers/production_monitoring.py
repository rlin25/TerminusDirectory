"""
Production monitoring and alerting system for rental property scraping.

This module provides comprehensive monitoring, metrics collection, alerting,
and dashboarding capabilities for the production scraping system.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp
import psutil
import os

from .config import get_config, ProductionScrapingConfig

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Represents an alert"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    metric_name: str
    threshold_value: float
    current_value: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    """Represents a metric"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


@dataclass
class HealthCheck:
    """Represents a health check result"""
    name: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores metrics"""
    
    def __init__(self, config: ProductionScrapingConfig = None):
        self.config = config or get_config()
        self.metrics: Dict[str, List[Metric]] = {}
        self.metric_retention = timedelta(hours=24)
        
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Dict[str, str] = None,
        description: str = ""
    ):
        """Record a metric"""
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            description=description
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        # Clean old metrics
        self._clean_old_metrics(name)
        
        logger.debug(f"Recorded metric {name}: {value}")
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        current_value = self.get_latest_metric_value(name, labels) or 0
        self.record_metric(name, current_value + 1, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def get_latest_metric_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get the latest value for a metric"""
        
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        # Filter by labels if provided
        if labels:
            matching_metrics = [
                m for m in self.metrics[name]
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]
            if matching_metrics:
                return matching_metrics[-1].value
            return None
        
        return self.metrics[name][-1].value
    
    def get_metric_history(
        self,
        name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        labels: Dict[str, str] = None
    ) -> List[Metric]:
        """Get metric history within time range"""
        
        if name not in self.metrics:
            return []
        
        metrics = self.metrics[name]
        
        # Filter by time range
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        # Filter by labels
        if labels:
            metrics = [
                m for m in metrics
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]
        
        return metrics
    
    def _clean_old_metrics(self, name: str):
        """Remove old metrics beyond retention period"""
        
        if name not in self.metrics:
            return
        
        cutoff_time = datetime.utcnow() - self.metric_retention
        self.metrics[name] = [
            m for m in self.metrics[name]
            if m.timestamp > cutoff_time
        ]
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        
        summary = {}
        
        for name, metrics in self.metrics.items():
            if not metrics:
                continue
            
            latest = metrics[-1]
            
            # Calculate statistics for recent metrics (last hour)
            recent_cutoff = datetime.utcnow() - timedelta(hours=1)
            recent_metrics = [m for m in metrics if m.timestamp > recent_cutoff]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary[name] = {
                    'latest_value': latest.value,
                    'latest_timestamp': latest.timestamp.isoformat(),
                    'count': len(recent_metrics),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'type': latest.metric_type.value,
                    'description': latest.description
                }
        
        return summary


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: ProductionScrapingConfig = None, metrics_collector: MetricsCollector = None):
        self.config = config or get_config()
        self.metrics_collector = metrics_collector or MetricsCollector(config)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        
        # Alert rules configuration
        self.alert_rules = self._load_alert_rules()
        
    def _load_alert_rules(self) -> List[Dict[str, Any]]:
        """Load alert rules configuration"""
        
        return [
            {
                'name': 'high_error_rate',
                'metric': 'scraping_error_rate',
                'threshold': self.config.monitoring.alert_on_error_rate,
                'severity': AlertSeverity.HIGH,
                'description': 'Scraping error rate is above threshold'
            },
            {
                'name': 'slow_response_time',
                'metric': 'scraping_avg_response_time',
                'threshold': self.config.monitoring.alert_on_slow_response,
                'severity': AlertSeverity.MEDIUM,
                'description': 'Average response time is above threshold'
            },
            {
                'name': 'low_property_count',
                'metric': 'properties_scraped_per_hour',
                'threshold': 10,  # Minimum properties per hour
                'severity': AlertSeverity.MEDIUM,
                'description': 'Property scraping rate is below threshold',
                'operator': 'less_than'
            },
            {
                'name': 'high_duplicate_rate',
                'metric': 'duplicate_detection_rate',
                'threshold': 0.5,  # 50% duplicates
                'severity': AlertSeverity.LOW,
                'description': 'Duplicate detection rate is high'
            },
            {
                'name': 'database_connection_failed',
                'metric': 'database_connection_status',
                'threshold': 0,  # 0 = disconnected
                'severity': AlertSeverity.CRITICAL,
                'description': 'Database connection failed'
            }
        ]
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler"""
        self.notification_handlers.append(handler)
    
    async def check_alerts(self):
        """Check all alert rules against current metrics"""
        
        for rule in self.alert_rules:
            try:
                await self._check_alert_rule(rule)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    async def _check_alert_rule(self, rule: Dict[str, Any]):
        """Check a specific alert rule"""
        
        metric_name = rule['metric']
        threshold = rule['threshold']
        operator = rule.get('operator', 'greater_than')
        
        # Get current metric value
        current_value = self.metrics_collector.get_latest_metric_value(metric_name)
        
        if current_value is None:
            return
        
        # Check threshold
        should_alert = False
        
        if operator == 'greater_than':
            should_alert = current_value > threshold
        elif operator == 'less_than':
            should_alert = current_value < threshold
        elif operator == 'equals':
            should_alert = current_value == threshold
        
        alert_id = f"{rule['name']}_{metric_name}"
        
        if should_alert:
            # Create or update alert
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    title=rule['name'].replace('_', ' ').title(),
                    description=rule['description'],
                    severity=rule['severity'],
                    source='scraping_monitor',
                    metric_name=metric_name,
                    threshold_value=threshold,
                    current_value=current_value
                )
                
                self.active_alerts[alert_id] = alert
                await self._send_alert_notification(alert)
                
                logger.warning(f"Alert triggered: {alert.title} - {alert.description}")
        else:
            # Resolve alert if it exists
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                
                await self._send_alert_resolution_notification(alert)
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert.title}")
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification"""
        
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, 'triggered')
                else:
                    handler(alert, 'triggered')
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    async def _send_alert_resolution_notification(self, alert: Alert):
        """Send alert resolution notification"""
        
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, 'resolved')
                else:
                    handler(alert, 'resolved')
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return self.alert_history[-limit:]


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self, config: ProductionScrapingConfig = None):
        self.config = config or get_config()
        self.health_checks: Dict[str, HealthCheck] = {}
    
    async def check_database_health(self, database_connector) -> HealthCheck:
        """Check database health"""
        
        start_time = time.time()
        
        try:
            if database_connector and database_connector.connection_pool:
                # Simple query to test connection
                async with database_connector.connection_pool.acquire() as connection:
                    await connection.fetchval("SELECT 1")
                
                response_time = (time.time() - start_time) * 1000
                
                health_check = HealthCheck(
                    name='database',
                    status='healthy',
                    message='Database connection is working',
                    response_time_ms=response_time
                )
            else:
                health_check = HealthCheck(
                    name='database',
                    status='critical',
                    message='Database connector not initialized'
                )
                
        except Exception as e:
            health_check = HealthCheck(
                name='database',
                status='critical',
                message=f'Database connection failed: {str(e)}',
                response_time_ms=(time.time() - start_time) * 1000
            )
        
        self.health_checks['database'] = health_check
        return health_check
    
    async def check_redis_health(self, redis_client) -> HealthCheck:
        """Check Redis health"""
        
        start_time = time.time()
        
        try:
            if redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, redis_client.ping
                )
                
                response_time = (time.time() - start_time) * 1000
                
                health_check = HealthCheck(
                    name='redis',
                    status='healthy',
                    message='Redis connection is working',
                    response_time_ms=response_time
                )
            else:
                health_check = HealthCheck(
                    name='redis',
                    status='warning',
                    message='Redis client not configured'
                )
                
        except Exception as e:
            health_check = HealthCheck(
                name='redis',
                status='critical',
                message=f'Redis connection failed: {str(e)}',
                response_time_ms=(time.time() - start_time) * 1000
            )
        
        self.health_checks['redis'] = health_check
        return health_check
    
    def check_system_resources(self) -> HealthCheck:
        """Check system resource usage"""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine status
            status = 'healthy'
            messages = []
            
            if cpu_percent > 80:
                status = 'warning'
                messages.append(f'High CPU usage: {cpu_percent:.1f}%')
            
            if memory_percent > 80:
                status = 'warning'
                messages.append(f'High memory usage: {memory_percent:.1f}%')
            
            if disk_percent > 90:
                status = 'critical'
                messages.append(f'High disk usage: {disk_percent:.1f}%')
            
            if status == 'healthy':
                message = 'System resources are within normal limits'
            else:
                message = '; '.join(messages)
            
            health_check = HealthCheck(
                name='system_resources',
                status=status,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent
                }
            )
            
        except Exception as e:
            health_check = HealthCheck(
                name='system_resources',
                status='critical',
                message=f'Failed to check system resources: {str(e)}'
            )
        
        self.health_checks['system_resources'] = health_check
        return health_check
    
    async def check_external_services(self) -> HealthCheck:
        """Check external service connectivity"""
        
        external_services = [
            'https://www.apartments.com',
            'https://www.rentals.com',
            'https://www.rent.com'
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for service in external_services:
                try:
                    start_time = time.time()
                    async with session.get(service, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            results.append({
                                'service': service,
                                'status': 'healthy',
                                'response_time_ms': response_time
                            })
                        else:
                            results.append({
                                'service': service,
                                'status': 'warning',
                                'message': f'HTTP {response.status}'
                            })
                            
                except Exception as e:
                    results.append({
                        'service': service,
                        'status': 'critical',
                        'message': str(e)
                    })
        
        # Determine overall status
        healthy_count = sum(1 for r in results if r['status'] == 'healthy')
        total_count = len(results)
        
        if healthy_count == total_count:
            status = 'healthy'
            message = 'All external services are accessible'
        elif healthy_count > 0:
            status = 'warning'
            message = f'{healthy_count}/{total_count} external services are accessible'
        else:
            status = 'critical'
            message = 'No external services are accessible'
        
        health_check = HealthCheck(
            name='external_services',
            status=status,
            message=message,
            details={'services': results}
        )
        
        self.health_checks['external_services'] = health_check
        return health_check
    
    def get_overall_health(self) -> HealthCheck:
        """Get overall system health"""
        
        if not self.health_checks:
            return HealthCheck(
                name='overall',
                status='warning',
                message='No health checks performed yet'
            )
        
        # Determine overall status
        critical_count = sum(1 for hc in self.health_checks.values() if hc.status == 'critical')
        warning_count = sum(1 for hc in self.health_checks.values() if hc.status == 'warning')
        
        if critical_count > 0:
            status = 'critical'
            message = f'{critical_count} critical issues detected'
        elif warning_count > 0:
            status = 'warning'
            message = f'{warning_count} warnings detected'
        else:
            status = 'healthy'
            message = 'All systems are healthy'
        
        return HealthCheck(
            name='overall',
            status=status,
            message=message,
            details={name: hc.status for name, hc in self.health_checks.items()}
        )


class ProductionMonitor:
    """Main production monitoring system"""
    
    def __init__(
        self,
        config: ProductionScrapingConfig = None,
        database_connector=None,
        redis_client=None
    ):
        self.config = config or get_config()
        self.database_connector = database_connector
        self.redis_client = redis_client
        
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config, self.metrics_collector)
        self.health_checker = HealthChecker(config)
        
        self.monitoring_active = False
        self.monitoring_interval = 60  # 1 minute
        
        # Setup notification handlers
        self._setup_notification_handlers()
        
        logger.info("Initialized production monitoring system")
    
    def _setup_notification_handlers(self):
        """Setup notification handlers"""
        
        # Email notifications
        if self.config.monitoring.enable_alerts:
            self.alert_manager.add_notification_handler(self._email_notification_handler)
        
        # Log notifications
        self.alert_manager.add_notification_handler(self._log_notification_handler)
    
    async def _email_notification_handler(self, alert: Alert, action: str):
        """Handle email notifications"""
        
        # Implementation would depend on your email configuration
        logger.info(f"Email notification: Alert {alert.title} was {action}")
    
    async def _log_notification_handler(self, alert: Alert, action: str):
        """Handle log notifications"""
        
        if action == 'triggered':
            logger.warning(
                f"ALERT TRIGGERED: {alert.title} - {alert.description} "
                f"(Current: {alert.current_value}, Threshold: {alert.threshold_value})"
            )
        elif action == 'resolved':
            logger.info(f"ALERT RESOLVED: {alert.title}")
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        
        self.monitoring_active = True
        logger.info("Starting production monitoring")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                await self._collect_system_metrics()
                
                # Perform health checks
                await self._perform_health_checks()
                
                # Check alerts
                await self.alert_manager.check_alerts()
                
                # Sleep until next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Short sleep on error
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        
        # System resource metrics
        health_check = self.health_checker.check_system_resources()
        
        if health_check.details:
            self.metrics_collector.set_gauge('system_cpu_percent', health_check.details['cpu_percent'])
            self.metrics_collector.set_gauge('system_memory_percent', health_check.details['memory_percent'])
            self.metrics_collector.set_gauge('system_disk_percent', health_check.details['disk_percent'])
        
        # Database metrics
        if self.database_connector:
            try:
                stats = await self.database_connector.get_property_stats()
                self.metrics_collector.set_gauge('total_properties', stats['total_properties'])
                self.metrics_collector.set_gauge('recent_properties_24h', stats['recent_properties_24h'])
                self.metrics_collector.set_gauge('total_duplicates', stats['total_duplicates'])
            except Exception as e:
                logger.error(f"Error collecting database metrics: {e}")
    
    async def _perform_health_checks(self):
        """Perform all health checks"""
        
        # Database health
        await self.health_checker.check_database_health(self.database_connector)
        
        # Redis health
        await self.health_checker.check_redis_health(self.redis_client)
        
        # System resources
        self.health_checker.check_system_resources()
        
        # External services
        await self.health_checker.check_external_services()
        
        # Update health metrics
        overall_health = self.health_checker.get_overall_health()
        health_score = 1 if overall_health.status == 'healthy' else 0
        self.metrics_collector.set_gauge('overall_health_status', health_score)
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("Stopping production monitoring")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_health': self.health_checker.get_overall_health().__dict__,
            'health_checks': {
                name: hc.__dict__ for name, hc in self.health_checker.health_checks.items()
            },
            'active_alerts': [alert.__dict__ for alert in self.alert_manager.get_active_alerts()],
            'metrics_summary': self.metrics_collector.get_all_metrics_summary(),
            'system_info': {
                'monitoring_active': self.monitoring_active,
                'uptime': self._get_system_uptime(),
                'python_version': self._get_python_version()
            }
        }
    
    def _get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_hours = uptime_seconds / 3600
            return f"{uptime_hours:.1f} hours"
        except:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Event callbacks for scraping orchestrator
    async def on_job_started(self, data: Dict[str, Any]):
        """Handle job started event"""
        self.metrics_collector.increment_counter('scraping_jobs_started')
    
    async def on_job_completed(self, data: Dict[str, Any]):
        """Handle job completed event"""
        job = data['job']
        
        self.metrics_collector.increment_counter('scraping_jobs_completed')
        self.metrics_collector.set_gauge('properties_found_last_job', job.properties_found)
        self.metrics_collector.set_gauge('properties_valid_last_job', job.properties_valid)
        
        if job.duration():
            duration_seconds = job.duration().total_seconds()
            self.metrics_collector.set_gauge('job_duration_seconds', duration_seconds)
            
            # Calculate properties per hour
            if duration_seconds > 0:
                properties_per_hour = (job.properties_valid / duration_seconds) * 3600
                self.metrics_collector.set_gauge('properties_scraped_per_hour', properties_per_hour)
    
    async def on_job_failed(self, data: Dict[str, Any]):
        """Handle job failed event"""
        self.metrics_collector.increment_counter('scraping_jobs_failed')
    
    async def on_property_found(self, data: Dict[str, Any]):
        """Handle property found event"""
        validation_result = data.get('validation_result')
        
        self.metrics_collector.increment_counter('properties_processed')
        
        if validation_result and validation_result.is_valid:
            self.metrics_collector.increment_counter('properties_valid')
            self.metrics_collector.set_gauge('data_quality_score_latest', validation_result.score)
        else:
            self.metrics_collector.increment_counter('properties_invalid')
        
        # Track duplicates
        if validation_result and 'potential_duplicates' in validation_result.cleaned_data:
            self.metrics_collector.increment_counter('duplicates_detected')