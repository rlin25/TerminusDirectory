#!/usr/bin/env python3
"""
Database Monitoring and Health Check System for Rental ML System
Created: 2025-07-12

This script provides comprehensive database monitoring including:
- PostgreSQL performance metrics and health checks
- Redis monitoring and memory usage tracking
- Connection pool monitoring
- Query performance analysis
- Automated alerting and reporting
- Database health dashboards
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncpg
import redis.asyncio as redis
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    timestamp: str
    database_type: str  # 'postgresql' or 'redis'
    
    # Connection metrics
    active_connections: int = 0
    max_connections: int = 0
    connection_usage_percent: float = 0.0
    
    # Performance metrics
    queries_per_second: float = 0.0
    slow_queries: int = 0
    cache_hit_ratio: float = 0.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0
    
    # Database-specific metrics
    database_size_mb: float = 0.0
    table_count: int = 0
    index_count: int = 0
    
    # Health indicators
    is_healthy: bool = True
    health_issues: List[str] = None
    
    def __post_init__(self):
        if self.health_issues is None:
            self.health_issues = []


@dataclass
class AlertRule:
    """Alert rule configuration"""
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str    # 'critical', 'warning', 'info'
    description: str
    enabled: bool = True


class PostgreSQLMonitor:
    """PostgreSQL database monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.metrics_history: List[DatabaseMetrics] = []
    
    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                min_size=2,
                max_size=10
            )
            logger.info("PostgreSQL monitor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL monitor: {e}")
            raise
    
    async def close(self):
        """Close connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
    
    async def collect_metrics(self) -> DatabaseMetrics:
        """Collect comprehensive PostgreSQL metrics"""
        timestamp = datetime.utcnow().isoformat()
        metrics = DatabaseMetrics(timestamp=timestamp, database_type='postgresql')
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Connection metrics
                connection_stats = await self._get_connection_stats(conn)
                metrics.active_connections = connection_stats['active_connections']
                metrics.max_connections = connection_stats['max_connections']
                metrics.connection_usage_percent = (
                    metrics.active_connections / metrics.max_connections * 100
                    if metrics.max_connections > 0 else 0
                )
                
                # Performance metrics
                perf_stats = await self._get_performance_stats(conn)
                metrics.queries_per_second = perf_stats['queries_per_second']
                metrics.slow_queries = perf_stats['slow_queries']
                metrics.cache_hit_ratio = perf_stats['cache_hit_ratio']
                
                # Database size and structure
                db_stats = await self._get_database_stats(conn)
                metrics.database_size_mb = db_stats['database_size_mb']
                metrics.table_count = db_stats['table_count']
                metrics.index_count = db_stats['index_count']
                
                # System resource usage
                system_stats = await self._get_system_stats()
                metrics.memory_usage_mb = system_stats['memory_usage_mb']
                metrics.memory_usage_percent = system_stats['memory_usage_percent']
                metrics.cpu_usage_percent = system_stats['cpu_usage_percent']
                
                # Health assessment
                health_issues = await self._assess_health(metrics)
                metrics.health_issues = health_issues
                metrics.is_healthy = len(health_issues) == 0
        
        except Exception as e:
            logger.error(f"Failed to collect PostgreSQL metrics: {e}")
            metrics.is_healthy = False
            metrics.health_issues = [f"Metrics collection failed: {str(e)}"]
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    async def _get_connection_stats(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Get connection statistics"""
        try:
            # Active connections
            active_result = await conn.fetchval("""
                SELECT count(*) FROM pg_stat_activity 
                WHERE state = 'active' AND pid != pg_backend_pid()
            """)
            
            # Max connections
            max_result = await conn.fetchval("SHOW max_connections")
            
            return {
                'active_connections': active_result or 0,
                'max_connections': int(max_result) if max_result else 100
            }
        except Exception as e:
            logger.error(f"Failed to get connection stats: {e}")
            return {'active_connections': 0, 'max_connections': 100}
    
    async def _get_performance_stats(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            # Query performance stats
            stats_result = await conn.fetchrow("""
                SELECT 
                    sum(calls) as total_calls,
                    sum(total_time) as total_time,
                    sum(CASE WHEN mean_time > 1000 THEN calls ELSE 0 END) as slow_queries
                FROM pg_stat_statements 
                WHERE calls > 0
            """)
            
            # Calculate QPS (simplified)
            total_calls = stats_result['total_calls'] if stats_result else 0
            queries_per_second = total_calls / 3600 if total_calls else 0  # Rough estimate
            
            # Cache hit ratio
            cache_stats = await conn.fetchrow("""
                SELECT 
                    sum(heap_blks_hit) as heap_hit,
                    sum(heap_blks_read) as heap_read
                FROM pg_statio_user_tables
            """)
            
            if cache_stats and (cache_stats['heap_hit'] + cache_stats['heap_read']) > 0:
                cache_hit_ratio = (
                    cache_stats['heap_hit'] / 
                    (cache_stats['heap_hit'] + cache_stats['heap_read']) * 100
                )
            else:
                cache_hit_ratio = 0.0
            
            return {
                'queries_per_second': queries_per_second,
                'slow_queries': stats_result['slow_queries'] if stats_result else 0,
                'cache_hit_ratio': cache_hit_ratio
            }
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {
                'queries_per_second': 0.0,
                'slow_queries': 0,
                'cache_hit_ratio': 0.0
            }
    
    async def _get_database_stats(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Get database size and structure statistics"""
        try:
            # Database size
            db_size = await conn.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)
            
            # Extract size in MB (simplified parsing)
            db_size_mb = 0.0
            if db_size:
                if 'MB' in db_size:
                    db_size_mb = float(db_size.split()[0])
                elif 'GB' in db_size:
                    db_size_mb = float(db_size.split()[0]) * 1024
                elif 'kB' in db_size:
                    db_size_mb = float(db_size.split()[0]) / 1024
            
            # Table count
            table_count = await conn.fetchval("""
                SELECT count(*) FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            # Index count
            index_count = await conn.fetchval("""
                SELECT count(*) FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            
            return {
                'database_size_mb': db_size_mb,
                'table_count': table_count or 0,
                'index_count': index_count or 0
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                'database_size_mb': 0.0,
                'table_count': 0,
                'index_count': 0
            }
    
    async def _get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory_usage_mb': memory.used / (1024 * 1024),
                'memory_usage_percent': memory.percent,
                'cpu_usage_percent': cpu_percent
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                'memory_usage_mb': 0.0,
                'memory_usage_percent': 0.0,
                'cpu_usage_percent': 0.0
            }
    
    async def _assess_health(self, metrics: DatabaseMetrics) -> List[str]:
        """Assess database health and return issues"""
        issues = []
        
        # Connection pool usage
        if metrics.connection_usage_percent > 80:
            issues.append(f"High connection usage: {metrics.connection_usage_percent:.1f}%")
        
        # Cache hit ratio
        if metrics.cache_hit_ratio < 80:
            issues.append(f"Low cache hit ratio: {metrics.cache_hit_ratio:.1f}%")
        
        # Memory usage
        if metrics.memory_usage_percent > 90:
            issues.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
        
        # CPU usage
        if metrics.cpu_usage_percent > 80:
            issues.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        return issues
    
    async def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slow queries from pg_stat_statements"""
        try:
            async with self.connection_pool.acquire() as conn:
                slow_queries = await conn.fetch("""
                    SELECT 
                        query,
                        calls,
                        total_time,
                        mean_time,
                        stddev_time,
                        rows
                    FROM pg_stat_statements 
                    WHERE calls > 10
                    ORDER BY mean_time DESC 
                    LIMIT $1
                """, limit)
                
                return [dict(row) for row in slow_queries]
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []
    
    async def get_table_sizes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get largest tables by size"""
        try:
            async with self.connection_pool.acquire() as conn:
                table_sizes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
                    LIMIT $1
                """, limit)
                
                return [dict(row) for row in table_sizes]
        except Exception as e:
            logger.error(f"Failed to get table sizes: {e}")
            return []


class RedisMonitor:
    """Redis monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[redis.Redis] = None
        self.metrics_history: List[DatabaseMetrics] = []
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                db=self.config['db'],
                password=self.config.get('password')
            )
            await self.client.ping()
            logger.info("Redis monitor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis monitor: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.aclose()
    
    async def collect_metrics(self) -> DatabaseMetrics:
        """Collect comprehensive Redis metrics"""
        timestamp = datetime.utcnow().isoformat()
        metrics = DatabaseMetrics(timestamp=timestamp, database_type='redis')
        
        try:
            # Get Redis info
            info = await self.client.info()
            
            # Connection metrics
            metrics.active_connections = info.get('connected_clients', 0)
            metrics.max_connections = info.get('maxclients', 10000)
            metrics.connection_usage_percent = (
                metrics.active_connections / metrics.max_connections * 100
                if metrics.max_connections > 0 else 0
            )
            
            # Performance metrics
            metrics.queries_per_second = info.get('instantaneous_ops_per_sec', 0)
            
            # Cache hit ratio
            keyspace_hits = info.get('keyspace_hits', 0)
            keyspace_misses = info.get('keyspace_misses', 0)
            total_requests = keyspace_hits + keyspace_misses
            metrics.cache_hit_ratio = (
                keyspace_hits / total_requests * 100 if total_requests > 0 else 0
            )
            
            # Memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            metrics.memory_usage_mb = used_memory / (1024 * 1024)
            
            if max_memory > 0:
                metrics.memory_usage_percent = used_memory / max_memory * 100
            else:
                # Use system memory as fallback
                system_memory = psutil.virtual_memory()
                metrics.memory_usage_percent = used_memory / system_memory.total * 100
            
            # Database size (number of keys)
            db_info = info.get(f'db{self.config["db"]}', {})
            if isinstance(db_info, dict):
                metrics.table_count = db_info.get('keys', 0)
            
            # CPU usage
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            # Health assessment
            health_issues = await self._assess_health(metrics, info)
            metrics.health_issues = health_issues
            metrics.is_healthy = len(health_issues) == 0
            
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
            metrics.is_healthy = False
            metrics.health_issues = [f"Metrics collection failed: {str(e)}"]
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    async def _assess_health(self, metrics: DatabaseMetrics, info: Dict[str, Any]) -> List[str]:
        """Assess Redis health"""
        issues = []
        
        # Memory usage
        if metrics.memory_usage_percent > 90:
            issues.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
        
        # Connection usage
        if metrics.connection_usage_percent > 80:
            issues.append(f"High connection usage: {metrics.connection_usage_percent:.1f}%")
        
        # Cache hit ratio
        if metrics.cache_hit_ratio < 80:
            issues.append(f"Low cache hit ratio: {metrics.cache_hit_ratio:.1f}%")
        
        # Check for Redis-specific issues
        if info.get('loading', 0) == 1:
            issues.append("Redis is loading data from disk")
        
        if info.get('rdb_last_bgsave_status') == 'err':
            issues.append("Last background save failed")
        
        return issues
    
    async def get_key_statistics(self) -> Dict[str, Any]:
        """Get Redis key statistics"""
        try:
            info = await self.client.info('keyspace')
            
            key_stats = {}
            for db_key, db_info in info.items():
                if db_key.startswith('db'):
                    db_num = db_key[2:]  # Remove 'db' prefix
                    if isinstance(db_info, dict):
                        key_stats[db_num] = db_info
            
            return key_stats
        except Exception as e:
            logger.error(f"Failed to get key statistics: {e}")
            return {}
    
    async def get_memory_usage_by_type(self) -> Dict[str, Any]:
        """Get memory usage breakdown by data type"""
        try:
            memory_info = await self.client.info('memory')
            
            memory_breakdown = {
                'used_memory': memory_info.get('used_memory', 0),
                'used_memory_rss': memory_info.get('used_memory_rss', 0),
                'used_memory_peak': memory_info.get('used_memory_peak', 0),
                'used_memory_overhead': memory_info.get('used_memory_overhead', 0),
                'used_memory_dataset': memory_info.get('used_memory_dataset', 0)
            }
            
            return memory_breakdown
        except Exception as e:
            logger.error(f"Failed to get memory usage breakdown: {e}")
            return {}


class DatabaseMonitoringSystem:
    """Main monitoring system that coordinates PostgreSQL and Redis monitoring"""
    
    def __init__(self, pg_config: Dict[str, Any], redis_config: Dict[str, Any]):
        self.pg_monitor = PostgreSQLMonitor(pg_config)
        self.redis_monitor = RedisMonitor(redis_config)
        self.alert_rules = self._load_default_alert_rules()
        self.monitoring_active = False
    
    async def initialize(self):
        """Initialize both monitors"""
        await self.pg_monitor.initialize()
        await self.redis_monitor.initialize()
        logger.info("Database monitoring system initialized")
    
    async def close(self):
        """Close all monitors"""
        await self.pg_monitor.close()
        await self.redis_monitor.close()
        self.monitoring_active = False
    
    async def collect_all_metrics(self) -> Dict[str, DatabaseMetrics]:
        """Collect metrics from all databases"""
        metrics = {}
        
        try:
            # Collect PostgreSQL metrics
            pg_metrics = await self.pg_monitor.collect_metrics()
            metrics['postgresql'] = pg_metrics
            
            # Collect Redis metrics
            redis_metrics = await self.redis_monitor.collect_metrics()
            metrics['redis'] = redis_metrics
            
            # Check alerts
            await self._check_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    async def start_continuous_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring loop"""
        self.monitoring_active = True
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        
        while self.monitoring_active:
            try:
                metrics = await self.collect_all_metrics()
                
                # Log summary
                self._log_metrics_summary(metrics)
                
                # Save metrics to file
                await self._save_metrics(metrics)
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        logger.info("Stopping continuous monitoring")
    
    def _load_default_alert_rules(self) -> List[AlertRule]:
        """Load default alert rules"""
        return [
            AlertRule(
                metric_name='connection_usage_percent',
                threshold=80.0,
                comparison='gt',
                severity='warning',
                description='High connection pool usage'
            ),
            AlertRule(
                metric_name='memory_usage_percent',
                threshold=90.0,
                comparison='gt',
                severity='critical',
                description='High memory usage'
            ),
            AlertRule(
                metric_name='cpu_usage_percent',
                threshold=80.0,
                comparison='gt',
                severity='warning',
                description='High CPU usage'
            ),
            AlertRule(
                metric_name='cache_hit_ratio',
                threshold=80.0,
                comparison='lt',
                severity='warning',
                description='Low cache hit ratio'
            )
        ]
    
    async def _check_alerts(self, metrics: Dict[str, DatabaseMetrics]):
        """Check alert rules against current metrics"""
        for db_type, db_metrics in metrics.items():
            for alert_rule in self.alert_rules:
                if not alert_rule.enabled:
                    continue
                
                metric_value = getattr(db_metrics, alert_rule.metric_name, None)
                if metric_value is None:
                    continue
                
                # Check if alert condition is met
                condition_met = False
                if alert_rule.comparison == 'gt':
                    condition_met = metric_value > alert_rule.threshold
                elif alert_rule.comparison == 'lt':
                    condition_met = metric_value < alert_rule.threshold
                elif alert_rule.comparison == 'eq':
                    condition_met = metric_value == alert_rule.threshold
                
                if condition_met:
                    await self._trigger_alert(db_type, alert_rule, metric_value)
    
    async def _trigger_alert(self, db_type: str, alert_rule: AlertRule, metric_value: float):
        """Trigger an alert"""
        alert_message = (
            f"ALERT [{alert_rule.severity.upper()}] {db_type}: "
            f"{alert_rule.description} - "
            f"{alert_rule.metric_name} = {metric_value:.2f} "
            f"(threshold: {alert_rule.threshold})"
        )
        
        if alert_rule.severity == 'critical':
            logger.critical(alert_message)
        elif alert_rule.severity == 'warning':
            logger.warning(alert_message)
        else:
            logger.info(alert_message)
        
        # Here you could add additional alert mechanisms:
        # - Send email
        # - Send to Slack
        # - Write to alert file
        # - Send to monitoring system
    
    def _log_metrics_summary(self, metrics: Dict[str, DatabaseMetrics]):
        """Log a summary of current metrics"""
        summary_parts = []
        
        for db_type, db_metrics in metrics.items():
            health_status = "HEALTHY" if db_metrics.is_healthy else "UNHEALTHY"
            summary_parts.append(
                f"{db_type.upper()}: {health_status} | "
                f"Conn: {db_metrics.connection_usage_percent:.1f}% | "
                f"Mem: {db_metrics.memory_usage_percent:.1f}% | "
                f"CPU: {db_metrics.cpu_usage_percent:.1f}% | "
                f"Cache: {db_metrics.cache_hit_ratio:.1f}%"
            )
        
        logger.info("METRICS: " + " || ".join(summary_parts))
    
    async def _save_metrics(self, metrics: Dict[str, DatabaseMetrics]):
        """Save metrics to file for historical analysis"""
        try:
            metrics_file = Path("/tmp/db_metrics.jsonl")
            
            # Convert metrics to JSON
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': {
                    db_type: asdict(db_metrics) 
                    for db_type, db_metrics in metrics.items()
                }
            }
            
            # Append to JSONL file
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics_data, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        metrics = await self.collect_all_metrics()
        
        # Get additional details
        pg_slow_queries = await self.pg_monitor.get_slow_queries()
        pg_table_sizes = await self.pg_monitor.get_table_sizes()
        redis_key_stats = await self.redis_monitor.get_key_statistics()
        redis_memory = await self.redis_monitor.get_memory_usage_by_type()
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'overall_health': all(m.is_healthy for m in metrics.values()),
            'databases': {},
            'recommendations': []
        }
        
        # Add database-specific information
        for db_type, db_metrics in metrics.items():
            report['databases'][db_type] = {
                'metrics': asdict(db_metrics),
                'health_status': 'healthy' if db_metrics.is_healthy else 'unhealthy',
                'issues': db_metrics.health_issues
            }
        
        # Add PostgreSQL specifics
        if 'postgresql' in report['databases']:
            report['databases']['postgresql'].update({
                'slow_queries': pg_slow_queries,
                'largest_tables': pg_table_sizes
            })
        
        # Add Redis specifics
        if 'redis' in report['databases']:
            report['databases']['redis'].update({
                'key_statistics': redis_key_stats,
                'memory_breakdown': redis_memory
            })
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(metrics)
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, DatabaseMetrics]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for db_type, db_metrics in metrics.items():
            if db_type == 'postgresql':
                if db_metrics.cache_hit_ratio < 90:
                    recommendations.append(
                        "Consider increasing PostgreSQL shared_buffers for better cache performance"
                    )
                if db_metrics.connection_usage_percent > 70:
                    recommendations.append(
                        "Consider implementing connection pooling or increasing max_connections"
                    )
            
            elif db_type == 'redis':
                if db_metrics.memory_usage_percent > 80:
                    recommendations.append(
                        "Consider implementing Redis memory optimization strategies"
                    )
                if db_metrics.cache_hit_ratio < 95:
                    recommendations.append(
                        "Review Redis cache TTL settings and key distribution"
                    )
        
        return recommendations


async def main():
    """Main monitoring function"""
    # Load configuration
    from dotenv import load_dotenv
    
    env_file = Path(__file__).parent.parent / '.env.production'
    if env_file.exists():
        load_dotenv(env_file)
    
    pg_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'rental_ml'),
        'user': os.getenv('DB_USERNAME', 'rental_ml_user'),
        'password': os.getenv('DB_PASSWORD', 'password123'),
    }
    
    redis_config = {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', '6379')),
        'db': int(os.getenv('REDIS_DB', '0')),
        'password': os.getenv('REDIS_PASSWORD'),
    }
    
    # Initialize monitoring system
    monitor = DatabaseMonitoringSystem(pg_config, redis_config)
    
    try:
        await monitor.initialize()
        
        # Generate and print health report
        report = await monitor.generate_health_report()
        
        print("\n" + "="*80)
        print("DATABASE HEALTH REPORT")
        print("="*80)
        
        print(f"\nGenerated: {report['generated_at']}")
        print(f"Overall Health: {'HEALTHY' if report['overall_health'] else 'UNHEALTHY'}")
        
        for db_type, db_info in report['databases'].items():
            print(f"\n{db_type.upper()} Database:")
            print(f"  Status: {db_info['health_status'].upper()}")
            
            metrics = db_info['metrics']
            print(f"  Connections: {metrics['active_connections']}/{metrics['max_connections']} ({metrics['connection_usage_percent']:.1f}%)")
            print(f"  Memory: {metrics['memory_usage_mb']:.1f} MB ({metrics['memory_usage_percent']:.1f}%)")
            print(f"  CPU: {metrics['cpu_usage_percent']:.1f}%")
            print(f"  Cache Hit Ratio: {metrics['cache_hit_ratio']:.1f}%")
            
            if db_info['issues']:
                print(f"  Issues:")
                for issue in db_info['issues']:
                    print(f"    - {issue}")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)
    finally:
        await monitor.close()


if __name__ == "__main__":
    # Install required package if not available
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        os.system("pip install psutil")
        import psutil
    
    asyncio.run(main())