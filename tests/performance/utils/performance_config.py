"""
Performance testing configuration and constants.

Centralized configuration for all performance tests with environment-specific
settings and realistic load patterns.
"""

import os
import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from contextlib import contextmanager
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceTestEnvironment(Enum):
    """Performance test environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    LOAD_TEST = "load_test"


class LoadPattern(Enum):
    """Load testing patterns."""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    STRESS = "stress"
    ENDURANCE = "endurance"


@dataclass
class DatabaseConfig:
    """Database performance testing configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "rental_ml_test"
    username: str = "postgres"
    password: str = "test_password"
    pool_size: int = 20
    max_overflow: int = 40
    connection_timeout: int = 30
    query_timeout: int = 60


@dataclass
class RedisConfig:
    """Redis performance testing configuration."""
    host: str = "localhost"
    port: int = 6379
    password: str = "test_password"
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 30
    socket_connect_timeout: int = 30


@dataclass
class APIConfig:
    """API performance testing configuration."""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    rate_limit_per_second: int = 100
    concurrent_users: int = 50


@dataclass
class MLConfig:
    """ML performance testing configuration."""
    model_path: str = "/tmp/test_models"
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    embedding_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    max_inference_time_ms: int = 100
    max_training_time_minutes: int = 30
    memory_limit_mb: int = 2048


@dataclass
class LoadTestConfig:
    """Load testing configuration."""
    duration_seconds: int = 300  # 5 minutes default
    warm_up_seconds: int = 30
    cool_down_seconds: int = 30
    concurrent_users: int = 50
    requests_per_user: int = 100
    ramp_up_time_seconds: int = 60
    think_time_seconds: float = 1.0
    pattern: LoadPattern = LoadPattern.CONSTANT


@dataclass
class StressTestConfig:
    """Stress testing configuration."""
    max_concurrent_users: int = 500
    step_size: int = 50
    step_duration_seconds: int = 60
    failure_threshold_percent: float = 5.0
    response_time_threshold_ms: int = 5000


@dataclass
class EnduranceTestConfig:
    """Endurance testing configuration."""
    duration_hours: int = 4
    concurrent_users: int = 100
    memory_leak_threshold_mb: int = 100
    cpu_threshold_percent: float = 80.0
    error_rate_threshold_percent: float = 1.0


@dataclass
class BenchmarkConfig:
    """Benchmarking configuration."""
    baseline_commit: Optional[str] = None
    performance_regression_threshold: float = 0.1  # 10% regression
    accuracy_regression_threshold: float = 0.05   # 5% accuracy regression
    iterations: int = 5
    confidence_level: float = 0.95


@dataclass 
class PerformanceConfig:
    """Main performance testing configuration."""
    environment: PerformanceTestEnvironment = PerformanceTestEnvironment.DEVELOPMENT
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    load_test: LoadTestConfig = field(default_factory=LoadTestConfig)
    stress_test: StressTestConfig = field(default_factory=StressTestConfig)
    endurance_test: EnduranceTestConfig = field(default_factory=EnduranceTestConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    
    # Performance thresholds
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'api_response_time_ms': {
            'p50': 100,
            'p95': 500,
            'p99': 1000
        },
        'ml_inference_time_ms': {
            'single': 50,
            'batch_32': 200,
            'batch_64': 400
        },
        'database_query_time_ms': {
            'simple_select': 10,
            'complex_join': 100,
            'aggregation': 500
        },
        'throughput_rps': {
            'min_search': 100,
            'min_recommendations': 50,
            'min_user_operations': 200
        },
        'resource_usage': {
            'cpu_percent': 70.0,
            'memory_mb': 1024,
            'disk_io_mbps': 100
        }
    })
    
    # Test data configuration
    test_data: Dict[str, Any] = field(default_factory=lambda: {
        'users': {
            'small': 100,
            'medium': 1000,
            'large': 10000
        },
        'properties': {
            'small': 500,
            'medium': 5000,
            'large': 50000
        },
        'interactions': {
            'density': 0.05,  # 5% user-property interaction rate
            'min_per_user': 5,
            'max_per_user': 100
        }
    })

    @classmethod
    def from_environment(cls, env: str = None) -> 'PerformanceConfig':
        """Create configuration based on environment."""
        env = env or os.getenv('PERFORMANCE_TEST_ENV', 'development')
        environment = PerformanceTestEnvironment(env)
        
        config = cls(environment=environment)
        
        if environment == PerformanceTestEnvironment.DEVELOPMENT:
            config._configure_development()
        elif environment == PerformanceTestEnvironment.STAGING:
            config._configure_staging()
        elif environment == PerformanceTestEnvironment.PRODUCTION:
            config._configure_production()
        elif environment == PerformanceTestEnvironment.LOAD_TEST:
            config._configure_load_test()
        
        return config
    
    def _configure_development(self):
        """Configure for development environment."""
        self.load_test.duration_seconds = 60
        self.load_test.concurrent_users = 10
        self.stress_test.max_concurrent_users = 50
        self.endurance_test.duration_hours = 1
        
        # Lower thresholds for development
        self.thresholds['api_response_time_ms'] = {
            'p50': 200,
            'p95': 1000,
            'p99': 2000
        }
    
    def _configure_staging(self):
        """Configure for staging environment."""
        self.load_test.duration_seconds = 180
        self.load_test.concurrent_users = 25
        self.stress_test.max_concurrent_users = 200
        self.endurance_test.duration_hours = 2
    
    def _configure_production(self):
        """Configure for production environment."""
        self.load_test.duration_seconds = 600
        self.load_test.concurrent_users = 100
        self.stress_test.max_concurrent_users = 1000
        self.endurance_test.duration_hours = 8
        
        # Stricter thresholds for production
        self.thresholds['api_response_time_ms'] = {
            'p50': 50,
            'p95': 200,
            'p99': 500
        }
    
    def _configure_load_test(self):
        """Configure for dedicated load testing."""
        self.load_test.duration_seconds = 1800  # 30 minutes
        self.load_test.concurrent_users = 200
        self.stress_test.max_concurrent_users = 2000
        self.endurance_test.duration_hours = 12


# Predefined test scenarios
PERFORMANCE_SCENARIOS = {
    'api_baseline': {
        'name': 'API Baseline Performance',
        'endpoints': ['/health', '/api/v1/search', '/api/v1/recommendations'],
        'load_pattern': LoadPattern.CONSTANT,
        'duration_seconds': 300,
        'concurrent_users': 50
    },
    'ml_inference_baseline': {
        'name': 'ML Inference Baseline',
        'models': ['content_recommender', 'collaborative_filter', 'hybrid'],
        'batch_sizes': [1, 8, 16, 32],
        'iterations': 100
    },
    'database_baseline': {
        'name': 'Database Baseline Performance',
        'operations': ['read_heavy', 'write_heavy', 'mixed'],
        'concurrent_connections': [10, 50, 100],
        'duration_seconds': 300
    },
    'search_load_test': {
        'name': 'Search Load Test',
        'endpoint': '/api/v1/search',
        'load_pattern': LoadPattern.RAMP_UP,
        'max_users': 200,
        'duration_seconds': 600
    },
    'recommendation_load_test': {
        'name': 'Recommendation Load Test',
        'endpoint': '/api/v1/recommendations',
        'load_pattern': LoadPattern.CONSTANT,
        'concurrent_users': 100,
        'duration_seconds': 900
    },
    'stress_test_api': {
        'name': 'API Stress Test',
        'load_pattern': LoadPattern.STRESS,
        'max_users': 500,
        'step_size': 50,
        'step_duration': 60
    },
    'endurance_test_full_system': {
        'name': 'Full System Endurance Test',
        'duration_hours': 4,
        'concurrent_users': 100,
        'all_endpoints': True
    }
}


# Real-world traffic patterns
TRAFFIC_PATTERNS = {
    'business_hours': {
        'description': 'Typical business hours traffic pattern',
        'pattern': [
            (0, 0.1),   # 12 AM - 10% traffic
            (6, 0.2),   # 6 AM - 20% traffic  
            (9, 0.8),   # 9 AM - 80% traffic
            (12, 1.0),  # 12 PM - 100% traffic (peak)
            (14, 0.9),  # 2 PM - 90% traffic
            (17, 0.7),  # 5 PM - 70% traffic
            (20, 0.4),  # 8 PM - 40% traffic
            (23, 0.2),  # 11 PM - 20% traffic
        ]
    },
    'weekend_pattern': {
        'description': 'Weekend traffic pattern',
        'pattern': [
            (0, 0.05),  # 12 AM - 5% traffic
            (8, 0.1),   # 8 AM - 10% traffic
            (10, 0.6),  # 10 AM - 60% traffic
            (14, 0.8),  # 2 PM - 80% traffic (weekend peak)
            (18, 0.7),  # 6 PM - 70% traffic
            (22, 0.3),  # 10 PM - 30% traffic
        ]
    },
    'flash_sale': {
        'description': 'Flash sale spike pattern',
        'pattern': [
            (0, 0.2),   # Normal traffic
            (1, 5.0),   # 5x spike
            (2, 8.0),   # 8x spike
            (3, 3.0),   # 3x spike
            (4, 1.0),   # Back to normal
        ]
    }
}


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""
    
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    
    # Resource usage
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Throughput metrics
    operations_count: Optional[int] = None
    throughput_ops_per_sec: Optional[float] = None
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (1.0 for single operation)."""
        return 1.0 if self.success else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'success': self.success,
            'error_message': self.error_message,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'operations_count': self.operations_count,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'context': self.context
        }


class PerformanceTimer:
    """High-precision timer for performance measurements."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.process = psutil.Process()
        self.start_memory: Optional[float] = None
        self.start_cpu: Optional[float] = None
    
    def start(self) -> 'PerformanceTimer':
        """Start the timer and resource monitoring."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return duration."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time
    
    def get_metrics(self, success: bool = True, error_message: Optional[str] = None, 
                   operations_count: Optional[int] = None) -> PerformanceMetrics:
        """Get complete performance metrics."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer not properly started/stopped")
        
        duration = self.end_time - self.start_time
        
        # Calculate resource usage
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = current_memory - (self.start_memory or 0)
        
        current_cpu = self.process.cpu_percent()
        
        # Calculate throughput if operations count provided
        throughput = None
        if operations_count is not None and duration > 0:
            throughput = operations_count / duration
        
        return PerformanceMetrics(
            operation_name=self.operation_name,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=duration,
            success=success,
            error_message=error_message,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=current_cpu,
            operations_count=operations_count,
            throughput_ops_per_sec=throughput
        )
    
    def __enter__(self) -> 'PerformanceTimer':
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PerformanceMonitor:
    """System-wide performance monitoring utility."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitor_interval = 1.0  # seconds
        self._monitor_task: Optional[asyncio.Task] = None
    
    def add_metric(self, metric: PerformanceMetrics):
        """Add a performance metric to the collection."""
        self.metrics.append(metric)
        logger.debug(f"Added performance metric: {metric.operation_name} - {metric.duration:.3f}s")
    
    async def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring."""
        self.monitor_interval = interval
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_system())
        logger.info("Started performance monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped performance monitoring")
    
    async def _monitor_system(self):
        """Continuously monitor system resources."""
        process = psutil.Process()
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                metric = PerformanceMetrics(
                    operation_name="system_monitoring",
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0.0,
                    success=True,
                    memory_usage_mb=memory_info.rss / 1024 / 1024,
                    cpu_usage_percent=cpu_percent,
                    context={
                        'virtual_memory': memory_info.vms / 1024 / 1024,
                        'available_memory': psutil.virtual_memory().available / 1024 / 1024,
                        'system_cpu_percent': psutil.cpu_percent()
                    }
                )
                
                self.add_metric(metric)
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    def get_summary_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for collected metrics."""
        filtered_metrics = self.metrics
        if operation_name:
            filtered_metrics = [m for m in self.metrics if m.operation_name == operation_name]
        
        if not filtered_metrics:
            return {}
        
        durations = [m.duration for m in filtered_metrics if m.duration > 0]
        success_count = sum(1 for m in filtered_metrics if m.success)
        
        stats = {
            'total_operations': len(filtered_metrics),
            'successful_operations': success_count,
            'success_rate': success_count / len(filtered_metrics) if filtered_metrics else 0,
            'error_count': len(filtered_metrics) - success_count
        }
        
        if durations:
            stats.update({
                'avg_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'p50_duration': np.percentile(durations, 50),
                'p90_duration': np.percentile(durations, 90),
                'p95_duration': np.percentile(durations, 95),
                'p99_duration': np.percentile(durations, 99)
            })
        
        # Memory statistics
        memory_values = [m.memory_usage_mb for m in filtered_metrics if m.memory_usage_mb is not None]
        if memory_values:
            stats.update({
                'avg_memory_mb': np.mean(memory_values),
                'max_memory_mb': np.max(memory_values),
                'min_memory_mb': np.min(memory_values)
            })
        
        # Throughput statistics
        throughput_values = [m.throughput_ops_per_sec for m in filtered_metrics if m.throughput_ops_per_sec is not None]
        if throughput_values:
            stats.update({
                'avg_throughput': np.mean(throughput_values),
                'max_throughput': np.max(throughput_values),
                'min_throughput': np.min(throughput_values)
            })
        
        return stats
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file."""
        import json
        
        metrics_data = {
            'summary': self.get_summary_stats(),
            'metrics': [m.to_dict() for m in self.metrics],
            'export_time': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Exported {len(self.metrics)} metrics to {filename}")
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        logger.info("Cleared all performance metrics")


def assert_performance_threshold(duration: float, threshold: float, operation: str):
    """Assert that operation duration is within acceptable threshold."""
    if duration > threshold:
        raise AssertionError(
            f"Performance threshold exceeded for {operation}: "
            f"{duration:.3f}s > {threshold:.3f}s (threshold)"
        )
    logger.info(f"Performance OK for {operation}: {duration:.3f}s <= {threshold:.3f}s")


def assert_throughput_threshold(actual_throughput: float, expected_throughput: float, operation: str):
    """Assert that throughput meets minimum requirements."""
    if actual_throughput < expected_throughput:
        raise AssertionError(
            f"Throughput below threshold for {operation}: "
            f"{actual_throughput:.1f} ops/s < {expected_throughput:.1f} ops/s (threshold)"
        )
    logger.info(f"Throughput OK for {operation}: {actual_throughput:.1f} ops/s >= {expected_throughput:.1f} ops/s")


def assert_memory_threshold(memory_usage: float, threshold: float, operation: str):
    """Assert that memory usage is within acceptable limits."""
    if memory_usage > threshold:
        raise AssertionError(
            f"Memory usage exceeded for {operation}: "
            f"{memory_usage:.1f}MB > {threshold:.1f}MB (threshold)"
        )
    logger.info(f"Memory usage OK for {operation}: {memory_usage:.1f}MB <= {threshold:.1f}MB")


@contextmanager
def measure_performance(operation_name: str, 
                      success_callback: Optional[Callable[[PerformanceMetrics], None]] = None):
    """Context manager for measuring operation performance."""
    timer = PerformanceTimer(operation_name)
    timer.start()
    
    success = True
    error_message = None
    
    try:
        yield timer
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        timer.stop()
        metrics = timer.get_metrics(success=success, error_message=error_message)
        
        if success_callback:
            success_callback(metrics)
        
        # Log performance info
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Performance [{status}] {operation_name}: {metrics.duration:.3f}s")
        if metrics.memory_usage_mb:
            logger.info(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
        if metrics.throughput_ops_per_sec:
            logger.info(f"  Throughput: {metrics.throughput_ops_per_sec:.1f} ops/s")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()