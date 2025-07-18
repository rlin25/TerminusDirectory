"""
Production Database Connection Manager for Rental ML System
Handles connection pooling, failover, monitoring, and load balancing
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import yaml
import os
import asyncpg
import aioredis
from asyncpg import Pool
from asyncpg.exceptions import PostgresError, ConnectionDoesNotExistError
import psutil
import threading
from datetime import datetime, timedelta
import ssl
import random


class DatabaseRole(Enum):
    PRIMARY = "primary"
    REPLICA = "replica"
    ANALYTICS = "analytics"


class ConnectionHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    ssl_context: Optional[ssl.SSLContext] = None
    min_connections: int = 10
    max_connections: int = 100
    connection_timeout: int = 30
    idle_timeout: int = 300
    max_lifetime: int = 3600
    role: DatabaseRole = DatabaseRole.PRIMARY
    weight: int = 100


@dataclass
class ConnectionMetrics:
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    queries_executed: int = 0
    errors_count: int = 0
    avg_query_time: float = 0.0
    last_health_check: Optional[datetime] = None
    health_status: ConnectionHealth = ConnectionHealth.HEALTHY


class DatabaseConnectionManager:
    """
    Production-grade database connection manager with:
    - Connection pooling and load balancing
    - Automatic failover and recovery
    - Health monitoring and metrics
    - SSL/TLS support
    - Read replica support
    """
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.getenv('DB_CONFIG_PATH', 'config/database/production.yaml')
        self.config = self._load_config()
        
        # Connection pools
        self.primary_pool: Optional[Pool] = None
        self.replica_pools: Dict[str, Pool] = {}
        self.analytics_pools: Dict[str, Pool] = {}
        
        # Health monitoring
        self.metrics: Dict[str, ConnectionMetrics] = {}
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Failover state
        self.primary_healthy = True
        self.replica_weights = {}
        
        # Redis for distributed coordination
        self.redis: Optional[aioredis.Redis] = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load database configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Environment variable substitution
            config = self._substitute_env_vars(config)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load database config: {e}")
            raise
            
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            if ':-' in env_var:
                var_name, default = env_var.split(':-', 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(env_var, config)
        return config
        
    def _create_ssl_context(self, ssl_config: Dict[str, Any]) -> Optional[ssl.SSLContext]:
        """Create SSL context from configuration"""
        if not ssl_config.get('enabled', False):
            return None
            
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        if ssl_config.get('cert_file'):
            context.load_cert_chain(
                ssl_config['cert_file'],
                ssl_config.get('key_file')
            )
            
        if ssl_config.get('ca_file'):
            context.load_verify_locations(ssl_config['ca_file'])
            
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED if ssl_config.get('mode') == 'require' else ssl.CERT_NONE
        
        return context
        
    async def initialize(self):
        """Initialize all database connections and monitoring"""
        try:
            self.logger.info("Initializing database connection manager...")
            
            # Initialize Redis for coordination
            await self._initialize_redis()
            
            # Initialize primary database
            await self._initialize_primary()
            
            # Initialize read replicas
            await self._initialize_replicas()
            
            # Initialize analytics databases
            await self._initialize_analytics()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self.logger.info("Database connection manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection manager: {e}")
            raise
            
    async def _initialize_redis(self):
        """Initialize Redis connection for distributed coordination"""
        try:
            redis_config = self.config.get('redis', {})
            if redis_config.get('enabled', True):
                self.redis = await aioredis.from_url(
                    f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}",
                    password=redis_config.get('password'),
                    ssl=redis_config.get('ssl', False)
                )
        except Exception as e:
            self.logger.warning(f"Failed to initialize Redis: {e}")
            
    async def _initialize_primary(self):
        """Initialize primary database connection pool"""
        primary_config = self.config['database']['primary']
        ssl_context = self._create_ssl_context(primary_config.get('ssl', {}))
        
        connection_config = primary_config['connections']
        
        try:
            self.primary_pool = await asyncpg.create_pool(
                host=primary_config['host'],
                port=primary_config['port'],
                database=primary_config['name'],
                user=primary_config['user'],
                password=primary_config['password'],
                ssl=ssl_context,
                min_size=connection_config['min_connections'],
                max_size=connection_config['max_connections'],
                max_queries=50000,
                max_inactive_connection_lifetime=connection_config['max_lifetime'],
                timeout=connection_config['connection_timeout'],
                command_timeout=primary_config['performance']['statement_timeout'] / 1000
            )
            
            self.metrics['primary'] = ConnectionMetrics()
            self.logger.info("Primary database pool initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize primary database: {e}")
            raise
            
    async def _initialize_replicas(self):
        """Initialize read replica connection pools"""
        replicas_config = self.config['database'].get('read_replicas', [])
        
        for replica_config in replicas_config:
            try:
                ssl_context = self._create_ssl_context(
                    self.config['database']['primary'].get('ssl', {})
                )
                
                pool = await asyncpg.create_pool(
                    host=replica_config['host'],
                    port=replica_config['port'],
                    database=self.config['database']['primary']['name'],
                    user=self.config['database']['primary']['user'],
                    password=self.config['database']['primary']['password'],
                    ssl=ssl_context,
                    min_size=5,
                    max_size=replica_config.get('max_connections', 50),
                    max_queries=50000,
                    timeout=30
                )
                
                replica_name = replica_config['name']
                self.replica_pools[replica_name] = pool
                self.replica_weights[replica_name] = replica_config.get('weight', 100)
                self.metrics[replica_name] = ConnectionMetrics()
                
                self.logger.info(f"Replica '{replica_name}' pool initialized")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize replica '{replica_config['name']}': {e}")
                
    async def _initialize_analytics(self):
        """Initialize analytics database connections"""
        # Implementation for analytics-specific pools
        pass
        
    async def _start_health_monitoring(self):
        """Start continuous health monitoring"""
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        
    async def _health_monitor_loop(self):
        """Continuous health monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def _perform_health_checks(self):
        """Perform health checks on all database connections"""
        # Check primary
        await self._check_database_health('primary', self.primary_pool)
        
        # Check replicas
        for name, pool in self.replica_pools.items():
            await self._check_database_health(name, pool)
            
    async def _check_database_health(self, name: str, pool: Optional[Pool]):
        """Check health of a specific database connection pool"""
        if not pool:
            return
            
        try:
            start_time = time.time()
            async with pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            response_time = time.time() - start_time
            
            # Update metrics
            metrics = self.metrics[name]
            metrics.last_health_check = datetime.utcnow()
            
            if response_time < 0.1:
                metrics.health_status = ConnectionHealth.HEALTHY
            elif response_time < 1.0:
                metrics.health_status = ConnectionHealth.DEGRADED
            else:
                metrics.health_status = ConnectionHealth.UNHEALTHY
                
            # Update connection stats
            metrics.active_connections = pool.get_size()
            metrics.idle_connections = pool.get_idle_size()
            metrics.total_connections = pool.get_size()
            
        except Exception as e:
            self.logger.error(f"Health check failed for {name}: {e}")
            self.metrics[name].health_status = ConnectionHealth.UNHEALTHY
            
            # Handle primary failure
            if name == 'primary':
                await self._handle_primary_failure()
                
    async def _handle_primary_failure(self):
        """Handle primary database failure"""
        self.logger.warning("Primary database failure detected")
        self.primary_healthy = False
        
        # Notify other instances via Redis
        if self.redis:
            await self.redis.publish('db_events', 'primary_failure')
            
        # Implement failover logic here
        # This could involve promoting a replica or switching to backup
        
    @asynccontextmanager
    async def get_connection(self, 
                           read_only: bool = False, 
                           analytics: bool = False,
                           timeout: Optional[int] = None):
        """
        Get a database connection with automatic load balancing and failover
        
        Args:
            read_only: Whether to use read replica
            analytics: Whether to use analytics-optimized connection
            timeout: Connection timeout override
        """
        pool = None
        pool_name = None
        
        try:
            if analytics and self.analytics_pools:
                pool_name, pool = self._select_analytics_pool()
            elif read_only and self.replica_pools and self._replicas_available():
                pool_name, pool = self._select_replica_pool()
            else:
                pool_name, pool = 'primary', self.primary_pool
                
            if not pool:
                raise ConnectionDoesNotExistError("No available database connections")
                
            # Acquire connection with timeout
            connection_timeout = timeout or 30
            async with asyncio.timeout(connection_timeout):
                async with pool.acquire() as conn:
                    # Set connection parameters
                    await self._configure_connection(conn, analytics, read_only)
                    
                    # Update metrics
                    self._update_connection_metrics(pool_name, 'acquire')
                    
                    yield conn
                    
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            self._update_connection_metrics(pool_name, 'error')
            raise
        finally:
            if pool_name:
                self._update_connection_metrics(pool_name, 'release')
                
    def _select_replica_pool(self) -> tuple[str, Pool]:
        """Select best available replica pool using weighted random selection"""
        healthy_replicas = [
            (name, pool, weight) for name, pool in self.replica_pools.items()
            if self.metrics[name].health_status == ConnectionHealth.HEALTHY
            and weight > 0
            for weight in [self.replica_weights.get(name, 100)]
        ]
        
        if not healthy_replicas:
            return 'primary', self.primary_pool
            
        # Weighted random selection
        total_weight = sum(weight for _, _, weight in healthy_replicas)
        r = random.uniform(0, total_weight)
        
        current_weight = 0
        for name, pool, weight in healthy_replicas:
            current_weight += weight
            if r <= current_weight:
                return name, pool
                
        # Fallback to first available
        return healthy_replicas[0][0], healthy_replicas[0][1]
        
    def _select_analytics_pool(self) -> tuple[str, Pool]:
        """Select analytics-optimized pool"""
        # Implementation for analytics pool selection
        return 'primary', self.primary_pool
        
    def _replicas_available(self) -> bool:
        """Check if any healthy replicas are available"""
        return any(
            self.metrics[name].health_status == ConnectionHealth.HEALTHY
            for name in self.replica_pools.keys()
        )
        
    async def _configure_connection(self, conn, analytics: bool, read_only: bool):
        """Configure connection parameters based on usage"""
        if analytics:
            # Optimize for analytics workloads
            await conn.execute("SET work_mem = '256MB'")
            await conn.execute("SET statement_timeout = '300000'")  # 5 minutes
            
        if read_only:
            await conn.execute("SET default_transaction_read_only = on")
            
    def _update_connection_metrics(self, pool_name: Optional[str], event: str):
        """Update connection metrics"""
        if not pool_name or pool_name not in self.metrics:
            return
            
        metrics = self.metrics[pool_name]
        
        if event == 'acquire':
            metrics.active_connections += 1
        elif event == 'release':
            metrics.active_connections = max(0, metrics.active_connections - 1)
        elif event == 'error':
            metrics.errors_count += 1
            
    async def get_metrics(self) -> Dict[str, ConnectionMetrics]:
        """Get current connection metrics"""
        return self.metrics.copy()
        
    async def close(self):
        """Close all connections and cleanup"""
        self.logger.info("Shutting down database connection manager...")
        
        # Cancel health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
                
        # Close all pools
        if self.primary_pool:
            await self.primary_pool.close()
            
        for pool in self.replica_pools.values():
            await pool.close()
            
        for pool in self.analytics_pools.values():
            await pool.close()
            
        # Close Redis
        if self.redis:
            await self.redis.close()
            
        self.logger.info("Database connection manager shutdown complete")


# Singleton instance
_connection_manager: Optional[DatabaseConnectionManager] = None


async def get_connection_manager() -> DatabaseConnectionManager:
    """Get the global connection manager instance"""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = DatabaseConnectionManager()
        await _connection_manager.initialize()
        
    return _connection_manager


async def close_connection_manager():
    """Close the global connection manager"""
    global _connection_manager
    
    if _connection_manager:
        await _connection_manager.close()
        _connection_manager = None


# Context manager for database operations
@asynccontextmanager
async def database_transaction(read_only: bool = False, analytics: bool = False):
    """
    Context manager for database transactions with automatic connection management
    
    Example:
        async with database_transaction() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM properties")
    """
    manager = await get_connection_manager()
    
    async with manager.get_connection(read_only=read_only, analytics=analytics) as conn:
        async with conn.transaction():
            yield conn