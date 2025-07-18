"""
Automated Database Maintenance Scheduler
Intelligent scheduling and execution of database maintenance tasks
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import croniter
import asyncpg


class MaintenanceType(Enum):
    VACUUM = "vacuum"
    ANALYZE = "analyze"
    REINDEX = "reindex"
    PARTITION_MAINTENANCE = "partition_maintenance"
    STATISTICS_UPDATE = "statistics_update"
    INDEX_OPTIMIZATION = "index_optimization"
    QUERY_PLAN_CACHE_CLEAR = "query_plan_cache_clear"
    CONNECTION_POOL_REFRESH = "connection_pool_refresh"


class MaintenancePriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MaintenanceStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MaintenanceTask:
    id: str
    name: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    schedule_cron: str  # Cron expression
    target_tables: List[str]
    parameters: Dict[str, Any]
    max_duration_minutes: int
    skip_if_high_load: bool = True
    require_low_activity: bool = True
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0
    
    
@dataclass
class MaintenanceExecution:
    task_id: str
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: MaintenanceStatus = MaintenanceStatus.PENDING
    duration_seconds: Optional[float] = None
    affected_tables: List[str] = None
    rows_processed: int = 0
    error_message: Optional[str] = None
    system_load_before: Optional[Dict[str, float]] = None
    system_load_after: Optional[Dict[str, float]] = None
    performance_impact: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.affected_tables is None:
            self.affected_tables = []


class DatabaseMaintenanceScheduler:
    """
    Intelligent database maintenance scheduler with:
    - Automated task scheduling based on system load
    - Performance-aware execution
    - Load balancing and resource management
    - Comprehensive monitoring and reporting
    """
    
    def __init__(self, connection_manager, performance_optimizer=None, partition_manager=None):
        self.connection_manager = connection_manager
        self.performance_optimizer = performance_optimizer
        self.partition_manager = partition_manager
        self.logger = logging.getLogger(__name__)
        
        # Scheduler state
        self.tasks: Dict[str, MaintenanceTask] = {}
        self.executions: List[MaintenanceExecution] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.scheduler_running = False
        
        # Performance thresholds
        self.max_cpu_usage = 80.0
        self.max_memory_usage = 85.0
        self.max_active_connections_percent = 70.0
        self.min_cache_hit_ratio = 95.0
        
        # Initialize default maintenance tasks
        self._initialize_default_tasks()
        
    def _initialize_default_tasks(self):
        """Initialize default maintenance tasks"""
        
        # Daily VACUUM for high-traffic tables
        self.tasks['daily_vacuum_high_traffic'] = MaintenanceTask(
            id='daily_vacuum_high_traffic',
            name='Daily VACUUM for High-Traffic Tables',
            maintenance_type=MaintenanceType.VACUUM,
            priority=MaintenancePriority.HIGH,
            schedule_cron='0 2 * * *',  # 2 AM daily
            target_tables=['properties', 'user_interactions'],
            parameters={'vacuum_type': 'VACUUM ANALYZE', 'verbose': True},
            max_duration_minutes=60,
            skip_if_high_load=True,
            require_low_activity=True
        )
        
        # Weekly VACUUM for all tables
        self.tasks['weekly_vacuum_all'] = MaintenanceTask(
            id='weekly_vacuum_all',
            name='Weekly VACUUM for All Tables',
            maintenance_type=MaintenanceType.VACUUM,
            priority=MaintenancePriority.MEDIUM,
            schedule_cron='0 3 * * 0',  # 3 AM Sunday
            target_tables=[],  # Empty means all tables
            parameters={'vacuum_type': 'VACUUM FULL ANALYZE', 'verbose': True},
            max_duration_minutes=180,
            skip_if_high_load=True,
            require_low_activity=True
        )
        
        # Daily statistics update
        self.tasks['daily_analyze'] = MaintenanceTask(
            id='daily_analyze',
            name='Daily Statistics Update',
            maintenance_type=MaintenanceType.ANALYZE,
            priority=MaintenancePriority.MEDIUM,
            schedule_cron='0 1 * * *',  # 1 AM daily
            target_tables=['properties', 'user_interactions', 'search_queries'],
            parameters={'analyze_type': 'ANALYZE'},
            max_duration_minutes=30,
            skip_if_high_load=False,
            require_low_activity=False
        )
        
        # Weekly REINDEX for critical indexes
        self.tasks['weekly_reindex'] = MaintenanceTask(
            id='weekly_reindex',
            name='Weekly REINDEX for Critical Indexes',
            maintenance_type=MaintenanceType.REINDEX,
            priority=MaintenancePriority.MEDIUM,
            schedule_cron='0 4 * * 6',  # 4 AM Saturday
            target_tables=['properties', 'user_interactions'],
            parameters={'reindex_type': 'REINDEX CONCURRENTLY'},
            max_duration_minutes=120,
            skip_if_high_load=True,
            require_low_activity=True
        )
        
        # Daily partition maintenance
        self.tasks['daily_partition_maintenance'] = MaintenanceTask(
            id='daily_partition_maintenance',
            name='Daily Partition Maintenance',
            maintenance_type=MaintenanceType.PARTITION_MAINTENANCE,
            priority=MaintenancePriority.HIGH,
            schedule_cron='0 0 * * *',  # Midnight daily
            target_tables=['properties', 'user_interactions', 'search_queries'],
            parameters={},
            max_duration_minutes=30,
            skip_if_high_load=False,
            require_low_activity=False
        )
        
        # Hourly statistics update for pg_stat_statements
        self.tasks['hourly_stats_reset'] = MaintenanceTask(
            id='hourly_stats_reset',
            name='Hourly Statistics Reset',
            maintenance_type=MaintenanceType.STATISTICS_UPDATE,
            priority=MaintenancePriority.LOW,
            schedule_cron='0 * * * *',  # Top of every hour
            target_tables=[],
            parameters={'reset_pg_stat_statements': True, 'reset_threshold_hours': 24},
            max_duration_minutes=5,
            skip_if_high_load=False,
            require_low_activity=False
        )
        
        # Weekly index optimization
        self.tasks['weekly_index_optimization'] = MaintenanceTask(
            id='weekly_index_optimization',
            name='Weekly Index Optimization',
            maintenance_type=MaintenanceType.INDEX_OPTIMIZATION,
            priority=MaintenancePriority.MEDIUM,
            schedule_cron='0 5 * * 1',  # 5 AM Monday
            target_tables=[],
            parameters={'analyze_unused_indexes': True, 'create_missing_indexes': True},
            max_duration_minutes=90,
            skip_if_high_load=True,
            require_low_activity=True
        )
        
        # Update next run times
        for task in self.tasks.values():
            task.next_run = self._calculate_next_run(task)
            
    def _calculate_next_run(self, task: MaintenanceTask) -> datetime:
        """Calculate next run time based on cron schedule"""
        try:
            cron = croniter.croniter(task.schedule_cron, datetime.now())
            return cron.get_next(datetime)
        except Exception as e:
            self.logger.error(f"Invalid cron expression for task {task.id}: {e}")
            # Default to 24 hours from now
            return datetime.now() + timedelta(hours=24)
            
    async def start_scheduler(self):
        """Start the maintenance scheduler"""
        if self.scheduler_running:
            self.logger.warning("Scheduler is already running")
            return
            
        self.scheduler_running = True
        self.logger.info("Starting database maintenance scheduler")
        
        try:
            while self.scheduler_running:
                await self._scheduler_loop()
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            self.logger.info("Scheduler cancelled")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
        finally:
            self.scheduler_running = False
            
    async def stop_scheduler(self):
        """Stop the maintenance scheduler"""
        self.logger.info("Stopping database maintenance scheduler")
        self.scheduler_running = False
        
        # Cancel running tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        self.running_tasks.clear()
        
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        current_time = datetime.now()
        
        # Check for tasks that need to run
        for task in self.tasks.values():
            if not task.enabled:
                continue
                
            if task.next_run <= current_time and task.id not in self.running_tasks:
                # Check if we should run this task
                should_run = await self._should_run_task(task)
                
                if should_run:
                    # Start task execution
                    task_coroutine = self._execute_task(task)
                    self.running_tasks[task.id] = asyncio.create_task(task_coroutine)
                else:
                    # Skip this run and schedule next
                    task.next_run = self._calculate_next_run(task)
                    self.logger.info(f"Skipped task {task.id} due to system conditions")
                    
        # Clean up completed tasks
        completed_tasks = []
        for task_id, task in self.running_tasks.items():
            if task.done():
                completed_tasks.append(task_id)
                try:
                    await task  # Get any exceptions
                except Exception as e:
                    self.logger.error(f"Task {task_id} failed: {e}")
                    
        for task_id in completed_tasks:
            del self.running_tasks[task_id]
            
    async def _should_run_task(self, task: MaintenanceTask) -> bool:
        """Determine if a task should run based on system conditions"""
        
        # Check if task requires low activity and system is busy
        if task.require_low_activity:
            if not await self._is_system_activity_low():
                return False
                
        # Check if task should skip during high load
        if task.skip_if_high_load:
            if await self._is_system_load_high():
                return False
                
        # Check maximum concurrent maintenance tasks
        if len(self.running_tasks) >= 2:  # Max 2 concurrent maintenance tasks
            return False
            
        return True
        
    async def _is_system_activity_low(self) -> bool:
        """Check if system activity is low enough for maintenance"""
        try:
            async with self.connection_manager.get_connection(analytics=True) as conn:
                # Check active connections
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as active_connections,
                        (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections,
                        COUNT(*) FILTER (WHERE state = 'active') as running_queries
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                """)
                
                connection_usage = (stats['active_connections'] / stats['max_connections']) * 100
                
                # Consider low activity if connection usage < 30% and running queries < 5
                return connection_usage < 30 and stats['running_queries'] < 5
                
        except Exception as e:
            self.logger.error(f"Failed to check system activity: {e}")
            return False
            
    async def _is_system_load_high(self) -> bool:
        """Check if system load is too high for maintenance"""
        try:
            # Check database load
            async with self.connection_manager.get_connection(analytics=True) as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as active_connections,
                        (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections,
                        COUNT(*) FILTER (WHERE state = 'active' AND query NOT LIKE '%pg_stat%') as user_queries
                    FROM pg_stat_activity
                """)
                
                connection_usage = (stats['active_connections'] / stats['max_connections']) * 100
                
                if connection_usage > self.max_active_connections_percent:
                    return True
                    
                if stats['user_queries'] > 10:  # More than 10 active user queries
                    return True
                    
            # Could also check system CPU/memory if psutil is available
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            if cpu_usage > self.max_cpu_usage or memory_usage > self.max_memory_usage:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check system load: {e}")
            return True  # Err on the side of caution
            
    async def _execute_task(self, task: MaintenanceTask):
        """Execute a maintenance task"""
        execution_id = f"{task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        execution = MaintenanceExecution(
            task_id=task.id,
            execution_id=execution_id,
            started_at=datetime.now(),
            status=MaintenanceStatus.RUNNING
        )
        
        # Record system load before execution
        execution.system_load_before = await self._get_system_metrics()
        
        self.executions.append(execution)
        self.logger.info(f"Starting maintenance task: {task.name}")
        
        try:
            # Execute based on maintenance type
            if task.maintenance_type == MaintenanceType.VACUUM:
                await self._execute_vacuum_task(task, execution)
            elif task.maintenance_type == MaintenanceType.ANALYZE:
                await self._execute_analyze_task(task, execution)
            elif task.maintenance_type == MaintenanceType.REINDEX:
                await self._execute_reindex_task(task, execution)
            elif task.maintenance_type == MaintenanceType.PARTITION_MAINTENANCE:
                await self._execute_partition_maintenance_task(task, execution)
            elif task.maintenance_type == MaintenanceType.STATISTICS_UPDATE:
                await self._execute_statistics_update_task(task, execution)
            elif task.maintenance_type == MaintenanceType.INDEX_OPTIMIZATION:
                await self._execute_index_optimization_task(task, execution)
            else:
                raise ValueError(f"Unknown maintenance type: {task.maintenance_type}")
                
            execution.status = MaintenanceStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            
            # Update task statistics
            task.last_run = execution.started_at
            task.next_run = self._calculate_next_run(task)
            task.run_count += 1
            
            self.logger.info(f"Completed maintenance task: {task.name} in {execution.duration_seconds:.2f}s")
            
        except asyncio.TimeoutError:
            execution.status = MaintenanceStatus.FAILED
            execution.error_message = f"Task timed out after {task.max_duration_minutes} minutes"
            task.failure_count += 1
            self.logger.error(f"Task {task.name} timed out")
            
        except Exception as e:
            execution.status = MaintenanceStatus.FAILED
            execution.error_message = str(e)
            task.failure_count += 1
            self.logger.error(f"Task {task.name} failed: {e}")
            
        finally:
            # Record system load after execution
            execution.system_load_after = await self._get_system_metrics()
            
            # Calculate performance impact
            if execution.system_load_before and execution.system_load_after:
                execution.performance_impact = self._calculate_performance_impact(
                    execution.system_load_before, execution.system_load_after
                )
                
    async def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        metrics = {}
        
        try:
            async with self.connection_manager.get_connection(analytics=True) as conn:
                # Database metrics
                db_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as active_connections,
                        COUNT(*) FILTER (WHERE state = 'active') as running_queries,
                        ROUND(
                            100.0 * SUM(blks_hit) / NULLIF(SUM(blks_hit + blks_read), 0), 2
                        ) as cache_hit_ratio
                    FROM pg_stat_activity, pg_stat_database
                    WHERE pg_stat_database.datname = current_database()
                """)
                
                metrics.update({
                    'active_connections': db_stats['active_connections'],
                    'running_queries': db_stats['running_queries'],
                    'cache_hit_ratio': db_stats['cache_hit_ratio'] or 0
                })
                
            # System metrics
            import psutil
            metrics.update({
                'cpu_usage_percent': psutil.cpu_percent(),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'disk_io_read_mb': psutil.disk_io_counters().read_bytes / (1024*1024),
                'disk_io_write_mb': psutil.disk_io_counters().write_bytes / (1024*1024)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            
        return metrics
        
    def _calculate_performance_impact(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Any]:
        """Calculate performance impact of maintenance task"""
        impact = {}
        
        for metric in ['cpu_usage_percent', 'memory_usage_percent', 'active_connections']:
            if metric in before and metric in after:
                change = after[metric] - before[metric]
                impact[f"{metric}_change"] = round(change, 2)
                
        return impact
        
    async def _execute_vacuum_task(self, task: MaintenanceTask, execution: MaintenanceExecution):
        """Execute VACUUM maintenance task"""
        vacuum_type = task.parameters.get('vacuum_type', 'VACUUM ANALYZE')
        verbose = task.parameters.get('verbose', False)
        
        tables_to_vacuum = task.target_tables if task.target_tables else await self._get_all_user_tables()
        
        async with self.connection_manager.get_connection() as conn:
            for table in tables_to_vacuum:
                try:
                    vacuum_sql = f"{vacuum_type} {table}"
                    if verbose:
                        vacuum_sql += " VERBOSE"
                        
                    await conn.execute(vacuum_sql)
                    execution.affected_tables.append(table)
                    self.logger.debug(f"Vacuumed table: {table}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to vacuum table {table}: {e}")
                    
    async def _execute_analyze_task(self, task: MaintenanceTask, execution: MaintenanceExecution):
        """Execute ANALYZE maintenance task"""
        tables_to_analyze = task.target_tables if task.target_tables else await self._get_all_user_tables()
        
        async with self.connection_manager.get_connection() as conn:
            for table in tables_to_analyze:
                try:
                    await conn.execute(f"ANALYZE {table}")
                    execution.affected_tables.append(table)
                    self.logger.debug(f"Analyzed table: {table}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze table {table}: {e}")
                    
    async def _execute_reindex_task(self, task: MaintenanceTask, execution: MaintenanceExecution):
        """Execute REINDEX maintenance task"""
        reindex_type = task.parameters.get('reindex_type', 'REINDEX TABLE CONCURRENTLY')
        
        async with self.connection_manager.get_connection() as conn:
            for table in task.target_tables:
                try:
                    await conn.execute(f"{reindex_type} {table}")
                    execution.affected_tables.append(table)
                    self.logger.debug(f"Reindexed table: {table}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to reindex table {table}: {e}")
                    
    async def _execute_partition_maintenance_task(self, task: MaintenanceTask, execution: MaintenanceExecution):
        """Execute partition maintenance task"""
        if not self.partition_manager:
            self.logger.warning("Partition manager not available")
            return
            
        try:
            results = await self.partition_manager.run_partition_maintenance()
            execution.affected_tables.extend(results.get('tables_processed', []))
            
            # Log results
            self.logger.info(f"Partition maintenance: {len(results.get('new_partitions_created', []))} created, "
                           f"{len(results.get('old_partitions_dropped', []))} dropped")
                           
        except Exception as e:
            self.logger.error(f"Partition maintenance failed: {e}")
            raise
            
    async def _execute_statistics_update_task(self, task: MaintenanceTask, execution: MaintenanceExecution):
        """Execute statistics update task"""
        reset_pg_stat_statements = task.parameters.get('reset_pg_stat_statements', False)
        reset_threshold_hours = task.parameters.get('reset_threshold_hours', 24)
        
        async with self.connection_manager.get_connection() as conn:
            if reset_pg_stat_statements:
                try:
                    # Check if pg_stat_statements data is old enough to reset
                    oldest_query = await conn.fetchval("""
                        SELECT EXTRACT(EPOCH FROM (NOW() - query_start)) / 3600
                        FROM pg_stat_activity 
                        WHERE state != 'idle' 
                        ORDER BY query_start 
                        LIMIT 1
                    """)
                    
                    if not oldest_query or oldest_query > reset_threshold_hours:
                        await conn.execute("SELECT pg_stat_statements_reset()")
                        self.logger.info("Reset pg_stat_statements")
                        
                except Exception as e:
                    self.logger.error(f"Failed to reset pg_stat_statements: {e}")
                    
    async def _execute_index_optimization_task(self, task: MaintenanceTask, execution: MaintenanceExecution):
        """Execute index optimization task"""
        if not self.performance_optimizer:
            self.logger.warning("Performance optimizer not available")
            return
            
        try:
            # Analyze index performance
            index_analyses = await self.performance_optimizer.analyze_index_performance()
            
            # Find unused indexes (except primary keys and unique constraints)
            unused_indexes = [
                idx for idx in index_analyses 
                if idx.index_scans == 0 and not idx.is_primary and not idx.is_unique
                and idx.index_size > 10 * 1024 * 1024  # Only drop if > 10MB
            ]
            
            # Drop unused indexes (with caution)
            async with self.connection_manager.get_connection() as conn:
                for index in unused_indexes[:5]:  # Limit to 5 per run
                    try:
                        await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {index.index_name}")
                        execution.affected_tables.append(index.table_name)
                        self.logger.info(f"Dropped unused index: {index.index_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to drop index {index.index_name}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            raise
            
    async def _get_all_user_tables(self) -> List[str]:
        """Get list of all user tables"""
        async with self.connection_manager.get_connection(analytics=True) as conn:
            rows = await conn.fetch("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            return [row['tablename'] for row in rows]
            
    async def add_task(self, task: MaintenanceTask):
        """Add a new maintenance task"""
        task.next_run = self._calculate_next_run(task)
        self.tasks[task.id] = task
        self.logger.info(f"Added maintenance task: {task.name}")
        
    async def remove_task(self, task_id: str):
        """Remove a maintenance task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.logger.info(f"Removed maintenance task: {task_id}")
            
        # Cancel if currently running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            
    async def enable_task(self, task_id: str):
        """Enable a maintenance task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.tasks[task_id].next_run = self._calculate_next_run(self.tasks[task_id])
            
    async def disable_task(self, task_id: str):
        """Disable a maintenance task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            
    async def run_task_immediately(self, task_id: str) -> MaintenanceExecution:
        """Run a specific task immediately"""
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")
            
        task = self.tasks[task_id]
        
        if task_id in self.running_tasks:
            raise ValueError(f"Task is already running: {task_id}")
            
        # Execute task immediately
        await self._execute_task(task)
        
        # Return the most recent execution
        task_executions = [exec for exec in self.executions if exec.task_id == task_id]
        return task_executions[-1] if task_executions else None
        
    async def get_task_status(self) -> Dict[str, Any]:
        """Get status of all maintenance tasks"""
        status = {
            'tasks': [],
            'running_tasks': list(self.running_tasks.keys()),
            'recent_executions': [],
            'summary': {
                'total_tasks': len(self.tasks),
                'enabled_tasks': len([t for t in self.tasks.values() if t.enabled]),
                'running_tasks': len(self.running_tasks),
                'failed_executions_last_24h': 0,
                'successful_executions_last_24h': 0
            }
        }
        
        # Task information
        for task in self.tasks.values():
            task_info = {
                'id': task.id,
                'name': task.name,
                'type': task.maintenance_type.value,
                'priority': task.priority.value,
                'enabled': task.enabled,
                'schedule': task.schedule_cron,
                'next_run': task.next_run.isoformat() if task.next_run else None,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'run_count': task.run_count,
                'failure_count': task.failure_count
            }
            status['tasks'].append(task_info)
            
        # Recent executions (last 10)
        recent_executions = sorted(self.executions, key=lambda x: x.started_at, reverse=True)[:10]
        for execution in recent_executions:
            exec_info = {
                'task_id': execution.task_id,
                'execution_id': execution.execution_id,
                'started_at': execution.started_at.isoformat(),
                'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
                'status': execution.status.value,
                'duration_seconds': execution.duration_seconds,
                'affected_tables': execution.affected_tables,
                'error_message': execution.error_message
            }
            status['recent_executions'].append(exec_info)
            
        # Calculate summary statistics
        last_24h = datetime.now() - timedelta(hours=24)
        recent_executions = [exec for exec in self.executions if exec.started_at > last_24h]
        
        status['summary']['failed_executions_last_24h'] = len([
            exec for exec in recent_executions if exec.status == MaintenanceStatus.FAILED
        ])
        status['summary']['successful_executions_last_24h'] = len([
            exec for exec in recent_executions if exec.status == MaintenanceStatus.COMPLETED
        ])
        
        return status