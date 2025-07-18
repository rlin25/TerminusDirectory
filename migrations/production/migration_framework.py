"""
Production-Safe Database Migration Framework
Supports rollback, validation, monitoring, and safe deployment
"""

import asyncio
import logging
import os
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import yaml
import asyncpg
from asyncpg import Connection
import psutil


class MigrationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationSafety(Enum):
    SAFE = "safe"              # No locks, no data loss risk
    CAUTION = "caution"        # Might cause brief locks
    DANGEROUS = "dangerous"    # Could cause extended downtime


@dataclass
class MigrationMetadata:
    version: str
    name: str
    description: str
    author: str
    created_at: datetime
    safety_level: MigrationSafety
    estimated_duration: int  # seconds
    requires_downtime: bool
    dependencies: List[str]
    affects_tables: List[str]
    rollback_strategy: str
    validation_queries: List[str]
    pre_checks: List[str]
    post_checks: List[str]
    
    
@dataclass
class MigrationExecution:
    version: str
    status: MigrationStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    rollback_executed: bool = False
    validation_passed: bool = False
    affected_rows: int = 0
    execution_log: List[str] = None
    
    def __post_init__(self):
        if self.execution_log is None:
            self.execution_log = []


class ProductionMigrationManager:
    """
    Production-grade migration manager with:
    - Rollback capabilities
    - Safety checks and validations
    - Performance monitoring
    - Lock detection and management
    - Incremental deployment support
    """
    
    def __init__(self, connection_manager, config_path: str = None):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config_path = config_path or "migrations/production/config.yaml"
        self.migrations_dir = Path("migrations/production/scripts")
        self.config = self._load_config()
        
        # State tracking
        self.current_executions: Dict[str, MigrationExecution] = {}
        self.migration_history: List[MigrationExecution] = []
        
        # Safety limits
        self.max_lock_wait_time = self.config.get('max_lock_wait_time', 300)  # 5 minutes
        self.max_migration_time = self.config.get('max_migration_time', 3600)  # 1 hour
        self.max_affected_rows = self.config.get('max_affected_rows', 1000000)  # 1M rows
        
    def _load_config(self) -> Dict[str, Any]:
        """Load migration configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"Could not load migration config: {e}")
            return {}
            
    async def initialize(self):
        """Initialize migration tracking tables"""
        async with self.connection_manager.get_connection() as conn:
            await self._create_migration_tables(conn)
            await self._load_migration_history(conn)
            
    async def _create_migration_tables(self, conn: Connection):
        """Create migration tracking tables if they don't exist"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_versions (
                version VARCHAR(255) PRIMARY KEY,
                name VARCHAR(500) NOT NULL,
                description TEXT,
                author VARCHAR(255),
                safety_level VARCHAR(50),
                estimated_duration INTEGER,
                requires_downtime BOOLEAN DEFAULT FALSE,
                dependencies TEXT[], -- JSON array of dependency versions
                affects_tables TEXT[],
                rollback_strategy TEXT,
                file_hash VARCHAR(64),
                metadata JSONB,
                applied_at TIMESTAMP WITH TIME ZONE,
                applied_by VARCHAR(255),
                execution_time_seconds REAL,
                rollback_executed BOOLEAN DEFAULT FALSE,
                rollback_at TIMESTAMP WITH TIME ZONE,
                rollback_by VARCHAR(255),
                status VARCHAR(50) DEFAULT 'pending',
                error_message TEXT,
                affected_rows INTEGER DEFAULT 0,
                validation_passed BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_execution_log (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                migration_version VARCHAR(255) REFERENCES migration_versions(version),
                execution_step VARCHAR(255),
                started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP WITH TIME ZONE,
                duration_seconds REAL,
                status VARCHAR(50),
                message TEXT,
                error_details JSONB,
                query_executed TEXT,
                rows_affected INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_locks (
                lock_name VARCHAR(255) PRIMARY KEY,
                migration_version VARCHAR(255),
                locked_by VARCHAR(255),
                locked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP WITH TIME ZONE,
                metadata JSONB
            );
        """)
        
        # Create indexes for performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_migration_versions_status 
            ON migration_versions(status);
            
            CREATE INDEX IF NOT EXISTS idx_migration_versions_applied_at 
            ON migration_versions(applied_at);
            
            CREATE INDEX IF NOT EXISTS idx_migration_execution_log_version 
            ON migration_execution_log(migration_version);
            
            CREATE INDEX IF NOT EXISTS idx_migration_execution_log_started_at 
            ON migration_execution_log(started_at);
        """)
        
    async def _load_migration_history(self, conn: Connection):
        """Load existing migration history from database"""
        rows = await conn.fetch("""
            SELECT version, status, applied_at, execution_time_seconds, 
                   error_message, rollback_executed, affected_rows
            FROM migration_versions 
            ORDER BY applied_at
        """)
        
        self.migration_history = []
        for row in rows:
            execution = MigrationExecution(
                version=row['version'],
                status=MigrationStatus(row['status']),
                started_at=row['applied_at'],
                completed_at=row['applied_at'],
                duration_seconds=row['execution_time_seconds'],
                error_message=row['error_message'],
                rollback_executed=row['rollback_executed'],
                affected_rows=row['affected_rows'] or 0
            )
            self.migration_history.append(execution)
            
    def discover_migrations(self) -> List[MigrationMetadata]:
        """Discover all available migration files"""
        migrations = []
        
        if not self.migrations_dir.exists():
            self.logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return migrations
            
        for migration_file in sorted(self.migrations_dir.glob("*.sql")):
            try:
                metadata = self._parse_migration_file(migration_file)
                migrations.append(metadata)
            except Exception as e:
                self.logger.error(f"Failed to parse migration {migration_file}: {e}")
                
        return migrations
        
    def _parse_migration_file(self, file_path: Path) -> MigrationMetadata:
        """Parse migration file and extract metadata"""
        content = file_path.read_text()
        lines = content.split('\n')
        
        metadata = {
            'version': file_path.stem,
            'name': '',
            'description': '',
            'author': '',
            'created_at': datetime.utcnow(),
            'safety_level': MigrationSafety.CAUTION,
            'estimated_duration': 60,
            'requires_downtime': False,
            'dependencies': [],
            'affects_tables': [],
            'rollback_strategy': 'manual',
            'validation_queries': [],
            'pre_checks': [],
            'post_checks': []
        }
        
        # Parse header comments for metadata
        for line in lines:
            line = line.strip()
            if not line.startswith('--'):
                break
                
            if ':' in line:
                key, value = line[2:].split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in metadata:
                    if key in ['dependencies', 'affects_tables', 'validation_queries', 'pre_checks', 'post_checks']:
                        metadata[key] = [v.strip() for v in value.split(',') if v.strip()]
                    elif key == 'safety_level':
                        metadata[key] = MigrationSafety(value.lower())
                    elif key in ['estimated_duration', 'requires_downtime']:
                        if key == 'estimated_duration':
                            metadata[key] = int(value)
                        else:
                            metadata[key] = value.lower() in ['true', 'yes', '1']
                    elif key == 'created_at':
                        metadata[key] = datetime.fromisoformat(value)
                    else:
                        metadata[key] = value
                        
        return MigrationMetadata(**metadata)
        
    async def get_pending_migrations(self) -> List[MigrationMetadata]:
        """Get list of migrations that haven't been applied"""
        all_migrations = self.discover_migrations()
        applied_versions = {exec.version for exec in self.migration_history 
                          if exec.status == MigrationStatus.COMPLETED}
        
        pending = [m for m in all_migrations if m.version not in applied_versions]
        
        # Sort by dependencies and version
        return self._sort_migrations_by_dependencies(pending)
        
    def _sort_migrations_by_dependencies(self, migrations: List[MigrationMetadata]) -> List[MigrationMetadata]:
        """Sort migrations by their dependencies"""
        # Simple topological sort
        sorted_migrations = []
        remaining = migrations.copy()
        
        while remaining:
            # Find migrations with satisfied dependencies
            ready = []
            for migration in remaining:
                dependencies_satisfied = all(
                    dep in [m.version for m in sorted_migrations] or 
                    dep in [exec.version for exec in self.migration_history 
                           if exec.status == MigrationStatus.COMPLETED]
                    for dep in migration.dependencies
                )
                if dependencies_satisfied:
                    ready.append(migration)
                    
            if not ready:
                # Circular dependency or missing dependency
                self.logger.warning("Circular dependency detected in migrations")
                ready = remaining  # Add remaining as-is
                
            for migration in ready:
                sorted_migrations.append(migration)
                remaining.remove(migration)
                
        return sorted_migrations
        
    async def validate_migration(self, metadata: MigrationMetadata) -> Tuple[bool, List[str]]:
        """Validate migration before execution"""
        issues = []
        
        # Check dependencies
        for dep in metadata.dependencies:
            if not any(exec.version == dep and exec.status == MigrationStatus.COMPLETED 
                      for exec in self.migration_history):
                issues.append(f"Dependency not satisfied: {dep}")
                
        # Check if already applied
        if any(exec.version == metadata.version and exec.status == MigrationStatus.COMPLETED 
               for exec in self.migration_history):
            issues.append(f"Migration {metadata.version} already applied")
            
        # Safety checks
        if metadata.safety_level == MigrationSafety.DANGEROUS:
            if not self.config.get('allow_dangerous_migrations', False):
                issues.append("Dangerous migrations not allowed in current configuration")
                
        # Check system resources
        try:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=1)
            
            if memory_usage > 85:
                issues.append(f"High memory usage: {memory_usage}%")
            if cpu_usage > 80:
                issues.append(f"High CPU usage: {cpu_usage}%")
        except Exception as e:
            self.logger.warning(f"Could not check system resources: {e}")
            
        # Pre-flight checks
        if metadata.pre_checks:
            async with self.connection_manager.get_connection() as conn:
                for check_query in metadata.pre_checks:
                    try:
                        result = await conn.fetchval(check_query)
                        if not result:
                            issues.append(f"Pre-flight check failed: {check_query}")
                    except Exception as e:
                        issues.append(f"Pre-flight check error: {e}")
                        
        return len(issues) == 0, issues
        
    async def execute_migration(self, metadata: MigrationMetadata, 
                              dry_run: bool = False) -> MigrationExecution:
        """Execute a single migration with full monitoring and rollback support"""
        execution = MigrationExecution(
            version=metadata.version,
            status=MigrationStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        self.current_executions[metadata.version] = execution
        
        try:
            # Validation
            valid, issues = await self.validate_migration(metadata)
            if not valid:
                raise Exception(f"Migration validation failed: {', '.join(issues)}")
                
            # Acquire migration lock
            lock_acquired = await self._acquire_migration_lock(metadata.version)
            if not lock_acquired:
                raise Exception("Could not acquire migration lock")
                
            try:
                # Load migration SQL
                migration_file = self.migrations_dir / f"{metadata.version}.sql"
                sql_content = migration_file.read_text()
                
                if dry_run:
                    execution.status = MigrationStatus.COMPLETED
                    execution.completed_at = datetime.utcnow()
                    execution.duration_seconds = 0
                    self.logger.info(f"Dry run completed for migration {metadata.version}")
                    return execution
                    
                # Execute migration with monitoring
                await self._execute_migration_sql(execution, metadata, sql_content)
                
                # Validation checks
                if metadata.validation_queries:
                    await self._run_validation_checks(execution, metadata)
                    
                # Post-execution checks
                if metadata.post_checks:
                    await self._run_post_checks(execution, metadata)
                    
                execution.status = MigrationStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.duration_seconds = (
                    execution.completed_at - execution.started_at
                ).total_seconds()
                
                # Record successful migration
                await self._record_migration_success(execution, metadata)
                
                self.logger.info(f"Migration {metadata.version} completed successfully")
                
            finally:
                await self._release_migration_lock(metadata.version)
                
        except Exception as e:
            execution.status = MigrationStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds() if execution.started_at else 0
            
            # Record failed migration
            await self._record_migration_failure(execution, metadata)
            
            self.logger.error(f"Migration {metadata.version} failed: {e}")
            
        finally:
            if metadata.version in self.current_executions:
                del self.current_executions[metadata.version]
                
        return execution
        
    async def _execute_migration_sql(self, execution: MigrationExecution, 
                                   metadata: MigrationMetadata, sql_content: str):
        """Execute migration SQL with monitoring and timeout"""
        async with self.connection_manager.get_connection() as conn:
            # Start transaction
            async with conn.transaction():
                # Set timeout
                await conn.execute(f"SET statement_timeout = '{self.max_migration_time * 1000}'")
                
                # Parse and execute SQL statements
                statements = self._parse_sql_statements(sql_content)
                
                for i, statement in enumerate(statements):
                    if not statement.strip():
                        continue
                        
                    step_start = time.time()
                    execution.execution_log.append(f"Executing step {i+1}: {statement[:100]}...")
                    
                    try:
                        # Execute with monitoring
                        if statement.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                            result = await conn.execute(statement)
                            # Extract affected rows from result
                            if hasattr(result, 'split'):
                                parts = result.split()
                                if len(parts) >= 2 and parts[1].isdigit():
                                    step_affected = int(parts[1])
                                    execution.affected_rows += step_affected
                                    
                                    # Check affected rows limit
                                    if execution.affected_rows > self.max_affected_rows:
                                        raise Exception(f"Too many rows affected: {execution.affected_rows}")
                        else:
                            await conn.execute(statement)
                            
                        step_duration = time.time() - step_start
                        execution.execution_log.append(f"Step {i+1} completed in {step_duration:.2f}s")
                        
                        # Log to execution table
                        await self._log_execution_step(conn, execution.version, 
                                                     f"step_{i+1}", statement, 
                                                     step_duration, "completed")
                        
                    except Exception as e:
                        step_duration = time.time() - step_start
                        execution.execution_log.append(f"Step {i+1} failed: {e}")
                        
                        # Log failed step
                        await self._log_execution_step(conn, execution.version, 
                                                     f"step_{i+1}", statement, 
                                                     step_duration, "failed", str(e))
                        raise
                        
    def _parse_sql_statements(self, sql_content: str) -> List[str]:
        """Parse SQL content into individual statements"""
        # Simple SQL statement parser (could be enhanced)
        statements = []
        current_statement = []
        
        for line in sql_content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('--'):
                continue
                
            current_statement.append(line)
            
            # Statement ends with semicolon
            if line.endswith(';'):
                statements.append(' '.join(current_statement))
                current_statement = []
                
        # Add any remaining statement
        if current_statement:
            statements.append(' '.join(current_statement))
            
        return statements
        
    async def _run_validation_checks(self, execution: MigrationExecution, 
                                   metadata: MigrationMetadata):
        """Run post-migration validation checks"""
        async with self.connection_manager.get_connection(read_only=True) as conn:
            for query in metadata.validation_queries:
                try:
                    result = await conn.fetchval(query)
                    if not result:
                        execution.validation_passed = False
                        raise Exception(f"Validation check failed: {query}")
                        
                    execution.execution_log.append(f"Validation passed: {query}")
                    
                except Exception as e:
                    execution.validation_passed = False
                    execution.execution_log.append(f"Validation failed: {query} - {e}")
                    raise
                    
    async def _run_post_checks(self, execution: MigrationExecution, 
                             metadata: MigrationMetadata):
        """Run post-migration checks"""
        async with self.connection_manager.get_connection(read_only=True) as conn:
            for check_query in metadata.post_checks:
                try:
                    result = await conn.fetchval(check_query)
                    execution.execution_log.append(f"Post-check result: {check_query} = {result}")
                except Exception as e:
                    execution.execution_log.append(f"Post-check error: {check_query} - {e}")
                    
    async def rollback_migration(self, version: str) -> MigrationExecution:
        """Rollback a previously applied migration"""
        # Find the migration in history
        migration_exec = None
        for exec in reversed(self.migration_history):
            if exec.version == version and exec.status == MigrationStatus.COMPLETED:
                migration_exec = exec
                break
                
        if not migration_exec:
            raise Exception(f"Migration {version} not found or not applied")
            
        if migration_exec.rollback_executed:
            raise Exception(f"Migration {version} already rolled back")
            
        # Load migration metadata
        metadata = None
        for m in self.discover_migrations():
            if m.version == version:
                metadata = m
                break
                
        if not metadata:
            raise Exception(f"Migration metadata not found for {version}")
            
        # Create rollback execution record
        rollback_execution = MigrationExecution(
            version=f"{version}_rollback",
            status=MigrationStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Load rollback SQL
            rollback_file = self.migrations_dir / f"{version}_rollback.sql"
            if not rollback_file.exists():
                raise Exception(f"Rollback file not found: {rollback_file}")
                
            rollback_sql = rollback_file.read_text()
            
            # Execute rollback
            await self._execute_migration_sql(rollback_execution, metadata, rollback_sql)
            
            # Update original migration record
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("""
                    UPDATE migration_versions 
                    SET rollback_executed = TRUE, 
                        rollback_at = $1,
                        status = 'rolled_back'
                    WHERE version = $2
                """, datetime.utcnow(), version)
                
            rollback_execution.status = MigrationStatus.COMPLETED
            rollback_execution.completed_at = datetime.utcnow()
            rollback_execution.duration_seconds = (
                rollback_execution.completed_at - rollback_execution.started_at
            ).total_seconds()
            
            # Update local history
            migration_exec.rollback_executed = True
            migration_exec.status = MigrationStatus.ROLLED_BACK
            
            self.logger.info(f"Migration {version} rolled back successfully")
            
        except Exception as e:
            rollback_execution.status = MigrationStatus.FAILED
            rollback_execution.error_message = str(e)
            rollback_execution.completed_at = datetime.utcnow()
            rollback_execution.duration_seconds = (
                rollback_execution.completed_at - rollback_execution.started_at
            ).total_seconds() if rollback_execution.started_at else 0
            
            self.logger.error(f"Rollback failed for migration {version}: {e}")
            raise
            
        return rollback_execution
        
    async def _acquire_migration_lock(self, version: str) -> bool:
        """Acquire migration lock to prevent concurrent executions"""
        try:
            async with self.connection_manager.get_connection() as conn:
                # Try to acquire lock
                result = await conn.execute("""
                    INSERT INTO migration_locks (lock_name, migration_version, locked_by, expires_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (lock_name) DO NOTHING
                """, f"migration_{version}", version, "migration_system", 
                    datetime.utcnow() + timedelta(seconds=self.max_migration_time))
                
                return "INSERT" in result
                
        except Exception as e:
            self.logger.error(f"Failed to acquire migration lock: {e}")
            return False
            
    async def _release_migration_lock(self, version: str):
        """Release migration lock"""
        try:
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("""
                    DELETE FROM migration_locks 
                    WHERE lock_name = $1
                """, f"migration_{version}")
        except Exception as e:
            self.logger.error(f"Failed to release migration lock: {e}")
            
    async def _record_migration_success(self, execution: MigrationExecution, 
                                      metadata: MigrationMetadata):
        """Record successful migration in database"""
        async with self.connection_manager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO migration_versions (
                    version, name, description, author, safety_level,
                    estimated_duration, requires_downtime, dependencies,
                    affects_tables, rollback_strategy, metadata,
                    applied_at, execution_time_seconds, status,
                    affected_rows, validation_passed
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (version) DO UPDATE SET
                    applied_at = $12,
                    execution_time_seconds = $13,
                    status = $14,
                    affected_rows = $15,
                    validation_passed = $16,
                    updated_at = CURRENT_TIMESTAMP
            """, 
            metadata.version, metadata.name, metadata.description, metadata.author,
            metadata.safety_level.value, metadata.estimated_duration, metadata.requires_downtime,
            metadata.dependencies, metadata.affects_tables, metadata.rollback_strategy,
            json.dumps(asdict(metadata), default=str),
            execution.completed_at, execution.duration_seconds, execution.status.value,
            execution.affected_rows, execution.validation_passed)
            
        # Add to local history
        self.migration_history.append(execution)
        
    async def _record_migration_failure(self, execution: MigrationExecution, 
                                      metadata: MigrationMetadata):
        """Record failed migration in database"""
        async with self.connection_manager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO migration_versions (
                    version, name, description, author, safety_level,
                    estimated_duration, requires_downtime, dependencies,
                    affects_tables, rollback_strategy, metadata,
                    applied_at, execution_time_seconds, status,
                    error_message, affected_rows
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (version) DO UPDATE SET
                    applied_at = $12,
                    execution_time_seconds = $13,
                    status = $14,
                    error_message = $15,
                    affected_rows = $16,
                    updated_at = CURRENT_TIMESTAMP
            """, 
            metadata.version, metadata.name, metadata.description, metadata.author,
            metadata.safety_level.value, metadata.estimated_duration, metadata.requires_downtime,
            metadata.dependencies, metadata.affects_tables, metadata.rollback_strategy,
            json.dumps(asdict(metadata), default=str),
            execution.started_at, execution.duration_seconds, execution.status.value,
            execution.error_message, execution.affected_rows)
            
    async def _log_execution_step(self, conn: Connection, migration_version: str,
                                step: str, query: str, duration: float, 
                                status: str, error: str = None):
        """Log individual execution step"""
        await conn.execute("""
            INSERT INTO migration_execution_log (
                migration_version, execution_step, completed_at, 
                duration_seconds, status, query_executed, error_details
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, migration_version, step, datetime.utcnow(), duration, 
            status, query[:1000], json.dumps({"error": error}) if error else None)
        
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status and statistics"""
        pending_migrations = await self.get_pending_migrations()
        
        return {
            "total_migrations": len(self.migration_history),
            "completed_migrations": len([m for m in self.migration_history 
                                        if m.status == MigrationStatus.COMPLETED]),
            "failed_migrations": len([m for m in self.migration_history 
                                    if m.status == MigrationStatus.FAILED]),
            "rolled_back_migrations": len([m for m in self.migration_history 
                                         if m.rollback_executed]),
            "pending_migrations": len(pending_migrations),
            "currently_running": len(self.current_executions),
            "last_migration": max([m.completed_at for m in self.migration_history 
                                 if m.completed_at], default=None),
            "pending_migration_versions": [m.version for m in pending_migrations],
            "running_migrations": list(self.current_executions.keys())
        }