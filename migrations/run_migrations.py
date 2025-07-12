#!/usr/bin/env python3
"""
Database Migration Runner for Rental ML System
Created: 2025-07-12

This script manages database migrations with proper error handling,
rollback capabilities, and migration tracking.
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import asyncpg
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.data.config import DataConfig, DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationRunner:
    """Handles database migration execution and tracking"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.migrations_dir = Path(__file__).parent
        self.connection: Optional[asyncpg.Connection] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Establish database connection"""
        try:
            self.connection = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password
            )
            logger.info(f"Connected to database: {self.config.database}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")
    
    async def create_migrations_table(self):
        """Create migrations tracking table if it doesn't exist"""
        try:
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    migration_file VARCHAR(255) NOT NULL UNIQUE,
                    checksum VARCHAR(64) NOT NULL,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms INTEGER,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    applied_by VARCHAR(255) DEFAULT CURRENT_USER
                )
            """)
            
            # Create index for faster lookups
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_file 
                ON schema_migrations(migration_file)
            """)
            
            logger.info("Migration tracking table ensured")
        except Exception as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise
    
    def get_migration_files(self) -> List[Path]:
        """Get all migration files in order"""
        migration_files = []
        for file_path in sorted(self.migrations_dir.glob("*.sql")):
            if file_path.name.startswith(("001_", "002_", "003_", "004_", "005_", "006_", "007_", "008_", "009_")):
                migration_files.append(file_path)
        
        logger.info(f"Found {len(migration_files)} migration files")
        return migration_files
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of migration file"""
        import hashlib
        
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    
    async def get_applied_migrations(self) -> set:
        """Get set of already applied migrations"""
        try:
            rows = await self.connection.fetch(
                "SELECT migration_file FROM schema_migrations WHERE success = TRUE"
            )
            return {row['migration_file'] for row in rows}
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return set()
    
    async def execute_migration(self, file_path: Path) -> Tuple[bool, Optional[str], int]:
        """Execute a single migration file"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing migration: {file_path.name}")
            
            # Read migration content
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Execute migration in transaction
            async with self.connection.transaction():
                await self.connection.execute(sql_content)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(f"Migration {file_path.name} completed in {execution_time}ms")
            
            return True, None, execution_time
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = str(e)
            logger.error(f"Migration {file_path.name} failed after {execution_time}ms: {error_msg}")
            return False, error_msg, execution_time
    
    async def record_migration(self, file_path: Path, checksum: str, 
                             success: bool, error_message: Optional[str], 
                             execution_time: int):
        """Record migration execution in tracking table"""
        try:
            await self.connection.execute("""
                INSERT INTO schema_migrations 
                (migration_file, checksum, success, error_message, execution_time_ms)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (migration_file) 
                DO UPDATE SET
                    checksum = EXCLUDED.checksum,
                    applied_at = CURRENT_TIMESTAMP,
                    success = EXCLUDED.success,
                    error_message = EXCLUDED.error_message,
                    execution_time_ms = EXCLUDED.execution_time_ms
            """, file_path.name, checksum, success, error_message, execution_time)
            
        except Exception as e:
            logger.error(f"Failed to record migration: {e}")
            raise
    
    async def validate_migration_integrity(self, file_path: Path) -> bool:
        """Validate that migration hasn't been modified since last run"""
        try:
            current_checksum = self.calculate_checksum(file_path)
            
            row = await self.connection.fetchrow(
                "SELECT checksum, success FROM schema_migrations WHERE migration_file = $1",
                file_path.name
            )
            
            if row is None:
                return True  # New migration
            
            if row['checksum'] != current_checksum:
                logger.warning(f"Migration {file_path.name} has been modified since last execution")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate migration integrity: {e}")
            return False
    
    async def run_migrations(self, force: bool = False, dry_run: bool = False) -> bool:
        """Run all pending migrations"""
        try:
            await self.create_migrations_table()
            applied_migrations = await self.get_applied_migrations()
            migration_files = self.get_migration_files()
            
            if not migration_files:
                logger.info("No migration files found")
                return True
            
            pending_migrations = []
            for file_path in migration_files:
                if file_path.name not in applied_migrations:
                    pending_migrations.append(file_path)
                elif not force:
                    # Validate integrity of applied migrations
                    if not await self.validate_migration_integrity(file_path):
                        logger.error(f"Migration integrity check failed for {file_path.name}")
                        return False
            
            if not pending_migrations:
                logger.info("All migrations are up to date")
                return True
            
            logger.info(f"Found {len(pending_migrations)} pending migrations")
            
            if dry_run:
                logger.info("DRY RUN - Would execute the following migrations:")
                for file_path in pending_migrations:
                    logger.info(f"  - {file_path.name}")
                return True
            
            # Execute pending migrations
            successful_migrations = 0
            for file_path in pending_migrations:
                checksum = self.calculate_checksum(file_path)
                success, error_message, execution_time = await self.execute_migration(file_path)
                
                # Record migration result
                await self.record_migration(file_path, checksum, success, error_message, execution_time)
                
                if success:
                    successful_migrations += 1
                else:
                    logger.error(f"Migration failed: {file_path.name}")
                    return False
            
            logger.info(f"Successfully executed {successful_migrations} migrations")
            return True
            
        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            return False
    
    async def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a specific migration (if rollback script exists)"""
        try:
            rollback_file = self.migrations_dir / f"rollback_{migration_name}"
            
            if not rollback_file.exists():
                logger.error(f"No rollback script found for migration: {migration_name}")
                return False
            
            logger.info(f"Rolling back migration: {migration_name}")
            
            with open(rollback_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            async with self.connection.transaction():
                await self.connection.execute(sql_content)
            
            # Mark migration as not applied
            await self.connection.execute(
                "UPDATE schema_migrations SET success = FALSE WHERE migration_file = $1",
                migration_name
            )
            
            logger.info(f"Successfully rolled back migration: {migration_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def get_migration_status(self) -> List[dict]:
        """Get status of all migrations"""
        try:
            await self.create_migrations_table()
            
            migration_files = self.get_migration_files()
            applied_migrations = await self.connection.fetch(
                "SELECT * FROM schema_migrations ORDER BY applied_at"
            )
            
            applied_dict = {row['migration_file']: row for row in applied_migrations}
            
            status = []
            for file_path in migration_files:
                if file_path.name in applied_dict:
                    row = applied_dict[file_path.name]
                    status.append({
                        'file': file_path.name,
                        'status': 'applied' if row['success'] else 'failed',
                        'applied_at': row['applied_at'],
                        'execution_time_ms': row['execution_time_ms'],
                        'error': row['error_message']
                    })
                else:
                    status.append({
                        'file': file_path.name,
                        'status': 'pending',
                        'applied_at': None,
                        'execution_time_ms': None,
                        'error': None
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return []


async def main():
    """Main migration runner function"""
    parser = argparse.ArgumentParser(description='Database Migration Runner')
    parser.add_argument('--env', default='production', help='Environment (production, development)')
    parser.add_argument('--force', action='store_true', help='Force re-run all migrations')
    parser.add_argument('--dry-run', action='store_true', help='Show pending migrations without executing')
    parser.add_argument('--status', action='store_true', help='Show migration status')
    parser.add_argument('--rollback', help='Rollback specific migration')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.env == 'production':
        # Load production environment
        from dotenv import load_dotenv
        env_file = Path(__file__).parent.parent / '.env.production'
        if env_file.exists():
            load_dotenv(env_file)
        else:
            logger.warning("Production .env file not found, using environment variables")
    
    config = DataConfig().database
    
    try:
        async with MigrationRunner(config) as runner:
            if args.status:
                status = await runner.get_migration_status()
                print("\nMigration Status:")
                print("-" * 80)
                for migration in status:
                    status_symbol = "✓" if migration['status'] == 'applied' else "✗" if migration['status'] == 'failed' else "○"
                    print(f"{status_symbol} {migration['file']:30} {migration['status']:10} {migration['applied_at'] or 'N/A'}")
                
            elif args.rollback:
                success = await runner.rollback_migration(args.rollback)
                sys.exit(0 if success else 1)
                
            else:
                success = await runner.run_migrations(force=args.force, dry_run=args.dry_run)
                sys.exit(0 if success else 1)
    
    except Exception as e:
        logger.error(f"Migration runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())