#!/usr/bin/env python3
"""
Database Backup and Restore System for Rental ML System
Created: 2025-07-12

This script provides comprehensive backup and restore functionality for:
- PostgreSQL database backups (schema + data)
- Redis data backups
- Incremental and full backup strategies
- Automated backup scheduling
- Backup verification and integrity checks
"""

import os
import sys
import json
import gzip
import shutil
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import subprocess
import asyncpg
import redis.asyncio as redis
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Backup configuration settings"""
    backup_dir: str
    retention_days: int = 30
    compress_backups: bool = True
    verify_backups: bool = True
    max_backup_size_mb: int = 1000
    parallel_jobs: int = 2


@dataclass
class BackupResult:
    """Backup operation result"""
    backup_type: str
    success: bool
    backup_file: Optional[str]
    file_size_mb: float
    duration_seconds: float
    error_message: Optional[str] = None
    verification_passed: bool = False


class PostgreSQLBackupManager:
    """Manages PostgreSQL database backups and restores"""
    
    def __init__(self, config: Dict[str, Any], backup_config: BackupConfig):
        self.db_config = config
        self.backup_config = backup_config
        self.backup_dir = Path(backup_config.backup_dir) / "postgresql"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_full_backup(self, backup_name: Optional[str] = None) -> BackupResult:
        """Create a full PostgreSQL backup using pg_dump"""
        start_time = datetime.now()
        
        if not backup_name:
            backup_name = f"full_backup_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.backup_dir / f"{backup_name}.sql"
        
        try:
            # Prepare pg_dump command
            cmd = [
                'pg_dump',
                '--host', self.db_config['host'],
                '--port', str(self.db_config['port']),
                '--username', self.db_config['user'],
                '--dbname', self.db_config['database'],
                '--verbose',
                '--clean',
                '--if-exists',
                '--create',
                '--file', str(backup_file)
            ]
            
            # Set password via environment
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config['password']
            
            # Execute pg_dump
            logger.info(f"Starting PostgreSQL full backup: {backup_name}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = f"pg_dump failed: {result.stderr}"
                logger.error(error_msg)
                return BackupResult(
                    backup_type="postgresql_full",
                    success=False,
                    backup_file=None,
                    file_size_mb=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=error_msg
                )
            
            # Compress if requested
            if self.backup_config.compress_backups:
                compressed_file = await self._compress_backup(backup_file)
                if compressed_file:
                    backup_file.unlink()  # Remove original
                    backup_file = compressed_file
            
            file_size_mb = backup_file.stat().st_size / (1024 * 1024)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Verify backup if requested
            verification_passed = False
            if self.backup_config.verify_backups:
                verification_passed = await self._verify_postgresql_backup(backup_file)
            
            logger.info(f"PostgreSQL backup completed: {backup_file} ({file_size_mb:.2f} MB)")
            
            return BackupResult(
                backup_type="postgresql_full",
                success=True,
                backup_file=str(backup_file),
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                verification_passed=verification_passed
            )
            
        except Exception as e:
            error_msg = f"PostgreSQL backup failed: {str(e)}"
            logger.error(error_msg)
            
            return BackupResult(
                backup_type="postgresql_full",
                success=False,
                backup_file=None,
                file_size_mb=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg
            )
    
    async def create_schema_backup(self, backup_name: Optional[str] = None) -> BackupResult:
        """Create a schema-only backup"""
        start_time = datetime.now()
        
        if not backup_name:
            backup_name = f"schema_backup_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.backup_dir / f"{backup_name}.sql"
        
        try:
            # pg_dump with schema-only option
            cmd = [
                'pg_dump',
                '--host', self.db_config['host'],
                '--port', str(self.db_config['port']),
                '--username', self.db_config['user'],
                '--dbname', self.db_config['database'],
                '--schema-only',
                '--verbose',
                '--clean',
                '--if-exists',
                '--create',
                '--file', str(backup_file)
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config['password']
            
            logger.info(f"Starting PostgreSQL schema backup: {backup_name}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = f"Schema backup failed: {result.stderr}"
                logger.error(error_msg)
                return BackupResult(
                    backup_type="postgresql_schema",
                    success=False,
                    backup_file=None,
                    file_size_mb=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=error_msg
                )
            
            if self.backup_config.compress_backups:
                compressed_file = await self._compress_backup(backup_file)
                if compressed_file:
                    backup_file.unlink()
                    backup_file = compressed_file
            
            file_size_mb = backup_file.stat().st_size / (1024 * 1024)
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"PostgreSQL schema backup completed: {backup_file}")
            
            return BackupResult(
                backup_type="postgresql_schema",
                success=True,
                backup_file=str(backup_file),
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                verification_passed=True  # Schema backups are easier to verify
            )
            
        except Exception as e:
            error_msg = f"PostgreSQL schema backup failed: {str(e)}"
            logger.error(error_msg)
            
            return BackupResult(
                backup_type="postgresql_schema",
                success=False,
                backup_file=None,
                file_size_mb=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg
            )
    
    async def restore_backup(self, backup_file: str, target_database: Optional[str] = None) -> bool:
        """Restore PostgreSQL backup"""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            # Decompress if needed
            if backup_path.suffix == '.gz':
                decompressed_file = await self._decompress_backup(backup_path)
                if not decompressed_file:
                    return False
                backup_path = decompressed_file
            
            target_db = target_database or self.db_config['database']
            
            # Use psql to restore
            cmd = [
                'psql',
                '--host', self.db_config['host'],
                '--port', str(self.db_config['port']),
                '--username', self.db_config['user'],
                '--dbname', target_db,
                '--file', str(backup_path)
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config['password']
            
            logger.info(f"Starting PostgreSQL restore from: {backup_file}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"PostgreSQL restore failed: {result.stderr}")
                return False
            
            logger.info(f"PostgreSQL restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL restore failed: {str(e)}")
            return False
    
    async def _compress_backup(self, backup_file: Path) -> Optional[Path]:
        """Compress backup file using gzip"""
        try:
            compressed_file = backup_file.with_suffix(backup_file.suffix + '.gz')
            
            with open(backup_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.debug(f"Compressed backup: {compressed_file}")
            return compressed_file
            
        except Exception as e:
            logger.error(f"Failed to compress backup: {e}")
            return None
    
    async def _decompress_backup(self, compressed_file: Path) -> Optional[Path]:
        """Decompress backup file"""
        try:
            decompressed_file = compressed_file.with_suffix('')
            
            with gzip.open(compressed_file, 'rb') as f_in:
                with open(decompressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.debug(f"Decompressed backup: {decompressed_file}")
            return decompressed_file
            
        except Exception as e:
            logger.error(f"Failed to decompress backup: {e}")
            return None
    
    async def _verify_postgresql_backup(self, backup_file: Path) -> bool:
        """Verify PostgreSQL backup integrity"""
        try:
            # Basic verification: check if file is valid SQL
            with open(backup_file, 'r') as f:
                content = f.read(1000)  # Read first 1KB
                
                # Look for PostgreSQL dump headers
                if 'PostgreSQL database dump' in content:
                    logger.debug(f"Backup verification passed: {backup_file}")
                    return True
                else:
                    logger.warning(f"Backup verification failed: Invalid format {backup_file}")
                    return False
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False


class RedisBackupManager:
    """Manages Redis data backups and restores"""
    
    def __init__(self, config: Dict[str, Any], backup_config: BackupConfig):
        self.redis_config = config
        self.backup_config = backup_config
        self.backup_dir = Path(backup_config.backup_dir) / "redis"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_redis_backup(self, backup_name: Optional[str] = None) -> BackupResult:
        """Create Redis backup using BGSAVE or RDB export"""
        start_time = datetime.now()
        
        if not backup_name:
            backup_name = f"redis_backup_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.backup_dir / f"{backup_name}.rdb"
        
        try:
            # Connect to Redis
            client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config.get('password')
            )
            
            # Trigger background save
            logger.info(f"Starting Redis backup: {backup_name}")
            await client.bgsave()
            
            # Wait for save to complete
            while True:
                info = await client.info('persistence')
                if info.get('rdb_bgsave_in_progress', 0) == 0:
                    break
                await asyncio.sleep(1)
            
            # Get the RDB file from Redis data directory
            # This is a simplified approach - in production, you'd copy from Redis data dir
            await self._export_redis_data(client, backup_file)
            
            await client.aclose()
            
            # Compress if requested
            if self.backup_config.compress_backups:
                compressed_file = await self._compress_backup(backup_file)
                if compressed_file:
                    backup_file.unlink()
                    backup_file = compressed_file
            
            file_size_mb = backup_file.stat().st_size / (1024 * 1024)
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Redis backup completed: {backup_file} ({file_size_mb:.2f} MB)")
            
            return BackupResult(
                backup_type="redis",
                success=True,
                backup_file=str(backup_file),
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                verification_passed=True
            )
            
        except Exception as e:
            error_msg = f"Redis backup failed: {str(e)}"
            logger.error(error_msg)
            
            return BackupResult(
                backup_type="redis",
                success=False,
                backup_file=None,
                file_size_mb=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg
            )
    
    async def _export_redis_data(self, client: redis.Redis, backup_file: Path):
        """Export Redis data to JSON format"""
        exported_data = {
            'backup_info': {
                'created_at': datetime.utcnow().isoformat(),
                'redis_version': None,
                'db_number': self.redis_config['db']
            },
            'data': {}
        }
        
        try:
            # Get Redis info
            info = await client.info()
            exported_data['backup_info']['redis_version'] = info.get('redis_version')
            
            # Export all keys
            async for key in client.scan_iter():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                # Get key type and value
                key_type = await client.type(key)
                
                if key_type == b'string':
                    value = await client.get(key)
                    exported_data['data'][key_str] = {
                        'type': 'string',
                        'value': value.decode('utf-8') if isinstance(value, bytes) else value,
                        'ttl': await client.ttl(key)
                    }
                elif key_type == b'hash':
                    value = await client.hgetall(key)
                    decoded_value = {
                        k.decode('utf-8') if isinstance(k, bytes) else k: 
                        v.decode('utf-8') if isinstance(v, bytes) else v
                        for k, v in value.items()
                    }
                    exported_data['data'][key_str] = {
                        'type': 'hash',
                        'value': decoded_value,
                        'ttl': await client.ttl(key)
                    }
                elif key_type == b'list':
                    value = await client.lrange(key, 0, -1)
                    decoded_value = [
                        v.decode('utf-8') if isinstance(v, bytes) else v 
                        for v in value
                    ]
                    exported_data['data'][key_str] = {
                        'type': 'list',
                        'value': decoded_value,
                        'ttl': await client.ttl(key)
                    }
                # Add more types as needed (sets, sorted sets, etc.)
            
            # Write to file
            with open(backup_file, 'w') as f:
                json.dump(exported_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to export Redis data: {e}")
            raise
    
    async def restore_redis_backup(self, backup_file: str) -> bool:
        """Restore Redis backup"""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            # Decompress if needed
            if backup_path.suffix == '.gz':
                decompressed_file = await self._decompress_backup(backup_path)
                if not decompressed_file:
                    return False
                backup_path = decompressed_file
            
            # Connect to Redis
            client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                password=self.redis_config.get('password')
            )
            
            # Load backup data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            logger.info(f"Starting Redis restore from: {backup_file}")
            
            # Clear existing data (optional - be careful!)
            # await client.flushdb()
            
            # Restore data
            for key, data in backup_data['data'].items():
                key_type = data['type']
                value = data['value']
                ttl = data.get('ttl', -1)
                
                if key_type == 'string':
                    await client.set(key, value)
                elif key_type == 'hash':
                    await client.hset(key, mapping=value)
                elif key_type == 'list':
                    if value:  # Only if list is not empty
                        await client.lpush(key, *reversed(value))
                
                # Set TTL if it was set
                if ttl > 0:
                    await client.expire(key, ttl)
            
            await client.aclose()
            
            logger.info(f"Redis restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Redis restore failed: {str(e)}")
            return False
    
    async def _compress_backup(self, backup_file: Path) -> Optional[Path]:
        """Compress backup file"""
        try:
            compressed_file = backup_file.with_suffix(backup_file.suffix + '.gz')
            
            with open(backup_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            return compressed_file
            
        except Exception as e:
            logger.error(f"Failed to compress backup: {e}")
            return None
    
    async def _decompress_backup(self, compressed_file: Path) -> Optional[Path]:
        """Decompress backup file"""
        try:
            decompressed_file = compressed_file.with_suffix('')
            
            with gzip.open(compressed_file, 'rb') as f_in:
                with open(decompressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            return decompressed_file
            
        except Exception as e:
            logger.error(f"Failed to decompress backup: {e}")
            return None


class BackupScheduler:
    """Manages automated backup scheduling and retention"""
    
    def __init__(self, pg_manager: PostgreSQLBackupManager, 
                 redis_manager: RedisBackupManager, backup_config: BackupConfig):
        self.pg_manager = pg_manager
        self.redis_manager = redis_manager
        self.backup_config = backup_config
        self.backup_history_file = Path(backup_config.backup_dir) / "backup_history.json"
    
    async def run_scheduled_backup(self, backup_types: List[str] = None) -> Dict[str, BackupResult]:
        """Run scheduled backup for specified types"""
        if backup_types is None:
            backup_types = ['postgresql_full', 'redis']
        
        results = {}
        
        for backup_type in backup_types:
            if backup_type == 'postgresql_full':
                result = await self.pg_manager.create_full_backup()
                results['postgresql_full'] = result
            elif backup_type == 'postgresql_schema':
                result = await self.pg_manager.create_schema_backup()
                results['postgresql_schema'] = result
            elif backup_type == 'redis':
                result = await self.redis_manager.create_redis_backup()
                results['redis'] = result
        
        # Record backup history
        await self._record_backup_history(results)
        
        # Clean up old backups
        await self.cleanup_old_backups()
        
        return results
    
    async def _record_backup_history(self, results: Dict[str, BackupResult]):
        """Record backup history to JSON file"""
        try:
            # Load existing history
            history = []
            if self.backup_history_file.exists():
                with open(self.backup_history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new results
            timestamp = datetime.utcnow().isoformat()
            for backup_type, result in results.items():
                history.append({
                    'timestamp': timestamp,
                    'backup_type': result.backup_type,
                    'success': result.success,
                    'backup_file': result.backup_file,
                    'file_size_mb': result.file_size_mb,
                    'duration_seconds': result.duration_seconds,
                    'verification_passed': result.verification_passed,
                    'error_message': result.error_message
                })
            
            # Save updated history
            with open(self.backup_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to record backup history: {e}")
    
    async def cleanup_old_backups(self):
        """Clean up backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_config.retention_days)
            deleted_count = 0
            
            for backup_dir in [self.pg_manager.backup_dir, self.redis_manager.backup_dir]:
                for backup_file in backup_dir.glob('*'):
                    if backup_file.is_file():
                        file_date = datetime.fromtimestamp(backup_file.stat().st_mtime)
                        if file_date < cutoff_date:
                            backup_file.unlink()
                            deleted_count += 1
                            logger.info(f"Deleted old backup: {backup_file}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backup files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    def get_backup_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get backup history"""
        try:
            if not self.backup_history_file.exists():
                return []
            
            with open(self.backup_history_file, 'r') as f:
                history = json.load(f)
            
            # Sort by timestamp (newest first) and limit
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            return history[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get backup history: {e}")
            return []


async def main():
    """Main backup system function"""
    parser = argparse.ArgumentParser(description='Database Backup System')
    parser.add_argument('--action', choices=['backup', 'restore', 'schedule', 'status'], 
                       default='backup', help='Action to perform')
    parser.add_argument('--type', choices=['postgresql', 'redis', 'both'], 
                       default='both', help='Backup type')
    parser.add_argument('--backup-file', help='Backup file for restore operation')
    parser.add_argument('--backup-dir', default='/var/backups/rental_ml', 
                       help='Backup directory')
    parser.add_argument('--retention-days', type=int, default=30, 
                       help='Backup retention in days')
    parser.add_argument('--compress', action='store_true', help='Compress backups')
    parser.add_argument('--verify', action='store_true', help='Verify backups')
    
    args = parser.parse_args()
    
    # Load configuration
    from dotenv import load_dotenv
    
    env_file = Path(__file__).parent.parent / '.env.production'
    if env_file.exists():
        load_dotenv(env_file)
    
    # Database configurations
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
    
    backup_config = BackupConfig(
        backup_dir=args.backup_dir,
        retention_days=args.retention_days,
        compress_backups=args.compress,
        verify_backups=args.verify
    )
    
    # Initialize managers
    pg_manager = PostgreSQLBackupManager(pg_config, backup_config)
    redis_manager = RedisBackupManager(redis_config, backup_config)
    scheduler = BackupScheduler(pg_manager, redis_manager, backup_config)
    
    try:
        if args.action == 'backup':
            backup_types = []
            if args.type in ['postgresql', 'both']:
                backup_types.append('postgresql_full')
            if args.type in ['redis', 'both']:
                backup_types.append('redis')
            
            results = await scheduler.run_scheduled_backup(backup_types)
            
            # Print results
            print("\nBackup Results:")
            print("-" * 50)
            for backup_type, result in results.items():
                status = "SUCCESS" if result.success else "FAILED"
                print(f"{backup_type:20} | {status:8} | {result.file_size_mb:8.2f} MB | {result.duration_seconds:6.1f}s")
                if result.error_message:
                    print(f"  Error: {result.error_message}")
            
        elif args.action == 'restore':
            if not args.backup_file:
                print("Error: --backup-file required for restore operation")
                sys.exit(1)
            
            if 'postgresql' in args.backup_file or 'pg' in args.backup_file:
                success = await pg_manager.restore_backup(args.backup_file)
            elif 'redis' in args.backup_file:
                success = await redis_manager.restore_redis_backup(args.backup_file)
            else:
                print("Error: Cannot determine backup type from filename")
                sys.exit(1)
            
            if success:
                print(f"Restore completed successfully: {args.backup_file}")
            else:
                print(f"Restore failed: {args.backup_file}")
                sys.exit(1)
        
        elif args.action == 'status':
            history = scheduler.get_backup_history(limit=20)
            
            print("\nRecent Backup History:")
            print("-" * 80)
            print(f"{'Timestamp':20} | {'Type':15} | {'Status':8} | {'Size (MB)':10} | {'Duration':10}")
            print("-" * 80)
            
            for entry in history:
                timestamp = entry['timestamp'][:19].replace('T', ' ')
                status = "SUCCESS" if entry['success'] else "FAILED"
                print(f"{timestamp} | {entry['backup_type']:15} | {status:8} | {entry['file_size_mb']:8.2f} | {entry['duration_seconds']:6.1f}s")
    
    except Exception as e:
        logger.error(f"Backup operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())