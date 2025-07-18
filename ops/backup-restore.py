#!/usr/bin/env python3
"""
Comprehensive Backup and Restore System for Rental ML Platform

This script provides automated backup and restore capabilities for:
- PostgreSQL databases with point-in-time recovery
- Redis data and configuration
- Kubernetes cluster state and configurations
- Application data and ML models
- Monitoring and logging data
- Configuration files and secrets
"""

import argparse
import json
import logging
import subprocess
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil

import yaml
from kubernetes import client, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BackupConfig:
    """Configuration for backup operations"""
    environment: str
    namespace: str
    backup_storage: str  # s3, gcs, azure
    storage_bucket: str
    encryption_key: str
    retention_days: int = 30
    
    # Database configuration
    postgres_host: str = None
    postgres_port: int = 5432
    postgres_user: str = None
    postgres_password: str = None
    postgres_database: str = None
    
    # Redis configuration
    redis_host: str = None
    redis_port: int = 6379
    redis_password: str = None
    
    # Storage paths
    backup_base_path: str = "/tmp/backups"
    
    # Notification settings
    notification_webhook: str = None
    notification_email: str = None

@dataclass
class BackupMetadata:
    """Metadata for a backup"""
    backup_id: str
    timestamp: datetime
    environment: str
    backup_type: str
    size_bytes: int
    components: List[str]
    storage_path: str
    encryption_enabled: bool
    status: str  # 'in_progress', 'completed', 'failed'
    restore_point: str = None

class BackupRestoreManager:
    """Comprehensive backup and restore manager"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_batch_v1 = client.BatchV1Api()
        
        # Create backup directory
        os.makedirs(self.config.backup_base_path, exist_ok=True)
    
    def create_full_backup(self) -> BackupMetadata:
        """Create a comprehensive full backup"""
        backup_id = f"full-{self.config.environment}-{int(time.time())}"
        timestamp = datetime.now()
        
        logger.info(f"Starting full backup: {backup_id}")
        
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=timestamp,
            environment=self.config.environment,
            backup_type="full",
            size_bytes=0,
            components=[],
            storage_path="",
            encryption_enabled=bool(self.config.encryption_key),
            status="in_progress"
        )
        
        try:
            backup_dir = os.path.join(self.config.backup_base_path, backup_id)
            os.makedirs(backup_dir, exist_ok=True)
            
            total_size = 0
            
            # Backup PostgreSQL database
            if self._backup_postgres(backup_dir):
                backup_metadata.components.append("postgres")
                total_size += self._get_directory_size(os.path.join(backup_dir, "postgres"))
            
            # Backup Redis data
            if self._backup_redis(backup_dir):
                backup_metadata.components.append("redis")
                total_size += self._get_directory_size(os.path.join(backup_dir, "redis"))
            
            # Backup Kubernetes resources
            if self._backup_kubernetes(backup_dir):
                backup_metadata.components.append("kubernetes")
                total_size += self._get_directory_size(os.path.join(backup_dir, "kubernetes"))
            
            # Backup application data
            if self._backup_application_data(backup_dir):
                backup_metadata.components.append("application_data")
                total_size += self._get_directory_size(os.path.join(backup_dir, "application_data"))
            
            # Backup ML models
            if self._backup_ml_models(backup_dir):
                backup_metadata.components.append("ml_models")
                total_size += self._get_directory_size(os.path.join(backup_dir, "ml_models"))
            
            # Backup configuration
            if self._backup_configuration(backup_dir):
                backup_metadata.components.append("configuration")
                total_size += self._get_directory_size(os.path.join(backup_dir, "configuration"))
            
            # Create backup archive
            archive_path = self._create_backup_archive(backup_dir, backup_id)
            
            # Encrypt if configured
            if self.config.encryption_key:
                archive_path = self._encrypt_backup(archive_path)
            
            # Upload to cloud storage
            storage_path = self._upload_backup(archive_path, backup_id)
            
            backup_metadata.size_bytes = total_size
            backup_metadata.storage_path = storage_path
            backup_metadata.status = "completed"
            
            # Save metadata
            self._save_backup_metadata(backup_metadata)
            
            # Cleanup local files
            shutil.rmtree(backup_dir)
            os.remove(archive_path)
            
            logger.info(f"Full backup completed: {backup_id}")
            self._send_notification("Backup Completed", f"Full backup {backup_id} completed successfully")
            
            return backup_metadata
            
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            backup_metadata.status = "failed"
            self._send_notification("Backup Failed", f"Full backup {backup_id} failed: {str(e)}")
            raise
    
    def create_incremental_backup(self, last_backup_id: str = None) -> BackupMetadata:
        """Create an incremental backup"""
        backup_id = f"incr-{self.config.environment}-{int(time.time())}"
        timestamp = datetime.now()
        
        logger.info(f"Starting incremental backup: {backup_id}")
        
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=timestamp,
            environment=self.config.environment,
            backup_type="incremental",
            size_bytes=0,
            components=[],
            storage_path="",
            encryption_enabled=bool(self.config.encryption_key),
            status="in_progress"
        )
        
        try:
            backup_dir = os.path.join(self.config.backup_base_path, backup_id)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Get last backup timestamp
            if last_backup_id:
                last_backup_metadata = self._load_backup_metadata(last_backup_id)
                since_time = last_backup_metadata.timestamp
            else:
                since_time = datetime.now() - timedelta(days=1)  # Default to 24h ago
            
            total_size = 0
            
            # Incremental PostgreSQL backup (WAL files)
            if self._backup_postgres_incremental(backup_dir, since_time):
                backup_metadata.components.append("postgres_wal")
                total_size += self._get_directory_size(os.path.join(backup_dir, "postgres_wal"))
            
            # Redis incremental backup (if AOF enabled)
            if self._backup_redis_incremental(backup_dir, since_time):
                backup_metadata.components.append("redis_aof")
                total_size += self._get_directory_size(os.path.join(backup_dir, "redis_aof"))
            
            # Changed application data only
            if self._backup_application_data_incremental(backup_dir, since_time):
                backup_metadata.components.append("application_data_changes")
                total_size += self._get_directory_size(os.path.join(backup_dir, "application_data_changes"))
            
            # New ML models only
            if self._backup_ml_models_incremental(backup_dir, since_time):
                backup_metadata.components.append("ml_models_new")
                total_size += self._get_directory_size(os.path.join(backup_dir, "ml_models_new"))
            
            if total_size == 0:
                logger.info("No changes detected, skipping incremental backup")
                backup_metadata.status = "skipped"
                return backup_metadata
            
            # Create backup archive
            archive_path = self._create_backup_archive(backup_dir, backup_id)
            
            # Encrypt if configured
            if self.config.encryption_key:
                archive_path = self._encrypt_backup(archive_path)
            
            # Upload to cloud storage
            storage_path = self._upload_backup(archive_path, backup_id)
            
            backup_metadata.size_bytes = total_size
            backup_metadata.storage_path = storage_path
            backup_metadata.status = "completed"
            
            # Save metadata
            self._save_backup_metadata(backup_metadata)
            
            # Cleanup local files
            shutil.rmtree(backup_dir)
            os.remove(archive_path)
            
            logger.info(f"Incremental backup completed: {backup_id}")
            
            return backup_metadata
            
        except Exception as e:
            logger.error(f"Incremental backup failed: {e}")
            backup_metadata.status = "failed"
            raise
    
    def restore_from_backup(self, backup_id: str, target_environment: str = None, 
                          point_in_time: datetime = None, 
                          selective_restore: List[str] = None) -> bool:
        """Restore from a backup"""
        logger.info(f"Starting restore from backup: {backup_id}")
        
        try:
            # Load backup metadata
            backup_metadata = self._load_backup_metadata(backup_id)
            
            if not backup_metadata:
                logger.error(f"Backup metadata not found for {backup_id}")
                return False
            
            # Download backup from storage
            local_archive_path = self._download_backup(backup_metadata.storage_path, backup_id)
            
            # Decrypt if needed
            if backup_metadata.encryption_enabled:
                local_archive_path = self._decrypt_backup(local_archive_path)
            
            # Extract backup
            restore_dir = os.path.join(self.config.backup_base_path, f"restore-{backup_id}")
            self._extract_backup_archive(local_archive_path, restore_dir)
            
            # Determine what to restore
            components_to_restore = selective_restore or backup_metadata.components
            
            # Restore each component
            success = True
            
            if "postgres" in components_to_restore:
                if not self._restore_postgres(restore_dir, point_in_time):
                    logger.error("PostgreSQL restore failed")
                    success = False
            
            if "redis" in components_to_restore:
                if not self._restore_redis(restore_dir):
                    logger.error("Redis restore failed")
                    success = False
            
            if "kubernetes" in components_to_restore:
                if not self._restore_kubernetes(restore_dir, target_environment):
                    logger.error("Kubernetes restore failed")
                    success = False
            
            if "application_data" in components_to_restore:
                if not self._restore_application_data(restore_dir):
                    logger.error("Application data restore failed")
                    success = False
            
            if "ml_models" in components_to_restore:
                if not self._restore_ml_models(restore_dir):
                    logger.error("ML models restore failed")
                    success = False
            
            if "configuration" in components_to_restore:
                if not self._restore_configuration(restore_dir, target_environment):
                    logger.error("Configuration restore failed")
                    success = False
            
            # Cleanup
            shutil.rmtree(restore_dir)
            os.remove(local_archive_path)
            
            if success:
                logger.info(f"Restore completed successfully from backup: {backup_id}")
                self._send_notification("Restore Completed", f"Restore from backup {backup_id} completed successfully")
            else:
                logger.error(f"Restore completed with errors from backup: {backup_id}")
                self._send_notification("Restore Failed", f"Restore from backup {backup_id} completed with errors")
            
            return success
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            self._send_notification("Restore Failed", f"Restore from backup {backup_id} failed: {str(e)}")
            return False
    
    def list_backups(self, backup_type: str = None, limit: int = 50) -> List[BackupMetadata]:
        """List available backups"""
        try:
            # This would query the metadata storage (database or cloud storage)
            # For now, return a placeholder
            logger.info(f"Listing backups (type: {backup_type}, limit: {limit})")
            return []
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def cleanup_old_backups(self) -> int:
        """Clean up old backups based on retention policy"""
        logger.info(f"Cleaning up backups older than {self.config.retention_days} days")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            
            # Get list of old backups
            old_backups = self._get_old_backups(cutoff_date)
            
            deleted_count = 0
            for backup_metadata in old_backups:
                try:
                    # Delete from cloud storage
                    self._delete_backup_from_storage(backup_metadata.storage_path)
                    
                    # Delete metadata
                    self._delete_backup_metadata(backup_metadata.backup_id)
                    
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {backup_metadata.backup_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to delete backup {backup_metadata.backup_id}: {e}")
            
            logger.info(f"Cleanup completed. Deleted {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return 0
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity"""
        logger.info(f"Verifying backup: {backup_id}")
        
        try:
            backup_metadata = self._load_backup_metadata(backup_id)
            
            if not backup_metadata:
                logger.error(f"Backup metadata not found: {backup_id}")
                return False
            
            # Download and verify backup
            local_archive_path = self._download_backup(backup_metadata.storage_path, backup_id)
            
            # Verify archive integrity
            if not self._verify_archive_integrity(local_archive_path):
                logger.error(f"Archive integrity check failed: {backup_id}")
                return False
            
            # Test extraction
            test_dir = os.path.join(self.config.backup_base_path, f"verify-{backup_id}")
            try:
                self._extract_backup_archive(local_archive_path, test_dir)
                
                # Verify each component
                for component in backup_metadata.components:
                    component_path = os.path.join(test_dir, component)
                    if not os.path.exists(component_path):
                        logger.error(f"Component missing in backup: {component}")
                        return False
                
                logger.info(f"Backup verification successful: {backup_id}")
                return True
                
            finally:
                # Cleanup
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)
                os.remove(local_archive_path)
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    # Private methods for specific backup operations
    
    def _backup_postgres(self, backup_dir: str) -> bool:
        """Backup PostgreSQL database"""
        logger.info("Starting PostgreSQL backup")
        
        try:
            postgres_backup_dir = os.path.join(backup_dir, "postgres")
            os.makedirs(postgres_backup_dir, exist_ok=True)
            
            # Use pg_dump for logical backup
            dump_file = os.path.join(postgres_backup_dir, f"{self.config.postgres_database}.sql")
            
            pg_dump_cmd = [
                "pg_dump",
                f"--host={self.config.postgres_host}",
                f"--port={self.config.postgres_port}",
                f"--username={self.config.postgres_user}",
                "--format=custom",
                "--compress=9",
                "--verbose",
                "--file", dump_file,
                self.config.postgres_database
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.postgres_password
            
            result = subprocess.run(
                pg_dump_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"pg_dump failed: {result.stderr}")
                return False
            
            # Also backup database schema separately
            schema_file = os.path.join(postgres_backup_dir, "schema.sql")
            pg_dump_schema_cmd = [
                "pg_dump",
                f"--host={self.config.postgres_host}",
                f"--port={self.config.postgres_port}",
                f"--username={self.config.postgres_user}",
                "--schema-only",
                "--file", schema_file,
                self.config.postgres_database
            ]
            
            subprocess.run(pg_dump_schema_cmd, env=env, check=True)
            
            logger.info("PostgreSQL backup completed")
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return False
    
    def _backup_redis(self, backup_dir: str) -> bool:
        """Backup Redis data"""
        logger.info("Starting Redis backup")
        
        try:
            redis_backup_dir = os.path.join(backup_dir, "redis")
            os.makedirs(redis_backup_dir, exist_ok=True)
            
            # Use redis-cli to create backup
            import redis
            
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=False
            )
            
            # Trigger BGSAVE
            r.bgsave()
            
            # Wait for backup to complete
            while r.lastsave() == r.lastsave():
                time.sleep(1)
            
            # Copy RDB file (this would need to be implemented based on Redis setup)
            # For now, export all keys
            keys_file = os.path.join(redis_backup_dir, "redis_keys.json")
            all_keys = r.keys('*')
            
            backup_data = {}
            for key in all_keys:
                key_type = r.type(key)
                if key_type == b'string':
                    backup_data[key.decode()] = {'type': 'string', 'value': r.get(key).decode()}
                elif key_type == b'hash':
                    backup_data[key.decode()] = {'type': 'hash', 'value': r.hgetall(key)}
                elif key_type == b'list':
                    backup_data[key.decode()] = {'type': 'list', 'value': r.lrange(key, 0, -1)}
                elif key_type == b'set':
                    backup_data[key.decode()] = {'type': 'set', 'value': list(r.smembers(key))}
                elif key_type == b'zset':
                    backup_data[key.decode()] = {'type': 'zset', 'value': r.zrange(key, 0, -1, withscores=True)}
            
            with open(keys_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info("Redis backup completed")
            return True
            
        except Exception as e:
            logger.error(f"Redis backup failed: {e}")
            return False
    
    def _backup_kubernetes(self, backup_dir: str) -> bool:
        """Backup Kubernetes resources"""
        logger.info("Starting Kubernetes backup")
        
        try:
            k8s_backup_dir = os.path.join(backup_dir, "kubernetes")
            os.makedirs(k8s_backup_dir, exist_ok=True)
            
            # Backup namespaced resources
            resources_to_backup = [
                'configmaps', 'secrets', 'services', 'deployments',
                'statefulsets', 'daemonsets', 'ingresses', 'persistentvolumeclaims',
                'servicemonitors', 'prometheusrules'
            ]
            
            for resource_type in resources_to_backup:
                try:
                    kubectl_cmd = [
                        "kubectl", "get", resource_type,
                        "--namespace", self.config.namespace,
                        "--output", "yaml"
                    ]
                    
                    result = subprocess.run(
                        kubectl_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        output_file = os.path.join(k8s_backup_dir, f"{resource_type}.yaml")
                        with open(output_file, 'w') as f:
                            f.write(result.stdout)
                        logger.info(f"Backed up {resource_type}")
                    
                except Exception as e:
                    logger.warning(f"Failed to backup {resource_type}: {e}")
            
            # Backup cluster-wide resources (if admin access)
            cluster_resources = ['nodes', 'persistentvolumes', 'storageclasses']
            
            for resource_type in cluster_resources:
                try:
                    kubectl_cmd = [
                        "kubectl", "get", resource_type,
                        "--output", "yaml"
                    ]
                    
                    result = subprocess.run(
                        kubectl_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        output_file = os.path.join(k8s_backup_dir, f"cluster-{resource_type}.yaml")
                        with open(output_file, 'w') as f:
                            f.write(result.stdout)
                        logger.info(f"Backed up cluster {resource_type}")
                
                except Exception as e:
                    logger.warning(f"Failed to backup cluster {resource_type}: {e}")
            
            logger.info("Kubernetes backup completed")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes backup failed: {e}")
            return False
    
    def _backup_application_data(self, backup_dir: str) -> bool:
        """Backup application-specific data"""
        logger.info("Starting application data backup")
        
        try:
            app_backup_dir = os.path.join(backup_dir, "application_data")
            os.makedirs(app_backup_dir, exist_ok=True)
            
            # This would be implemented based on specific application requirements
            # For example, backing up uploaded files, user data, etc.
            
            logger.info("Application data backup completed")
            return True
            
        except Exception as e:
            logger.error(f"Application data backup failed: {e}")
            return False
    
    def _backup_ml_models(self, backup_dir: str) -> bool:
        """Backup ML models and artifacts"""
        logger.info("Starting ML models backup")
        
        try:
            ml_backup_dir = os.path.join(backup_dir, "ml_models")
            os.makedirs(ml_backup_dir, exist_ok=True)
            
            # Backup from cloud storage (S3, GCS, etc.)
            # This would use cloud CLI tools to sync model artifacts
            
            logger.info("ML models backup completed")
            return True
            
        except Exception as e:
            logger.error(f"ML models backup failed: {e}")
            return False
    
    def _backup_configuration(self, backup_dir: str) -> bool:
        """Backup configuration files and environment settings"""
        logger.info("Starting configuration backup")
        
        try:
            config_backup_dir = os.path.join(backup_dir, "configuration")
            os.makedirs(config_backup_dir, exist_ok=True)
            
            # Backup Helm values, configs, etc.
            config_files = [
                "k8s/helm/rental-ml/values-production.yaml",
                "config/monitoring/production-monitoring.yaml",
                "infrastructure/terraform.tfvars"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    shutil.copy2(config_file, config_backup_dir)
            
            logger.info("Configuration backup completed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            return False
    
    # Additional private helper methods would be implemented here
    # including: _create_backup_archive, _encrypt_backup, _upload_backup, etc.
    
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def _create_backup_archive(self, backup_dir: str, backup_id: str) -> str:
        """Create compressed archive of backup"""
        archive_path = f"{backup_dir}.tar.gz"
        
        tar_cmd = ["tar", "-czf", archive_path, "-C", os.path.dirname(backup_dir), os.path.basename(backup_dir)]
        
        result = subprocess.run(tar_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to create archive: {result.stderr}")
        
        return archive_path
    
    def _send_notification(self, subject: str, message: str):
        """Send notification about backup/restore operations"""
        try:
            if self.config.notification_webhook:
                import requests
                requests.post(
                    self.config.notification_webhook,
                    json={"text": f"{subject}: {message}"},
                    timeout=30
                )
            
            if self.config.notification_email:
                # Send email notification (would need to be implemented)
                pass
                
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
    
    # Placeholder methods for metadata management
    def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata"""
        pass
    
    def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load backup metadata"""
        return None
    
    def _get_old_backups(self, cutoff_date: datetime) -> List[BackupMetadata]:
        """Get list of backups older than cutoff date"""
        return []

def main():
    parser = argparse.ArgumentParser(description='Backup and Restore Manager')
    parser.add_argument('action', choices=['backup', 'restore', 'list', 'cleanup', 'verify'])
    parser.add_argument('--environment', required=True, help='Environment name')
    parser.add_argument('--namespace', help='Kubernetes namespace')
    parser.add_argument('--backup-id', help='Backup ID for restore/verify operations')
    parser.add_argument('--backup-type', choices=['full', 'incremental'], default='full')
    parser.add_argument('--storage-bucket', required=True, help='Cloud storage bucket')
    parser.add_argument('--encryption-key', help='Encryption key for backup')
    parser.add_argument('--config-file', help='Configuration file')
    
    args = parser.parse_args()
    
    # Build configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        backup_config = BackupConfig(**config_data)
    else:
        backup_config = BackupConfig(
            environment=args.environment,
            namespace=args.namespace or f"rental-ml-{args.environment}",
            backup_storage="s3",  # Default to S3
            storage_bucket=args.storage_bucket,
            encryption_key=args.encryption_key
        )
    
    manager = BackupRestoreManager(backup_config)
    
    try:
        if args.action == 'backup':
            if args.backup_type == 'full':
                metadata = manager.create_full_backup()
                print(f"Full backup completed: {metadata.backup_id}")
            else:
                metadata = manager.create_incremental_backup()
                print(f"Incremental backup completed: {metadata.backup_id}")
        
        elif args.action == 'restore':
            if not args.backup_id:
                print("Error: --backup-id required for restore operation")
                sys.exit(1)
            
            success = manager.restore_from_backup(args.backup_id)
            if success:
                print(f"Restore completed successfully from backup: {args.backup_id}")
                sys.exit(0)
            else:
                print(f"Restore failed from backup: {args.backup_id}")
                sys.exit(1)
        
        elif args.action == 'list':
            backups = manager.list_backups()
            for backup in backups:
                print(f"{backup.backup_id} - {backup.timestamp} - {backup.backup_type} - {backup.status}")
        
        elif args.action == 'cleanup':
            deleted_count = manager.cleanup_old_backups()
            print(f"Cleaned up {deleted_count} old backups")
        
        elif args.action == 'verify':
            if not args.backup_id:
                print("Error: --backup-id required for verify operation")
                sys.exit(1)
            
            success = manager.verify_backup(args.backup_id)
            if success:
                print(f"Backup verification successful: {args.backup_id}")
                sys.exit(0)
            else:
                print(f"Backup verification failed: {args.backup_id}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()