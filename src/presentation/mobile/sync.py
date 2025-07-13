"""
Offline synchronization manager for mobile applications.

This module provides offline-first capabilities with data synchronization,
conflict resolution, and delta sync for mobile devices.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class SyncAction(str, Enum):
    """Synchronization actions"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"


class SyncStatus(str, Enum):
    """Synchronization status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies"""
    CLIENT_WINS = "client_wins"
    SERVER_WINS = "server_wins"
    MERGE = "merge"
    MANUAL = "manual"


class EntityType(str, Enum):
    """Supported entity types for sync"""
    PROPERTY = "property"
    USER_PROFILE = "user_profile"
    SEARCH_HISTORY = "search_history"
    FAVORITES = "favorites"
    RECOMMENDATIONS = "recommendations"
    SETTINGS = "settings"


class SyncRecord(BaseModel):
    """Individual sync record"""
    id: str = Field(..., description="Unique sync record ID")
    entity_type: EntityType = Field(..., description="Type of entity being synced")
    entity_id: str = Field(..., description="Entity identifier")
    action: SyncAction = Field(..., description="Sync action")
    data: Dict[str, Any] = Field(default={}, description="Entity data")
    checksum: str = Field(..., description="Data checksum for integrity")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    device_id: str = Field(..., description="Source device ID")
    version: int = Field(default=1, description="Entity version")
    status: SyncStatus = Field(default=SyncStatus.PENDING)


class SyncConflict(BaseModel):
    """Sync conflict information"""
    conflict_id: str = Field(..., description="Unique conflict ID")
    entity_type: EntityType = Field(..., description="Conflicting entity type")
    entity_id: str = Field(..., description="Conflicting entity ID")
    client_record: SyncRecord = Field(..., description="Client-side record")
    server_record: SyncRecord = Field(..., description="Server-side record")
    conflict_type: str = Field(..., description="Type of conflict")
    resolution_strategy: Optional[ConflictResolution] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SyncBatch(BaseModel):
    """Batch of sync records"""
    batch_id: str = Field(..., description="Unique batch ID")
    user_id: str = Field(..., description="User ID")
    device_id: str = Field(..., description="Device ID")
    records: List[SyncRecord] = Field(default=[], description="Sync records")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None)
    status: SyncStatus = Field(default=SyncStatus.PENDING)


class SyncRequestDTO(BaseModel):
    """Sync request from mobile client"""
    device_id: str = Field(..., description="Device identifier")
    last_sync_timestamp: Optional[datetime] = Field(None, description="Last successful sync")
    client_records: List[SyncRecord] = Field(default=[], description="Client-side changes")
    requested_entities: List[EntityType] = Field(default=[], description="Entities to sync")
    batch_size: int = Field(default=50, description="Maximum records per batch")
    include_deleted: bool = Field(default=False, description="Include deleted records")


class SyncResponseDTO(BaseModel):
    """Sync response to mobile client"""
    sync_id: str = Field(..., description="Sync session ID")
    server_records: List[SyncRecord] = Field(default=[], description="Server-side changes")
    conflicts: List[SyncConflict] = Field(default=[], description="Sync conflicts")
    last_sync_timestamp: datetime = Field(default_factory=datetime.utcnow)
    has_more: bool = Field(default=False, description="More records available")
    next_batch_token: Optional[str] = Field(None, description="Token for next batch")
    has_conflicts: bool = Field(default=False, description="Has unresolved conflicts")
    sync_status: SyncStatus = Field(default=SyncStatus.COMPLETED)


class DeltaChange(BaseModel):
    """Delta change for efficient sync"""
    field_path: str = Field(..., description="JSON path to changed field")
    old_value: Any = Field(None, description="Previous value")
    new_value: Any = Field(..., description="New value")
    operation: str = Field(..., description="Change operation (set, unset, append, etc.)")


class DeltaRecord(BaseModel):
    """Delta-based sync record"""
    entity_type: EntityType = Field(..., description="Entity type")
    entity_id: str = Field(..., description="Entity ID")
    changes: List[DeltaChange] = Field(..., description="List of changes")
    base_version: int = Field(..., description="Base version for changes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OfflineSyncManager:
    """Offline synchronization manager"""
    
    def __init__(self, repository_factory):
        self.repository_factory = repository_factory
        self.redis_client: Optional[redis.Redis] = None
        self.sync_config = {
            "max_batch_size": 100,
            "conflict_retention_days": 30,
            "sync_timeout_minutes": 10,
            "max_retries": 3
        }
        
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the sync manager"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Offline sync manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize sync manager: {e}")
            raise
    
    async def process_sync(
        self,
        user_id: str,
        sync_request: SyncRequestDTO
    ) -> SyncResponseDTO:
        """Process synchronization request from mobile client"""
        try:
            sync_id = f"sync_{datetime.utcnow().timestamp()}_{user_id}"
            
            # Create sync batch
            batch = SyncBatch(
                batch_id=sync_id,
                user_id=user_id,
                device_id=sync_request.device_id,
                records=sync_request.client_records
            )
            
            # Store sync batch
            await self.store_sync_batch(batch)
            
            # Process client records
            conflicts = await self.process_client_records(user_id, sync_request.client_records)
            
            # Get server changes
            server_records = await self.get_server_changes(
                user_id,
                sync_request.device_id,
                sync_request.last_sync_timestamp,
                sync_request.requested_entities,
                sync_request.batch_size
            )
            
            # Create response
            response = SyncResponseDTO(
                sync_id=sync_id,
                server_records=server_records,
                conflicts=conflicts,
                has_conflicts=len(conflicts) > 0,
                sync_status=SyncStatus.CONFLICT if conflicts else SyncStatus.COMPLETED
            )
            
            # Update sync status
            await self.update_sync_status(sync_id, response.sync_status)
            
            logger.info(f"Sync processed: {sync_id} - {len(server_records)} records, {len(conflicts)} conflicts")
            return response
            
        except Exception as e:
            logger.error(f"Sync processing failed: {e}")
            raise
    
    async def process_client_records(
        self,
        user_id: str,
        client_records: List[SyncRecord]
    ) -> List[SyncConflict]:
        """Process client-side records and detect conflicts"""
        conflicts = []
        
        for record in client_records:
            try:
                # Verify record integrity
                if not self.verify_record_integrity(record):
                    logger.warning(f"Invalid record checksum: {record.id}")
                    continue
                
                # Check for conflicts
                conflict = await self.check_for_conflicts(user_id, record)
                if conflict:
                    conflicts.append(conflict)
                    continue
                
                # Apply record
                await self.apply_record(user_id, record)
                
                # Update record status
                record.status = SyncStatus.COMPLETED
                await self.update_record_status(record.id, SyncStatus.COMPLETED)
                
            except Exception as e:
                logger.error(f"Failed to process record {record.id}: {e}")
                record.status = SyncStatus.FAILED
                await self.update_record_status(record.id, SyncStatus.FAILED)
        
        return conflicts
    
    async def check_for_conflicts(
        self,
        user_id: str,
        client_record: SyncRecord
    ) -> Optional[SyncConflict]:
        """Check if client record conflicts with server state"""
        try:
            # Get current server record
            server_record = await self.get_server_record(
                user_id,
                client_record.entity_type,
                client_record.entity_id
            )
            
            if not server_record:
                # No server record exists, no conflict
                return None
            
            # Check for version conflicts
            if server_record.version > client_record.version:
                return SyncConflict(
                    conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                    entity_type=client_record.entity_type,
                    entity_id=client_record.entity_id,
                    client_record=client_record,
                    server_record=server_record,
                    conflict_type="version_mismatch"
                )
            
            # Check for concurrent modifications
            if (server_record.timestamp > client_record.timestamp and 
                server_record.checksum != client_record.checksum):
                return SyncConflict(
                    conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                    entity_type=client_record.entity_type,
                    entity_id=client_record.entity_id,
                    client_record=client_record,
                    server_record=server_record,
                    conflict_type="concurrent_modification"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Conflict check failed: {e}")
            return None
    
    async def apply_record(self, user_id: str, record: SyncRecord) -> bool:
        """Apply sync record to server state"""
        try:
            if record.action == SyncAction.CREATE:
                return await self.create_entity(user_id, record)
            elif record.action == SyncAction.UPDATE:
                return await self.update_entity(user_id, record)
            elif record.action == SyncAction.DELETE:
                return await self.delete_entity(user_id, record)
            else:
                logger.warning(f"Unknown sync action: {record.action}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply record: {e}")
            return False
    
    async def create_entity(self, user_id: str, record: SyncRecord) -> bool:
        """Create entity from sync record"""
        try:
            if record.entity_type == EntityType.FAVORITES:
                # Handle favorites
                user_repo = self.repository_factory.get_user_repository()
                await user_repo.add_user_favorite(user_id, record.entity_id, record.data)
                return True
            
            elif record.entity_type == EntityType.SEARCH_HISTORY:
                # Handle search history
                search_repo = self.repository_factory.get_search_repository()
                await search_repo.add_search_history(user_id, record.data)
                return True
            
            elif record.entity_type == EntityType.USER_PROFILE:
                # Handle user profile updates
                user_repo = self.repository_factory.get_user_repository()
                await user_repo.update_user_profile(user_id, record.data)
                return True
            
            # Add more entity types as needed
            return True
            
        except Exception as e:
            logger.error(f"Entity creation failed: {e}")
            return False
    
    async def update_entity(self, user_id: str, record: SyncRecord) -> bool:
        """Update entity from sync record"""
        try:
            if record.entity_type == EntityType.USER_PROFILE:
                user_repo = self.repository_factory.get_user_repository()
                await user_repo.update_user_profile(user_id, record.data)
                return True
            
            elif record.entity_type == EntityType.SETTINGS:
                # Handle settings updates
                await self.update_user_settings(user_id, record.data)
                return True
            
            # Add more entity types as needed
            return True
            
        except Exception as e:
            logger.error(f"Entity update failed: {e}")
            return False
    
    async def delete_entity(self, user_id: str, record: SyncRecord) -> bool:
        """Delete entity from sync record"""
        try:
            if record.entity_type == EntityType.FAVORITES:
                user_repo = self.repository_factory.get_user_repository()
                await user_repo.remove_user_favorite(user_id, record.entity_id)
                return True
            
            elif record.entity_type == EntityType.SEARCH_HISTORY:
                search_repo = self.repository_factory.get_search_repository()
                await search_repo.delete_search_history(user_id, record.entity_id)
                return True
            
            # Add more entity types as needed
            return True
            
        except Exception as e:
            logger.error(f"Entity deletion failed: {e}")
            return False
    
    async def get_server_changes(
        self,
        user_id: str,
        device_id: str,
        last_sync_timestamp: Optional[datetime],
        requested_entities: List[EntityType],
        batch_size: int
    ) -> List[SyncRecord]:
        """Get server-side changes since last sync"""
        try:
            since_timestamp = last_sync_timestamp or datetime.utcnow() - timedelta(days=30)
            records = []
            
            for entity_type in requested_entities:
                entity_records = await self.get_entity_changes(
                    user_id,
                    entity_type,
                    since_timestamp,
                    device_id
                )
                records.extend(entity_records)
                
                # Respect batch size limit
                if len(records) >= batch_size:
                    records = records[:batch_size]
                    break
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get server changes: {e}")
            return []
    
    async def get_entity_changes(
        self,
        user_id: str,
        entity_type: EntityType,
        since_timestamp: datetime,
        device_id: str
    ) -> List[SyncRecord]:
        """Get changes for specific entity type"""
        try:
            changes_key = f"entity_changes:{user_id}:{entity_type}"
            
            # Get changes from Redis sorted set (timestamp-based)
            min_score = since_timestamp.timestamp()
            changes = await self.redis_client.zrangebyscore(
                changes_key,
                min_score,
                "+inf",
                withscores=True
            )
            
            records = []
            for change_data, timestamp in changes:
                try:
                    change_dict = json.loads(change_data)
                    
                    # Skip changes from the same device to avoid loops
                    if change_dict.get("device_id") == device_id:
                        continue
                    
                    record = SyncRecord(**change_dict)
                    records.append(record)
                    
                except Exception as e:
                    logger.error(f"Invalid change record: {e}")
                    continue
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get entity changes: {e}")
            return []
    
    async def resolve_conflicts(
        self,
        user_id: str,
        conflicts: List[SyncConflict]
    ) -> Dict[str, bool]:
        """Resolve sync conflicts"""
        results = {}
        
        for conflict in conflicts:
            try:
                resolution_strategy = conflict.resolution_strategy or ConflictResolution.SERVER_WINS
                
                if resolution_strategy == ConflictResolution.CLIENT_WINS:
                    success = await self.apply_record(user_id, conflict.client_record)
                elif resolution_strategy == ConflictResolution.SERVER_WINS:
                    # Server version is already current, mark as resolved
                    success = True
                elif resolution_strategy == ConflictResolution.MERGE:
                    success = await self.merge_records(conflict.client_record, conflict.server_record)
                else:
                    # Manual resolution required
                    await self.store_conflict_for_manual_resolution(conflict)
                    success = False
                
                results[conflict.conflict_id] = success
                
                if success:
                    await self.mark_conflict_resolved(conflict.conflict_id)
                
            except Exception as e:
                logger.error(f"Conflict resolution failed for {conflict.conflict_id}: {e}")
                results[conflict.conflict_id] = False
        
        return results
    
    async def merge_records(
        self,
        client_record: SyncRecord,
        server_record: SyncRecord
    ) -> bool:
        """Merge client and server records"""
        try:
            # Simple merge strategy - combine non-conflicting fields
            merged_data = {**server_record.data}
            
            for key, value in client_record.data.items():
                if key not in server_record.data or server_record.data[key] is None:
                    merged_data[key] = value
                elif isinstance(value, list) and isinstance(server_record.data[key], list):
                    # Merge lists (remove duplicates)
                    merged_data[key] = list(set(server_record.data[key] + value))
            
            # Create merged record
            merged_record = SyncRecord(
                id=f"merged_{datetime.utcnow().timestamp()}",
                entity_type=client_record.entity_type,
                entity_id=client_record.entity_id,
                action=SyncAction.UPDATE,
                data=merged_data,
                checksum=self.calculate_checksum(merged_data),
                device_id="server",
                version=max(client_record.version, server_record.version) + 1
            )
            
            # Apply merged record
            return await self.apply_record("system", merged_record)
            
        except Exception as e:
            logger.error(f"Record merge failed: {e}")
            return False
    
    def verify_record_integrity(self, record: SyncRecord) -> bool:
        """Verify record data integrity using checksum"""
        try:
            calculated_checksum = self.calculate_checksum(record.data)
            return calculated_checksum == record.checksum
        except Exception:
            return False
    
    def calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data"""
        try:
            # Sort keys for consistent hash
            sorted_data = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(sorted_data.encode()).hexdigest()
        except Exception:
            return ""
    
    async def store_sync_batch(self, batch: SyncBatch) -> None:
        """Store sync batch for tracking"""
        try:
            batch_key = f"sync_batch:{batch.batch_id}"
            await self.redis_client.setex(
                batch_key,
                86400 * 7,  # 7 days
                batch.model_dump_json()
            )
        except Exception as e:
            logger.error(f"Failed to store sync batch: {e}")
    
    async def update_sync_status(self, sync_id: str, status: SyncStatus) -> None:
        """Update sync batch status"""
        try:
            batch_key = f"sync_batch:{sync_id}"
            batch_data = await self.redis_client.get(batch_key)
            
            if batch_data:
                batch = SyncBatch.model_validate_json(batch_data)
                batch.status = status
                batch.processed_at = datetime.utcnow()
                
                await self.redis_client.setex(
                    batch_key,
                    86400 * 7,
                    batch.model_dump_json()
                )
        except Exception as e:
            logger.error(f"Failed to update sync status: {e}")
    
    async def update_record_status(self, record_id: str, status: SyncStatus) -> None:
        """Update individual record status"""
        try:
            status_key = f"record_status:{record_id}"
            await self.redis_client.setex(
                status_key,
                3600,  # 1 hour
                status.value
            )
        except Exception as e:
            logger.error(f"Failed to update record status: {e}")
    
    async def get_server_record(
        self,
        user_id: str,
        entity_type: EntityType,
        entity_id: str
    ) -> Optional[SyncRecord]:
        """Get current server record for entity"""
        try:
            record_key = f"server_record:{user_id}:{entity_type}:{entity_id}"
            record_data = await self.redis_client.get(record_key)
            
            if record_data:
                return SyncRecord.model_validate_json(record_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get server record: {e}")
            return None
    
    async def store_conflict_for_manual_resolution(self, conflict: SyncConflict) -> None:
        """Store conflict for manual resolution"""
        try:
            conflict_key = f"manual_conflict:{conflict.conflict_id}"
            await self.redis_client.setex(
                conflict_key,
                86400 * self.sync_config["conflict_retention_days"],
                conflict.model_dump_json()
            )
            
            # Add to user's conflicts list
            user_conflicts_key = f"user_conflicts:{conflict.client_record.device_id}"
            await self.redis_client.sadd(user_conflicts_key, conflict.conflict_id)
            
        except Exception as e:
            logger.error(f"Failed to store manual conflict: {e}")
    
    async def mark_conflict_resolved(self, conflict_id: str) -> None:
        """Mark conflict as resolved"""
        try:
            conflict_key = f"manual_conflict:{conflict_id}"
            await self.redis_client.delete(conflict_key)
            
            # Remove from all user conflicts lists
            pattern = "user_conflicts:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                await self.redis_client.srem(key, conflict_id)
                
        except Exception as e:
            logger.error(f"Failed to mark conflict resolved: {e}")
    
    async def get_sync_statistics(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get sync statistics for user"""
        try:
            since = datetime.utcnow() - timedelta(days=days)
            pattern = f"sync_batch:*"
            keys = await self.redis_client.keys(pattern)
            
            stats = {
                "total_syncs": 0,
                "successful_syncs": 0,
                "failed_syncs": 0,
                "conflicts": 0,
                "last_sync": None,
                "avg_records_per_sync": 0
            }
            
            total_records = 0
            
            for key in keys:
                batch_data = await self.redis_client.get(key)
                if batch_data:
                    try:
                        batch = SyncBatch.model_validate_json(batch_data)
                        
                        if batch.user_id != user_id or batch.created_at < since:
                            continue
                        
                        stats["total_syncs"] += 1
                        total_records += len(batch.records)
                        
                        if batch.status == SyncStatus.COMPLETED:
                            stats["successful_syncs"] += 1
                        elif batch.status == SyncStatus.FAILED:
                            stats["failed_syncs"] += 1
                        elif batch.status == SyncStatus.CONFLICT:
                            stats["conflicts"] += 1
                        
                        if not stats["last_sync"] or batch.created_at > stats["last_sync"]:
                            stats["last_sync"] = batch.created_at
                            
                    except Exception as e:
                        logger.error(f"Invalid batch data: {e}")
                        continue
            
            if stats["total_syncs"] > 0:
                stats["avg_records_per_sync"] = total_records / stats["total_syncs"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get sync statistics: {e}")
            return {}
    
    async def cleanup_old_sync_data(self, days: int = 30) -> None:
        """Clean up old sync data"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old sync batches
            pattern = "sync_batch:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                batch_data = await self.redis_client.get(key)
                if batch_data:
                    try:
                        batch = SyncBatch.model_validate_json(batch_data)
                        if batch.created_at < cutoff:
                            await self.redis_client.delete(key)
                    except Exception:
                        # Delete invalid data
                        await self.redis_client.delete(key)
            
            # Clean up old conflicts
            pattern = "manual_conflict:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                conflict_data = await self.redis_client.get(key)
                if conflict_data:
                    try:
                        conflict = SyncConflict.model_validate_json(conflict_data)
                        if conflict.created_at < cutoff:
                            await self.redis_client.delete(key)
                    except Exception:
                        await self.redis_client.delete(key)
            
            logger.info(f"Cleaned up sync data older than {days} days")
            
        except Exception as e:
            logger.error(f"Sync cleanup failed: {e}")
    
    # Utility methods for specific sync operations
    
    async def update_user_settings(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """Update user settings"""
        try:
            settings_key = f"user_settings:{user_id}"
            await self.redis_client.hmset(settings_key, settings)
            await self.redis_client.expire(settings_key, 86400 * 365)  # 1 year
            return True
        except Exception as e:
            logger.error(f"Failed to update user settings: {e}")
            return False
    
    async def force_sync_for_user(self, user_id: str) -> bool:
        """Force complete sync for user (emergency/recovery)"""
        try:
            # Mark all user data for re-sync
            pattern = f"server_record:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                # Update timestamp to trigger sync
                record_data = await self.redis_client.get(key)
                if record_data:
                    record = SyncRecord.model_validate_json(record_data)
                    record.timestamp = datetime.utcnow()
                    
                    await self.redis_client.setex(
                        key,
                        86400 * 30,
                        record.model_dump_json()
                    )
            
            logger.info(f"Forced sync initiated for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Force sync failed: {e}")
            return False