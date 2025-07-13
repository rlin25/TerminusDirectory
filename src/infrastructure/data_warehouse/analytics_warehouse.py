"""
Analytics Warehouse for comprehensive data storage and processing.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import redis.asyncio as aioredis
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import boto3
from pathlib import Path


class StorageType(Enum):
    """Storage layer types."""
    HOT = "hot"          # Active data (Redis, fast access)
    WARM = "warm"        # Recent data (PostgreSQL, balanced)
    COLD = "cold"        # Historical data (S3, slow access)
    ARCHIVE = "archive"  # Long-term storage (Glacier, very slow)


class CompressionType(Enum):
    """Data compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    PARQUET = "parquet"


@dataclass
class DataPartition:
    """Data partition configuration."""
    partition_key: str
    partition_value: str
    partition_type: str  # 'time', 'hash', 'range'
    storage_type: StorageType
    compression: CompressionType
    row_count: int
    size_bytes: int
    created_at: datetime
    last_accessed: datetime


@dataclass
class WarehouseConfig:
    """Analytics warehouse configuration."""
    # Database connections
    primary_db_url: str
    replica_db_urls: List[str]
    redis_url: str
    
    # Storage configuration
    hot_storage_ttl: int = 86400  # 24 hours
    warm_storage_ttl: int = 2592000  # 30 days
    cold_storage_path: str = "s3://analytics-cold-storage"
    archive_storage_path: str = "s3://analytics-archive"
    
    # Performance settings
    batch_size: int = 10000
    max_workers: int = 8
    query_timeout: int = 300
    connection_pool_size: int = 20
    
    # Partitioning
    time_partition_interval: str = "day"
    hash_partition_count: int = 32
    
    # Compression
    default_compression: CompressionType = CompressionType.SNAPPY
    archive_compression: CompressionType = CompressionType.GZIP


class AnalyticsWarehouse:
    """
    Analytics Warehouse for comprehensive data storage and processing.
    
    Provides multi-tier storage, automated data lifecycle management,
    and optimized query processing for analytics workloads.
    """
    
    def __init__(
        self,
        config: WarehouseConfig,
        db_session: AsyncSession,
        redis_client: aioredis.Redis
    ):
        self.config = config
        self.db_session = db_session
        self.redis_client = redis_client
        
        # Initialize engines for different storage tiers
        self.primary_engine = create_async_engine(
            config.primary_db_url,
            pool_size=config.connection_pool_size,
            max_overflow=config.connection_pool_size,
            echo=False
        )
        
        self.replica_engines = [
            create_async_engine(url, pool_size=5, max_overflow=10)
            for url in config.replica_db_urls
        ]
        
        # Initialize S3 client for cold storage
        self.s3_client = boto3.client('s3')
        
        # Thread pool for parallel processing
        self.thread_executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Partition tracking
        self.partitions: Dict[str, List[DataPartition]] = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the analytics warehouse."""
        try:
            # Load partition metadata
            await self._load_partition_metadata()
            
            # Initialize storage tiers
            await self._initialize_storage_tiers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Analytics warehouse initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics warehouse: {e}")
            raise
    
    async def store_analytics_data(
        self,
        table_name: str,
        data: pd.DataFrame,
        storage_type: StorageType = StorageType.WARM,
        partition_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store analytics data with automatic partitioning and compression."""
        start_time = datetime.utcnow()
        
        try:
            # Validate and prepare data
            validated_data = await self._validate_and_prepare_data(data, table_name)
            
            # Determine partitioning strategy
            partitions = await self._create_partitions(
                validated_data, 
                table_name, 
                partition_config or {}
            )
            
            # Store data in appropriate storage tier
            storage_results = []
            for partition_data, partition_info in partitions:
                result = await self._store_partition(
                    table_name,
                    partition_data,
                    partition_info,
                    storage_type
                )
                storage_results.append(result)
            
            # Update partition metadata
            await self._update_partition_metadata(table_name, partitions)
            
            # Update table statistics
            await self._update_table_statistics(table_name, validated_data)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "status": "success",
                "table_name": table_name,
                "records_stored": len(validated_data),
                "partitions_created": len(partitions),
                "storage_type": storage_type.value,
                "execution_time_seconds": execution_time,
                "storage_results": storage_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to store analytics data for {table_name}: {e}")
            return {
                "status": "error",
                "table_name": table_name,
                "error_message": str(e),
                "execution_time_seconds": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def query_analytics_data(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        storage_types: List[StorageType] = None,
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> Dict[str, Any]:
        """Query analytics data across storage tiers with optimization."""
        start_time = datetime.utcnow()
        cache_key = f"query:{hash(query + str(parameters))}"
        
        try:
            # Check cache first
            if use_cache:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    self.logger.info(f"Query cache hit for key: {cache_key}")
                    return json.loads(cached_result)
            
            # Parse and optimize query
            optimized_query = await self._optimize_query(query, parameters or {})
            
            # Determine which storage tiers to query
            target_storage_types = storage_types or [StorageType.HOT, StorageType.WARM]
            
            # Execute query across storage tiers
            results = []
            for storage_type in target_storage_types:
                tier_result = await self._query_storage_tier(
                    optimized_query,
                    parameters or {},
                    storage_type
                )
                if tier_result["data"]:
                    results.append(tier_result)
            
            # Combine and deduplicate results
            combined_data = await self._combine_query_results(results)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            query_result = {
                "status": "success",
                "data": combined_data,
                "row_count": len(combined_data) if combined_data else 0,
                "execution_time_seconds": execution_time,
                "cache_used": False,
                "storage_tiers_queried": [st.value for st in target_storage_types]
            }
            
            # Cache successful results
            if use_cache and query_result["status"] == "success":
                await self.redis_client.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps(query_result, default=str)
                )
            
            return query_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute analytics query: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "execution_time_seconds": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def aggregate_data(
        self,
        table_name: str,
        aggregation_config: Dict[str, Any],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform aggregations on analytics data."""
        try:
            # Build aggregation query
            agg_query = await self._build_aggregation_query(
                table_name,
                aggregation_config,
                time_range,
                group_by
            )
            
            # Execute aggregation
            result = await self.query_analytics_data(
                agg_query["query"],
                agg_query["parameters"],
                use_cache=True,
                cache_ttl=600  # 10 minutes cache for aggregations
            )
            
            if result["status"] == "success":
                # Post-process aggregation results
                processed_data = await self._post_process_aggregation(
                    result["data"],
                    aggregation_config
                )
                result["data"] = processed_data
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate data for {table_name}: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    async def manage_data_lifecycle(self) -> Dict[str, Any]:
        """Manage data lifecycle across storage tiers."""
        try:
            lifecycle_stats = {
                "hot_to_warm": 0,
                "warm_to_cold": 0,
                "cold_to_archive": 0,
                "deleted": 0
            }
            
            # Move hot data to warm storage
            hot_to_warm = await self._migrate_hot_to_warm()
            lifecycle_stats["hot_to_warm"] = hot_to_warm
            
            # Move warm data to cold storage
            warm_to_cold = await self._migrate_warm_to_cold()
            lifecycle_stats["warm_to_cold"] = warm_to_cold
            
            # Move cold data to archive
            cold_to_archive = await self._migrate_cold_to_archive()
            lifecycle_stats["cold_to_archive"] = cold_to_archive
            
            # Delete expired data
            deleted = await self._delete_expired_data()
            lifecycle_stats["deleted"] = deleted
            
            # Update partition metadata
            await self._update_lifecycle_metadata()
            
            return {
                "status": "success",
                "lifecycle_stats": lifecycle_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Data lifecycle management failed: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    async def get_warehouse_statistics(self) -> Dict[str, Any]:
        """Get comprehensive warehouse statistics."""
        try:
            stats = {
                "storage_tiers": {},
                "table_statistics": {},
                "partition_statistics": {},
                "performance_metrics": {},
                "data_quality_metrics": {}
            }
            
            # Storage tier statistics
            for storage_type in StorageType:
                tier_stats = await self._get_storage_tier_stats(storage_type)
                stats["storage_tiers"][storage_type.value] = tier_stats
            
            # Table statistics
            stats["table_statistics"] = await self._get_table_statistics()
            
            # Partition statistics
            stats["partition_statistics"] = await self._get_partition_statistics()
            
            # Performance metrics
            stats["performance_metrics"] = await self._get_performance_metrics()
            
            # Data quality metrics
            stats["data_quality_metrics"] = await self._get_data_quality_metrics()
            
            return {
                "status": "success",
                "statistics": stats,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get warehouse statistics: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    # Private methods
    async def _validate_and_prepare_data(
        self,
        data: pd.DataFrame,
        table_name: str
    ) -> pd.DataFrame:
        """Validate and prepare data for storage."""
        # Data validation logic
        if data.empty:
            raise ValueError("Cannot store empty DataFrame")
        
        # Add metadata columns
        data = data.copy()
        data['_ingested_at'] = datetime.utcnow()
        data['_table_name'] = table_name
        
        # Handle missing values
        data = data.fillna("")
        
        # Ensure proper data types
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_datetime(data[col])
                except:
                    pass  # Keep as object/string
        
        return data
    
    async def _create_partitions(
        self,
        data: pd.DataFrame,
        table_name: str,
        partition_config: Dict[str, Any]
    ) -> List[Tuple[pd.DataFrame, DataPartition]]:
        """Create data partitions based on configuration."""
        partitions = []
        
        partition_strategy = partition_config.get("strategy", "time")
        
        if partition_strategy == "time":
            # Time-based partitioning
            time_column = partition_config.get("time_column", "timestamp")
            interval = partition_config.get("interval", "day")
            
            if time_column in data.columns:
                data[time_column] = pd.to_datetime(data[time_column])
                
                if interval == "hour":
                    data['partition_key'] = data[time_column].dt.strftime('%Y%m%d%H')
                elif interval == "day":
                    data['partition_key'] = data[time_column].dt.strftime('%Y%m%d')
                elif interval == "month":
                    data['partition_key'] = data[time_column].dt.strftime('%Y%m')
                
                for partition_key, group_data in data.groupby('partition_key'):
                    partition_info = DataPartition(
                        partition_key=str(partition_key),
                        partition_value=str(partition_key),
                        partition_type="time",
                        storage_type=StorageType.WARM,
                        compression=self.config.default_compression,
                        row_count=len(group_data),
                        size_bytes=group_data.memory_usage(deep=True).sum(),
                        created_at=datetime.utcnow(),
                        last_accessed=datetime.utcnow()
                    )
                    partitions.append((group_data.drop('partition_key', axis=1), partition_info))
            
        elif partition_strategy == "hash":
            # Hash-based partitioning
            hash_column = partition_config.get("hash_column", "id")
            partition_count = partition_config.get("partition_count", 8)
            
            if hash_column in data.columns:
                data['partition_key'] = data[hash_column].apply(hash) % partition_count
                
                for partition_key, group_data in data.groupby('partition_key'):
                    partition_info = DataPartition(
                        partition_key=str(partition_key),
                        partition_value=str(partition_key),
                        partition_type="hash",
                        storage_type=StorageType.WARM,
                        compression=self.config.default_compression,
                        row_count=len(group_data),
                        size_bytes=group_data.memory_usage(deep=True).sum(),
                        created_at=datetime.utcnow(),
                        last_accessed=datetime.utcnow()
                    )
                    partitions.append((group_data.drop('partition_key', axis=1), partition_info))
        
        # If no partitioning strategy worked, create single partition
        if not partitions:
            partition_info = DataPartition(
                partition_key="default",
                partition_value="default",
                partition_type="none",
                storage_type=StorageType.WARM,
                compression=self.config.default_compression,
                row_count=len(data),
                size_bytes=data.memory_usage(deep=True).sum(),
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            partitions.append((data, partition_info))
        
        return partitions
    
    async def _store_partition(
        self,
        table_name: str,
        data: pd.DataFrame,
        partition_info: DataPartition,
        storage_type: StorageType
    ) -> Dict[str, Any]:
        """Store a data partition in the specified storage tier."""
        try:
            if storage_type == StorageType.HOT:
                return await self._store_to_hot_storage(table_name, data, partition_info)
            elif storage_type == StorageType.WARM:
                return await self._store_to_warm_storage(table_name, data, partition_info)
            elif storage_type == StorageType.COLD:
                return await self._store_to_cold_storage(table_name, data, partition_info)
            elif storage_type == StorageType.ARCHIVE:
                return await self._store_to_archive_storage(table_name, data, partition_info)
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
                
        except Exception as e:
            return {
                "status": "error",
                "partition_key": partition_info.partition_key,
                "error_message": str(e)
            }
    
    async def _store_to_hot_storage(
        self,
        table_name: str,
        data: pd.DataFrame,
        partition_info: DataPartition
    ) -> Dict[str, Any]:
        """Store data to hot storage (Redis)."""
        try:
            # Convert data to JSON for Redis storage
            records = data.to_dict('records')
            
            # Store each record with expiration
            pipeline = self.redis_client.pipeline()
            for i, record in enumerate(records):
                key = f"{table_name}:{partition_info.partition_key}:{i}"
                pipeline.setex(
                    key,
                    self.config.hot_storage_ttl,
                    json.dumps(record, default=str)
                )
            
            await pipeline.execute()
            
            return {
                "status": "success",
                "partition_key": partition_info.partition_key,
                "records_stored": len(records),
                "storage_location": "redis"
            }
            
        except Exception as e:
            raise Exception(f"Failed to store to hot storage: {e}")
    
    async def _store_to_warm_storage(
        self,
        table_name: str,
        data: pd.DataFrame,
        partition_info: DataPartition
    ) -> Dict[str, Any]:
        """Store data to warm storage (PostgreSQL)."""
        try:
            # Create partition table if it doesn't exist
            partition_table_name = f"{table_name}_{partition_info.partition_key}"
            
            # Store data using pandas to_sql
            async with self.primary_engine.begin() as conn:
                # Use run_sync for pandas operations
                await conn.run_sync(
                    lambda sync_conn: data.to_sql(
                        partition_table_name,
                        sync_conn,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=self.config.batch_size
                    )
                )
            
            return {
                "status": "success",
                "partition_key": partition_info.partition_key,
                "records_stored": len(data),
                "storage_location": f"postgresql:{partition_table_name}"
            }
            
        except Exception as e:
            raise Exception(f"Failed to store to warm storage: {e}")
    
    async def _store_to_cold_storage(
        self,
        table_name: str,
        data: pd.DataFrame,
        partition_info: DataPartition
    ) -> Dict[str, Any]:
        """Store data to cold storage (S3)."""
        try:
            # Convert to parquet for efficient storage
            file_name = f"{table_name}/{partition_info.partition_key}.parquet"
            s3_key = f"{self.config.cold_storage_path.replace('s3://', '')}/{file_name}"
            
            # Use thread executor for blocking operations
            await asyncio.get_event_loop().run_in_executor(
                self.thread_executor,
                self._write_parquet_to_s3,
                data,
                s3_key
            )
            
            return {
                "status": "success",
                "partition_key": partition_info.partition_key,
                "records_stored": len(data),
                "storage_location": f"s3://{s3_key}"
            }
            
        except Exception as e:
            raise Exception(f"Failed to store to cold storage: {e}")
    
    async def _store_to_archive_storage(
        self,
        table_name: str,
        data: pd.DataFrame,
        partition_info: DataPartition
    ) -> Dict[str, Any]:
        """Store data to archive storage (S3 Glacier)."""
        try:
            # Similar to cold storage but with Glacier storage class
            file_name = f"{table_name}/{partition_info.partition_key}.parquet.gz"
            s3_key = f"{self.config.archive_storage_path.replace('s3://', '')}/{file_name}"
            
            await asyncio.get_event_loop().run_in_executor(
                self.thread_executor,
                self._write_compressed_parquet_to_s3,
                data,
                s3_key,
                'GLACIER'
            )
            
            return {
                "status": "success",
                "partition_key": partition_info.partition_key,
                "records_stored": len(data),
                "storage_location": f"s3://{s3_key}"
            }
            
        except Exception as e:
            raise Exception(f"Failed to store to archive storage: {e}")
    
    def _write_parquet_to_s3(self, data: pd.DataFrame, s3_key: str) -> None:
        """Write DataFrame to S3 as Parquet (blocking operation)."""
        import io
        buffer = io.BytesIO()
        data.to_parquet(buffer, compression='snappy')
        buffer.seek(0)
        
        bucket, key = s3_key.split('/', 1)
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
            StorageClass='STANDARD_IA'
        )
    
    def _write_compressed_parquet_to_s3(
        self,
        data: pd.DataFrame,
        s3_key: str,
        storage_class: str
    ) -> None:
        """Write compressed DataFrame to S3 (blocking operation)."""
        import io
        import gzip
        
        # Create parquet buffer
        parquet_buffer = io.BytesIO()
        data.to_parquet(parquet_buffer, compression='snappy')
        parquet_buffer.seek(0)
        
        # Compress with gzip
        compressed_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as gz_file:
            gz_file.write(parquet_buffer.getvalue())
        compressed_buffer.seek(0)
        
        bucket, key = s3_key.split('/', 1)
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=compressed_buffer.getvalue(),
            StorageClass=storage_class
        )
    
    # Additional helper methods would be implemented here
    async def _load_partition_metadata(self) -> None:
        """Load partition metadata from database."""
        pass
    
    async def _initialize_storage_tiers(self) -> None:
        """Initialize storage tier connections and configurations."""
        pass
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        pass
    
    async def _optimize_query(self, query: str, parameters: Dict[str, Any]) -> str:
        """Optimize query for analytics workload."""
        return query
    
    async def _query_storage_tier(
        self,
        query: str,
        parameters: Dict[str, Any],
        storage_type: StorageType
    ) -> Dict[str, Any]:
        """Query specific storage tier."""
        return {"data": [], "metadata": {}}
    
    async def _combine_query_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine and deduplicate query results from multiple tiers."""
        combined_data = []
        for result in results:
            combined_data.extend(result["data"])
        return combined_data
    
    async def _build_aggregation_query(
        self,
        table_name: str,
        aggregation_config: Dict[str, Any],
        time_range: Optional[Tuple[datetime, datetime]],
        group_by: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Build aggregation query."""
        return {"query": "", "parameters": {}}
    
    async def _post_process_aggregation(
        self,
        data: List[Dict[str, Any]],
        aggregation_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Post-process aggregation results."""
        return data
    
    async def _migrate_hot_to_warm(self) -> int:
        """Migrate data from hot to warm storage."""
        return 0
    
    async def _migrate_warm_to_cold(self) -> int:
        """Migrate data from warm to cold storage."""
        return 0
    
    async def _migrate_cold_to_archive(self) -> int:
        """Migrate data from cold to archive storage."""
        return 0
    
    async def _delete_expired_data(self) -> int:
        """Delete expired data."""
        return 0
    
    async def _update_partition_metadata(
        self,
        table_name: str,
        partitions: List[Tuple[pd.DataFrame, DataPartition]]
    ) -> None:
        """Update partition metadata."""
        pass
    
    async def _update_table_statistics(self, table_name: str, data: pd.DataFrame) -> None:
        """Update table statistics."""
        pass
    
    async def _update_lifecycle_metadata(self) -> None:
        """Update data lifecycle metadata."""
        pass
    
    async def _get_storage_tier_stats(self, storage_type: StorageType) -> Dict[str, Any]:
        """Get statistics for a storage tier."""
        return {}
    
    async def _get_table_statistics(self) -> Dict[str, Any]:
        """Get table statistics."""
        return {}
    
    async def _get_partition_statistics(self) -> Dict[str, Any]:
        """Get partition statistics."""
        return {}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {}
    
    async def _get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics."""
        return {}