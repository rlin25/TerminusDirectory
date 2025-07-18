"""
Advanced Partition Management System
Automated partition creation, maintenance, and optimization for time-series data
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import calendar
import asyncpg
from asyncpg import Connection


class PartitionType(Enum):
    RANGE = "range"
    HASH = "hash"
    LIST = "list"


class PartitionStrategy(Enum):
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    YEARLY = "yearly"


@dataclass
class PartitionConfig:
    table_name: str
    partition_type: PartitionType
    partition_strategy: PartitionStrategy
    partition_column: str
    retention_months: int
    pre_create_months: int = 3
    auto_vacuum_enabled: bool = True
    auto_analyze_enabled: bool = True
    compression_enabled: bool = False
    
    
@dataclass
class PartitionInfo:
    partition_name: str
    table_name: str
    start_value: str
    end_value: str
    row_count: int
    size_bytes: int
    created_at: datetime
    last_modified: datetime
    is_compressed: bool = False


class PartitionManager:
    """
    Advanced partition management for high-scale time-series data:
    - Automatic partition creation and maintenance
    - Intelligent retention policies
    - Performance optimization
    - Compression management
    - Constraint management
    """
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # Default partition configurations
        self.partition_configs = {
            'properties': PartitionConfig(
                table_name='properties',
                partition_type=PartitionType.RANGE,
                partition_strategy=PartitionStrategy.MONTHLY,
                partition_column='scraped_at',
                retention_months=24,
                pre_create_months=6
            ),
            'user_interactions': PartitionConfig(
                table_name='user_interactions',
                partition_type=PartitionType.RANGE,
                partition_strategy=PartitionStrategy.MONTHLY,
                partition_column='timestamp',
                retention_months=12,
                pre_create_months=3
            ),
            'search_queries': PartitionConfig(
                table_name='search_queries',
                partition_type=PartitionType.RANGE,
                partition_strategy=PartitionStrategy.WEEKLY,
                partition_column='created_at',
                retention_months=6,
                pre_create_months=2
            ),
            'training_metrics': PartitionConfig(
                table_name='training_metrics',
                partition_type=PartitionType.RANGE,
                partition_strategy=PartitionStrategy.MONTHLY,
                partition_column='training_date',
                retention_months=36,
                pre_create_months=6
            ),
            'audit_log': PartitionConfig(
                table_name='audit_log',
                partition_type=PartitionType.RANGE,
                partition_strategy=PartitionStrategy.DAILY,
                partition_column='changed_at',
                retention_months=24,
                pre_create_months=1
            )
        }
        
    async def initialize_partitioning(self, table_name: str) -> Dict[str, Any]:
        """Initialize partitioning for a table"""
        if table_name not in self.partition_configs:
            raise ValueError(f"No partition config found for table: {table_name}")
            
        config = self.partition_configs[table_name]
        result = {
            'table_name': table_name,
            'partitions_created': [],
            'indexes_created': [],
            'constraints_added': [],
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            try:
                # Check if table is already partitioned
                is_partitioned = await self._is_table_partitioned(conn, table_name)
                
                if is_partitioned:
                    self.logger.info(f"Table {table_name} is already partitioned")
                    return result
                    
                # Create partitioned table
                await self._create_partitioned_table(conn, config)
                
                # Create initial partitions
                partitions = await self._create_initial_partitions(conn, config)
                result['partitions_created'] = partitions
                
                # Migrate existing data
                await self._migrate_data_to_partitions(conn, config)
                
                # Create indexes on partitions
                indexes = await self._create_partition_indexes(conn, config)
                result['indexes_created'] = indexes
                
                # Set up constraints
                constraints = await self._setup_partition_constraints(conn, config)
                result['constraints_added'] = constraints
                
                # Replace original table
                await self._replace_original_table(conn, table_name)
                
                self.logger.info(f"Successfully initialized partitioning for {table_name}")
                
            except Exception as e:
                error_msg = f"Failed to initialize partitioning for {table_name}: {e}"
                result['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        return result
        
    async def _is_table_partitioned(self, conn: Connection, table_name: str) -> bool:
        """Check if table is already partitioned"""
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM pg_partitioned_table pt
                JOIN pg_class c ON pt.partrelid = c.oid
                WHERE c.relname = $1
            )
        """, table_name)
        return result
        
    async def _create_partitioned_table(self, conn: Connection, config: PartitionConfig):
        """Create the main partitioned table"""
        partition_clause = self._get_partition_clause(config)
        
        # Create partitioned table with same structure as original
        await conn.execute(f"""
            CREATE TABLE {config.table_name}_partitioned (
                LIKE {config.table_name} INCLUDING ALL
            ) {partition_clause}
        """)
        
        self.logger.info(f"Created partitioned table: {config.table_name}_partitioned")
        
    def _get_partition_clause(self, config: PartitionConfig) -> str:
        """Generate partition clause based on configuration"""
        if config.partition_type == PartitionType.RANGE:
            return f"PARTITION BY RANGE ({config.partition_column})"
        elif config.partition_type == PartitionType.HASH:
            return f"PARTITION BY HASH ({config.partition_column})"
        elif config.partition_type == PartitionType.LIST:
            return f"PARTITION BY LIST ({config.partition_column})"
        else:
            raise ValueError(f"Unsupported partition type: {config.partition_type}")
            
    async def _create_initial_partitions(self, conn: Connection, config: PartitionConfig) -> List[str]:
        """Create initial partitions based on configuration"""
        partitions_created = []
        
        if config.partition_type != PartitionType.RANGE:
            # For now, only handle RANGE partitions
            return partitions_created
            
        # Calculate date range for initial partitions
        start_date = self._calculate_start_date(config)
        end_date = self._calculate_end_date(config)
        
        current_date = start_date
        while current_date < end_date:
            partition_start, partition_end = self._get_partition_bounds(current_date, config.partition_strategy)
            partition_name = self._generate_partition_name(config.table_name, partition_start, config.partition_strategy)
            
            try:
                await conn.execute(f"""
                    CREATE TABLE {partition_name} PARTITION OF {config.table_name}_partitioned
                    FOR VALUES FROM ('{partition_start}') TO ('{partition_end}')
                """)
                
                partitions_created.append(partition_name)
                self.logger.debug(f"Created partition: {partition_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to create partition {partition_name}: {e}")
                
            current_date = partition_end
            
        # Create default partition for future dates
        default_partition = f"{config.table_name}_default"
        try:
            await conn.execute(f"""
                CREATE TABLE {default_partition} PARTITION OF {config.table_name}_partitioned DEFAULT
            """)
            partitions_created.append(default_partition)
            
        except Exception as e:
            self.logger.error(f"Failed to create default partition {default_partition}: {e}")
            
        return partitions_created
        
    def _calculate_start_date(self, config: PartitionConfig) -> date:
        """Calculate start date for initial partitions"""
        today = date.today()
        
        if config.partition_strategy == PartitionStrategy.MONTHLY:
            # Start from beginning of current month minus retention period
            start_date = date(today.year, today.month, 1) - timedelta(days=30 * config.retention_months)
            return date(start_date.year, start_date.month, 1)
        elif config.partition_strategy == PartitionStrategy.WEEKLY:
            # Start from beginning of current week minus retention period
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)
            return week_start - timedelta(weeks=config.retention_months * 4)
        elif config.partition_strategy == PartitionStrategy.DAILY:
            # Start from today minus retention period
            return today - timedelta(days=30 * config.retention_months)
        elif config.partition_strategy == PartitionStrategy.YEARLY:
            # Start from beginning of current year minus retention period
            start_year = today.year - (config.retention_months // 12)
            return date(start_year, 1, 1)
        else:
            return today - timedelta(days=30 * config.retention_months)
            
    def _calculate_end_date(self, config: PartitionConfig) -> date:
        """Calculate end date for initial partitions"""
        today = date.today()
        
        if config.partition_strategy == PartitionStrategy.MONTHLY:
            # End at beginning of month that's pre_create_months in the future
            end_month = today.month + config.pre_create_months
            end_year = today.year
            while end_month > 12:
                end_month -= 12
                end_year += 1
            return date(end_year, end_month, 1)
        elif config.partition_strategy == PartitionStrategy.WEEKLY:
            return today + timedelta(weeks=config.pre_create_months * 4)
        elif config.partition_strategy == PartitionStrategy.DAILY:
            return today + timedelta(days=30 * config.pre_create_months)
        elif config.partition_strategy == PartitionStrategy.YEARLY:
            return date(today.year + config.pre_create_months, 1, 1)
        else:
            return today + timedelta(days=30 * config.pre_create_months)
            
    def _get_partition_bounds(self, current_date: date, strategy: PartitionStrategy) -> Tuple[date, date]:
        """Get partition start and end bounds"""
        if strategy == PartitionStrategy.MONTHLY:
            start_date = date(current_date.year, current_date.month, 1)
            if current_date.month == 12:
                end_date = date(current_date.year + 1, 1, 1)
            else:
                end_date = date(current_date.year, current_date.month + 1, 1)
            return start_date, end_date
            
        elif strategy == PartitionStrategy.WEEKLY:
            # Start from Monday of the week
            days_since_monday = current_date.weekday()
            start_date = current_date - timedelta(days=days_since_monday)
            end_date = start_date + timedelta(days=7)
            return start_date, end_date
            
        elif strategy == PartitionStrategy.DAILY:
            start_date = current_date
            end_date = current_date + timedelta(days=1)
            return start_date, end_date
            
        elif strategy == PartitionStrategy.YEARLY:
            start_date = date(current_date.year, 1, 1)
            end_date = date(current_date.year + 1, 1, 1)
            return start_date, end_date
            
        else:
            raise ValueError(f"Unsupported partition strategy: {strategy}")
            
    def _generate_partition_name(self, table_name: str, partition_date: date, strategy: PartitionStrategy) -> str:
        """Generate partition name based on date and strategy"""
        if strategy == PartitionStrategy.MONTHLY:
            return f"{table_name}_{partition_date.year}_{partition_date.month:02d}"
        elif strategy == PartitionStrategy.WEEKLY:
            year, week, _ = partition_date.isocalendar()
            return f"{table_name}_{year}_w{week:02d}"
        elif strategy == PartitionStrategy.DAILY:
            return f"{table_name}_{partition_date.year}_{partition_date.month:02d}_{partition_date.day:02d}"
        elif strategy == PartitionStrategy.YEARLY:
            return f"{table_name}_{partition_date.year}"
        else:
            return f"{table_name}_{partition_date.strftime('%Y_%m_%d')}"
            
    async def _migrate_data_to_partitions(self, conn: Connection, config: PartitionConfig):
        """Migrate existing data from original table to partitioned table"""
        # Copy all data from original table
        row_count = await conn.fetchval(f"""
            INSERT INTO {config.table_name}_partitioned
            SELECT * FROM {config.table_name}
            RETURNING (SELECT COUNT(*) FROM {config.table_name}_partitioned)
        """)
        
        self.logger.info(f"Migrated {row_count} rows to partitioned table {config.table_name}")
        
        # Verify data integrity
        original_count = await conn.fetchval(f"SELECT COUNT(*) FROM {config.table_name}")
        partitioned_count = await conn.fetchval(f"SELECT COUNT(*) FROM {config.table_name}_partitioned")
        
        if original_count != partitioned_count:
            raise Exception(f"Data migration failed: original={original_count}, partitioned={partitioned_count}")
            
    async def _create_partition_indexes(self, conn: Connection, config: PartitionConfig) -> List[str]:
        """Create indexes on partition tables"""
        indexes_created = []
        
        # Get list of all partitions
        partitions = await conn.fetch("""
            SELECT schemaname, tablename 
            FROM pg_tables 
            WHERE tablename LIKE $1 
            AND schemaname = 'public'
        """, f"{config.table_name}_%")
        
        for partition in partitions:
            partition_name = partition['tablename']
            
            # Skip default partition for some indexes
            if partition_name.endswith('_default'):
                continue
                
            try:
                # Create common indexes based on table type
                if config.table_name == 'properties':
                    await self._create_properties_partition_indexes(conn, partition_name)
                elif config.table_name == 'user_interactions':
                    await self._create_user_interactions_partition_indexes(conn, partition_name)
                elif config.table_name == 'search_queries':
                    await self._create_search_queries_partition_indexes(conn, partition_name)
                    
                indexes_created.append(f"indexes_for_{partition_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to create indexes for partition {partition_name}: {e}")
                
        return indexes_created
        
    async def _create_properties_partition_indexes(self, conn: Connection, partition_name: str):
        """Create indexes for properties partition"""
        indexes = [
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_status_price ON {partition_name}(status, price)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_location_gin ON {partition_name} USING GIN(location gin_trgm_ops)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_bedrooms ON {partition_name}(bedrooms)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_property_type ON {partition_name}(property_type)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_scraped_at ON {partition_name}(scraped_at)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
            
    async def _create_user_interactions_partition_indexes(self, conn: Connection, partition_name: str):
        """Create indexes for user_interactions partition"""
        indexes = [
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_user_id ON {partition_name}(user_id)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_property_id ON {partition_name}(property_id)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_interaction_type ON {partition_name}(interaction_type)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_timestamp ON {partition_name}(timestamp)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
            
    async def _create_search_queries_partition_indexes(self, conn: Connection, partition_name: str):
        """Create indexes for search_queries partition"""
        indexes = [
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_user_id ON {partition_name}(user_id)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_created_at ON {partition_name}(created_at)",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{partition_name}_query_gin ON {partition_name} USING GIN(query_text gin_trgm_ops)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
            
    async def _setup_partition_constraints(self, conn: Connection, config: PartitionConfig) -> List[str]:
        """Set up constraints for better query planning"""
        constraints_added = []
        
        # Enable constraint exclusion
        await conn.execute("SET constraint_exclusion = partition")
        
        # Add check constraints for better partition pruning
        partitions = await conn.fetch("""
            SELECT schemaname, tablename 
            FROM pg_tables 
            WHERE tablename LIKE $1 
            AND schemaname = 'public'
            AND tablename != $2
        """, f"{config.table_name}_%", f"{config.table_name}_default")
        
        for partition in partitions:
            partition_name = partition['tablename']
            
            try:
                # Extract date range from partition name and add constraint
                constraint_sql = self._generate_partition_constraint(partition_name, config)
                if constraint_sql:
                    await conn.execute(constraint_sql)
                    constraints_added.append(f"constraint_for_{partition_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to add constraint for partition {partition_name}: {e}")
                
        return constraints_added
        
    def _generate_partition_constraint(self, partition_name: str, config: PartitionConfig) -> Optional[str]:
        """Generate check constraint for partition"""
        # This is a simplified implementation
        # In production, you'd parse the partition name to extract bounds
        return None
        
    async def _replace_original_table(self, conn: Connection, table_name: str):
        """Replace original table with partitioned version"""
        backup_name = f"{table_name}_backup_{datetime.now().strftime('%Y%m%d')}"
        
        # Rename original table as backup
        await conn.execute(f"ALTER TABLE {table_name} RENAME TO {backup_name}")
        
        # Rename partitioned table to original name
        await conn.execute(f"ALTER TABLE {table_name}_partitioned RENAME TO {table_name}")
        
        self.logger.info(f"Replaced original table {table_name} (backup: {backup_name})")
        
    async def create_new_partitions(self, table_name: str) -> List[str]:
        """Create new partitions for upcoming time periods"""
        if table_name not in self.partition_configs:
            raise ValueError(f"No partition config found for table: {table_name}")
            
        config = self.partition_configs[table_name]
        partitions_created = []
        
        async with self.connection_manager.get_connection() as conn:
            # Calculate future partitions needed
            today = date.today()
            future_dates = self._get_future_partition_dates(today, config)
            
            for future_date in future_dates:
                partition_start, partition_end = self._get_partition_bounds(future_date, config.partition_strategy)
                partition_name = self._generate_partition_name(table_name, partition_start, config.partition_strategy)
                
                # Check if partition already exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_tables 
                        WHERE tablename = $1
                    )
                """, partition_name)
                
                if not exists:
                    try:
                        await conn.execute(f"""
                            CREATE TABLE {partition_name} PARTITION OF {table_name}
                            FOR VALUES FROM ('{partition_start}') TO ('{partition_end}')
                        """)
                        
                        # Create indexes for new partition
                        if config.table_name == 'properties':
                            await self._create_properties_partition_indexes(conn, partition_name)
                        elif config.table_name == 'user_interactions':
                            await self._create_user_interactions_partition_indexes(conn, partition_name)
                        elif config.table_name == 'search_queries':
                            await self._create_search_queries_partition_indexes(conn, partition_name)
                            
                        partitions_created.append(partition_name)
                        self.logger.info(f"Created new partition: {partition_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to create partition {partition_name}: {e}")
                        
        return partitions_created
        
    def _get_future_partition_dates(self, start_date: date, config: PartitionConfig) -> List[date]:
        """Get list of future dates that need partitions"""
        dates = []
        current_date = start_date
        
        for i in range(config.pre_create_months):
            if config.partition_strategy == PartitionStrategy.MONTHLY:
                if current_date.month == 12:
                    current_date = date(current_date.year + 1, 1, 1)
                else:
                    current_date = date(current_date.year, current_date.month + 1, 1)
            elif config.partition_strategy == PartitionStrategy.WEEKLY:
                current_date = current_date + timedelta(weeks=1)
            elif config.partition_strategy == PartitionStrategy.DAILY:
                current_date = current_date + timedelta(days=1)
            elif config.partition_strategy == PartitionStrategy.YEARLY:
                current_date = date(current_date.year + 1, 1, 1)
                
            dates.append(current_date)
            
        return dates
        
    async def drop_old_partitions(self, table_name: str) -> List[str]:
        """Drop old partitions based on retention policy"""
        if table_name not in self.partition_configs:
            raise ValueError(f"No partition config found for table: {table_name}")
            
        config = self.partition_configs[table_name]
        partitions_dropped = []
        
        async with self.connection_manager.get_connection() as conn:
            # Calculate cutoff date
            cutoff_date = self._calculate_cutoff_date(config)
            
            # Get list of partitions
            partitions = await conn.fetch("""
                SELECT schemaname, tablename 
                FROM pg_tables 
                WHERE tablename LIKE $1 
                AND schemaname = 'public'
                AND tablename != $2
            """, f"{table_name}_%", f"{table_name}_default")
            
            for partition in partitions:
                partition_name = partition['tablename']
                
                # Extract date from partition name
                partition_date = self._extract_date_from_partition_name(partition_name, config.partition_strategy)
                
                if partition_date and partition_date < cutoff_date:
                    try:
                        await conn.execute(f"DROP TABLE IF EXISTS {partition_name}")
                        partitions_dropped.append(partition_name)
                        self.logger.info(f"Dropped old partition: {partition_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to drop partition {partition_name}: {e}")
                        
        return partitions_dropped
        
    def _calculate_cutoff_date(self, config: PartitionConfig) -> date:
        """Calculate cutoff date for partition retention"""
        today = date.today()
        
        if config.partition_strategy == PartitionStrategy.MONTHLY:
            cutoff_month = today.month - config.retention_months
            cutoff_year = today.year
            while cutoff_month <= 0:
                cutoff_month += 12
                cutoff_year -= 1
            return date(cutoff_year, cutoff_month, 1)
        elif config.partition_strategy == PartitionStrategy.WEEKLY:
            return today - timedelta(weeks=config.retention_months * 4)
        elif config.partition_strategy == PartitionStrategy.DAILY:
            return today - timedelta(days=30 * config.retention_months)
        elif config.partition_strategy == PartitionStrategy.YEARLY:
            return date(today.year - config.retention_months // 12, 1, 1)
        else:
            return today - timedelta(days=30 * config.retention_months)
            
    def _extract_date_from_partition_name(self, partition_name: str, strategy: PartitionStrategy) -> Optional[date]:
        """Extract date from partition name"""
        try:
            parts = partition_name.split('_')
            
            if strategy == PartitionStrategy.MONTHLY and len(parts) >= 3:
                year = int(parts[-2])
                month = int(parts[-1])
                return date(year, month, 1)
            elif strategy == PartitionStrategy.YEARLY and len(parts) >= 2:
                year = int(parts[-1])
                return date(year, 1, 1)
            elif strategy == PartitionStrategy.DAILY and len(parts) >= 4:
                year = int(parts[-3])
                month = int(parts[-2])
                day = int(parts[-1])
                return date(year, month, day)
            elif strategy == PartitionStrategy.WEEKLY and len(parts) >= 2:
                # Format: table_YYYY_wWW
                year_part = parts[-2]
                week_part = parts[-1]
                if week_part.startswith('w'):
                    year = int(year_part)
                    week = int(week_part[1:])
                    # Convert ISO week to date (approximate)
                    return date(year, 1, 1) + timedelta(weeks=week-1)
                    
        except (ValueError, IndexError):
            pass
            
        return None
        
    async def get_partition_info(self, table_name: str) -> List[PartitionInfo]:
        """Get information about all partitions for a table"""
        partition_info = []
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            rows = await conn.fetch("""
                SELECT 
                    pt.schemaname,
                    pt.tablename,
                    pg_size_pretty(pg_total_relation_size(pt.schemaname||'.'||pt.tablename)) as size,
                    pg_total_relation_size(pt.schemaname||'.'||pt.tablename) as size_bytes,
                    COALESCE(ps.n_tup_ins + ps.n_tup_upd + ps.n_tup_del, 0) as operations,
                    COALESCE(ps.n_live_tup, 0) as row_count,
                    ps.last_vacuum,
                    ps.last_analyze
                FROM pg_tables pt
                LEFT JOIN pg_stat_user_tables ps ON pt.tablename = ps.relname
                WHERE pt.tablename LIKE $1
                AND pt.schemaname = 'public'
                ORDER BY pt.tablename
            """, f"{table_name}_%")
            
            for row in rows:
                info = PartitionInfo(
                    partition_name=row['tablename'],
                    table_name=table_name,
                    start_value="",  # Would need to query pg_constraint for exact bounds
                    end_value="",
                    row_count=row['row_count'],
                    size_bytes=row['size_bytes'],
                    created_at=datetime.now(),  # Would need to track this separately
                    last_modified=row['last_analyze'] or datetime.now()
                )
                partition_info.append(info)
                
        return partition_info
        
    async def run_partition_maintenance(self) -> Dict[str, Any]:
        """Run comprehensive partition maintenance for all configured tables"""
        results = {
            'tables_processed': [],
            'new_partitions_created': [],
            'old_partitions_dropped': [],
            'errors': []
        }
        
        for table_name in self.partition_configs.keys():
            try:
                self.logger.info(f"Running partition maintenance for {table_name}")
                
                # Create new partitions
                new_partitions = await self.create_new_partitions(table_name)
                results['new_partitions_created'].extend(new_partitions)
                
                # Drop old partitions
                dropped_partitions = await self.drop_old_partitions(table_name)
                results['old_partitions_dropped'].extend(dropped_partitions)
                
                results['tables_processed'].append(table_name)
                
            except Exception as e:
                error_msg = f"Partition maintenance failed for {table_name}: {e}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        return results