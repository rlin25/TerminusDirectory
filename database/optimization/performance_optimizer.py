"""
Production Database Performance Optimizer
Advanced database optimization, monitoring, and maintenance for high-scale operations
"""

import asyncio
import logging
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncpg
from asyncpg import Connection, Record
import math


class OptimizationType(Enum):
    QUERY_OPTIMIZATION = "query_optimization"
    INDEX_OPTIMIZATION = "index_optimization"
    PARTITION_MAINTENANCE = "partition_maintenance"
    VACUUM_ANALYSIS = "vacuum_analysis"
    STATISTICS_UPDATE = "statistics_update"
    CONNECTION_OPTIMIZATION = "connection_optimization"


class PerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QueryPerformance:
    query_hash: str
    query_text: str
    calls: int
    total_time: float
    mean_time: float
    max_time: float
    min_time: float
    rows_affected: int
    cache_hit_ratio: float
    index_scan_ratio: float
    optimization_priority: int
    suggested_indexes: List[str] = None
    suggested_rewrites: List[str] = None
    
    def __post_init__(self):
        if self.suggested_indexes is None:
            self.suggested_indexes = []
        if self.suggested_rewrites is None:
            self.suggested_rewrites = []


@dataclass
class IndexAnalysis:
    schema_name: str
    table_name: str
    index_name: str
    index_size: int
    index_scans: int
    tuples_read: int
    tuples_fetched: int
    efficiency_ratio: float
    is_unique: bool
    is_primary: bool
    columns: List[str]
    usage_recommendation: str
    
    
@dataclass
class TableAnalysis:
    schema_name: str
    table_name: str
    total_size: int
    index_size: int
    row_count: int
    dead_tuples: int
    live_tuples: int
    bloat_percentage: float
    last_vacuum: Optional[datetime]
    last_analyze: Optional[datetime]
    needs_vacuum: bool
    needs_analyze: bool
    partition_candidate: bool
    suggested_partitioning: Optional[str] = None


class DatabasePerformanceOptimizer:
    """
    Production-grade database performance optimizer with:
    - Automatic query analysis and optimization
    - Index recommendation and management
    - Partition maintenance and optimization
    - Vacuum and analyze scheduling
    - Real-time performance monitoring
    """
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.slow_query_threshold = 1.0  # seconds
        self.inefficient_index_threshold = 0.1  # efficiency ratio
        self.bloat_threshold = 20  # percentage
        self.large_table_threshold = 1000000  # rows
        
        # Monitoring state
        self.last_analysis = {}
        self.optimization_history = []
        
    async def analyze_query_performance(self) -> List[QueryPerformance]:
        """Analyze query performance using pg_stat_statements"""
        queries = []
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            # Get slow and frequent queries
            rows = await conn.fetch("""
                SELECT 
                    queryid,
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    max_exec_time,
                    min_exec_time,
                    rows,
                    100.0 * shared_blks_hit / 
                        NULLIF(shared_blks_hit + shared_blks_read, 0) as cache_hit_ratio,
                    CASE 
                        WHEN calls > 0 THEN 
                            100.0 * (shared_blks_hit + shared_blks_read) / calls
                        ELSE 0 
                    END as avg_blocks_per_call
                FROM pg_stat_statements 
                WHERE calls > 10  -- Only analyze frequently called queries
                ORDER BY total_exec_time DESC
                LIMIT 100
            """)
            
            for row in rows:
                # Calculate optimization priority
                priority = self._calculate_optimization_priority(
                    row['total_exec_time'], row['calls'], row['mean_exec_time']
                )
                
                query_perf = QueryPerformance(
                    query_hash=str(row['queryid']),
                    query_text=row['query'],
                    calls=row['calls'],
                    total_time=row['total_exec_time'],
                    mean_time=row['mean_exec_time'],
                    max_time=row['max_exec_time'],
                    min_time=row['min_exec_time'],
                    rows_affected=row['rows'],
                    cache_hit_ratio=row['cache_hit_ratio'] or 0,
                    index_scan_ratio=0,  # Will be calculated separately
                    optimization_priority=priority
                )
                
                # Analyze query for index suggestions
                await self._analyze_query_for_indexes(conn, query_perf)
                
                queries.append(query_perf)
                
        return sorted(queries, key=lambda x: x.optimization_priority, reverse=True)
        
    def _calculate_optimization_priority(self, total_time: float, calls: int, mean_time: float) -> int:
        """Calculate optimization priority score"""
        # Weight factors
        total_time_weight = 0.4
        calls_weight = 0.3
        mean_time_weight = 0.3
        
        # Normalize values (simple scoring)
        total_time_score = min(total_time / 1000, 100)  # Cap at 100
        calls_score = min(calls / 100, 100)  # Cap at 100
        mean_time_score = min(mean_time, 100)  # Cap at 100
        
        priority = (
            total_time_score * total_time_weight +
            calls_score * calls_weight +
            mean_time_score * mean_time_weight
        )
        
        return int(priority)
        
    async def _analyze_query_for_indexes(self, conn: Connection, query: QueryPerformance):
        """Analyze query to suggest optimal indexes"""
        # Simple index analysis based on query patterns
        query_text = query.query_text.lower()
        
        # Look for WHERE clause patterns
        if 'where' in query_text:
            # Extract table and column patterns (simplified)
            if 'properties' in query_text:
                if 'price' in query_text and 'bedrooms' in query_text:
                    query.suggested_indexes.append("CREATE INDEX idx_properties_price_bedrooms ON properties(price, bedrooms);")
                elif 'location' in query_text and 'status' in query_text:
                    query.suggested_indexes.append("CREATE INDEX idx_properties_location_status ON properties(location, status);")
                    
            if 'user_interactions' in query_text:
                if 'user_id' in query_text and 'timestamp' in query_text:
                    query.suggested_indexes.append("CREATE INDEX idx_user_interactions_user_timestamp ON user_interactions(user_id, timestamp);")
                    
        # Look for JOIN patterns
        if 'join' in query_text:
            if 'properties' in query_text and 'user_interactions' in query_text:
                query.suggested_indexes.append("CREATE INDEX idx_user_interactions_property_id ON user_interactions(property_id);")
                
    async def analyze_index_performance(self) -> List[IndexAnalysis]:
        """Analyze index usage and efficiency"""
        indexes = []
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            rows = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                    pg_relation_size(indexrelid) as index_size_bytes,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    indisunique,
                    indisprimary,
                    CASE 
                        WHEN idx_scan > 0 THEN 
                            ROUND((idx_tup_fetch::numeric / idx_tup_read) * 100, 2)
                        ELSE 0 
                    END as efficiency_ratio,
                    pg_get_indexdef(indexrelid) as index_def
                FROM pg_stat_user_indexes psi
                JOIN pg_index pi ON psi.indexrelid = pi.indexrelid
                ORDER BY pg_relation_size(indexrelid) DESC
            """)
            
            for row in rows:
                # Extract column names from index definition
                columns = self._extract_index_columns(row['index_def'])
                
                # Determine usage recommendation
                recommendation = self._get_index_recommendation(
                    row['idx_scan'], row['efficiency_ratio'], row['index_size_bytes']
                )
                
                index_analysis = IndexAnalysis(
                    schema_name=row['schemaname'],
                    table_name=row['tablename'],
                    index_name=row['indexname'],
                    index_size=row['index_size_bytes'],
                    index_scans=row['idx_scan'],
                    tuples_read=row['idx_tup_read'],
                    tuples_fetched=row['idx_tup_fetch'],
                    efficiency_ratio=row['efficiency_ratio'],
                    is_unique=row['indisunique'],
                    is_primary=row['indisprimary'],
                    columns=columns,
                    usage_recommendation=recommendation
                )
                
                indexes.append(index_analysis)
                
        return indexes
        
    def _extract_index_columns(self, index_def: str) -> List[str]:
        """Extract column names from index definition"""
        # Simple extraction (could be enhanced with proper SQL parsing)
        if '(' in index_def and ')' in index_def:
            start = index_def.index('(') + 1
            end = index_def.rindex(')')
            columns_str = index_def[start:end]
            return [col.strip() for col in columns_str.split(',')]
        return []
        
    def _get_index_recommendation(self, scans: int, efficiency: float, size: int) -> str:
        """Get recommendation for index usage"""
        if scans == 0:
            return "UNUSED - Consider dropping"
        elif efficiency < 10 and size > 10 * 1024 * 1024:  # 10MB
            return "INEFFICIENT - Review query patterns"
        elif scans < 100 and size > 50 * 1024 * 1024:  # 50MB
            return "RARELY_USED - Consider dropping"
        elif efficiency > 80:
            return "OPTIMAL - Well utilized"
        else:
            return "MODERATE - Monitor usage"
            
    async def analyze_table_bloat(self) -> List[TableAnalysis]:
        """Analyze table bloat and maintenance needs"""
        tables = []
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            # Get table statistics and bloat estimation
            rows = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                    pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes,
                    pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
                    pg_indexes_size(schemaname||'.'||tablename) as index_size_bytes,
                    n_tup_ins + n_tup_upd + n_tup_del as total_ops,
                    n_dead_tup,
                    n_live_tup,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze,
                    CASE 
                        WHEN n_live_tup > 0 THEN 
                            ROUND((n_dead_tup::numeric / n_live_tup) * 100, 2)
                        ELSE 0 
                    END as bloat_percentage
                FROM pg_stat_user_tables 
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            
            for row in rows:
                # Determine maintenance needs
                needs_vacuum = (
                    row['bloat_percentage'] > self.bloat_threshold or
                    row['n_dead_tup'] > 1000
                )
                
                needs_analyze = (
                    row['last_analyze'] is None or
                    row['last_analyze'] < datetime.now() - timedelta(days=7)
                )
                
                # Check if table is a partition candidate
                partition_candidate = (
                    row['n_live_tup'] > self.large_table_threshold and
                    row['tablename'] in ['properties', 'user_interactions', 'search_queries']
                )
                
                # Suggest partitioning strategy
                suggested_partitioning = None
                if partition_candidate:
                    if row['tablename'] == 'properties':
                        suggested_partitioning = "PARTITION BY RANGE (scraped_at)"
                    elif row['tablename'] == 'user_interactions':
                        suggested_partitioning = "PARTITION BY RANGE (timestamp)"
                    elif row['tablename'] == 'search_queries':
                        suggested_partitioning = "PARTITION BY RANGE (created_at)"
                        
                # Get most recent vacuum/analyze time
                last_vacuum = max(
                    filter(None, [row['last_vacuum'], row['last_autovacuum']]),
                    default=None
                )
                
                last_analyze = max(
                    filter(None, [row['last_analyze'], row['last_autoanalyze']]),
                    default=None
                )
                
                table_analysis = TableAnalysis(
                    schema_name=row['schemaname'],
                    table_name=row['tablename'],
                    total_size=row['total_size_bytes'],
                    index_size=row['index_size_bytes'],
                    row_count=row['n_live_tup'],
                    dead_tuples=row['n_dead_tup'],
                    live_tuples=row['n_live_tup'],
                    bloat_percentage=row['bloat_percentage'],
                    last_vacuum=last_vacuum,
                    last_analyze=last_analyze,
                    needs_vacuum=needs_vacuum,
                    needs_analyze=needs_analyze,
                    partition_candidate=partition_candidate,
                    suggested_partitioning=suggested_partitioning
                )
                
                tables.append(table_analysis)
                
        return tables
        
    async def optimize_queries(self, query_performances: List[QueryPerformance]) -> Dict[str, Any]:
        """Apply query optimizations"""
        optimization_results = {
            'indexes_created': [],
            'queries_rewritten': [],
            'statistics_updated': [],
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            # Create suggested indexes for high-priority queries
            for query in query_performances[:10]:  # Top 10 queries
                if query.optimization_priority > 50:  # High priority threshold
                    for index_sql in query.suggested_indexes:
                        try:
                            # Check if index already exists
                            index_name = self._extract_index_name(index_sql)
                            exists = await conn.fetchval("""
                                SELECT EXISTS (
                                    SELECT 1 FROM pg_indexes 
                                    WHERE indexname = $1
                                )
                            """, index_name)
                            
                            if not exists:
                                # Create index concurrently to avoid blocking
                                concurrent_sql = index_sql.replace('CREATE INDEX', 'CREATE INDEX CONCURRENTLY')
                                await conn.execute(concurrent_sql)
                                optimization_results['indexes_created'].append(index_name)
                                self.logger.info(f"Created index: {index_name}")
                                
                        except Exception as e:
                            error_msg = f"Failed to create index: {e}"
                            optimization_results['errors'].append(error_msg)
                            self.logger.error(error_msg)
                            
        return optimization_results
        
    def _extract_index_name(self, index_sql: str) -> str:
        """Extract index name from CREATE INDEX statement"""
        # Simple extraction - in production, use proper SQL parsing
        parts = index_sql.split()
        if 'INDEX' in parts:
            idx = parts.index('INDEX')
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "unknown_index"
        
    async def optimize_tables(self, table_analyses: List[TableAnalysis]) -> Dict[str, Any]:
        """Apply table optimizations"""
        optimization_results = {
            'tables_vacuumed': [],
            'tables_analyzed': [],
            'tables_reindexed': [],
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            for table in table_analyses:
                table_name = f"{table.schema_name}.{table.table_name}"
                
                try:
                    # Vacuum tables with high bloat
                    if table.needs_vacuum and table.bloat_percentage > 30:
                        await conn.execute(f"VACUUM ANALYZE {table_name}")
                        optimization_results['tables_vacuumed'].append(table_name)
                        self.logger.info(f"Vacuumed table: {table_name}")
                        
                    # Analyze tables with stale statistics
                    elif table.needs_analyze:
                        await conn.execute(f"ANALYZE {table_name}")
                        optimization_results['tables_analyzed'].append(table_name)
                        self.logger.info(f"Analyzed table: {table_name}")
                        
                    # Reindex tables with very high bloat
                    if table.bloat_percentage > 50:
                        await conn.execute(f"REINDEX TABLE CONCURRENTLY {table_name}")
                        optimization_results['tables_reindexed'].append(table_name)
                        self.logger.info(f"Reindexed table: {table_name}")
                        
                except Exception as e:
                    error_msg = f"Failed to optimize table {table_name}: {e}"
                    optimization_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
        return optimization_results
        
    async def create_partition_maintenance_script(self, table_analyses: List[TableAnalysis]) -> str:
        """Create partition maintenance script for large tables"""
        script_lines = [
            "-- Partition Maintenance Script",
            "-- Generated by Performance Optimizer",
            f"-- Generated at: {datetime.now().isoformat()}",
            "",
        ]
        
        for table in table_analyses:
            if table.partition_candidate and table.suggested_partitioning:
                script_lines.extend([
                    f"-- Partition {table.table_name}",
                    f"-- Current size: {table.total_size / (1024**3):.2f} GB",
                    f"-- Rows: {table.row_count:,}",
                    f"-- Suggested: {table.suggested_partitioning}",
                    "",
                    self._generate_partition_script(table),
                    ""
                ])
                
        return "\n".join(script_lines)
        
    def _generate_partition_script(self, table: TableAnalysis) -> str:
        """Generate partition creation script for a table"""
        if table.table_name == 'properties':
            return f"""
-- Create partitioned properties table
CREATE TABLE {table.table_name}_partitioned (
    LIKE {table.table_name} INCLUDING ALL
) PARTITION BY RANGE (scraped_at);

-- Create monthly partitions for current and next 6 months
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
    i INTEGER;
BEGIN
    FOR i IN 0..6 LOOP
        start_date := DATE_TRUNC('month', CURRENT_DATE + (i || ' months')::INTERVAL);
        end_date := start_date + INTERVAL '1 month';
        partition_name := '{table.table_name}_' || TO_CHAR(start_date, 'YYYY_MM');
        
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF {table.table_name}_partitioned FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
    END LOOP;
END $$;

-- Migrate data
INSERT INTO {table.table_name}_partitioned SELECT * FROM {table.table_name};

-- Replace original table
BEGIN;
    ALTER TABLE {table.table_name} RENAME TO {table.table_name}_backup;
    ALTER TABLE {table.table_name}_partitioned RENAME TO {table.table_name};
COMMIT;
"""
        else:
            return f"-- Partitioning script for {table.table_name} not implemented"
            
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {}
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            # Database size and growth
            db_stats = await conn.fetchrow("""
                SELECT 
                    pg_database_size(current_database()) as db_size,
                    (SELECT COUNT(*) FROM pg_stat_activity) as active_connections,
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections
            """)
            
            # Query performance summary
            query_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(calls) as total_calls,
                    AVG(mean_exec_time) as avg_query_time,
                    MAX(max_exec_time) as slowest_query_time,
                    SUM(total_exec_time) as total_exec_time
                FROM pg_stat_statements
            """)
            
            # Cache hit ratios
            cache_stats = await conn.fetchrow("""
                SELECT 
                    ROUND(
                        100.0 * SUM(blks_hit) / NULLIF(SUM(blks_hit + blks_read), 0), 2
                    ) as cache_hit_ratio
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            
            # Index usage statistics
            index_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_indexes,
                    COUNT(*) FILTER (WHERE idx_scan = 0) as unused_indexes,
                    SUM(pg_relation_size(indexrelid)) as total_index_size
                FROM pg_stat_user_indexes
            """)
            
            # Table bloat summary
            bloat_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_tables,
                    AVG(CASE 
                        WHEN n_live_tup > 0 THEN 
                            (n_dead_tup::numeric / n_live_tup) * 100
                        ELSE 0 
                    END) as avg_bloat_percentage,
                    COUNT(*) FILTER (
                        WHERE (n_dead_tup::numeric / NULLIF(n_live_tup, 0)) * 100 > 20
                    ) as high_bloat_tables
                FROM pg_stat_user_tables
            """)
            
        # Compile metrics
        metrics = {
            'database': {
                'size_bytes': db_stats['db_size'],
                'size_gb': round(db_stats['db_size'] / (1024**3), 2),
                'active_connections': db_stats['active_connections'],
                'max_connections': db_stats['max_connections'],
                'connection_usage_percent': round(
                    (db_stats['active_connections'] / db_stats['max_connections']) * 100, 2
                )
            },
            'queries': {
                'total_unique_queries': query_stats['total_queries'] if query_stats else 0,
                'total_executions': query_stats['total_calls'] if query_stats else 0,
                'avg_execution_time_ms': round(query_stats['avg_query_time'] if query_stats else 0, 2),
                'slowest_query_time_ms': round(query_stats['slowest_query_time'] if query_stats else 0, 2),
                'total_execution_time_hours': round(
                    (query_stats['total_exec_time'] / 1000 / 3600) if query_stats else 0, 2
                )
            },
            'cache': {
                'hit_ratio_percent': cache_stats['cache_hit_ratio'] if cache_stats else 0
            },
            'indexes': {
                'total_count': index_stats['total_indexes'] if index_stats else 0,
                'unused_count': index_stats['unused_indexes'] if index_stats else 0,
                'total_size_gb': round(
                    (index_stats['total_index_size'] / (1024**3)) if index_stats else 0, 2
                ),
                'usage_efficiency_percent': round(
                    ((index_stats['total_indexes'] - index_stats['unused_indexes']) / 
                     max(index_stats['total_indexes'], 1)) * 100 if index_stats else 0, 2
                )
            },
            'maintenance': {
                'total_tables': bloat_stats['total_tables'] if bloat_stats else 0,
                'avg_bloat_percent': round(bloat_stats['avg_bloat_percentage'] if bloat_stats else 0, 2),
                'high_bloat_tables': bloat_stats['high_bloat_tables'] if bloat_stats else 0
            },
            'system': {
                'cpu_usage_percent': psutil.cpu_percent(),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
        
    async def get_performance_level(self, metrics: Dict[str, Any]) -> PerformanceLevel:
        """Determine overall performance level based on metrics"""
        score = 100
        
        # Deduct points for performance issues
        if metrics['queries']['avg_execution_time_ms'] > 100:
            score -= 20
        if metrics['cache']['hit_ratio_percent'] < 95:
            score -= 15
        if metrics['indexes']['usage_efficiency_percent'] < 80:
            score -= 10
        if metrics['maintenance']['avg_bloat_percent'] > 20:
            score -= 15
        if metrics['database']['connection_usage_percent'] > 80:
            score -= 10
        if metrics['system']['cpu_usage_percent'] > 80:
            score -= 10
        if metrics['system']['memory_usage_percent'] > 85:
            score -= 15
        if metrics['system']['disk_usage_percent'] > 85:
            score -= 5
            
        # Determine performance level
        if score >= 90:
            return PerformanceLevel.EXCELLENT
        elif score >= 75:
            return PerformanceLevel.GOOD
        elif score >= 60:
            return PerformanceLevel.MODERATE
        elif score >= 40:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
            
    async def run_full_optimization(self) -> Dict[str, Any]:
        """Run comprehensive performance optimization"""
        start_time = time.time()
        results = {
            'start_time': datetime.now().isoformat(),
            'query_analysis': {},
            'index_analysis': {},
            'table_analysis': {},
            'optimizations_applied': {},
            'performance_metrics': {},
            'recommendations': [],
            'errors': []
        }
        
        try:
            self.logger.info("Starting comprehensive performance optimization...")
            
            # Analyze query performance
            self.logger.info("Analyzing query performance...")
            query_performances = await self.analyze_query_performance()
            results['query_analysis'] = {
                'slow_queries_count': len([q for q in query_performances if q.mean_time > self.slow_query_threshold]),
                'top_queries': [
                    {
                        'query_hash': q.query_hash,
                        'calls': q.calls,
                        'mean_time': q.mean_time,
                        'total_time': q.total_time,
                        'optimization_priority': q.optimization_priority
                    }
                    for q in query_performances[:5]
                ]
            }
            
            # Analyze index performance
            self.logger.info("Analyzing index performance...")
            index_analyses = await self.analyze_index_performance()
            results['index_analysis'] = {
                'total_indexes': len(index_analyses),
                'unused_indexes': len([i for i in index_analyses if i.index_scans == 0]),
                'inefficient_indexes': len([i for i in index_analyses 
                                          if i.efficiency_ratio < self.inefficient_index_threshold * 100])
            }
            
            # Analyze table bloat
            self.logger.info("Analyzing table bloat...")
            table_analyses = await self.analyze_table_bloat()
            results['table_analysis'] = {
                'total_tables': len(table_analyses),
                'high_bloat_tables': len([t for t in table_analyses if t.bloat_percentage > self.bloat_threshold]),
                'partition_candidates': len([t for t in table_analyses if t.partition_candidate])
            }
            
            # Apply optimizations
            self.logger.info("Applying optimizations...")
            query_optimizations = await self.optimize_queries(query_performances)
            table_optimizations = await self.optimize_tables(table_analyses)
            
            results['optimizations_applied'] = {
                'query_optimizations': query_optimizations,
                'table_optimizations': table_optimizations
            }
            
            # Get final performance metrics
            self.logger.info("Collecting performance metrics...")
            metrics = await self.get_performance_metrics()
            performance_level = await self.get_performance_level(metrics)
            
            results['performance_metrics'] = metrics
            results['performance_level'] = performance_level.value
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                query_performances, index_analyses, table_analyses, metrics
            )
            results['recommendations'] = recommendations
            
            execution_time = time.time() - start_time
            results['execution_time_seconds'] = round(execution_time, 2)
            results['end_time'] = datetime.now().isoformat()
            
            self.logger.info(f"Performance optimization completed in {execution_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Performance optimization failed: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
            
        return results
        
    def _generate_recommendations(self, query_performances: List[QueryPerformance],
                                index_analyses: List[IndexAnalysis],
                                table_analyses: List[TableAnalysis],
                                metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Query recommendations
        slow_queries = [q for q in query_performances if q.mean_time > self.slow_query_threshold]
        if slow_queries:
            recommendations.append(
                f"Optimize {len(slow_queries)} slow queries with average execution time > {self.slow_query_threshold}s"
            )
            
        # Index recommendations
        unused_indexes = [i for i in index_analyses if i.index_scans == 0 and not i.is_primary]
        if unused_indexes:
            recommendations.append(f"Consider dropping {len(unused_indexes)} unused indexes to save space")
            
        # Table maintenance recommendations
        high_bloat_tables = [t for t in table_analyses if t.bloat_percentage > self.bloat_threshold]
        if high_bloat_tables:
            recommendations.append(f"Vacuum {len(high_bloat_tables)} tables with high bloat percentage")
            
        # Partitioning recommendations
        partition_candidates = [t for t in table_analyses if t.partition_candidate]
        if partition_candidates:
            recommendations.append(
                f"Consider partitioning {len(partition_candidates)} large tables: "
                f"{', '.join([t.table_name for t in partition_candidates])}"
            )
            
        # System recommendations
        if metrics['cache']['hit_ratio_percent'] < 95:
            recommendations.append("Increase shared_buffers to improve cache hit ratio")
            
        if metrics['database']['connection_usage_percent'] > 80:
            recommendations.append("Connection pool usage is high - consider increasing max_connections or optimizing connection usage")
            
        return recommendations