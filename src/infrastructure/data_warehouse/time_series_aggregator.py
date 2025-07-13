"""
Time-Series Data Aggregation and Processing for Analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import pytz


class TimeUnit(Enum):
    """Time aggregation units."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class AggregationType(Enum):
    """Types of aggregations."""
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STDDEV = "stddev"
    VARIANCE = "variance"
    PERCENTILE = "percentile"
    FIRST = "first"
    LAST = "last"
    DISTINCT_COUNT = "distinct_count"


@dataclass
class AggregationRule:
    """Configuration for time-series aggregation."""
    metric_name: str
    source_table: str
    source_column: str
    aggregation_type: AggregationType
    time_unit: TimeUnit
    time_column: str = "timestamp"
    group_by_columns: List[str] = None
    filter_conditions: Dict[str, Any] = None
    percentile_value: float = None  # For percentile aggregations
    window_size: int = 1  # For rolling aggregations
    
    def __post_init__(self):
        if self.group_by_columns is None:
            self.group_by_columns = []
        if self.filter_conditions is None:
            self.filter_conditions = {}


@dataclass
class AggregationResult:
    """Result of time-series aggregation."""
    metric_name: str
    timestamp: datetime
    value: float
    dimensions: Dict[str, str]
    aggregation_type: str
    time_unit: str
    window_start: datetime
    window_end: datetime
    record_count: int
    confidence_score: float = 1.0


class TimeSeriesAggregator:
    """
    Time-Series Data Aggregation and Processing.
    
    Provides real-time and batch aggregation capabilities for analytics data
    with support for multiple time granularities and aggregation functions.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        timezone: str = "UTC"
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.timezone = pytz.timezone(timezone)
        
        # Aggregation rules registry
        self.aggregation_rules: Dict[str, AggregationRule] = {}
        
        # Thread pool for parallel processing
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache configuration
        self.cache_ttl = {
            TimeUnit.MINUTE: 60,
            TimeUnit.HOUR: 3600,
            TimeUnit.DAY: 86400,
            TimeUnit.WEEK: 604800,
            TimeUnit.MONTH: 2592000
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def register_aggregation_rule(self, rule: AggregationRule) -> bool:
        """Register a new aggregation rule."""
        try:
            # Validate rule
            await self._validate_aggregation_rule(rule)
            
            # Store in registry
            self.aggregation_rules[rule.metric_name] = rule
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                "aggregation_rules",
                rule.metric_name,
                json.dumps(asdict(rule), default=str)
            )
            
            self.logger.info(f"Registered aggregation rule: {rule.metric_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register aggregation rule {rule.metric_name}: {e}")
            return False
    
    async def run_real_time_aggregation(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AggregationResult]:
        """Run real-time aggregation for a specific metric."""
        if metric_name not in self.aggregation_rules:
            raise ValueError(f"Aggregation rule not found: {metric_name}")
        
        rule = self.aggregation_rules[metric_name]
        
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = self._get_default_start_time(rule.time_unit, end_time)
        
        try:
            # Generate time windows
            time_windows = self._generate_time_windows(start_time, end_time, rule.time_unit)
            
            # Run aggregation for each window
            results = []
            for window_start, window_end in time_windows:
                result = await self._aggregate_time_window(
                    rule,
                    window_start,
                    window_end
                )
                if result:
                    results.extend(result)
            
            # Cache results
            await self._cache_aggregation_results(metric_name, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Real-time aggregation failed for {metric_name}: {e}")
            raise
    
    async def run_batch_aggregation(
        self,
        start_time: datetime,
        end_time: datetime,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, List[AggregationResult]]:
        """Run batch aggregation for multiple metrics."""
        # Determine which metrics to process
        target_metrics = metric_names or list(self.aggregation_rules.keys())
        
        results = {}
        
        # Process metrics in parallel
        tasks = []
        for metric_name in target_metrics:
            task = self.run_real_time_aggregation(metric_name, start_time, end_time)
            tasks.append((metric_name, task))
        
        # Execute all tasks
        for metric_name, task in tasks:
            try:
                metric_results = await task
                results[metric_name] = metric_results
            except Exception as e:
                self.logger.error(f"Batch aggregation failed for {metric_name}: {e}")
                results[metric_name] = []
        
        return results
    
    async def get_aggregated_data(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        time_unit: Optional[TimeUnit] = None,
        use_cache: bool = True
    ) -> List[AggregationResult]:
        """Get aggregated data for a metric within a time range."""
        # Check cache first
        if use_cache:
            cache_key = f"agg:{metric_name}:{start_time.isoformat()}:{end_time.isoformat()}"
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                try:
                    cached_results = json.loads(cached_data)
                    return [AggregationResult(**result) for result in cached_results]
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize cached data: {e}")
        
        # Get from database
        if metric_name not in self.aggregation_rules:
            # Try to get pre-computed aggregations
            return await self._get_precomputed_aggregations(
                metric_name,
                start_time,
                end_time,
                time_unit
            )
        else:
            # Run real-time aggregation
            return await self.run_real_time_aggregation(metric_name, start_time, end_time)
    
    async def compute_rolling_aggregations(
        self,
        metric_name: str,
        window_size: int,
        time_unit: TimeUnit,
        aggregation_type: AggregationType,
        start_time: datetime,
        end_time: datetime
    ) -> List[AggregationResult]:
        """Compute rolling aggregations over a time window."""
        try:
            # Get base data
            base_rule = self.aggregation_rules.get(metric_name)
            if not base_rule:
                raise ValueError(f"Base aggregation rule not found: {metric_name}")
            
            # Get detailed data for rolling calculation
            detailed_data = await self._get_detailed_data(
                base_rule,
                start_time,
                end_time
            )
            
            # Compute rolling aggregations
            rolling_results = await self._compute_rolling_window(
                detailed_data,
                window_size,
                time_unit,
                aggregation_type,
                base_rule
            )
            
            return rolling_results
            
        except Exception as e:
            self.logger.error(f"Rolling aggregation failed for {metric_name}: {e}")
            raise
    
    async def compute_percentile_aggregations(
        self,
        metric_name: str,
        percentiles: List[float],
        start_time: datetime,
        end_time: datetime,
        time_unit: TimeUnit = TimeUnit.HOUR
    ) -> Dict[float, List[AggregationResult]]:
        """Compute percentile aggregations for a metric."""
        try:
            rule = self.aggregation_rules.get(metric_name)
            if not rule:
                raise ValueError(f"Aggregation rule not found: {metric_name}")
            
            # Generate time windows
            time_windows = self._generate_time_windows(start_time, end_time, time_unit)
            
            results = {}
            for percentile in percentiles:
                results[percentile] = []
                
                for window_start, window_end in time_windows:
                    percentile_result = await self._compute_percentile_window(
                        rule,
                        percentile,
                        window_start,
                        window_end
                    )
                    if percentile_result:
                        results[percentile].extend(percentile_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Percentile aggregation failed for {metric_name}: {e}")
            raise
    
    async def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get statistics about aggregation performance and data."""
        try:
            stats = {
                "registered_rules": len(self.aggregation_rules),
                "rule_details": {},
                "cache_statistics": {},
                "performance_metrics": {}
            }
            
            # Rule details
            for metric_name, rule in self.aggregation_rules.items():
                stats["rule_details"][metric_name] = {
                    "aggregation_type": rule.aggregation_type.value,
                    "time_unit": rule.time_unit.value,
                    "source_table": rule.source_table,
                    "group_by_columns": rule.group_by_columns
                }
            
            # Cache statistics
            stats["cache_statistics"] = await self._get_cache_statistics()
            
            # Performance metrics
            stats["performance_metrics"] = await self._get_performance_statistics()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get aggregation statistics: {e}")
            return {}
    
    # Private methods
    async def _validate_aggregation_rule(self, rule: AggregationRule) -> None:
        """Validate aggregation rule configuration."""
        # Check if source table exists
        table_exists = await self.db_session.execute(
            text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """),
            {"table_name": rule.source_table}
        )
        
        if not table_exists.scalar():
            raise ValueError(f"Source table does not exist: {rule.source_table}")
        
        # Validate percentile value if needed
        if rule.aggregation_type == AggregationType.PERCENTILE:
            if rule.percentile_value is None or not (0 <= rule.percentile_value <= 100):
                raise ValueError("Valid percentile_value (0-100) required for percentile aggregation")
    
    def _get_default_start_time(self, time_unit: TimeUnit, end_time: datetime) -> datetime:
        """Get default start time based on time unit."""
        if time_unit == TimeUnit.MINUTE:
            return end_time - timedelta(minutes=60)
        elif time_unit == TimeUnit.HOUR:
            return end_time - timedelta(hours=24)
        elif time_unit == TimeUnit.DAY:
            return end_time - timedelta(days=30)
        elif time_unit == TimeUnit.WEEK:
            return end_time - timedelta(weeks=12)
        elif time_unit == TimeUnit.MONTH:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)
    
    def _generate_time_windows(
        self,
        start_time: datetime,
        end_time: datetime,
        time_unit: TimeUnit
    ) -> List[Tuple[datetime, datetime]]:
        """Generate time windows for aggregation."""
        windows = []
        current_time = start_time
        
        if time_unit == TimeUnit.MINUTE:
            delta = timedelta(minutes=1)
        elif time_unit == TimeUnit.HOUR:
            delta = timedelta(hours=1)
        elif time_unit == TimeUnit.DAY:
            delta = timedelta(days=1)
        elif time_unit == TimeUnit.WEEK:
            delta = timedelta(weeks=1)
        elif time_unit == TimeUnit.MONTH:
            delta = timedelta(days=30)  # Approximate
        elif time_unit == TimeUnit.QUARTER:
            delta = timedelta(days=90)  # Approximate
        elif time_unit == TimeUnit.YEAR:
            delta = timedelta(days=365)  # Approximate
        else:
            delta = timedelta(hours=1)
        
        while current_time < end_time:
            window_end = min(current_time + delta, end_time)
            windows.append((current_time, window_end))
            current_time = window_end
        
        return windows
    
    async def _aggregate_time_window(
        self,
        rule: AggregationRule,
        window_start: datetime,
        window_end: datetime
    ) -> List[AggregationResult]:
        """Aggregate data for a specific time window."""
        try:
            # Build aggregation query
            query = self._build_aggregation_query(rule, window_start, window_end)
            
            # Execute query
            result = await self.db_session.execute(
                text(query["sql"]),
                query["parameters"]
            )
            
            rows = result.fetchall()
            columns = result.keys()
            
            # Convert to AggregationResult objects
            aggregation_results = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                
                # Extract dimensions
                dimensions = {}
                for col in rule.group_by_columns:
                    if col in row_dict:
                        dimensions[col] = str(row_dict[col])
                
                aggregation_result = AggregationResult(
                    metric_name=rule.metric_name,
                    timestamp=window_start,
                    value=float(row_dict.get('agg_value', 0)),
                    dimensions=dimensions,
                    aggregation_type=rule.aggregation_type.value,
                    time_unit=rule.time_unit.value,
                    window_start=window_start,
                    window_end=window_end,
                    record_count=int(row_dict.get('record_count', 0)),
                    confidence_score=1.0
                )
                
                aggregation_results.append(aggregation_result)
            
            return aggregation_results
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate window {window_start}-{window_end}: {e}")
            return []
    
    def _build_aggregation_query(
        self,
        rule: AggregationRule,
        window_start: datetime,
        window_end: datetime
    ) -> Dict[str, Any]:
        """Build SQL query for aggregation."""
        # Base SELECT clause
        select_columns = []
        
        # Add group by columns
        for col in rule.group_by_columns:
            select_columns.append(col)
        
        # Add aggregation function
        if rule.aggregation_type == AggregationType.SUM:
            agg_func = f"SUM({rule.source_column})"
        elif rule.aggregation_type == AggregationType.COUNT:
            agg_func = f"COUNT({rule.source_column})"
        elif rule.aggregation_type == AggregationType.AVG:
            agg_func = f"AVG({rule.source_column})"
        elif rule.aggregation_type == AggregationType.MIN:
            agg_func = f"MIN({rule.source_column})"
        elif rule.aggregation_type == AggregationType.MAX:
            agg_func = f"MAX({rule.source_column})"
        elif rule.aggregation_type == AggregationType.MEDIAN:
            agg_func = f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {rule.source_column})"
        elif rule.aggregation_type == AggregationType.STDDEV:
            agg_func = f"STDDEV({rule.source_column})"
        elif rule.aggregation_type == AggregationType.VARIANCE:
            agg_func = f"VARIANCE({rule.source_column})"
        elif rule.aggregation_type == AggregationType.PERCENTILE:
            percentile = rule.percentile_value / 100.0
            agg_func = f"PERCENTILE_CONT({percentile}) WITHIN GROUP (ORDER BY {rule.source_column})"
        elif rule.aggregation_type == AggregationType.FIRST:
            agg_func = f"FIRST_VALUE({rule.source_column}) OVER (ORDER BY {rule.time_column})"
        elif rule.aggregation_type == AggregationType.LAST:
            agg_func = f"LAST_VALUE({rule.source_column}) OVER (ORDER BY {rule.time_column})"
        elif rule.aggregation_type == AggregationType.DISTINCT_COUNT:
            agg_func = f"COUNT(DISTINCT {rule.source_column})"
        else:
            agg_func = f"AVG({rule.source_column})"
        
        select_columns.append(f"{agg_func} AS agg_value")
        select_columns.append("COUNT(*) AS record_count")
        
        # Build WHERE clause
        where_conditions = [
            f"{rule.time_column} >= :window_start",
            f"{rule.time_column} < :window_end"
        ]
        
        # Add filter conditions
        for column, value in rule.filter_conditions.items():
            if isinstance(value, list):
                placeholders = ','.join([f':filter_{column}_{i}' for i in range(len(value))])
                where_conditions.append(f"{column} IN ({placeholders})")
            else:
                where_conditions.append(f"{column} = :filter_{column}")
        
        # Build GROUP BY clause
        group_by_clause = ""
        if rule.group_by_columns:
            group_by_clause = f"GROUP BY {', '.join(rule.group_by_columns)}"
        
        # Construct full query
        sql = f"""
            SELECT {', '.join(select_columns)}
            FROM {rule.source_table}
            WHERE {' AND '.join(where_conditions)}
            {group_by_clause}
        """
        
        # Build parameters
        parameters = {
            "window_start": window_start,
            "window_end": window_end
        }
        
        # Add filter parameters
        for column, value in rule.filter_conditions.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    parameters[f"filter_{column}_{i}"] = v
            else:
                parameters[f"filter_{column}"] = value
        
        return {"sql": sql, "parameters": parameters}
    
    async def _cache_aggregation_results(
        self,
        metric_name: str,
        results: List[AggregationResult]
    ) -> None:
        """Cache aggregation results."""
        try:
            rule = self.aggregation_rules[metric_name]
            cache_ttl = self.cache_ttl.get(rule.time_unit, 3600)
            
            # Group results by time window for efficient caching
            for result in results:
                cache_key = f"agg:{metric_name}:{result.timestamp.isoformat()}"
                cache_value = json.dumps(asdict(result), default=str)
                await self.redis_client.setex(cache_key, cache_ttl, cache_value)
                
        except Exception as e:
            self.logger.warning(f"Failed to cache aggregation results: {e}")
    
    async def _get_precomputed_aggregations(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        time_unit: Optional[TimeUnit]
    ) -> List[AggregationResult]:
        """Get pre-computed aggregations from database."""
        # This would query pre-computed aggregation tables
        return []
    
    async def _get_detailed_data(
        self,
        rule: AggregationRule,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get detailed data for rolling calculations."""
        # This would fetch raw data for rolling window calculations
        return pd.DataFrame()
    
    async def _compute_rolling_window(
        self,
        data: pd.DataFrame,
        window_size: int,
        time_unit: TimeUnit,
        aggregation_type: AggregationType,
        rule: AggregationRule
    ) -> List[AggregationResult]:
        """Compute rolling window aggregations."""
        # This would implement rolling window calculations
        return []
    
    async def _compute_percentile_window(
        self,
        rule: AggregationRule,
        percentile: float,
        window_start: datetime,
        window_end: datetime
    ) -> List[AggregationResult]:
        """Compute percentile for a specific window."""
        # This would implement percentile calculations
        return []
    
    async def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        try:
            info = await self.redis_client.info("memory")
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception:
            return {}
    
    async def _get_performance_statistics(self) -> Dict[str, Any]:
        """Get aggregation performance statistics."""
        # This would track aggregation performance metrics
        return {
            "avg_aggregation_time": 0.0,
            "total_aggregations_computed": 0,
            "cache_hit_rate": 0.0
        }