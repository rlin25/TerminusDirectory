"""
Real-time Aggregator for live data processing and metrics calculation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncio
import json
import logging
import numpy as np
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


@dataclass
class AggregationWindow:
    """Represents a time window for aggregation."""
    start_time: datetime
    end_time: datetime
    window_size_seconds: int
    aggregation_type: str
    metric_name: str
    value: float
    count: int
    dimensions: Dict[str, str]


@dataclass
class MetricDefinition:
    """Definition of a real-time metric."""
    name: str
    aggregation_type: str  # sum, count, avg, min, max, percentile
    source_field: str
    time_windows: List[int]  # Window sizes in seconds
    dimensions: List[str]    # Fields to group by
    filter_conditions: Dict[str, Any]
    percentile_value: Optional[float] = None


class RealTimeAggregator:
    """
    Real-time Aggregator for live analytics data processing.
    
    Provides low-latency aggregation of streaming data with support
    for sliding windows, multiple time granularities, and real-time updates.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        max_window_size: int = 3600  # 1 hour max window
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.max_window_size = max_window_size
        
        # Metric definitions
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # In-memory data structures for real-time aggregation
        self.sliding_windows: Dict[str, Dict[int, deque]] = defaultdict(lambda: defaultdict(deque))
        self.current_aggregations: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance tracking
        self.aggregation_stats = {
            "total_events_processed": 0,
            "total_aggregations_updated": 0,
            "avg_processing_time_ms": 0.0
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the real-time aggregator."""
        try:
            # Load metric definitions from Redis/database
            await self._load_metric_definitions()
            
            # Initialize sliding windows
            await self._initialize_sliding_windows()
            
            # Start background tasks
            asyncio.create_task(self._background_aggregation_flush())
            asyncio.create_task(self._background_cleanup())
            
            self.logger.info("Real-time aggregator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize real-time aggregator: {e}")
            raise
    
    async def register_metric(self, metric_def: MetricDefinition) -> bool:
        """Register a new metric for real-time aggregation."""
        try:
            # Validate metric definition
            self._validate_metric_definition(metric_def)
            
            # Store metric definition
            self.metric_definitions[metric_def.name] = metric_def
            
            # Initialize sliding windows for this metric
            await self._initialize_metric_windows(metric_def)
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                "realtime_metrics",
                metric_def.name,
                json.dumps(asdict(metric_def), default=str)
            )
            
            self.logger.info(f"Registered real-time metric: {metric_def.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register metric {metric_def.name}: {e}")
            return False
    
    async def update_aggregations(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Update aggregations based on an incoming event."""
        start_time = datetime.utcnow()
        updated_metrics = []
        
        try:
            event_timestamp = self._parse_event_timestamp(event)
            
            # Process each registered metric
            for metric_name, metric_def in self.metric_definitions.items():
                try:
                    # Check if event matches metric filters
                    if self._event_matches_filters(event, metric_def.filter_conditions):
                        # Extract metric value from event
                        metric_value = self._extract_metric_value(event, metric_def)
                        
                        if metric_value is not None:
                            # Update aggregations for all time windows
                            for window_size in metric_def.time_windows:
                                await self._update_window_aggregation(
                                    metric_name,
                                    metric_def,
                                    metric_value,
                                    event_timestamp,
                                    window_size,
                                    event
                                )
                            
                            updated_metrics.append(metric_name)
                
                except Exception as e:
                    self.logger.error(f"Error updating metric {metric_name}: {e}")
            
            # Update performance stats
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.aggregation_stats["total_events_processed"] += 1
            self.aggregation_stats["total_aggregations_updated"] += len(updated_metrics)
            
            # Update average processing time
            current_avg = self.aggregation_stats["avg_processing_time_ms"]
            total_events = self.aggregation_stats["total_events_processed"]
            self.aggregation_stats["avg_processing_time_ms"] = (
                (current_avg * (total_events - 1) + processing_time) / total_events
            )
            
            return {
                "status": "success",
                "updated_metrics": updated_metrics,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update aggregations: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    async def get_real_time_metric(
        self,
        metric_name: str,
        window_size: int,
        dimensions: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get current aggregated value for a metric."""
        try:
            if metric_name not in self.metric_definitions:
                return None
            
            metric_def = self.metric_definitions[metric_name]
            
            # Build cache key
            dimension_key = self._build_dimension_key(dimensions or {})
            cache_key = f"realtime:{metric_name}:{window_size}:{dimension_key}"
            
            # Try to get from Redis cache first
            cached_value = await self.redis_client.get(cache_key)
            if cached_value:
                try:
                    return json.loads(cached_value)
                except:
                    pass
            
            # Calculate from sliding window
            window_data = self.sliding_windows[metric_name][window_size]
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(seconds=window_size)
            
            # Filter data within window and matching dimensions
            filtered_data = [
                data_point for data_point in window_data
                if (data_point["timestamp"] >= window_start and 
                    self._dimensions_match(data_point.get("dimensions", {}), dimensions or {}))
            ]
            
            if not filtered_data:
                return {
                    "metric_name": metric_name,
                    "window_size": window_size,
                    "value": 0.0,
                    "count": 0,
                    "timestamp": current_time.isoformat(),
                    "dimensions": dimensions or {}
                }
            
            # Calculate aggregation based on type
            aggregated_value = self._calculate_aggregation(
                filtered_data,
                metric_def.aggregation_type,
                metric_def.percentile_value
            )
            
            result = {
                "metric_name": metric_name,
                "window_size": window_size,
                "value": aggregated_value,
                "count": len(filtered_data),
                "timestamp": current_time.isoformat(),
                "dimensions": dimensions or {}
            }
            
            # Cache the result
            await self.redis_client.setex(
                cache_key,
                30,  # 30 seconds cache
                json.dumps(result, default=str)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting real-time metric {metric_name}: {e}")
            return None
    
    async def get_multiple_metrics(
        self,
        metric_names: List[str],
        window_size: int,
        dimensions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get multiple real-time metrics efficiently."""
        results = {}
        
        # Process metrics in parallel
        tasks = []
        for metric_name in metric_names:
            task = self.get_real_time_metric(metric_name, window_size, dimensions)
            tasks.append((metric_name, task))
        
        # Execute all tasks
        for metric_name, task in tasks:
            try:
                result = await task
                results[metric_name] = result
            except Exception as e:
                self.logger.error(f"Error getting metric {metric_name}: {e}")
                results[metric_name] = None
        
        return results
    
    async def get_metric_trend(
        self,
        metric_name: str,
        window_size: int,
        points: int = 10,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Get trend data for a metric over multiple time points."""
        try:
            trend_data = []
            current_time = datetime.utcnow()
            
            # Calculate time intervals
            interval = window_size // points
            
            for i in range(points):
                point_time = current_time - timedelta(seconds=i * interval)
                
                # Get data point for this time
                data_point = await self._get_metric_at_time(
                    metric_name,
                    window_size,
                    point_time,
                    dimensions
                )
                
                if data_point:
                    trend_data.append(data_point)
            
            # Reverse to get chronological order
            trend_data.reverse()
            
            return trend_data
            
        except Exception as e:
            self.logger.error(f"Error getting metric trend for {metric_name}: {e}")
            return []
    
    async def get_top_dimensions(
        self,
        metric_name: str,
        window_size: int,
        dimension_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top dimension values for a metric."""
        try:
            if metric_name not in self.metric_definitions:
                return []
            
            metric_def = self.metric_definitions[metric_name]
            window_data = self.sliding_windows[metric_name][window_size]
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(seconds=window_size)
            
            # Group by dimension value
            dimension_aggregations = defaultdict(list)
            
            for data_point in window_data:
                if (data_point["timestamp"] >= window_start and 
                    dimension_name in data_point.get("dimensions", {})):
                    
                    dimension_value = data_point["dimensions"][dimension_name]
                    dimension_aggregations[dimension_value].append(data_point["value"])
            
            # Calculate aggregations for each dimension value
            top_dimensions = []
            for dim_value, values in dimension_aggregations.items():
                aggregated_value = self._calculate_aggregation(
                    [{"value": v} for v in values],
                    metric_def.aggregation_type,
                    metric_def.percentile_value
                )
                
                top_dimensions.append({
                    "dimension_value": dim_value,
                    "value": aggregated_value,
                    "count": len(values)
                })
            
            # Sort by value and return top N
            top_dimensions.sort(key=lambda x: x["value"], reverse=True)
            return top_dimensions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting top dimensions for {metric_name}: {e}")
            return []
    
    async def get_aggregator_status(self) -> Dict[str, Any]:
        """Get status and performance metrics of the aggregator."""
        try:
            # Calculate memory usage
            total_data_points = sum(
                sum(len(window_data) for window_data in metric_windows.values())
                for metric_windows in self.sliding_windows.values()
            )
            
            # Get Redis memory usage
            redis_info = await self.redis_client.info("memory")
            
            return {
                "status": "active",
                "registered_metrics": len(self.metric_definitions),
                "total_data_points": total_data_points,
                "performance_stats": self.aggregation_stats,
                "memory_usage": {
                    "redis_used_memory": redis_info.get("used_memory_human", "0B"),
                    "in_memory_data_points": total_data_points
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting aggregator status: {e}")
            return {"status": "error", "error": str(e)}
    
    # Private methods
    def _validate_metric_definition(self, metric_def: MetricDefinition) -> None:
        """Validate metric definition."""
        if not metric_def.name:
            raise ValueError("Metric name is required")
        
        if metric_def.aggregation_type not in ["sum", "count", "avg", "min", "max", "percentile"]:
            raise ValueError("Invalid aggregation type")
        
        if metric_def.aggregation_type == "percentile" and metric_def.percentile_value is None:
            raise ValueError("Percentile value required for percentile aggregation")
        
        if any(window > self.max_window_size for window in metric_def.time_windows):
            raise ValueError(f"Window size cannot exceed {self.max_window_size} seconds")
    
    async def _load_metric_definitions(self) -> None:
        """Load metric definitions from persistence."""
        try:
            metrics = await self.redis_client.hgetall("realtime_metrics")
            for metric_name, metric_json in metrics.items():
                try:
                    metric_dict = json.loads(metric_json)
                    metric_def = MetricDefinition(**metric_dict)
                    self.metric_definitions[metric_name] = metric_def
                except Exception as e:
                    self.logger.warning(f"Failed to load metric {metric_name}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load metric definitions: {e}")
    
    async def _initialize_sliding_windows(self) -> None:
        """Initialize sliding window data structures."""
        for metric_name, metric_def in self.metric_definitions.items():
            await self._initialize_metric_windows(metric_def)
    
    async def _initialize_metric_windows(self, metric_def: MetricDefinition) -> None:
        """Initialize sliding windows for a specific metric."""
        for window_size in metric_def.time_windows:
            self.sliding_windows[metric_def.name][window_size] = deque(maxlen=window_size * 10)
    
    def _parse_event_timestamp(self, event: Dict[str, Any]) -> datetime:
        """Parse timestamp from event."""
        timestamp_field = event.get("timestamp")
        if timestamp_field:
            try:
                if isinstance(timestamp_field, str):
                    return datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                elif isinstance(timestamp_field, datetime):
                    return timestamp_field
            except:
                pass
        return datetime.utcnow()
    
    def _event_matches_filters(
        self,
        event: Dict[str, Any],
        filter_conditions: Dict[str, Any]
    ) -> bool:
        """Check if event matches filter conditions."""
        for field, expected_value in filter_conditions.items():
            event_value = event.get(field)
            
            if isinstance(expected_value, list):
                if event_value not in expected_value:
                    return False
            else:
                if event_value != expected_value:
                    return False
        
        return True
    
    def _extract_metric_value(
        self,
        event: Dict[str, Any],
        metric_def: MetricDefinition
    ) -> Optional[float]:
        """Extract metric value from event."""
        try:
            if metric_def.aggregation_type == "count":
                return 1.0
            
            value = event.get(metric_def.source_field)
            if value is not None:
                return float(value)
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    async def _update_window_aggregation(
        self,
        metric_name: str,
        metric_def: MetricDefinition,
        value: float,
        timestamp: datetime,
        window_size: int,
        event: Dict[str, Any]
    ) -> None:
        """Update aggregation for a specific window."""
        # Extract dimensions
        dimensions = {}
        for dim_field in metric_def.dimensions:
            if dim_field in event:
                dimensions[dim_field] = str(event[dim_field])
        
        # Add data point to sliding window
        data_point = {
            "value": value,
            "timestamp": timestamp,
            "dimensions": dimensions
        }
        
        self.sliding_windows[metric_name][window_size].append(data_point)
        
        # Clean up old data points
        cutoff_time = datetime.utcnow() - timedelta(seconds=window_size)
        window_data = self.sliding_windows[metric_name][window_size]
        
        while window_data and window_data[0]["timestamp"] < cutoff_time:
            window_data.popleft()
    
    def _build_dimension_key(self, dimensions: Dict[str, str]) -> str:
        """Build a string key from dimensions."""
        if not dimensions:
            return "all"
        
        sorted_items = sorted(dimensions.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def _dimensions_match(
        self,
        data_dimensions: Dict[str, str],
        filter_dimensions: Dict[str, str]
    ) -> bool:
        """Check if data dimensions match filter dimensions."""
        for key, value in filter_dimensions.items():
            if data_dimensions.get(key) != value:
                return False
        return True
    
    def _calculate_aggregation(
        self,
        data_points: List[Dict[str, Any]],
        aggregation_type: str,
        percentile_value: Optional[float] = None
    ) -> float:
        """Calculate aggregation from data points."""
        if not data_points:
            return 0.0
        
        values = [point["value"] for point in data_points]
        
        if aggregation_type == "sum":
            return sum(values)
        elif aggregation_type == "count":
            return len(values)
        elif aggregation_type == "avg":
            return sum(values) / len(values)
        elif aggregation_type == "min":
            return min(values)
        elif aggregation_type == "max":
            return max(values)
        elif aggregation_type == "percentile" and percentile_value is not None:
            return np.percentile(values, percentile_value)
        else:
            return 0.0
    
    async def _get_metric_at_time(
        self,
        metric_name: str,
        window_size: int,
        target_time: datetime,
        dimensions: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metric value at a specific time."""
        # This would implement historical point-in-time queries
        # For now, return current metric (simplified)
        return await self.get_real_time_metric(metric_name, window_size, dimensions)
    
    async def _background_aggregation_flush(self) -> None:
        """Background task to flush aggregations to persistent storage."""
        while True:
            try:
                await asyncio.sleep(60)  # Flush every minute
                
                # Flush current aggregations to database/cache
                for metric_name in self.metric_definitions.keys():
                    for window_size in self.metric_definitions[metric_name].time_windows:
                        # Get current aggregation and store it
                        current_agg = await self.get_real_time_metric(metric_name, window_size)
                        if current_agg:
                            await self._store_aggregation_snapshot(current_agg)
                
            except Exception as e:
                self.logger.error(f"Error in background aggregation flush: {e}")
    
    async def _background_cleanup(self) -> None:
        """Background task to clean up old data."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                # Clean up old data points from sliding windows
                current_time = datetime.utcnow()
                
                for metric_name, metric_windows in self.sliding_windows.items():
                    for window_size, window_data in metric_windows.items():
                        cutoff_time = current_time - timedelta(seconds=window_size)
                        
                        while window_data and window_data[0]["timestamp"] < cutoff_time:
                            window_data.popleft()
                
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {e}")
    
    async def _store_aggregation_snapshot(self, aggregation: Dict[str, Any]) -> None:
        """Store aggregation snapshot to persistent storage."""
        try:
            # Store to database for historical analysis
            query = text("""
                INSERT INTO business_metrics (
                    metric_name, metric_value, timestamp, dimensions, source
                ) VALUES (
                    :metric_name, :value, :timestamp, :dimensions, 'realtime_aggregator'
                )
            """)
            
            await self.db_session.execute(query, {
                "metric_name": aggregation["metric_name"],
                "value": aggregation["value"],
                "timestamp": datetime.fromisoformat(aggregation["timestamp"].replace('Z', '+00:00')),
                "dimensions": json.dumps(aggregation["dimensions"])
            })
            
            await self.db_session.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing aggregation snapshot: {e}")
            await self.db_session.rollback()