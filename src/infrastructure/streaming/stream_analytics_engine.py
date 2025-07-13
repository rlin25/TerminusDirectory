"""
Real-time Stream Analytics Engine for processing live data streams.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from .kafka_consumer import KafkaEventConsumer
from .kafka_producer import KafkaEventProducer
from .redis_streams_processor import RedisStreamsProcessor
from .real_time_aggregator import RealTimeAggregator
from .event_processor import EventProcessor
from .anomaly_detector import AnomalyDetector
from .alert_manager import AlertManager


class StreamType(Enum):
    """Types of data streams."""
    USER_EVENTS = "user_events"
    PROPERTY_EVENTS = "property_events"
    SYSTEM_METRICS = "system_metrics"
    ML_PREDICTIONS = "ml_predictions"
    BUSINESS_METRICS = "business_metrics"
    ERROR_LOGS = "error_logs"
    AUDIT_LOGS = "audit_logs"


class ProcessingMode(Enum):
    """Stream processing modes."""
    REAL_TIME = "real_time"         # Immediate processing
    MICRO_BATCH = "micro_batch"     # Small batch processing
    SLIDING_WINDOW = "sliding_window"  # Window-based processing
    SESSION_WINDOW = "session_window"  # Session-based processing


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    stream_name: str
    stream_type: StreamType
    processing_mode: ProcessingMode
    batch_size: int = 100
    window_size_seconds: int = 60
    session_timeout_seconds: int = 1800
    parallelism: int = 4
    checkpoint_interval: int = 10000
    enable_exactly_once: bool = True
    enable_backpressure: bool = True
    max_lag_threshold: int = 10000


@dataclass
class StreamEvent:
    """Represents a stream event."""
    event_id: str
    stream_name: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    partition_key: Optional[str] = None
    sequence_number: Optional[int] = None


@dataclass
class ProcessingResult:
    """Result of stream processing."""
    processed_events: int
    failed_events: int
    processing_time_ms: float
    throughput_per_second: float
    lag_ms: float
    errors: List[str]
    metrics: Dict[str, Any]


class StreamAnalyticsEngine:
    """
    Real-time Stream Analytics Engine.
    
    Processes live data streams for real-time analytics, anomaly detection,
    alerting, and business intelligence with high throughput and low latency.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        kafka_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 8
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.kafka_config = kafka_config or {}
        
        # Initialize components
        self.kafka_consumer = KafkaEventConsumer(kafka_config) if kafka_config else None
        self.kafka_producer = KafkaEventProducer(kafka_config) if kafka_config else None
        self.redis_processor = RedisStreamsProcessor(redis_client)
        self.real_time_aggregator = RealTimeAggregator(db_session, redis_client)
        self.event_processor = EventProcessor(db_session, redis_client)
        self.anomaly_detector = AnomalyDetector(db_session, redis_client)
        self.alert_manager = AlertManager(db_session, redis_client)
        
        # Stream configurations
        self.stream_configs: Dict[str, StreamConfig] = {}
        
        # Processing state
        self.active_streams: Dict[str, bool] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.processing_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.event_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Thread pool for CPU-intensive operations
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the stream analytics engine."""
        try:
            # Initialize all components
            if self.kafka_consumer:
                await self.kafka_consumer.initialize()
            if self.kafka_producer:
                await self.kafka_producer.initialize()
            
            await self.redis_processor.initialize()
            await self.real_time_aggregator.initialize()
            await self.event_processor.initialize()
            await self.anomaly_detector.initialize()
            await self.alert_manager.initialize()
            
            # Load stream configurations
            await self._load_stream_configurations()
            
            # Start monitoring
            await self._start_monitoring()
            
            self.logger.info("Stream analytics engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stream analytics engine: {e}")
            raise
    
    async def register_stream(self, config: StreamConfig) -> bool:
        """Register a new stream for processing."""
        try:
            # Validate configuration
            await self._validate_stream_config(config)
            
            # Store configuration
            self.stream_configs[config.stream_name] = config
            
            # Initialize stream state
            self.active_streams[config.stream_name] = False
            self.processing_stats[config.stream_name] = {
                "events_processed": 0,
                "events_failed": 0,
                "total_processing_time": 0.0,
                "avg_latency": 0.0,
                "last_processed": None
            }
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                "stream_configs",
                config.stream_name,
                json.dumps(asdict(config), default=str)
            )
            
            self.logger.info(f"Registered stream: {config.stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register stream {config.stream_name}: {e}")
            return False
    
    async def start_stream(self, stream_name: str) -> bool:
        """Start processing a stream."""
        if stream_name not in self.stream_configs:
            self.logger.error(f"Stream configuration not found: {stream_name}")
            return False
        
        if self.active_streams.get(stream_name, False):
            self.logger.warning(f"Stream already active: {stream_name}")
            return True
        
        try:
            config = self.stream_configs[stream_name]
            
            # Create processing task
            if config.processing_mode == ProcessingMode.REAL_TIME:
                task = asyncio.create_task(self._process_real_time_stream(config))
            elif config.processing_mode == ProcessingMode.MICRO_BATCH:
                task = asyncio.create_task(self._process_micro_batch_stream(config))
            elif config.processing_mode == ProcessingMode.SLIDING_WINDOW:
                task = asyncio.create_task(self._process_sliding_window_stream(config))
            elif config.processing_mode == ProcessingMode.SESSION_WINDOW:
                task = asyncio.create_task(self._process_session_window_stream(config))
            else:
                raise ValueError(f"Unsupported processing mode: {config.processing_mode}")
            
            self.stream_tasks[stream_name] = task
            self.active_streams[stream_name] = True
            
            self.logger.info(f"Started stream processing: {stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start stream {stream_name}: {e}")
            return False
    
    async def stop_stream(self, stream_name: str) -> bool:
        """Stop processing a stream."""
        try:
            if stream_name in self.stream_tasks:
                task = self.stream_tasks[stream_name]
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                del self.stream_tasks[stream_name]
            
            self.active_streams[stream_name] = False
            
            self.logger.info(f"Stopped stream processing: {stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop stream {stream_name}: {e}")
            return False
    
    async def process_event(
        self,
        stream_name: str,
        event: StreamEvent
    ) -> ProcessingResult:
        """Process a single stream event."""
        start_time = datetime.utcnow()
        
        try:
            # Add to event buffer
            self.event_buffers[stream_name].append(event)
            
            # Process event through pipeline
            processing_results = []
            
            # 1. Event processing and enrichment
            enriched_event = await self.event_processor.process_event(event)
            processing_results.append(enriched_event)
            
            # 2. Real-time aggregation
            if enriched_event["status"] == "success":
                agg_result = await self.real_time_aggregator.update_aggregations(
                    enriched_event["event"]
                )
                processing_results.append(agg_result)
            
            # 3. Anomaly detection
            if enriched_event["status"] == "success":
                anomaly_result = await self.anomaly_detector.detect_anomalies(
                    enriched_event["event"]
                )
                processing_results.append(anomaly_result)
                
                # 4. Alert management if anomaly detected
                if anomaly_result.get("anomaly_detected", False):
                    alert_result = await self.alert_manager.process_anomaly(
                        anomaly_result
                    )
                    processing_results.append(alert_result)
            
            # Calculate processing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update stream statistics
            await self._update_stream_statistics(
                stream_name,
                1,  # processed_events
                0,  # failed_events
                processing_time
            )
            
            return ProcessingResult(
                processed_events=1,
                failed_events=0,
                processing_time_ms=processing_time,
                throughput_per_second=1000.0 / processing_time if processing_time > 0 else 0,
                lag_ms=0.0,
                errors=[],
                metrics={
                    "enrichment_status": enriched_event["status"],
                    "aggregation_updates": agg_result.get("updates", 0) if 'agg_result' in locals() else 0,
                    "anomaly_score": anomaly_result.get("anomaly_score", 0.0) if 'anomaly_result' in locals() else 0.0
                }
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update statistics with failure
            await self._update_stream_statistics(
                stream_name,
                0,  # processed_events
                1,  # failed_events
                processing_time
            )
            
            self.logger.error(f"Failed to process event for stream {stream_name}: {e}")
            
            return ProcessingResult(
                processed_events=0,
                failed_events=1,
                processing_time_ms=processing_time,
                throughput_per_second=0.0,
                lag_ms=0.0,
                errors=[str(e)],
                metrics={}
            )
    
    async def get_stream_status(self, stream_name: str) -> Dict[str, Any]:
        """Get current status of a stream."""
        if stream_name not in self.stream_configs:
            return {"error": f"Stream not found: {stream_name}"}
        
        config = self.stream_configs[stream_name]
        stats = self.processing_stats[stream_name]
        is_active = self.active_streams.get(stream_name, False)
        
        # Get real-time metrics
        recent_events = len(self.event_buffers[stream_name])
        
        # Calculate throughput
        current_time = datetime.utcnow()
        if stats.get("last_processed"):
            time_diff = (current_time - stats["last_processed"]).total_seconds()
            throughput = stats["events_processed"] / max(time_diff, 1.0)
        else:
            throughput = 0.0
        
        return {
            "stream_name": stream_name,
            "stream_type": config.stream_type.value,
            "processing_mode": config.processing_mode.value,
            "is_active": is_active,
            "configuration": asdict(config),
            "statistics": {
                **stats,
                "current_throughput": throughput,
                "buffer_size": recent_events,
                "success_rate": self._calculate_success_rate(stats)
            },
            "health_status": await self._get_stream_health(stream_name)
        }
    
    async def get_all_streams_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered streams."""
        status = {}
        for stream_name in self.stream_configs.keys():
            status[stream_name] = await self.get_stream_status(stream_name)
        return status
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time analytics metrics."""
        try:
            # Aggregate metrics across all streams
            total_processed = sum(
                stats["events_processed"] 
                for stats in self.processing_stats.values()
            )
            
            total_failed = sum(
                stats["events_failed"] 
                for stats in self.processing_stats.values()
            )
            
            active_streams_count = sum(1 for active in self.active_streams.values() if active)
            
            # Get recent performance metrics
            recent_metrics = await self._get_recent_performance_metrics()
            
            return {
                "summary": {
                    "total_events_processed": total_processed,
                    "total_events_failed": total_failed,
                    "active_streams": active_streams_count,
                    "total_streams": len(self.stream_configs),
                    "overall_success_rate": (total_processed / max(total_processed + total_failed, 1)) * 100
                },
                "performance": recent_metrics,
                "stream_health": await self._get_overall_stream_health(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time metrics: {e}")
            return {"error": str(e)}
    
    # Private methods for stream processing
    async def _process_real_time_stream(self, config: StreamConfig) -> None:
        """Process stream in real-time mode."""
        self.logger.info(f"Starting real-time processing for {config.stream_name}")
        
        try:
            if config.stream_type in [StreamType.USER_EVENTS, StreamType.PROPERTY_EVENTS]:
                # Use Kafka for high-volume event streams
                if self.kafka_consumer:
                    async for event in self.kafka_consumer.consume_events(config.stream_name):
                        if not self.active_streams.get(config.stream_name, False):
                            break
                        await self.process_event(config.stream_name, event)
            else:
                # Use Redis Streams for other types
                async for event in self.redis_processor.consume_stream(config.stream_name):
                    if not self.active_streams.get(config.stream_name, False):
                        break
                    await self.process_event(config.stream_name, event)
                    
        except asyncio.CancelledError:
            self.logger.info(f"Real-time processing cancelled for {config.stream_name}")
        except Exception as e:
            self.logger.error(f"Error in real-time processing for {config.stream_name}: {e}")
    
    async def _process_micro_batch_stream(self, config: StreamConfig) -> None:
        """Process stream in micro-batch mode."""
        self.logger.info(f"Starting micro-batch processing for {config.stream_name}")
        
        try:
            batch_events = []
            
            if config.stream_type in [StreamType.USER_EVENTS, StreamType.PROPERTY_EVENTS]:
                # Kafka-based micro-batching
                if self.kafka_consumer:
                    async for event in self.kafka_consumer.consume_events(config.stream_name):
                        if not self.active_streams.get(config.stream_name, False):
                            break
                            
                        batch_events.append(event)
                        
                        if len(batch_events) >= config.batch_size:
                            await self._process_event_batch(config.stream_name, batch_events)
                            batch_events = []
            else:
                # Redis Streams micro-batching
                async for event in self.redis_processor.consume_stream(config.stream_name):
                    if not self.active_streams.get(config.stream_name, False):
                        break
                        
                    batch_events.append(event)
                    
                    if len(batch_events) >= config.batch_size:
                        await self._process_event_batch(config.stream_name, batch_events)
                        batch_events = []
            
            # Process remaining events
            if batch_events:
                await self._process_event_batch(config.stream_name, batch_events)
                
        except asyncio.CancelledError:
            self.logger.info(f"Micro-batch processing cancelled for {config.stream_name}")
        except Exception as e:
            self.logger.error(f"Error in micro-batch processing for {config.stream_name}: {e}")
    
    async def _process_sliding_window_stream(self, config: StreamConfig) -> None:
        """Process stream with sliding window."""
        self.logger.info(f"Starting sliding window processing for {config.stream_name}")
        
        window_events = deque()
        window_start = datetime.utcnow()
        
        try:
            # Implementation for sliding window processing
            # This would maintain a time-based window of events
            pass
            
        except asyncio.CancelledError:
            self.logger.info(f"Sliding window processing cancelled for {config.stream_name}")
        except Exception as e:
            self.logger.error(f"Error in sliding window processing for {config.stream_name}: {e}")
    
    async def _process_session_window_stream(self, config: StreamConfig) -> None:
        """Process stream with session windows."""
        self.logger.info(f"Starting session window processing for {config.stream_name}")
        
        sessions = {}  # user_id -> session_events
        
        try:
            # Implementation for session-based processing
            # This would group events by user sessions
            pass
            
        except asyncio.CancelledError:
            self.logger.info(f"Session window processing cancelled for {config.stream_name}")
        except Exception as e:
            self.logger.error(f"Error in session window processing for {config.stream_name}: {e}")
    
    async def _process_event_batch(
        self,
        stream_name: str,
        events: List[StreamEvent]
    ) -> None:
        """Process a batch of events."""
        start_time = datetime.utcnow()
        
        try:
            # Process events in parallel
            tasks = []
            for event in events:
                task = asyncio.create_task(self.process_event(stream_name, event))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            total_processed = 0
            total_failed = 0
            total_processing_time = 0.0
            
            for result in results:
                if isinstance(result, ProcessingResult):
                    total_processed += result.processed_events
                    total_failed += result.failed_events
                    total_processing_time += result.processing_time_ms
                else:
                    total_failed += 1
            
            batch_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"Processed batch for {stream_name}: "
                f"{total_processed} successful, {total_failed} failed, "
                f"{batch_time:.2f}ms total time"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process event batch for {stream_name}: {e}")
    
    # Helper methods
    async def _validate_stream_config(self, config: StreamConfig) -> None:
        """Validate stream configuration."""
        if not config.stream_name:
            raise ValueError("Stream name is required")
        
        if config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.window_size_seconds <= 0:
            raise ValueError("Window size must be positive")
    
    async def _load_stream_configurations(self) -> None:
        """Load stream configurations from Redis."""
        try:
            configs = await self.redis_client.hgetall("stream_configs")
            for stream_name, config_json in configs.items():
                try:
                    config_dict = json.loads(config_json)
                    config = StreamConfig(**config_dict)
                    self.stream_configs[stream_name] = config
                    self.active_streams[stream_name] = False
                except Exception as e:
                    self.logger.warning(f"Failed to load config for {stream_name}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load stream configurations: {e}")
    
    async def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        # This would start tasks for health monitoring, metrics collection, etc.
        pass
    
    async def _update_stream_statistics(
        self,
        stream_name: str,
        processed: int,
        failed: int,
        processing_time: float
    ) -> None:
        """Update stream processing statistics."""
        stats = self.processing_stats[stream_name]
        stats["events_processed"] += processed
        stats["events_failed"] += failed
        stats["total_processing_time"] += processing_time
        stats["last_processed"] = datetime.utcnow()
        
        # Calculate average latency
        total_events = stats["events_processed"] + stats["events_failed"]
        if total_events > 0:
            stats["avg_latency"] = stats["total_processing_time"] / total_events
    
    def _calculate_success_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate success rate for a stream."""
        total = stats["events_processed"] + stats["events_failed"]
        if total == 0:
            return 100.0
        return (stats["events_processed"] / total) * 100.0
    
    async def _get_stream_health(self, stream_name: str) -> str:
        """Get health status of a stream."""
        stats = self.processing_stats[stream_name]
        success_rate = self._calculate_success_rate(stats)
        
        if success_rate >= 99.0:
            return "healthy"
        elif success_rate >= 95.0:
            return "warning"
        else:
            return "critical"
    
    async def _get_recent_performance_metrics(self) -> Dict[str, Any]:
        """Get recent performance metrics."""
        # This would calculate recent throughput, latency, etc.
        return {
            "avg_throughput_per_second": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "error_rate": 0.0
        }
    
    async def _get_overall_stream_health(self) -> str:
        """Get overall health of all streams."""
        if not self.stream_configs:
            return "unknown"
        
        health_scores = []
        for stream_name in self.stream_configs.keys():
            health = await self._get_stream_health(stream_name)
            if health == "healthy":
                health_scores.append(3)
            elif health == "warning":
                health_scores.append(2)
            elif health == "critical":
                health_scores.append(1)
            else:
                health_scores.append(0)
        
        avg_health = sum(health_scores) / len(health_scores)
        
        if avg_health >= 2.8:
            return "healthy"
        elif avg_health >= 2.0:
            return "warning"
        else:
            return "critical"