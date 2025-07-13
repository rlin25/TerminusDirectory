"""
Real-time Streaming Analytics Infrastructure.

This package provides real-time data processing capabilities using Kafka and Redis Streams
for live analytics, event processing, and instant insights.
"""

from .kafka_consumer import KafkaEventConsumer
from .kafka_producer import KafkaEventProducer
from .redis_streams_processor import RedisStreamsProcessor
from .stream_analytics_engine import StreamAnalyticsEngine
from .real_time_aggregator import RealTimeAggregator
from .event_processor import EventProcessor
from .anomaly_detector import AnomalyDetector
from .alert_manager import AlertManager

__all__ = [
    "KafkaEventConsumer",
    "KafkaEventProducer", 
    "RedisStreamsProcessor",
    "StreamAnalyticsEngine",
    "RealTimeAggregator",
    "EventProcessor",
    "AnomalyDetector",
    "AlertManager"
]