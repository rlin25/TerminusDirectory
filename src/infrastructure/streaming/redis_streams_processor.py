"""
Redis Streams Processor for real-time event processing.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
import asyncio
import json
import logging
import redis.asyncio as aioredis
from redis.asyncio import Redis


@dataclass
class StreamMessage:
    """Represents a Redis Stream message."""
    stream_name: str
    message_id: str
    timestamp: datetime
    fields: Dict[str, str]
    consumer_group: Optional[str] = None
    consumer_name: Optional[str] = None


class RedisStreamsProcessor:
    """
    Redis Streams Processor for real-time event processing.
    
    Provides high-performance stream processing capabilities using Redis Streams
    with support for consumer groups, acknowledgments, and dead letter queues.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        consumer_group: str = "analytics_group",
        consumer_name: str = "analytics_consumer",
        block_timeout: int = 1000,
        max_messages: int = 100
    ):
        self.redis_client = redis_client
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.block_timeout = block_timeout
        self.max_messages = max_messages
        
        # Stream configurations
        self.active_streams: Dict[str, bool] = {}
        self.consumer_groups: Dict[str, str] = {}
        
        # Dead letter queue
        self.dlq_suffix = "_dlq"
        self.max_retries = 3
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the Redis Streams processor."""
        try:
            # Test Redis connection
            await self.redis_client.ping()
            self.logger.info("Redis Streams processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis Streams processor: {e}")
            raise
    
    async def create_stream(
        self,
        stream_name: str,
        consumer_group: Optional[str] = None,
        start_id: str = "0"
    ) -> bool:
        """Create a new stream with consumer group."""
        try:
            group_name = consumer_group or self.consumer_group
            
            # Create consumer group (will create stream if it doesn't exist)
            try:
                await self.redis_client.xgroup_create(
                    stream_name,
                    group_name,
                    start_id,
                    mkstream=True
                )
                self.logger.info(f"Created stream {stream_name} with group {group_name}")
            except aioredis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    self.logger.info(f"Consumer group {group_name} already exists for {stream_name}")
                else:
                    raise
            
            # Track stream and consumer group
            self.consumer_groups[stream_name] = group_name
            self.active_streams[stream_name] = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create stream {stream_name}: {e}")
            return False
    
    async def add_message(
        self,
        stream_name: str,
        fields: Dict[str, Any],
        message_id: str = "*",
        maxlen: Optional[int] = None
    ) -> str:
        """Add a message to a stream."""
        try:
            # Convert all field values to strings
            string_fields = {k: str(v) for k, v in fields.items()}
            
            # Add timestamp if not present
            if "timestamp" not in string_fields:
                string_fields["timestamp"] = datetime.utcnow().isoformat()
            
            # Add message to stream
            kwargs = {"maxlen": maxlen} if maxlen else {}
            message_id = await self.redis_client.xadd(
                stream_name,
                string_fields,
                id=message_id,
                **kwargs
            )
            
            self.logger.debug(f"Added message {message_id} to stream {stream_name}")
            return message_id
            
        except Exception as e:
            self.logger.error(f"Failed to add message to stream {stream_name}: {e}")
            raise
    
    async def consume_stream(
        self,
        stream_name: str,
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
        start_id: str = ">"
    ) -> AsyncGenerator[StreamMessage, None]:
        """Consume messages from a stream."""
        group_name = consumer_group or self.consumer_groups.get(stream_name, self.consumer_group)
        consumer = consumer_name or self.consumer_name
        
        # Ensure consumer group exists
        if stream_name not in self.consumer_groups:
            await self.create_stream(stream_name, group_name)
        
        try:
            while self.active_streams.get(stream_name, True):
                try:
                    # Read messages from stream
                    messages = await self.redis_client.xreadgroup(
                        group_name,
                        consumer,
                        {stream_name: start_id},
                        count=self.max_messages,
                        block=self.block_timeout
                    )
                    
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            # Parse timestamp
                            timestamp_str = fields.get("timestamp")
                            if timestamp_str:
                                try:
                                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                except:
                                    timestamp = datetime.utcnow()
                            else:
                                timestamp = datetime.utcnow()
                            
                            # Create StreamMessage object
                            stream_message = StreamMessage(
                                stream_name=stream_name,
                                message_id=message_id,
                                timestamp=timestamp,
                                fields=fields,
                                consumer_group=group_name,
                                consumer_name=consumer
                            )
                            
                            yield stream_message
                
                except aioredis.ResponseError as e:
                    if "NOGROUP" in str(e):
                        # Consumer group doesn't exist, create it
                        await self.create_stream(stream_name, group_name)
                        continue
                    else:
                        self.logger.error(f"Redis error consuming stream {stream_name}: {e}")
                        await asyncio.sleep(1)
                
                except Exception as e:
                    self.logger.error(f"Error consuming stream {stream_name}: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            self.logger.info(f"Stream consumption cancelled for {stream_name}")
        except Exception as e:
            self.logger.error(f"Fatal error in stream consumption for {stream_name}: {e}")
    
    async def acknowledge_message(
        self,
        stream_name: str,
        message_id: str,
        consumer_group: Optional[str] = None
    ) -> bool:
        """Acknowledge a processed message."""
        try:
            group_name = consumer_group or self.consumer_groups.get(stream_name, self.consumer_group)
            
            result = await self.redis_client.xack(stream_name, group_name, message_id)
            
            if result > 0:
                self.logger.debug(f"Acknowledged message {message_id} from stream {stream_name}")
                return True
            else:
                self.logger.warning(f"Failed to acknowledge message {message_id} from stream {stream_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error acknowledging message {message_id}: {e}")
            return False
    
    async def get_pending_messages(
        self,
        stream_name: str,
        consumer_group: Optional[str] = None,
        start_id: str = "-",
        end_id: str = "+",
        count: int = 100
    ) -> List[Tuple[str, str, int, List[str]]]:
        """Get pending messages that haven't been acknowledged."""
        try:
            group_name = consumer_group or self.consumer_groups.get(stream_name, self.consumer_group)
            
            pending_info = await self.redis_client.xpending_range(
                stream_name,
                group_name,
                start_id,
                end_id,
                count
            )
            
            return pending_info
            
        except Exception as e:
            self.logger.error(f"Error getting pending messages for stream {stream_name}: {e}")
            return []
    
    async def claim_pending_messages(
        self,
        stream_name: str,
        consumer_name: Optional[str] = None,
        min_idle_time: int = 60000,  # 1 minute
        message_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, Dict[str, str]]]:
        """Claim pending messages from other consumers."""
        try:
            consumer = consumer_name or self.consumer_name
            group_name = self.consumer_groups.get(stream_name, self.consumer_group)
            
            if message_ids:
                # Claim specific messages
                claimed = await self.redis_client.xclaim(
                    stream_name,
                    group_name,
                    consumer,
                    min_idle_time,
                    message_ids
                )
            else:
                # Auto-claim based on idle time
                claimed = await self.redis_client.xautoclaim(
                    stream_name,
                    group_name,
                    consumer,
                    min_idle_time,
                    start_id="0-0",
                    count=100
                )
            
            return claimed
            
        except Exception as e:
            self.logger.error(f"Error claiming pending messages for stream {stream_name}: {e}")
            return []
    
    async def move_to_dead_letter_queue(
        self,
        stream_name: str,
        message: StreamMessage,
        error_reason: str
    ) -> bool:
        """Move a failed message to dead letter queue."""
        try:
            dlq_stream = f"{stream_name}{self.dlq_suffix}"
            
            # Add error information to the message
            dlq_fields = message.fields.copy()
            dlq_fields.update({
                "original_stream": stream_name,
                "original_message_id": message.message_id,
                "error_reason": error_reason,
                "failed_at": datetime.utcnow().isoformat(),
                "retry_count": str(int(dlq_fields.get("retry_count", "0")) + 1)
            })
            
            # Add to DLQ
            dlq_message_id = await self.add_message(dlq_stream, dlq_fields)
            
            # Acknowledge original message
            await self.acknowledge_message(stream_name, message.message_id, message.consumer_group)
            
            self.logger.warning(
                f"Moved message {message.message_id} to DLQ {dlq_stream} "
                f"with ID {dlq_message_id}: {error_reason}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move message to DLQ: {e}")
            return False
    
    async def reprocess_dlq_messages(
        self,
        stream_name: str,
        max_messages: int = 100
    ) -> int:
        """Reprocess messages from dead letter queue."""
        try:
            dlq_stream = f"{stream_name}{self.dlq_suffix}"
            reprocessed_count = 0
            
            # Read messages from DLQ
            messages = await self.redis_client.xrange(dlq_stream, count=max_messages)
            
            for message_id, fields in messages:
                try:
                    retry_count = int(fields.get("retry_count", "0"))
                    
                    if retry_count < self.max_retries:
                        # Remove DLQ-specific fields
                        original_fields = {k: v for k, v in fields.items() 
                                         if k not in ["original_stream", "original_message_id", 
                                                    "error_reason", "failed_at", "retry_count"]}
                        
                        # Add back to original stream
                        await self.add_message(stream_name, original_fields)
                        
                        # Remove from DLQ
                        await self.redis_client.xdel(dlq_stream, message_id)
                        
                        reprocessed_count += 1
                        
                    else:
                        self.logger.warning(
                            f"Message {message_id} exceeded max retries ({self.max_retries})"
                        )
                
                except Exception as e:
                    self.logger.error(f"Failed to reprocess DLQ message {message_id}: {e}")
            
            if reprocessed_count > 0:
                self.logger.info(f"Reprocessed {reprocessed_count} messages from DLQ {dlq_stream}")
            
            return reprocessed_count
            
        except Exception as e:
            self.logger.error(f"Error reprocessing DLQ messages for stream {stream_name}: {e}")
            return 0
    
    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about a stream."""
        try:
            info = await self.redis_client.xinfo_stream(stream_name)
            
            # Get consumer group info
            groups_info = []
            try:
                groups = await self.redis_client.xinfo_groups(stream_name)
                for group in groups:
                    consumers = await self.redis_client.xinfo_consumers(stream_name, group["name"])
                    groups_info.append({
                        "name": group["name"],
                        "consumers": len(consumers),
                        "pending": group["pending"],
                        "last_delivered_id": group["last-delivered-id"]
                    })
            except Exception as e:
                self.logger.warning(f"Could not get group info for stream {stream_name}: {e}")
            
            return {
                "length": info["length"],
                "first_entry": info["first-entry"],
                "last_entry": info["last-entry"],
                "groups": groups_info,
                "dlq_length": await self._get_dlq_length(stream_name)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stream info for {stream_name}: {e}")
            return {}
    
    async def trim_stream(
        self,
        stream_name: str,
        maxlen: int,
        approximate: bool = True
    ) -> int:
        """Trim a stream to a maximum length."""
        try:
            if approximate:
                trimmed = await self.redis_client.xtrim(stream_name, maxlen=maxlen, approximate=True)
            else:
                trimmed = await self.redis_client.xtrim(stream_name, maxlen=maxlen)
            
            self.logger.info(f"Trimmed {trimmed} messages from stream {stream_name}")
            return trimmed
            
        except Exception as e:
            self.logger.error(f"Error trimming stream {stream_name}: {e}")
            return 0
    
    async def delete_stream(self, stream_name: str) -> bool:
        """Delete a stream and its consumer groups."""
        try:
            # Delete the stream
            deleted = await self.redis_client.delete(stream_name)
            
            # Delete DLQ if it exists
            dlq_stream = f"{stream_name}{self.dlq_suffix}"
            await self.redis_client.delete(dlq_stream)
            
            # Remove from tracking
            if stream_name in self.active_streams:
                del self.active_streams[stream_name]
            if stream_name in self.consumer_groups:
                del self.consumer_groups[stream_name]
            
            self.logger.info(f"Deleted stream {stream_name}")
            return deleted > 0
            
        except Exception as e:
            self.logger.error(f"Error deleting stream {stream_name}: {e}")
            return False
    
    async def stop_stream(self, stream_name: str) -> None:
        """Stop consuming from a stream."""
        self.active_streams[stream_name] = False
        self.logger.info(f"Stopped stream consumption for {stream_name}")
    
    async def get_all_streams_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active streams."""
        status = {}
        
        for stream_name in self.active_streams.keys():
            status[stream_name] = {
                "active": self.active_streams[stream_name],
                "consumer_group": self.consumer_groups.get(stream_name),
                "info": await self.get_stream_info(stream_name)
            }
        
        return status
    
    # Private helper methods
    async def _get_dlq_length(self, stream_name: str) -> int:
        """Get the length of the dead letter queue for a stream."""
        try:
            dlq_stream = f"{stream_name}{self.dlq_suffix}"
            info = await self.redis_client.xinfo_stream(dlq_stream)
            return info["length"]
        except:
            return 0