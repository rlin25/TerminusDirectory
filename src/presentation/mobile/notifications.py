"""
Push notification service for mobile applications.

This module provides push notification capabilities for iOS and Android devices,
including property alerts, recommendation updates, and system notifications.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

import aiohttp
import jwt
from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Push notification types"""
    PROPERTY_ALERT = "property_alert"
    RECOMMENDATION = "recommendation"
    PRICE_CHANGE = "price_change"
    NEW_PROPERTY = "new_property"
    SAVED_SEARCH = "saved_search"
    SYSTEM = "system"
    MARKETING = "marketing"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class DevicePlatform(str, Enum):
    """Device platforms"""
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"


class NotificationPayload(BaseModel):
    """Push notification payload"""
    title: str = Field(..., description="Notification title")
    body: str = Field(..., description="Notification body")
    data: Dict[str, Any] = Field(default={}, description="Custom data")
    image_url: Optional[str] = Field(None, description="Notification image URL")
    action_url: Optional[str] = Field(None, description="Deep link URL")
    category: Optional[str] = Field(None, description="Notification category")
    sound: Optional[str] = Field("default", description="Notification sound")
    badge_count: Optional[int] = Field(None, description="Badge count for iOS")


class NotificationRequest(BaseModel):
    """Push notification request"""
    user_id: str = Field(..., description="Target user ID")
    notification_type: NotificationType = Field(..., description="Notification type")
    priority: NotificationPriority = Field(default=NotificationPriority.NORMAL)
    payload: NotificationPayload = Field(..., description="Notification payload")
    device_tokens: Optional[List[str]] = Field(None, description="Specific device tokens")
    schedule_time: Optional[datetime] = Field(None, description="Scheduled delivery time")
    expire_time: Optional[datetime] = Field(None, description="Notification expiry")
    collapse_key: Optional[str] = Field(None, description="Message collapse key")


class NotificationResult(BaseModel):
    """Push notification result"""
    notification_id: str = Field(..., description="Notification ID")
    user_id: str = Field(..., description="Target user ID")
    success_count: int = Field(default=0, description="Successful deliveries")
    failure_count: int = Field(default=0, description="Failed deliveries")
    device_results: Dict[str, Dict[str, Any]] = Field(default={}, description="Per-device results")
    sent_at: datetime = Field(default_factory=datetime.utcnow)


class DeviceToken(BaseModel):
    """Device token information"""
    token: str = Field(..., description="Device token")
    platform: DevicePlatform = Field(..., description="Device platform")
    app_version: str = Field(..., description="App version")
    os_version: str = Field(..., description="OS version")
    active: bool = Field(default=True, description="Token is active")
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: datetime = Field(default_factory=datetime.utcnow)


class NotificationPreferences(BaseModel):
    """User notification preferences"""
    user_id: str = Field(..., description="User ID")
    enabled: bool = Field(default=True, description="Notifications enabled")
    property_alerts: bool = Field(default=True, description="Property alerts enabled")
    recommendations: bool = Field(default=True, description="Recommendations enabled")
    price_changes: bool = Field(default=True, description="Price change alerts")
    marketing: bool = Field(default=False, description="Marketing notifications")
    quiet_hours_start: Optional[str] = Field(None, description="Quiet hours start (HH:MM)")
    quiet_hours_end: Optional[str] = Field(None, description="Quiet hours end (HH:MM)")
    timezone: str = Field(default="UTC", description="User timezone")


class PushNotificationService:
    """Push notification service for mobile applications"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.fcm_server_key: Optional[str] = None
        self.apns_key_id: Optional[str] = None
        self.apns_team_id: Optional[str] = None
        self.apns_private_key: Optional[str] = None
        self.web_push_private_key: Optional[str] = None
        self.web_push_public_key: Optional[str] = None
        
    async def initialize(
        self,
        redis_url: str = "redis://localhost:6379",
        fcm_server_key: Optional[str] = None,
        apns_config: Optional[Dict[str, str]] = None,
        web_push_config: Optional[Dict[str, str]] = None
    ):
        """Initialize the push notification service"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            
            # Set FCM configuration
            self.fcm_server_key = fcm_server_key
            
            # Set APNs configuration
            if apns_config:
                self.apns_key_id = apns_config.get("key_id")
                self.apns_team_id = apns_config.get("team_id")
                self.apns_private_key = apns_config.get("private_key")
            
            # Set Web Push configuration
            if web_push_config:
                self.web_push_private_key = web_push_config.get("private_key")
                self.web_push_public_key = web_push_config.get("public_key")
            
            logger.info("Push notification service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize push service: {e}")
            raise
    
    async def register_device(
        self,
        user_id: str,
        device_token: str,
        device_type: str,
        app_version: str = "1.0.0",
        os_version: str = "unknown"
    ) -> bool:
        """Register a device for push notifications"""
        try:
            platform = DevicePlatform(device_type.lower())
            
            device_info = DeviceToken(
                token=device_token,
                platform=platform,
                app_version=app_version,
                os_version=os_version
            )
            
            # Store device token
            token_key = f"device_token:{user_id}:{device_token}"
            await self.redis_client.setex(
                token_key,
                86400 * 365,  # 1 year
                device_info.model_dump_json()
            )
            
            # Add to user's device list
            user_tokens_key = f"user_tokens:{user_id}"
            await self.redis_client.sadd(user_tokens_key, device_token)
            await self.redis_client.expire(user_tokens_key, 86400 * 365)
            
            logger.info(f"Device registered: {user_id} -> {device_token[:20]}...")
            return True
            
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            return False
    
    async def unregister_device(self, user_id: str, device_token: str) -> bool:
        """Unregister a device from push notifications"""
        try:
            # Remove device token
            token_key = f"device_token:{user_id}:{device_token}"
            await self.redis_client.delete(token_key)
            
            # Remove from user's device list
            user_tokens_key = f"user_tokens:{user_id}"
            await self.redis_client.srem(user_tokens_key, device_token)
            
            logger.info(f"Device unregistered: {user_id} -> {device_token[:20]}...")
            return True
            
        except Exception as e:
            logger.error(f"Device unregistration failed: {e}")
            return False
    
    async def get_user_devices(self, user_id: str) -> List[DeviceToken]:
        """Get all registered devices for a user"""
        try:
            user_tokens_key = f"user_tokens:{user_id}"
            device_tokens = await self.redis_client.smembers(user_tokens_key)
            
            devices = []
            for token in device_tokens:
                token_key = f"device_token:{user_id}:{token}"
                device_data = await self.redis_client.get(token_key)
                
                if device_data:
                    try:
                        device = DeviceToken.model_validate_json(device_data)
                        devices.append(device)
                    except Exception as e:
                        logger.error(f"Invalid device data for {token}: {e}")
                        # Clean up invalid token
                        await self.redis_client.srem(user_tokens_key, token)
                        await self.redis_client.delete(token_key)
            
            return devices
            
        except Exception as e:
            logger.error(f"Failed to get user devices: {e}")
            return []
    
    async def send_notification(self, request: NotificationRequest) -> NotificationResult:
        """Send push notification to user devices"""
        try:
            notification_id = f"notif_{datetime.utcnow().timestamp()}_{request.user_id}"
            
            # Check user preferences
            if not await self.should_send_notification(request.user_id, request.notification_type):
                logger.info(f"Notification blocked by user preferences: {request.user_id}")
                return NotificationResult(
                    notification_id=notification_id,
                    user_id=request.user_id,
                    success_count=0,
                    failure_count=0
                )
            
            # Get device tokens
            if request.device_tokens:
                device_tokens = request.device_tokens
            else:
                devices = await self.get_user_devices(request.user_id)
                device_tokens = [d.token for d in devices if d.active]
            
            if not device_tokens:
                logger.info(f"No active devices for user: {request.user_id}")
                return NotificationResult(
                    notification_id=notification_id,
                    user_id=request.user_id,
                    success_count=0,
                    failure_count=0
                )
            
            # Check if notification should be scheduled
            if request.schedule_time and request.schedule_time > datetime.utcnow():
                await self.schedule_notification(notification_id, request)
                return NotificationResult(
                    notification_id=notification_id,
                    user_id=request.user_id,
                    success_count=0,
                    failure_count=0,
                    device_results={"scheduled": {"status": "scheduled", "time": request.schedule_time.isoformat()}}
                )
            
            # Send to all device tokens
            results = await self.send_to_devices(device_tokens, request.payload, request.priority)
            
            # Count successes and failures
            success_count = sum(1 for r in results.values() if r.get("success", False))
            failure_count = len(results) - success_count
            
            # Store notification record
            await self.store_notification_record(notification_id, request, results)
            
            logger.info(f"Notification sent: {notification_id} - {success_count} success, {failure_count} failed")
            
            return NotificationResult(
                notification_id=notification_id,
                user_id=request.user_id,
                success_count=success_count,
                failure_count=failure_count,
                device_results=results
            )
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise
    
    async def send_to_devices(
        self,
        device_tokens: List[str],
        payload: NotificationPayload,
        priority: NotificationPriority
    ) -> Dict[str, Dict[str, Any]]:
        """Send notification to specific device tokens"""
        results = {}
        
        # Group tokens by platform
        platform_tokens = await self.group_tokens_by_platform(device_tokens)
        
        # Send to each platform
        tasks = []
        
        if platform_tokens.get(DevicePlatform.ANDROID):
            tasks.append(self.send_fcm_notification(
                platform_tokens[DevicePlatform.ANDROID],
                payload,
                priority
            ))
        
        if platform_tokens.get(DevicePlatform.IOS):
            tasks.append(self.send_apns_notification(
                platform_tokens[DevicePlatform.IOS],
                payload,
                priority
            ))
        
        if platform_tokens.get(DevicePlatform.WEB):
            tasks.append(self.send_web_push_notification(
                platform_tokens[DevicePlatform.WEB],
                payload,
                priority
            ))
        
        # Execute all platform sends concurrently
        if tasks:
            platform_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results
            for result in platform_results:
                if isinstance(result, dict):
                    results.update(result)
                else:
                    logger.error(f"Platform send failed: {result}")
        
        return results
    
    async def send_fcm_notification(
        self,
        tokens: List[str],
        payload: NotificationPayload,
        priority: NotificationPriority
    ) -> Dict[str, Dict[str, Any]]:
        """Send FCM notification to Android devices"""
        if not self.fcm_server_key:
            logger.warning("FCM server key not configured")
            return {token: {"success": False, "error": "FCM not configured"} for token in tokens}
        
        results = {}
        
        try:
            # Build FCM payload
            fcm_payload = {
                "registration_ids": tokens,
                "notification": {
                    "title": payload.title,
                    "body": payload.body,
                    "icon": "ic_notification",
                    "sound": payload.sound or "default"
                },
                "data": {
                    **payload.data,
                    "notification_type": payload.category or "general",
                    "action_url": payload.action_url or ""
                },
                "priority": "high" if priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL] else "normal",
                "time_to_live": 86400  # 24 hours
            }
            
            if payload.image_url:
                fcm_payload["notification"]["image"] = payload.image_url
            
            # Send FCM request
            headers = {
                "Authorization": f"key={self.fcm_server_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://fcm.googleapis.com/fcm/send",
                    json=fcm_payload,
                    headers=headers
                ) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        # Process individual results
                        fcm_results = response_data.get("results", [])
                        for i, token in enumerate(tokens):
                            if i < len(fcm_results):
                                result = fcm_results[i]
                                if "message_id" in result:
                                    results[token] = {"success": True, "message_id": result["message_id"]}
                                else:
                                    error = result.get("error", "Unknown error")
                                    results[token] = {"success": False, "error": error}
                                    
                                    # Handle token cleanup
                                    if error in ["NotRegistered", "InvalidRegistration"]:
                                        await self.cleanup_invalid_token(token)
                            else:
                                results[token] = {"success": False, "error": "No result"}
                    else:
                        error_msg = f"FCM request failed: {response.status}"
                        for token in tokens:
                            results[token] = {"success": False, "error": error_msg}
                        
        except Exception as e:
            logger.error(f"FCM send failed: {e}")
            for token in tokens:
                results[token] = {"success": False, "error": str(e)}
        
        return results
    
    async def send_apns_notification(
        self,
        tokens: List[str],
        payload: NotificationPayload,
        priority: NotificationPriority
    ) -> Dict[str, Dict[str, Any]]:
        """Send APNs notification to iOS devices"""
        if not all([self.apns_key_id, self.apns_team_id, self.apns_private_key]):
            logger.warning("APNs configuration incomplete")
            return {token: {"success": False, "error": "APNs not configured"} for token in tokens}
        
        results = {}
        
        try:
            # Generate JWT token for APNs
            jwt_token = self.generate_apns_jwt()
            
            # Build APNs payload
            apns_payload = {
                "aps": {
                    "alert": {
                        "title": payload.title,
                        "body": payload.body
                    },
                    "sound": payload.sound or "default",
                    "mutable-content": 1
                },
                **payload.data
            }
            
            if payload.badge_count is not None:
                apns_payload["aps"]["badge"] = payload.badge_count
            
            if payload.category:
                apns_payload["aps"]["category"] = payload.category
            
            if payload.action_url:
                apns_payload["action_url"] = payload.action_url
            
            # Send to each token individually (APNs requirement)
            headers = {
                "authorization": f"bearer {jwt_token}",
                "apns-topic": "com.rentalml.app",  # Your app bundle ID
                "apns-priority": "10" if priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL] else "5"
            }
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for token in tokens:
                    task = self.send_single_apns(session, token, apns_payload, headers)
                    tasks.append(task)
                
                # Send all notifications concurrently
                token_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(token_results):
                    token = tokens[i]
                    if isinstance(result, dict):
                        results[token] = result
                    else:
                        results[token] = {"success": False, "error": str(result)}
                        
        except Exception as e:
            logger.error(f"APNs send failed: {e}")
            for token in tokens:
                results[token] = {"success": False, "error": str(e)}
        
        return results
    
    async def send_single_apns(
        self,
        session: aiohttp.ClientSession,
        token: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Send single APNs notification"""
        try:
            url = f"https://api.push.apple.com/3/device/{token}"
            
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return {"success": True, "message_id": response.headers.get("apns-id")}
                else:
                    error_data = await response.json() if response.content_type == "application/json" else {}
                    error_reason = error_data.get("reason", f"HTTP {response.status}")
                    
                    # Handle token cleanup
                    if error_reason in ["BadDeviceToken", "Unregistered"]:
                        await self.cleanup_invalid_token(token)
                    
                    return {"success": False, "error": error_reason}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def send_web_push_notification(
        self,
        tokens: List[str],
        payload: NotificationPayload,
        priority: NotificationPriority
    ) -> Dict[str, Dict[str, Any]]:
        """Send web push notification"""
        # Web push implementation would go here
        # This is a placeholder for web push notifications
        results = {}
        for token in tokens:
            results[token] = {"success": False, "error": "Web push not implemented"}
        
        return results
    
    def generate_apns_jwt(self) -> str:
        """Generate JWT token for APNs authentication"""
        now = datetime.utcnow()
        payload = {
            "iss": self.apns_team_id,
            "iat": now,
            "exp": now + timedelta(hours=1)
        }
        
        return jwt.encode(
            payload,
            self.apns_private_key,
            algorithm="ES256",
            headers={"kid": self.apns_key_id}
        )
    
    async def group_tokens_by_platform(self, tokens: List[str]) -> Dict[DevicePlatform, List[str]]:
        """Group device tokens by platform"""
        platform_tokens = {platform: [] for platform in DevicePlatform}
        
        for token in tokens:
            # Try to determine platform from token characteristics
            # This is a simplified approach - in practice, you'd store platform info with tokens
            if len(token) == 64:  # APNs tokens are typically 64 characters
                platform_tokens[DevicePlatform.IOS].append(token)
            elif len(token) > 100:  # FCM tokens are typically longer
                platform_tokens[DevicePlatform.ANDROID].append(token)
            else:
                platform_tokens[DevicePlatform.WEB].append(token)
        
        return platform_tokens
    
    async def should_send_notification(self, user_id: str, notification_type: NotificationType) -> bool:
        """Check if notification should be sent based on user preferences"""
        try:
            prefs_key = f"notification_prefs:{user_id}"
            prefs_data = await self.redis_client.get(prefs_key)
            
            if not prefs_data:
                return True  # Default to sending if no preferences set
            
            prefs = NotificationPreferences.model_validate_json(prefs_data)
            
            if not prefs.enabled:
                return False
            
            # Check specific notification type preferences
            type_mapping = {
                NotificationType.PROPERTY_ALERT: prefs.property_alerts,
                NotificationType.RECOMMENDATION: prefs.recommendations,
                NotificationType.PRICE_CHANGE: prefs.price_changes,
                NotificationType.MARKETING: prefs.marketing
            }
            
            return type_mapping.get(notification_type, True)
            
        except Exception as e:
            logger.error(f"Failed to check notification preferences: {e}")
            return True  # Default to sending on error
    
    async def update_notification_preferences(
        self,
        user_id: str,
        preferences: NotificationPreferences
    ) -> bool:
        """Update user notification preferences"""
        try:
            prefs_key = f"notification_prefs:{user_id}"
            await self.redis_client.setex(
                prefs_key,
                86400 * 365,  # 1 year
                preferences.model_dump_json()
            )
            
            logger.info(f"Notification preferences updated for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update notification preferences: {e}")
            return False
    
    async def cleanup_invalid_token(self, token: str) -> None:
        """Clean up invalid/expired device token"""
        try:
            # Find and remove token from all user token sets
            pattern = f"device_token:*:{token}"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                # Extract user_id from key
                parts = key.split(":")
                if len(parts) >= 3:
                    user_id = parts[2]
                    user_tokens_key = f"user_tokens:{user_id}"
                    await self.redis_client.srem(user_tokens_key, token)
                
                # Remove the device token record
                await self.redis_client.delete(key)
            
            logger.info(f"Cleaned up invalid token: {token[:20]}...")
            
        except Exception as e:
            logger.error(f"Failed to cleanup token: {e}")
    
    async def schedule_notification(self, notification_id: str, request: NotificationRequest) -> None:
        """Schedule notification for later delivery"""
        try:
            schedule_key = f"scheduled_notification:{notification_id}"
            schedule_data = {
                "request": request.model_dump_json(),
                "scheduled_for": request.schedule_time.isoformat()
            }
            
            # Calculate TTL until scheduled time
            now = datetime.utcnow()
            ttl = int((request.schedule_time - now).total_seconds())
            
            if ttl > 0:
                await self.redis_client.setex(schedule_key, ttl, json.dumps(schedule_data))
                logger.info(f"Notification scheduled: {notification_id} for {request.schedule_time}")
            
        except Exception as e:
            logger.error(f"Failed to schedule notification: {e}")
    
    async def store_notification_record(
        self,
        notification_id: str,
        request: NotificationRequest,
        results: Dict[str, Dict[str, Any]]
    ) -> None:
        """Store notification delivery record"""
        try:
            record_key = f"notification_record:{notification_id}"
            record_data = {
                "user_id": request.user_id,
                "type": request.notification_type,
                "payload": request.payload.model_dump(),
                "results": results,
                "sent_at": datetime.utcnow().isoformat()
            }
            
            # Store for 30 days
            await self.redis_client.setex(
                record_key,
                86400 * 30,
                json.dumps(record_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store notification record: {e}")
    
    # Utility methods for creating common notifications
    
    async def send_property_alert(
        self,
        user_id: str,
        property_id: str,
        property_title: str,
        alert_type: str = "new_match"
    ) -> NotificationResult:
        """Send property alert notification"""
        payload = NotificationPayload(
            title="New Property Match!",
            body=f"Found a property you might like: {property_title}",
            data={
                "property_id": property_id,
                "alert_type": alert_type
            },
            action_url=f"/property/{property_id}",
            category="property_alert"
        )
        
        request = NotificationRequest(
            user_id=user_id,
            notification_type=NotificationType.PROPERTY_ALERT,
            priority=NotificationPriority.HIGH,
            payload=payload
        )
        
        return await self.send_notification(request)
    
    async def send_price_change_alert(
        self,
        user_id: str,
        property_id: str,
        property_title: str,
        old_price: float,
        new_price: float
    ) -> NotificationResult:
        """Send price change notification"""
        price_change = new_price - old_price
        change_text = f"${abs(price_change):,.0f} {'decrease' if price_change < 0 else 'increase'}"
        
        payload = NotificationPayload(
            title="Price Change Alert",
            body=f"{property_title} - {change_text}",
            data={
                "property_id": property_id,
                "old_price": old_price,
                "new_price": new_price,
                "price_change": price_change
            },
            action_url=f"/property/{property_id}",
            category="price_change"
        )
        
        request = NotificationRequest(
            user_id=user_id,
            notification_type=NotificationType.PRICE_CHANGE,
            priority=NotificationPriority.HIGH,
            payload=payload
        )
        
        return await self.send_notification(request)
    
    async def send_recommendation_notification(
        self,
        user_id: str,
        property_count: int = 1
    ) -> NotificationResult:
        """Send new recommendations notification"""
        title = "New Recommendations"
        body = f"We found {property_count} new propert{'y' if property_count == 1 else 'ies'} for you!"
        
        payload = NotificationPayload(
            title=title,
            body=body,
            data={
                "property_count": property_count,
                "recommendation_type": "personalized"
            },
            action_url="/recommendations",
            category="recommendation"
        )
        
        request = NotificationRequest(
            user_id=user_id,
            notification_type=NotificationType.RECOMMENDATION,
            priority=NotificationPriority.NORMAL,
            payload=payload
        )
        
        return await self.send_notification(request)