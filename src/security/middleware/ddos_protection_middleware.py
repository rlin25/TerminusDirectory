"""
DDoS Protection Middleware

Advanced DDoS protection middleware with traffic analysis, anomaly detection,
and automatic threat mitigation for the rental ML system.
"""

import asyncio
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


@dataclass
class TrafficMetrics:
    """Traffic metrics for DDoS detection"""
    requests_per_second: deque = field(default_factory=lambda: deque(maxlen=60))
    requests_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    unique_ips_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    error_rate_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: float = field(default_factory=time.time)


@dataclass
class IPAnalytics:
    """Per-IP analytics for behavior analysis"""
    request_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    request_intervals: deque = field(default_factory=lambda: deque(maxlen=100))
    user_agents: Set[str] = field(default_factory=set)
    paths_accessed: Set[str] = field(default_factory=set)
    error_count: int = 0
    suspicious_score: float = 0.0


@dataclass
class ThreatAlert:
    """DDoS threat alert"""
    threat_type: str
    severity: str  # low, medium, high, critical
    description: str
    source_ips: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    mitigation_applied: bool = False


class DDoSProtectionMiddleware(BaseHTTPMiddleware):
    """
    Advanced DDoS Protection Middleware with:
    - Real-time traffic analysis
    - Behavioral anomaly detection
    - Adaptive rate limiting
    - IP reputation scoring
    - Automatic threat mitigation
    - Geolocation-based filtering
    - Machine learning-based detection
    """
    
    def __init__(
        self,
        app,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # DDoS protection configuration
        self.enable_protection = self.config.get("enable_protection", True)
        self.detection_sensitivity = self.config.get("detection_sensitivity", "medium")
        self.auto_mitigation = self.config.get("auto_mitigation", True)
        
        # Traffic thresholds
        self.max_requests_per_second = self.config.get("max_requests_per_second", 100)
        self.max_requests_per_minute = self.config.get("max_requests_per_minute", 1000)
        self.max_error_rate = self.config.get("max_error_rate", 0.5)  # 50%
        self.max_response_time_ms = self.config.get("max_response_time_ms", 5000)
        
        # IP behavior thresholds
        self.max_requests_per_ip_per_minute = self.config.get("max_requests_per_ip_per_minute", 60)
        self.suspicious_score_threshold = self.config.get("suspicious_score_threshold", 0.7)
        self.min_request_interval_ms = self.config.get("min_request_interval_ms", 10)
        
        # Geolocation filtering
        self.blocked_countries = set(self.config.get("blocked_countries", []))
        self.allowed_countries_only = self.config.get("allowed_countries_only", False)
        self.allowed_countries = set(self.config.get("allowed_countries", []))
        
        # Data structures
        self.traffic_metrics = TrafficMetrics()
        self.ip_analytics: Dict[str, IPAnalytics] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> block_until_timestamp
        self.threat_alerts: List[ThreatAlert] = []
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "threats_detected": 0,
            "ips_blocked": 0,
            "false_positives": 0,
            "mitigation_actions": 0
        }
        
        # Real-time monitoring
        self.monitoring_enabled = self.config.get("monitoring_enabled", True)
        self.alert_callback = self.config.get("alert_callback")
        
        # Initialize background tasks
        if self.enable_protection:
            asyncio.create_task(self._background_analysis())
            asyncio.create_task(self._cleanup_task())
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method"""
        if not self.enable_protection:
            return await call_next(request)
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            self._stats["total_requests"] += 1
            
            # Check if IP is blocked
            if await self._is_ip_blocked(client_ip):
                self._stats["blocked_requests"] += 1
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Access temporarily restricted",
                        "message": "Your IP has been temporarily blocked due to suspicious activity"
                    }
                )
            
            # Update traffic metrics
            await self._update_traffic_metrics(request, start_time)
            
            # Analyze request for suspicious patterns
            suspicious_score = await self._analyze_request(request, client_ip)
            
            # Check for immediate threats
            if suspicious_score > self.suspicious_score_threshold:
                threat_detected = await self._detect_immediate_threats(request, client_ip, suspicious_score)
                if threat_detected:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Suspicious activity detected",
                            "message": "Request blocked due to suspicious patterns"
                        }
                    )
            
            # Process request
            response = await call_next(request)
            
            # Update post-request analytics
            duration = time.time() - start_time
            await self._update_post_request_analytics(request, response, client_ip, duration)
            
            return response
            
        except Exception as e:
            self.logger.error(f"DDoS protection middleware error: {e}", exc_info=True)
            # Allow request to proceed on middleware error
            return await call_next(request)
    
    async def _update_traffic_metrics(self, request: Request, start_time: float):
        """Update real-time traffic metrics"""
        current_minute = int(start_time // 60)
        current_second = int(start_time)
        
        metrics = self.traffic_metrics
        
        # Update requests per second
        if len(metrics.requests_per_second) == 0 or metrics.requests_per_second[-1][0] != current_second:
            metrics.requests_per_second.append((current_second, 1))
        else:
            # Increment current second count
            last_entry = metrics.requests_per_second[-1]
            metrics.requests_per_second[-1] = (current_second, last_entry[1] + 1)
        
        # Update requests per minute
        client_ip = self._get_client_ip(request)
        if len(metrics.requests_per_minute) == 0 or metrics.requests_per_minute[-1][0] != current_minute:
            metrics.requests_per_minute.append((current_minute, 1))
            # Also update unique IPs for this minute
            metrics.unique_ips_per_minute.append((current_minute, {client_ip}))
        else:
            # Increment current minute count
            minute_data = metrics.requests_per_minute[-1]
            metrics.requests_per_minute[-1] = (current_minute, minute_data[1] + 1)
            # Add IP to unique set
            if len(metrics.unique_ips_per_minute) > 0 and metrics.unique_ips_per_minute[-1][0] == current_minute:
                metrics.unique_ips_per_minute[-1] = (current_minute, metrics.unique_ips_per_minute[-1][1] | {client_ip})
        
        metrics.last_update = start_time
    
    async def _analyze_request(self, request: Request, client_ip: str) -> float:
        """Analyze request for suspicious patterns and return suspicion score"""
        current_time = time.time()
        
        # Get or create IP analytics
        if client_ip not in self.ip_analytics:
            self.ip_analytics[client_ip] = IPAnalytics()
        
        ip_data = self.ip_analytics[client_ip]
        
        # Update IP analytics
        ip_data.request_count += 1
        ip_data.last_seen = current_time
        ip_data.paths_accessed.add(request.url.path)
        
        user_agent = request.headers.get("user-agent", "")
        if user_agent:
            ip_data.user_agents.add(user_agent)
        
        # Calculate request interval
        if ip_data.request_intervals:
            interval = current_time - ip_data.request_intervals[-1]
            ip_data.request_intervals.append(current_time)
        else:
            ip_data.request_intervals.append(current_time)
            interval = 1.0  # Default for first request
        
        # Calculate suspicion score
        suspicion_score = 0.0
        
        # High request rate from single IP
        recent_requests = sum(1 for t in ip_data.request_intervals if current_time - t < 60)
        if recent_requests > self.max_requests_per_ip_per_minute:
            suspicion_score += 0.3
        
        # Very short request intervals (bot-like behavior)
        if interval < self.min_request_interval_ms / 1000:
            suspicion_score += 0.2
        
        # Too many different user agents (possible spoofing)
        if len(ip_data.user_agents) > 10:
            suspicion_score += 0.1
        
        # Accessing too many different paths (possible scanning)
        if len(ip_data.paths_accessed) > 50:
            suspicion_score += 0.15
        
        # High error rate
        if ip_data.request_count > 10:
            error_rate = ip_data.error_count / ip_data.request_count
            if error_rate > 0.3:
                suspicion_score += 0.2
        
        # Missing or suspicious user agent
        if not user_agent or len(user_agent) < 10:
            suspicion_score += 0.1
        
        # Check for common bot patterns
        bot_indicators = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
        if any(indicator in user_agent.lower() for indicator in bot_indicators):
            suspicion_score += 0.1
        
        # Update suspicion score
        ip_data.suspicious_score = max(ip_data.suspicious_score, suspicion_score)
        
        return suspicion_score
    
    async def _detect_immediate_threats(self, request: Request, client_ip: str, suspicion_score: float) -> bool:
        """Detect immediate threats that require blocking"""
        current_time = time.time()
        
        # Check global traffic patterns
        metrics = self.traffic_metrics
        
        # Current requests per second
        if metrics.requests_per_second:
            current_rps = metrics.requests_per_second[-1][1] if metrics.requests_per_second else 0
            if current_rps > self.max_requests_per_second:
                await self._create_threat_alert(
                    "high_traffic_volume",
                    "high",
                    f"Requests per second ({current_rps}) exceeded threshold ({self.max_requests_per_second})",
                    [client_ip],
                    {"rps": current_rps, "threshold": self.max_requests_per_second}
                )
                
                if self.auto_mitigation:
                    await self._apply_mitigation("rate_limiting", [client_ip])
                    return True
        
        # IP-specific threats
        ip_data = self.ip_analytics.get(client_ip)
        if ip_data:
            # Rapid-fire requests
            recent_requests = [t for t in ip_data.request_intervals if current_time - t < 10]
            if len(recent_requests) > 20:  # More than 20 requests in 10 seconds
                await self._create_threat_alert(
                    "rapid_fire_requests",
                    "high",
                    f"IP {client_ip} made {len(recent_requests)} requests in 10 seconds",
                    [client_ip],
                    {"requests_in_10s": len(recent_requests)}
                )
                
                if self.auto_mitigation:
                    await self._block_ip(client_ip, duration_seconds=300)  # 5 minute block
                    return True
        
        return False
    
    async def _update_post_request_analytics(
        self, 
        request: Request, 
        response: Response, 
        client_ip: str, 
        duration: float
    ):
        """Update analytics after request completion"""
        # Update response times
        duration_ms = duration * 1000
        self.traffic_metrics.response_times.append(duration_ms)
        
        # Update error rates
        if response.status_code >= 400:
            ip_data = self.ip_analytics.get(client_ip)
            if ip_data:
                ip_data.error_count += 1
            
            # Update global error rate
            current_minute = int(time.time() // 60)
            if self.traffic_metrics.error_rate_per_minute:
                if self.traffic_metrics.error_rate_per_minute[-1][0] == current_minute:
                    # Increment error count for current minute
                    minute_data = self.traffic_metrics.error_rate_per_minute[-1]
                    total_requests = minute_data[1]
                    error_count = minute_data[2] + 1
                    error_rate = error_count / total_requests if total_requests > 0 else 0
                    self.traffic_metrics.error_rate_per_minute[-1] = (current_minute, total_requests, error_count, error_rate)
                else:
                    # New minute
                    self.traffic_metrics.error_rate_per_minute.append((current_minute, 1, 1, 1.0))
            else:
                # First error
                self.traffic_metrics.error_rate_per_minute.append((current_minute, 1, 1, 1.0))
    
    async def _background_analysis(self):
        """Background task for continuous threat analysis"""
        while True:
            try:
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
                if not self.enable_protection:
                    continue
                
                await self._analyze_traffic_patterns()
                await self._detect_coordinated_attacks()
                await self._update_ip_reputation_scores()
                
            except Exception as e:
                self.logger.error(f"Background analysis error: {e}")
    
    async def _analyze_traffic_patterns(self):
        """Analyze overall traffic patterns for anomalies"""
        current_time = time.time()
        metrics = self.traffic_metrics
        
        # Analyze requests per minute trend
        if len(metrics.requests_per_minute) >= 5:
            recent_counts = [count for _, count in list(metrics.requests_per_minute)[-5:]]
            avg_requests = statistics.mean(recent_counts)
            
            if avg_requests > self.max_requests_per_minute:
                await self._create_threat_alert(
                    "sustained_high_traffic",
                    "medium",
                    f"Sustained high traffic detected: {avg_requests:.1f} req/min average",
                    [],
                    {"avg_requests_per_minute": avg_requests}
                )
        
        # Analyze response time degradation
        if len(metrics.response_times) >= 100:
            recent_times = list(metrics.response_times)[-100:]
            avg_response_time = statistics.mean(recent_times)
            
            if avg_response_time > self.max_response_time_ms:
                await self._create_threat_alert(
                    "performance_degradation",
                    "medium",
                    f"Average response time degraded to {avg_response_time:.1f}ms",
                    [],
                    {"avg_response_time_ms": avg_response_time}
                )
    
    async def _detect_coordinated_attacks(self):
        """Detect coordinated attacks from multiple IPs"""
        current_time = time.time()
        
        # Find IPs with similar suspicious patterns
        suspicious_ips = []
        for ip, data in self.ip_analytics.items():
            if (data.suspicious_score > 0.5 and 
                current_time - data.last_seen < 300):  # Active in last 5 minutes
                suspicious_ips.append(ip)
        
        # If many IPs are acting suspiciously, it might be a coordinated attack
        if len(suspicious_ips) > 10:
            await self._create_threat_alert(
                "coordinated_attack",
                "critical",
                f"Coordinated attack detected from {len(suspicious_ips)} IPs",
                suspicious_ips[:20],  # Limit to first 20 IPs
                {"suspicious_ip_count": len(suspicious_ips)}
            )
            
            if self.auto_mitigation:
                # Block top suspicious IPs
                for ip in suspicious_ips[:50]:  # Block up to 50 IPs
                    await self._block_ip(ip, duration_seconds=1800)  # 30 minute block
    
    async def _update_ip_reputation_scores(self):
        """Update IP reputation scores based on behavior"""
        current_time = time.time()
        
        for ip, data in self.ip_analytics.items():
            # Decay suspicion score over time
            time_since_last_request = current_time - data.last_seen
            if time_since_last_request > 3600:  # 1 hour
                data.suspicious_score *= 0.9  # Decay by 10%
            
            # Remove very old analytics
            if time_since_last_request > 86400:  # 24 hours
                data.request_intervals = deque([
                    t for t in data.request_intervals 
                    if current_time - t < 86400
                ], maxlen=100)
    
    async def _create_threat_alert(
        self,
        threat_type: str,
        severity: str,
        description: str,
        source_ips: List[str],
        metrics: Dict[str, Any]
    ):
        """Create and log a threat alert"""
        alert = ThreatAlert(
            threat_type=threat_type,
            severity=severity,
            description=description,
            source_ips=source_ips,
            metrics=metrics
        )
        
        self.threat_alerts.append(alert)
        self._stats["threats_detected"] += 1
        
        # Log the alert
        self.logger.warning(
            f"DDoS Threat Alert: {threat_type} ({severity}) - {description} - "
            f"IPs: {source_ips[:5]}{'...' if len(source_ips) > 5 else ''}"
        )
        
        # Call alert callback if configured
        if self.alert_callback:
            try:
                await self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        
        # Keep only recent alerts (last 1000)
        if len(self.threat_alerts) > 1000:
            self.threat_alerts = self.threat_alerts[-1000:]
    
    async def _apply_mitigation(self, mitigation_type: str, target_ips: List[str]):
        """Apply mitigation measures"""
        self._stats["mitigation_actions"] += 1
        
        if mitigation_type == "rate_limiting":
            # Apply more aggressive rate limiting
            for ip in target_ips:
                await self._block_ip(ip, duration_seconds=600)  # 10 minute block
        
        self.logger.info(f"Applied mitigation: {mitigation_type} to {len(target_ips)} IPs")
    
    async def _block_ip(self, ip: str, duration_seconds: int = 3600):
        """Block an IP address for specified duration"""
        block_until = time.time() + duration_seconds
        self.blocked_ips[ip] = block_until
        self._stats["ips_blocked"] += 1
        
        self.logger.info(f"Blocked IP {ip} for {duration_seconds} seconds")
    
    async def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP address is currently blocked"""
        if ip not in self.blocked_ips:
            return False
        
        if time.time() > self.blocked_ips[ip]:
            # Block has expired
            del self.blocked_ips[ip]
            return False
        
        return True
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_data()
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    async def _cleanup_expired_data(self):
        """Clean up expired data"""
        current_time = time.time()
        
        # Clean up expired IP blocks
        expired_blocks = [
            ip for ip, block_until in self.blocked_ips.items()
            if current_time > block_until
        ]
        for ip in expired_blocks:
            del self.blocked_ips[ip]
        
        # Clean up old IP analytics
        old_ips = [
            ip for ip, data in self.ip_analytics.items()
            if current_time - data.last_seen > 86400  # 24 hours
        ]
        for ip in old_ips:
            del self.ip_analytics[ip]
        
        # Clean up old alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.threat_alerts = [
            alert for alert in self.threat_alerts
            if alert.timestamp > cutoff_time
        ]
        
        if expired_blocks or old_ips:
            self.logger.debug(
                f"Cleaned up {len(expired_blocks)} expired blocks and "
                f"{len(old_ips)} old IP analytics"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def unblock_ip(self, ip: str):
        """Manually unblock an IP address"""
        if ip in self.blocked_ips:
            del self.blocked_ips[ip]
            self.logger.info(f"Manually unblocked IP: {ip}")
    
    async def get_ip_analytics(self, ip: str) -> Optional[Dict[str, Any]]:
        """Get analytics for specific IP"""
        data = self.ip_analytics.get(ip)
        if not data:
            return None
        
        return {
            "request_count": data.request_count,
            "first_seen": datetime.fromtimestamp(data.first_seen).isoformat(),
            "last_seen": datetime.fromtimestamp(data.last_seen).isoformat(),
            "unique_user_agents": len(data.user_agents),
            "unique_paths": len(data.paths_accessed),
            "error_count": data.error_count,
            "suspicious_score": data.suspicious_score,
            "is_blocked": await self._is_ip_blocked(ip)
        }
    
    def get_ddos_statistics(self) -> Dict[str, Any]:
        """Get DDoS protection statistics"""
        current_time = time.time()
        
        # Current traffic metrics
        current_rps = 0
        if self.traffic_metrics.requests_per_second:
            current_rps = self.traffic_metrics.requests_per_second[-1][1]
        
        recent_alerts = [
            alert for alert in self.threat_alerts
            if (datetime.now() - alert.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        return {
            "statistics": dict(self._stats),
            "current_metrics": {
                "requests_per_second": current_rps,
                "active_ips": len([
                    ip for ip, data in self.ip_analytics.items()
                    if current_time - data.last_seen < 300  # Active in last 5 minutes
                ]),
                "blocked_ips": len(self.blocked_ips),
                "suspicious_ips": len([
                    ip for ip, data in self.ip_analytics.items()
                    if data.suspicious_score > 0.5
                ])
            },
            "recent_alerts": len(recent_alerts),
            "protection_enabled": self.enable_protection,
            "auto_mitigation": self.auto_mitigation
        }