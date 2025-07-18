"""
Security Monitor

Central security monitoring system that aggregates security events,
detects threats, and coordinates incident response.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from ..auth.models import SecurityEvent, SecurityEventType, ThreatLevel


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityAlert:
    """Security alert model"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_events: List[SecurityEvent]
    affected_resources: List[str]
    recommended_actions: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """Security metrics aggregation"""
    total_events: int = 0
    events_by_type: Dict[SecurityEventType, int] = field(default_factory=lambda: defaultdict(int))
    events_by_severity: Dict[ThreatLevel, int] = field(default_factory=lambda: defaultdict(int))
    active_alerts: int = 0
    resolved_alerts: int = 0
    unique_users_affected: int = 0
    unique_ips_involved: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class SecurityMonitor:
    """
    Central Security Monitoring System with:
    - Real-time event processing
    - Threat correlation and analysis
    - Alert generation and management
    - Metrics aggregation
    - Incident tracking
    - Response coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Event processing
        self.event_queue = asyncio.Queue()
        self.event_buffer = deque(maxlen=10000)
        self.processing_enabled = True
        
        # Alert management
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        
        # Metrics
        self.metrics = SecurityMetrics()
        self.metrics_history: deque = deque(maxlen=100)
        
        # Correlation rules
        self.correlation_rules = self._load_correlation_rules()
        
        # Threat detection
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        self.load_threat_patterns()
        
        # Configuration
        self.alert_cooldown_minutes = self.config.get("alert_cooldown_minutes", 10)
        self.max_events_per_second = self.config.get("max_events_per_second", 100)
        self.enable_auto_response = self.config.get("enable_auto_response", True)
        
        # Start background tasks
        self.background_tasks = []
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        tasks = [
            self._event_processor(),
            self._metrics_aggregator(),
            self._alert_manager(),
            self._threat_correlator(),
            self._cleanup_task()
        ]
        
        for task in tasks:
            self.background_tasks.append(asyncio.create_task(task))
    
    async def process_event(self, event: SecurityEvent):
        """Process a security event"""
        try:
            # Add to queue for processing
            await self.event_queue.put(event)
            
            # Add to buffer for historical analysis
            self.event_buffer.append(event)
            
            # Update metrics
            self.metrics.total_events += 1
            self.metrics.events_by_type[event.event_type] += 1
            self.metrics.events_by_severity[event.threat_level] += 1
            
            # Immediate processing for critical events
            if event.threat_level == ThreatLevel.CRITICAL:
                await self._process_critical_event(event)
            
        except Exception as e:
            self.logger.error(f"Failed to process security event: {e}")
    
    async def _event_processor(self):
        """Background event processor"""
        while self.processing_enabled:
            try:
                # Process events from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Analyze event
                await self._analyze_event(event)
                
                # Check for patterns
                await self._check_threat_patterns(event)
                
                # Correlate with other events
                await self._correlate_events(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Event processor error: {e}")
                await asyncio.sleep(1)
    
    async def _analyze_event(self, event: SecurityEvent):
        """Analyze individual security event"""
        # Basic event analysis
        analysis = {
            "event_id": str(event.id),
            "severity": event.threat_level.value,
            "timestamp": event.timestamp.isoformat(),
            "anomaly_score": 0.0,
            "context": {}
        }
        
        # Calculate anomaly score based on event characteristics
        anomaly_score = 0.0
        
        # Check event frequency
        recent_events = [
            e for e in self.event_buffer
            if (e.event_type == event.event_type and 
                datetime.now() - e.timestamp < timedelta(minutes=5))
        ]
        
        if len(recent_events) > 10:
            anomaly_score += 0.3
        
        # Check for unusual patterns
        if event.event_type == SecurityEventType.LOGIN_FAILURE:
            # Check for brute force patterns
            user_failures = [
                e for e in recent_events
                if e.username == event.username
            ]
            if len(user_failures) > 5:
                anomaly_score += 0.4
        
        # Check IP reputation
        if event.ip_address:
            ip_events = [
                e for e in self.event_buffer
                if (e.ip_address == event.ip_address and 
                    datetime.now() - e.timestamp < timedelta(hours=1))
            ]
            if len(ip_events) > 50:
                anomaly_score += 0.5
        
        analysis["anomaly_score"] = anomaly_score
        
        # Generate alert if score is high
        if anomaly_score > 0.7:
            await self._generate_alert(event, analysis)
    
    async def _check_threat_patterns(self, event: SecurityEvent):
        """Check event against known threat patterns"""
        for pattern_name, pattern_config in self.threat_patterns.items():
            if await self._matches_pattern(event, pattern_config):
                await self._handle_pattern_match(event, pattern_name, pattern_config)
    
    async def _matches_pattern(self, event: SecurityEvent, pattern: Dict[str, Any]) -> bool:
        """Check if event matches threat pattern"""
        # Check event type
        if "event_types" in pattern:
            if event.event_type not in pattern["event_types"]:
                return False
        
        # Check threat level
        if "min_threat_level" in pattern:
            threat_levels = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            min_level = pattern["min_threat_level"]
            if threat_levels.index(event.threat_level) < threat_levels.index(min_level):
                return False
        
        # Check time window
        if "time_window_minutes" in pattern:
            time_window = timedelta(minutes=pattern["time_window_minutes"])
            matching_events = [
                e for e in self.event_buffer
                if (e.event_type == event.event_type and 
                    datetime.now() - e.timestamp < time_window)
            ]
            
            if len(matching_events) < pattern.get("min_occurrences", 1):
                return False
        
        return True
    
    async def _handle_pattern_match(self, event: SecurityEvent, pattern_name: str, pattern_config: Dict[str, Any]):
        """Handle threat pattern match"""
        alert = SecurityAlert(
            id=f"pattern_{pattern_name}_{event.id}",
            title=f"Threat Pattern Detected: {pattern_name}",
            description=pattern_config.get("description", f"Pattern {pattern_name} detected"),
            severity=AlertSeverity(pattern_config.get("severity", "medium")),
            event_type=event.event_type,
            threat_level=event.threat_level,
            source_events=[event],
            affected_resources=[event.resource or "unknown"],
            recommended_actions=pattern_config.get("recommended_actions", [])
        )
        
        await self._create_alert(alert)
    
    async def _correlate_events(self, event: SecurityEvent):
        """Correlate events to identify complex threats"""
        # Look for related events in recent history
        related_events = []
        
        # Correlate by IP address
        if event.ip_address:
            ip_events = [
                e for e in self.event_buffer
                if (e.ip_address == event.ip_address and 
                    datetime.now() - e.timestamp < timedelta(hours=1))
            ]
            related_events.extend(ip_events)
        
        # Correlate by user
        if event.username:
            user_events = [
                e for e in self.event_buffer
                if (e.username == event.username and 
                    datetime.now() - e.timestamp < timedelta(hours=1))
            ]
            related_events.extend(user_events)
        
        # Apply correlation rules
        for rule in self.correlation_rules:
            if await self._apply_correlation_rule(event, related_events, rule):
                await self._handle_correlation_match(event, related_events, rule)
    
    async def _apply_correlation_rule(self, event: SecurityEvent, related_events: List[SecurityEvent], rule: Dict[str, Any]) -> bool:
        """Apply correlation rule to events"""
        # This is a simplified correlation rule engine
        # In production, you might use a more sophisticated system
        
        rule_type = rule.get("type")
        
        if rule_type == "sequence":
            # Check for sequence of events
            required_sequence = rule.get("sequence", [])
            event_types = [e.event_type for e in related_events]
            
            # Check if required sequence appears in events
            for i in range(len(event_types) - len(required_sequence) + 1):
                if event_types[i:i+len(required_sequence)] == required_sequence:
                    return True
        
        elif rule_type == "frequency":
            # Check for high frequency of specific events
            event_type = rule.get("event_type")
            threshold = rule.get("threshold", 10)
            
            matching_events = [e for e in related_events if e.event_type == event_type]
            return len(matching_events) >= threshold
        
        elif rule_type == "combination":
            # Check for combination of different event types
            required_types = set(rule.get("event_types", []))
            observed_types = set(e.event_type for e in related_events)
            
            return required_types.issubset(observed_types)
        
        return False
    
    async def _handle_correlation_match(self, event: SecurityEvent, related_events: List[SecurityEvent], rule: Dict[str, Any]):
        """Handle correlation rule match"""
        alert = SecurityAlert(
            id=f"correlation_{rule['name']}_{event.id}",
            title=f"Correlated Threat: {rule['name']}",
            description=rule.get("description", f"Correlation rule {rule['name']} triggered"),
            severity=AlertSeverity(rule.get("severity", "high")),
            event_type=event.event_type,
            threat_level=ThreatLevel.HIGH,
            source_events=[event] + related_events[:10],  # Limit to avoid huge alerts
            affected_resources=list(set(e.resource for e in related_events if e.resource)),
            recommended_actions=rule.get("recommended_actions", [])
        )
        
        await self._create_alert(alert)
    
    async def _process_critical_event(self, event: SecurityEvent):
        """Process critical security event immediately"""
        alert = SecurityAlert(
            id=f"critical_{event.id}",
            title=f"Critical Security Event: {event.event_type.value}",
            description=f"Critical security event detected: {event.message}",
            severity=AlertSeverity.CRITICAL,
            event_type=event.event_type,
            threat_level=event.threat_level,
            source_events=[event],
            affected_resources=[event.resource or "unknown"],
            recommended_actions=[
                "Investigate immediately",
                "Check for additional indicators",
                "Consider blocking IP if malicious",
                "Escalate to security team"
            ]
        )
        
        await self._create_alert(alert)
    
    async def _generate_alert(self, event: SecurityEvent, analysis: Dict[str, Any]):
        """Generate security alert from event analysis"""
        severity = AlertSeverity.MEDIUM
        
        # Determine severity based on anomaly score
        anomaly_score = analysis.get("anomaly_score", 0.0)
        if anomaly_score > 0.9:
            severity = AlertSeverity.CRITICAL
        elif anomaly_score > 0.7:
            severity = AlertSeverity.HIGH
        elif anomaly_score > 0.5:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        alert = SecurityAlert(
            id=f"anomaly_{event.id}",
            title=f"Anomalous Activity Detected",
            description=f"Unusual security event pattern detected (score: {anomaly_score:.2f})",
            severity=severity,
            event_type=event.event_type,
            threat_level=event.threat_level,
            source_events=[event],
            affected_resources=[event.resource or "unknown"],
            recommended_actions=[
                "Review event details",
                "Check for related activities",
                "Validate with user if applicable"
            ],
            metadata=analysis
        )
        
        await self._create_alert(alert)
    
    async def _create_alert(self, alert: SecurityAlert):
        """Create and process security alert"""
        # Check for duplicate alerts (cooldown)
        similar_alerts = [
            a for a in self.active_alerts.values()
            if (a.event_type == alert.event_type and 
                a.severity == alert.severity and
                datetime.now() - a.created_at < timedelta(minutes=self.alert_cooldown_minutes))
        ]
        
        if similar_alerts:
            # Update existing alert instead of creating new one
            existing_alert = similar_alerts[0]
            existing_alert.source_events.extend(alert.source_events)
            existing_alert.updated_at = datetime.now()
            existing_alert.description += f"\n\nAdditional occurrence at {alert.created_at.isoformat()}"
            return
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        self.metrics.active_alerts += 1
        
        # Log alert
        self.logger.warning(f"Security Alert Created: {alert.title} (Severity: {alert.severity.value})")
        
        # Trigger alert handlers
        await self._trigger_alert_handlers(alert)
        
        # Auto-response if enabled
        if self.enable_auto_response:
            await self._auto_respond(alert)
    
    async def _trigger_alert_handlers(self, alert: SecurityAlert):
        """Trigger registered alert handlers"""
        handlers = self.alert_handlers.get(alert.severity, [])
        
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
    
    async def _auto_respond(self, alert: SecurityAlert):
        """Automatic response to security alerts"""
        # This is a simplified auto-response system
        # In production, you'd have more sophisticated response actions
        
        if alert.severity == AlertSeverity.CRITICAL:
            # Critical alerts might trigger immediate blocking
            for event in alert.source_events:
                if event.ip_address:
                    await self._block_ip(event.ip_address, duration_minutes=60)
        
        elif alert.severity == AlertSeverity.HIGH:
            # High severity alerts might trigger additional monitoring
            for event in alert.source_events:
                if event.username:
                    await self._flag_user_for_monitoring(event.username)
    
    async def _block_ip(self, ip_address: str, duration_minutes: int = 60):
        """Block IP address (would integrate with firewall/WAF)"""
        self.logger.info(f"Auto-response: Blocking IP {ip_address} for {duration_minutes} minutes")
        # Implementation would depend on your infrastructure
    
    async def _flag_user_for_monitoring(self, username: str):
        """Flag user for additional monitoring"""
        self.logger.info(f"Auto-response: Flagging user {username} for enhanced monitoring")
        # Implementation would flag user in your monitoring system
    
    async def _metrics_aggregator(self):
        """Background metrics aggregation"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(60)  # Aggregate every minute
                
                # Create metrics snapshot
                current_metrics = SecurityMetrics(
                    total_events=self.metrics.total_events,
                    events_by_type=dict(self.metrics.events_by_type),
                    events_by_severity=dict(self.metrics.events_by_severity),
                    active_alerts=len(self.active_alerts),
                    resolved_alerts=self.metrics.resolved_alerts,
                    unique_users_affected=len(set(e.username for e in self.event_buffer if e.username)),
                    unique_ips_involved=len(set(e.ip_address for e in self.event_buffer if e.ip_address))
                )
                
                self.metrics_history.append(current_metrics)
                
            except Exception as e:
                self.logger.error(f"Metrics aggregator error: {e}")
    
    async def _alert_manager(self):
        """Background alert management"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Auto-resolve old alerts
                cutoff_time = datetime.now() - timedelta(hours=24)
                resolved_alerts = []
                
                for alert_id, alert in list(self.active_alerts.items()):
                    if alert.created_at < cutoff_time and not alert.acknowledged:
                        alert.resolved = True
                        resolved_alerts.append(alert_id)
                
                # Remove resolved alerts
                for alert_id in resolved_alerts:
                    del self.active_alerts[alert_id]
                    self.metrics.active_alerts -= 1
                    self.metrics.resolved_alerts += 1
                
                if resolved_alerts:
                    self.logger.info(f"Auto-resolved {len(resolved_alerts)} old alerts")
                
            except Exception as e:
                self.logger.error(f"Alert manager error: {e}")
    
    async def _threat_correlator(self):
        """Background threat correlation"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(120)  # Correlate every 2 minutes
                
                # Run advanced correlation analysis
                await self._advanced_threat_correlation()
                
            except Exception as e:
                self.logger.error(f"Threat correlator error: {e}")
    
    async def _advanced_threat_correlation(self):
        """Advanced threat correlation analysis"""
        # This would contain more sophisticated correlation logic
        # For now, just a placeholder
        pass
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                
                # Clean up old events from buffer
                cutoff_time = datetime.now() - timedelta(hours=24)
                old_events = [e for e in self.event_buffer if e.timestamp < cutoff_time]
                
                for event in old_events:
                    self.event_buffer.remove(event)
                
                if old_events:
                    self.logger.info(f"Cleaned up {len(old_events)} old events")
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    def _load_correlation_rules(self) -> List[Dict[str, Any]]:
        """Load correlation rules from configuration"""
        default_rules = [
            {
                "name": "brute_force_attack",
                "type": "frequency",
                "event_type": SecurityEventType.LOGIN_FAILURE,
                "threshold": 5,
                "time_window_minutes": 10,
                "severity": "high",
                "description": "Multiple failed login attempts detected",
                "recommended_actions": [
                    "Block IP address",
                    "Notify user of suspicious activity",
                    "Check for credential stuffing"
                ]
            },
            {
                "name": "privilege_escalation",
                "type": "sequence",
                "sequence": [
                    SecurityEventType.LOGIN_SUCCESS,
                    SecurityEventType.PERMISSION_DENIED,
                    SecurityEventType.PERMISSION_GRANTED
                ],
                "severity": "critical",
                "description": "Potential privilege escalation detected",
                "recommended_actions": [
                    "Investigate user permissions",
                    "Check for unauthorized access",
                    "Review role assignments"
                ]
            },
            {
                "name": "data_exfiltration",
                "type": "combination",
                "event_types": [
                    SecurityEventType.DATA_ACCESS,
                    SecurityEventType.EXPORT_DATA,
                    SecurityEventType.SUSPICIOUS_ACTIVITY
                ],
                "severity": "critical",
                "description": "Potential data exfiltration detected",
                "recommended_actions": [
                    "Block user access",
                    "Investigate data access patterns",
                    "Contact security team immediately"
                ]
            }
        ]
        
        # Load additional rules from config
        custom_rules = self.config.get("correlation_rules", [])
        
        return default_rules + custom_rules
    
    def load_threat_patterns(self):
        """Load threat detection patterns"""
        self.threat_patterns = {
            "sql_injection": {
                "event_types": [SecurityEventType.SECURITY_VIOLATION],
                "min_threat_level": ThreatLevel.MEDIUM,
                "description": "SQL injection attempt detected",
                "severity": "high",
                "recommended_actions": [
                    "Block IP address",
                    "Check input validation",
                    "Review database queries"
                ]
            },
            "xss_attack": {
                "event_types": [SecurityEventType.SECURITY_VIOLATION],
                "min_threat_level": ThreatLevel.MEDIUM,
                "description": "Cross-site scripting attack detected",
                "severity": "high",
                "recommended_actions": [
                    "Block IP address",
                    "Check input sanitization",
                    "Review CSP headers"
                ]
            },
            "account_takeover": {
                "event_types": [SecurityEventType.LOGIN_SUCCESS],
                "min_threat_level": ThreatLevel.MEDIUM,
                "time_window_minutes": 60,
                "min_occurrences": 3,
                "description": "Potential account takeover detected",
                "severity": "critical",
                "recommended_actions": [
                    "Force password reset",
                    "Enable MFA",
                    "Check for credential compromise"
                ]
            }
        }
    
    def register_alert_handler(self, severity: AlertSeverity, handler: Callable):
        """Register alert handler for specific severity"""
        self.alert_handlers[severity].append(handler)
        self.logger.info(f"Registered alert handler for {severity.value} severity")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge security alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.assigned_to = acknowledged_by
            alert.updated_at = datetime.now()
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str, resolved_by: str):
        """Resolve security alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.assigned_to = resolved_by
            alert.updated_at = datetime.now()
            del self.active_alerts[alert_id]
            self.metrics.active_alerts -= 1
            self.metrics.resolved_alerts += 1
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
    
    def get_active_alerts(self) -> List[SecurityAlert]:
        """Get all active security alerts"""
        return list(self.active_alerts.values())
    
    def get_metrics(self) -> SecurityMetrics:
        """Get current security metrics"""
        return self.metrics
    
    def get_metrics_history(self) -> List[SecurityMetrics]:
        """Get historical security metrics"""
        return list(self.metrics_history)
    
    async def stop(self):
        """Stop security monitoring"""
        self.processing_enabled = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        self.logger.info("Security monitoring stopped")