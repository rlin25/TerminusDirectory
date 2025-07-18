"""
Threat Detector

Advanced threat detection system using machine learning and rule-based
approaches to identify security threats and anomalies.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from ..auth.models import SecurityEvent, SecurityEventType, ThreatLevel


class ThreatType(Enum):
    """Types of threats that can be detected"""
    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    ACCOUNT_TAKEOVER = "account_takeover"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    INSIDER_THREAT = "insider_threat"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    MALWARE_ACTIVITY = "malware_activity"
    RECONNAISSANCE = "reconnaissance"
    DENIAL_OF_SERVICE = "denial_of_service"


@dataclass
class ThreatDetection:
    """Threat detection result"""
    threat_type: ThreatType
    confidence: float
    severity: ThreatLevel
    description: str
    indicators: List[str]
    affected_entities: List[str]
    source_events: List[SecurityEvent]
    detection_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorProfile:
    """User behavior profile for anomaly detection"""
    user_id: str
    username: str
    typical_login_times: List[int] = field(default_factory=list)  # Hours of day
    typical_locations: Set[str] = field(default_factory=set)  # IP addresses/locations
    typical_user_agents: Set[str] = field(default_factory=set)
    typical_resources: Set[str] = field(default_factory=set)  # Accessed resources
    average_session_duration: float = 0.0
    login_frequency: float = 0.0  # Logins per day
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    profile_version: int = 1


class ThreatDetector:
    """
    Advanced Threat Detection System with:
    - Machine learning-based anomaly detection
    - Behavioral analysis
    - Pattern recognition
    - Real-time threat scoring
    - Adaptive learning
    - False positive reduction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Detection configuration
        self.detection_enabled = self.config.get("detection_enabled", True)
        self.ml_detection_enabled = self.config.get("ml_detection_enabled", True)
        self.behavioral_analysis_enabled = self.config.get("behavioral_analysis_enabled", True)
        
        # Thresholds
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.7)
        self.threat_confidence_threshold = self.config.get("threat_confidence_threshold", 0.8)
        self.max_events_for_analysis = self.config.get("max_events_for_analysis", 10000)
        
        # Data storage
        self.event_history = deque(maxlen=self.max_events_for_analysis)
        self.behavior_profiles: Dict[str, BehaviorProfile] = {}
        self.threat_detections: List[ThreatDetection] = []
        
        # Rule-based detection
        self.detection_rules = self._load_detection_rules()
        
        # ML models (simplified - in production use proper ML framework)
        self.anomaly_detector = None
        self.behavior_analyzer = None
        
        # Statistics
        self.stats = {
            "total_events_analyzed": 0,
            "threats_detected": 0,
            "false_positives": 0,
            "true_positives": 0,
            "behavior_profiles_created": 0,
            "behavior_profiles_updated": 0
        }
        
        # Background tasks
        self.background_tasks = []
        if self.detection_enabled:
            self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background detection tasks"""
        tasks = [
            self._behavior_profiler(),
            self._threat_analyzer(),
            self._model_trainer(),
            self._cleanup_task()
        ]
        
        for task in tasks:
            self.background_tasks.append(asyncio.create_task(task))
    
    async def analyze_event(self, event: SecurityEvent) -> List[ThreatDetection]:
        """Analyze security event for threats"""
        if not self.detection_enabled:
            return []
        
        self.stats["total_events_analyzed"] += 1
        self.event_history.append(event)
        
        detections = []
        
        # Rule-based detection
        rule_detections = await self._rule_based_detection(event)
        detections.extend(rule_detections)
        
        # Behavioral analysis
        if self.behavioral_analysis_enabled:
            behavioral_detections = await self._behavioral_analysis(event)
            detections.extend(behavioral_detections)
        
        # ML-based detection
        if self.ml_detection_enabled:
            ml_detections = await self._ml_based_detection(event)
            detections.extend(ml_detections)
        
        # Store detections
        for detection in detections:
            self.threat_detections.append(detection)
            self.stats["threats_detected"] += 1
            
            self.logger.warning(
                f"Threat detected: {detection.threat_type.value} "
                f"(Confidence: {detection.confidence:.2f}) - {detection.description}"
            )
        
        return detections
    
    async def _rule_based_detection(self, event: SecurityEvent) -> List[ThreatDetection]:
        """Rule-based threat detection"""
        detections = []
        
        for rule in self.detection_rules:
            if await self._evaluate_rule(event, rule):
                detection = ThreatDetection(
                    threat_type=ThreatType(rule["threat_type"]),
                    confidence=rule.get("confidence", 0.8),
                    severity=ThreatLevel(rule.get("severity", "medium")),
                    description=rule["description"],
                    indicators=rule.get("indicators", []),
                    affected_entities=self._extract_affected_entities(event),
                    source_events=[event],
                    metadata={"rule_name": rule["name"]}
                )
                detections.append(detection)
        
        return detections
    
    async def _evaluate_rule(self, event: SecurityEvent, rule: Dict[str, Any]) -> bool:
        """Evaluate detection rule against event"""
        conditions = rule.get("conditions", [])
        
        for condition in conditions:
            condition_type = condition.get("type")
            
            if condition_type == "event_type":
                if event.event_type != SecurityEventType(condition["value"]):
                    return False
            
            elif condition_type == "threat_level":
                min_level = ThreatLevel(condition["value"])
                threat_levels = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
                if threat_levels.index(event.threat_level) < threat_levels.index(min_level):
                    return False
            
            elif condition_type == "frequency":
                # Check frequency of similar events
                time_window = timedelta(minutes=condition.get("time_window_minutes", 10))
                similar_events = [
                    e for e in self.event_history
                    if (e.event_type == event.event_type and 
                        datetime.now() - e.timestamp < time_window)
                ]
                
                if len(similar_events) < condition.get("threshold", 5):
                    return False
            
            elif condition_type == "pattern":
                # Check for specific patterns in event data
                pattern = condition.get("pattern", "")
                if pattern not in (event.message or ""):
                    return False
        
        return True
    
    async def _behavioral_analysis(self, event: SecurityEvent) -> List[ThreatDetection]:
        """Behavioral analysis for anomaly detection"""
        detections = []
        
        if not event.username:
            return detections
        
        # Get or create behavior profile
        profile = self.behavior_profiles.get(event.username)
        if not profile:
            profile = BehaviorProfile(
                user_id=event.user_id or event.username,
                username=event.username
            )
            self.behavior_profiles[event.username] = profile
            self.stats["behavior_profiles_created"] += 1
        
        # Analyze behavior
        anomalies = await self._analyze_user_behavior(event, profile)
        
        for anomaly in anomalies:
            detection = ThreatDetection(
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                confidence=anomaly["confidence"],
                severity=ThreatLevel.MEDIUM,
                description=anomaly["description"],
                indicators=anomaly["indicators"],
                affected_entities=[event.username],
                source_events=[event],
                metadata={"anomaly_type": anomaly["type"]}
            )
            detections.append(detection)
        
        # Update profile
        await self._update_behavior_profile(event, profile)
        
        return detections
    
    async def _analyze_user_behavior(self, event: SecurityEvent, profile: BehaviorProfile) -> List[Dict[str, Any]]:
        """Analyze user behavior against profile"""
        anomalies = []
        
        # Check login time anomaly
        if event.event_type == SecurityEventType.LOGIN_SUCCESS:
            current_hour = event.timestamp.hour
            
            if profile.typical_login_times:
                # Calculate deviation from typical times
                time_deviations = [abs(current_hour - t) for t in profile.typical_login_times]
                min_deviation = min(time_deviations)
                
                if min_deviation > 4:  # More than 4 hours from typical
                    anomalies.append({
                        "type": "unusual_login_time",
                        "confidence": min(min_deviation / 12, 1.0),
                        "description": f"Login at unusual time: {current_hour:02d}:00",
                        "indicators": [f"typical_times: {profile.typical_login_times}"]
                    })
        
        # Check location anomaly
        if event.ip_address and profile.typical_locations:
            if event.ip_address not in profile.typical_locations:
                anomalies.append({
                    "type": "unusual_location",
                    "confidence": 0.7,
                    "description": f"Login from unusual location: {event.ip_address}",
                    "indicators": [f"typical_locations: {len(profile.typical_locations)}"]
                })
        
        # Check user agent anomaly
        if event.user_agent and profile.typical_user_agents:
            if event.user_agent not in profile.typical_user_agents:
                anomalies.append({
                    "type": "unusual_user_agent",
                    "confidence": 0.6,
                    "description": "Login from unusual device/browser",
                    "indicators": [f"new_user_agent: {event.user_agent[:50]}..."]
                })
        
        # Check resource access anomaly
        if event.resource and profile.typical_resources:
            if event.resource not in profile.typical_resources:
                anomalies.append({
                    "type": "unusual_resource_access",
                    "confidence": 0.5,
                    "description": f"Access to unusual resource: {event.resource}",
                    "indicators": [f"typical_resources: {len(profile.typical_resources)}"]
                })
        
        return anomalies
    
    async def _update_behavior_profile(self, event: SecurityEvent, profile: BehaviorProfile):
        """Update user behavior profile with new event"""
        self.stats["behavior_profiles_updated"] += 1
        
        # Update login times
        if event.event_type == SecurityEventType.LOGIN_SUCCESS:
            hour = event.timestamp.hour
            if len(profile.typical_login_times) < 24:
                profile.typical_login_times.append(hour)
            else:
                # Use moving average approach
                profile.typical_login_times = profile.typical_login_times[-23:] + [hour]
        
        # Update typical locations
        if event.ip_address:
            profile.typical_locations.add(event.ip_address)
            # Keep only recent locations (max 10)
            if len(profile.typical_locations) > 10:
                profile.typical_locations = set(list(profile.typical_locations)[-10:])
        
        # Update typical user agents
        if event.user_agent:
            profile.typical_user_agents.add(event.user_agent)
            # Keep only recent user agents (max 5)
            if len(profile.typical_user_agents) > 5:
                profile.typical_user_agents = set(list(profile.typical_user_agents)[-5:])
        
        # Update typical resources
        if event.resource:
            profile.typical_resources.add(event.resource)
            # Keep only recent resources (max 50)
            if len(profile.typical_resources) > 50:
                profile.typical_resources = set(list(profile.typical_resources)[-50:])
        
        profile.updated_at = datetime.now()
    
    async def _ml_based_detection(self, event: SecurityEvent) -> List[ThreatDetection]:
        """Machine learning-based threat detection"""
        detections = []
        
        # This is a simplified ML approach
        # In production, you'd use proper ML models
        
        # Feature extraction
        features = self._extract_features(event)
        
        # Anomaly detection
        anomaly_score = self._calculate_anomaly_score(features)
        
        if anomaly_score > self.anomaly_threshold:
            detection = ThreatDetection(
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                confidence=anomaly_score,
                severity=self._score_to_severity(anomaly_score),
                description=f"ML-based anomaly detected (score: {anomaly_score:.2f})",
                indicators=[f"anomaly_score: {anomaly_score:.2f}"],
                affected_entities=self._extract_affected_entities(event),
                source_events=[event],
                metadata={"ml_model": "anomaly_detector", "features": features}
            )
            detections.append(detection)
        
        return detections
    
    def _extract_features(self, event: SecurityEvent) -> Dict[str, float]:
        """Extract features from security event for ML analysis"""
        features = {
            "event_type_numeric": list(SecurityEventType).index(event.event_type),
            "threat_level_numeric": list(ThreatLevel).index(event.threat_level),
            "hour_of_day": event.timestamp.hour,
            "day_of_week": event.timestamp.weekday(),
            "message_length": len(event.message or ""),
            "has_username": 1.0 if event.username else 0.0,
            "has_ip_address": 1.0 if event.ip_address else 0.0,
            "has_user_agent": 1.0 if event.user_agent else 0.0,
            "has_resource": 1.0 if event.resource else 0.0,
        }
        
        # Add contextual features
        recent_events = [
            e for e in self.event_history
            if datetime.now() - e.timestamp < timedelta(hours=1)
        ]
        
        features.update({
            "recent_events_count": len(recent_events),
            "recent_failures_count": len([e for e in recent_events if e.event_type == SecurityEventType.LOGIN_FAILURE]),
            "recent_successes_count": len([e for e in recent_events if e.event_type == SecurityEventType.LOGIN_SUCCESS]),
        })
        
        return features
    
    def _calculate_anomaly_score(self, features: Dict[str, float]) -> float:
        """Calculate anomaly score (simplified implementation)"""
        # This is a very simplified anomaly scoring
        # In production, you'd use proper ML models
        
        score = 0.0
        
        # High activity score
        if features["recent_events_count"] > 50:
            score += 0.3
        
        # High failure rate
        if features["recent_failures_count"] > 10:
            score += 0.4
        
        # Unusual time
        hour = features["hour_of_day"]
        if hour < 6 or hour > 22:  # Outside normal hours
            score += 0.2
        
        # Weekend activity
        if features["day_of_week"] >= 5:  # Weekend
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_to_severity(self, score: float) -> ThreatLevel:
        """Convert anomaly score to threat severity"""
        if score > 0.9:
            return ThreatLevel.CRITICAL
        elif score > 0.8:
            return ThreatLevel.HIGH
        elif score > 0.7:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _extract_affected_entities(self, event: SecurityEvent) -> List[str]:
        """Extract affected entities from event"""
        entities = []
        
        if event.username:
            entities.append(f"user:{event.username}")
        
        if event.ip_address:
            entities.append(f"ip:{event.ip_address}")
        
        if event.resource:
            entities.append(f"resource:{event.resource}")
        
        return entities
    
    async def _behavior_profiler(self):
        """Background behavior profiling task"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Update behavior profiles based on recent events
                await self._update_behavior_profiles()
                
            except Exception as e:
                self.logger.error(f"Behavior profiler error: {e}")
    
    async def _update_behavior_profiles(self):
        """Update behavior profiles from recent events"""
        # This would contain more sophisticated profiling logic
        pass
    
    async def _threat_analyzer(self):
        """Background threat analysis task"""
        while True:
            try:
                await asyncio.sleep(120)  # Run every 2 minutes
                
                # Perform batch analysis of recent events
                await self._batch_threat_analysis()
                
            except Exception as e:
                self.logger.error(f"Threat analyzer error: {e}")
    
    async def _batch_threat_analysis(self):
        """Batch analysis of recent events"""
        # Analyze patterns across multiple events
        recent_events = [
            e for e in self.event_history
            if datetime.now() - e.timestamp < timedelta(minutes=10)
        ]
        
        if len(recent_events) < 5:
            return
        
        # Look for coordinated attacks
        await self._detect_coordinated_attacks(recent_events)
        
        # Look for data exfiltration patterns
        await self._detect_data_exfiltration(recent_events)
    
    async def _detect_coordinated_attacks(self, events: List[SecurityEvent]):
        """Detect coordinated attacks across multiple events"""
        # Group events by IP address
        ip_groups = defaultdict(list)
        for event in events:
            if event.ip_address:
                ip_groups[event.ip_address].append(event)
        
        # Look for IPs with multiple types of suspicious activity
        for ip, ip_events in ip_groups.items():
            if len(ip_events) > 5:
                event_types = set(e.event_type for e in ip_events)
                if len(event_types) > 2:
                    # Multiple event types from same IP
                    detection = ThreatDetection(
                        threat_type=ThreatType.RECONNAISSANCE,
                        confidence=0.8,
                        severity=ThreatLevel.HIGH,
                        description=f"Coordinated attack detected from IP {ip}",
                        indicators=[f"event_types: {[et.value for et in event_types]}"],
                        affected_entities=[f"ip:{ip}"],
                        source_events=ip_events,
                        metadata={"analysis_type": "coordinated_attack"}
                    )
                    self.threat_detections.append(detection)
    
    async def _detect_data_exfiltration(self, events: List[SecurityEvent]):
        """Detect potential data exfiltration patterns"""
        # Look for unusual data access patterns
        data_access_events = [
            e for e in events
            if e.event_type == SecurityEventType.DATA_ACCESS
        ]
        
        if len(data_access_events) > 10:
            # High volume of data access
            detection = ThreatDetection(
                threat_type=ThreatType.DATA_EXFILTRATION,
                confidence=0.7,
                severity=ThreatLevel.HIGH,
                description=f"Potential data exfiltration detected ({len(data_access_events)} access events)",
                indicators=[f"data_access_count: {len(data_access_events)}"],
                affected_entities=list(set(e.username for e in data_access_events if e.username)),
                source_events=data_access_events,
                metadata={"analysis_type": "data_exfiltration"}
            )
            self.threat_detections.append(detection)
    
    async def _model_trainer(self):
        """Background ML model training task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Retrain ML models with new data
                await self._retrain_models()
                
            except Exception as e:
                self.logger.error(f"Model trainer error: {e}")
    
    async def _retrain_models(self):
        """Retrain ML models with recent data"""
        # This would implement model retraining logic
        pass
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old threat detections
                cutoff_time = datetime.now() - timedelta(days=7)
                self.threat_detections = [
                    d for d in self.threat_detections
                    if d.detection_time > cutoff_time
                ]
                
                # Clean up old behavior profiles
                cutoff_time = datetime.now() - timedelta(days=30)
                inactive_profiles = [
                    username for username, profile in self.behavior_profiles.items()
                    if profile.updated_at < cutoff_time
                ]
                
                for username in inactive_profiles:
                    del self.behavior_profiles[username]
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    def _load_detection_rules(self) -> List[Dict[str, Any]]:
        """Load threat detection rules"""
        return [
            {
                "name": "multiple_failed_logins",
                "threat_type": "brute_force",
                "confidence": 0.8,
                "severity": "high",
                "description": "Multiple failed login attempts detected",
                "conditions": [
                    {"type": "event_type", "value": "login_failure"},
                    {"type": "frequency", "threshold": 5, "time_window_minutes": 10}
                ],
                "indicators": ["failed_login_attempts"]
            },
            {
                "name": "privilege_escalation_attempt",
                "threat_type": "privilege_escalation",
                "confidence": 0.9,
                "severity": "critical",
                "description": "Privilege escalation attempt detected",
                "conditions": [
                    {"type": "event_type", "value": "permission_denied"},
                    {"type": "threat_level", "value": "high"}
                ],
                "indicators": ["permission_denied", "privilege_escalation"]
            },
            {
                "name": "suspicious_data_access",
                "threat_type": "data_exfiltration",
                "confidence": 0.7,
                "severity": "high",
                "description": "Suspicious data access pattern detected",
                "conditions": [
                    {"type": "event_type", "value": "data_access"},
                    {"type": "frequency", "threshold": 10, "time_window_minutes": 5}
                ],
                "indicators": ["high_volume_data_access"]
            }
        ]
    
    def get_threat_detections(self, limit: int = 100) -> List[ThreatDetection]:
        """Get recent threat detections"""
        return self.threat_detections[-limit:]
    
    def get_behavior_profile(self, username: str) -> Optional[BehaviorProfile]:
        """Get behavior profile for user"""
        return self.behavior_profiles.get(username)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        return {
            "statistics": self.stats.copy(),
            "configuration": {
                "detection_enabled": self.detection_enabled,
                "ml_detection_enabled": self.ml_detection_enabled,
                "behavioral_analysis_enabled": self.behavioral_analysis_enabled,
                "anomaly_threshold": self.anomaly_threshold,
                "threat_confidence_threshold": self.threat_confidence_threshold
            },
            "current_state": {
                "event_history_size": len(self.event_history),
                "behavior_profiles_count": len(self.behavior_profiles),
                "threat_detections_count": len(self.threat_detections),
                "detection_rules_count": len(self.detection_rules)
            }
        }
    
    async def stop(self):
        """Stop threat detection"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        self.logger.info("Threat detection stopped")