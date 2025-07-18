"""
Audit Logger

Comprehensive audit logging system for security events, compliance,
and forensic analysis with secure, tamper-evident logging.
"""

import json
import logging
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from ..auth.models import SecurityEvent, SecurityEventType, ThreatLevel


class AuditEventType(Enum):
    """Types of audit events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    ADMIN_ACTION = "admin_action"
    USER_ACTION = "user_action"
    API_ACCESS = "api_access"


@dataclass
class AuditEvent:
    """Audit event model"""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"  # success, failure, partial
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    sequence_number: Optional[int] = None


@dataclass
class AuditLog:
    """Audit log container"""
    log_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[AuditEvent] = field(default_factory=list)
    integrity_hash: Optional[str] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """
    Comprehensive Audit Logging System with:
    - Tamper-evident logging
    - Integrity verification
    - Compliance reporting
    - Forensic analysis support
    - Secure log storage
    - Log rotation and archival
    - Real-time alerting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Audit configuration
        self.audit_enabled = self.config.get("audit_enabled", True)
        self.integrity_checking = self.config.get("integrity_checking", True)
        self.real_time_alerts = self.config.get("real_time_alerts", True)
        
        # Storage configuration
        self.log_file_path = self.config.get("log_file_path", "audit.log")
        self.max_log_size = self.config.get("max_log_size", 100 * 1024 * 1024)  # 100MB
        self.retention_days = self.config.get("retention_days", 2555)  # 7 years
        
        # Security configuration
        self.signing_key = self.config.get("signing_key") or secrets.token_hex(32)
        self.encryption_enabled = self.config.get("encryption_enabled", True)
        self.compression_enabled = self.config.get("compression_enabled", True)
        
        # Compliance configuration
        self.compliance_standards = self.config.get("compliance_standards", ["SOX", "PCI-DSS", "GDPR"])
        self.required_fields = self.config.get("required_fields", [
            "timestamp", "user_id", "action", "resource", "outcome"
        ])
        
        # Internal state
        self.sequence_counter = 0
        self.current_log = None
        self.event_queue = asyncio.Queue()
        self.processing_enabled = True
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_outcome": {"success": 0, "failure": 0, "partial": 0},
            "integrity_violations": 0,
            "compliance_violations": 0,
            "log_rotations": 0,
            "alerts_sent": 0
        }
        
        # Initialize audit logger
        self._initialize_audit_logger()
        
        # Start background tasks
        self.background_tasks = []
        if self.audit_enabled:
            self._start_background_tasks()
    
    def _initialize_audit_logger(self):
        """Initialize audit logging system"""
        # Create audit logger
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                self.log_file_path,
                maxBytes=self.max_log_size,
                backupCount=10
            )
        except ImportError:
            handler = logging.FileHandler(self.log_file_path)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.audit_logger.addHandler(handler)
        
        # Create new audit log
        self.current_log = AuditLog(
            log_id=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now()
        )
    
    def _start_background_tasks(self):
        """Start background audit tasks"""
        tasks = [
            self._event_processor(),
            self._integrity_checker(),
            self._log_rotator(),
            self._compliance_monitor(),
            self._cleanup_task()
        ]
        
        for task in tasks:
            self.background_tasks.append(asyncio.create_task(task))
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log audit event"""
        if not self.audit_enabled:
            return
        
        # Generate event ID
        event_id = self._generate_event_id()
        
        # Create audit event
        audit_event = AuditEvent(
            id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            username=username,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            metadata=metadata or {},
            sequence_number=self._get_next_sequence_number()
        )
        
        # Add integrity checksum
        if self.integrity_checking:
            audit_event.checksum = self._calculate_checksum(audit_event)
        
        # Validate event
        if not self._validate_event(audit_event):
            self.logger.warning(f"Invalid audit event: {event_id}")
            return
        
        # Queue event for processing
        await self.event_queue.put(audit_event)
    
    async def log_security_event(self, security_event: SecurityEvent):
        """Log security event to audit log"""
        await self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            user_id=str(security_event.user_id) if security_event.user_id else None,
            username=security_event.username,
            ip_address=security_event.ip_address,
            user_agent=security_event.user_agent,
            resource=security_event.resource,
            action=security_event.action,
            outcome=security_event.result,
            details={
                "event_type": security_event.event_type.value,
                "threat_level": security_event.threat_level.value,
                "message": security_event.message
            },
            metadata=security_event.metadata
        )
    
    async def _event_processor(self):
        """Background event processor"""
        while self.processing_enabled:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Event processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: AuditEvent):
        """Process individual audit event"""
        # Update statistics
        self.stats["total_events"] += 1
        
        event_type_str = event.event_type.value
        if event_type_str not in self.stats["events_by_type"]:
            self.stats["events_by_type"][event_type_str] = 0
        self.stats["events_by_type"][event_type_str] += 1
        
        self.stats["events_by_outcome"][event.outcome] += 1
        
        # Add to current log
        self.current_log.events.append(event)
        
        # Write to audit log file
        await self._write_to_audit_log(event)
        
        # Check for compliance violations
        if await self._check_compliance_violations(event):
            self.stats["compliance_violations"] += 1
        
        # Send real-time alerts if configured
        if self.real_time_alerts and await self._should_alert(event):
            await self._send_alert(event)
    
    async def _write_to_audit_log(self, event: AuditEvent):
        """Write event to audit log file"""
        try:
            # Serialize event
            log_entry = self._serialize_event(event)
            
            # Write to log
            self.audit_logger.info(log_entry)
            
            # Also write to structured log if configured
            if self.config.get("structured_logging", False):
                await self._write_structured_log(event)
            
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def _serialize_event(self, event: AuditEvent) -> str:
        """Serialize audit event for logging"""
        event_dict = {
            "id": event.id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "username": event.username,
            "session_id": event.session_id,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "resource": event.resource,
            "action": event.action,
            "outcome": event.outcome,
            "details": event.details,
            "metadata": event.metadata,
            "sequence_number": event.sequence_number,
            "checksum": event.checksum
        }
        
        # Remove None values
        event_dict = {k: v for k, v in event_dict.items() if v is not None}
        
        return json.dumps(event_dict)
    
    async def _write_structured_log(self, event: AuditEvent):
        """Write to structured log storage (database, elasticsearch, etc.)"""
        # This would implement structured logging to databases or search engines
        pass
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_hex(8)
        return f"audit_{timestamp}_{random_part}"
    
    def _get_next_sequence_number(self) -> int:
        """Get next sequence number for event ordering"""
        self.sequence_counter += 1
        return self.sequence_counter
    
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate integrity checksum for event"""
        # Create canonical representation
        canonical_data = {
            "id": event.id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "username": event.username,
            "resource": event.resource,
            "action": event.action,
            "outcome": event.outcome,
            "sequence_number": event.sequence_number
        }
        
        # Convert to string
        canonical_str = json.dumps(canonical_data, sort_keys=True)
        
        # Calculate HMAC
        return hmac.new(
            self.signing_key.encode(),
            canonical_str.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _validate_event(self, event: AuditEvent) -> bool:
        """Validate audit event"""
        # Check required fields
        for field in self.required_fields:
            if not hasattr(event, field) or getattr(event, field) is None:
                return False
        
        # Validate checksum if present
        if event.checksum and self.integrity_checking:
            expected_checksum = self._calculate_checksum(event)
            if event.checksum != expected_checksum:
                self.stats["integrity_violations"] += 1
                return False
        
        # Additional validation rules
        if event.event_type not in AuditEventType:
            return False
        
        if event.outcome not in ["success", "failure", "partial"]:
            return False
        
        return True
    
    async def _check_compliance_violations(self, event: AuditEvent) -> bool:
        """Check for compliance violations"""
        violations = []
        
        # SOX compliance checks
        if "SOX" in self.compliance_standards:
            if event.event_type == AuditEventType.CONFIGURATION_CHANGE:
                if not event.details.get("change_approval"):
                    violations.append("SOX: Configuration change without approval")
        
        # PCI-DSS compliance checks
        if "PCI-DSS" in self.compliance_standards:
            if event.event_type == AuditEventType.DATA_ACCESS:
                if "credit_card" in str(event.resource).lower():
                    if not event.details.get("authorized_access"):
                        violations.append("PCI-DSS: Unauthorized credit card data access")
        
        # GDPR compliance checks
        if "GDPR" in self.compliance_standards:
            if event.event_type == AuditEventType.DATA_ACCESS:
                if "personal_data" in event.details.get("data_type", ""):
                    if not event.details.get("consent_verified"):
                        violations.append("GDPR: Personal data access without consent")
        
        # Log violations
        if violations:
            self.logger.warning(f"Compliance violations: {violations}")
            return True
        
        return False
    
    async def _should_alert(self, event: AuditEvent) -> bool:
        """Check if event should trigger real-time alert"""
        # Alert on failures
        if event.outcome == "failure":
            return True
        
        # Alert on admin actions
        if event.event_type == AuditEventType.ADMIN_ACTION:
            return True
        
        # Alert on configuration changes
        if event.event_type == AuditEventType.CONFIGURATION_CHANGE:
            return True
        
        # Alert on high-privilege actions
        if event.details.get("privilege_level") == "high":
            return True
        
        return False
    
    async def _send_alert(self, event: AuditEvent):
        """Send real-time alert for event"""
        self.stats["alerts_sent"] += 1
        
        alert_data = {
            "alert_type": "audit_event",
            "event_id": event.id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user": event.username or event.user_id,
            "action": event.action,
            "resource": event.resource,
            "outcome": event.outcome,
            "severity": self._determine_alert_severity(event)
        }
        
        # Send alert (implementation depends on alerting system)
        self.logger.info(f"AUDIT ALERT: {json.dumps(alert_data)}")
    
    def _determine_alert_severity(self, event: AuditEvent) -> str:
        """Determine alert severity based on event"""
        if event.outcome == "failure":
            return "high"
        elif event.event_type == AuditEventType.ADMIN_ACTION:
            return "medium"
        elif event.event_type == AuditEventType.CONFIGURATION_CHANGE:
            return "medium"
        else:
            return "low"
    
    async def _integrity_checker(self):
        """Background integrity checker"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Verify log integrity
                await self._verify_log_integrity()
                
            except Exception as e:
                self.logger.error(f"Integrity checker error: {e}")
    
    async def _verify_log_integrity(self):
        """Verify audit log integrity"""
        if not self.current_log.events:
            return
        
        # Check event sequence
        for i, event in enumerate(self.current_log.events):
            if event.sequence_number != i + 1:
                self.logger.error(f"Sequence number mismatch: expected {i + 1}, got {event.sequence_number}")
                self.stats["integrity_violations"] += 1
        
        # Verify checksums
        for event in self.current_log.events:
            if event.checksum:
                expected_checksum = self._calculate_checksum(event)
                if event.checksum != expected_checksum:
                    self.logger.error(f"Checksum mismatch for event {event.id}")
                    self.stats["integrity_violations"] += 1
    
    async def _log_rotator(self):
        """Background log rotation"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                # Rotate logs if needed
                if await self._should_rotate_log():
                    await self._rotate_log()
                
            except Exception as e:
                self.logger.error(f"Log rotator error: {e}")
    
    async def _should_rotate_log(self) -> bool:
        """Check if log should be rotated"""
        # Rotate daily
        if self.current_log.start_time.date() != datetime.now().date():
            return True
        
        # Rotate if too many events
        if len(self.current_log.events) > 100000:
            return True
        
        return False
    
    async def _rotate_log(self):
        """Rotate audit log"""
        # Finalize current log
        self.current_log.end_time = datetime.now()
        self.current_log.integrity_hash = self._calculate_log_hash()
        
        # Archive current log
        await self._archive_log(self.current_log)
        
        # Create new log
        self.current_log = AuditLog(
            log_id=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now()
        )
        
        self.stats["log_rotations"] += 1
        self.logger.info("Audit log rotated")
    
    def _calculate_log_hash(self) -> str:
        """Calculate integrity hash for entire log"""
        # Create hash of all event checksums
        event_checksums = [event.checksum for event in self.current_log.events if event.checksum]
        combined_checksums = "".join(event_checksums)
        
        return hashlib.sha256(combined_checksums.encode()).hexdigest()
    
    async def _archive_log(self, log: AuditLog):
        """Archive completed audit log"""
        # This would implement log archival to secure storage
        pass
    
    async def _compliance_monitor(self):
        """Background compliance monitoring"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Generate compliance reports
                await self._generate_compliance_reports()
                
            except Exception as e:
                self.logger.error(f"Compliance monitor error: {e}")
    
    async def _generate_compliance_reports(self):
        """Generate compliance reports"""
        # This would generate reports for different compliance standards
        pass
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Clean up old audit data
                await self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old audit data"""
        # Clean up events older than retention period
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # In a real implementation, this would clean up archived logs
        # and old events from databases
        pass
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events"""
        # Filter events from current log
        filtered_events = []
        
        for event in self.current_log.events:
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if username and event.username != username:
                continue
            if resource and event.resource != resource:
                continue
            if action and event.action != action:
                continue
            if outcome and event.outcome != outcome:
                continue
            
            filtered_events.append(event)
            
            if len(filtered_events) >= limit:
                break
        
        return filtered_events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        return {
            "statistics": self.stats.copy(),
            "configuration": {
                "audit_enabled": self.audit_enabled,
                "integrity_checking": self.integrity_checking,
                "real_time_alerts": self.real_time_alerts,
                "compliance_standards": self.compliance_standards,
                "retention_days": self.retention_days
            },
            "current_log": {
                "log_id": self.current_log.log_id,
                "start_time": self.current_log.start_time.isoformat(),
                "events_count": len(self.current_log.events),
                "sequence_counter": self.sequence_counter
            }
        }
    
    async def stop(self):
        """Stop audit logging"""
        self.processing_enabled = False
        
        # Process remaining events
        while not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                break
        
        # Finalize current log
        if self.current_log.events:
            await self._rotate_log()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        self.logger.info("Audit logging stopped")