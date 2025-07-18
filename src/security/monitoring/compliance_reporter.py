"""
Compliance Reporter

Automated compliance reporting system for various security standards
including SOX, PCI-DSS, GDPR, HIPAA, and other regulatory requirements.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .audit_logger import AuditEvent, AuditEventType
from ..auth.models import SecurityEvent, SecurityEventType


class ComplianceStandard(Enum):
    """Supported compliance standards"""
    SOX = "sox"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CCPA = "ccpa"
    FISMA = "fisma"


@dataclass
class ComplianceRequirement:
    """Compliance requirement definition"""
    id: str
    standard: ComplianceStandard
    title: str
    description: str
    category: str
    severity: str  # critical, high, medium, low
    automated_check: bool = True
    check_frequency: str = "daily"  # daily, weekly, monthly
    evidence_types: List[str] = field(default_factory=list)
    control_objectives: List[str] = field(default_factory=list)


@dataclass
class ComplianceEvidence:
    """Evidence supporting compliance"""
    requirement_id: str
    evidence_type: str
    timestamp: datetime
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "automated"
    verified: bool = False
    verifier: Optional[str] = None


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    id: str
    requirement_id: str
    standard: ComplianceStandard
    severity: str
    description: str
    detected_at: datetime
    evidence: List[ComplianceEvidence] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


@dataclass
class ComplianceReport:
    """Compliance report"""
    id: str
    standard: ComplianceStandard
    report_period_start: datetime
    report_period_end: datetime
    generated_at: datetime
    total_requirements: int
    compliant_requirements: int
    violations: List[ComplianceViolation] = field(default_factory=list)
    evidence_summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceReporter:
    """
    Compliance Reporter with:
    - Multi-standard compliance monitoring
    - Automated evidence collection
    - Violation detection and tracking
    - Periodic compliance reporting
    - Remediation tracking
    - Audit trail generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Configuration
        self.enabled_standards = [
            ComplianceStandard(std) for std in self.config.get("enabled_standards", ["sox", "pci_dss", "gdpr"])
        ]
        self.auto_reporting = self.config.get("auto_reporting", True)
        self.report_frequency = self.config.get("report_frequency", "monthly")
        
        # Data storage
        self.requirements = self._load_requirements()
        self.evidence_store: List[ComplianceEvidence] = []
        self.violations: List[ComplianceViolation] = []
        self.reports: List[ComplianceReport] = []
        
        # Processing state
        self.processing_enabled = True
        self.last_report_dates = {}
        
        # Statistics
        self.stats = {
            "total_requirements": len(self.requirements),
            "active_violations": 0,
            "resolved_violations": 0,
            "evidence_collected": 0,
            "reports_generated": 0,
            "compliance_score": 0.0
        }
        
        # Background tasks
        self.background_tasks = []
        if self.auto_reporting:
            self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background compliance tasks"""
        tasks = [
            self._compliance_monitor(),
            self._evidence_collector(),
            self._report_generator(),
            self._violation_tracker(),
            self._cleanup_task()
        ]
        
        for task in tasks:
            self.background_tasks.append(asyncio.create_task(task))
    
    def _load_requirements(self) -> List[ComplianceRequirement]:
        """Load compliance requirements for enabled standards"""
        requirements = []
        
        for standard in self.enabled_standards:
            if standard == ComplianceStandard.SOX:
                requirements.extend(self._load_sox_requirements())
            elif standard == ComplianceStandard.PCI_DSS:
                requirements.extend(self._load_pci_dss_requirements())
            elif standard == ComplianceStandard.GDPR:
                requirements.extend(self._load_gdpr_requirements())
            elif standard == ComplianceStandard.HIPAA:
                requirements.extend(self._load_hipaa_requirements())
            elif standard == ComplianceStandard.ISO_27001:
                requirements.extend(self._load_iso27001_requirements())
        
        return requirements
    
    def _load_sox_requirements(self) -> List[ComplianceRequirement]:
        """Load SOX compliance requirements"""
        return [
            ComplianceRequirement(
                id="sox_404_internal_controls",
                standard=ComplianceStandard.SOX,
                title="Internal Controls Over Financial Reporting",
                description="Maintain adequate internal controls over financial reporting",
                category="financial_controls",
                severity="critical",
                evidence_types=["audit_logs", "access_reviews", "change_management"],
                control_objectives=["access_control", "segregation_of_duties", "change_approval"]
            ),
            ComplianceRequirement(
                id="sox_302_disclosure_controls",
                standard=ComplianceStandard.SOX,
                title="Disclosure Controls and Procedures",
                description="Maintain disclosure controls and procedures",
                category="disclosure",
                severity="high",
                evidence_types=["audit_logs", "data_access_logs"],
                control_objectives=["data_integrity", "access_control"]
            ),
            ComplianceRequirement(
                id="sox_906_audit_trail",
                standard=ComplianceStandard.SOX,
                title="Audit Trail Requirements",
                description="Maintain comprehensive audit trails",
                category="audit_logging",
                severity="critical",
                evidence_types=["audit_logs", "system_logs"],
                control_objectives=["auditability", "non_repudiation"]
            )
        ]
    
    def _load_pci_dss_requirements(self) -> List[ComplianceRequirement]:
        """Load PCI-DSS compliance requirements"""
        return [
            ComplianceRequirement(
                id="pci_dss_req_2",
                standard=ComplianceStandard.PCI_DSS,
                title="Do not use vendor-supplied defaults",
                description="Change vendor-supplied defaults and remove unnecessary accounts",
                category="configuration",
                severity="high",
                evidence_types=["configuration_logs", "account_reviews"],
                control_objectives=["secure_configuration", "account_management"]
            ),
            ComplianceRequirement(
                id="pci_dss_req_7",
                standard=ComplianceStandard.PCI_DSS,
                title="Restrict access to cardholder data",
                description="Restrict access to cardholder data by business need-to-know",
                category="access_control",
                severity="critical",
                evidence_types=["access_logs", "authorization_logs"],
                control_objectives=["access_control", "least_privilege"]
            ),
            ComplianceRequirement(
                id="pci_dss_req_10",
                standard=ComplianceStandard.PCI_DSS,
                title="Track and monitor all access",
                description="Track and monitor all access to network resources and cardholder data",
                category="monitoring",
                severity="critical",
                evidence_types=["audit_logs", "security_logs"],
                control_objectives=["monitoring", "audit_trail"]
            )
        ]
    
    def _load_gdpr_requirements(self) -> List[ComplianceRequirement]:
        """Load GDPR compliance requirements"""
        return [
            ComplianceRequirement(
                id="gdpr_art_32_security",
                standard=ComplianceStandard.GDPR,
                title="Security of processing",
                description="Implement appropriate technical and organizational measures",
                category="security",
                severity="critical",
                evidence_types=["security_logs", "access_logs", "encryption_logs"],
                control_objectives=["data_protection", "access_control", "encryption"]
            ),
            ComplianceRequirement(
                id="gdpr_art_30_records",
                standard=ComplianceStandard.GDPR,
                title="Records of processing activities",
                description="Maintain records of processing activities",
                category="documentation",
                severity="high",
                evidence_types=["audit_logs", "data_processing_logs"],
                control_objectives=["documentation", "audit_trail"]
            ),
            ComplianceRequirement(
                id="gdpr_art_17_erasure",
                standard=ComplianceStandard.GDPR,
                title="Right to erasure",
                description="Implement right to erasure (right to be forgotten)",
                category="data_rights",
                severity="high",
                evidence_types=["deletion_logs", "audit_logs"],
                control_objectives=["data_erasure", "audit_trail"]
            )
        ]
    
    def _load_hipaa_requirements(self) -> List[ComplianceRequirement]:
        """Load HIPAA compliance requirements"""
        return [
            ComplianceRequirement(
                id="hipaa_164_308_admin_safeguards",
                standard=ComplianceStandard.HIPAA,
                title="Administrative Safeguards",
                description="Implement administrative safeguards for PHI",
                category="administrative",
                severity="critical",
                evidence_types=["access_logs", "training_logs"],
                control_objectives=["access_control", "workforce_training"]
            ),
            ComplianceRequirement(
                id="hipaa_164_312_technical_safeguards",
                standard=ComplianceStandard.HIPAA,
                title="Technical Safeguards",
                description="Implement technical safeguards for PHI",
                category="technical",
                severity="critical",
                evidence_types=["audit_logs", "encryption_logs"],
                control_objectives=["access_control", "encryption", "audit_trail"]
            )
        ]
    
    def _load_iso27001_requirements(self) -> List[ComplianceRequirement]:
        """Load ISO 27001 compliance requirements"""
        return [
            ComplianceRequirement(
                id="iso27001_a9_access_control",
                standard=ComplianceStandard.ISO_27001,
                title="Access Control",
                description="Implement access control measures",
                category="access_control",
                severity="high",
                evidence_types=["access_logs", "authorization_logs"],
                control_objectives=["access_control", "authentication"]
            ),
            ComplianceRequirement(
                id="iso27001_a12_operations_security",
                standard=ComplianceStandard.ISO_27001,
                title="Operations Security",
                description="Implement operations security procedures",
                category="operations",
                severity="high",
                evidence_types=["operational_logs", "security_logs"],
                control_objectives=["operational_security", "incident_response"]
            )
        ]
    
    async def process_audit_event(self, event: AuditEvent):
        """Process audit event for compliance evidence"""
        evidence_items = []
        
        # Check each requirement
        for requirement in self.requirements:
            if await self._event_supports_requirement(event, requirement):
                evidence = ComplianceEvidence(
                    requirement_id=requirement.id,
                    evidence_type="audit_event",
                    timestamp=event.timestamp,
                    description=f"Audit event supporting {requirement.title}",
                    data={
                        "event_id": event.id,
                        "event_type": event.event_type.value,
                        "user": event.username or event.user_id,
                        "action": event.action,
                        "resource": event.resource,
                        "outcome": event.outcome
                    }
                )
                evidence_items.append(evidence)
        
        # Store evidence
        self.evidence_store.extend(evidence_items)
        self.stats["evidence_collected"] += len(evidence_items)
        
        # Check for violations
        violations = await self._detect_violations(event)
        if violations:
            self.violations.extend(violations)
            self.stats["active_violations"] += len(violations)
    
    async def process_security_event(self, event: SecurityEvent):
        """Process security event for compliance monitoring"""
        # Convert to audit event for processing
        audit_event = AuditEvent(
            id=str(event.id),
            event_type=AuditEventType.SECURITY_EVENT,
            timestamp=event.timestamp,
            user_id=str(event.user_id) if event.user_id else None,
            username=event.username,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            resource=event.resource,
            action=event.action,
            outcome=event.result,
            details={
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "message": event.message
            },
            metadata=event.metadata
        )
        
        await self.process_audit_event(audit_event)
    
    async def _event_supports_requirement(self, event: AuditEvent, requirement: ComplianceRequirement) -> bool:
        """Check if event provides evidence for requirement"""
        # Check evidence types
        if "audit_logs" in requirement.evidence_types:
            return True
        
        if "access_logs" in requirement.evidence_types:
            if event.event_type in [AuditEventType.AUTHENTICATION, AuditEventType.AUTHORIZATION]:
                return True
        
        if "security_logs" in requirement.evidence_types:
            if event.event_type == AuditEventType.SECURITY_EVENT:
                return True
        
        if "configuration_logs" in requirement.evidence_types:
            if event.event_type == AuditEventType.CONFIGURATION_CHANGE:
                return True
        
        if "data_access_logs" in requirement.evidence_types:
            if event.event_type == AuditEventType.DATA_ACCESS:
                return True
        
        return False
    
    async def _detect_violations(self, event: AuditEvent) -> List[ComplianceViolation]:
        """Detect compliance violations from event"""
        violations = []
        
        # SOX violations
        if ComplianceStandard.SOX in self.enabled_standards:
            sox_violations = await self._detect_sox_violations(event)
            violations.extend(sox_violations)
        
        # PCI-DSS violations
        if ComplianceStandard.PCI_DSS in self.enabled_standards:
            pci_violations = await self._detect_pci_dss_violations(event)
            violations.extend(pci_violations)
        
        # GDPR violations
        if ComplianceStandard.GDPR in self.enabled_standards:
            gdpr_violations = await self._detect_gdpr_violations(event)
            violations.extend(gdpr_violations)
        
        return violations
    
    async def _detect_sox_violations(self, event: AuditEvent) -> List[ComplianceViolation]:
        """Detect SOX compliance violations"""
        violations = []
        
        # Check for unauthorized configuration changes
        if event.event_type == AuditEventType.CONFIGURATION_CHANGE:
            if event.outcome == "success" and not event.details.get("approval_id"):
                violation = ComplianceViolation(
                    id=f"sox_violation_{event.id}",
                    requirement_id="sox_404_internal_controls",
                    standard=ComplianceStandard.SOX,
                    severity="critical",
                    description="Configuration change without proper approval",
                    detected_at=event.timestamp,
                    evidence=[ComplianceEvidence(
                        requirement_id="sox_404_internal_controls",
                        evidence_type="violation_evidence",
                        timestamp=event.timestamp,
                        description="Unauthorized configuration change detected",
                        data={"event_id": event.id, "user": event.username}
                    )],
                    remediation_actions=[
                        "Implement change approval process",
                        "Review change management procedures",
                        "Audit recent configuration changes"
                    ]
                )
                violations.append(violation)
        
        return violations
    
    async def _detect_pci_dss_violations(self, event: AuditEvent) -> List[ComplianceViolation]:
        """Detect PCI-DSS compliance violations"""
        violations = []
        
        # Check for unauthorized cardholder data access
        if event.event_type == AuditEventType.DATA_ACCESS:
            if "cardholder" in str(event.resource).lower():
                if not event.details.get("pci_authorized"):
                    violation = ComplianceViolation(
                        id=f"pci_violation_{event.id}",
                        requirement_id="pci_dss_req_7",
                        standard=ComplianceStandard.PCI_DSS,
                        severity="critical",
                        description="Unauthorized access to cardholder data",
                        detected_at=event.timestamp,
                        evidence=[ComplianceEvidence(
                            requirement_id="pci_dss_req_7",
                            evidence_type="violation_evidence",
                            timestamp=event.timestamp,
                            description="Unauthorized cardholder data access",
                            data={"event_id": event.id, "user": event.username, "resource": event.resource}
                        )],
                        remediation_actions=[
                            "Revoke unauthorized access",
                            "Review access controls",
                            "Implement need-to-know principle"
                        ]
                    )
                    violations.append(violation)
        
        return violations
    
    async def _detect_gdpr_violations(self, event: AuditEvent) -> List[ComplianceViolation]:
        """Detect GDPR compliance violations"""
        violations = []
        
        # Check for unauthorized personal data processing
        if event.event_type == AuditEventType.DATA_ACCESS:
            if "personal_data" in event.details.get("data_type", ""):
                if not event.details.get("consent_verified"):
                    violation = ComplianceViolation(
                        id=f"gdpr_violation_{event.id}",
                        requirement_id="gdpr_art_32_security",
                        standard=ComplianceStandard.GDPR,
                        severity="high",
                        description="Personal data processing without consent",
                        detected_at=event.timestamp,
                        evidence=[ComplianceEvidence(
                            requirement_id="gdpr_art_32_security",
                            evidence_type="violation_evidence",
                            timestamp=event.timestamp,
                            description="Unauthorized personal data processing",
                            data={"event_id": event.id, "user": event.username, "data_type": event.details.get("data_type")}
                        )],
                        remediation_actions=[
                            "Verify user consent",
                            "Implement consent management",
                            "Review data processing procedures"
                        ]
                    )
                    violations.append(violation)
        
        return violations
    
    async def _compliance_monitor(self):
        """Background compliance monitoring"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Update compliance scores
                await self._update_compliance_scores()
                
                # Check for overdue remediation
                await self._check_overdue_remediation()
                
            except Exception as e:
                self.logger.error(f"Compliance monitor error: {e}")
    
    async def _update_compliance_scores(self):
        """Update compliance scores for all standards"""
        for standard in self.enabled_standards:
            standard_requirements = [r for r in self.requirements if r.standard == standard]
            standard_violations = [v for v in self.violations if v.standard == standard and not v.resolved]
            
            if standard_requirements:
                compliance_score = max(0, 100 - (len(standard_violations) / len(standard_requirements)) * 100)
                self.stats["compliance_score"] = compliance_score
    
    async def _check_overdue_remediation(self):
        """Check for overdue remediation actions"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        overdue_violations = [
            v for v in self.violations
            if not v.resolved and v.detected_at < cutoff_time
        ]
        
        if overdue_violations:
            self.logger.warning(f"Found {len(overdue_violations)} overdue compliance violations")
    
    async def _evidence_collector(self):
        """Background evidence collection"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Collect additional evidence
                await self._collect_system_evidence()
                
            except Exception as e:
                self.logger.error(f"Evidence collector error: {e}")
    
    async def _collect_system_evidence(self):
        """Collect system-level evidence"""
        # This would collect evidence from various system sources
        pass
    
    async def _report_generator(self):
        """Background report generation"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                # Generate reports if due
                await self._generate_due_reports()
                
            except Exception as e:
                self.logger.error(f"Report generator error: {e}")
    
    async def _generate_due_reports(self):
        """Generate reports that are due"""
        for standard in self.enabled_standards:
            if await self._is_report_due(standard):
                report = await self.generate_compliance_report(standard)
                if report:
                    self.reports.append(report)
                    self.stats["reports_generated"] += 1
                    self.last_report_dates[standard] = datetime.now()
    
    async def _is_report_due(self, standard: ComplianceStandard) -> bool:
        """Check if report is due for standard"""
        last_report = self.last_report_dates.get(standard)
        
        if not last_report:
            return True
        
        if self.report_frequency == "daily":
            return datetime.now() - last_report >= timedelta(days=1)
        elif self.report_frequency == "weekly":
            return datetime.now() - last_report >= timedelta(weeks=1)
        elif self.report_frequency == "monthly":
            return datetime.now() - last_report >= timedelta(days=30)
        
        return False
    
    async def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ComplianceReport:
        """Generate compliance report for standard"""
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Filter requirements for this standard
        standard_requirements = [r for r in self.requirements if r.standard == standard]
        
        # Filter violations for this period
        period_violations = [
            v for v in self.violations
            if (v.standard == standard and 
                start_date <= v.detected_at <= end_date)
        ]
        
        # Calculate compliance metrics
        total_requirements = len(standard_requirements)
        violated_requirements = len(set(v.requirement_id for v in period_violations))
        compliant_requirements = total_requirements - violated_requirements
        
        # Generate evidence summary
        evidence_summary = {}
        for evidence in self.evidence_store:
            if evidence.timestamp >= start_date and evidence.timestamp <= end_date:
                evidence_type = evidence.evidence_type
                evidence_summary[evidence_type] = evidence_summary.get(evidence_type, 0) + 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(standard, period_violations)
        
        # Create report
        report = ComplianceReport(
            id=f"compliance_report_{standard.value}_{end_date.strftime('%Y%m%d')}",
            standard=standard,
            report_period_start=start_date,
            report_period_end=end_date,
            generated_at=datetime.now(),
            total_requirements=total_requirements,
            compliant_requirements=compliant_requirements,
            violations=period_violations,
            evidence_summary=evidence_summary,
            recommendations=recommendations,
            metadata={
                "compliance_score": (compliant_requirements / total_requirements) * 100 if total_requirements > 0 else 0,
                "violation_count": len(period_violations),
                "evidence_count": len(self.evidence_store)
            }
        )
        
        self.logger.info(f"Generated compliance report for {standard.value}")
        return report
    
    def _generate_recommendations(self, standard: ComplianceStandard, violations: List[ComplianceViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        if not violations:
            recommendations.append(f"Maintain current compliance posture for {standard.value}")
            return recommendations
        
        # Analyze violation patterns
        violation_by_category = {}
        for violation in violations:
            requirement = next((r for r in self.requirements if r.id == violation.requirement_id), None)
            if requirement:
                category = requirement.category
                if category not in violation_by_category:
                    violation_by_category[category] = []
                violation_by_category[category].append(violation)
        
        # Generate category-specific recommendations
        for category, category_violations in violation_by_category.items():
            if category == "access_control":
                recommendations.append("Strengthen access control policies and procedures")
                recommendations.append("Implement regular access reviews")
            elif category == "audit_logging":
                recommendations.append("Enhance audit logging and monitoring")
                recommendations.append("Implement log integrity controls")
            elif category == "configuration":
                recommendations.append("Implement change management processes")
                recommendations.append("Regular configuration reviews")
            elif category == "data_protection":
                recommendations.append("Enhance data protection measures")
                recommendations.append("Implement data classification")
        
        return recommendations
    
    async def _violation_tracker(self):
        """Background violation tracking"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Track violation remediation
                await self._track_remediation_progress()
                
            except Exception as e:
                self.logger.error(f"Violation tracker error: {e}")
    
    async def _track_remediation_progress(self):
        """Track remediation progress for violations"""
        # This would track the progress of remediation actions
        pass
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.processing_enabled:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Clean up old evidence
                cutoff_date = datetime.now() - timedelta(days=365)
                self.evidence_store = [
                    e for e in self.evidence_store
                    if e.timestamp > cutoff_date
                ]
                
                # Clean up old reports
                cutoff_date = datetime.now() - timedelta(days=730)  # 2 years
                self.reports = [
                    r for r in self.reports
                    if r.generated_at > cutoff_date
                ]
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    async def resolve_violation(self, violation_id: str, resolved_by: str, notes: str = ""):
        """Resolve a compliance violation"""
        violation = next((v for v in self.violations if v.id == violation_id), None)
        if violation:
            violation.resolved = True
            violation.resolved_at = datetime.now()
            violation.resolved_by = resolved_by
            
            self.stats["active_violations"] -= 1
            self.stats["resolved_violations"] += 1
            
            self.logger.info(f"Resolved compliance violation {violation_id} by {resolved_by}")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status"""
        status = {}
        
        for standard in self.enabled_standards:
            standard_requirements = [r for r in self.requirements if r.standard == standard]
            standard_violations = [v for v in self.violations if v.standard == standard and not v.resolved]
            
            compliance_score = 0
            if standard_requirements:
                compliance_score = max(0, 100 - (len(standard_violations) / len(standard_requirements)) * 100)
            
            status[standard.value] = {
                "compliance_score": compliance_score,
                "total_requirements": len(standard_requirements),
                "active_violations": len(standard_violations),
                "last_report": self.last_report_dates.get(standard)
            }
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compliance statistics"""
        return {
            "statistics": self.stats.copy(),
            "configuration": {
                "enabled_standards": [std.value for std in self.enabled_standards],
                "auto_reporting": self.auto_reporting,
                "report_frequency": self.report_frequency
            },
            "current_state": {
                "evidence_store_size": len(self.evidence_store),
                "active_violations": len([v for v in self.violations if not v.resolved]),
                "total_violations": len(self.violations),
                "reports_generated": len(self.reports)
            }
        }
    
    async def stop(self):
        """Stop compliance reporting"""
        self.processing_enabled = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        self.logger.info("Compliance reporting stopped")