"""
GDPR compliance and data protection features for rental property scraping.

This module provides comprehensive GDPR compliance features including
data anonymization, consent management, data retention, and audit trails.
"""

import asyncio
import logging
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .config import get_config, ProductionScrapingConfig

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"


class ProcessingPurpose(Enum):
    """Data processing purposes"""
    PROPERTY_LISTING = "property_listing"
    MARKET_ANALYSIS = "market_analysis"
    RECOMMENDATION = "recommendation"
    ANALYTICS = "analytics"
    RESEARCH = "research"


class LegalBasis(Enum):
    """Legal basis for processing under GDPR"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataRetentionPolicy:
    """Data retention policy"""
    purpose: ProcessingPurpose
    retention_period_days: int
    description: str
    legal_basis: LegalBasis


@dataclass
class PIIField:
    """Personally Identifiable Information field"""
    field_name: str
    classification: DataClassification
    anonymization_method: str  # 'hash', 'mask', 'remove', 'pseudonymize'
    retention_days: Optional[int] = None


@dataclass
class DataProcessingRecord:
    """Record of data processing activities"""
    id: str
    timestamp: datetime
    operation: str  # 'collect', 'process', 'anonymize', 'delete'
    data_subject: Optional[str] = None
    purpose: ProcessingPurpose = ProcessingPurpose.PROPERTY_LISTING
    legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTERESTS
    data_categories: List[str] = field(default_factory=list)
    retention_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PIIDetector:
    """Detects and classifies personally identifiable information"""
    
    def __init__(self):
        # Regex patterns for PII detection
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'driver_license': r'\b[A-Z]{1,2}\d{6,8}\b',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
        }
        
        # Field classifications
        self.field_classifications = {
            'phone': DataClassification.PERSONAL,
            'email': DataClassification.PERSONAL,
            'contact_info': DataClassification.PERSONAL,
            'name': DataClassification.PERSONAL,
            'address': DataClassification.PERSONAL,
            'location': DataClassification.INTERNAL,
            'price': DataClassification.PUBLIC,
            'description': DataClassification.PUBLIC,
            'amenities': DataClassification.PUBLIC,
            'images': DataClassification.PUBLIC
        }
    
    def detect_pii_in_text(self, text: str) -> Dict[str, List[str]]:
        """Detect PII patterns in text"""
        
        if not text:
            return {}
        
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def classify_field(self, field_name: str, field_value: Any) -> DataClassification:
        """Classify a field based on name and content"""
        
        field_name_lower = field_name.lower()
        
        # Direct field name classification
        for key, classification in self.field_classifications.items():
            if key in field_name_lower:
                return classification
        
        # Content-based classification
        if isinstance(field_value, str):
            pii_detected = self.detect_pii_in_text(field_value)
            if pii_detected:
                return DataClassification.PERSONAL
        
        # Default classification
        return DataClassification.INTERNAL
    
    def get_pii_fields(self, data: Dict[str, Any]) -> List[PIIField]:
        """Get all PII fields from data"""
        
        pii_fields = []
        
        for field_name, field_value in data.items():
            classification = self.classify_field(field_name, field_value)
            
            if classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE]:
                
                # Determine anonymization method
                if 'email' in field_name.lower():
                    method = 'hash'
                elif 'phone' in field_name.lower():
                    method = 'mask'
                elif 'contact' in field_name.lower():
                    method = 'pseudonymize'
                else:
                    method = 'hash'
                
                pii_field = PIIField(
                    field_name=field_name,
                    classification=classification,
                    anonymization_method=method
                )
                
                pii_fields.append(pii_field)
        
        return pii_fields


class DataAnonymizer:
    """Anonymizes and pseudonymizes personal data"""
    
    def __init__(self, salt: str = None):
        self.salt = salt or "rental-ml-system-salt"
    
    def hash_value(self, value: str) -> str:
        """Hash a value with salt"""
        if not value:
            return ""
        
        combined = f"{value}{self.salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def mask_phone(self, phone: str) -> str:
        """Mask phone number"""
        if not phone:
            return ""
        
        # Keep area code, mask the rest
        phone_digits = re.sub(r'[^\d]', '', phone)
        if len(phone_digits) >= 10:
            return f"({phone_digits[:3]}) ***-****"
        
        return "***-***-****"
    
    def mask_email(self, email: str) -> str:
        """Mask email address"""
        if not email or '@' not in email:
            return "***@***.***"
        
        username, domain = email.split('@', 1)
        
        if len(username) > 2:
            masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
        else:
            masked_username = '*' * len(username)
        
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            masked_domain = '***.' + domain_parts[-1]
        else:
            masked_domain = '***'
        
        return f"{masked_username}@{masked_domain}"
    
    def pseudonymize_name(self, name: str) -> str:
        """Pseudonymize a name"""
        if not name:
            return ""
        
        # Create a consistent pseudonym based on hash
        hash_value = self.hash_value(name)
        return f"Person_{hash_value[:8]}"
    
    def anonymize_field(self, field: PIIField, value: Any) -> Any:
        """Anonymize a field based on its classification"""
        
        if not value:
            return value
        
        value_str = str(value)
        
        if field.anonymization_method == 'hash':
            return self.hash_value(value_str)
        
        elif field.anonymization_method == 'mask':
            if 'phone' in field.field_name.lower():
                return self.mask_phone(value_str)
            elif 'email' in field.field_name.lower():
                return self.mask_email(value_str)
            else:
                # Generic masking
                if len(value_str) > 4:
                    return value_str[:2] + '*' * (len(value_str) - 4) + value_str[-2:]
                else:
                    return '*' * len(value_str)
        
        elif field.anonymization_method == 'pseudonymize':
            if 'name' in field.field_name.lower():
                return self.pseudonymize_name(value_str)
            else:
                return f"PSEUDO_{self.hash_value(value_str)[:8]}"
        
        elif field.anonymization_method == 'remove':
            return None
        
        return value
    
    def anonymize_property_data(self, property_data: Dict[str, Any], pii_fields: List[PIIField]) -> Dict[str, Any]:
        """Anonymize property data"""
        
        anonymized_data = property_data.copy()
        
        for pii_field in pii_fields:
            if pii_field.field_name in anonymized_data:
                original_value = anonymized_data[pii_field.field_name]
                anonymized_value = self.anonymize_field(pii_field, original_value)
                
                if anonymized_value is None:
                    del anonymized_data[pii_field.field_name]
                else:
                    anonymized_data[pii_field.field_name] = anonymized_value
        
        return anonymized_data


class GDPRCompliance:
    """Main GDPR compliance manager"""
    
    def __init__(self, config: ProductionScrapingConfig = None):
        self.config = config or get_config()
        self.pii_detector = PIIDetector()
        self.anonymizer = DataAnonymizer()
        
        # Processing records
        self.processing_records: List[DataProcessingRecord] = []
        
        # Retention policies
        self.retention_policies = self._load_retention_policies()
        
        # Consent tracking (simplified for property data)
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized GDPR compliance manager")
    
    def _load_retention_policies(self) -> List[DataRetentionPolicy]:
        """Load data retention policies"""
        
        return [
            DataRetentionPolicy(
                purpose=ProcessingPurpose.PROPERTY_LISTING,
                retention_period_days=365,  # 1 year
                description="Property listing data for rental marketplace",
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS
            ),
            DataRetentionPolicy(
                purpose=ProcessingPurpose.MARKET_ANALYSIS,
                retention_period_days=1095,  # 3 years
                description="Market analysis and trend data",
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS
            ),
            DataRetentionPolicy(
                purpose=ProcessingPurpose.ANALYTICS,
                retention_period_days=730,  # 2 years
                description="Analytics and performance metrics",
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS
            )
        ]
    
    def record_processing_activity(
        self,
        operation: str,
        purpose: ProcessingPurpose = ProcessingPurpose.PROPERTY_LISTING,
        legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTERESTS,
        data_categories: List[str] = None,
        data_subject: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Record a data processing activity"""
        
        # Calculate retention date
        retention_policy = next(
            (p for p in self.retention_policies if p.purpose == purpose),
            self.retention_policies[0]  # Default policy
        )
        
        retention_date = datetime.utcnow() + timedelta(days=retention_policy.retention_period_days)
        
        record = DataProcessingRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            operation=operation,
            purpose=purpose,
            legal_basis=legal_basis,
            data_categories=data_categories or [],
            data_subject=data_subject,
            retention_date=retention_date,
            metadata=metadata or {}
        )
        
        self.processing_records.append(record)
        
        logger.debug(f"Recorded processing activity: {operation} for {purpose.value}")
        return record.id
    
    def process_property_for_gdpr(
        self,
        property_data: Dict[str, Any],
        anonymize: bool = True,
        purpose: ProcessingPurpose = ProcessingPurpose.PROPERTY_LISTING
    ) -> Dict[str, Any]:
        """Process property data for GDPR compliance"""
        
        # Detect PII fields
        pii_fields = self.pii_detector.get_pii_fields(property_data)
        
        # Record processing activity
        data_categories = [field.field_name for field in pii_fields]
        processing_id = self.record_processing_activity(
            operation='process',
            purpose=purpose,
            data_categories=data_categories,
            metadata={
                'pii_fields_detected': len(pii_fields),
                'anonymization_enabled': anonymize
            }
        )
        
        # Anonymize if requested
        if anonymize and pii_fields:
            processed_data = self.anonymizer.anonymize_property_data(property_data, pii_fields)
            
            # Record anonymization
            self.record_processing_activity(
                operation='anonymize',
                purpose=purpose,
                data_categories=data_categories,
                metadata={'processing_id': processing_id}
            )
        else:
            processed_data = property_data.copy()
        
        # Add GDPR metadata
        processed_data['_gdpr_metadata'] = {
            'processing_id': processing_id,
            'processed_at': datetime.utcnow().isoformat(),
            'pii_fields_detected': len(pii_fields),
            'anonymized': anonymize and len(pii_fields) > 0,
            'purpose': purpose.value,
            'legal_basis': LegalBasis.LEGITIMATE_INTERESTS.value
        }
        
        return processed_data
    
    def check_retention_compliance(self) -> List[Dict[str, Any]]:
        """Check for data that should be deleted based on retention policies"""
        
        expired_data = []
        current_time = datetime.utcnow()
        
        for record in self.processing_records:
            if record.retention_date and current_time > record.retention_date:
                expired_data.append({
                    'processing_id': record.id,
                    'operation': record.operation,
                    'purpose': record.purpose.value,
                    'expired_since': current_time - record.retention_date,
                    'data_categories': record.data_categories
                })
        
        return expired_data
    
    def generate_privacy_notice(self) -> str:
        """Generate privacy notice for data processing"""
        
        notice = """
        PRIVACY NOTICE - Rental Property Data Processing
        
        This notice explains how we process rental property data in compliance with GDPR.
        
        WHAT DATA WE COLLECT:
        - Property listings (title, description, price, location)
        - Contact information (when publicly available)
        - Property specifications (bedrooms, bathrooms, amenities)
        - Property images
        
        WHY WE COLLECT IT:
        - To provide rental property search and recommendation services
        - For market analysis and trend identification
        - To improve our services and user experience
        
        LEGAL BASIS:
        - Legitimate interests for property listing aggregation
        - Performance of contract for user services
        
        DATA RETENTION:
        - Property listings: 1 year from collection
        - Market analysis data: 3 years
        - Analytics data: 2 years
        
        YOUR RIGHTS:
        - Access: Request access to your personal data
        - Rectification: Request correction of inaccurate data
        - Erasure: Request deletion of your personal data
        - Restriction: Request limitation of processing
        - Portability: Request transfer of your data
        - Objection: Object to processing based on legitimate interests
        
        CONTACT:
        For any privacy-related requests, contact: privacy@rental-ml-system.com
        
        Data Protection Officer: dpo@rental-ml-system.com
        """
        
        return notice.strip()
    
    def handle_data_subject_request(
        self,
        request_type: str,
        data_subject_id: str,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle data subject rights requests"""
        
        if request_type == 'access':
            # Find all processing records for this data subject
            subject_records = [
                record for record in self.processing_records
                if record.data_subject == data_subject_id
            ]
            
            return {
                'request_type': 'access',
                'data_subject_id': data_subject_id,
                'processing_records': [
                    {
                        'id': record.id,
                        'timestamp': record.timestamp.isoformat(),
                        'operation': record.operation,
                        'purpose': record.purpose.value,
                        'data_categories': record.data_categories
                    }
                    for record in subject_records
                ],
                'handled_at': datetime.utcnow().isoformat()
            }
        
        elif request_type == 'erasure':
            # Mark records for deletion
            deleted_records = []
            for record in self.processing_records:
                if record.data_subject == data_subject_id:
                    # In production, this would trigger actual data deletion
                    deleted_records.append(record.id)
            
            return {
                'request_type': 'erasure',
                'data_subject_id': data_subject_id,
                'deleted_records': deleted_records,
                'handled_at': datetime.utcnow().isoformat()
            }
        
        elif request_type == 'portability':
            # Export data in structured format
            return {
                'request_type': 'portability',
                'data_subject_id': data_subject_id,
                'export_format': 'JSON',
                'data_categories': ['property_listings', 'search_history'],
                'handled_at': datetime.utcnow().isoformat()
            }
        
        else:
            return {
                'error': f'Unsupported request type: {request_type}',
                'supported_types': ['access', 'erasure', 'portability']
            }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        
        current_time = datetime.utcnow()
        
        # Processing statistics
        total_records = len(self.processing_records)
        recent_records = len([
            r for r in self.processing_records
            if (current_time - r.timestamp).days <= 30
        ])
        
        # Purpose breakdown
        purpose_counts = {}
        for record in self.processing_records:
            purpose = record.purpose.value
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        
        # Check retention compliance
        expired_data = self.check_retention_compliance()
        
        return {
            'report_generated_at': current_time.isoformat(),
            'processing_statistics': {
                'total_processing_records': total_records,
                'recent_processing_records_30d': recent_records,
                'purpose_breakdown': purpose_counts
            },
            'retention_compliance': {
                'expired_data_count': len(expired_data),
                'next_retention_check': (current_time + timedelta(days=1)).isoformat()
            },
            'data_subject_rights': {
                'requests_handled': 0,  # Would track actual requests
                'average_response_time': '< 30 days'
            },
            'privacy_by_design': {
                'anonymization_enabled': True,
                'pii_detection_enabled': True,
                'retention_policies_active': len(self.retention_policies)
            }
        }
    
    def get_processing_records(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        purpose: ProcessingPurpose = None
    ) -> List[DataProcessingRecord]:
        """Get processing records with optional filters"""
        
        records = self.processing_records
        
        if start_date:
            records = [r for r in records if r.timestamp >= start_date]
        
        if end_date:
            records = [r for r in records if r.timestamp <= end_date]
        
        if purpose:
            records = [r for r in records if r.purpose == purpose]
        
        return records