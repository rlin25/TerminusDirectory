"""
Regional data processor for compliance and localized processing.

This module handles region-specific data processing, GDPR compliance,
data sovereignty requirements, and localized business logic.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum

from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class DataRegion(str, Enum):
    """Supported data regions"""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    CANADA = "canada"
    UK = "uk"
    AUSTRALIA = "australia"


class ComplianceFramework(str, Enum):
    """Compliance frameworks"""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PIPEDA = "pipeda"      # Personal Information Protection (Canada)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"          # Lei Geral de Proteção de Dados (Brazil)


class DataClassification(str, Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"            # Personally Identifiable Information
    SENSITIVE = "sensitive"


class ProcessingPurpose(str, Enum):
    """Data processing purposes"""
    PERSONALIZATION = "personalization"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    SECURITY = "security"
    LEGAL_COMPLIANCE = "legal_compliance"
    SERVICE_PROVISION = "service_provision"


class ConsentStatus(str, Enum):
    """User consent status"""
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class RegionalConfig(BaseModel):
    """Regional processing configuration"""
    region: DataRegion = Field(..., description="Data region")
    compliance_frameworks: List[ComplianceFramework] = Field(..., description="Applicable compliance frameworks")
    
    # Data residency requirements
    data_residency_required: bool = Field(default=False, description="Data must remain in region")
    cross_border_transfer_allowed: bool = Field(default=True, description="Cross-border transfers allowed")
    approved_transfer_regions: List[DataRegion] = Field(default=[], description="Approved regions for data transfer")
    
    # Retention policies
    default_retention_days: int = Field(default=2555, description="Default data retention in days (7 years)")
    pii_retention_days: int = Field(default=1095, description="PII retention in days (3 years)")
    analytics_retention_days: int = Field(default=365, description="Analytics data retention")
    
    # Processing restrictions
    automated_decision_making: bool = Field(default=True, description="Automated decision making allowed")
    profiling_allowed: bool = Field(default=True, description="User profiling allowed")
    marketing_processing: bool = Field(default=False, description="Marketing processing allowed by default")
    
    # Localization settings
    currency: str = Field(default="USD", description="Local currency")
    date_format: str = Field(default="%Y-%m-%d", description="Local date format")
    timezone: str = Field(default="UTC", description="Local timezone")
    language: str = Field(default="en", description="Default language")


class UserConsent(BaseModel):
    """User consent record"""
    user_id: str = Field(..., description="User identifier")
    region: DataRegion = Field(..., description="User's region")
    purpose: ProcessingPurpose = Field(..., description="Processing purpose")
    status: ConsentStatus = Field(..., description="Consent status")
    
    # Consent details
    granted_at: Optional[datetime] = Field(None, description="When consent was granted")
    withdrawn_at: Optional[datetime] = Field(None, description="When consent was withdrawn")
    expires_at: Optional[datetime] = Field(None, description="When consent expires")
    legal_basis: str = Field(..., description="Legal basis for processing")
    
    # Audit trail
    consent_version: str = Field(..., description="Version of consent form")
    ip_address: Optional[str] = Field(None, description="IP address when consent given")
    user_agent: Optional[str] = Field(None, description="User agent when consent given")
    evidence: Dict[str, Any] = Field(default={}, description="Evidence of consent")


class DataProcessingRecord(BaseModel):
    """Data processing activity record"""
    record_id: str = Field(..., description="Unique record identifier")
    user_id: str = Field(..., description="User identifier")
    region: DataRegion = Field(..., description="Processing region")
    
    # Processing details
    purpose: ProcessingPurpose = Field(..., description="Processing purpose")
    data_types: List[DataClassification] = Field(..., description="Types of data processed")
    legal_basis: str = Field(..., description="Legal basis for processing")
    
    # Data flow
    data_source: str = Field(..., description="Source of data")
    data_destination: Optional[str] = Field(None, description="Destination of data")
    cross_border_transfer: bool = Field(default=False, description="Cross-border transfer occurred")
    
    # Timing
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    retention_until: datetime = Field(..., description="Data retention until date")
    
    # Security measures
    encryption_used: bool = Field(default=True, description="Data encrypted")
    anonymization_applied: bool = Field(default=False, description="Data anonymized")
    pseudonymization_applied: bool = Field(default=False, description="Data pseudonymized")


class DataSubjectRequest(BaseModel):
    """Data subject request (GDPR Article 15-22)"""
    request_id: str = Field(..., description="Request identifier")
    user_id: str = Field(..., description="Data subject user ID")
    region: DataRegion = Field(..., description="Request region")
    
    # Request details
    request_type: str = Field(..., description="Type of request (access, rectification, erasure, etc.)")
    description: str = Field(..., description="Request description")
    status: str = Field(default="pending", description="Request status")
    
    # Processing
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None, description="When request was processed")
    response_due_date: datetime = Field(..., description="Response due date")
    
    # Verification
    identity_verified: bool = Field(default=False, description="Identity verified")
    verification_method: Optional[str] = Field(None, description="Verification method used")
    
    # Response
    response: Optional[str] = Field(None, description="Response to request")
    documents_provided: List[str] = Field(default=[], description="Documents provided")


class RegionalDataProcessor:
    """Regional data processor for compliance and localization"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.regional_configs: Dict[DataRegion, RegionalConfig] = {}
        self.user_consents: Dict[str, List[UserConsent]] = {}
        
        # Initialize default regional configurations
        self._initialize_regional_configs()
    
    def _initialize_regional_configs(self):
        """Initialize default regional configurations"""
        
        # EU GDPR Configuration
        self.regional_configs[DataRegion.EU_WEST] = RegionalConfig(
            region=DataRegion.EU_WEST,
            compliance_frameworks=[ComplianceFramework.GDPR],
            data_residency_required=True,
            cross_border_transfer_allowed=False,
            approved_transfer_regions=[DataRegion.EU_CENTRAL, DataRegion.UK],
            pii_retention_days=1095,
            automated_decision_making=False,  # Requires explicit consent
            profiling_allowed=False,
            marketing_processing=False,
            currency="EUR",
            timezone="Europe/London",
            language="en"
        )
        
        # US CCPA Configuration
        self.regional_configs[DataRegion.US_WEST] = RegionalConfig(
            region=DataRegion.US_WEST,
            compliance_frameworks=[ComplianceFramework.CCPA],
            data_residency_required=False,
            cross_border_transfer_allowed=True,
            approved_transfer_regions=[DataRegion.US_EAST, DataRegion.CANADA],
            pii_retention_days=1825,  # 5 years
            automated_decision_making=True,
            profiling_allowed=True,
            marketing_processing=True,
            currency="USD",
            timezone="America/Los_Angeles",
            language="en"
        )
        
        # Canada PIPEDA Configuration
        self.regional_configs[DataRegion.CANADA] = RegionalConfig(
            region=DataRegion.CANADA,
            compliance_frameworks=[ComplianceFramework.PIPEDA],
            data_residency_required=True,
            cross_border_transfer_allowed=False,
            approved_transfer_regions=[DataRegion.US_EAST, DataRegion.US_WEST],
            pii_retention_days=2555,  # 7 years
            automated_decision_making=True,
            profiling_allowed=True,
            marketing_processing=False,
            currency="CAD",
            timezone="America/Toronto",
            language="en"
        )
        
        # Asia Pacific Configuration
        self.regional_configs[DataRegion.ASIA_PACIFIC] = RegionalConfig(
            region=DataRegion.ASIA_PACIFIC,
            compliance_frameworks=[ComplianceFramework.PDPA],
            data_residency_required=True,
            cross_border_transfer_allowed=True,
            approved_transfer_regions=[DataRegion.AUSTRALIA],
            pii_retention_days=1095,
            automated_decision_making=True,
            profiling_allowed=True,
            marketing_processing=False,
            currency="SGD",
            timezone="Asia/Singapore",
            language="en"
        )
    
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the regional data processor"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            
            # Load existing consent records
            await self.load_consent_records()
            
            # Start background compliance tasks
            asyncio.create_task(self.compliance_monitoring_loop())
            asyncio.create_task(self.data_retention_cleanup())
            
            logger.info("Regional data processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize regional processor: {e}")
            raise
    
    async def process_data_request(
        self,
        user_id: str,
        data: Dict[str, Any],
        purpose: ProcessingPurpose,
        region: DataRegion,
        data_types: List[DataClassification]
    ) -> Dict[str, Any]:
        """Process data request with regional compliance"""
        try:
            # Get regional configuration
            config = self.regional_configs.get(region)
            if not config:
                raise ValueError(f"No configuration for region: {region}")
            
            # Check user consent
            consent_valid = await self.check_user_consent(user_id, purpose, region)
            if not consent_valid:
                return {
                    "success": False,
                    "error": "User consent required for this processing purpose",
                    "consent_required": True
                }
            
            # Apply regional data processing rules
            processed_data = await self.apply_regional_rules(data, config, data_types)
            
            # Record processing activity
            await self.record_processing_activity(
                user_id, region, purpose, data_types, processed_data
            )
            
            # Apply data minimization
            minimized_data = await self.apply_data_minimization(
                processed_data, purpose, config
            )
            
            return {
                "success": True,
                "data": minimized_data,
                "region": region,
                "processing_id": f"proc_{datetime.utcnow().timestamp()}"
            }
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_user_consent(
        self,
        user_id: str,
        purpose: ProcessingPurpose,
        region: DataRegion
    ) -> bool:
        """Check if user has valid consent for processing purpose"""
        try:
            consent_key = f"consent:{user_id}:{purpose}:{region}"
            consent_data = await self.redis_client.get(consent_key)
            
            if not consent_data:
                return False
            
            consent = UserConsent.model_validate_json(consent_data)
            
            # Check consent status
            if consent.status != ConsentStatus.GRANTED:
                return False
            
            # Check expiration
            if consent.expires_at and consent.expires_at < datetime.utcnow():
                # Mark as expired
                consent.status = ConsentStatus.EXPIRED
                await self.store_consent(consent)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Consent check failed: {e}")
            return False
    
    async def grant_consent(
        self,
        user_id: str,
        purpose: ProcessingPurpose,
        region: DataRegion,
        legal_basis: str,
        consent_version: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expiry_days: Optional[int] = None
    ) -> UserConsent:
        """Grant user consent for processing purpose"""
        try:
            # Create consent record
            consent = UserConsent(
                user_id=user_id,
                region=region,
                purpose=purpose,
                status=ConsentStatus.GRANTED,
                granted_at=datetime.utcnow(),
                legal_basis=legal_basis,
                consent_version=consent_version,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Set expiry if specified
            if expiry_days:
                consent.expires_at = datetime.utcnow() + timedelta(days=expiry_days)
            
            # Store consent
            await self.store_consent(consent)
            
            logger.info(f"Consent granted: {user_id} for {purpose} in {region}")
            return consent
            
        except Exception as e:
            logger.error(f"Consent granting failed: {e}")
            raise
    
    async def withdraw_consent(
        self,
        user_id: str,
        purpose: ProcessingPurpose,
        region: DataRegion
    ) -> bool:
        """Withdraw user consent"""
        try:
            consent_key = f"consent:{user_id}:{purpose}:{region}"
            consent_data = await self.redis_client.get(consent_key)
            
            if not consent_data:
                return False
            
            consent = UserConsent.model_validate_json(consent_data)
            consent.status = ConsentStatus.WITHDRAWN
            consent.withdrawn_at = datetime.utcnow()
            
            await self.store_consent(consent)
            
            # Trigger data cleanup for withdrawn consent
            await self.cleanup_data_for_withdrawn_consent(user_id, purpose, region)
            
            logger.info(f"Consent withdrawn: {user_id} for {purpose} in {region}")
            return True
            
        except Exception as e:
            logger.error(f"Consent withdrawal failed: {e}")
            return False
    
    async def apply_regional_rules(
        self,
        data: Dict[str, Any],
        config: RegionalConfig,
        data_types: List[DataClassification]
    ) -> Dict[str, Any]:
        """Apply regional processing rules to data"""
        try:
            processed_data = data.copy()
            
            # Apply pseudonymization for PII
            if DataClassification.PII in data_types:
                processed_data = await self.pseudonymize_pii(processed_data)
            
            # Apply anonymization for analytics in GDPR regions
            if (ComplianceFramework.GDPR in config.compliance_frameworks and
                DataClassification.SENSITIVE in data_types):
                processed_data = await self.anonymize_sensitive_data(processed_data)
            
            # Remove restricted fields for certain regions
            if config.region in [DataRegion.EU_WEST, DataRegion.EU_CENTRAL]:
                processed_data = await self.remove_restricted_fields_eu(processed_data)
            
            # Apply currency conversion
            if "price" in processed_data and config.currency != "USD":
                processed_data["price"] = await self.convert_currency(
                    processed_data["price"], "USD", config.currency
                )
                processed_data["currency"] = config.currency
            
            # Apply date format localization
            for key, value in processed_data.items():
                if isinstance(value, datetime):
                    processed_data[key] = value.strftime(config.date_format)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Regional rules application failed: {e}")
            return data
    
    async def apply_data_minimization(
        self,
        data: Dict[str, Any],
        purpose: ProcessingPurpose,
        config: RegionalConfig
    ) -> Dict[str, Any]:
        """Apply data minimization principle"""
        try:
            # Define essential fields for each purpose
            essential_fields = {
                ProcessingPurpose.PERSONALIZATION: [
                    "user_id", "preferences", "location", "search_history"
                ],
                ProcessingPurpose.ANALYTICS: [
                    "user_id", "event_type", "timestamp", "page_views"
                ],
                ProcessingPurpose.MARKETING: [
                    "user_id", "interests", "demographics", "contact_preferences"
                ],
                ProcessingPurpose.SECURITY: [
                    "user_id", "ip_address", "device_info", "login_attempts"
                ],
                ProcessingPurpose.SERVICE_PROVISION: [
                    "user_id", "service_data", "transaction_data", "support_history"
                ]
            }
            
            allowed_fields = essential_fields.get(purpose, list(data.keys()))
            
            # Filter data to only include essential fields
            minimized_data = {
                key: value for key, value in data.items()
                if key in allowed_fields
            }
            
            return minimized_data
            
        except Exception as e:
            logger.error(f"Data minimization failed: {e}")
            return data
    
    async def pseudonymize_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pseudonymize personally identifiable information"""
        try:
            pii_fields = ["email", "phone", "name", "address", "ssn"]
            
            for field in pii_fields:
                if field in data:
                    # Generate consistent pseudonym
                    original_value = str(data[field])
                    pseudonym = hashlib.sha256(
                        f"{field}:{original_value}:salt".encode()
                    ).hexdigest()[:16]
                    
                    data[f"{field}_pseudonym"] = pseudonym
                    del data[field]
            
            return data
            
        except Exception as e:
            logger.error(f"PII pseudonymization failed: {e}")
            return data
    
    async def anonymize_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data"""
        try:
            sensitive_fields = ["precise_location", "financial_data", "health_data"]
            
            for field in sensitive_fields:
                if field in data:
                    if field == "precise_location" and isinstance(data[field], dict):
                        # Reduce location precision
                        if "latitude" in data[field]:
                            data[field]["latitude"] = round(data[field]["latitude"], 2)
                        if "longitude" in data[field]:
                            data[field]["longitude"] = round(data[field]["longitude"], 2)
                    else:
                        # Remove sensitive field
                        del data[field]
            
            return data
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return data
    
    async def remove_restricted_fields_eu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove fields restricted under EU regulations"""
        try:
            restricted_fields = [
                "political_opinions", "religious_beliefs", "trade_union_membership",
                "genetic_data", "biometric_data", "health_data", "sexual_orientation"
            ]
            
            for field in restricted_fields:
                if field in data:
                    del data[field]
            
            return data
            
        except Exception as e:
            logger.error(f"EU field restriction failed: {e}")
            return data
    
    async def convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str
    ) -> float:
        """Convert currency (simplified implementation)"""
        try:
            # In a real implementation, this would use a currency conversion API
            conversion_rates = {
                ("USD", "EUR"): 0.85,
                ("USD", "CAD"): 1.25,
                ("USD", "SGD"): 1.35,
                ("EUR", "USD"): 1.18,
                ("CAD", "USD"): 0.80,
                ("SGD", "USD"): 0.74
            }
            
            rate = conversion_rates.get((from_currency, to_currency), 1.0)
            return round(amount * rate, 2)
            
        except Exception as e:
            logger.error(f"Currency conversion failed: {e}")
            return amount
    
    async def record_processing_activity(
        self,
        user_id: str,
        region: DataRegion,
        purpose: ProcessingPurpose,
        data_types: List[DataClassification],
        processed_data: Dict[str, Any]
    ) -> None:
        """Record data processing activity for audit"""
        try:
            config = self.regional_configs[region]
            
            # Calculate retention date based on data types
            retention_days = config.default_retention_days
            if DataClassification.PII in data_types:
                retention_days = config.pii_retention_days
            elif purpose == ProcessingPurpose.ANALYTICS:
                retention_days = config.analytics_retention_days
            
            record = DataProcessingRecord(
                record_id=f"proc_{datetime.utcnow().timestamp()}_{user_id}",
                user_id=user_id,
                region=region,
                purpose=purpose,
                data_types=data_types,
                legal_basis="consent",  # Simplified
                data_source="api_request",
                retention_until=datetime.utcnow() + timedelta(days=retention_days),
                encryption_used=True,
                anonymization_applied=DataClassification.SENSITIVE in data_types,
                pseudonymization_applied=DataClassification.PII in data_types
            )
            
            # Store processing record
            record_key = f"processing_record:{record.record_id}"
            await self.redis_client.setex(
                record_key,
                86400 * retention_days,
                record.model_dump_json()
            )
            
            # Add to user's processing history
            user_records_key = f"user_processing:{user_id}"
            await self.redis_client.sadd(user_records_key, record.record_id)
            await self.redis_client.expire(user_records_key, 86400 * retention_days)
            
        except Exception as e:
            logger.error(f"Processing activity recording failed: {e}")
    
    async def handle_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        region: DataRegion,
        description: str
    ) -> DataSubjectRequest:
        """Handle data subject request (GDPR, CCPA, etc.)"""
        try:
            config = self.regional_configs[region]
            
            # Determine response timeframe based on compliance framework
            response_days = 30  # Default
            if ComplianceFramework.GDPR in config.compliance_frameworks:
                response_days = 30  # GDPR Article 12
            elif ComplianceFramework.CCPA in config.compliance_frameworks:
                response_days = 45  # CCPA
            
            request = DataSubjectRequest(
                request_id=f"dsr_{datetime.utcnow().timestamp()}_{user_id}",
                user_id=user_id,
                region=region,
                request_type=request_type,
                description=description,
                response_due_date=datetime.utcnow() + timedelta(days=response_days)
            )
            
            # Store request
            await self.store_data_subject_request(request)
            
            # Process specific request types
            if request_type == "access":
                await self.process_access_request(request)
            elif request_type == "erasure":
                await self.process_erasure_request(request)
            elif request_type == "rectification":
                await self.process_rectification_request(request)
            elif request_type == "portability":
                await self.process_portability_request(request)
            
            logger.info(f"Data subject request created: {request.request_id}")
            return request
            
        except Exception as e:
            logger.error(f"Data subject request handling failed: {e}")
            raise
    
    async def process_access_request(self, request: DataSubjectRequest) -> None:
        """Process data access request"""
        try:
            user_id = request.user_id
            
            # Collect all user data
            user_data = {
                "personal_data": await self.get_user_personal_data(user_id),
                "consent_records": await self.get_user_consent_records(user_id),
                "processing_activities": await self.get_user_processing_records(user_id),
                "data_sources": await self.get_user_data_sources(user_id)
            }
            
            # Store compiled data
            data_key = f"access_data:{request.request_id}"
            await self.redis_client.setex(
                data_key,
                86400 * 90,  # Keep for 90 days
                json.dumps(user_data, default=str)
            )
            
            # Update request status
            request.status = "completed"
            request.processed_at = datetime.utcnow()
            request.response = f"Data access package prepared. Available at: {data_key}"
            
            await self.store_data_subject_request(request)
            
        except Exception as e:
            logger.error(f"Access request processing failed: {e}")
    
    async def process_erasure_request(self, request: DataSubjectRequest) -> None:
        """Process data erasure request (right to be forgotten)"""
        try:
            user_id = request.user_id
            
            # Get all user-related keys
            patterns = [
                f"user_data:{user_id}*",
                f"user_processing:{user_id}*",
                f"consent:{user_id}*",
                f"cache:*user:{user_id}*"
            ]
            
            deleted_count = 0
            for pattern in patterns:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    deleted_count += await self.redis_client.delete(*keys)
            
            # Update request status
            request.status = "completed"
            request.processed_at = datetime.utcnow()
            request.response = f"User data erased. {deleted_count} records deleted."
            
            await self.store_data_subject_request(request)
            
            logger.info(f"Erasure request completed: {deleted_count} records deleted for {user_id}")
            
        except Exception as e:
            logger.error(f"Erasure request processing failed: {e}")
    
    async def store_consent(self, consent: UserConsent) -> None:
        """Store user consent record"""
        try:
            consent_key = f"consent:{consent.user_id}:{consent.purpose}:{consent.region}"
            await self.redis_client.setex(
                consent_key,
                86400 * 2555,  # 7 years
                consent.model_dump_json()
            )
            
            # Add to user's consent index
            user_consents_key = f"user_consents:{consent.user_id}"
            await self.redis_client.sadd(user_consents_key, consent_key)
            
        except Exception as e:
            logger.error(f"Consent storage failed: {e}")
    
    async def store_data_subject_request(self, request: DataSubjectRequest) -> None:
        """Store data subject request"""
        try:
            request_key = f"dsr:{request.request_id}"
            await self.redis_client.setex(
                request_key,
                86400 * 2555,  # 7 years
                request.model_dump_json()
            )
            
            # Add to user's request index
            user_requests_key = f"user_requests:{request.user_id}"
            await self.redis_client.sadd(user_requests_key, request.request_id)
            
        except Exception as e:
            logger.error(f"Data subject request storage failed: {e}")
    
    async def compliance_monitoring_loop(self) -> None:
        """Background compliance monitoring"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Check for expired consents
                await self.check_expired_consents()
                
                # Check overdue data subject requests
                await self.check_overdue_requests()
                
                # Generate compliance reports
                await self.generate_compliance_reports()
                
            except Exception as e:
                logger.error(f"Compliance monitoring failed: {e}")
    
    async def data_retention_cleanup(self) -> None:
        """Background data retention cleanup"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Clean up expired data
                await self.cleanup_expired_data()
                
                # Archive old processing records
                await self.archive_old_records()
                
            except Exception as e:
                logger.error(f"Data retention cleanup failed: {e}")
    
    async def check_expired_consents(self) -> None:
        """Check for and handle expired consents"""
        try:
            pattern = "consent:*"
            keys = await self.redis_client.keys(pattern)
            
            expired_count = 0
            for key in keys:
                consent_data = await self.redis_client.get(key)
                if consent_data:
                    consent = UserConsent.model_validate_json(consent_data)
                    
                    if (consent.expires_at and 
                        consent.expires_at < datetime.utcnow() and
                        consent.status == ConsentStatus.GRANTED):
                        
                        consent.status = ConsentStatus.EXPIRED
                        await self.store_consent(consent)
                        expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Marked {expired_count} consents as expired")
                
        except Exception as e:
            logger.error(f"Expired consent check failed: {e}")
    
    async def cleanup_expired_data(self) -> None:
        """Clean up data that has exceeded retention period"""
        try:
            pattern = "processing_record:*"
            keys = await self.redis_client.keys(pattern)
            
            cleaned_count = 0
            for key in keys:
                record_data = await self.redis_client.get(key)
                if record_data:
                    record = DataProcessingRecord.model_validate_json(record_data)
                    
                    if record.retention_until < datetime.utcnow():
                        await self.redis_client.delete(key)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired processing records")
                
        except Exception as e:
            logger.error(f"Expired data cleanup failed: {e}")
    
    async def get_user_personal_data(self, user_id: str) -> Dict[str, Any]:
        """Get all personal data for a user"""
        try:
            # This would collect data from various sources
            personal_data = {
                "profile": await self.redis_client.hgetall(f"user_profile:{user_id}"),
                "preferences": await self.redis_client.hgetall(f"user_preferences:{user_id}"),
                "search_history": await self.redis_client.lrange(f"search_history:{user_id}", 0, -1),
                "favorites": await self.redis_client.smembers(f"user_favorites:{user_id}")
            }
            
            return personal_data
            
        except Exception as e:
            logger.error(f"Personal data retrieval failed: {e}")
            return {}
    
    async def get_user_consent_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all consent records for a user"""
        try:
            user_consents_key = f"user_consents:{user_id}"
            consent_keys = await self.redis_client.smembers(user_consents_key)
            
            consents = []
            for consent_key in consent_keys:
                consent_data = await self.redis_client.get(consent_key.decode())
                if consent_data:
                    consent = UserConsent.model_validate_json(consent_data)
                    consents.append(consent.model_dump())
            
            return consents
            
        except Exception as e:
            logger.error(f"Consent records retrieval failed: {e}")
            return []
    
    async def get_user_processing_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all processing records for a user"""
        try:
            user_records_key = f"user_processing:{user_id}"
            record_ids = await self.redis_client.smembers(user_records_key)
            
            records = []
            for record_id in record_ids:
                record_key = f"processing_record:{record_id.decode()}"
                record_data = await self.redis_client.get(record_key)
                if record_data:
                    record = DataProcessingRecord.model_validate_json(record_data)
                    records.append(record.model_dump())
            
            return records
            
        except Exception as e:
            logger.error(f"Processing records retrieval failed: {e}")
            return []
    
    async def load_consent_records(self) -> None:
        """Load existing consent records from Redis"""
        try:
            pattern = "consent:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                consent_data = await self.redis_client.get(key)
                if consent_data:
                    consent = UserConsent.model_validate_json(consent_data)
                    
                    if consent.user_id not in self.user_consents:
                        self.user_consents[consent.user_id] = []
                    
                    self.user_consents[consent.user_id].append(consent)
            
            logger.info(f"Loaded consent records for {len(self.user_consents)} users")
            
        except Exception as e:
            logger.error(f"Consent records loading failed: {e}")
    
    async def cleanup_data_for_withdrawn_consent(
        self,
        user_id: str,
        purpose: ProcessingPurpose,
        region: DataRegion
    ) -> None:
        """Clean up data when consent is withdrawn"""
        try:
            # Get processing records for this purpose
            user_records_key = f"user_processing:{user_id}"
            record_ids = await self.redis_client.smembers(user_records_key)
            
            cleaned_count = 0
            for record_id in record_ids:
                record_key = f"processing_record:{record_id.decode()}"
                record_data = await self.redis_client.get(record_key)
                
                if record_data:
                    record = DataProcessingRecord.model_validate_json(record_data)
                    
                    if record.purpose == purpose and record.region == region:
                        # Mark data for deletion or anonymization
                        await self.anonymize_processing_record(record)
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} records for withdrawn consent")
            
        except Exception as e:
            logger.error(f"Consent withdrawal cleanup failed: {e}")
    
    async def anonymize_processing_record(self, record: DataProcessingRecord) -> None:
        """Anonymize processing record when consent is withdrawn"""
        try:
            # Remove personally identifiable information
            record.user_id = "anonymous"
            record.anonymization_applied = True
            
            # Store anonymized record
            record_key = f"processing_record:{record.record_id}"
            await self.redis_client.setex(
                record_key,
                86400 * 30,  # Keep anonymized for 30 days
                record.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Record anonymization failed: {e}")