"""
Database Compliance Manager
GDPR, SOX, HIPAA, and other regulatory compliance management
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import asyncpg
from asyncpg import Connection


class ComplianceFramework(Enum):
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    SENSITIVE = "sensitive"


class RetentionAction(Enum):
    ARCHIVE = "archive"
    DELETE = "delete"
    ANONYMIZE = "anonymize"
    PSEUDONYMIZE = "pseudonymize"


@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    framework: ComplianceFramework
    description: str
    applies_to_tables: List[str]
    applies_to_columns: List[str]
    data_classification: DataClassification
    retention_period_days: int
    retention_action: RetentionAction
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class DataInventoryItem:
    table_name: str
    column_name: str
    data_type: str
    classification: DataClassification
    contains_pii: bool
    encryption_required: bool
    retention_period_days: int
    legal_basis: Optional[str] = None
    purpose: Optional[str] = None
    last_reviewed: Optional[datetime] = None


@dataclass
class ConsentRecord:
    consent_id: str
    user_id: str
    purpose: str
    consent_given: bool
    consent_date: datetime
    withdrawal_date: Optional[datetime] = None
    legal_basis: str = "consent"
    data_categories: List[str] = None
    
    def __post_init__(self):
        if self.data_categories is None:
            self.data_categories = []


@dataclass
class DataProcessingActivity:
    activity_id: str
    name: str
    purpose: str
    legal_basis: str
    data_categories: List[str]
    data_subjects: List[str]
    recipients: List[str]
    retention_period: str
    security_measures: List[str]
    transfer_countries: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.transfer_countries is None:
            self.transfer_countries = []
        if self.created_at is None:
            self.created_at = datetime.now()


class DatabaseComplianceManager:
    """
    Comprehensive compliance management for:
    - GDPR (General Data Protection Regulation)
    - SOX (Sarbanes-Oxley Act)
    - HIPAA (Health Insurance Portability and Accountability Act)
    - PCI DSS (Payment Card Industry Data Security Standard)
    - SOC 2 (Service Organization Control 2)
    - ISO 27001 (Information Security Management)
    """
    
    def __init__(self, connection_manager, security_manager=None):
        self.connection_manager = connection_manager
        self.security_manager = security_manager
        self.logger = logging.getLogger(__name__)
        
        # Compliance rules by framework
        self.compliance_rules: Dict[ComplianceFramework, List[ComplianceRule]] = {}
        
        # Data inventory
        self.data_inventory: List[DataInventoryItem] = []
        
        # Initialize compliance frameworks
        self._initialize_compliance_rules()
        
    async def initialize_compliance(self):
        """Initialize compliance management system"""
        try:
            self.logger.info("Initializing database compliance system...")
            
            # Create compliance schema and tables
            await self._create_compliance_schema()
            
            # Initialize data inventory
            await self._initialize_data_inventory()
            
            # Set up retention policies
            await self._setup_retention_policies()
            
            # Configure compliance monitoring
            await self._setup_compliance_monitoring()
            
            self.logger.info("Database compliance system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize compliance system: {e}")
            raise
            
    async def _create_compliance_schema(self):
        """Create compliance-related database schema"""
        async with self.connection_manager.get_connection() as conn:
            # Create compliance schema
            await conn.execute("CREATE SCHEMA IF NOT EXISTS compliance")
            
            # Data inventory table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance.data_inventory (
                    inventory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    schema_name VARCHAR(255) NOT NULL,
                    table_name VARCHAR(255) NOT NULL,
                    column_name VARCHAR(255),
                    data_type VARCHAR(100),
                    classification VARCHAR(50) NOT NULL,
                    contains_pii BOOLEAN DEFAULT FALSE,
                    encryption_required BOOLEAN DEFAULT FALSE,
                    retention_period_days INTEGER,
                    legal_basis VARCHAR(255),
                    purpose TEXT,
                    last_reviewed TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE (schema_name, table_name, column_name)
                )
            """)
            
            # Consent management table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance.consent_records (
                    consent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID NOT NULL,
                    purpose VARCHAR(255) NOT NULL,
                    consent_given BOOLEAN NOT NULL,
                    consent_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    withdrawal_date TIMESTAMP WITH TIME ZONE,
                    legal_basis VARCHAR(100) DEFAULT 'consent',
                    data_categories TEXT[],
                    consent_text TEXT,
                    version VARCHAR(20),
                    ip_address INET,
                    user_agent TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Data processing activities table (GDPR Article 30)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance.processing_activities (
                    activity_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL,
                    purpose TEXT NOT NULL,
                    legal_basis VARCHAR(100) NOT NULL,
                    data_categories TEXT[] NOT NULL,
                    data_subjects TEXT[] NOT NULL,
                    recipients TEXT[],
                    retention_period VARCHAR(255),
                    security_measures TEXT[],
                    transfer_countries TEXT[],
                    dpo_contact VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Data retention schedule table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance.retention_schedule (
                    schedule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    table_name VARCHAR(255) NOT NULL,
                    column_name VARCHAR(255),
                    retention_period_days INTEGER NOT NULL,
                    retention_action VARCHAR(50) NOT NULL,
                    legal_basis VARCHAR(255),
                    next_review_date DATE,
                    last_executed TIMESTAMP WITH TIME ZONE,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Compliance violations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance.violations (
                    violation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    framework VARCHAR(50) NOT NULL,
                    violation_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    description TEXT NOT NULL,
                    table_name VARCHAR(255),
                    column_name VARCHAR(255),
                    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP WITH TIME ZONE,
                    resolution_notes TEXT,
                    impact_assessment TEXT
                )
            """)
            
            # Data subject requests table (GDPR)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance.data_subject_requests (
                    request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    request_type VARCHAR(50) NOT NULL, -- access, rectification, erasure, portability, restriction
                    user_id UUID,
                    user_email VARCHAR(255),
                    request_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'pending', -- pending, in_progress, completed, rejected
                    completed_date TIMESTAMP WITH TIME ZONE,
                    request_details JSONB,
                    response_data JSONB,
                    legal_basis VARCHAR(255),
                    processed_by VARCHAR(255),
                    verification_method VARCHAR(100),
                    notes TEXT
                )
            """)
            
            # Compliance audit log table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance.audit_log (
                    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    framework VARCHAR(50) NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    table_name VARCHAR(255),
                    user_id UUID,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    details JSONB,
                    compliance_status VARCHAR(20) -- compliant, non_compliant, requires_review
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_inventory_table ON compliance.data_inventory(schema_name, table_name);
                CREATE INDEX IF NOT EXISTS idx_consent_records_user ON compliance.consent_records(user_id);
                CREATE INDEX IF NOT EXISTS idx_consent_records_purpose ON compliance.consent_records(purpose);
                CREATE INDEX IF NOT EXISTS idx_processing_activities_legal_basis ON compliance.processing_activities(legal_basis);
                CREATE INDEX IF NOT EXISTS idx_retention_schedule_table ON compliance.retention_schedule(table_name);
                CREATE INDEX IF NOT EXISTS idx_violations_framework ON compliance.violations(framework);
                CREATE INDEX IF NOT EXISTS idx_data_subject_requests_user ON compliance.data_subject_requests(user_id);
                CREATE INDEX IF NOT EXISTS idx_data_subject_requests_type ON compliance.data_subject_requests(request_type);
                CREATE INDEX IF NOT EXISTS idx_audit_log_framework ON compliance.audit_log(framework);
                CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON compliance.audit_log(timestamp);
            """)
            
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for different frameworks"""
        
        # GDPR Rules
        gdpr_rules = [
            ComplianceRule(
                rule_id="gdpr_001",
                name="PII Data Retention",
                framework=ComplianceFramework.GDPR,
                description="Personal data must be kept for no longer than necessary",
                applies_to_tables=["users", "user_interactions"],
                applies_to_columns=["email", "ip_address"],
                data_classification=DataClassification.PII,
                retention_period_days=1095,  # 3 years
                retention_action=RetentionAction.ANONYMIZE
            ),
            ComplianceRule(
                rule_id="gdpr_002", 
                name="Sensitive Data Encryption",
                framework=ComplianceFramework.GDPR,
                description="Sensitive personal data must be encrypted",
                applies_to_tables=["users"],
                applies_to_columns=["email"],
                data_classification=DataClassification.PII,
                retention_period_days=1095,
                retention_action=RetentionAction.PSEUDONYMIZE
            ),
            ComplianceRule(
                rule_id="gdpr_003",
                name="Marketing Data Consent",
                framework=ComplianceFramework.GDPR,
                description="Marketing communications require explicit consent",
                applies_to_tables=["users"],
                applies_to_columns=["email", "preferred_locations"],
                data_classification=DataClassification.PII,
                retention_period_days=730,  # 2 years
                retention_action=RetentionAction.DELETE
            )
        ]
        
        # SOX Rules (Sarbanes-Oxley)
        sox_rules = [
            ComplianceRule(
                rule_id="sox_001",
                name="Financial Data Retention",
                framework=ComplianceFramework.SOX,
                description="Financial records must be retained for 7 years",
                applies_to_tables=["training_metrics", "audit_log"],
                applies_to_columns=["*"],
                data_classification=DataClassification.CONFIDENTIAL,
                retention_period_days=2555,  # 7 years
                retention_action=RetentionAction.ARCHIVE
            ),
            ComplianceRule(
                rule_id="sox_002",
                name="Audit Trail Integrity",
                framework=ComplianceFramework.SOX,
                description="Audit trails must be tamper-proof and complete",
                applies_to_tables=["audit_log"],
                applies_to_columns=["*"],
                data_classification=DataClassification.RESTRICTED,
                retention_period_days=2555,
                retention_action=RetentionAction.ARCHIVE
            )
        ]
        
        # SOC 2 Rules
        soc2_rules = [
            ComplianceRule(
                rule_id="soc2_001",
                name="Access Control Logging",
                framework=ComplianceFramework.SOC2,
                description="All access to customer data must be logged",
                applies_to_tables=["users", "properties", "user_interactions"],
                applies_to_columns=["*"],
                data_classification=DataClassification.CONFIDENTIAL,
                retention_period_days=1095,
                retention_action=RetentionAction.ARCHIVE
            ),
            ComplianceRule(
                rule_id="soc2_002",
                name="Data Processing Monitoring",
                framework=ComplianceFramework.SOC2,
                description="Data processing activities must be monitored",
                applies_to_tables=["ml_models", "training_metrics"],
                applies_to_columns=["*"],
                data_classification=DataClassification.INTERNAL,
                retention_period_days=1095,
                retention_action=RetentionAction.ARCHIVE
            )
        ]
        
        self.compliance_rules[ComplianceFramework.GDPR] = gdpr_rules
        self.compliance_rules[ComplianceFramework.SOX] = sox_rules
        self.compliance_rules[ComplianceFramework.SOC2] = soc2_rules
        
    async def _initialize_data_inventory(self):
        """Initialize data inventory with table and column classifications"""
        
        # Define data inventory items
        inventory_items = [
            # Users table
            DataInventoryItem("users", "id", "UUID", DataClassification.PII, True, True, 1095),
            DataInventoryItem("users", "email", "VARCHAR", DataClassification.PII, True, True, 1095, "consent", "user_account"),
            DataInventoryItem("users", "created_at", "TIMESTAMP", DataClassification.INTERNAL, False, False, 2555),
            DataInventoryItem("users", "preferred_locations", "TEXT[]", DataClassification.PII, True, False, 730, "consent", "personalization"),
            DataInventoryItem("users", "required_amenities", "TEXT[]", DataClassification.INTERNAL, False, False, 730),
            
            # Properties table
            DataInventoryItem("properties", "id", "UUID", DataClassification.PUBLIC, False, False, 2555),
            DataInventoryItem("properties", "title", "VARCHAR", DataClassification.PUBLIC, False, False, 2555),
            DataInventoryItem("properties", "description", "TEXT", DataClassification.PUBLIC, False, False, 2555),
            DataInventoryItem("properties", "contact_info", "JSONB", DataClassification.CONFIDENTIAL, True, True, 1095),
            DataInventoryItem("properties", "latitude", "DECIMAL", DataClassification.INTERNAL, False, False, 2555),
            DataInventoryItem("properties", "longitude", "DECIMAL", DataClassification.INTERNAL, False, False, 2555),
            
            # User interactions table
            DataInventoryItem("user_interactions", "id", "UUID", DataClassification.INTERNAL, False, False, 1095),
            DataInventoryItem("user_interactions", "user_id", "UUID", DataClassification.PII, True, False, 1095, "consent", "analytics"),
            DataInventoryItem("user_interactions", "property_id", "UUID", DataClassification.INTERNAL, False, False, 1095),
            DataInventoryItem("user_interactions", "ip_address", "INET", DataClassification.PII, True, True, 365, "legitimate_interest", "security"),
            DataInventoryItem("user_interactions", "user_agent", "TEXT", DataClassification.PII, True, False, 365, "legitimate_interest", "analytics"),
            
            # Search queries table
            DataInventoryItem("search_queries", "user_id", "UUID", DataClassification.PII, True, False, 730, "consent", "personalization"),
            DataInventoryItem("search_queries", "query_text", "TEXT", DataClassification.INTERNAL, False, False, 730),
            DataInventoryItem("search_queries", "filters", "JSONB", DataClassification.INTERNAL, False, False, 730),
            
            # ML models table
            DataInventoryItem("ml_models", "id", "UUID", DataClassification.CONFIDENTIAL, False, False, 2555),
            DataInventoryItem("ml_models", "model_data", "BYTEA", DataClassification.CONFIDENTIAL, False, True, 2555),
            DataInventoryItem("ml_models", "metadata", "JSONB", DataClassification.INTERNAL, False, False, 2555),
            
            # Training metrics table
            DataInventoryItem("training_metrics", "id", "UUID", DataClassification.INTERNAL, False, False, 2555),
            DataInventoryItem("training_metrics", "metrics", "JSONB", DataClassification.CONFIDENTIAL, False, False, 2555),
            DataInventoryItem("training_metrics", "hyperparameters", "JSONB", DataClassification.CONFIDENTIAL, False, False, 2555),
            
            # Audit log table
            DataInventoryItem("audit_log", "id", "UUID", DataClassification.RESTRICTED, False, False, 2555),
            DataInventoryItem("audit_log", "table_name", "VARCHAR", DataClassification.RESTRICTED, False, False, 2555),
            DataInventoryItem("audit_log", "old_values", "JSONB", DataClassification.RESTRICTED, False, True, 2555),
            DataInventoryItem("audit_log", "new_values", "JSONB", DataClassification.RESTRICTED, False, True, 2555),
            DataInventoryItem("audit_log", "ip_address", "INET", DataClassification.PII, True, True, 2555, "legitimate_interest", "security")
        ]
        
        self.data_inventory = inventory_items
        
        # Store in database
        async with self.connection_manager.get_connection() as conn:
            for item in inventory_items:
                await conn.execute("""
                    INSERT INTO compliance.data_inventory (
                        schema_name, table_name, column_name, data_type,
                        classification, contains_pii, encryption_required,
                        retention_period_days, legal_basis, purpose
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (schema_name, table_name, column_name) DO UPDATE SET
                        classification = EXCLUDED.classification,
                        contains_pii = EXCLUDED.contains_pii,
                        encryption_required = EXCLUDED.encryption_required,
                        retention_period_days = EXCLUDED.retention_period_days,
                        legal_basis = EXCLUDED.legal_basis,
                        purpose = EXCLUDED.purpose,
                        updated_at = CURRENT_TIMESTAMP
                """, "public", item.table_name, item.column_name, item.data_type,
                    item.classification.value, item.contains_pii, item.encryption_required,
                    item.retention_period_days, item.legal_basis, item.purpose)
                    
    async def _setup_retention_policies(self):
        """Set up data retention policies based on compliance rules"""
        
        async with self.connection_manager.get_connection() as conn:
            # Clear existing retention schedules
            await conn.execute("DELETE FROM compliance.retention_schedule")
            
            # Create retention schedules based on compliance rules
            for framework, rules in self.compliance_rules.items():
                for rule in rules:
                    for table in rule.applies_to_tables:
                        if rule.applies_to_columns == ["*"]:
                            # Apply to entire table
                            await conn.execute("""
                                INSERT INTO compliance.retention_schedule (
                                    table_name, retention_period_days, retention_action,
                                    legal_basis, next_review_date
                                ) VALUES ($1, $2, $3, $4, $5)
                            """, table, rule.retention_period_days, rule.retention_action.value,
                                framework.value, datetime.now().date() + timedelta(days=90))
                        else:
                            # Apply to specific columns
                            for column in rule.applies_to_columns:
                                await conn.execute("""
                                    INSERT INTO compliance.retention_schedule (
                                        table_name, column_name, retention_period_days,
                                        retention_action, legal_basis, next_review_date
                                    ) VALUES ($1, $2, $3, $4, $5, $6)
                                """, table, column, rule.retention_period_days,
                                    rule.retention_action.value, framework.value,
                                    datetime.now().date() + timedelta(days=90))
                                    
    async def _setup_compliance_monitoring(self):
        """Set up monitoring for compliance violations"""
        
        async with self.connection_manager.get_connection() as conn:
            # Create compliance monitoring views
            await conn.execute("""
                CREATE OR REPLACE VIEW compliance.gdpr_compliance_status AS
                SELECT 
                    'consent_coverage' as metric,
                    COUNT(DISTINCT u.id) as total_users,
                    COUNT(DISTINCT cr.user_id) as users_with_consent,
                    ROUND(
                        (COUNT(DISTINCT cr.user_id)::decimal / COUNT(DISTINCT u.id)) * 100, 2
                    ) as compliance_percentage
                FROM users u
                LEFT JOIN compliance.consent_records cr ON u.id = cr.user_id 
                    AND cr.consent_given = true 
                    AND cr.withdrawal_date IS NULL
            """)
            
            await conn.execute("""
                CREATE OR REPLACE VIEW compliance.data_retention_violations AS
                SELECT 
                    rs.table_name,
                    rs.column_name,
                    rs.retention_period_days,
                    rs.legal_basis,
                    COUNT(*) as violation_count
                FROM compliance.retention_schedule rs
                JOIN information_schema.tables t ON t.table_name = rs.table_name
                WHERE rs.enabled = true
                  AND rs.last_executed < NOW() - INTERVAL '1 day' * rs.retention_period_days
                GROUP BY rs.table_name, rs.column_name, rs.retention_period_days, rs.legal_basis
            """)
            
    async def record_consent(self, user_id: str, purpose: str, consent_given: bool,
                           data_categories: List[str], legal_basis: str = "consent",
                           ip_address: str = None, user_agent: str = None) -> str:
        """Record user consent for data processing"""
        
        async with self.connection_manager.get_connection() as conn:
            consent_id = await conn.fetchval("""
                INSERT INTO compliance.consent_records (
                    user_id, purpose, consent_given, legal_basis,
                    data_categories, ip_address, user_agent
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING consent_id
            """, user_id, purpose, consent_given, legal_basis,
                data_categories, ip_address, user_agent)
            
            # Log compliance event
            await self._log_compliance_event(
                framework=ComplianceFramework.GDPR,
                event_type="consent_recorded",
                user_id=user_id,
                details={
                    "purpose": purpose,
                    "consent_given": consent_given,
                    "data_categories": data_categories
                }
            )
            
            self.logger.info(f"Recorded consent for user {user_id}: {purpose} = {consent_given}")
            return str(consent_id)
            
    async def withdraw_consent(self, user_id: str, purpose: str) -> bool:
        """Withdraw user consent for specific purpose"""
        
        async with self.connection_manager.get_connection() as conn:
            # Update existing consent record
            updated = await conn.execute("""
                UPDATE compliance.consent_records 
                SET withdrawal_date = CURRENT_TIMESTAMP
                WHERE user_id = $1 AND purpose = $2 
                  AND consent_given = true 
                  AND withdrawal_date IS NULL
            """, user_id, purpose)
            
            if "UPDATE 1" in updated:
                # Log compliance event
                await self._log_compliance_event(
                    framework=ComplianceFramework.GDPR,
                    event_type="consent_withdrawn",
                    user_id=user_id,
                    details={"purpose": purpose}
                )
                
                self.logger.info(f"Consent withdrawn for user {user_id}: {purpose}")
                return True
                
        return False
        
    async def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given valid consent for purpose"""
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            consent_valid = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM compliance.consent_records
                    WHERE user_id = $1 AND purpose = $2
                      AND consent_given = true
                      AND withdrawal_date IS NULL
                )
            """, user_id, purpose)
            
            return consent_valid
            
    async def handle_data_subject_request(self, request_type: str, user_email: str,
                                        user_id: str = None) -> str:
        """Handle GDPR data subject requests"""
        
        async with self.connection_manager.get_connection() as conn:
            request_id = await conn.fetchval("""
                INSERT INTO compliance.data_subject_requests (
                    request_type, user_id, user_email, request_details
                ) VALUES ($1, $2, $3, $4)
                RETURNING request_id
            """, request_type, user_id, user_email, {
                "submitted_at": datetime.now().isoformat(),
                "auto_generated": True
            })
            
            # Process request based on type
            if request_type == "access":
                await self._process_access_request(str(request_id), user_id or user_email)
            elif request_type == "erasure":
                await self._process_erasure_request(str(request_id), user_id or user_email)
            elif request_type == "portability":
                await self._process_portability_request(str(request_id), user_id or user_email)
            elif request_type == "rectification":
                await self._process_rectification_request(str(request_id), user_id or user_email)
                
            # Log compliance event
            await self._log_compliance_event(
                framework=ComplianceFramework.GDPR,
                event_type=f"data_subject_request_{request_type}",
                user_id=user_id,
                details={
                    "request_id": str(request_id),
                    "user_email": user_email
                }
            )
            
            self.logger.info(f"Created data subject request: {request_type} for {user_email}")
            return str(request_id)
            
    async def _process_access_request(self, request_id: str, user_identifier: str):
        """Process GDPR Article 15 - Right of access"""
        
        async with self.connection_manager.get_connection() as conn:
            # Collect all data for the user
            user_data = {}
            
            # User profile data
            user_profile = await conn.fetchrow("""
                SELECT email, created_at, updated_at, min_price, max_price,
                       min_bedrooms, max_bedrooms, preferred_locations,
                       required_amenities, property_types, last_login
                FROM users WHERE id = $1 OR email = $1
            """, user_identifier)
            
            if user_profile:
                user_data['profile'] = dict(user_profile)
                user_id = user_profile.get('id')  # Get actual user ID
                
                # User interactions
                interactions = await conn.fetch("""
                    SELECT interaction_type, timestamp, duration_seconds,
                           session_id, ip_address
                    FROM user_interactions 
                    WHERE user_id = $1
                    ORDER BY timestamp DESC
                """, user_id)
                
                user_data['interactions'] = [dict(i) for i in interactions]
                
                # Search queries
                searches = await conn.fetch("""
                    SELECT query_text, filters, results_count, created_at
                    FROM search_queries 
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                """, user_id)
                
                user_data['search_history'] = [dict(s) for s in searches]
                
                # Consent records
                consents = await conn.fetch("""
                    SELECT purpose, consent_given, consent_date, withdrawal_date,
                           legal_basis, data_categories
                    FROM compliance.consent_records
                    WHERE user_id = $1
                    ORDER BY consent_date DESC
                """, user_id)
                
                user_data['consents'] = [dict(c) for c in consents]
                
            # Update request with collected data
            await conn.execute("""
                UPDATE compliance.data_subject_requests
                SET status = 'completed',
                    completed_date = CURRENT_TIMESTAMP,
                    response_data = $1
                WHERE request_id = $2
            """, user_data, request_id)
            
    async def _process_erasure_request(self, request_id: str, user_identifier: str):
        """Process GDPR Article 17 - Right to erasure"""
        
        async with self.connection_manager.get_connection() as conn:
            # Find user
            user_data = await conn.fetchrow("""
                SELECT id, email FROM users WHERE id = $1 OR email = $1
            """, user_identifier)
            
            if not user_data:
                await conn.execute("""
                    UPDATE compliance.data_subject_requests
                    SET status = 'completed',
                        completed_date = CURRENT_TIMESTAMP,
                        notes = 'No user data found'
                    WHERE request_id = $1
                """, request_id)
                return
                
            user_id = user_data['id']
            
            # Check if erasure is legally required or if there are legitimate interests
            # In a real system, this would involve complex business logic
            
            # For now, anonymize rather than delete to preserve analytics
            anonymized_email = f"anonymized_{user_id}@deleted.local"
            
            # Anonymize user data
            await conn.execute("""
                UPDATE users 
                SET email = $1,
                    preferred_locations = '{}',
                    required_amenities = '{}'
                WHERE id = $2
            """, anonymized_email, user_id)
            
            # Anonymize interaction data
            await conn.execute("""
                UPDATE user_interactions
                SET ip_address = '127.0.0.1',
                    user_agent = 'anonymized'
                WHERE user_id = $1
            """, user_id)
            
            # Record erasure completion
            await conn.execute("""
                UPDATE compliance.data_subject_requests
                SET status = 'completed',
                    completed_date = CURRENT_TIMESTAMP,
                    notes = 'Data anonymized as per retention policy'
                WHERE request_id = $1
            """, request_id)
            
    async def _process_portability_request(self, request_id: str, user_identifier: str):
        """Process GDPR Article 20 - Right to data portability"""
        
        # This would generate a machine-readable export of user data
        # Similar to access request but in structured format (JSON, CSV, etc.)
        await self._process_access_request(request_id, user_identifier)
        
    async def _process_rectification_request(self, request_id: str, user_identifier: str):
        """Process GDPR Article 16 - Right to rectification"""
        
        # This would require additional input about what needs to be corrected
        # For now, just mark as requiring manual processing
        
        async with self.connection_manager.get_connection() as conn:
            await conn.execute("""
                UPDATE compliance.data_subject_requests
                SET status = 'requires_manual_processing',
                    notes = 'Rectification requests require manual review'
                WHERE request_id = $1
            """, request_id)
            
    async def run_retention_cleanup(self) -> Dict[str, Any]:
        """Run data retention cleanup based on policies"""
        
        cleanup_results = {
            'tables_processed': [],
            'records_affected': 0,
            'errors': []
        }
        
        async with self.connection_manager.get_connection() as conn:
            # Get active retention schedules
            schedules = await conn.fetch("""
                SELECT table_name, column_name, retention_period_days, 
                       retention_action, legal_basis
                FROM compliance.retention_schedule
                WHERE enabled = true
                  AND (last_executed IS NULL OR 
                       last_executed < NOW() - INTERVAL '1 day')
            """)
            
            for schedule in schedules:
                table_name = schedule['table_name']
                retention_days = schedule['retention_period_days']
                action = schedule['retention_action']
                
                try:
                    cutoff_date = datetime.now() - timedelta(days=retention_days)
                    
                    if action == 'delete':
                        result = await conn.execute(f"""
                            DELETE FROM {table_name} 
                            WHERE created_at < $1
                        """, cutoff_date)
                        
                    elif action == 'anonymize':
                        if table_name == 'users':
                            result = await conn.execute(f"""
                                UPDATE {table_name}
                                SET email = 'anonymized_' || id || '@deleted.local',
                                    preferred_locations = '{{}}',
                                    required_amenities = '{{}}'
                                WHERE created_at < $1
                            """, cutoff_date)
                        elif table_name == 'user_interactions':
                            result = await conn.execute(f"""
                                UPDATE {table_name}
                                SET ip_address = '127.0.0.1',
                                    user_agent = 'anonymized'
                                WHERE timestamp < $1
                            """, cutoff_date)
                            
                    elif action == 'archive':
                        # In a real system, this would move data to archive storage
                        # For now, just mark as archived
                        result = await conn.execute(f"""
                            UPDATE {table_name}
                            SET status = 'archived'
                            WHERE created_at < $1
                              AND status != 'archived'
                        """, cutoff_date)
                        
                    # Extract affected rows count
                    affected_rows = int(result.split()[-1]) if result.split() else 0
                    cleanup_results['records_affected'] += affected_rows
                    
                    # Update last executed
                    await conn.execute("""
                        UPDATE compliance.retention_schedule
                        SET last_executed = CURRENT_TIMESTAMP
                        WHERE table_name = $1 AND column_name IS NOT DISTINCT FROM $2
                    """, table_name, schedule['column_name'])
                    
                    cleanup_results['tables_processed'].append({
                        'table': table_name,
                        'action': action,
                        'records_affected': affected_rows
                    })
                    
                    self.logger.info(f"Retention cleanup: {action} {affected_rows} records from {table_name}")
                    
                except Exception as e:
                    error_msg = f"Failed retention cleanup for {table_name}: {e}"
                    cleanup_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
        return cleanup_results
        
    async def _log_compliance_event(self, framework: ComplianceFramework, event_type: str,
                                  table_name: str = None, user_id: str = None,
                                  details: Dict[str, Any] = None):
        """Log compliance-related events"""
        
        try:
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO compliance.audit_log (
                        framework, event_type, table_name, user_id, details
                    ) VALUES ($1, $2, $3, $4, $5)
                """, framework.value, event_type, table_name, user_id, details)
                
        except Exception as e:
            self.logger.error(f"Failed to log compliance event: {e}")
            
    async def generate_compliance_report(self, framework: ComplianceFramework,
                                       start_date: datetime = None,
                                       end_date: datetime = None) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        report = {
            'framework': framework.value,
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat()
        }
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            if framework == ComplianceFramework.GDPR:
                # GDPR-specific metrics
                
                # Consent metrics
                consent_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_consent_records,
                        COUNT(*) FILTER (WHERE consent_given = true) as active_consents,
                        COUNT(*) FILTER (WHERE withdrawal_date IS NOT NULL) as withdrawn_consents,
                        COUNT(DISTINCT user_id) as users_with_consent
                    FROM compliance.consent_records
                    WHERE consent_date BETWEEN $1 AND $2
                """, start_date, end_date)
                
                # Data subject requests
                dsr_stats = await conn.fetch("""
                    SELECT request_type, status, COUNT(*) as count
                    FROM compliance.data_subject_requests
                    WHERE request_date BETWEEN $1 AND $2
                    GROUP BY request_type, status
                """, start_date, end_date)
                
                # Data retention compliance
                retention_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_schedules,
                        COUNT(*) FILTER (WHERE last_executed IS NOT NULL) as executed_schedules,
                        COUNT(*) FILTER (WHERE enabled = true) as active_schedules
                    FROM compliance.retention_schedule
                """)
                
                report['gdpr_metrics'] = {
                    'consent_statistics': dict(consent_stats) if consent_stats else {},
                    'data_subject_requests': [dict(d) for d in dsr_stats],
                    'retention_compliance': dict(retention_stats) if retention_stats else {}
                }
                
            elif framework == ComplianceFramework.SOX:
                # SOX-specific metrics
                
                # Audit trail completeness
                audit_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_audit_events,
                        COUNT(DISTINCT username) as unique_users,
                        COUNT(DISTINCT table_name) as tables_audited
                    FROM security.audit_log
                    WHERE timestamp BETWEEN $1 AND $2
                """, start_date, end_date)
                
                report['sox_metrics'] = {
                    'audit_trail_statistics': dict(audit_stats) if audit_stats else {}
                }
                
            # Common compliance events
            compliance_events = await conn.fetch("""
                SELECT event_type, COUNT(*) as count
                FROM compliance.audit_log
                WHERE framework = $1 
                  AND timestamp BETWEEN $2 AND $3
                GROUP BY event_type
                ORDER BY count DESC
            """, framework.value, start_date, end_date)
            
            report['compliance_events'] = [dict(e) for e in compliance_events]
            
            # Violations
            violations = await conn.fetch("""
                SELECT violation_type, severity, COUNT(*) as count
                FROM compliance.violations
                WHERE framework = $1
                  AND detected_at BETWEEN $2 AND $3
                GROUP BY violation_type, severity
            """, framework.value, start_date, end_date)
            
            report['violations'] = [dict(v) for v in violations]
            
        return report
        
    async def check_compliance_status(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Check current compliance status for framework"""
        
        status = {
            'framework': framework.value,
            'overall_status': 'compliant',
            'checked_at': datetime.now().isoformat(),
            'issues': [],
            'recommendations': []
        }
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            if framework == ComplianceFramework.GDPR:
                # Check consent coverage
                consent_coverage = await conn.fetchval("""
                    SELECT 
                        COALESCE(
                            (COUNT(DISTINCT cr.user_id)::decimal / NULLIF(COUNT(DISTINCT u.id), 0)) * 100,
                            0
                        )
                    FROM users u
                    LEFT JOIN compliance.consent_records cr ON u.id = cr.user_id 
                        AND cr.consent_given = true 
                        AND cr.withdrawal_date IS NULL
                """)
                
                if consent_coverage < 95:
                    status['overall_status'] = 'non_compliant'
                    status['issues'].append(f"Low consent coverage: {consent_coverage:.1f}%")
                    status['recommendations'].append("Implement consent collection for existing users")
                    
                # Check for overdue data subject requests
                overdue_requests = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM compliance.data_subject_requests
                    WHERE status IN ('pending', 'in_progress')
                      AND request_date < NOW() - INTERVAL '30 days'
                """)
                
                if overdue_requests > 0:
                    status['overall_status'] = 'non_compliant'
                    status['issues'].append(f"{overdue_requests} overdue data subject requests")
                    status['recommendations'].append("Process pending data subject requests within 30 days")
                    
            # Check retention policy execution
            overdue_retention = await conn.fetchval("""
                SELECT COUNT(*)
                FROM compliance.retention_schedule
                WHERE enabled = true
                  AND (last_executed IS NULL OR 
                       last_executed < NOW() - INTERVAL '7 days')
            """)
            
            if overdue_retention > 0:
                if status['overall_status'] == 'compliant':
                    status['overall_status'] = 'requires_attention'
                status['issues'].append(f"{overdue_retention} overdue retention policies")
                status['recommendations'].append("Execute overdue data retention policies")
                
        return status