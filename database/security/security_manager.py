"""
Database Security Manager
Comprehensive security implementation with RBAC, encryption, audit logging, and compliance
"""

import asyncio
import logging
import hashlib
import secrets
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncpg
from asyncpg import Connection
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt


class SecurityRole(Enum):
    ADMIN = "admin"
    DBA = "dba"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    API_USER = "api_user"
    READ_ONLY = "read_only"
    ML_TRAINER = "ml_trainer"
    SCRAPER = "scraper"


class PermissionType(Enum):
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    USAGE = "USAGE"
    CREATE = "CREATE"
    CONNECT = "CONNECT"


class AuditEventType(Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    QUERY_EXECUTE = "query_execute"
    SCHEMA_CHANGE = "schema_change"
    PERMISSION_CHANGE = "permission_change"
    DATA_ACCESS = "data_access"
    SECURITY_VIOLATION = "security_violation"
    EXPORT = "export"
    IMPORT = "import"


class EncryptionLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    SENSITIVE = "sensitive"
    HIGHLY_SENSITIVE = "highly_sensitive"


@dataclass
class SecurityPolicy:
    name: str
    description: str
    roles: List[SecurityRole]
    tables: List[str]
    permissions: List[PermissionType]
    conditions: Optional[Dict[str, Any]] = None
    time_restrictions: Optional[Dict[str, Any]] = None
    ip_restrictions: Optional[List[str]] = None
    enabled: bool = True


@dataclass
class UserAccount:
    username: str
    email: str
    roles: List[SecurityRole]
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    account_expires: Optional[datetime] = None
    must_change_password: bool = False
    two_factor_enabled: bool = False
    api_key: Optional[str] = None
    session_token: Optional[str] = None


@dataclass
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    username: str
    table_name: Optional[str]
    operation: Optional[str]
    timestamp: datetime
    ip_address: str
    user_agent: Optional[str]
    query_text: Optional[str]
    affected_rows: int = 0
    success: bool = True
    error_message: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class DatabaseSecurityManager:
    """
    Enterprise-grade database security manager with:
    - Role-based access control (RBAC)
    - Data encryption at rest and in transit
    - Comprehensive audit logging
    - Security policy enforcement
    - Compliance reporting
    - Threat detection and response
    """
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # Encryption setup
        self._encryption_key = None
        self._field_encryption_keys: Dict[str, Fernet] = {}
        
        # Security policies
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.user_accounts: Dict[str, UserAccount] = {}
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Security configuration
        self.password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special': True,
            'max_age_days': 90,
            'history_count': 5
        }
        
        self.session_policy = {
            'timeout_minutes': 480,  # 8 hours
            'max_concurrent_sessions': 5,
            'require_secure_transport': True
        }
        
        # Initialize default security policies
        self._initialize_security_policies()
        
    async def initialize_security(self):
        """Initialize database security components"""
        try:
            self.logger.info("Initializing database security system...")
            
            # Create security schema and tables
            await self._create_security_schema()
            
            # Set up encryption
            await self._initialize_encryption()
            
            # Create default roles and users
            await self._create_default_roles()
            
            # Set up audit logging
            await self._initialize_audit_logging()
            
            # Configure row-level security
            await self._configure_row_level_security()
            
            # Set up security monitoring
            await self._initialize_security_monitoring()
            
            self.logger.info("Database security system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security system: {e}")
            raise
            
    async def _create_security_schema(self):
        """Create security-related database schema"""
        async with self.connection_manager.get_connection() as conn:
            # Create security schema
            await conn.execute("CREATE SCHEMA IF NOT EXISTS security")
            
            # User accounts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security.user_accounts (
                    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    username VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    roles TEXT[] NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP WITH TIME ZONE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    account_locked BOOLEAN DEFAULT FALSE,
                    account_expires TIMESTAMP WITH TIME ZONE,
                    must_change_password BOOLEAN DEFAULT TRUE,
                    two_factor_enabled BOOLEAN DEFAULT FALSE,
                    two_factor_secret TEXT,
                    api_key_hash TEXT,
                    password_history TEXT[] DEFAULT '{}',
                    created_by VARCHAR(255),
                    
                    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
                    CONSTRAINT valid_username CHECK (username ~* '^[a-zA-Z0-9_-]{3,50}$')
                )
            """)
            
            # User sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security.user_sessions (
                    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES security.user_accounts(user_id) ON DELETE CASCADE,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    ip_address INET,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Security policies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security.security_policies (
                    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    policy_name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    policy_definition JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    created_by VARCHAR(255),
                    enabled BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Audit log table (partitioned by date)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security.audit_log (
                    audit_id UUID DEFAULT uuid_generate_v4(),
                    event_type VARCHAR(50) NOT NULL,
                    username VARCHAR(255),
                    user_id UUID,
                    table_name VARCHAR(255),
                    operation VARCHAR(20),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    ip_address INET,
                    user_agent TEXT,
                    query_text TEXT,
                    query_params JSONB,
                    affected_rows INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    session_id UUID,
                    additional_data JSONB,
                    
                    PRIMARY KEY (audit_id, timestamp)
                ) PARTITION BY RANGE (timestamp)
            """)
            
            # Create audit log partitions for current and next months
            current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = (current_month + timedelta(days=32)).replace(day=1)
            
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS security.audit_log_{current_month.strftime('%Y_%m')}
                PARTITION OF security.audit_log
                FOR VALUES FROM ('{current_month}') TO ('{next_month}')
            """)
            
            # Data classification table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security.data_classification (
                    classification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    schema_name VARCHAR(255) NOT NULL,
                    table_name VARCHAR(255) NOT NULL,
                    column_name VARCHAR(255),
                    classification_level VARCHAR(50) NOT NULL,
                    encryption_required BOOLEAN DEFAULT FALSE,
                    masking_required BOOLEAN DEFAULT FALSE,
                    retention_period_days INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE (schema_name, table_name, column_name)
                )
            """)
            
            # Encryption keys table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security.encryption_keys (
                    key_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    key_name VARCHAR(255) UNIQUE NOT NULL,
                    key_purpose VARCHAR(100) NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    key_version INTEGER DEFAULT 1,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP WITH TIME ZONE,
                    is_active BOOLEAN DEFAULT TRUE,
                    rotation_schedule VARCHAR(50)
                )
            """)
            
            # Security violations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security.security_violations (
                    violation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    violation_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    username VARCHAR(255),
                    ip_address INET,
                    description TEXT NOT NULL,
                    detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP WITH TIME ZONE,
                    resolved_by VARCHAR(255),
                    additional_data JSONB
                )
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_accounts_username ON security.user_accounts(username);
                CREATE INDEX IF NOT EXISTS idx_user_accounts_email ON security.user_accounts(email);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON security.user_sessions(session_token);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON security.user_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON security.audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_log_username ON security.audit_log(username);
                CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON security.audit_log(event_type);
                CREATE INDEX IF NOT EXISTS idx_security_violations_timestamp ON security.security_violations(detection_timestamp);
                CREATE INDEX IF NOT EXISTS idx_data_classification_table ON security.data_classification(schema_name, table_name);
            """)
            
    async def _initialize_encryption(self):
        """Initialize encryption system"""
        # Generate master encryption key (in production, use proper key management)
        password = b"your-secret-key-here"  # Should be from secure config
        salt = b"stable-salt-value"  # Should be from secure config
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self._encryption_key = Fernet(key)
        
        # Initialize field-level encryption keys
        sensitive_fields = [
            'users.email',
            'properties.contact_info',
            'user_interactions.ip_address',
            'audit_log.query_text'
        ]
        
        for field in sensitive_fields:
            field_key = Fernet.generate_key()
            self._field_encryption_keys[field] = Fernet(field_key)
            
        self.logger.info("Encryption system initialized")
        
    def _initialize_security_policies(self):
        """Initialize default security policies"""
        
        # Admin policy - full access
        self.security_policies['admin_policy'] = SecurityPolicy(
            name='admin_policy',
            description='Full administrative access',
            roles=[SecurityRole.ADMIN, SecurityRole.DBA],
            tables=['*'],
            permissions=[p for p in PermissionType],
            enabled=True
        )
        
        # Developer policy - development access
        self.security_policies['developer_policy'] = SecurityPolicy(
            name='developer_policy',
            description='Development environment access',
            roles=[SecurityRole.DEVELOPER],
            tables=['users', 'properties', 'user_interactions', 'search_queries', 'ml_models'],
            permissions=[PermissionType.SELECT, PermissionType.INSERT, PermissionType.UPDATE],
            conditions={'environment': 'development'}
        )
        
        # Analyst policy - read-only analytics
        self.security_policies['analyst_policy'] = SecurityPolicy(
            name='analyst_policy',
            description='Read-only access for analytics',
            roles=[SecurityRole.ANALYST],
            tables=['properties', 'user_interactions', 'search_queries', 'training_metrics'],
            permissions=[PermissionType.SELECT],
            time_restrictions={
                'allowed_hours': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 8 AM to 5 PM
            }
        )
        
        # API user policy - limited API access
        self.security_policies['api_user_policy'] = SecurityPolicy(
            name='api_user_policy',
            description='Limited API access for applications',
            roles=[SecurityRole.API_USER],
            tables=['properties', 'user_interactions', 'search_queries'],
            permissions=[PermissionType.SELECT, PermissionType.INSERT]
        )
        
        # ML trainer policy - ML model access
        self.security_policies['ml_trainer_policy'] = SecurityPolicy(
            name='ml_trainer_policy',
            description='ML training and model management',
            roles=[SecurityRole.ML_TRAINER],
            tables=['properties', 'user_interactions', 'ml_models', 'training_metrics', 'embeddings'],
            permissions=[PermissionType.SELECT, PermissionType.INSERT, PermissionType.UPDATE]
        )
        
        # Scraper policy - data ingestion
        self.security_policies['scraper_policy'] = SecurityPolicy(
            name='scraper_policy',
            description='Data scraping and ingestion',
            roles=[SecurityRole.SCRAPER],
            tables=['properties'],
            permissions=[PermissionType.INSERT, PermissionType.UPDATE],
            ip_restrictions=['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']  # Private networks only
        )
        
    async def _create_default_roles(self):
        """Create default database roles"""
        async with self.connection_manager.get_connection() as conn:
            roles_to_create = [
                ('rental_ml_admin', 'Administrative role with full access'),
                ('rental_ml_developer', 'Development role with limited access'),
                ('rental_ml_analyst', 'Read-only analytics role'),
                ('rental_ml_api', 'API application role'),
                ('rental_ml_ml_trainer', 'ML training role'),
                ('rental_ml_scraper', 'Data scraping role'),
                ('rental_ml_readonly', 'Read-only role')
            ]
            
            for role_name, description in roles_to_create:
                try:
                    await conn.execute(f"CREATE ROLE {role_name}")
                    self.logger.info(f"Created role: {role_name}")
                except Exception as e:
                    if "already exists" not in str(e):
                        self.logger.error(f"Failed to create role {role_name}: {e}")
                        
            # Configure role permissions
            await self._configure_role_permissions(conn)
            
    async def _configure_role_permissions(self, conn: Connection):
        """Configure permissions for database roles"""
        
        # Admin role - full access
        admin_permissions = [
            "GRANT ALL PRIVILEGES ON DATABASE rental_ml TO rental_ml_admin",
            "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rental_ml_admin",
            "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rental_ml_admin",
            "GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO rental_ml_admin",
            "GRANT ALL PRIVILEGES ON SCHEMA security TO rental_ml_admin",
            "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA security TO rental_ml_admin"
        ]
        
        for permission in admin_permissions:
            try:
                await conn.execute(permission)
            except Exception as e:
                self.logger.warning(f"Permission already exists or failed: {e}")
                
        # Developer role - limited access
        developer_permissions = [
            "GRANT CONNECT ON DATABASE rental_ml TO rental_ml_developer",
            "GRANT USAGE ON SCHEMA public TO rental_ml_developer",
            "GRANT SELECT, INSERT, UPDATE ON users, properties, user_interactions TO rental_ml_developer",
            "GRANT SELECT ON ALL TABLES IN SCHEMA public TO rental_ml_developer",
            "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rental_ml_developer"
        ]
        
        for permission in developer_permissions:
            try:
                await conn.execute(permission)
            except Exception as e:
                self.logger.warning(f"Permission already exists or failed: {e}")
                
        # Analyst role - read-only
        analyst_permissions = [
            "GRANT CONNECT ON DATABASE rental_ml TO rental_ml_analyst",
            "GRANT USAGE ON SCHEMA public TO rental_ml_analyst",
            "GRANT SELECT ON properties, user_interactions, search_queries, training_metrics TO rental_ml_analyst"
        ]
        
        for permission in analyst_permissions:
            try:
                await conn.execute(permission)
            except Exception as e:
                self.logger.warning(f"Permission already exists or failed: {e}")
                
        # API role - application access
        api_permissions = [
            "GRANT CONNECT ON DATABASE rental_ml TO rental_ml_api",
            "GRANT USAGE ON SCHEMA public TO rental_ml_api",
            "GRANT SELECT, INSERT ON properties, user_interactions, search_queries TO rental_ml_api",
            "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rental_ml_api"
        ]
        
        for permission in api_permissions:
            try:
                await conn.execute(permission)
            except Exception as e:
                self.logger.warning(f"Permission already exists or failed: {e}")
                
    async def _initialize_audit_logging(self):
        """Initialize audit logging system"""
        async with self.connection_manager.get_connection() as conn:
            # Create audit trigger function
            await conn.execute("""
                CREATE OR REPLACE FUNCTION security.audit_trigger_function()
                RETURNS TRIGGER AS $$
                BEGIN
                    INSERT INTO security.audit_log (
                        event_type, username, table_name, operation,
                        timestamp, affected_rows, additional_data
                    ) VALUES (
                        'data_access',
                        current_user,
                        TG_TABLE_NAME,
                        TG_OP,
                        CURRENT_TIMESTAMP,
                        CASE WHEN TG_OP = 'DELETE' THEN 1 ELSE 1 END,
                        jsonb_build_object(
                            'old_values', CASE WHEN TG_OP IN ('UPDATE', 'DELETE') THEN row_to_json(OLD) ELSE NULL END,
                            'new_values', CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END
                        )
                    );
                    
                    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
                END;
                $$ LANGUAGE plpgsql SECURITY DEFINER;
            """)
            
            # Add audit triggers to sensitive tables
            sensitive_tables = ['users', 'properties', 'user_interactions', 'ml_models']
            
            for table in sensitive_tables:
                try:
                    await conn.execute(f"""
                        CREATE TRIGGER audit_trigger_{table}
                        AFTER INSERT OR UPDATE OR DELETE ON {table}
                        FOR EACH ROW EXECUTE FUNCTION security.audit_trigger_function()
                    """)
                    self.logger.info(f"Created audit trigger for table: {table}")
                except Exception as e:
                    if "already exists" not in str(e):
                        self.logger.error(f"Failed to create audit trigger for {table}: {e}")
                        
    async def _configure_row_level_security(self):
        """Configure row-level security policies"""
        async with self.connection_manager.get_connection() as conn:
            # Enable RLS on sensitive tables
            rls_tables = ['users', 'user_interactions', 'search_queries']
            
            for table in rls_tables:
                try:
                    await conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
                    self.logger.info(f"Enabled RLS for table: {table}")
                except Exception as e:
                    self.logger.error(f"Failed to enable RLS for {table}: {e}")
                    
            # Create RLS policies
            
            # Users can only see their own data
            await conn.execute("""
                CREATE POLICY user_isolation_policy ON users
                FOR ALL TO rental_ml_api
                USING (id = current_setting('app.current_user_id')::uuid)
            """)
            
            # User interactions policy
            await conn.execute("""
                CREATE POLICY user_interactions_policy ON user_interactions
                FOR ALL TO rental_ml_api
                USING (user_id = current_setting('app.current_user_id')::uuid)
            """)
            
            # Allow admin and analyst roles to see all data
            for table in rls_tables:
                await conn.execute(f"""
                    CREATE POLICY admin_access_policy_{table} ON {table}
                    FOR ALL TO rental_ml_admin, rental_ml_analyst
                    USING (true)
                """)
                
    async def _initialize_security_monitoring(self):
        """Initialize security monitoring and alerting"""
        # This would set up real-time monitoring for security events
        # For now, we'll create a basic monitoring structure
        
        async with self.connection_manager.get_connection() as conn:
            # Create security monitoring views
            await conn.execute("""
                CREATE OR REPLACE VIEW security.recent_failed_logins AS
                SELECT 
                    username,
                    COUNT(*) as failed_attempts,
                    MAX(timestamp) as last_attempt,
                    array_agg(DISTINCT ip_address::text) as source_ips
                FROM security.audit_log 
                WHERE event_type = 'login' 
                  AND success = false 
                  AND timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY username
                HAVING COUNT(*) >= 3
            """)
            
            await conn.execute("""
                CREATE OR REPLACE VIEW security.suspicious_activity AS
                SELECT 
                    username,
                    ip_address,
                    COUNT(*) as query_count,
                    array_agg(DISTINCT table_name) as accessed_tables,
                    MIN(timestamp) as first_activity,
                    MAX(timestamp) as last_activity
                FROM security.audit_log 
                WHERE timestamp > NOW() - INTERVAL '10 minutes'
                  AND event_type = 'query_execute'
                GROUP BY username, ip_address
                HAVING COUNT(*) > 100  -- More than 100 queries in 10 minutes
            """)
            
    async def create_user_account(self, username: str, email: str, 
                                roles: List[SecurityRole], password: str,
                                created_by: str) -> UserAccount:
        """Create a new user account with proper security"""
        
        # Validate password policy
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")
            
        # Generate salt and hash password
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)
        
        # Generate API key
        api_key = secrets.token_urlsafe(32)
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        async with self.connection_manager.get_connection() as conn:
            user_id = await conn.fetchval("""
                INSERT INTO security.user_accounts (
                    username, email, password_hash, salt, roles,
                    api_key_hash, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING user_id
            """, username, email, password_hash, salt, 
                [role.value for role in roles], api_key_hash, created_by)
            
            # Log the account creation
            await self._log_audit_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                username=created_by,
                operation="CREATE_USER",
                success=True,
                additional_data={'new_username': username, 'roles': [r.value for r in roles]}
            )
            
        user_account = UserAccount(
            username=username,
            email=email,
            roles=roles,
            password_hash=password_hash,
            created_at=datetime.now()
        )
        
        user_account.api_key = api_key  # Return unhashed API key
        self.user_accounts[username] = user_account
        
        self.logger.info(f"Created user account: {username}")
        return user_account
        
    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy"""
        policy = self.password_policy
        
        if len(password) < policy['min_length']:
            return False
            
        if policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
            
        if policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
            
        if policy['require_digits'] and not any(c.isdigit() for c in password):
            return False
            
        if policy['require_special'] and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            return False
            
        return True
        
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using bcrypt"""
        combined = (password + salt).encode('utf-8')
        hashed = bcrypt.hashpw(combined, bcrypt.gensalt())
        return hashed.decode('utf-8')
        
    def _verify_password(self, password: str, salt: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        combined = (password + salt).encode('utf-8')
        return bcrypt.checkpw(combined, stored_hash.encode('utf-8'))
        
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str, user_agent: str) -> Optional[str]:
        """Authenticate user and return session token"""
        
        async with self.connection_manager.get_connection() as conn:
            user_data = await conn.fetchrow("""
                SELECT user_id, username, email, password_hash, salt, roles,
                       failed_login_attempts, account_locked, account_expires,
                       must_change_password
                FROM security.user_accounts 
                WHERE username = $1
            """, username)
            
            if not user_data:
                await self._log_audit_event(
                    event_type=AuditEventType.LOGIN,
                    username=username,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    error_message="User not found"
                )
                return None
                
            # Check account status
            if user_data['account_locked']:
                await self._log_audit_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    username=username,
                    ip_address=ip_address,
                    operation="LOGIN_ATTEMPT_LOCKED_ACCOUNT",
                    success=False
                )
                return None
                
            if user_data['account_expires'] and user_data['account_expires'] < datetime.now():
                return None
                
            # Verify password
            if not self._verify_password(password, user_data['salt'], user_data['password_hash']):
                # Increment failed login attempts
                await conn.execute("""
                    UPDATE security.user_accounts 
                    SET failed_login_attempts = failed_login_attempts + 1
                    WHERE username = $1
                """, username)
                
                await self._log_audit_event(
                    event_type=AuditEventType.LOGIN,
                    username=username,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    error_message="Invalid password"
                )
                
                # Lock account after too many failures
                if user_data['failed_login_attempts'] >= 4:  # 5 total attempts
                    await conn.execute("""
                        UPDATE security.user_accounts 
                        SET account_locked = true 
                        WHERE username = $1
                    """, username)
                    
                    await self._log_audit_event(
                        event_type=AuditEventType.SECURITY_VIOLATION,
                        username=username,
                        operation="ACCOUNT_LOCKED_EXCESSIVE_FAILURES",
                        ip_address=ip_address
                    )
                    
                return None
                
            # Reset failed login attempts on successful authentication
            await conn.execute("""
                UPDATE security.user_accounts 
                SET failed_login_attempts = 0, last_login = CURRENT_TIMESTAMP
                WHERE username = $1
            """, username)
            
            # Create session
            session_token = secrets.token_urlsafe(64)
            expires_at = datetime.now() + timedelta(minutes=self.session_policy['timeout_minutes'])
            
            session_id = await conn.fetchval("""
                INSERT INTO security.user_sessions (
                    user_id, session_token, expires_at, ip_address, user_agent
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING session_id
            """, user_data['user_id'], session_token, expires_at, ip_address, user_agent)
            
            # Store session in memory
            self.active_sessions[session_token] = {
                'user_id': user_data['user_id'],
                'username': username,
                'roles': user_data['roles'],
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'ip_address': ip_address
            }
            
            await self._log_audit_event(
                event_type=AuditEventType.LOGIN,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                success=True,
                additional_data={'session_id': str(session_id)}
            )
            
            self.logger.info(f"User authenticated successfully: {username}")
            return session_token
            
    async def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user info"""
        
        if session_token not in self.active_sessions:
            # Check database for session
            async with self.connection_manager.get_connection() as conn:
                session_data = await conn.fetchrow("""
                    SELECT s.session_id, s.user_id, s.expires_at, s.ip_address,
                           u.username, u.roles
                    FROM security.user_sessions s
                    JOIN security.user_accounts u ON s.user_id = u.user_id
                    WHERE s.session_token = $1 AND s.is_active = true
                """, session_token)
                
                if not session_data or session_data['expires_at'] < datetime.now():
                    return None
                    
                # Load session into memory
                self.active_sessions[session_token] = {
                    'user_id': session_data['user_id'],
                    'username': session_data['username'],
                    'roles': session_data['roles'],
                    'expires_at': session_data['expires_at'],
                    'ip_address': session_data['ip_address']
                }
                
        session = self.active_sessions.get(session_token)
        if session and session['expires_at'] > datetime.now():
            # Update last activity
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("""
                    UPDATE security.user_sessions 
                    SET last_activity = CURRENT_TIMESTAMP 
                    WHERE session_token = $1
                """, session_token)
                
            return session
            
        return None
        
    async def logout_user(self, session_token: str):
        """Logout user and invalidate session"""
        session = self.active_sessions.get(session_token)
        
        if session:
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("""
                    UPDATE security.user_sessions 
                    SET is_active = false 
                    WHERE session_token = $1
                """, session_token)
                
            await self._log_audit_event(
                event_type=AuditEventType.LOGOUT,
                username=session['username'],
                ip_address=session['ip_address'],
                success=True
            )
            
            del self.active_sessions[session_token]
            
    async def encrypt_sensitive_data(self, table_name: str, column_name: str, data: str) -> str:
        """Encrypt sensitive data using field-level encryption"""
        field_key = f"{table_name}.{column_name}"
        
        if field_key in self._field_encryption_keys:
            encrypted_data = self._field_encryption_keys[field_key].encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        else:
            # Use general encryption
            encrypted_data = self._encryption_key.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
    async def decrypt_sensitive_data(self, table_name: str, column_name: str, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        field_key = f"{table_name}.{column_name}"
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            
            if field_key in self._field_encryption_keys:
                decrypted_data = self._field_encryption_keys[field_key].decrypt(decoded_data)
            else:
                decrypted_data = self._encryption_key.decrypt(decoded_data)
                
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data for {field_key}: {e}")
            raise
            
    async def _log_audit_event(self, event_type: AuditEventType, username: str = None,
                             table_name: str = None, operation: str = None,
                             ip_address: str = None, user_agent: str = None,
                             query_text: str = None, affected_rows: int = 0,
                             success: bool = True, error_message: str = None,
                             additional_data: Dict[str, Any] = None):
        """Log audit event to audit log"""
        
        try:
            async with self.connection_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO security.audit_log (
                        event_type, username, table_name, operation,
                        ip_address, user_agent, query_text, affected_rows,
                        success, error_message, additional_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, event_type.value, username, table_name, operation,
                    ip_address, user_agent, query_text, affected_rows,
                    success, error_message, additional_data)
                    
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            
    async def check_permission(self, username: str, table_name: str, 
                             operation: PermissionType) -> bool:
        """Check if user has permission for operation on table"""
        
        if username not in self.user_accounts:
            # Load user from database
            async with self.connection_manager.get_connection() as conn:
                user_data = await conn.fetchrow("""
                    SELECT username, email, roles FROM security.user_accounts 
                    WHERE username = $1
                """, username)
                
                if not user_data:
                    return False
                    
                user_roles = [SecurityRole(role) for role in user_data['roles']]
                
        else:
            user_roles = self.user_accounts[username].roles
            
        # Check against security policies
        for policy in self.security_policies.values():
            if not policy.enabled:
                continue
                
            # Check if user has required role
            if not any(role in policy.roles for role in user_roles):
                continue
                
            # Check table access
            if '*' not in policy.tables and table_name not in policy.tables:
                continue
                
            # Check operation permission
            if operation not in policy.permissions:
                continue
                
            # Additional checks (time, IP, etc.) would go here
            
            return True
            
        return False
        
    async def get_security_report(self, start_date: datetime = None, 
                                end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        async with self.connection_manager.get_connection(analytics=True) as conn:
            # Login statistics
            login_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) FILTER (WHERE success = true) as successful_logins,
                    COUNT(*) FILTER (WHERE success = false) as failed_logins,
                    COUNT(DISTINCT username) as unique_users,
                    COUNT(DISTINCT ip_address) as unique_ips
                FROM security.audit_log 
                WHERE event_type = 'login' 
                  AND timestamp BETWEEN $1 AND $2
            """, start_date, end_date)
            
            # Security violations
            violations = await conn.fetch("""
                SELECT violation_type, severity, COUNT(*) as count
                FROM security.security_violations
                WHERE detection_timestamp BETWEEN $1 AND $2
                GROUP BY violation_type, severity
                ORDER BY count DESC
            """, start_date, end_date)
            
            # Most accessed tables
            table_access = await conn.fetch("""
                SELECT table_name, COUNT(*) as access_count
                FROM security.audit_log 
                WHERE event_type = 'data_access' 
                  AND timestamp BETWEEN $1 AND $2
                  AND table_name IS NOT NULL
                GROUP BY table_name
                ORDER BY access_count DESC
                LIMIT 10
            """, start_date, end_date)
            
            # Active sessions
            active_sessions = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM security.user_sessions 
                WHERE is_active = true 
                  AND expires_at > CURRENT_TIMESTAMP
            """)
            
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'login_statistics': dict(login_stats) if login_stats else {},
            'security_violations': [dict(v) for v in violations],
            'table_access_summary': [dict(t) for t in table_access],
            'active_sessions_count': active_sessions,
            'security_policies_count': len([p for p in self.security_policies.values() if p.enabled]),
            'user_accounts_count': len(self.user_accounts),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
        
    async def detect_security_threats(self) -> List[Dict[str, Any]]:
        """Detect potential security threats"""
        threats = []
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            # Check for brute force attacks
            brute_force = await conn.fetch("""
                SELECT username, ip_address, COUNT(*) as failed_attempts
                FROM security.audit_log 
                WHERE event_type = 'login' 
                  AND success = false 
                  AND timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY username, ip_address
                HAVING COUNT(*) >= 5
            """)
            
            for attack in brute_force:
                threats.append({
                    'type': 'brute_force_attack',
                    'severity': 'high',
                    'description': f"Multiple failed login attempts for {attack['username']} from {attack['ip_address']}",
                    'details': dict(attack)
                })
                
            # Check for unusual data access patterns
            unusual_access = await conn.fetch("""
                SELECT username, COUNT(DISTINCT table_name) as tables_accessed
                FROM security.audit_log 
                WHERE event_type = 'data_access' 
                  AND timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY username
                HAVING COUNT(DISTINCT table_name) > 5
            """)
            
            for access in unusual_access:
                threats.append({
                    'type': 'unusual_data_access',
                    'severity': 'medium',
                    'description': f"User {access['username']} accessed {access['tables_accessed']} different tables in 1 hour",
                    'details': dict(access)
                })
                
        return threats