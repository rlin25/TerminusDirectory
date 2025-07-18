# Rental ML System - Security Components

A comprehensive, production-ready security system providing authentication, authorization, session management, and security monitoring for the Rental ML System.

## 🔐 Features

### Authentication Methods
- **Password-based authentication** with secure hashing and account lockout
- **JWT token authentication** with RSA256 signing and refresh tokens
- **API key authentication** with rate limiting and IP whitelisting
- **OAuth2 integration** (Google, Facebook, Apple, Microsoft)
- **Multi-factor authentication** (TOTP, SMS, Email)

### Authorization & Access Control
- **Role-based access control (RBAC)** with flexible permission system
- **Fine-grained permissions** for different resources and actions
- **Dynamic authorization** with context-aware decisions
- **Session-based authorization** with automatic expiration

### Session Management
- **Secure session tokens** with configurable expiration
- **Concurrent session limits** per user
- **Session hijacking detection** with IP and device tracking
- **Geographic anomaly detection** for suspicious logins
- **Session analytics** and monitoring

### Security Monitoring
- **Real-time threat detection** with suspicious activity monitoring
- **Rate limiting** with Redis-backed sliding window algorithm
- **Security event logging** with threat level classification
- **Audit trail** for all security-related operations
- **Comprehensive analytics** and reporting

### Multi-Factor Authentication (MFA)
- **TOTP support** (Google Authenticator, Authy)
- **SMS verification** with multiple provider support
- **Email verification** with templated messages
- **Backup codes** for account recovery
- **QR code generation** for easy setup

### Production Features
- **Database persistence** with PostgreSQL integration
- **Redis caching** for high-performance operations
- **Email/SMS services** with multiple provider support
- **Comprehensive middleware** with FastAPI integration
- **Health checks** and monitoring endpoints
- **Docker support** with production configurations

## 📁 Component Structure

```
src/security/auth/
├── README.md                    # This file
├── __init__.py                 # Package initialization
├── models.py                   # Core security models and enums
├── security_integration.py    # Unified security manager
├── enhanced_middleware.py     # FastAPI middleware integration
├── 
├── # Core Authentication Components
├── mfa_manager.py             # Multi-factor authentication
├── api_key_manager.py         # API key management
├── oauth2_manager.py          # OAuth2 provider integration
├── session_manager.py         # Session management
├── jwt_manager.py             # JWT token management
├── 
├── # Infrastructure Integration
├── database_integration.py    # PostgreSQL persistence
├── redis_integration.py       # Redis caching and rate limiting
├── email_sms_services.py      # Email/SMS service providers
├── 
├── # Configuration and Examples
├── config_example.py          # Configuration examples
├── usage_examples.py          # Complete usage examples
└── example_usage.py           # Basic usage examples
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install fastapi uvicorn
pip install asyncpg redis
pip install pyotp qrcode phonenumbers
pip install cryptography python-jose
pip install httpx aiosmtplib jinja2
pip install sqlalchemy alembic
```

### 2. Basic Setup

```python
from fastapi import FastAPI
from src.security.auth.security_integration import SecurityManager
from src.security.auth.enhanced_middleware import SecurityMiddleware
from src.security.auth.config_example import get_security_config

# Initialize FastAPI app
app = FastAPI()

# Initialize security manager
security_config = get_security_config("development")
security_manager = SecurityManager(security_config)

# Add security middleware
app.add_middleware(SecurityMiddleware, security_manager=security_manager)

@app.on_event("startup")
async def startup():
    await security_manager.initialize()

@app.on_event("shutdown")
async def shutdown():
    await security_manager.shutdown()
```

### 3. Environment Configuration

Create a `.env` file:

```env
ENVIRONMENT=development
DATABASE_URL=postgresql+asyncpg://user:password@localhost/rental_ml
REDIS_URL=redis://localhost:6379

# OAuth2 (optional)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Email service (optional)
EMAIL_PROVIDER=smtp
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# SMS service (optional)
SMS_PROVIDER=twilio
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
```

### 4. Protected Endpoints

```python
from src.security.auth.enhanced_middleware import get_current_user, require_permissions
from src.security.auth.models import Permission

@app.get("/api/properties")
@require_permissions(Permission.READ_PROPERTY)
async def get_properties(current_user=Depends(get_current_user)):
    return {"properties": [], "user": current_user.username}
```

## 📚 Detailed Usage

### Authentication

#### Password Authentication
```python
auth_result = await security_manager.authenticate_password(
    username="user@example.com",
    password="secure_password",
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0..."
)

if auth_result.success:
    access_token = auth_result.access_token
    refresh_token = auth_result.refresh_token
```

#### JWT Token Authentication
```python
auth_result = await security_manager.authenticate_jwt_token(token)

if auth_result.success:
    security_context = auth_result.security_context
    user_permissions = security_context.permissions
```

#### API Key Authentication
```python
# Create API key
api_key_result = await security_manager.api_key_manager.create_api_key(
    name="Mobile App",
    user_id=user_id,
    permissions=[Permission.READ_PROPERTY, Permission.SEARCH_PROPERTIES],
    rate_limit=1000,
    expires_in_days=365
)

# Authenticate with API key
auth_result = await security_manager.authenticate_api_key(
    api_key="sk_...",
    ip_address="192.168.1.1"
)
```

### Multi-Factor Authentication

#### TOTP Setup
```python
# Setup TOTP
totp_result = await security_manager.setup_mfa(
    user_id=user_id,
    method=MFAMethod.TOTP,
    contact_info={"username": "user", "email": "user@example.com"}
)

# Returns QR code, secret, and backup codes
qr_code = totp_result["qr_code_data"]
backup_codes = totp_result["backup_codes"]
```

#### SMS Verification
```python
# Setup SMS MFA
sms_result = await security_manager.setup_mfa(
    user_id=user_id,
    method=MFAMethod.SMS,
    contact_info={"phone": "+1234567890"}
)

# Verify SMS code
verification_result = await security_manager.verify_mfa(
    token_id=sms_result["token_id"],
    verification_code="123456"
)
```

### Session Management

```python
# Create session
session = await security_manager.create_session(
    user_id=user_id,
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0...",
    device_fingerprint="optional_fingerprint"
)

# Validate session
session = await security_manager.validate_session(session_token)

# Invalidate session
await security_manager.invalidate_session(session_token, user_id)
```

### Authorization

```python
# Check permissions
authz_result = await security_manager.authorize_request(
    security_context=current_user_context,
    required_permissions=[Permission.CREATE_PROPERTY],
    resource="properties",
    action="create"
)

if authz_result.allowed:
    # User has permission
    pass
```

### Rate Limiting

```python
# Check rate limit
rate_result = await security_manager.check_rate_limit(
    identifier="192.168.1.1",
    limit=100,
    window_seconds=60,
    limit_type="api_requests"
)

if not rate_result["allowed"]:
    # Rate limit exceeded
    retry_after = rate_result["retry_after"]
```

### OAuth2 Integration

```python
# Get authorization URL
auth_url_data = security_manager.oauth2_manager.get_authorization_url(
    provider_name="google",
    use_pkce=True
)

# Handle callback
auth_result = await security_manager.authenticate_oauth2_callback(
    provider="google",
    code="auth_code",
    state="state_parameter",
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0..."
)
```

## 🔧 Configuration

### Security Settings

```python
security_config = {
    "jwt": {
        "algorithm": "RS256",
        "access_token_expire_minutes": 30,
        "refresh_token_expire_days": 7
    },
    "sessions": {
        "session_timeout_minutes": 480,
        "max_concurrent_sessions": 5,
        "require_secure": True
    },
    "mfa": {
        "totp_issuer": "Rental ML System",
        "token_validity_minutes": 5,
        "max_attempts": 3
    }
}
```

### Database Configuration

```python
database_config = {
    "enabled": True,
    "url": "postgresql+asyncpg://user:password@localhost/rental_ml",
    "pool_size": 10,
    "max_overflow": 20
}
```

### Redis Configuration

```python
redis_config = {
    "enabled": True,
    "url": "redis://localhost:6379",
    "socket_timeout": 5,
    "health_check_interval": 30
}
```

## 🛡️ Security Best Practices

### Password Security
- Minimum 12 characters with complexity requirements
- Password history tracking (prevents reuse of last 12 passwords)
- Automatic password expiration (90 days default)
- Account lockout after 5 failed attempts

### Token Security
- RSA256 signing for JWT tokens
- Automatic token rotation
- Token blacklisting for immediate revocation
- Short-lived access tokens (30 minutes default)

### Session Security
- Secure session token generation
- IP address and user agent tracking
- Geographic anomaly detection
- Automatic session cleanup

### Rate Limiting
- IP-based rate limiting
- User-based rate limiting
- Endpoint-specific limits
- Distributed rate limiting with Redis

### Monitoring & Logging
- All security events logged with threat levels
- Real-time suspicious activity detection
- Comprehensive audit trails
- Performance metrics and analytics

## 📊 Monitoring & Analytics

### Security Statistics
```python
stats = security_manager.get_security_statistics()
# Returns comprehensive security metrics including:
# - Authentication attempts and success rates
# - Authorization checks and denials
# - MFA usage statistics
# - API key usage
# - Session statistics
# - Rate limiting metrics
```

### Health Checks
```python
health_status = await security_manager.health_check()
# Returns health status of all components:
# - Database connectivity
# - Redis connectivity
# - Service availability
# - Response times
```

## 🐳 Docker Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  app:
    build: .
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/rental_ml
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: rental_ml
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
```

## 🧪 Testing

### Unit Tests
```python
# Example test for authentication
async def test_password_authentication():
    config = get_security_config("testing")
    async with security_manager_context(config) as sm:
        result = await sm.authenticate_password(
            username="testuser",
            password="testpassword",
            ip_address="127.0.0.1",
            user_agent="test"
        )
        assert result.success
        assert result.access_token is not None
```

### Integration Tests
```python
# Example test for API endpoint
def test_protected_endpoint():
    response = client.get(
        "/api/properties",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
```

## 📈 Performance

### Optimization Features
- Redis caching for frequently accessed data
- Database connection pooling
- Async/await throughout for non-blocking operations
- Efficient rate limiting algorithms
- Optimized JWT token verification

### Scalability
- Horizontal scaling support with Redis
- Database read/write separation ready
- Load balancer friendly (stateless design)
- Microservice architecture compatible

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Follow security best practices
5. Ensure backward compatibility

## 📄 License

This security system is part of the Rental ML System project. See the main project license for details.

## 🆘 Support

For security-related issues:
1. Check the logs for detailed error messages
2. Verify configuration settings
3. Test with minimal configuration first
4. Review the examples in `usage_examples.py`

For production deployments:
1. Use environment variables for sensitive configuration
2. Enable database and Redis persistence
3. Configure proper SSL/TLS certificates
4. Set up monitoring and alerting
5. Regularly rotate keys and secrets

## 🔗 Related Components

- **User Management**: Integration with user repository
- **Property Management**: Authorization for property operations  
- **Analytics**: Security metrics and reporting
- **Monitoring**: Health checks and performance metrics
- **API Documentation**: Swagger/OpenAPI integration