# Production API Endpoints

This directory contains comprehensive production-ready API endpoints for the Rental ML System. The API is built with FastAPI and provides a complete suite of endpoints for machine learning inference, property management, user management, administration, and monitoring.

## 🏗️ Architecture Overview

The API follows a modular architecture with separate routers for different functional areas:

- **Production API Server** (`production_api.py`) - Main FastAPI application with security and middleware
- **ML Endpoints** (`ml_endpoints.py`) - Machine learning inference and model management
- **Property Endpoints** (`property_endpoints.py`) - Property CRUD operations and management
- **User Endpoints** (`user_endpoints.py`) - User authentication and profile management
- **Admin Endpoints** (`admin_endpoints.py`) - Administrative functions and system management
- **Monitoring Endpoints** (`monitoring_endpoints.py`) - Real-time monitoring and metrics

## 🚀 Quick Start

### Running the Production API

```bash
# Install dependencies
pip install -r requirements/prod.txt

# Set environment variables
export SECRET_KEY="your-production-secret-key"
export DATABASE_URL="postgresql://user:pass@localhost/rental_ml"
export REDIS_URL="redis://localhost:6379"

# Run the production server
python -m src.presentation.api.production_api
```

### Running with Gunicorn (Production)

```bash
gunicorn src.presentation.api.production_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## 🔐 Authentication & Security

### Authentication Methods

1. **JWT Tokens** - For user authentication
   ```bash
   Authorization: Bearer <jwt_token>
   ```

2. **API Keys** - For service-to-service communication
   ```bash
   X-API-Key: <api_key>
   ```

### Security Features

- **Rate Limiting**: 100 requests per minute per IP (configurable)
- **CORS Protection**: Configurable allowed origins
- **Request Validation**: Comprehensive Pydantic models
- **Error Handling**: Structured error responses
- **Security Headers**: HSTS, XSS protection, etc.
- **Input Sanitization**: Automatic SQL injection prevention

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: `GET /docs`
- **ReDoc**: `GET /redoc`
- **OpenAPI Schema**: `GET /api/v2/openapi.json`

### Health Checks

- **Quick Health**: `GET /health`
- **Detailed Health**: `GET /api/v2/monitoring/health`
- **System Status**: `GET /info`

## 🤖 ML Endpoints (`/api/v2/ml`)

### Recommendations

```http
POST /api/v2/ml/recommendations
Content-Type: application/json
Authorization: Bearer <token>

{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "recommendation_type": "hybrid",
  "num_recommendations": 10,
  "include_explanations": true
}
```

### Property Search with ML Ranking

```http
POST /api/v2/ml/search
Content-Type: application/json
Authorization: Bearer <token>

{
  "query": "2 bedroom apartment downtown",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "use_ml_ranking": true,
  "include_scores": true
}
```

### Batch Predictions

```http
POST /api/v2/ml/batch-predictions
Content-Type: application/json
Authorization: Bearer <token>

{
  "user_ids": ["uuid1", "uuid2", "uuid3"],
  "model_type": "recommender",
  "batch_size": 100,
  "async_processing": true
}
```

### Model Metrics

```http
POST /api/v2/ml/model-metrics
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_type": "recommender",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z"
}
```

## 🏠 Property Endpoints (`/api/v2/properties`)

### Create Property

```http
POST /api/v2/properties
Content-Type: application/json
Authorization: Bearer <token>

{
  "title": "Beautiful 2BR Apartment",
  "description": "Spacious apartment with modern amenities",
  "price": 2500.00,
  "location": "Downtown Seattle",
  "bedrooms": 2,
  "bathrooms": 2.0,
  "property_type": "apartment",
  "amenities": ["parking", "gym", "pool"]
}
```

### Search Properties

```http
POST /api/v2/properties/search
Content-Type: application/json
Authorization: Bearer <token>

{
  "min_price": 1500,
  "max_price": 3000,
  "min_bedrooms": 2,
  "locations": ["Seattle", "Bellevue"],
  "amenities": ["parking"],
  "sort_by": "price",
  "sort_order": "asc"
}
```

### Bulk Operations

```http
POST /api/v2/properties/bulk-operation
Content-Type: application/json
Authorization: Bearer <token>

{
  "property_ids": ["uuid1", "uuid2"],
  "operation": "activate",
  "parameters": {}
}
```

### Property Analytics

```http
POST /api/v2/properties/analytics
Content-Type: application/json
Authorization: Bearer <token>

{
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "location_filter": ["Seattle"],
  "group_by": "location"
}
```

## 👤 User Endpoints (`/api/v2/users`)

### User Registration

```http
POST /api/v2/users/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "first_name": "John",
  "last_name": "Doe"
}
```

### User Login

```http
POST /api/v2/users/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "remember_me": false
}
```

### Update User Preferences

```http
PUT /api/v2/users/preferences
Content-Type: application/json
Authorization: Bearer <token>

{
  "min_price": 1500,
  "max_price": 3000,
  "preferred_locations": ["Seattle", "Bellevue"],
  "required_amenities": ["parking", "gym"]
}
```

### Record User Interaction

```http
POST /api/v2/users/interactions
Content-Type: application/json
Authorization: Bearer <token>

{
  "property_id": "123e4567-e89b-12d3-a456-426614174000",
  "interaction_type": "like",
  "duration_seconds": 120
}
```

## ⚙️ Admin Endpoints (`/api/v2/admin`)

### System Health

```http
GET /api/v2/admin/health?detailed=true
X-API-Key: <admin_api_key>
```

### Model Management

```http
POST /api/v2/admin/models/manage
Content-Type: application/json
X-API-Key: <admin_api_key>

{
  "model_name": "hybrid_recommender",
  "action": "deploy",
  "parameters": {
    "version": "v2.1.0"
  }
}
```

### User Moderation

```http
POST /api/v2/admin/users/moderate
Content-Type: application/json
X-API-Key: <admin_api_key>

{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "action": "suspend",
  "reason": "Terms of service violation",
  "duration_hours": 24
}
```

### Data Pipeline Control

```http
POST /api/v2/admin/pipelines/control
Content-Type: application/json
X-API-Key: <admin_api_key>

{
  "pipeline_name": "property_scraping",
  "action": "restart",
  "parameters": {}
}
```

## 📊 Monitoring Endpoints (`/api/v2/monitoring`)

### System Metrics

```http
GET /api/v2/monitoring/metrics/system
Authorization: Bearer <token>
```

### ML Model Metrics

```http
GET /api/v2/monitoring/metrics/ml
Authorization: Bearer <token>
```

### Business Metrics

```http
GET /api/v2/monitoring/metrics/business
Authorization: Bearer <token>
```

### Dashboard Metrics

```http
GET /api/v2/monitoring/dashboard
Authorization: Bearer <token>
```

### Real-time Metrics (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v2/monitoring/realtime');

ws.onmessage = function(event) {
    const metrics = JSON.parse(event.data);
    console.log('Real-time metrics:', metrics);
};
```

### Custom Metric Queries

```http
POST /api/v2/monitoring/metrics/query
Content-Type: application/json
Authorization: Bearer <token>

{
  "metric_name": "cpu_usage_percent",
  "time_range": "24h",
  "aggregation": "avg",
  "filters": {"environment": "production"}
}
```

### Alert Management

```http
POST /api/v2/monitoring/alerts/rules
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "high_error_rate",
  "metric": "error_rate_percent",
  "condition": ">",
  "threshold": 5.0,
  "severity": "critical",
  "notification_channels": ["email", "slack"]
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Database
DATABASE_URL=postgresql://user:pass@localhost/rental_ml
REDIS_URL=redis://localhost:6379

# API Configuration
PORT=8000
WORKERS=4
LOG_LEVEL=info
ENVIRONMENT=production

# CORS
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
ALLOWED_HOSTS=yourdomain.com,*.yourdomain.com

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements/ requirements/
RUN pip install -r requirements/prod.txt

COPY src/ src/
EXPOSE 8000

CMD ["gunicorn", "src.presentation.api.production_api:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

## 📈 Performance Considerations

### Scalability

- **Async Operations**: All endpoints use async/await for optimal performance
- **Database Connection Pooling**: Configured for high concurrency
- **Redis Caching**: Intelligent caching for frequently accessed data
- **Background Tasks**: Non-blocking operations for heavy workloads

### Monitoring

- **Request Tracing**: Comprehensive logging with request IDs
- **Performance Metrics**: Real-time monitoring of response times
- **Error Tracking**: Automatic error reporting and alerting
- **Resource Monitoring**: CPU, memory, and database performance

### Security

- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Protection against abuse
- **Authentication**: Multi-factor authentication support
- **Audit Logging**: Complete audit trail for all operations

## 🐛 Error Handling

### Standard Error Response Format

```json
{
  "error": "Detailed error message",
  "status_code": 400,
  "path": "/api/v2/properties",
  "method": "POST",
  "timestamp": "2023-12-01T10:30:00Z",
  "request_id": "req_123456789"
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

## 🚀 Deployment

### Production Checklist

- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Redis cluster configured
- [ ] SSL certificates installed
- [ ] Load balancer configured
- [ ] Monitoring alerts configured
- [ ] Backup procedures tested
- [ ] Security audit completed

### Health Checks

The API provides multiple health check endpoints for load balancers:

- `/health` - Basic health check
- `/api/v2/monitoring/health` - Detailed component health
- `/info` - System information

### Logging

Structured JSON logging is configured for production environments:

```json
{
  "timestamp": "2023-12-01T10:30:00Z",
  "level": "INFO",
  "message": "Request processed successfully",
  "request_id": "req_123456789",
  "user_id": "user_123",
  "endpoint": "/api/v2/properties",
  "response_time_ms": 245.5
}
```

## 📞 Support

For technical support or questions about the API:

1. Check the interactive documentation at `/docs`
2. Review the health check endpoints for system status
3. Check the monitoring dashboard for system metrics
4. Review error logs for detailed error information

## 🔄 API Versioning

The API uses URL versioning (`/api/v2/`) to ensure backward compatibility. When breaking changes are introduced, a new version will be created while maintaining support for previous versions.

Current version: **v2.0.0**

## 📝 License

This API is part of the Rental ML System and is subject to the project's license terms.