# Docker Configuration for Rental ML System

This document provides comprehensive information about the Docker setup for the Rental ML System, including production and development configurations.

## Overview

The Docker configuration includes:

- **Multi-stage Dockerfile** with separate builds for development, production, and ML training
- **Production docker-compose.yml** with full stack including PostgreSQL, Redis, Nginx, and monitoring
- **Development docker-compose.dev.yml** with hot reload, debugging tools, and development services
- **Environment configuration** with .env.example and .env.docker templates
- **Startup scripts** for proper application initialization
- **Configuration files** for all services (Nginx, Redis, Prometheus, etc.)

## Quick Start

### Development Environment

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your specific values
   ```

2. **Start development stack:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Access services:**
   - **API:** http://localhost:8000
   - **API Documentation:** http://localhost:8000/docs
   - **Jupyter Notebooks:** http://localhost:8888
   - **Flower (Celery Monitor):** http://localhost:5555
   - **pgAdmin:** http://localhost:8082
   - **Redis Commander:** http://localhost:8081
   - **MailHog:** http://localhost:8025

### Production Environment

1. **Set up environment:**
   ```bash
   cp .env.docker .env
   # Configure production values in .env
   ```

2. **Start production stack:**
   ```bash
   docker-compose up -d
   ```

3. **Access services:**
   - **Application:** http://localhost (via Nginx)
   - **Prometheus:** http://localhost:9090

## Docker Images

### Main Application Image

The Dockerfile uses multi-stage builds:

- **python-base:** Base Python 3.11 with system dependencies
- **deps:** Python dependencies installation
- **development:** Development target with dev tools
- **builder:** Production build stage
- **production:** Optimized production runtime
- **ml-training:** ML training environment with Jupyter

### Build Targets

```bash
# Development build
docker build --target development -t rental-ml:dev .

# Production build  
docker build --target production -t rental-ml:prod .

# ML training build
docker build --target ml-training -t rental-ml:ml .
```

## Services Architecture

### Production Stack (docker-compose.yml)

- **postgres:** PostgreSQL 15 database with optimized configuration
- **redis:** Redis 7 cache with persistence
- **app:** Main FastAPI application (4 workers)
- **worker:** Celery background task worker
- **scheduler:** Celery beat scheduler
- **nginx:** Reverse proxy with rate limiting
- **prometheus:** Metrics collection

### Development Stack (docker-compose.dev.yml)

All production services plus:
- **jupyter:** Jupyter Lab for ML development
- **flower:** Celery task monitoring
- **redis-commander:** Redis GUI
- **pgadmin:** PostgreSQL GUI
- **mailhog:** Email testing

## Environment Configuration

### Key Environment Variables

#### Database
```bash
DB_HOST=postgres
DB_PORT=5432
DB_NAME=rental_ml
DB_USERNAME=postgres
DB_PASSWORD=your_secure_password
```

#### Redis
```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
```

#### Application
```bash
APP_ENV=production
LOG_LEVEL=INFO
SECRET_KEY=your_secret_key
```

#### ML Configuration
```bash
ML_MODEL_PATH=/app/models
ML_BATCH_SIZE=32
ML_CACHE_TTL=3600
```

### Environment Files

- **`.env.example`:** Template with all available variables
- **`.env.docker`:** Docker-specific defaults
- **`.env`:** Your local configuration (not tracked)

## Volumes and Data Persistence

### Persistent Volumes

- **postgres_data:** Database files
- **redis_data:** Redis persistence
- **ml_models:** Trained ML models
- **app_logs:** Application logs
- **app_data:** Application data

### Development Volumes

Additional volumes for development:
- **Source code mounting** for hot reload
- **Jupyter notebooks** for ML development
- **Development tools** configuration

## Health Checks

All services include health checks:

- **Database:** `pg_isready` check
- **Redis:** `redis-cli ping`
- **Application:** `/health` endpoint
- **Nginx:** HTTP status check

## Networking

### Production Network
- **rental-ml-network:** Bridge network for service communication
- Services communicate via service names (e.g., `postgres`, `redis`)

### Port Mapping

#### Production
- **80:** Nginx (HTTP)
- **443:** Nginx (HTTPS)
- **5432:** PostgreSQL (optional, for external access)
- **6379:** Redis (optional, for external access)
- **9090:** Prometheus

#### Development
- **8000:** FastAPI application
- **8888:** Jupyter Lab
- **5555:** Flower
- **8081:** Redis Commander
- **8082:** pgAdmin
- **8025:** MailHog

## Security

### Production Security Features

- **Non-root users** in all containers
- **Security headers** via Nginx
- **Rate limiting** for API endpoints
- **Secret management** via environment variables
- **Network isolation** via Docker networks

### Security Checklist

- [ ] Change default passwords
- [ ] Set strong SECRET_KEY
- [ ] Configure CORS properly
- [ ] Use HTTPS in production
- [ ] Regularly update base images
- [ ] Monitor security logs

## Monitoring and Logging

### Logging

- **Structured JSON logging** for all services
- **Log rotation** configured (10MB max, 3 files)
- **Centralized logs** in `/app/logs` volume

### Monitoring

- **Prometheus** for metrics collection
- **Health checks** for all services
- **Application metrics** via FastAPI middleware

## Development Workflow

### Hot Reload Development

1. Start development stack:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

2. Code changes are automatically reflected in the running container

3. Access development tools:
   - Jupyter for ML experimentation
   - pgAdmin for database management
   - Flower for task monitoring

### Debugging

1. **Enable debugging:**
   ```bash
   export DEBUG_PORT=5678
   export ENABLE_REMOTE_DEBUGGING=true
   ```

2. **Attach debugger** to port 5678

3. **View logs:**
   ```bash
   docker-compose logs -f app-dev
   ```

## Deployment

### Local Deployment

```bash
# Production stack
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Cloud Deployment

The Docker configuration is cloud-ready:

- **Environment variables** for configuration
- **Health checks** for orchestration
- **Persistent volumes** for data
- **Horizontal scaling** support

## Troubleshooting

### Common Issues

1. **Database connection failed:**
   ```bash
   # Check database status
   docker-compose ps postgres
   # Check logs
   docker-compose logs postgres
   ```

2. **Application not starting:**
   ```bash
   # Check application logs
   docker-compose logs app
   # Verify environment variables
   docker-compose exec app env | grep -E '^(DB_|REDIS_)'
   ```

3. **Redis connection issues:**
   ```bash
   # Test Redis connectivity
   docker-compose exec redis redis-cli ping
   ```

### Debug Commands

```bash
# Enter application container
docker-compose exec app bash

# Check application health
docker-compose exec app curl http://localhost:8000/health

# View database
docker-compose exec postgres psql -U postgres -d rental_ml

# Monitor Redis
docker-compose exec redis redis-cli monitor
```

## Maintenance

### Database Backups

```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres rental_ml > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres rental_ml < backup.sql
```

### Image Updates

```bash
# Pull latest images
docker-compose pull

# Rebuild application image
docker-compose build app

# Restart with new images
docker-compose up -d
```

### Cleanup

```bash
# Remove stopped containers
docker-compose down

# Remove volumes (⚠️ This deletes data!)
docker-compose down -v

# Clean build cache
docker system prune -f
```

## Performance Tuning

### Resource Limits

Configure in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '1.0'
```

### Database Performance

- **Connection pooling** configured
- **Optimized PostgreSQL settings** in init script
- **Indexes** for common queries

### Caching

- **Redis caching** for ML models
- **Application-level caching** for frequent queries
- **Nginx caching** for static content

## Best Practices

1. **Use specific image tags** in production
2. **Monitor resource usage** regularly
3. **Backup data** before major updates
4. **Test configuration changes** in development first
5. **Keep secrets secure** and rotate regularly
6. **Monitor logs** for errors and security issues
7. **Update dependencies** regularly for security patches

## Support

For issues with the Docker configuration:

1. Check this documentation
2. Review logs: `docker-compose logs <service>`
3. Verify environment configuration
4. Test individual services
5. Check Docker and Docker Compose versions

## Version Compatibility

- **Docker:** 20.10+
- **Docker Compose:** 2.0+
- **Python:** 3.11
- **PostgreSQL:** 15
- **Redis:** 7