#!/bin/bash

# =============================================================================
# Docker Entrypoint Script for Rental ML System
# =============================================================================
# This script handles application initialization and startup in Docker containers

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [ "${LOG_LEVEL:-INFO}" = "DEBUG" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Configuration
APP_ENV="${APP_ENV:-production}"
DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
MAX_RETRIES="${MAX_RETRIES:-30}"
RETRY_INTERVAL="${RETRY_INTERVAL:-2}"

log_info "Starting Rental ML System..."
log_info "Environment: ${APP_ENV}"
log_info "Database: ${DB_HOST}:${DB_PORT}"
log_info "Redis: ${REDIS_HOST}:${REDIS_PORT}"

# =============================================================================
# Health Check Functions
# =============================================================================

check_database() {
    log_debug "Checking database connectivity..."
    python3 -c "
import asyncio
import asyncpg
import os
import sys

async def check_db():
    try:
        db_url = f'postgresql://{os.getenv(\"DB_USERNAME\", \"postgres\")}:{os.getenv(\"DB_PASSWORD\", \"\")}@{os.getenv(\"DB_HOST\", \"postgres\")}:{os.getenv(\"DB_PORT\", \"5432\")}/{os.getenv(\"DB_NAME\", \"rental_ml\")}'
        conn = await asyncpg.connect(db_url, server_settings={'application_name': 'health_check'})
        await conn.fetchval('SELECT 1')
        await conn.close()
        print('Database connection successful')
        return True
    except Exception as e:
        print(f'Database connection failed: {e}')
        return False

result = asyncio.run(check_db())
sys.exit(0 if result else 1)
"
}

check_redis() {
    log_debug "Checking Redis connectivity..."
    python3 -c "
import redis
import os
import sys

try:
    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', '6379')),
        password=os.getenv('REDIS_PASSWORD', None),
        db=int(os.getenv('REDIS_DB', '0')),
        socket_timeout=5,
        socket_connect_timeout=5
    )
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    sys.exit(1)
"
}

wait_for_service() {
    local service_name=$1
    local check_function=$2
    local retries=0
    
    log_info "Waiting for ${service_name}..."
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if $check_function; then
            log_info "${service_name} is ready!"
            return 0
        fi
        
        retries=$((retries + 1))
        log_warn "${service_name} not ready. Retrying in ${RETRY_INTERVAL}s... (${retries}/${MAX_RETRIES})"
        sleep $RETRY_INTERVAL
    done
    
    log_error "Failed to connect to ${service_name} after ${MAX_RETRIES} attempts"
    return 1
}

# =============================================================================
# Application Initialization
# =============================================================================

initialize_database() {
    log_info "Initializing database..."
    
    # Run database initialization script
    if [ -f "/app/scripts/init_database.py" ]; then
        log_debug "Running database initialization script..."
        python3 /app/scripts/init_database.py
        if [ $? -eq 0 ]; then
            log_info "Database initialization completed successfully"
        else
            log_error "Database initialization failed"
            return 1
        fi
    else
        log_warn "Database initialization script not found"
    fi
}

setup_directories() {
    log_info "Setting up application directories..."
    
    # Create necessary directories
    mkdir -p /app/logs
    mkdir -p /app/data
    mkdir -p /app/models
    mkdir -p /app/tmp
    
    # Set proper permissions
    chmod -R 755 /app/logs /app/data /app/models /app/tmp
    
    log_debug "Application directories created and configured"
}

check_environment() {
    log_info "Validating environment configuration..."
    
    # Required environment variables
    required_vars=(
        "DB_HOST"
        "DB_NAME"
        "DB_USERNAME"
        "DB_PASSWORD"
        "REDIS_HOST"
        "SECRET_KEY"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    log_debug "Environment validation passed"
}

load_ml_models() {
    log_info "Loading ML models..."
    
    # Check if models directory exists and has models
    if [ -d "/app/models" ] && [ "$(ls -A /app/models)" ]; then
        log_info "ML models found in /app/models"
    else
        log_warn "No ML models found. The application will download/train models on first use."
    fi
    
    # Optionally pre-load models
    if [ "${PRELOAD_MODELS:-false}" = "true" ]; then
        log_info "Pre-loading ML models..."
        python3 -c "
import sys
sys.path.append('/app')
try:
    from src.infrastructure.ml.models.hybrid_recommender import HybridRecommender
    # Initialize models (this will load them into memory)
    recommender = HybridRecommender()
    print('ML models pre-loaded successfully')
except Exception as e:
    print(f'Failed to pre-load ML models: {e}')
    # Don't fail the startup, just warn
"
    fi
}

run_health_check() {
    log_info "Running application health check..."
    
    # Start the application in background for health check
    python3 -m uvicorn src.application.api.main:app --host 0.0.0.0 --port 8000 &
    APP_PID=$!
    
    # Wait for application to start
    sleep 10
    
    # Check health endpoint
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_info "Application health check passed"
        kill $APP_PID 2>/dev/null || true
        wait $APP_PID 2>/dev/null || true
        return 0
    else
        log_error "Application health check failed"
        kill $APP_PID 2>/dev/null || true
        wait $APP_PID 2>/dev/null || true
        return 1
    fi
}

# =============================================================================
# Signal Handlers
# =============================================================================

cleanup() {
    log_info "Received shutdown signal. Cleaning up..."
    
    # Kill any background processes
    if [ ! -z "$APP_PID" ]; then
        kill $APP_PID 2>/dev/null || true
        wait $APP_PID 2>/dev/null || true
    fi
    
    log_info "Cleanup completed"
    exit 0
}

# Setup signal handlers
trap cleanup SIGTERM SIGINT

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_info "Executing Docker entrypoint..."
    
    # Validate environment
    if ! check_environment; then
        log_error "Environment validation failed"
        exit 1
    fi
    
    # Setup directories
    setup_directories
    
    # Wait for external services
    if ! wait_for_service "PostgreSQL" check_database; then
        exit 1
    fi
    
    if ! wait_for_service "Redis" check_redis; then
        exit 1
    fi
    
    # Initialize application
    if [ "${SKIP_DB_INIT:-false}" != "true" ]; then
        if ! initialize_database; then
            log_error "Database initialization failed"
            exit 1
        fi
    else
        log_info "Skipping database initialization (SKIP_DB_INIT=true)"
    fi
    
    # Load ML models
    load_ml_models
    
    # Run health check if not in development
    if [ "$APP_ENV" != "development" ] && [ "${SKIP_HEALTH_CHECK:-false}" != "true" ]; then
        if ! run_health_check; then
            log_error "Initial health check failed"
            exit 1
        fi
    fi
    
    log_info "Initialization completed successfully!"
    
    # Execute the main command
    log_info "Starting application with command: $*"
    exec "$@"
}

# =============================================================================
# Script Entry Point
# =============================================================================

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
    log_error "No command provided"
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 gunicorn src.application.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"
    exit 1
fi

# Run main function with all arguments
main "$@"