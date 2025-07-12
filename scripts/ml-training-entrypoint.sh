#!/bin/bash
set -e

# ML Training Service Entrypoint Script
# Handles initialization, GPU detection, and training environment setup

echo "=== ML Training Service Initialization ==="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "Checking GPU availability..."
        nvidia-smi
        export CUDA_AVAILABLE=true
        export TF_GPU_ALLOCATOR=cuda_malloc_async
    else
        log "No GPU detected, using CPU-only mode"
        export CUDA_AVAILABLE=false
        export CUDA_VISIBLE_DEVICES=""
    fi
}

# Function to setup ML environment
setup_ml_environment() {
    log "Setting up ML training environment..."
    
    # Create necessary directories
    mkdir -p /app/models/checkpoints
    mkdir -p /app/models/saved_models
    mkdir -p /app/artifacts/experiments
    mkdir -p /app/data/training
    mkdir -p /app/data/validation
    mkdir -p /app/tensorboard
    
    # Set ML-specific environment variables
    export PYTHONPATH="/app/src:$PYTHONPATH"
    export TF_CPP_MIN_LOG_LEVEL=2
    export MLFLOW_TRACKING_URI="file:///app/artifacts/mlruns"
    export TENSORBOARD_LOG_DIR="/app/tensorboard"
    
    # Configure TensorFlow memory growth for GPU
    if [ "$CUDA_AVAILABLE" = "true" ]; then
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export TF_GPU_MEMORY_ALLOW_GROWTH=true
    fi
    
    log "ML environment setup completed"
}

# Function to validate Python dependencies
validate_dependencies() {
    log "Validating ML dependencies..."
    
    python -c "
import sys
import importlib

required_packages = [
    'tensorflow', 'torch', 'sklearn', 'pandas', 'numpy',
    'mlflow', 'optuna', 'xgboost', 'lightgbm'
]

missing_packages = []
for package in required_packages:
    try:
        importlib.import_module(package)
        print(f'✓ {package} imported successfully')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package} is missing')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All required packages are available')
"
    
    if [ $? -eq 0 ]; then
        log "All ML dependencies validated successfully"
    else
        log "ERROR: Missing required ML dependencies"
        exit 1
    fi
}

# Function to initialize MLflow
setup_mlflow() {
    log "Setting up MLflow tracking..."
    
    # Initialize MLflow tracking directory
    if [ ! -d "/app/artifacts/mlruns" ]; then
        mkdir -p /app/artifacts/mlruns
        log "Created MLflow tracking directory"
    fi
    
    # Start MLflow server in background if not running
    if ! pgrep -f "mlflow server" > /dev/null; then
        log "Starting MLflow tracking server..."
        mlflow server \
            --backend-store-uri file:///app/artifacts/mlruns \
            --default-artifact-root file:///app/artifacts/mlflow-artifacts \
            --host 0.0.0.0 \
            --port 5000 \
            --workers 2 &
        
        # Wait for MLflow to start
        sleep 10
        log "MLflow tracking server started on port 5000"
    fi
}

# Function to setup Jupyter if needed
setup_jupyter() {
    if [ "$ENABLE_JUPYTER" = "true" ]; then
        log "Setting up Jupyter environment..."
        
        # Generate Jupyter config
        jupyter notebook --generate-config
        
        # Set Jupyter password if provided
        if [ -n "$JUPYTER_PASSWORD" ]; then
            python -c "
from notebook.auth import passwd
import os
password_hash = passwd('$JUPYTER_PASSWORD')
config_file = os.path.expanduser('~/.jupyter/jupyter_notebook_config.py')
with open(config_file, 'a') as f:
    f.write(f\"\\nc.NotebookApp.password = '{password_hash}'\\n\")
"
            log "Jupyter password configured"
        fi
        
        # Start Jupyter in background
        jupyter notebook \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --notebook-dir=/app/notebooks &
        
        log "Jupyter notebook started on port 8888"
    fi
}

# Function to run database migrations
run_migrations() {
    log "Checking database connectivity..."
    
    python -c "
import sys
import os
sys.path.append('/app/src')

try:
    from infrastructure.data.config import get_database_url
    from sqlalchemy import create_engine, text
    
    engine = create_engine(get_database_url())
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1'))
        print('Database connection successful')
        
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log "Database connectivity verified"
        
        # Run any pending migrations
        if [ -f "/app/migrations/run_migrations.py" ]; then
            log "Running database migrations..."
            python /app/migrations/run_migrations.py
        fi
    else
        log "ERROR: Database connectivity check failed"
        exit 1
    fi
}

# Function to perform health check
health_check() {
    log "Performing initial health check..."
    
    # Check if training scripts exist
    if [ ! -f "/app/src/application/ml_training/production_training_pipeline.py" ]; then
        log "ERROR: Training pipeline script not found"
        exit 1
    fi
    
    # Test Python imports
    python -c "
import sys
sys.path.append('/app/src')
from application.ml_training.production_training_pipeline import *
print('Training pipeline imports successful')
"
    
    if [ $? -eq 0 ]; then
        log "Health check passed"
    else
        log "ERROR: Health check failed"
        exit 1
    fi
}

# Main initialization sequence
main() {
    log "Starting ML Training Service..."
    
    # Run initialization steps
    check_gpu
    setup_ml_environment
    validate_dependencies
    run_migrations
    setup_mlflow
    setup_jupyter
    health_check
    
    log "ML Training Service initialization completed"
    log "Executing command: $@"
    
    # Execute the main command
    exec "$@"
}

# Handle shutdown signals gracefully
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Stop background processes
    pkill -f "mlflow server" || true
    pkill -f "jupyter notebook" || true
    
    log "Cleanup completed"
    exit 0
}

# Set signal handlers
trap cleanup SIGTERM SIGINT

# Run main function
main "$@"