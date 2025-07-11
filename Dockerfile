# Multi-stage Dockerfile for Rental ML System
# Optimized for production with separate build and runtime stages

# ================================
# Base Python stage
# ================================
FROM python:3.11-slim as python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    libc6-dev \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Dependencies stage
# ================================
FROM python-base as deps

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Copy dependency files
COPY requirements/ ./requirements/
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/base.txt && \
    pip install --no-cache-dir -r requirements/prod.txt

# ================================
# Development stage
# ================================
FROM deps as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements/dev.txt

# Create non-root user for development
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Expose development ports
EXPOSE 8000 5678

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command
CMD ["python", "-m", "uvicorn", "src.application.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================
# Production build stage
# ================================
FROM deps as builder

# Copy application source
COPY . .

# Build application wheel
RUN pip install build && \
    python -m build --wheel

# ================================
# Production runtime stage
# ================================
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Add startup script
COPY --chown=appuser:appuser scripts/docker-entrypoint.sh /docker-entrypoint.sh
USER root
RUN chmod +x /docker-entrypoint.sh
USER appuser

# Default command
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["gunicorn", "src.application.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# ================================
# ML Training stage (optional)
# ================================
FROM deps as ml-training

# Install additional ML training dependencies
RUN pip install --no-cache-dir jupyter notebook mlflow

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser
RUN chown -R mluser:mluser /app
USER mluser

# Copy application code
COPY --chown=mluser:mluser . .

# Expose Jupyter port
EXPOSE 8888

# Training command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]