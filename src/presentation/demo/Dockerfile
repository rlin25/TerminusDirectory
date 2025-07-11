# Dockerfile for Rental ML System Streamlit Demo
# Optimized for production deployment of the demo application

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        software-properties-common \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements/base.txt \
    && pip install -r src/presentation/demo/requirements-demo.txt

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
USER user

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set environment variables for demo
ENV DEMO_PROPERTY_COUNT=100 \
    DEMO_USER_COUNT=50 \
    RANDOM_SEED=42

# Default command to run the demo
CMD ["streamlit", "run", "src/presentation/demo/app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]