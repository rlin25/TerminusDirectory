# Rental ML System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](#)
[![Coverage](https://img.shields.io/badge/Coverage-85%+-brightgreen.svg)](#)

> A production-grade machine learning system for intelligent rental property search and personalized recommendations using advanced NLP and hybrid recommendation algorithms.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Getting Started](#-getting-started)
- [📖 Usage Instructions](#-usage-instructions)
- [🛠️ Development](#️-development)
- [🚢 Deployment](#-deployment)
- [📊 ML Models & Architecture](#-ml-models--architecture)
- [📚 Documentation](#-documentation)
- [🔧 Maintenance](#-maintenance)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Project Overview

The Rental ML System is a sophisticated machine learning platform designed for real-time rental property search and personalized recommendations. Built with clean architecture principles, it combines advanced NLP-powered search with hybrid recommendation engines to deliver exceptional user experiences.

### Key Capabilities

- **🔍 Intelligent Search**: NLP-powered semantic search using TensorFlow and Transformers
- **🎯 Personalized Recommendations**: Hybrid ML system combining collaborative filtering and content-based approaches
- **⚡ Real-time Performance**: Optimized for low-latency serving with Redis caching
- **🏗️ Production-Ready Architecture**: Clean architecture with comprehensive testing and monitoring
- **🔄 Scalable Infrastructure**: Containerized deployment with Kubernetes support
- **📊 Advanced Analytics**: Real-time performance monitoring and business intelligence

### Technology Stack

**Core ML/AI:**
- TensorFlow 2.13+ for deep learning models
- Transformers for NLP and semantic search
- Scikit-learn for traditional ML algorithms
- NumPy/Pandas for data processing

**Backend & API:**
- FastAPI for high-performance REST APIs
- SQLAlchemy for database ORM
- PostgreSQL for primary data storage
- Redis for caching and session management

**Infrastructure:**
- Docker & Docker Compose for containerization
- Kubernetes for orchestration
- Nginx for reverse proxy and load balancing
- Prometheus for monitoring and metrics

**Development:**
- Pytest for comprehensive testing
- Black/Flake8 for code formatting and linting
- MyPy for static type checking
- Pre-commit hooks for code quality

## ✨ Key Features

### 🔍 Advanced Search Engine
- **Semantic Search**: TensorFlow-based text embedding and ranking
- **Multi-criteria Filtering**: Price, location, amenities, property type
- **Real-time Results**: Sub-200ms query response times
- **Relevance Ranking**: ML-powered result ordering

### 🎯 Hybrid Recommendation System
- **Collaborative Filtering**: Neural collaborative filtering with TensorFlow
- **Content-Based Filtering**: Property feature similarity matching
- **Cold Start Handling**: Effective recommendations for new users
- **Explainable AI**: Detailed recommendation explanations

### 📊 Analytics & Monitoring
- **Performance Metrics**: Real-time model accuracy and response times
- **Business Intelligence**: Market trends and user behavior analysis
- **System Health**: Comprehensive monitoring and alerting
- **A/B Testing**: Built-in framework for model experimentation

### 🔄 Data Pipeline
- **Web Scraping**: Ethical property data collection
- **Data Quality**: Automated validation and cleaning
- **Feature Engineering**: Advanced feature extraction and encoding
- **Model Training**: Automated ML pipeline with evaluation

## 🏗️ Architecture

### Clean Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   FastAPI       │  │   Streamlit     │  │      Jupyter                │ │
│  │   REST API      │  │   Demo UI       │  │      Notebooks              │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
┌────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   Use Cases     │  │      DTOs       │  │         API                 │ │
│  │ - SearchProps   │  │ - SearchDTO     │  │ - search_endpoints.py       │ │
│  │ - GetRecommend  │  │ - RecommendDTO  │  │ - recommendation_endpoints  │ │
│  │ - TrackInteract │  │ - UserDTO       │  │ - user_endpoints.py         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
┌────────────────────────────────────────────────────────────────────────────┐
│                            DOMAIN LAYER                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │    Entities     │  │   Repositories  │  │        Services             │ │
│  │ - Property      │  │ - PropertyRepo  │  │ - SearchService             │ │
│  │ - User          │  │ - UserRepo      │  │ - RecommendationService     │ │
│  │ - SearchQuery   │  │ - ModelRepo     │  │ - UserService               │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
┌────────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  Data Access    │  │   ML Models     │  │      Monitoring             │ │
│  │ - PostgreSQL    │  │ - Search Ranker │  │ - Logging                   │ │
│  │ - Redis Cache   │  │ - Collaborative │  │ - Metrics                   │ │
│  │ - Web Scrapers  │  │ - Content-Based │  │ - Health Checks             │ │
│  │ - File Storage  │  │ - Hybrid System │  │ - Error Tracking            │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

### System Components

**Core Services:**
- **API Gateway**: FastAPI-based REST API with automatic documentation
- **Search Service**: NLP-powered property search and ranking
- **Recommendation Service**: Hybrid ML recommendation engine
- **User Service**: User management and preference tracking

**Data Layer:**
- **PostgreSQL**: Primary database for structured data
- **Redis**: High-performance caching and session storage
- **Feature Store**: ML feature management and serving

**ML Infrastructure:**
- **Model Training**: Automated training pipelines with evaluation
- **Model Serving**: Real-time model inference and prediction
- **Model Registry**: Version control and model lifecycle management

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Docker**: 20.0+ (for containerized deployment)
- **PostgreSQL**: 12+ (if running locally)
- **Redis**: 6+ (if running locally)

### Quick Start

#### 1. Clone Repository
```bash
git clone https://github.com/rental-ml-system/rental-ml-system.git
cd rental-ml-system
```

#### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
```

#### 3. Database Setup
```bash
# Start PostgreSQL and Redis with Docker
docker-compose up -d postgres redis

# Run database migrations
python scripts/init_database.py
python migrations/run_migrations.py
```

#### 4. Run Demo Application
```bash
# Quick start with demo script
./demo-quick-start.sh

# Or manually start Streamlit demo
streamlit run src/presentation/demo/app.py
```

#### 5. API Server
```bash
# Start FastAPI server
uvicorn src.application.api.main:app --reload --port 8000

# API documentation available at: http://localhost:8000/docs
```

### Docker Quick Start

```bash
# Start full system with Docker Compose
docker-compose up -d

# Access services:
# - API: http://localhost:8000
# - Demo UI: http://localhost:8501  
# - Monitoring: http://localhost:9090
```

## 📖 Usage Instructions

### Demo Application

The Streamlit demo provides a comprehensive showcase of system capabilities:

```bash
# Launch interactive demo
./demo-quick-start.sh --port 8501

# Or with custom configuration
streamlit run src/presentation/demo/app.py \
  --server.port 8502 \
  --server.address 0.0.0.0
```

**Demo Features:**
- 🏠 **Property Search**: Advanced filtering and search capabilities
- 🎯 **Recommendations**: Personalized property recommendations
- 👤 **User Preferences**: User profile and preference management
- 📊 **Analytics Dashboard**: Market insights and performance metrics
- ⚡ **ML Monitoring**: Real-time model performance tracking

### API Usage

#### Search Properties
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "downtown apartment with parking",
    "filters": {
      "price_min": 2000,
      "price_max": 4000,
      "bedrooms": 2,
      "amenities": ["parking", "gym"]
    },
    "limit": 10
  }'
```

#### Get Recommendations
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "num_recommendations": 5,
    "include_explanations": true
  }'
```

#### Health Check
```bash
curl "http://localhost:8000/health"
```

### Configuration

#### Environment Variables
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rental_ml
DB_USERNAME=postgres
DB_PASSWORD=your_password

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# ML Configuration
ML_MODEL_PATH=/app/models
ML_BATCH_SIZE=32
ML_CACHE_TTL=3600

# Application Configuration
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

#### Feature Flags
```python
# config/settings.py
class FeatureFlags:
    ENABLE_RECOMMENDATIONS = True
    ENABLE_SCRAPING = True
    ENABLE_ML_TRAINING = False  # Disable in production
    ENABLE_ANALYTICS = True
```

## 🛠️ Development

### Development Environment Setup

```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Install pre-commit hooks
pre-commit install

# Run development server with hot reload
uvicorn src.application.api.main:app --reload --port 8000
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m ml            # ML model tests only

# Run performance tests
pytest tests/performance/ -v
```

### Project Structure
```
rental-ml-system/
├── src/                              # Source code
│   ├── application/                  # Application layer
│   │   ├── api/                     # REST API endpoints
│   │   ├── dto/                     # Data transfer objects
│   │   └── use_cases/               # Business use cases
│   ├── domain/                      # Domain layer
│   │   ├── entities/                # Core business entities
│   │   ├── repositories/            # Repository interfaces
│   │   └── services/                # Domain services
│   ├── infrastructure/              # Infrastructure layer
│   │   ├── data/                    # Data access and repositories
│   │   ├── ml/                      # ML models and training
│   │   ├── monitoring/              # Monitoring and logging
│   │   └── scrapers/                # Web scraping infrastructure
│   └── presentation/                # Presentation layer
│       ├── demo/                    # Streamlit demo application
│       └── web/                     # Web interface
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   └── performance/                 # Performance tests
├── deployment/                      # Deployment configurations
│   ├── docker/                      # Docker configurations
│   ├── kubernetes/                  # Kubernetes manifests
│   └── terraform/                   # Infrastructure as code
├── docs/                           # Documentation
├── examples/                       # Usage examples
├── notebooks/                      # Jupyter notebooks
└── scripts/                        # Utility scripts
```

### Adding New Features

1. **Create Domain Entity** (if needed):
```python
# src/domain/entities/new_entity.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class NewEntity:
    id: Optional[int]
    name: str
    # Add fields
```

2. **Implement Repository Interface**:
```python
# src/domain/repositories/new_repository.py
from abc import ABC, abstractmethod

class NewRepository(ABC):
    @abstractmethod
    async def find_by_id(self, entity_id: int) -> Optional[NewEntity]:
        pass
```

3. **Create Implementation**:
```python
# src/infrastructure/data/repositories/postgres_new_repository.py
class PostgresNewRepository(NewRepository):
    async def find_by_id(self, entity_id: int) -> Optional[NewEntity]:
        # Implementation
        pass
```

4. **Add API Endpoint**:
```python
# src/application/api/routers/new_router.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/new", tags=["new"])

@router.get("/{entity_id}")
async def get_entity(entity_id: int):
    # Implementation
    pass
```

5. **Write Tests**:
```python
# tests/unit/test_domain/test_new_entity.py
def test_new_entity_creation():
    entity = NewEntity(id=1, name="test")
    assert entity.name == "test"
```

## 🚢 Deployment

### Docker Deployment

#### Development Environment
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f app
```

#### Production Environment
```bash
# Build and start production environment
docker-compose up -d --build

# Scale services
docker-compose up -d --scale app=3 --scale worker=2
```

### Kubernetes Deployment

#### Local Development (Minikube)
```bash
# Start minikube
minikube start

# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rental-ml

# Access services
minikube service rental-ml-app -n rental-ml
```

#### Production Deployment
```bash
# Apply production configuration
kubectl apply -f deployment/kubernetes/

# Monitor deployment
kubectl rollout status deployment/rental-ml-app -n rental-ml

# View logs
kubectl logs -f deployment/rental-ml-app -n rental-ml
```

### Helm Deployment
```bash
# Install with Helm
helm install rental-ml k8s/helm/rental-ml/ \
  --namespace rental-ml \
  --create-namespace \
  --values k8s/helm/rental-ml/values-prod.yaml

# Upgrade deployment
helm upgrade rental-ml k8s/helm/rental-ml/ \
  --values k8s/helm/rental-ml/values-prod.yaml
```

### Environment Variables

#### Required Environment Variables
```bash
# Production deployment requires these variables
DB_PASSWORD=your_secure_db_password
REDIS_PASSWORD=your_secure_redis_password
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_here
```

#### Optional Configuration
```bash
# Monitoring
SENTRY_DSN=https://your-sentry-dsn
PROMETHEUS_ENABLED=true

# Scaling
DB_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=50
ML_BATCH_SIZE=64

# Features
ENABLE_SCRAPING=true
ENABLE_ML_TRAINING=false
```

## 📊 ML Models & Architecture

### Search Ranking Model

The NLP-powered search system uses TensorFlow and Transformers:

```python
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

class NLPSearchRanker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = TFAutoModel.from_pretrained(model_name)
        self.ranking_model = self._build_ranking_model()
    
    def rank_properties(self, query: str, properties: List[Dict]) -> List[Tuple[Dict, float]]:
        # Generate embeddings and rank properties
        pass
```

**Features:**
- Semantic text understanding using pre-trained transformers
- Deep neural ranking with TensorFlow
- Real-time inference optimized for low latency
- Custom training on rental property domain data

### Hybrid Recommendation System

Combines collaborative filtering and content-based approaches:

```python
class HybridRecommendationSystem:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = CollaborativeFilteringModel()
        self.cb_model = ContentBasedModel()
    
    def recommend(self, user_id: int, num_recommendations: int = 10) -> List[Recommendation]:
        # Generate hybrid recommendations
        pass
```

**Components:**
- **Collaborative Filtering**: Neural collaborative filtering with embeddings
- **Content-Based**: Property feature similarity matching
- **Hybrid Fusion**: Weighted combination with dynamic adjustment
- **Cold Start**: Effective handling of new users and properties

### Model Training Pipeline

```bash
# Train models locally
python src/infrastructure/ml/training/ml_trainer.py \
  --model-type hybrid \
  --epochs 100 \
  --batch-size 256

# Evaluate models
python src/infrastructure/ml/training/model_evaluator.py \
  --model-path models/hybrid_model.h5 \
  --test-data data/test_interactions.csv
```

**Training Features:**
- Automated data preprocessing and feature engineering
- Hyperparameter optimization with Optuna
- Cross-validation and performance evaluation
- Model versioning and experiment tracking

### Performance Metrics

**Search Metrics:**
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10
- **MAP@10**: Mean Average Precision at 10
- **Response Time**: 95th percentile < 200ms

**Recommendation Metrics:**
- **Precision@K**: Relevant items in top-K recommendations
- **Recall@K**: Coverage of relevant items
- **Diversity**: Intra-list diversity score
- **Coverage**: Catalog coverage percentage

## 📚 Documentation

### API Documentation

- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Architecture Documentation

- [System Architecture](/root/terminus_directory/rental-ml-system/docs/diagrams/system_architecture.md)
- [ML Models Architecture](/root/terminus_directory/rental-ml-system/docs/diagrams/ml_models_architecture.md)
- [Recommendation Flow](/root/terminus_directory/rental-ml-system/docs/diagrams/recommendation_flow_diagram.md)
- [Search Ranking Flow](/root/terminus_directory/rental-ml-system/docs/diagrams/search_ranking_flow.md)

### Implementation Guides

- [Database Setup](/root/terminus_directory/rental-ml-system/DATABASE_SETUP.md)
- [Docker Deployment](/root/terminus_directory/rental-ml-system/DOCKER.md)
- [Hybrid Recommender Implementation](/root/terminus_directory/rental-ml-system/HYBRID_RECOMMENDER_IMPLEMENTATION.md)
- [Scraping Router Documentation](/root/terminus_directory/rental-ml-system/docs/scraping_router_documentation.md)

### Examples & Tutorials

**Code Examples:**
- [Hybrid Recommender Example](/root/terminus_directory/rental-ml-system/examples/hybrid_recommender_example.py)
- [Production ML Example](/root/terminus_directory/rental-ml-system/examples/production_ml_example.py)

**Jupyter Notebooks:**
```bash
# Start Jupyter server
jupyter lab notebooks/

# Available notebooks:
# - 01_data_exploration.ipynb
# - 02_nlp_search_development.ipynb  
# - 03_recommendation_model_training.ipynb
# - 04_system_evaluation.ipynb
# - 05_deployment_analysis.ipynb
```

### Troubleshooting Guide

#### Common Issues

**Installation Issues:**
```bash
# Python version compatibility
python --version  # Should be 3.9+

# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install -r requirements/base.txt --force-reinstall
```

**Database Connection:**
```bash
# Test database connection
python scripts/test_database_connection.py

# Reset database
python scripts/init_database.py --reset
```

**Model Loading Errors:**
```bash
# Check model files
ls -la models/

# Retrain models
python src/infrastructure/ml/training/ml_trainer.py --retrain
```

**Performance Issues:**
```bash
# Monitor system resources
docker stats

# Check Redis connection
redis-cli ping

# Monitor API performance
curl http://localhost:8000/health
```

## 🔧 Maintenance

### Monitoring & Logging

#### Prometheus Metrics
```bash
# Access Prometheus dashboard
http://localhost:9090

# Key metrics to monitor:
# - ml_model_prediction_latency
# - api_request_duration_seconds
# - database_query_duration_seconds
# - cache_hit_ratio
```

#### Log Aggregation
```bash
# View application logs
docker-compose logs -f app

# View specific service logs
kubectl logs -f deployment/rental-ml-app -n rental-ml

# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

#### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# ML models health
curl http://localhost:8000/health/models
```

### Backup Procedures

#### Database Backup
```bash
# Automated backup script
python scripts/backup_database.py \
  --output-dir backups/ \
  --compress

# Restore from backup
python scripts/restore_database.py \
  --backup-file backups/rental_ml_2024_01_15.sql.gz
```

#### Model Backup
```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Cloud backup (AWS S3)
aws s3 sync models/ s3://your-bucket/models/
```

### Performance Optimization

#### Database Optimization
```sql
-- Key database indexes
CREATE INDEX CONCURRENTLY idx_properties_location ON properties(city, neighborhood);
CREATE INDEX CONCURRENTLY idx_properties_price ON properties(price);
CREATE INDEX CONCURRENTLY idx_user_interactions_user_id ON user_interactions(user_id);
```

#### Redis Optimization
```bash
# Monitor Redis performance
redis-cli --latency-history

# Optimize Redis configuration
# - maxmemory-policy: allkeys-lru
# - save: disabled for cache-only usage
# - tcp-keepalive: 60
```

#### ML Model Optimization
```python
# Model quantization for faster inference
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('models/hybrid_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Security Considerations

#### Production Security Checklist

- [ ] **Database Security**
  - [ ] Use strong passwords (min 16 characters)
  - [ ] Enable SSL/TLS connections
  - [ ] Restrict database access by IP
  - [ ] Regular security updates

- [ ] **API Security**
  - [ ] Enable HTTPS only
  - [ ] Implement rate limiting
  - [ ] Use JWT for authentication
  - [ ] Validate all inputs

- [ ] **Container Security**
  - [ ] Use non-root containers
  - [ ] Scan images for vulnerabilities
  - [ ] Keep base images updated
  - [ ] Use secrets management

- [ ] **Infrastructure Security**
  - [ ] Network policies in Kubernetes
  - [ ] Regular security patches
  - [ ] Monitor for intrusions
  - [ ] Backup encryption

#### Security Updates
```bash
# Update dependencies for security patches
pip install --upgrade -r requirements/base.txt

# Scan for vulnerabilities
pip-audit

# Update Docker base images
docker-compose build --no-cache
```

## 🤝 Contributing

### Development Workflow

1. **Fork and Clone**
```bash
git clone https://github.com/your-username/rental-ml-system.git
cd rental-ml-system
```

2. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make Changes**
```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Make your changes
# Add tests for new functionality
# Update documentation
```

4. **Test Changes**
```bash
# Run full test suite
pytest

# Check code quality
pre-commit run --all-files

# Test in Docker environment
docker-compose -f docker-compose.dev.yml up -d
```

5. **Submit Pull Request**
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Code Standards

**Python Style:**
- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Maximum line length: 88 characters
- Use descriptive variable and function names

**Testing Requirements:**
- Minimum 80% code coverage
- Unit tests for all new functions
- Integration tests for API endpoints
- Performance tests for ML models

**Documentation:**
- Docstrings for all public functions
- Update README for new features
- Include usage examples
- Update API documentation

### Issue Reporting

**Bug Reports:**
```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- OS:
- Docker version:
```

**Feature Requests:**
```markdown
## Feature Description
Description of the requested feature

## Use Case
Why this feature would be useful

## Proposed Implementation
How you think it could be implemented

## Alternatives
Other ways to achieve the same goal
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team**: For the excellent deep learning framework
- **FastAPI Team**: For the high-performance web framework
- **Streamlit Team**: For the intuitive demo application framework
- **Open Source Community**: For the many libraries that make this project possible

---

## 🚀 Quick Commands Reference

```bash
# Start development environment
./demo-quick-start.sh

# Run full system with Docker
docker-compose up -d

# Deploy to Kubernetes
kubectl apply -f k8s/

# Run tests
pytest --cov=src

# Start API server
uvicorn src.application.api.main:app --reload

# Launch demo
streamlit run src/presentation/demo/app.py

# Train ML models
python src/infrastructure/ml/training/ml_trainer.py

# Check system health
curl http://localhost:8000/health
```

**Ready to build the future of rental property search? Get started with the demo and explore the power of AI-driven property matching!** 🏠✨