# Intelligent Rental Property Recommendation & Search System
## Production-Grade ML System for Realtor.com Application

### 🎯 Project Overview
A real-time rental property recommendation and search system that combines sophisticated ML algorithms with production-ready architecture. This system demonstrates advanced NLP-powered search, hybrid recommendation engines, and scalable deployment using modern MLOps practices.

### 🏗️ Architecture & Design Philosophy

#### Clean Architecture Principles
```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   FastAPI       │  │   Streamlit     │  │   Jupyter   │ │
│  │   REST API      │  │   Demo UI       │  │   Notebooks │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Search        │  │  Recommendation │  │   User      │ │
│  │   Service       │  │   Service       │  │   Service   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Domain Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Property      │  │   User          │  │   Search    │ │
│  │   Models        │  │   Models        │  │   Models    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Data Access   │  │   ML Models     │  │   Caching   │ │
│  │   Repository    │  │   Repository    │  │   Layer     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 📁 Project Structure
```
rental-ml-system/
├── src/
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── property.py
│   │   │   ├── user.py
│   │   │   └── search_query.py
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── property_repository.py
│   │   │   ├── user_repository.py
│   │   │   └── model_repository.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── search_service.py
│   │       ├── recommendation_service.py
│   │       └── user_service.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── scrapers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── zillow_scraper.py
│   │   │   │   └── apartments_scraper.py
│   │   │   ├── repositories/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── postgres_property_repository.py
│   │   │   │   └── redis_cache_repository.py
│   │   │   └── processors/
│   │   │       ├── __init__.py
│   │   │       ├── text_processor.py
│   │   │       └── feature_engineer.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── search_ranker.py
│   │   │   │   ├── collaborative_filter.py
│   │   │   │   ├── content_recommender.py
│   │   │   │   └── hybrid_recommender.py
│   │   │   ├── training/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── trainer.py
│   │   │   │   └── evaluator.py
│   │   │   └── serving/
│   │   │       ├── __init__.py
│   │   │       ├── model_server.py
│   │   │       └── batch_inference.py
│   │   └── monitoring/
│   │       ├── __init__.py
│   │       ├── metrics.py
│   │       └── logging.py
│   ├── application/
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── search_endpoints.py
│   │   │   ├── recommendation_endpoints.py
│   │   │   └── user_endpoints.py
│   │   ├── dto/
│   │   │   ├── __init__.py
│   │   │   ├── search_dto.py
│   │   │   └── recommendation_dto.py
│   │   └── use_cases/
│   │       ├── __init__.py
│   │       ├── search_properties.py
│   │       ├── get_recommendations.py
│   │       └── track_user_interaction.py
│   └── presentation/
│       ├── __init__.py
│       ├── web/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── config.py
│       └── demo/
│           ├── __init__.py
│           └── streamlit_app.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_nlp_search_development.ipynb
│   ├── 03_recommendation_model_training.ipynb
│   ├── 04_system_evaluation.ipynb
│   └── 05_deployment_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_domain/
│   │   ├── test_infrastructure/
│   │   └── test_application/
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api/
│   │   └── test_ml_pipeline/
│   └── performance/
│       ├── __init__.py
│       └── load_tests.py
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── docker-compose.prod.yml
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── model_config.yaml
├── scripts/
│   ├── setup_environment.sh
│   ├── train_models.py
│   ├── deploy_local.sh
│   └── deploy_production.sh
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
└── README.md
```

### 🧠 Core ML Components

#### 1. NLP-Powered Search System
```python
# src/infrastructure/ml/models/search_ranker.py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from typing import List, Dict, Tuple

class NLPSearchRanker:
    """Advanced NLP-powered search ranking using TensorFlow and Transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = TFAutoModel.from_pretrained(model_name)
        self.ranking_model = self._build_ranking_model()
        
    def _build_ranking_model(self) -> tf.keras.Model:
        """Build TensorFlow ranking model"""
        query_input = tf.keras.layers.Input(shape=(384,), name="query_embedding")
        property_input = tf.keras.layers.Input(shape=(384,), name="property_embedding")
        
        # Feature interaction layers
        concat_features = tf.keras.layers.Concatenate()([query_input, property_input])
        
        # Deep ranking network
        hidden = tf.keras.layers.Dense(512, activation='relu')(concat_features)
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        
        # Ranking score output
        score = tf.keras.layers.Dense(1, activation='sigmoid', name="relevance_score")(hidden)
        
        model = tf.keras.Model(
            inputs=[query_input, property_input],
            outputs=score
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def encode_text(self, texts: List[str]) -> tf.Tensor:
        """Encode texts using transformer model"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='tf',
            max_length=512
        )
        
        outputs = self.encoder(**inputs)
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    
    def rank_properties(self, query: str, properties: List[Dict]) -> List[Tuple[Dict, float]]:
        """Rank properties based on search query using ML"""
        # Encode query
        query_embedding = self.encode_text([query])
        
        # Encode property descriptions
        property_texts = [
            f"{prop['title']} {prop['description']} {prop['location']}"
            for prop in properties
        ]
        property_embeddings = self.encode_text(property_texts)
        
        # Get ranking scores
        scores = self.ranking_model.predict([
            tf.repeat(query_embedding, len(properties), axis=0),
            property_embeddings
        ])
        
        # Combine properties with scores and sort
        ranked_properties = list(zip(properties, scores.flatten()))
        ranked_properties.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_properties
```

#### 2. Hybrid Recommendation Engine
```python
# src/infrastructure/ml/models/hybrid_recommender.py
import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    """Base class for recommendation models"""
    
    @abstractmethod
    def fit(self, user_item_matrix: np.ndarray, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        pass

class CollaborativeFilteringModel(BaseRecommender):
    """Neural Collaborative Filtering using TensorFlow"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build neural collaborative filtering model"""
        # User and item embeddings
        user_input = tf.keras.layers.Input(shape=(), name='user_id')
        item_input = tf.keras.layers.Input(shape=(), name='item_id')
        
        user_embedding = tf.keras.layers.Embedding(
            self.num_users, self.embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )(user_input)
        
        item_embedding = tf.keras.layers.Embedding(
            self.num_items, self.embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )(item_input)
        
        # Flatten embeddings
        user_vec = tf.keras.layers.Flatten()(user_embedding)
        item_vec = tf.keras.layers.Flatten()(item_embedding)
        
        # Neural MF layers
        concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
        hidden = tf.keras.layers.Dense(128, activation='relu')(concat)
        hidden = tf.keras.layers.Dropout(0.2)(hidden)
        hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
        hidden = tf.keras.layers.Dropout(0.2)(hidden)
        
        # Output layer
        output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
        
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def fit(self, user_item_matrix: np.ndarray, epochs: int = 100, **kwargs):
        """Train the collaborative filtering model"""
        # Prepare training data
        users, items, ratings = [], [], []
        
        for user_id in range(user_item_matrix.shape[0]):
            for item_id in range(user_item_matrix.shape[1]):
                if user_item_matrix[user_id, item_id] > 0:
                    users.append(user_id)
                    items.append(item_id)
                    ratings.append(user_item_matrix[user_id, item_id])
        
        # Convert to numpy arrays
        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings)
        
        # Train model
        self.model.fit(
            [users, items], ratings,
            epochs=epochs,
            batch_size=256,
            validation_split=0.2,
            verbose=1
        )
    
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Predict ratings for user-item pairs"""
        user_ids = np.array([user_id] * len(item_ids))
        item_ids = np.array(item_ids)
        
        predictions = self.model.predict([user_ids, item_ids])
        return predictions.flatten()

class HybridRecommendationSystem:
    """Combines collaborative filtering and content-based recommendations"""
    
    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = None
        self.cb_model = None
    
    def fit(self, user_item_matrix: np.ndarray, item_features: np.ndarray):
        """Train both collaborative filtering and content-based models"""
        # Train collaborative filtering
        self.cf_model = CollaborativeFilteringModel(
            num_users=user_item_matrix.shape[0],
            num_items=user_item_matrix.shape[1]
        )
        self.cf_model.fit(user_item_matrix)
        
        # Train content-based model
        self.cb_model = ContentBasedModel(item_features.shape[1])
        self.cb_model.fit(item_features, user_item_matrix)
    
    def recommend(self, user_id: int, item_ids: List[int], 
                 item_features: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Generate hybrid recommendations"""
        # Get collaborative filtering predictions
        cf_scores = self.cf_model.predict(user_id, item_ids)
        
        # Get content-based predictions
        cb_scores = self.cb_model.predict(user_id, item_features)
        
        # Combine scores
        hybrid_scores = (self.cf_weight * cf_scores + 
                        self.cb_weight * cb_scores)
        
        # Get top-k recommendations
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'item_id': item_ids[idx],
                'score': hybrid_scores[idx],
                'cf_score': cf_scores[idx],
                'cb_score': cb_scores[idx]
            })
        
        return recommendations
```

### 🛠️ Production Infrastructure

#### 1. Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/base.txt requirements/prod.txt ./
RUN pip install --no-cache-dir -r prod.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "src.presentation.web.main"]
```

#### 2. Kubernetes Deployment
```yaml
# deployment/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rental-ml-system
  namespace: rental-ml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rental-ml-system
  template:
    metadata:
      labels:
        app: rental-ml-system
    spec:
      containers:
      - name: ml-api
        image: rental-ml-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: redis_url
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 📊 Data Pipeline & Processing

#### 1. Data Scraping & Ingestion
```python
# src/infrastructure/data/scrapers/apartments_scraper.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PropertyListing:
    title: str
    description: str
    price: float
    location: str
    bedrooms: int
    bathrooms: float
    square_feet: Optional[int]
    amenities: List[str]
    contact_info: Dict[str, str]
    images: List[str]
    scraped_at: datetime

class RentalPropertyScraper:
    """Ethical web scraper for rental property data"""
    
    def __init__(self, max_concurrent: int = 5, delay: float = 1.0):
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.session = None
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (research purposes)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_property_listings(self, urls: List[str]) -> List[PropertyListing]:
        """Scrape property listings from multiple URLs"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._scrape_single_property(url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        properties = [r for r in results if isinstance(r, PropertyListing)]
        return properties
    
    async def _scrape_single_property(self, url: str, semaphore: asyncio.Semaphore) -> Optional[PropertyListing]:
        """Scrape a single property listing"""
        async with semaphore:
            try:
                await asyncio.sleep(self.delay)  # Rate limiting
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_property_html(html)
                    else:
                        self.logger.warning(f"Failed to scrape {url}: {response.status}")
                        return None
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {e}")
                return None
    
    def _parse_property_html(self, html: str) -> Optional[PropertyListing]:
        """Parse property information from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # Extract property details (site-specific selectors)
            title = soup.find('h1', class_='property-title')?.get_text(strip=True)
            description = soup.find('div', class_='property-description')?.get_text(strip=True)
            price_text = soup.find('span', class_='price')?.get_text(strip=True)
            
            # Parse price
            price = self._extract_price(price_text) if price_text else None
            
            # Extract other details
            bedrooms = self._extract_bedrooms(soup)
            bathrooms = self._extract_bathrooms(soup)
            square_feet = self._extract_square_feet(soup)
            
            if title and description and price:
                return PropertyListing(
                    title=title,
                    description=description,
                    price=price,
                    location=self._extract_location(soup),
                    bedrooms=bedrooms,
                    bathrooms=bathrooms,
                    square_feet=square_feet,
                    amenities=self._extract_amenities(soup),
                    contact_info=self._extract_contact_info(soup),
                    images=self._extract_images(soup),
                    scraped_at=datetime.now()
                )
        except Exception as e:
            self.logger.error(f"Error parsing property HTML: {e}")
            return None
    
    def _extract_price(self, price_text: str) -> Optional[float]:
        """Extract numeric price from text"""
        import re
        price_match = re.search(r'[\$]?([0-9,]+)', price_text.replace(',', ''))
        return float(price_match.group(1)) if price_match else None
```

### 🚀 Deployment Strategy

#### Progressive Deployment Plan:

1. **Local Development (Week 1-2)**
   - Set up development environment with Docker
   - Implement core ML models and API endpoints
   - Create comprehensive test suite

2. **Containerization (Week 3)**
   - Dockerize all components
   - Set up local Kubernetes cluster (minikube)
   - Implement monitoring and logging

3. **AWS Deployment (Week 4)**
   - Deploy to AWS EKS (free tier)
   - Set up CI/CD pipeline with GitHub Actions
   - Configure monitoring and alerting

#### Budget-Conscious AWS Setup:
- **EKS Cluster**: Free tier (pay only for EC2 instances)
- **EC2 Instances**: t3.small instances (~$15/month each)
- **RDS PostgreSQL**: db.t3.micro (~$13/month)
- **ElastiCache Redis**: cache.t3.micro (~$12/month)
- **S3 Storage**: Minimal costs for model artifacts
- **Total Estimated Cost**: ~$50-70/month

### 📈 Success Metrics & Evaluation

#### Technical Metrics:
- **Search Relevance**: NDCG@10, MAP@10
- **Recommendation Quality**: Precision@K, Recall@K
- **System Performance**: 95th percentile latency < 200ms
- **Scalability**: Handle 1000+ concurrent requests

#### Business Metrics:
- **User Engagement**: CTR, time on property pages
- **Conversion Simulation**: Mock inquiry generation
- **Recommendation Diversity**: Intra-list diversity scores
- **Search Success Rate**: Queries with clicks

### 🎯 Key Differentiators for Realtor.com

1. **Production-Ready Architecture**: Clean, maintainable, scalable code
2. **Advanced NLP**: TensorFlow-based semantic search
3. **Hybrid ML Approach**: Combines multiple recommendation strategies
4. **Real-time Performance**: Optimized for low-latency serving
5. **Comprehensive Testing**: Unit, integration, and performance tests
6. **Modern DevOps**: Kubernetes, CI/CD, monitoring
7. **Cost-Effective Scaling**: Efficient resource utilization

This system demonstrates exactly what Realtor.com is looking for: a senior-level ML engineer who can build sophisticated, production-ready systems that directly impact user experience and business metrics.

### 📋 Implementation Timeline

**Week 1**: Core ML models and local development
**Week 2**: API development and testing
**Week 3**: Containerization and Kubernetes setup
**Week 4**: AWS deployment and documentation

Ready to start building? I recommend beginning with the NLP search ranker as it showcases your TensorFlow learning goal while building on your existing Python expertise.
