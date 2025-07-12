"""
Production model serving infrastructure for rental ML system.

This module provides high-performance model serving capabilities including:
- Real-time inference APIs
- Batch prediction processing
- Model loading and versioning
- A/B testing framework
- Caching and optimization
- Performance monitoring
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import redis
from sklearn.preprocessing import StandardScaler

from ..models.collaborative_filter import CollaborativeFilteringModel
from ..models.content_recommender import ContentBasedRecommender
from ..models.hybrid_recommender import HybridRecommendationSystem
from ..models.search_ranker import NLPSearchRanker
from ..training.data_loader import ProductionDataLoader


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    model_type: str
    version: str
    model_path: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    is_active: bool = False
    a_b_test_ratio: float = 0.0


@dataclass
class InferenceRequest:
    """Request for model inference"""
    user_id: Optional[UUID] = None
    property_ids: Optional[List[UUID]] = None
    query_text: Optional[str] = None
    num_recommendations: int = 10
    model_type: str = "hybrid"
    model_version: Optional[str] = None
    include_explanations: bool = True
    exclude_seen: bool = True


@dataclass
class InferenceResponse:
    """Response from model inference"""
    request_id: str
    user_id: Optional[UUID]
    recommendations: List[Dict]
    model_used: str
    model_version: str
    inference_time_ms: float
    cached: bool = False
    explanation: Optional[str] = None


class ModelCache:
    """In-memory model cache with LRU eviction"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str):
        """Get model from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, model):
        """Put model in cache with LRU eviction"""
        with self.lock:
            # Evict if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = model
            self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


class ABTestManager:
    """A/B testing manager for model experiments"""
    
    def __init__(self):
        self.experiments = {}
        self.user_assignments = {}
        self.metrics = {}
    
    def create_experiment(self, 
                         experiment_id: str,
                         model_a: str,
                         model_b: str,
                         traffic_split: float = 0.5):
        """Create A/B test experiment"""
        self.experiments[experiment_id] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'created_at': datetime.utcnow(),
            'active': True
        }
    
    def get_model_for_user(self, user_id: UUID, experiment_id: str) -> str:
        """Get model assignment for user in experiment"""
        if experiment_id not in self.experiments:
            return self.experiments[experiment_id]['model_a']
        
        # Consistent assignment based on user ID hash
        user_hash = hash(str(user_id)) % 100
        traffic_split = self.experiments[experiment_id]['traffic_split']
        
        if user_hash < traffic_split * 100:
            assigned_model = self.experiments[experiment_id]['model_a']
        else:
            assigned_model = self.experiments[experiment_id]['model_b']
        
        # Store assignment
        self.user_assignments[str(user_id)] = {
            'experiment_id': experiment_id,
            'model': assigned_model,
            'assigned_at': datetime.utcnow()
        }
        
        return assigned_model
    
    def record_metric(self, user_id: UUID, metric_name: str, value: float):
        """Record metric for A/B test analysis"""
        user_key = str(user_id)
        if user_key in self.user_assignments:
            experiment_id = self.user_assignments[user_key]['experiment_id']
            model = self.user_assignments[user_key]['model']
            
            if experiment_id not in self.metrics:
                self.metrics[experiment_id] = {}
            
            if model not in self.metrics[experiment_id]:
                self.metrics[experiment_id][model] = {}
            
            if metric_name not in self.metrics[experiment_id][model]:
                self.metrics[experiment_id][model][metric_name] = []
            
            self.metrics[experiment_id][model][metric_name].append(value)


class ModelServer:
    """
    Production model serving infrastructure.
    
    This class provides:
    - Model loading and management
    - Real-time inference with caching
    - Batch prediction processing
    - A/B testing capabilities
    - Performance monitoring
    - Model versioning
    """
    
    def __init__(self,
                 database_url: str,
                 models_dir: str = "/tmp/models",
                 redis_url: str = "redis://localhost:6379",
                 cache_ttl: int = 3600):
        self.database_url = database_url
        self.models_dir = Path(models_dir)
        self.cache_ttl = cache_ttl
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = ProductionDataLoader(database_url)
        self.model_cache = ModelCache(max_size=10)
        self.ab_test_manager = ABTestManager()
        
        # Redis for response caching
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Model registry
        self.model_registry = {}
        self.active_models = {}
        
        # Performance metrics
        self.inference_metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'average_latency': 0.0,
            'error_count': 0
        }
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self):
        """Initialize the model server"""
        try:
            await self.data_loader.initialize()
            await self._load_available_models()
            self.logger.info("Model server initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize model server: {e}")
            raise
    
    async def close(self):
        """Close resources"""
        await self.data_loader.close()
        if self.redis_client:
            await self.redis_client.close()
        self.executor.shutdown(wait=True)
    
    async def _load_available_models(self):
        """Load available models from models directory"""
        try:
            model_files = list(self.models_dir.glob("*.h5"))
            
            for model_file in model_files:
                try:
                    # Parse model info from filename
                    filename = model_file.stem
                    parts = filename.split('_')
                    
                    if len(parts) >= 3:
                        model_type = parts[0]
                        if model_type in ['collaborative', 'content', 'hybrid', 'search']:
                            version = parts[-1]  # timestamp
                            
                            model_version = ModelVersion(
                                model_id=str(uuid4()),
                                model_type=model_type,
                                version=version,
                                model_path=str(model_file),
                                created_at=datetime.utcnow(),
                                performance_metrics={},
                                is_active=True
                            )
                            
                            # Register model
                            if model_type not in self.model_registry:
                                self.model_registry[model_type] = []
                            
                            self.model_registry[model_type].append(model_version)
                            
                            # Set as active if first model of this type
                            if model_type not in self.active_models:
                                self.active_models[model_type] = model_version
                                
                except Exception as e:
                    self.logger.warning(f"Failed to load model {model_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.model_registry)} model types")
            
        except Exception as e:
            self.logger.error(f"Failed to load available models: {e}")
    
    async def load_model(self, model_type: str, version: Optional[str] = None):
        """Load model into memory"""
        try:
            # Get model version
            if version:
                model_version = self._get_model_version(model_type, version)
            else:
                model_version = self.active_models.get(model_type)
            
            if not model_version:
                raise ValueError(f"No model found for type {model_type}, version {version}")
            
            # Check cache first
            cache_key = f"{model_type}_{model_version.version}"
            cached_model = self.model_cache.get(cache_key)
            if cached_model:
                return cached_model
            
            # Load model based on type
            if model_type == 'collaborative':
                model = self._load_collaborative_model(model_version.model_path)
            elif model_type == 'content':
                model = self._load_content_model(model_version.model_path)
            elif model_type == 'hybrid':
                model = self._load_hybrid_model(model_version.model_path)
            elif model_type == 'search':
                model = self._load_search_model(model_version.model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cache model
            self.model_cache.put(cache_key, model)
            
            self.logger.info(f"Loaded {model_type} model version {model_version.version}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_type}: {e}")
            raise
    
    def _load_collaborative_model(self, model_path: str):
        """Load collaborative filtering model"""
        # Note: Would need to store model metadata to reconstruct properly
        # For now, create a placeholder
        model = CollaborativeFilteringModel(
            num_users=1000,  # Would be loaded from metadata
            num_items=500,
            embedding_dim=128
        )
        try:
            model.load_model(model_path)
        except Exception as e:
            self.logger.warning(f"Failed to load model weights: {e}")
        return model
    
    def _load_content_model(self, model_path: str):
        """Load content-based model"""
        model = ContentBasedRecommender()
        try:
            model.load_model(model_path)
        except Exception as e:
            self.logger.warning(f"Failed to load model weights: {e}")
        return model
    
    def _load_hybrid_model(self, model_path: str):
        """Load hybrid model"""
        model = HybridRecommendationSystem()
        # Would need to load both CF and CB models
        return model
    
    def _load_search_model(self, model_path: str):
        """Load search ranking model"""
        model = NLPSearchRanker()
        try:
            model.load_model(model_path)
        except Exception as e:
            self.logger.warning(f"Failed to load model weights: {e}")
        return model
    
    def _get_model_version(self, model_type: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version"""
        if model_type in self.model_registry:
            for model_version in self.model_registry[model_type]:
                if model_version.version == version:
                    return model_version
        return None
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """
        Make prediction using appropriate model.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response with predictions
        """
        start_time = time.time()
        request_id = str(uuid4())
        
        try:
            # Update metrics
            self.inference_metrics['total_requests'] += 1
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                self.inference_metrics['cache_hits'] += 1
                cached_response.request_id = request_id
                cached_response.cached = True
                return cached_response
            
            # A/B test model selection
            model_type = request.model_type
            if request.user_id and 'recommendation_ab_test' in self.ab_test_manager.experiments:
                model_type = self.ab_test_manager.get_model_for_user(
                    request.user_id, 'recommendation_ab_test'
                )
            
            # Load model
            model = await self.load_model(model_type, request.model_version)
            model_version = self.active_models.get(model_type)
            
            # Make prediction based on request type
            if request.query_text and model_type == 'search':
                recommendations = await self._search_predict(model, request)
            elif request.user_id:
                recommendations = await self._user_predict(model, request, model_type)
            else:
                raise ValueError("Either user_id or query_text must be provided")
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000
            
            # Create response
            response = InferenceResponse(
                request_id=request_id,
                user_id=request.user_id,
                recommendations=recommendations,
                model_used=model_type,
                model_version=model_version.version if model_version else "unknown",
                inference_time_ms=inference_time,
                cached=False
            )
            
            # Cache response
            await self._cache_response(cache_key, response)
            
            # Update metrics
            self._update_latency_metric(inference_time)
            
            self.logger.debug(f"Prediction completed in {inference_time:.2f}ms")
            return response
            
        except Exception as e:
            self.inference_metrics['error_count'] += 1
            self.logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _search_predict(self, model, request: InferenceRequest) -> List[Dict]:
        """Make search ranking prediction"""
        try:
            # Get properties for ranking (simplified)
            # In production, this would use proper search indexing
            properties_sample = []
            
            # Mock properties for demonstration
            for i in range(min(50, request.num_recommendations * 5)):
                properties_sample.append({
                    'id': str(uuid4()),
                    'title': f'Property {i}',
                    'description': f'Description for property {i}',
                    'location': f'Location {i % 10}',
                    'price': 1000 + i * 100,
                    'amenities': ['parking', 'gym'] if i % 2 == 0 else ['pool']
                })
            
            # Rank properties
            ranking_results = model.rank_properties(request.query_text, properties_sample)
            
            # Convert to recommendation format
            recommendations = []
            for result in ranking_results[:request.num_recommendations]:
                recommendations.append({
                    'property_id': result.property_id,
                    'score': result.relevance_score,
                    'explanation': f"Search relevance: {result.relevance_score:.3f}",
                    'property_data': result.property_data
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Search prediction failed: {e}")
            return []
    
    async def _user_predict(self, model, request: InferenceRequest, model_type: str) -> List[Dict]:
        """Make user-based recommendation prediction"""
        try:
            recommendations = []
            
            if model_type == 'collaborative':
                # Get collaborative filtering recommendations
                if hasattr(model, 'recommend'):
                    results = model.recommend(
                        user_id=0,  # Would map UUID to int
                        num_recommendations=request.num_recommendations,
                        exclude_seen=request.exclude_seen
                    )
                    
                    for result in results:
                        recommendations.append({
                            'property_id': str(uuid4()),  # Would be actual property ID
                            'score': result.predicted_rating,
                            'confidence': result.confidence_score,
                            'explanation': result.explanation if request.include_explanations else None
                        })
                        
            elif model_type == 'content':
                # Get content-based recommendations
                if hasattr(model, 'recommend'):
                    results = model.recommend(
                        user_id=0,  # Would map UUID to int
                        num_recommendations=request.num_recommendations,
                        exclude_seen=request.exclude_seen
                    )
                    
                    for result in results:
                        recommendations.append({
                            'property_id': str(uuid4()),
                            'score': result.predicted_rating,
                            'confidence': result.confidence_score,
                            'explanation': result.explanation if request.include_explanations else None
                        })
                        
            elif model_type == 'hybrid':
                # Get hybrid recommendations
                if hasattr(model, 'recommend'):
                    results = model.recommend(
                        user_id=0,  # Would map UUID to int
                        num_recommendations=request.num_recommendations,
                        exclude_seen=request.exclude_seen,
                        include_explanations=request.include_explanations
                    )
                    
                    for result in results:
                        recommendations.append({
                            'property_id': str(uuid4()),
                            'score': result.predicted_rating,
                            'confidence': result.confidence_score,
                            'explanation': result.explanation if request.include_explanations else None,
                            'cf_score': result.cf_score,
                            'cb_score': result.cb_score,
                            'hybrid_method': result.hybrid_method
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"User prediction failed: {e}")
            return []
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        key_parts = [
            str(request.user_id) if request.user_id else "no_user",
            request.query_text or "no_query",
            request.model_type,
            str(request.num_recommendations),
            str(request.exclude_seen)
        ]
        return ":".join(key_parts)
    
    async def _get_cached_response(self, cache_key: str) -> Optional[InferenceResponse]:
        """Get cached response"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                response_dict = json.loads(cached_data)
                return InferenceResponse(**response_dict)
        except Exception as e:
            self.logger.warning(f"Cache get failed: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: InferenceResponse):
        """Cache response"""
        if not self.redis_client:
            return
        
        try:
            # Convert response to dict for JSON serialization
            response_dict = asdict(response)
            # Handle UUID serialization
            if response_dict['user_id']:
                response_dict['user_id'] = str(response_dict['user_id'])
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(response_dict, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Cache set failed: {e}")
    
    def _update_latency_metric(self, latency: float):
        """Update average latency metric"""
        current_avg = self.inference_metrics['average_latency']
        total_requests = self.inference_metrics['total_requests']
        
        # Running average
        new_avg = ((current_avg * (total_requests - 1)) + latency) / total_requests
        self.inference_metrics['average_latency'] = new_avg
    
    async def batch_predict(self, 
                          requests: List[InferenceRequest],
                          batch_size: int = 32) -> List[InferenceResponse]:
        """
        Process multiple prediction requests in batch.
        
        Args:
            requests: List of inference requests
            batch_size: Size of processing batches
            
        Returns:
            List of inference responses
        """
        try:
            responses = []
            
            # Process requests in batches
            for i in range(0, len(requests), batch_size):
                batch = requests[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = [self.predict(request) for request in batch]
                batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle exceptions
                for response in batch_responses:
                    if isinstance(response, Exception):
                        self.logger.error(f"Batch prediction error: {response}")
                        # Create error response
                        error_response = InferenceResponse(
                            request_id=str(uuid4()),
                            user_id=None,
                            recommendations=[],
                            model_used="error",
                            model_version="error",
                            inference_time_ms=0.0,
                            explanation=str(response)
                        )
                        responses.append(error_response)
                    else:
                        responses.append(response)
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'model_registry': {
                model_type: [asdict(version) for version in versions]
                for model_type, versions in self.model_registry.items()
            },
            'active_models': {
                model_type: asdict(version)
                for model_type, version in self.active_models.items()
            },
            'cache_stats': {
                'cache_size': len(self.model_cache.cache),
                'max_cache_size': self.model_cache.max_size
            },
            'performance_metrics': self.inference_metrics
        }
    
    def get_ab_test_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test results"""
        if experiment_id not in self.ab_test_manager.metrics:
            return {'error': 'Experiment not found'}
        
        results = {}
        experiment_metrics = self.ab_test_manager.metrics[experiment_id]
        
        for model, metrics in experiment_metrics.items():
            model_results = {}
            for metric_name, values in metrics.items():
                if values:
                    model_results[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            results[model] = model_results
        
        return results
    
    async def deploy_model(self, 
                          model_type: str, 
                          model_path: str,
                          version: str,
                          performance_metrics: Dict[str, float]) -> bool:
        """
        Deploy new model version.
        
        Args:
            model_type: Type of model
            model_path: Path to model file
            version: Model version
            performance_metrics: Model performance metrics
            
        Returns:
            Success status
        """
        try:
            # Create model version
            model_version = ModelVersion(
                model_id=str(uuid4()),
                model_type=model_type,
                version=version,
                model_path=model_path,
                created_at=datetime.utcnow(),
                performance_metrics=performance_metrics,
                is_active=False
            )
            
            # Add to registry
            if model_type not in self.model_registry:
                self.model_registry[model_type] = []
            
            self.model_registry[model_type].append(model_version)
            
            # Clear cache for this model type
            cache_keys_to_remove = [
                key for key in self.model_cache.cache.keys()
                if key.startswith(f"{model_type}_")
            ]
            for key in cache_keys_to_remove:
                self.model_cache.cache.pop(key, None)
                self.model_cache.access_times.pop(key, None)
            
            self.logger.info(f"Deployed {model_type} model version {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False
    
    async def activate_model(self, model_type: str, version: str) -> bool:
        """Activate specific model version"""
        try:
            model_version = self._get_model_version(model_type, version)
            if not model_version:
                return False
            
            # Deactivate current model
            if model_type in self.active_models:
                self.active_models[model_type].is_active = False
            
            # Activate new model
            model_version.is_active = True
            self.active_models[model_type] = model_version
            
            self.logger.info(f"Activated {model_type} model version {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model activation failed: {e}")
            return False