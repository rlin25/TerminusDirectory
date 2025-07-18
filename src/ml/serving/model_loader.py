"""
Model Loader for Production ML Serving Infrastructure.

This module provides advanced model loading capabilities including:
- Lazy loading with memory optimization
- Model versioning and caching
- TensorFlow Serving integration
- Model validation and health checks
- Multi-threading for concurrent loading
"""

import asyncio
import logging
import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import hashlib

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
import joblib

from ...infrastructure.ml.models.collaborative_filter import CollaborativeFilteringModel
from ...infrastructure.ml.models.content_recommender import ContentBasedRecommender
from ...infrastructure.ml.models.hybrid_recommender import HybridRecommendationSystem
from ...infrastructure.ml.models.search_ranker import NLPSearchRanker


@dataclass
class ModelInfo:
    """Information about a discovered model"""
    type: str
    version: str
    path: str
    size_bytes: int
    created_at: datetime
    checksum: str
    metadata: Dict[str, Any]
    format: str  # 'tensorflow', 'sklearn', 'custom'


class ModelCache:
    """Thread-safe model cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10, max_memory_gb: float = 8.0):
        self.max_size = max_size
        self.max_memory_gb = max_memory_gb
        self.cache = {}
        self.access_times = {}
        self.memory_usage = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str):
        """Get model from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, model, memory_mb: float = 0):
        """Put model in cache with memory tracking"""
        with self.lock:
            # Check memory constraints
            current_memory = sum(self.memory_usage.values())
            if current_memory + memory_mb > self.max_memory_gb * 1024:
                self._evict_by_memory()
            
            # Check size constraints
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = model
            self.access_times[key] = time.time()
            self.memory_usage[key] = memory_mb
            
            self.logger.debug(f"Cached model {key} ({memory_mb:.1f}MB)")
    
    def remove(self, key: str):
        """Remove model from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                del self.memory_usage[key]
    
    def _evict_lru(self):
        """Evict least recently used model"""
        if self.access_times:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.remove(oldest_key)
            self.logger.debug(f"Evicted LRU model {oldest_key}")
    
    def _evict_by_memory(self):
        """Evict models to free memory"""
        # Sort by access time, evict oldest first
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        for key in sorted_keys:
            self.remove(key)
            current_memory = sum(self.memory_usage.values())
            if current_memory < self.max_memory_gb * 1024 * 0.8:  # 80% threshold
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': sum(self.memory_usage.values()),
                'max_memory_gb': self.max_memory_gb,
                'cached_models': list(self.cache.keys())
            }


class ModelLoader:
    """
    Advanced model loader for production ML serving.
    
    Features:
    - Async model loading with thread pool
    - Model caching with memory management
    - Format detection and validation
    - Model health checks
    - Version management
    - Lazy loading optimization
    """
    
    def __init__(self, 
                 models_dir: str,
                 cache_size: int = 10,
                 max_memory_gb: float = 8.0,
                 num_workers: int = 4):
        
        self.models_dir = Path(models_dir)
        self.cache = ModelCache(cache_size, max_memory_gb)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.discovered_models = {}
        self.model_metadata = {}
        
        # Loading state
        self.loading_locks = {}
        self.health_status = {}
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover all available models in the models directory"""
        try:
            models = []
            
            if not self.models_dir.exists():
                self.logger.warning(f"Models directory {self.models_dir} does not exist")
                return models
            
            # Scan for model files
            model_files = []
            model_files.extend(self.models_dir.glob("*.h5"))         # TensorFlow/Keras
            model_files.extend(self.models_dir.glob("*.pkl"))        # Pickle
            model_files.extend(self.models_dir.glob("*.joblib"))     # Joblib
            model_files.extend(self.models_dir.glob("*.pb"))         # TensorFlow SavedModel
            
            for model_file in model_files:
                try:
                    model_info = await self._analyze_model_file(model_file)
                    if model_info:
                        models.append(model_info)
                        
                        # Store in registry
                        key = f"{model_info['type']}_{model_info['version']}"
                        self.discovered_models[key] = model_info
                        
                except Exception as e:
                    self.logger.warning(f"Failed to analyze model file {model_file}: {e}")
            
            self.logger.info(f"Discovered {len(models)} models")
            return models
            
        except Exception as e:
            self.logger.error(f"Model discovery failed: {e}")
            return []
    
    async def _analyze_model_file(self, model_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze a model file to extract metadata"""
        try:
            # Parse filename to extract type and version
            filename = model_file.stem
            parts = filename.split('_')
            
            if len(parts) < 3:
                return None
            
            model_type = parts[0]
            if model_type not in ['collaborative', 'content', 'hybrid', 'search']:
                return None
            
            # Extract version (usually timestamp)
            version = parts[-1]
            
            # Get file stats
            stat = model_file.stat()
            
            # Calculate checksum
            checksum = await self._calculate_checksum(model_file)
            
            # Determine format
            format_type = self._detect_format(model_file)
            
            # Load metadata if available
            metadata_file = model_file.parent / f"{filename}_metadata.json"
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata for {model_file}: {e}")
            
            return {
                'type': model_type,
                'version': version,
                'path': str(model_file),
                'size_bytes': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_mtime),
                'checksum': checksum,
                'metadata': metadata,
                'format': format_type
            }
            
        except Exception as e:
            self.logger.error(f"Model file analysis failed: {e}")
            return None
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        def _hash_file():
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _hash_file)
    
    def _detect_format(self, model_file: Path) -> str:
        """Detect model format from file extension"""
        suffix = model_file.suffix.lower()
        
        if suffix == '.h5':
            return 'tensorflow'
        elif suffix in ['.pkl', '.pickle']:
            return 'pickle'
        elif suffix == '.joblib':
            return 'joblib'
        elif suffix == '.pb':
            return 'tensorflow_pb'
        else:
            return 'unknown'
    
    async def load_model(self, model_type: str, version: str, model_path: str):
        """Load a specific model into memory"""
        try:
            cache_key = f"{model_type}_{version}"
            
            # Check if already cached
            cached_model = self.cache.get(cache_key)
            if cached_model:
                return cached_model
            
            # Prevent concurrent loading of same model
            if cache_key in self.loading_locks:
                # Wait for ongoing load
                while cache_key in self.loading_locks:
                    await asyncio.sleep(0.1)
                return self.cache.get(cache_key)
            
            self.loading_locks[cache_key] = True
            
            try:
                # Load model in thread pool
                model = await self._load_model_sync(model_type, version, model_path)
                
                # Estimate memory usage
                memory_mb = self._estimate_model_memory(model)
                
                # Cache the model
                self.cache.put(cache_key, model, memory_mb)
                
                # Update health status
                self.health_status[cache_key] = {
                    'status': 'healthy',
                    'last_loaded': datetime.utcnow(),
                    'memory_mb': memory_mb
                }
                
                self.logger.info(f"Loaded model {cache_key} ({memory_mb:.1f}MB)")
                return model
                
            finally:
                del self.loading_locks[cache_key]
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_type} v{version}: {e}")
            
            # Update health status
            cache_key = f"{model_type}_{version}"
            self.health_status[cache_key] = {
                'status': 'unhealthy',
                'error': str(e),
                'last_attempt': datetime.utcnow()
            }
            raise
    
    async def _load_model_sync(self, model_type: str, version: str, model_path: str):
        """Synchronous model loading (runs in thread pool)"""
        def _load():
            try:
                # Detect format
                path_obj = Path(model_path)
                format_type = self._detect_format(path_obj)
                
                if format_type == 'tensorflow':
                    return self._load_tensorflow_model(model_type, model_path)
                elif format_type in ['pickle', 'joblib']:
                    return self._load_sklearn_model(model_type, model_path, format_type)
                else:
                    raise ValueError(f"Unsupported model format: {format_type}")
                    
            except Exception as e:
                self.logger.error(f"Model loading failed: {e}")
                raise
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _load)
    
    def _load_tensorflow_model(self, model_type: str, model_path: str):
        """Load TensorFlow/Keras model"""
        try:
            if model_type == 'collaborative':
                # Create model instance first
                model = CollaborativeFilteringModel(
                    num_users=1000,  # Will be updated from metadata
                    num_items=500,
                    embedding_dim=128
                )
                # Load weights
                if Path(model_path).exists():
                    model.load_model(model_path)
                return model
                
            elif model_type == 'content':
                model = ContentBasedRecommender()
                if Path(model_path).exists():
                    model.load_model(model_path)
                return model
                
            elif model_type == 'hybrid':
                model = HybridRecommendationSystem()
                # For hybrid models, we need to load both CF and CB components
                # This is simplified - in production you'd have separate paths
                return model
                
            elif model_type == 'search':
                model = NLPSearchRanker()
                if Path(model_path).exists():
                    model.load_model(model_path)
                return model
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"TensorFlow model loading failed: {e}")
            raise
    
    def _load_sklearn_model(self, model_type: str, model_path: str, format_type: str):
        """Load scikit-learn model"""
        try:
            if format_type == 'pickle':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif format_type == 'joblib':
                model = joblib.load(model_path)
            else:
                raise ValueError(f"Unsupported sklearn format: {format_type}")
            
            # Validate model
            if not hasattr(model, 'predict'):
                raise ValueError("Loaded model doesn't have predict method")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Sklearn model loading failed: {e}")
            raise
    
    def _estimate_model_memory(self, model) -> float:
        """Estimate model memory usage in MB"""
        try:
            # For TensorFlow models
            if hasattr(model, 'model') and hasattr(model.model, 'count_params'):
                params = model.model.count_params()
                # Rough estimate: 4 bytes per parameter (float32)
                return (params * 4) / (1024 * 1024)
            
            # For sklearn models
            elif hasattr(model, '__sizeof__'):
                return model.__sizeof__() / (1024 * 1024)
            
            # Default estimate
            return 50.0  # 50MB default
            
        except Exception:
            return 50.0
    
    async def get_model(self, model_type: str, version: str):
        """Get model from cache or load if not available"""
        cache_key = f"{model_type}_{version}"
        
        # Try cache first
        model = self.cache.get(cache_key)
        if model:
            return model
        
        # Find model path
        model_info = self.discovered_models.get(cache_key)
        if not model_info:
            raise ValueError(f"Model {cache_key} not found")
        
        # Load model
        return await self.load_model(model_type, version, model_info['path'])
    
    async def unload_model(self, model_type: str, version: str):
        """Unload model from cache"""
        cache_key = f"{model_type}_{version}"
        self.cache.remove(cache_key)
        
        # Clear health status
        if cache_key in self.health_status:
            del self.health_status[cache_key]
        
        self.logger.info(f"Unloaded model {cache_key}")
    
    async def validate_model(self, model_type: str, version: str) -> Dict[str, Any]:
        """Validate model health and functionality"""
        try:
            cache_key = f"{model_type}_{version}"
            model = await self.get_model(model_type, version)
            
            if not model:
                return {
                    'status': 'unhealthy',
                    'error': 'Model not found or failed to load'
                }
            
            # Basic validation - check if model has required methods
            validation_results = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'checks': {}
            }
            
            # Check predict method
            if hasattr(model, 'predict'):
                validation_results['checks']['predict_method'] = True
            else:
                validation_results['checks']['predict_method'] = False
                validation_results['status'] = 'unhealthy'
            
            # Check recommend method for recommendation models
            if model_type in ['collaborative', 'content', 'hybrid']:
                if hasattr(model, 'recommend'):
                    validation_results['checks']['recommend_method'] = True
                else:
                    validation_results['checks']['recommend_method'] = False
                    validation_results['status'] = 'degraded'
            
            # Model-specific checks
            if model_type == 'collaborative':
                validation_results['checks']['user_item_matrix'] = hasattr(model, 'user_item_matrix')
            elif model_type == 'content':
                validation_results['checks']['property_features'] = hasattr(model, 'property_embeddings')
            elif model_type == 'hybrid':
                validation_results['checks']['cf_model'] = hasattr(model, 'cf_model')
                validation_results['checks']['cb_model'] = hasattr(model, 'cb_model')
            
            # Performance test with dummy data
            try:
                start_time = time.time()
                
                if model_type == 'search':
                    # Test search ranking
                    if hasattr(model, 'rank_properties'):
                        test_properties = [{'title': 'test', 'description': 'test property'}]
                        model.rank_properties("test query", test_properties)
                else:
                    # Test recommendation
                    if hasattr(model, 'predict'):
                        model.predict(0, [0, 1, 2])
                
                inference_time = (time.time() - start_time) * 1000
                validation_results['checks']['inference_test'] = True
                validation_results['inference_time_ms'] = inference_time
                
                # Check if inference time is reasonable (< 1000ms)
                if inference_time > 1000:
                    validation_results['status'] = 'degraded'
                    validation_results['warning'] = 'High inference latency'
                
            except Exception as e:
                validation_results['checks']['inference_test'] = False
                validation_results['status'] = 'unhealthy'
                validation_results['error'] = f"Inference test failed: {str(e)}"
            
            # Update health status
            self.health_status[cache_key] = {
                'status': validation_results['status'],
                'last_validated': datetime.utcnow(),
                'validation_results': validation_results
            }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation failed for {model_type} v{version}: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_model_info(self, model_type: str, version: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            cache_key = f"{model_type}_{version}"
            
            # Get discovered model info
            model_info = self.discovered_models.get(cache_key, {})
            
            # Get health status
            health = self.health_status.get(cache_key, {'status': 'unknown'})
            
            # Check if model is cached
            is_cached = cache_key in self.cache.cache
            
            return {
                'model_type': model_type,
                'version': version,
                'file_info': model_info,
                'health_status': health,
                'cached': is_cached,
                'cache_stats': self.cache.get_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    async def list_loaded_models(self) -> List[Dict[str, Any]]:
        """List all currently loaded models"""
        try:
            loaded_models = []
            cache_stats = self.cache.get_stats()
            
            for model_key in cache_stats['cached_models']:
                parts = model_key.split('_', 1)
                if len(parts) == 2:
                    model_type, version = parts
                    
                    model_info = {
                        'model_type': model_type,
                        'version': version,
                        'cache_key': model_key,
                        'health_status': self.health_status.get(model_key, {'status': 'unknown'})
                    }
                    
                    loaded_models.append(model_info)
            
            return loaded_models
            
        except Exception as e:
            self.logger.error(f"Failed to list loaded models: {e}")
            return []
    
    async def cleanup_unused_models(self, max_age_hours: int = 24):
        """Clean up models that haven't been accessed recently"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            models_to_remove = []
            
            with self.cache.lock:
                for cache_key, access_time in self.cache.access_times.items():
                    if current_time - access_time > max_age_seconds:
                        models_to_remove.append(cache_key)
            
            for cache_key in models_to_remove:
                self.cache.remove(cache_key)
                if cache_key in self.health_status:
                    del self.health_status[cache_key]
                
                self.logger.info(f"Cleaned up unused model {cache_key}")
            
            return {
                'cleaned_models': models_to_remove,
                'count': len(models_to_remove)
            }
            
        except Exception as e:
            self.logger.error(f"Model cleanup failed: {e}")
            return {'error': str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    async def close(self):
        """Close the model loader"""
        try:
            # Clear cache
            with self.cache.lock:
                self.cache.cache.clear()
                self.cache.access_times.clear()
                self.cache.memory_usage.clear()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Model loader closed")
            
        except Exception as e:
            self.logger.error(f"Error closing model loader: {e}")