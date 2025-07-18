"""
Production ML Training Pipeline for scalable model training and deployment.

This pipeline provides:
- Multi-model training orchestration (collaborative filtering, content-based, hybrid)
- Distributed training across multiple GPUs/nodes
- Hyperparameter optimization with Optuna and Ray Tune
- Advanced validation strategies including cross-validation
- Model performance comparison and selection
- Automated feature engineering and data preprocessing
- Integration with MLflow for experiment tracking
- Model versioning and artifact management
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from uuid import uuid4

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import KFold, TimeSeriesSplit
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
import mlflow
import mlflow.tensorflow
import psutil

from ...infrastructure.ml.models.collaborative_filter import CollaborativeFilteringModel
from ...infrastructure.ml.models.content_recommender import ContentBasedRecommender
from ...infrastructure.ml.models.hybrid_recommender import HybridRecommendationSystem
from ...infrastructure.ml.models.search_ranker import NLPSearchRanker
from ...infrastructure.ml.training.data_loader import ProductionDataLoader, MLDataset


@dataclass
class TrainingConfig:
    """Advanced training configuration for production ML pipeline"""
    # Model configuration
    model_types: List[str]  # ['collaborative', 'content', 'hybrid', 'search_ranker']
    model_variants: Dict[str, List[str]] = None  # Model architecture variants
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 15
    
    # Advanced training features
    use_mixed_precision: bool = True
    gradient_clipping: float = 1.0
    weight_decay: float = 1e-4
    dropout_rate: float = 0.2
    batch_norm: bool = True
    
    # Model architecture
    embedding_dims: List[int] = None
    hidden_layers: List[int] = None
    activation: str = 'relu'
    optimizer: str = 'adam'
    
    # Distributed training
    use_distributed: bool = False
    num_gpus: int = 1
    num_workers: int = 4
    distributed_backend: str = 'nccl'
    
    # Hyperparameter optimization
    use_hyperopt: bool = False
    hyperopt_trials: int = 50
    hyperopt_algorithm: str = 'optuna'  # 'optuna', 'ray_tune'
    
    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5
    cv_strategy: str = 'kfold'  # 'kfold', 'time_series', 'stratified'
    
    # Data preprocessing
    feature_engineering: bool = True
    data_augmentation: bool = False
    handle_imbalance: bool = True
    feature_selection: bool = True
    
    # Performance and scaling
    max_memory_gb: float = 16.0
    prefetch_size: int = 10
    num_parallel_calls: int = 4
    cache_dataset: bool = True
    
    # Experiment tracking
    experiment_name: str = "production_ml_training"
    run_name: Optional[str] = None
    track_artifacts: bool = True
    log_model: bool = True
    
    # Model selection criteria
    primary_metric: str = 'val_rmse'
    metric_direction: str = 'minimize'  # 'minimize' or 'maximize'
    ensemble_models: bool = False
    
    # Production settings
    model_registry_stage: str = 'staging'  # 'staging', 'production'
    auto_deploy_threshold: float = 0.05  # Improvement threshold for auto-deployment


@dataclass
class ModelResult:
    """Results from model training and evaluation"""
    model_type: str
    model_variant: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]
    cv_metrics: Optional[Dict[str, float]]
    training_time: float
    model_path: str
    model_size_mb: float
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    config: TrainingConfig
    metadata: Dict[str, Any]


class ProductionMLPipeline:
    """
    Enterprise-grade ML training pipeline for rental property recommendation system.
    
    Features:
    - Multi-model training with parallel execution
    - Distributed training across multiple GPUs/nodes
    - Automated hyperparameter optimization
    - Advanced validation strategies
    - Model performance monitoring and comparison
    - Automated feature engineering
    - MLflow integration for experiment tracking
    - Model versioning and deployment automation
    """
    
    def __init__(self, 
                 database_url: str,
                 models_dir: str = "/app/models",
                 artifacts_dir: str = "/app/artifacts",
                 mlflow_tracking_uri: Optional[str] = None,
                 ray_address: Optional[str] = None):
        
        self.database_url = database_url
        self.models_dir = Path(models_dir)
        self.artifacts_dir = Path(artifacts_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = ProductionDataLoader(database_url)
        
        # Initialize distributed computing
        if ray_address:
            ray.init(address=ray_address)
        elif not ray.is_initialized():
            ray.init(num_cpus=mp.cpu_count(), num_gpus=tf.config.list_physical_devices('GPU').__len__())
        
        # Initialize MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Training state
        self.trained_models = {}
        self.training_results = {}
        self.best_models = {}
        
        # Performance monitoring
        self.system_monitor = SystemMonitor()
        
    async def initialize(self):
        """Initialize the training pipeline"""
        try:
            await self.data_loader.initialize()
            self.logger.info("Production ML Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def close(self):
        """Close resources"""
        try:
            await self.data_loader.close()
            if ray.is_initialized():
                ray.shutdown()
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    async def train_models(self, config: TrainingConfig) -> Dict[str, ModelResult]:
        """
        Train multiple models with the given configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary of model results keyed by model type
        """
        try:
            start_time = time.time()
            self.logger.info("Starting production ML training pipeline")
            
            # Set up MLflow experiment
            mlflow.set_experiment(config.experiment_name)
            
            with mlflow.start_run(run_name=config.run_name) as run:
                # Log configuration
                mlflow.log_params(asdict(config))
                
                # Load and preprocess data
                dataset = await self._load_and_preprocess_data(config)
                
                # Log dataset metadata
                self._log_dataset_metadata(dataset)
                
                # Initialize model results
                model_results = {}
                
                # Train models (parallel or sequential based on config)
                if config.use_distributed and len(config.model_types) > 1:
                    model_results = await self._train_models_distributed(config, dataset)
                else:
                    model_results = await self._train_models_sequential(config, dataset)
                
                # Model comparison and selection
                best_model = self._select_best_model(model_results, config)
                
                # Ensemble modeling if enabled
                if config.ensemble_models and len(model_results) > 1:
                    ensemble_result = await self._create_ensemble_model(model_results, config, dataset)
                    model_results['ensemble'] = ensemble_result
                
                # Log final results
                self._log_training_summary(model_results, best_model, run)
                
                # Auto-deployment if conditions are met
                if self._should_auto_deploy(best_model, config):
                    await self._auto_deploy_model(best_model, config)
                
                training_time = time.time() - start_time
                self.logger.info(f"Training pipeline completed in {training_time:.2f}s")
                
                return model_results
                
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise
    
    async def _load_and_preprocess_data(self, config: TrainingConfig) -> MLDataset:
        """Load and preprocess training data"""
        try:
            self.logger.info("Loading and preprocessing data")
            
            # Load raw dataset
            dataset = await self.data_loader.load_training_dataset(
                train_split=1.0 - config.validation_split - config.test_split,
                validation_split=config.validation_split,
                test_split=config.test_split,
                min_interactions=5,
                max_users=config.max_memory_gb * 1000,  # Scale based on memory
                max_properties=config.max_memory_gb * 500
            )
            
            # Feature engineering
            if config.feature_engineering:
                dataset = await self._engineer_features(dataset, config)
            
            # Data augmentation
            if config.data_augmentation:
                dataset = await self._augment_data(dataset, config)
            
            # Handle class imbalance
            if config.handle_imbalance:
                dataset = await self._handle_imbalance(dataset, config)
            
            # Feature selection
            if config.feature_selection:
                dataset = await self._select_features(dataset, config)
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    async def _engineer_features(self, dataset: MLDataset, config: TrainingConfig) -> MLDataset:
        """Advanced feature engineering"""
        try:
            # Time-based features
            current_time = datetime.utcnow()
            
            # User behavior features
            user_features = {}
            for user_idx, user_data in enumerate(dataset.train_data.user_metadata):
                user_features[user_idx] = {
                    'interaction_count': np.sum(dataset.train_data.user_item_matrix[user_idx] > 0),
                    'avg_rating': np.mean(dataset.train_data.user_item_matrix[user_idx][dataset.train_data.user_item_matrix[user_idx] > 0]) if np.any(dataset.train_data.user_item_matrix[user_idx] > 0) else 0,
                    'rating_variance': np.var(dataset.train_data.user_item_matrix[user_idx][dataset.train_data.user_item_matrix[user_idx] > 0]) if np.any(dataset.train_data.user_item_matrix[user_idx] > 0) else 0,
                    'account_age_days': (current_time - datetime.fromisoformat(user_data.get('created_at', current_time.isoformat()))).days if user_data.get('created_at') else 0
                }
            
            # Property features
            property_features = {}
            for prop_idx, prop_data in enumerate(dataset.train_data.property_metadata):
                property_features[prop_idx] = {
                    'popularity_score': np.sum(dataset.train_data.user_item_matrix[:, prop_idx] > 0),
                    'avg_rating': np.mean(dataset.train_data.user_item_matrix[:, prop_idx][dataset.train_data.user_item_matrix[:, prop_idx] > 0]) if np.any(dataset.train_data.user_item_matrix[:, prop_idx] > 0) else 0,
                    'price_per_sqft': prop_data.get('price', 0) / max(prop_data.get('sqft', 1), 1),
                    'listing_age_days': (current_time - datetime.fromisoformat(prop_data.get('created_at', current_time.isoformat()))).days if prop_data.get('created_at') else 0
                }
            
            # Add engineered features to dataset
            dataset.train_data.user_features = user_features
            dataset.train_data.property_features = property_features
            
            if dataset.validation_data:
                # Apply same feature engineering to validation data
                dataset.validation_data.user_features = user_features  # Reuse training features
                dataset.validation_data.property_features = property_features
            
            if dataset.test_data:
                # Apply same feature engineering to test data
                dataset.test_data.user_features = user_features
                dataset.test_data.property_features = property_features
            
            self.logger.info("Feature engineering completed")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return dataset
    
    async def _augment_data(self, dataset: MLDataset, config: TrainingConfig) -> MLDataset:
        """Data augmentation for improving model robustness"""
        try:
            # Noise injection for ratings
            noise_factor = 0.1
            augmented_matrix = dataset.train_data.user_item_matrix.copy()
            
            # Add Gaussian noise to non-zero ratings
            mask = augmented_matrix > 0
            noise = np.random.normal(0, noise_factor, augmented_matrix.shape)
            augmented_matrix[mask] += noise[mask]
            
            # Clip to valid rating range
            augmented_matrix = np.clip(augmented_matrix, 0, 1)
            
            # Combine original and augmented data
            combined_matrix = np.vstack([dataset.train_data.user_item_matrix, augmented_matrix])
            
            # Update dataset
            dataset.train_data.user_item_matrix = combined_matrix
            
            self.logger.info("Data augmentation completed")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Data augmentation failed: {e}")
            return dataset
    
    async def _handle_imbalance(self, dataset: MLDataset, config: TrainingConfig) -> MLDataset:
        """Handle class imbalance in recommendation data"""
        try:
            # Calculate interaction sparsity
            total_interactions = np.sum(dataset.train_data.user_item_matrix > 0)
            total_possible = dataset.train_data.user_item_matrix.shape[0] * dataset.train_data.user_item_matrix.shape[1]
            sparsity = 1 - (total_interactions / total_possible)
            
            self.logger.info(f"Data sparsity: {sparsity:.4f}")
            
            # For highly sparse data, use negative sampling
            if sparsity > 0.95:
                dataset = await self._apply_negative_sampling(dataset, num_negatives=5)
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Imbalance handling failed: {e}")
            return dataset
    
    async def _apply_negative_sampling(self, dataset: MLDataset, num_negatives: int = 5) -> MLDataset:
        """Apply negative sampling to balance positive and negative interactions"""
        try:
            user_item_matrix = dataset.train_data.user_item_matrix
            num_users, num_items = user_item_matrix.shape
            
            # Create negative samples
            negative_samples = []
            
            for user_idx in range(num_users):
                # Get positive items for this user
                positive_items = set(np.where(user_item_matrix[user_idx] > 0)[0])
                
                # Sample negative items
                all_items = set(range(num_items))
                negative_items = all_items - positive_items
                
                if len(negative_items) >= num_negatives:
                    sampled_negatives = np.random.choice(
                        list(negative_items), 
                        size=min(num_negatives, len(positive_items)), 
                        replace=False
                    )
                    
                    for item_idx in sampled_negatives:
                        negative_samples.append((user_idx, item_idx, 0.0))
            
            # Add negative samples to dataset
            for user_idx, item_idx, rating in negative_samples:
                user_item_matrix[user_idx, item_idx] = rating
            
            dataset.train_data.user_item_matrix = user_item_matrix
            
            self.logger.info(f"Added {len(negative_samples)} negative samples")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Negative sampling failed: {e}")
            return dataset
    
    async def _select_features(self, dataset: MLDataset, config: TrainingConfig) -> MLDataset:
        """Feature selection based on importance"""
        try:
            # This is a placeholder for advanced feature selection
            # In production, you would use techniques like:
            # - Mutual information
            # - Recursive feature elimination
            # - SHAP values
            # - Correlation analysis
            
            self.logger.info("Feature selection completed (placeholder)")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return dataset
    
    async def _train_models_sequential(self, config: TrainingConfig, dataset: MLDataset) -> Dict[str, ModelResult]:
        """Train models sequentially"""
        try:
            model_results = {}
            
            for model_type in config.model_types:
                self.logger.info(f"Training {model_type} model")
                
                # Train model with hyperparameter optimization if enabled
                if config.use_hyperopt:
                    result = await self._train_with_hyperopt(model_type, config, dataset)
                else:
                    result = await self._train_single_model(model_type, config, dataset)
                
                model_results[model_type] = result
                
                # Log intermediate results
                mlflow.log_metrics({
                    f"{model_type}_val_{config.primary_metric}": result.validation_metrics.get(config.primary_metric, 0),
                    f"{model_type}_training_time": result.training_time
                })
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Sequential training failed: {e}")
            raise
    
    async def _train_models_distributed(self, config: TrainingConfig, dataset: MLDataset) -> Dict[str, ModelResult]:
        """Train models in parallel using Ray"""
        try:
            # Create Ray tasks for each model
            training_tasks = []
            
            for model_type in config.model_types:
                if config.use_hyperopt:
                    task = self._train_with_hyperopt_remote.remote(self, model_type, config, dataset)
                else:
                    task = self._train_single_model_remote.remote(self, model_type, config, dataset)
                
                training_tasks.append((model_type, task))
            
            # Wait for all tasks to complete
            model_results = {}
            for model_type, task in training_tasks:
                try:
                    result = await task
                    model_results[model_type] = result
                    self.logger.info(f"Completed training {model_type} model")
                except Exception as e:
                    self.logger.error(f"Failed to train {model_type} model: {e}")
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Distributed training failed: {e}")
            raise
    
    @ray.remote
    def _train_single_model_remote(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> ModelResult:
        """Remote wrapper for single model training"""
        return self._train_single_model(model_type, config, dataset)
    
    @ray.remote  
    def _train_with_hyperopt_remote(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> ModelResult:
        """Remote wrapper for hyperparameter optimization training"""
        return self._train_with_hyperopt(model_type, config, dataset)
    
    async def _train_single_model(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> ModelResult:
        """Train a single model without hyperparameter optimization"""
        try:
            start_time = time.time()
            
            # Initialize model
            model = self._create_model(model_type, config, dataset)
            
            # Train model
            training_history = await self._train_model_instance(model, model_type, config, dataset)
            
            # Evaluate model
            validation_metrics = await self._evaluate_model(model, model_type, dataset.validation_data, config)
            test_metrics = await self._evaluate_model(model, model_type, dataset.test_data, config) if dataset.test_data else None
            
            # Cross-validation if enabled
            cv_metrics = None
            if config.use_cross_validation:
                cv_metrics = await self._cross_validate_model(model_type, config, dataset)
            
            # Save model
            model_path = await self._save_model(model, model_type, config)
            
            # Calculate model size
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024) if Path(model_path).exists() else 0
            
            # Extract feature importance if available
            feature_importance = self._extract_feature_importance(model, model_type)
            
            training_time = time.time() - start_time
            
            # Create result
            result = ModelResult(
                model_type=model_type,
                model_variant='default',
                training_metrics=training_history,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                cv_metrics=cv_metrics,
                training_time=training_time,
                model_path=model_path,
                model_size_mb=model_size_mb,
                hyperparameters=self._get_model_hyperparameters(config),
                feature_importance=feature_importance,
                config=config,
                metadata={
                    'training_start': start_time,
                    'training_end': time.time(),
                    'system_info': self.system_monitor.get_system_info()
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Single model training failed for {model_type}: {e}")
            raise
    
    async def _train_with_hyperopt(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> ModelResult:
        """Train model with hyperparameter optimization"""
        try:
            self.logger.info(f"Starting hyperparameter optimization for {model_type}")
            
            if config.hyperopt_algorithm == 'optuna':
                return await self._train_with_optuna(model_type, config, dataset)
            elif config.hyperopt_algorithm == 'ray_tune':
                return await self._train_with_ray_tune(model_type, config, dataset)
            else:
                raise ValueError(f"Unknown hyperopt algorithm: {config.hyperopt_algorithm}")
                
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed for {model_type}: {e}")
            raise
    
    async def _train_with_optuna(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> ModelResult:
        """Train model using Optuna for hyperparameter optimization"""
        try:
            def objective(trial):
                # Define hyperparameter search space
                hyperparams = self._get_optuna_hyperparams(trial, model_type)
                
                # Create and train model with suggested hyperparameters
                temp_config = self._create_temp_config(config, hyperparams)
                model = self._create_model(model_type, temp_config, dataset)
                
                # Train model
                training_history = asyncio.run(self._train_model_instance(model, model_type, temp_config, dataset))
                
                # Evaluate model
                val_metrics = asyncio.run(self._evaluate_model(model, model_type, dataset.validation_data, temp_config))
                
                # Return primary metric for optimization
                return val_metrics.get(config.primary_metric, float('inf'))
            
            # Create study
            direction = 'minimize' if config.metric_direction == 'minimize' else 'maximize'
            study = optuna.create_study(direction=direction)
            
            # Optimize
            study.optimize(objective, n_trials=config.hyperopt_trials)
            
            # Train final model with best hyperparameters
            best_params = study.best_params
            best_config = self._create_temp_config(config, best_params)
            
            return await self._train_single_model(model_type, best_config, dataset)
            
        except Exception as e:
            self.logger.error(f"Optuna optimization failed: {e}")
            raise
    
    async def _train_with_ray_tune(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> ModelResult:
        """Train model using Ray Tune for hyperparameter optimization"""
        try:
            # Define search space
            search_space = self._get_ray_tune_search_space(model_type)
            
            # Define training function
            def train_fn(hyperparams):
                temp_config = self._create_temp_config(config, hyperparams)
                model = self._create_model(model_type, temp_config, dataset)
                
                # Train and evaluate
                training_history = asyncio.run(self._train_model_instance(model, model_type, temp_config, dataset))
                val_metrics = asyncio.run(self._evaluate_model(model, model_type, dataset.validation_data, temp_config))
                
                # Report metric to Tune
                tune.report(**val_metrics)
            
            # Configure scheduler
            scheduler = ASHAScheduler(
                metric=config.primary_metric,
                mode=config.metric_direction,
                max_t=config.epochs,
                grace_period=10
            )
            
            # Run optimization
            analysis = tune.run(
                train_fn,
                config=search_space,
                num_samples=config.hyperopt_trials,
                scheduler=scheduler,
                resources_per_trial={"cpu": 2, "gpu": 0.5}
            )
            
            # Get best configuration
            best_config_dict = analysis.best_config
            best_config = self._create_temp_config(config, best_config_dict)
            
            return await self._train_single_model(model_type, best_config, dataset)
            
        except Exception as e:
            self.logger.error(f"Ray Tune optimization failed: {e}")
            raise
    
    def _create_model(self, model_type: str, config: TrainingConfig, dataset: MLDataset):
        """Create model instance based on type and configuration"""
        try:
            if model_type == 'collaborative':
                return CollaborativeFilteringModel(
                    num_users=dataset.train_data.user_item_matrix.shape[0],
                    num_items=dataset.train_data.user_item_matrix.shape[1],
                    embedding_dim=config.embedding_dims[0] if config.embedding_dims else 128,
                    reg_lambda=config.weight_decay
                )
            elif model_type == 'content':
                return ContentBasedRecommender(
                    embedding_dim=config.embedding_dims[0] if config.embedding_dims else 128,
                    reg_lambda=config.weight_decay,
                    learning_rate=config.learning_rate
                )
            elif model_type == 'hybrid':
                hybrid_system = HybridRecommendationSystem(
                    cf_weight=0.6,
                    cb_weight=0.4,
                    min_cf_interactions=5
                )
                hybrid_system.initialize_models(
                    num_users=dataset.train_data.user_item_matrix.shape[0],
                    num_items=dataset.train_data.user_item_matrix.shape[1],
                    cf_embedding_dim=config.embedding_dims[0] if config.embedding_dims else 128,
                    cb_embedding_dim=config.embedding_dims[1] if len(config.embedding_dims or []) > 1 else 128
                )
                return hybrid_system
            elif model_type == 'search_ranker':
                return NLPSearchRanker()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Model creation failed for {model_type}: {e}")
            raise
    
    async def _train_model_instance(self, model, model_type: str, config: TrainingConfig, dataset: MLDataset) -> Dict[str, float]:
        """Train a specific model instance"""
        try:
            if model_type in ['collaborative', 'content']:
                # Prepare property data for content-based model
                property_data = []
                if model_type == 'content':
                    for prop_meta in dataset.train_data.property_metadata:
                        property_data.append({
                            'title': prop_meta.get('title', ''),
                            'location': prop_meta.get('location', ''),
                            'price': prop_meta.get('price', 0),
                            'bedrooms': prop_meta.get('bedrooms', 0),
                            'bathrooms': prop_meta.get('bathrooms', 0),
                            'amenities': prop_meta.get('amenities', []),
                            'property_type': prop_meta.get('property_type', 'apartment')
                        })
                
                # Train model
                if model_type == 'content' and property_data:
                    training_history = model.fit(
                        user_item_matrix=dataset.train_data.user_item_matrix,
                        property_data=property_data,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_split=0.0  # We handle validation separately
                    )
                else:
                    training_history = model.fit(
                        user_item_matrix=dataset.train_data.user_item_matrix,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_split=0.0
                    )
                
                return training_history
                
            elif model_type == 'hybrid':
                # Train hybrid model
                property_data = []
                for prop_meta in dataset.train_data.property_metadata:
                    property_data.append({
                        'title': prop_meta.get('title', ''),
                        'location': prop_meta.get('location', ''),
                        'price': prop_meta.get('price', 0),
                        'bedrooms': prop_meta.get('bedrooms', 0),
                        'bathrooms': prop_meta.get('bathrooms', 0),
                        'amenities': prop_meta.get('amenities', []),
                        'property_type': prop_meta.get('property_type', 'apartment')
                    })
                
                training_results = model.fit(
                    user_item_matrix=dataset.train_data.user_item_matrix,
                    property_data=property_data,
                    cf_epochs=config.epochs,
                    cb_epochs=config.epochs,
                    cf_batch_size=config.batch_size,
                    cb_batch_size=config.batch_size,
                    validation_split=0.0
                )
                
                return training_results
                
            elif model_type == 'search_ranker':
                # Generate synthetic search training data
                training_data = self._generate_search_training_data(dataset)
                
                if training_data:
                    split_idx = int(len(training_data) * 0.8)
                    train_data = training_data[:split_idx]
                    
                    training_metrics = model.train(
                        training_data=train_data,
                        validation_data=[],
                        epochs=min(config.epochs, 20),
                        batch_size=config.batch_size
                    )
                    
                    return training_metrics
                else:
                    return {}
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    def _generate_search_training_data(self, dataset: MLDataset) -> List[Tuple[str, Dict, float]]:
        """Generate synthetic search training data from property metadata"""
        try:
            training_data = []
            
            for prop_meta in dataset.train_data.property_metadata[:200]:  # Limit for efficiency
                location = prop_meta.get('location', '').split(',')[0]
                bedrooms = prop_meta.get('bedrooms', 0)
                
                if location:
                    # High relevance: exact location match
                    training_data.append((
                        f"apartment in {location}",
                        prop_meta,
                        1.0
                    ))
                    
                    # Medium relevance: bedroom match
                    training_data.append((
                        f"{bedrooms} bedroom apartment",
                        prop_meta,
                        0.7
                    ))
                    
                    # Low relevance: general query
                    training_data.append((
                        "cheap apartment",
                        prop_meta,
                        0.3 if prop_meta.get('price', 0) < 2000 else 0.1
                    ))
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate search training data: {e}")
            return []
    
    async def _evaluate_model(self, model, model_type: str, test_data, config: TrainingConfig) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if not test_data:
                return {}
            
            predictions = []
            actuals = []
            
            # Sample for efficiency
            sample_size = min(100, test_data.user_item_matrix.shape[0])
            
            for user_idx in range(sample_size):
                user_items = np.where(test_data.user_item_matrix[user_idx] > 0)[0]
                if len(user_items) > 0:
                    sample_items = user_items[:10]  # Limit items per user
                    
                    if model_type == 'hybrid':
                        preds = model.predict(user_idx, sample_items.tolist())
                    else:
                        preds = model.predict(user_idx, sample_items.tolist())
                    
                    if len(preds) > 0:
                        actuals_user = test_data.user_item_matrix[user_idx, sample_items]
                        predictions.extend(preds)
                        actuals.extend(actuals_user)
            
            if not predictions:
                return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            mse = float(mean_squared_error(actuals, predictions))
            mae = float(mean_absolute_error(actuals, predictions))
            rmse = float(np.sqrt(mse))
            
            # Additional metrics for recommendations
            precision_at_5 = self._calculate_precision_at_k(actuals, predictions, k=5)
            recall_at_5 = self._calculate_recall_at_k(actuals, predictions, k=5)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'precision_at_5': precision_at_5,
                'recall_at_5': recall_at_5
            }
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
    
    def _calculate_precision_at_k(self, actuals: np.ndarray, predictions: np.ndarray, k: int = 5) -> float:
        """Calculate precision@k metric"""
        try:
            # Get top-k predictions
            top_k_indices = np.argsort(predictions)[-k:]
            
            # Count relevant items in top-k
            relevant_in_top_k = np.sum(actuals[top_k_indices] > 0.5)
            
            return float(relevant_in_top_k / k)
            
        except Exception:
            return 0.0
    
    def _calculate_recall_at_k(self, actuals: np.ndarray, predictions: np.ndarray, k: int = 5) -> float:
        """Calculate recall@k metric"""
        try:
            # Get top-k predictions
            top_k_indices = np.argsort(predictions)[-k:]
            
            # Count total relevant items
            total_relevant = np.sum(actuals > 0.5)
            if total_relevant == 0:
                return 0.0
            
            # Count relevant items in top-k
            relevant_in_top_k = np.sum(actuals[top_k_indices] > 0.5)
            
            return float(relevant_in_top_k / total_relevant)
            
        except Exception:
            return 0.0
    
    async def _cross_validate_model(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> Dict[str, float]:
        """Perform cross-validation"""
        try:
            if config.cv_strategy == 'time_series':
                return await self._time_series_cv(model_type, config, dataset)
            else:
                return await self._kfold_cv(model_type, config, dataset)
                
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {}
    
    async def _kfold_cv(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> Dict[str, float]:
        """K-fold cross-validation"""
        try:
            user_item_matrix = dataset.train_data.user_item_matrix
            num_users = user_item_matrix.shape[0]
            
            kfold = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
            
            cv_scores = []
            
            for fold, (train_indices, val_indices) in enumerate(kfold.split(range(num_users))):
                # Create fold-specific data
                fold_train_matrix = user_item_matrix[train_indices]
                fold_val_matrix = user_item_matrix[val_indices]
                
                # Create model for this fold
                fold_config = self._create_temp_config(config, {})
                model = self._create_model(model_type, fold_config, dataset)
                
                # Simplified training for CV
                if model_type in ['collaborative', 'content']:
                    model.fit(
                        user_item_matrix=fold_train_matrix,
                        epochs=config.epochs // 2,  # Fewer epochs for CV
                        batch_size=config.batch_size
                    )
                
                # Evaluate fold
                fold_scores = await self._evaluate_fold(model, model_type, fold_val_matrix, config)
                cv_scores.append(fold_scores)
            
            # Aggregate CV results
            aggregated_scores = {}
            if cv_scores:
                for metric in cv_scores[0].keys():
                    values = [score[metric] for score in cv_scores if metric in score]
                    aggregated_scores[f"cv_{metric}_mean"] = float(np.mean(values))
                    aggregated_scores[f"cv_{metric}_std"] = float(np.std(values))
            
            return aggregated_scores
            
        except Exception as e:
            self.logger.error(f"K-fold CV failed: {e}")
            return {}
    
    async def _time_series_cv(self, model_type: str, config: TrainingConfig, dataset: MLDataset) -> Dict[str, float]:
        """Time series cross-validation"""
        try:
            # This is a simplified version - in production you would use actual timestamps
            user_item_matrix = dataset.train_data.user_item_matrix
            
            tscv = TimeSeriesSplit(n_splits=config.cv_folds)
            
            cv_scores = []
            
            for train_indices, val_indices in tscv.split(user_item_matrix):
                # Time series split on users
                fold_train_matrix = user_item_matrix[train_indices]
                fold_val_matrix = user_item_matrix[val_indices]
                
                # Create and train model
                fold_config = self._create_temp_config(config, {})
                model = self._create_model(model_type, fold_config, dataset)
                
                if model_type in ['collaborative', 'content']:
                    model.fit(
                        user_item_matrix=fold_train_matrix,
                        epochs=config.epochs // 2,
                        batch_size=config.batch_size
                    )
                
                # Evaluate
                fold_scores = await self._evaluate_fold(model, model_type, fold_val_matrix, config)
                cv_scores.append(fold_scores)
            
            # Aggregate results
            aggregated_scores = {}
            if cv_scores:
                for metric in cv_scores[0].keys():
                    values = [score[metric] for score in cv_scores if metric in score]
                    aggregated_scores[f"ts_cv_{metric}_mean"] = float(np.mean(values))
                    aggregated_scores[f"ts_cv_{metric}_std"] = float(np.std(values))
            
            return aggregated_scores
            
        except Exception as e:
            self.logger.error(f"Time series CV failed: {e}")
            return {}
    
    async def _evaluate_fold(self, model, model_type: str, val_matrix: np.ndarray, config: TrainingConfig) -> Dict[str, float]:
        """Evaluate model on a CV fold"""
        try:
            predictions = []
            actuals = []
            
            # Sample users for efficiency
            sample_size = min(50, val_matrix.shape[0])
            
            for user_idx in range(sample_size):
                user_items = np.where(val_matrix[user_idx] > 0)[0]
                if len(user_items) > 0:
                    sample_items = user_items[:5]  # Small sample
                    preds = model.predict(user_idx, sample_items.tolist())
                    actuals_user = val_matrix[user_idx, sample_items]
                    
                    predictions.extend(preds)
                    actuals.extend(actuals_user)
            
            if not predictions:
                return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            return {
                'mse': float(mean_squared_error(actuals, predictions)),
                'mae': float(mean_absolute_error(actuals, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(actuals, predictions)))
            }
            
        except Exception as e:
            self.logger.error(f"Fold evaluation failed: {e}")
            return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
    
    async def _save_model(self, model, model_type: str, config: TrainingConfig) -> str:
        """Save trained model"""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_type}_model_{timestamp}.h5"
            model_path = self.models_dir / model_filename
            
            if hasattr(model, 'save_model'):
                model.save_model(str(model_path))
            else:
                # Fallback to pickle for models without save_model method
                with open(str(model_path).replace('.h5', '.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                model_path = str(model_path).replace('.h5', '.pkl')
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            return ""
    
    def _extract_feature_importance(self, model, model_type: str) -> Optional[Dict[str, float]]:
        """Extract feature importance from model"""
        try:
            # This is model-specific and would need to be implemented
            # based on the specific model architectures
            return None
            
        except Exception as e:
            self.logger.warning(f"Feature importance extraction failed: {e}")
            return None
    
    def _get_model_hyperparameters(self, config: TrainingConfig) -> Dict[str, Any]:
        """Get hyperparameters from config"""
        return {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'dropout_rate': config.dropout_rate,
            'embedding_dims': config.embedding_dims,
            'optimizer': config.optimizer,
            'activation': config.activation
        }
    
    def _get_optuna_hyperparams(self, trial, model_type: str) -> Dict[str, Any]:
        """Define Optuna hyperparameter search space"""
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        }
        
        if model_type in ['collaborative', 'content', 'hybrid']:
            hyperparams['embedding_dim'] = trial.suggest_categorical('embedding_dim', [64, 128, 256, 512])
        
        return hyperparams
    
    def _get_ray_tune_search_space(self, model_type: str) -> Dict[str, Any]:
        """Define Ray Tune search space"""
        search_space = {
            'learning_rate': tune.loguniform(1e-5, 1e-1),
            'batch_size': tune.choice([64, 128, 256, 512]),
            'weight_decay': tune.loguniform(1e-6, 1e-3),
            'dropout_rate': tune.uniform(0.1, 0.5),
        }
        
        if model_type in ['collaborative', 'content', 'hybrid']:
            search_space['embedding_dim'] = tune.choice([64, 128, 256, 512])
        
        return search_space
    
    def _create_temp_config(self, base_config: TrainingConfig, hyperparams: Dict[str, Any]) -> TrainingConfig:
        """Create temporary config with updated hyperparameters"""
        config_dict = asdict(base_config)
        config_dict.update(hyperparams)
        
        # Handle embedding_dims special case
        if 'embedding_dim' in hyperparams:
            config_dict['embedding_dims'] = [hyperparams['embedding_dim']] * 2
        
        return TrainingConfig(**config_dict)
    
    def _select_best_model(self, model_results: Dict[str, ModelResult], config: TrainingConfig) -> ModelResult:
        """Select best model based on validation metrics"""
        try:
            best_model = None
            best_score = float('inf') if config.metric_direction == 'minimize' else float('-inf')
            
            for model_type, result in model_results.items():
                metric_value = result.validation_metrics.get(config.primary_metric, 
                                                           float('inf') if config.metric_direction == 'minimize' else float('-inf'))
                
                if config.metric_direction == 'minimize':
                    if metric_value < best_score:
                        best_score = metric_value
                        best_model = result
                else:
                    if metric_value > best_score:
                        best_score = metric_value
                        best_model = result
            
            if best_model:
                self.logger.info(f"Best model: {best_model.model_type} with {config.primary_metric}={best_score:.4f}")
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            return list(model_results.values())[0] if model_results else None
    
    async def _create_ensemble_model(self, model_results: Dict[str, ModelResult], config: TrainingConfig, dataset: MLDataset) -> ModelResult:
        """Create ensemble model from multiple trained models"""
        try:
            self.logger.info("Creating ensemble model")
            
            # Simple ensemble approach - weighted average based on validation performance
            ensemble_weights = {}
            total_weight = 0
            
            for model_type, result in model_results.items():
                # Weight based on inverse of primary metric (for minimization metrics)
                if config.metric_direction == 'minimize':
                    weight = 1.0 / (result.validation_metrics.get(config.primary_metric, 1.0) + 1e-8)
                else:
                    weight = result.validation_metrics.get(config.primary_metric, 0.0)
                
                ensemble_weights[model_type] = weight
                total_weight += weight
            
            # Normalize weights
            for model_type in ensemble_weights:
                ensemble_weights[model_type] /= total_weight
            
            # Create ensemble metadata
            ensemble_result = ModelResult(
                model_type='ensemble',
                model_variant='weighted_average',
                training_metrics={},
                validation_metrics={},
                test_metrics=None,
                cv_metrics=None,
                training_time=sum(result.training_time for result in model_results.values()),
                model_path='',  # Ensemble doesn't have a single model file
                model_size_mb=sum(result.model_size_mb for result in model_results.values()),
                hyperparameters={'ensemble_weights': ensemble_weights},
                feature_importance=None,
                config=config,
                metadata={
                    'component_models': list(model_results.keys()),
                    'ensemble_weights': ensemble_weights
                }
            )
            
            # Evaluate ensemble (simplified)
            ensemble_val_metrics = {}
            for metric in ['mse', 'mae', 'rmse']:
                if all(metric in result.validation_metrics for result in model_results.values()):
                    weighted_score = sum(
                        ensemble_weights[model_type] * result.validation_metrics[metric]
                        for model_type, result in model_results.items()
                    )
                    ensemble_val_metrics[metric] = weighted_score
            
            ensemble_result.validation_metrics = ensemble_val_metrics
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            raise
    
    def _log_dataset_metadata(self, dataset: MLDataset):
        """Log dataset metadata to MLflow"""
        try:
            mlflow.log_metrics({
                "dataset_users": dataset.metadata['total_users'],
                "dataset_properties": dataset.metadata['total_properties'], 
                "dataset_interactions": dataset.metadata['total_interactions'],
                "data_sparsity": dataset.metadata['data_quality']['interaction_sparsity']
            })
        except Exception as e:
            self.logger.warning(f"Failed to log dataset metadata: {e}")
    
    def _log_training_summary(self, model_results: Dict[str, ModelResult], best_model: ModelResult, mlflow_run):
        """Log training summary to MLflow"""
        try:
            # Log final metrics for all models
            for model_type, result in model_results.items():
                mlflow.log_metrics({
                    f"{model_type}_final_val_{self.config.primary_metric}": result.validation_metrics.get(self.config.primary_metric, 0),
                    f"{model_type}_training_time": result.training_time,
                    f"{model_type}_model_size_mb": result.model_size_mb
                })
                
                # Log model artifact
                if result.model_path:
                    mlflow.log_artifact(result.model_path)
            
            # Log best model info
            if best_model:
                mlflow.log_metrics({
                    "best_model_score": best_model.validation_metrics.get(self.config.primary_metric, 0),
                    "total_training_time": sum(result.training_time for result in model_results.values())
                })
                
                mlflow.set_tag("best_model_type", best_model.model_type)
                
        except Exception as e:
            self.logger.warning(f"Failed to log training summary: {e}")
    
    def _should_auto_deploy(self, best_model: ModelResult, config: TrainingConfig) -> bool:
        """Determine if model should be automatically deployed"""
        try:
            if not best_model:
                return False
            
            # Check if improvement threshold is met
            current_baseline = self._get_current_baseline_score(config.primary_metric)
            if current_baseline is None:
                return True  # No baseline, deploy first model
            
            best_score = best_model.validation_metrics.get(config.primary_metric, float('inf'))
            
            if config.metric_direction == 'minimize':
                improvement = (current_baseline - best_score) / current_baseline
            else:
                improvement = (best_score - current_baseline) / current_baseline
            
            return improvement >= config.auto_deploy_threshold
            
        except Exception as e:
            self.logger.warning(f"Auto-deploy check failed: {e}")
            return False
    
    def _get_current_baseline_score(self, metric_name: str) -> Optional[float]:
        """Get current production model baseline score"""
        # This would query the model registry for current production model performance
        # For now, return None (no baseline)
        return None
    
    async def _auto_deploy_model(self, best_model: ModelResult, config: TrainingConfig):
        """Automatically deploy the best model"""
        try:
            self.logger.info(f"Auto-deploying {best_model.model_type} model")
            
            # Register model in MLflow Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model.model_type}_model"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=f"rental_ml_{best_model.model_type}",
                tags={
                    "auto_deployed": "true",
                    "deployment_time": datetime.utcnow().isoformat(),
                    "validation_score": str(best_model.validation_metrics.get(config.primary_metric, 0))
                }
            )
            
            # Transition to staging stage
            mlflow.tracking.MlflowClient().transition_model_version_stage(
                name=registered_model.name,
                version=registered_model.version,
                stage=config.model_registry_stage
            )
            
            self.logger.info(f"Model {registered_model.name} v{registered_model.version} deployed to {config.model_registry_stage}")
            
        except Exception as e:
            self.logger.error(f"Auto-deployment failed: {e}")


class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'gpu_count': len(tf.config.list_physical_devices('GPU')),
                'tensorflow_version': tf.__version__
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {}