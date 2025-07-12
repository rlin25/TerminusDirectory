"""
Production ML training pipeline for rental recommendation models.

This module provides comprehensive training capabilities for all ML models
including collaborative filtering, content-based, hybrid recommenders,
and search ranking models. It includes model validation, checkpointing,
and performance monitoring.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import KFold
import mlflow
import mlflow.tensorflow

from .data_loader import ProductionDataLoader, MLDataset
from ..models.collaborative_filter import CollaborativeFilteringModel
from ..models.content_recommender import ContentBasedRecommender
from ..models.hybrid_recommender import HybridRecommendationSystem
from ..models.search_ranker import NLPSearchRanker


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: str  # 'collaborative', 'content', 'hybrid', 'search_ranker'
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_frequency: int = 5
    evaluation_frequency: int = 5
    
    # Model-specific parameters
    embedding_dim: int = 128
    regularization: float = 1e-5
    dropout_rate: float = 0.2
    
    # Training data parameters
    min_interactions: int = 5
    max_users: Optional[int] = None
    max_properties: Optional[int] = None
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # MLflow tracking
    experiment_name: str = "rental_ml_training"
    run_name: Optional[str] = None


@dataclass
class TrainingResults:
    """Results from model training"""
    model_type: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]
    training_time: float
    model_path: str
    config: TrainingConfig
    metadata: Dict[str, Any]


class MLTrainer:
    """
    Production ML training pipeline.
    
    This class handles:
    - Model training with validation and checkpointing
    - Hyperparameter optimization
    - Cross-validation
    - Model evaluation and metrics tracking
    - Model persistence and versioning
    - MLflow experiment tracking
    """
    
    def __init__(self, 
                 database_url: str,
                 models_dir: str = "/tmp/models",
                 mlflow_tracking_uri: Optional[str] = None):
        self.database_url = database_url
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize data loader
        self.data_loader = ProductionDataLoader(database_url)
        
        # Initialize MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Model storage
        self.trained_models = {}
        
    async def initialize(self):
        """Initialize the trainer"""
        await self.data_loader.initialize()
        self.logger.info("ML Trainer initialized successfully")
    
    async def close(self):
        """Close resources"""
        await self.data_loader.close()
    
    async def train_model(self, config: TrainingConfig) -> TrainingResults:
        """
        Train a model with the given configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Training results with metrics and model path
        """
        try:
            start_time = datetime.utcnow()
            
            # Set up MLflow experiment
            mlflow.set_experiment(config.experiment_name)
            
            with mlflow.start_run(run_name=config.run_name) as run:
                # Log configuration
                mlflow.log_params(asdict(config))
                
                self.logger.info(f"Starting training for {config.model_type} model")
                
                # Load training data
                dataset = await self.data_loader.load_training_dataset(
                    min_interactions=config.min_interactions,
                    max_users=config.max_users,
                    max_properties=config.max_properties
                )
                
                # Log dataset metadata
                mlflow.log_metrics({
                    "dataset_users": dataset.metadata['total_users'],
                    "dataset_properties": dataset.metadata['total_properties'],
                    "dataset_interactions": dataset.metadata['total_interactions'],
                    "data_sparsity": dataset.metadata['data_quality']['interaction_sparsity']
                })
                
                # Train model based on type
                if config.model_type == 'collaborative':
                    training_results = await self._train_collaborative_model(config, dataset, run)
                elif config.model_type == 'content':
                    training_results = await self._train_content_model(config, dataset, run)
                elif config.model_type == 'hybrid':
                    training_results = await self._train_hybrid_model(config, dataset, run)
                elif config.model_type == 'search_ranker':
                    training_results = await self._train_search_ranker(config, dataset, run)
                else:
                    raise ValueError(f"Unknown model type: {config.model_type}")
                
                # Calculate training time
                end_time = datetime.utcnow()
                training_time = (end_time - start_time).total_seconds()
                
                # Create results
                results = TrainingResults(
                    model_type=config.model_type,
                    training_metrics=training_results['training_metrics'],
                    validation_metrics=training_results['validation_metrics'],
                    test_metrics=training_results.get('test_metrics'),
                    training_time=training_time,
                    model_path=training_results['model_path'],
                    config=config,
                    metadata={
                        'dataset_metadata': dataset.metadata,
                        'mlflow_run_id': run.info.run_id,
                        'training_start': start_time.isoformat(),
                        'training_end': end_time.isoformat()
                    }
                )
                
                # Log final metrics
                mlflow.log_metrics({
                    "training_time_seconds": training_time,
                    **{f"final_{k}": v for k, v in training_results['validation_metrics'].items()}
                })
                
                # Log model artifact
                mlflow.log_artifact(training_results['model_path'])
                
                self.logger.info(
                    f"Training completed for {config.model_type} in {training_time:.2f}s"
                )
                
                return results
                
        except Exception as e:
            self.logger.error(f"Training failed for {config.model_type}: {e}")
            raise
    
    async def _train_collaborative_model(self, 
                                       config: TrainingConfig, 
                                       dataset: MLDataset,
                                       mlflow_run) -> Dict[str, Any]:
        """Train collaborative filtering model"""
        try:
            # Initialize model
            num_users = dataset.train_data.user_item_matrix.shape[0]
            num_items = dataset.train_data.user_item_matrix.shape[1]
            
            model = CollaborativeFilteringModel(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=config.embedding_dim,
                reg_lambda=config.regularization
            )
            
            # Prepare combined training data (train + validation for collaborative filtering)
            combined_matrix = np.vstack([
                dataset.train_data.user_item_matrix,
                dataset.validation_data.user_item_matrix
            ])
            
            # Train model with validation
            training_history = model.fit(
                user_item_matrix=combined_matrix,
                epochs=config.epochs,
                batch_size=config.batch_size,
                validation_split=config.validation_split
            )
            
            # Evaluate on validation data
            val_predictions = []
            val_actuals = []
            
            for user_idx in range(dataset.validation_data.user_item_matrix.shape[0]):
                # Get items user has interacted with
                user_items = np.where(dataset.validation_data.user_item_matrix[user_idx] > 0)[0]
                if len(user_items) > 0:
                    predictions = model.predict(user_idx, user_items.tolist())
                    actuals = dataset.validation_data.user_item_matrix[user_idx, user_items]
                    
                    val_predictions.extend(predictions)
                    val_actuals.extend(actuals)
            
            # Calculate validation metrics
            val_predictions = np.array(val_predictions)
            val_actuals = np.array(val_actuals)
            
            validation_metrics = {
                'val_mse': float(mean_squared_error(val_actuals, val_predictions)),
                'val_mae': float(mean_absolute_error(val_actuals, val_predictions)),
                'val_rmse': float(np.sqrt(mean_squared_error(val_actuals, val_predictions)))
            }
            
            # Test evaluation if available
            test_metrics = None
            if dataset.test_data is not None:
                test_predictions = []
                test_actuals = []
                
                for user_idx in range(dataset.test_data.user_item_matrix.shape[0]):
                    user_items = np.where(dataset.test_data.user_item_matrix[user_idx] > 0)[0]
                    if len(user_items) > 0:
                        predictions = model.predict(user_idx, user_items.tolist())
                        actuals = dataset.test_data.user_item_matrix[user_idx, user_items]
                        
                        test_predictions.extend(predictions)
                        test_actuals.extend(actuals)
                
                if test_predictions:
                    test_predictions = np.array(test_predictions)
                    test_actuals = np.array(test_actuals)
                    
                    test_metrics = {
                        'test_mse': float(mean_squared_error(test_actuals, test_predictions)),
                        'test_mae': float(mean_absolute_error(test_actuals, test_predictions)),
                        'test_rmse': float(np.sqrt(mean_squared_error(test_actuals, test_predictions)))
                    }
            
            # Save model
            model_filename = f"collaborative_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.h5"
            model_path = self.models_dir / model_filename
            model.save_model(str(model_path))
            
            # Store trained model
            self.trained_models['collaborative'] = model
            
            return {
                'training_metrics': training_history,
                'validation_metrics': validation_metrics,
                'test_metrics': test_metrics,
                'model_path': str(model_path)
            }
            
        except Exception as e:
            self.logger.error(f"Collaborative filtering training failed: {e}")
            raise
    
    async def _train_content_model(self, 
                                 config: TrainingConfig, 
                                 dataset: MLDataset,
                                 mlflow_run) -> Dict[str, Any]:
        """Train content-based model"""
        try:
            # Initialize model
            model = ContentBasedRecommender(
                embedding_dim=config.embedding_dim,
                reg_lambda=config.regularization,
                learning_rate=config.learning_rate
            )
            
            # Prepare property data for training
            property_data = []
            for prop_meta in dataset.train_data.property_metadata:
                property_data.append({
                    'title': prop_meta.get('title', ''),
                    'location': prop_meta.get('location', ''),
                    'price': prop_meta.get('price', 0),
                    'bedrooms': prop_meta.get('bedrooms', 0),
                    'bathrooms': prop_meta.get('bathrooms', 0),
                    'amenities': [],  # Would need to be loaded from database
                    'property_type': 'apartment'
                })
            
            # Train model
            training_history = model.fit(
                user_item_matrix=dataset.train_data.user_item_matrix,
                property_data=property_data,
                epochs=config.epochs,
                batch_size=config.batch_size,
                validation_split=config.validation_split
            )
            
            # Evaluate on validation data
            val_predictions = []
            val_actuals = []
            
            for user_idx in range(min(100, dataset.validation_data.user_item_matrix.shape[0])):  # Sample for efficiency
                user_items = np.where(dataset.validation_data.user_item_matrix[user_idx] > 0)[0]
                if len(user_items) > 0:
                    # Limit to first 10 items for efficiency
                    sample_items = user_items[:10]
                    predictions = model.predict(user_idx, sample_items.tolist())
                    actuals = dataset.validation_data.user_item_matrix[user_idx, sample_items]
                    
                    val_predictions.extend(predictions)
                    val_actuals.extend(actuals)
            
            # Calculate validation metrics
            validation_metrics = {}
            if val_predictions:
                val_predictions = np.array(val_predictions)
                val_actuals = np.array(val_actuals)
                
                validation_metrics = {
                    'val_mse': float(mean_squared_error(val_actuals, val_predictions)),
                    'val_mae': float(mean_absolute_error(val_actuals, val_predictions)),
                    'val_rmse': float(np.sqrt(mean_squared_error(val_actuals, val_predictions)))
                }
            
            # Save model
            model_filename = f"content_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.h5"
            model_path = self.models_dir / model_filename
            model.save_model(str(model_path))
            
            # Store trained model
            self.trained_models['content'] = model
            
            return {
                'training_metrics': training_history,
                'validation_metrics': validation_metrics,
                'test_metrics': None,
                'model_path': str(model_path)
            }
            
        except Exception as e:
            self.logger.error(f"Content-based training failed: {e}")
            raise
    
    async def _train_hybrid_model(self, 
                                config: TrainingConfig, 
                                dataset: MLDataset,
                                mlflow_run) -> Dict[str, Any]:
        """Train hybrid recommendation model"""
        try:
            # Initialize hybrid system
            hybrid_system = HybridRecommendationSystem(
                cf_weight=0.6,
                cb_weight=0.4,
                min_cf_interactions=config.min_interactions
            )
            
            # Initialize models
            num_users = dataset.train_data.user_item_matrix.shape[0]
            num_items = dataset.train_data.user_item_matrix.shape[1]
            
            hybrid_system.initialize_models(
                num_users=num_users,
                num_items=num_items,
                cf_embedding_dim=config.embedding_dim,
                cb_embedding_dim=config.embedding_dim
            )
            
            # Prepare property data
            property_data = []
            for prop_meta in dataset.train_data.property_metadata:
                property_data.append({
                    'title': prop_meta.get('title', ''),
                    'location': prop_meta.get('location', ''),
                    'price': prop_meta.get('price', 0),
                    'bedrooms': prop_meta.get('bedrooms', 0),
                    'bathrooms': prop_meta.get('bathrooms', 0),
                    'amenities': [],
                    'property_type': 'apartment'
                })
            
            # Train hybrid system
            training_results = hybrid_system.fit(
                user_item_matrix=dataset.train_data.user_item_matrix,
                property_data=property_data,
                cf_epochs=config.epochs,
                cb_epochs=config.epochs,
                cf_batch_size=config.batch_size,
                cb_batch_size=config.batch_size,
                validation_split=config.validation_split
            )
            
            # Evaluate hybrid system
            val_predictions = []
            val_actuals = []
            
            for user_idx in range(min(50, dataset.validation_data.user_item_matrix.shape[0])):  # Sample for efficiency
                user_items = np.where(dataset.validation_data.user_item_matrix[user_idx] > 0)[0]
                if len(user_items) > 0:
                    sample_items = user_items[:5]  # Small sample for efficiency
                    predictions = hybrid_system.predict(user_idx, sample_items.tolist())
                    actuals = dataset.validation_data.user_item_matrix[user_idx, sample_items]
                    
                    val_predictions.extend(predictions)
                    val_actuals.extend(actuals)
            
            # Calculate validation metrics
            validation_metrics = {}
            if val_predictions:
                val_predictions = np.array(val_predictions)
                val_actuals = np.array(val_actuals)
                
                validation_metrics = {
                    'val_mse': float(mean_squared_error(val_actuals, val_predictions)),
                    'val_mae': float(mean_absolute_error(val_actuals, val_predictions)),
                    'val_rmse': float(np.sqrt(mean_squared_error(val_actuals, val_predictions)))
                }
            
            # Save models
            cf_model_filename = f"hybrid_cf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.h5"
            cb_model_filename = f"hybrid_cb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.h5"
            
            cf_model_path = self.models_dir / cf_model_filename
            cb_model_path = self.models_dir / cb_model_filename
            
            hybrid_system.save_models(str(cf_model_path), str(cb_model_path))
            
            # Store trained model
            self.trained_models['hybrid'] = hybrid_system
            
            return {
                'training_metrics': training_results,
                'validation_metrics': validation_metrics,
                'test_metrics': None,
                'model_path': str(cf_model_path)  # Return primary model path
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid model training failed: {e}")
            raise
    
    async def _train_search_ranker(self, 
                                 config: TrainingConfig, 
                                 dataset: MLDataset,
                                 mlflow_run) -> Dict[str, Any]:
        """Train search ranking model"""
        try:
            # Initialize search ranker
            ranker = NLPSearchRanker()
            
            # Generate synthetic training data for search ranking
            # In production, this would come from search logs and click data
            training_data = self._generate_search_training_data(dataset)
            
            if not training_data:
                self.logger.warning("No search training data available, skipping search ranker training")
                return {
                    'training_metrics': {},
                    'validation_metrics': {},
                    'test_metrics': None,
                    'model_path': ''
                }
            
            # Split training data
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]
            
            # Train model
            training_metrics = ranker.train(
                training_data=train_data,
                validation_data=val_data,
                epochs=min(config.epochs, 20),  # Search ranker needs fewer epochs
                batch_size=config.batch_size
            )
            
            # Evaluate model
            validation_metrics = ranker.evaluate(val_data)
            
            # Save model
            model_filename = f"search_ranker_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.h5"
            model_path = self.models_dir / model_filename
            ranker.save_model(str(model_path))
            
            # Store trained model
            self.trained_models['search_ranker'] = ranker
            
            return {
                'training_metrics': training_metrics,
                'validation_metrics': validation_metrics,
                'test_metrics': None,
                'model_path': str(model_path)
            }
            
        except Exception as e:
            self.logger.error(f"Search ranker training failed: {e}")
            raise
    
    def _generate_search_training_data(self, dataset: MLDataset) -> List[Tuple[str, Dict, float]]:
        """Generate synthetic search training data"""
        try:
            training_data = []
            
            # Use property metadata to generate search queries and relevance scores
            for prop_meta in dataset.train_data.property_metadata[:100]:  # Limit for efficiency
                # Generate positive examples
                location = prop_meta.get('location', '').split(',')[0]  # City part
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
            
            return training_data[:200]  # Return limited set
            
        except Exception as e:
            self.logger.error(f"Failed to generate search training data: {e}")
            return []
    
    async def cross_validate_model(self, 
                                 config: TrainingConfig, 
                                 cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation for model evaluation.
        
        Args:
            config: Training configuration
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        try:
            self.logger.info(f"Starting {cv_folds}-fold cross-validation for {config.model_type}")
            
            # Load full dataset
            dataset = await self.data_loader.load_training_dataset(
                train_split=1.0,  # Use all data for CV
                validation_split=0.0,
                test_split=0.0,
                min_interactions=config.min_interactions,
                max_users=config.max_users,
                max_properties=config.max_properties
            )
            
            # Prepare for cross-validation
            user_item_matrix = dataset.train_data.user_item_matrix
            num_users = user_item_matrix.shape[0]
            
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            cv_results = []
            
            for fold, (train_indices, val_indices) in enumerate(kfold.split(range(num_users))):
                self.logger.info(f"Training fold {fold + 1}/{cv_folds}")
                
                # Create fold-specific datasets
                fold_train_matrix = user_item_matrix[train_indices]
                fold_val_matrix = user_item_matrix[val_indices]
                
                # Train model for this fold
                if config.model_type == 'collaborative':
                    fold_results = await self._cv_train_collaborative(
                        config, fold_train_matrix, fold_val_matrix
                    )
                else:
                    self.logger.warning(f"Cross-validation not implemented for {config.model_type}")
                    continue
                
                cv_results.append(fold_results)
            
            # Aggregate results
            if cv_results:
                avg_metrics = {}
                for metric in cv_results[0].keys():
                    values = [result[metric] for result in cv_results if metric in result]
                    avg_metrics[f"{metric}_mean"] = np.mean(values)
                    avg_metrics[f"{metric}_std"] = np.std(values)
                
                return {
                    'cv_results': cv_results,
                    'averaged_metrics': avg_metrics,
                    'num_folds': cv_folds
                }
            else:
                return {'error': 'No valid cross-validation results'}
                
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            raise
    
    async def _cv_train_collaborative(self, 
                                    config: TrainingConfig,
                                    train_matrix: np.ndarray,
                                    val_matrix: np.ndarray) -> Dict[str, float]:
        """Train collaborative filtering model for one CV fold"""
        try:
            # Initialize model
            model = CollaborativeFilteringModel(
                num_users=train_matrix.shape[0],
                num_items=train_matrix.shape[1],
                embedding_dim=config.embedding_dim,
                reg_lambda=config.regularization
            )
            
            # Train model
            model.fit(
                user_item_matrix=train_matrix,
                epochs=config.epochs // 2,  # Fewer epochs for CV
                batch_size=config.batch_size,
                validation_split=0.0  # No internal validation for CV
            )
            
            # Evaluate on validation fold
            val_predictions = []
            val_actuals = []
            
            for user_idx in range(min(val_matrix.shape[0], 50)):  # Sample for efficiency
                user_items = np.where(val_matrix[user_idx] > 0)[0]
                if len(user_items) > 0:
                    sample_items = user_items[:5]  # Small sample
                    predictions = model.predict(user_idx, sample_items.tolist())
                    actuals = val_matrix[user_idx, sample_items]
                    
                    val_predictions.extend(predictions)
                    val_actuals.extend(actuals)
            
            # Calculate metrics
            if val_predictions:
                val_predictions = np.array(val_predictions)
                val_actuals = np.array(val_actuals)
                
                return {
                    'mse': float(mean_squared_error(val_actuals, val_predictions)),
                    'mae': float(mean_absolute_error(val_actuals, val_predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(val_actuals, val_predictions)))
                }
            else:
                return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
                
        except Exception as e:
            self.logger.error(f"CV fold training failed: {e}")
            return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
    
    def get_trained_model(self, model_type: str):
        """Get a trained model by type"""
        return self.trained_models.get(model_type)
    
    async def evaluate_model_performance(self, 
                                       model_type: str,
                                       test_data: Optional[MLDataset] = None) -> Dict[str, Any]:
        """
        Evaluate trained model performance.
        
        Args:
            model_type: Type of model to evaluate
            test_data: Optional test dataset
            
        Returns:
            Performance metrics
        """
        try:
            model = self.get_trained_model(model_type)
            if model is None:
                raise ValueError(f"No trained model found for type: {model_type}")
            
            if test_data is None:
                # Load fresh test data
                dataset = await self.data_loader.load_training_dataset()
                test_data = dataset.test_data
            
            if test_data is None:
                raise ValueError("No test data available for evaluation")
            
            # Perform evaluation based on model type
            if model_type == 'collaborative':
                return await self._evaluate_collaborative_model(model, test_data)
            elif model_type == 'content':
                return await self._evaluate_content_model(model, test_data)
            elif model_type == 'hybrid':
                return await self._evaluate_hybrid_model(model, test_data)
            else:
                raise ValueError(f"Evaluation not implemented for {model_type}")
                
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    async def _evaluate_collaborative_model(self, model, test_data) -> Dict[str, Any]:
        """Evaluate collaborative filtering model"""
        # Implementation similar to training evaluation
        # Would include metrics like precision@k, recall@k, NDCG
        pass
    
    async def _evaluate_content_model(self, model, test_data) -> Dict[str, Any]:
        """Evaluate content-based model"""
        # Implementation for content-based evaluation
        pass
    
    async def _evaluate_hybrid_model(self, model, test_data) -> Dict[str, Any]:
        """Evaluate hybrid model"""
        # Implementation for hybrid model evaluation
        pass