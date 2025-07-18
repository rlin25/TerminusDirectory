"""
MLflow Manager for comprehensive experiment tracking and model management.

This module provides:
- Experiment lifecycle management
- Model registry operations
- Model versioning and staging
- Metrics and artifact tracking
- Model lineage and governance
- Integration with CI/CD pipelines
"""

import asyncio
import logging
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil

import mlflow
import mlflow.tensorflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
import pandas as pd
import numpy as np

from ..src.ml.production.training_pipeline import TrainingConfig, ModelResult


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiments"""
    name: str
    description: str
    tags: Dict[str, str]
    artifact_location: Optional[str] = None


@dataclass
class ModelRegistryConfig:
    """Configuration for model registry operations"""
    model_name: str
    model_version: Optional[str] = None
    stage: str = "None"  # None, Staging, Production, Archived
    description: Optional[str] = None
    tags: Dict[str, str] = None


class MLflowManager:
    """
    Comprehensive MLflow management for production ML workflows.
    
    Features:
    - Experiment lifecycle management
    - Model registry operations with staging
    - Metrics and artifacts tracking
    - Model lineage and governance
    - Performance monitoring integration
    - CI/CD pipeline integration
    """
    
    def __init__(self,
                 tracking_uri: str = "http://localhost:5000",
                 registry_uri: Optional[str] = None,
                 default_artifact_root: str = "/app/mlflow-artifacts"):
        
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        self.default_artifact_root = Path(default_artifact_root)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        # MLflow client
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri
        )
        
        # Ensure artifact directory exists
        self.default_artifact_root.mkdir(parents=True, exist_ok=True)
        
        # Active experiments cache
        self._experiments_cache = {}
        self._cache_expiry = {}
        
    async def initialize(self):
        """Initialize MLflow manager"""
        try:
            # Test connection
            experiments = self.client.search_experiments()
            self.logger.info(f"Connected to MLflow. Found {len(experiments)} experiments.")
            
            # Create default experiment if not exists
            await self.create_experiment_if_not_exists(
                ExperimentConfig(
                    name="rental_ml_default",
                    description="Default experiment for rental ML system",
                    tags={"project": "rental-ml", "environment": "production"}
                )
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow manager: {e}")
            raise
    
    async def create_experiment_if_not_exists(self, config: ExperimentConfig) -> str:
        """Create experiment if it doesn't exist"""
        try:
            # Check if experiment exists
            try:
                experiment = self.client.get_experiment_by_name(config.name)
                if experiment:
                    self.logger.info(f"Experiment '{config.name}' already exists")
                    return experiment.experiment_id
            except Exception:
                pass
            
            # Create new experiment
            experiment_id = self.client.create_experiment(
                name=config.name,
                artifact_location=config.artifact_location or str(self.default_artifact_root / config.name),
                tags=config.tags
            )
            
            # Set description if provided
            if config.description:
                self.client.set_experiment_tag(experiment_id, "mlflow.note.content", config.description)
            
            self.logger.info(f"Created experiment '{config.name}' with ID {experiment_id}")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def start_training_run(self,
                                experiment_name: str,
                                run_name: Optional[str] = None,
                                nested: bool = False) -> str:
        """Start a new training run"""
        try:
            # Set experiment
            mlflow.set_experiment(experiment_name)
            
            # Start run
            run = mlflow.start_run(
                run_name=run_name,
                nested=nested
            )
            
            # Log system info
            await self._log_system_info()
            
            self.logger.info(f"Started MLflow run {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            self.logger.error(f"Failed to start training run: {e}")
            raise
    
    async def log_training_config(self, config: TrainingConfig):
        """Log training configuration to MLflow"""
        try:
            # Convert config to dict and log as parameters
            config_dict = asdict(config)
            
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(config_dict)
            
            # Log parameters (MLflow has limits on param value length)
            for key, value in flat_params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value)[:250])  # Truncate long values
            
            # Log full config as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_dict, f, indent=2, default=str)
                temp_path = f.name
            
            try:
                mlflow.log_artifact(temp_path, "config")
            finally:
                os.unlink(temp_path)
            
            self.logger.debug("Logged training configuration to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log training config: {e}")
    
    async def log_training_results(self, results: Dict[str, ModelResult]):
        """Log training results to MLflow"""
        try:
            for model_type, result in results.items():
                # Log metrics with model type prefix
                for metric_name, value in result.validation_metrics.items():
                    mlflow.log_metric(f"{model_type}_{metric_name}", value)
                
                # Log training time
                mlflow.log_metric(f"{model_type}_training_time", result.training_time)
                
                # Log model size
                mlflow.log_metric(f"{model_type}_model_size_mb", result.model_size_mb)
                
                # Log test metrics if available
                if result.test_metrics:
                    for metric_name, value in result.test_metrics.items():
                        mlflow.log_metric(f"{model_type}_test_{metric_name}", value)
                
                # Log CV metrics if available
                if result.cv_metrics:
                    for metric_name, value in result.cv_metrics.items():
                        mlflow.log_metric(f"{model_type}_{metric_name}", value)
                
                # Log hyperparameters
                for param_name, value in result.hyperparameters.items():
                    mlflow.log_param(f"{model_type}_{param_name}", value)
                
                # Log model artifact
                if result.model_path and os.path.exists(result.model_path):
                    mlflow.log_artifact(result.model_path, f"{model_type}_model")
                
                # Log feature importance if available
                if result.feature_importance:
                    await self._log_feature_importance(result.feature_importance, model_type)
            
            # Log best model information
            best_model = self._select_best_model(results)
            if best_model:
                mlflow.set_tag("best_model_type", best_model.model_type)
                mlflow.log_metric("best_model_score", best_model.validation_metrics.get('rmse', 0))
            
            self.logger.info("Logged training results to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log training results: {e}")
    
    async def log_dataset_info(self, dataset_metadata: Dict[str, Any]):
        """Log dataset information"""
        try:
            # Log dataset metrics
            mlflow.log_metric("dataset_users", dataset_metadata.get('total_users', 0))
            mlflow.log_metric("dataset_properties", dataset_metadata.get('total_properties', 0))
            mlflow.log_metric("dataset_interactions", dataset_metadata.get('total_interactions', 0))
            
            # Log data quality metrics
            if 'data_quality' in dataset_metadata:
                quality_metrics = dataset_metadata['data_quality']
                for metric_name, value in quality_metrics.items():
                    mlflow.log_metric(f"data_quality_{metric_name}", value)
            
            # Log dataset as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dataset_metadata, f, indent=2, default=str)
                temp_path = f.name
            
            try:
                mlflow.log_artifact(temp_path, "dataset_info")
            finally:
                os.unlink(temp_path)
            
        except Exception as e:
            self.logger.error(f"Failed to log dataset info: {e}")
    
    async def register_model(self,
                           model_path: str,
                           config: ModelRegistryConfig,
                           model_type: str = "tensorflow") -> str:
        """Register model in MLflow Model Registry"""
        try:
            # Log model first if not already logged
            if not mlflow.active_run():
                raise ValueError("No active MLflow run. Start a run before registering model.")
            
            # Log model based on type
            if model_type == "tensorflow":
                model_uri = mlflow.tensorflow.log_model(
                    model=None,  # Model will be loaded from path
                    artifact_path="model",
                    saved_model_dir=model_path
                ).model_uri
            elif model_type == "sklearn":
                # Load and log sklearn model
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                model_uri = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                ).model_uri
            else:
                # Generic artifact logging
                mlflow.log_artifact(model_path, "model")
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
            
            # Register model
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=config.model_name,
                tags=config.tags or {}
            )
            
            # Set description if provided
            if config.description:
                self.client.update_model_version(
                    name=config.model_name,
                    version=registered_model.version,
                    description=config.description
                )
            
            # Transition to specified stage
            if config.stage != "None":
                self.client.transition_model_version_stage(
                    name=config.model_name,
                    version=registered_model.version,
                    stage=config.stage
                )
            
            self.logger.info(f"Registered model {config.model_name} v{registered_model.version}")
            return registered_model.version
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    async def promote_model(self,
                          model_name: str,
                          version: str,
                          stage: str,
                          archive_existing: bool = True) -> bool:
        """Promote model to different stage"""
        try:
            # Archive existing models in target stage if requested
            if archive_existing and stage == "Production":
                await self._archive_existing_production_models(model_name)
            
            # Transition model to new stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            # Add promotion tags
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_at",
                value=datetime.utcnow().isoformat()
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_to",
                value=stage
            )
            
            self.logger.info(f"Promoted model {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to promote model: {e}")
            return False
    
    async def get_latest_model_version(self,
                                     model_name: str,
                                     stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get latest model version for a given stage"""
        try:
            if stage:
                versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
            else:
                # Get all versions and find latest
                versions = self.client.search_model_versions(f"name='{model_name}'")
                if versions:
                    versions = [max(versions, key=lambda v: int(v.version))]
            
            if not versions:
                return None
            
            latest_version = versions[0]
            
            return {
                'name': latest_version.name,
                'version': latest_version.version,
                'stage': latest_version.current_stage,
                'creation_timestamp': latest_version.creation_timestamp,
                'description': latest_version.description,
                'run_id': latest_version.run_id,
                'source': latest_version.source,
                'tags': dict(latest_version.tags) if latest_version.tags else {}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get latest model version: {e}")
            return None
    
    async def compare_model_performance(self,
                                      model_name: str,
                                      version_a: str,
                                      version_b: str,
                                      metrics: List[str]) -> Dict[str, Any]:
        """Compare performance between two model versions"""
        try:
            comparison_results = {
                'model_name': model_name,
                'version_a': version_a,
                'version_b': version_b,
                'metrics_comparison': {},
                'recommendation': None
            }
            
            # Get run IDs for both versions
            version_a_info = self.client.get_model_version(model_name, version_a)
            version_b_info = self.client.get_model_version(model_name, version_b)
            
            run_a = self.client.get_run(version_a_info.run_id)
            run_b = self.client.get_run(version_b_info.run_id)
            
            # Compare metrics
            better_count_a = 0
            better_count_b = 0
            
            for metric in metrics:
                value_a = run_a.data.metrics.get(metric)
                value_b = run_b.data.metrics.get(metric)
                
                if value_a is not None and value_b is not None:
                    improvement = ((value_b - value_a) / value_a) * 100
                    
                    comparison_results['metrics_comparison'][metric] = {
                        'version_a': value_a,
                        'version_b': value_b,
                        'improvement_percent': improvement,
                        'better_version': version_b if improvement > 0 else version_a
                    }
                    
                    # Count which version is better (assuming lower is better for most metrics)
                    if metric.lower() in ['loss', 'error', 'mse', 'mae', 'rmse']:
                        if value_b < value_a:
                            better_count_b += 1
                        else:
                            better_count_a += 1
                    else:  # Higher is better for metrics like accuracy, precision, etc.
                        if value_b > value_a:
                            better_count_b += 1
                        else:
                            better_count_a += 1
            
            # Make recommendation
            if better_count_b > better_count_a:
                comparison_results['recommendation'] = f"Version {version_b} performs better"
            elif better_count_a > better_count_b:
                comparison_results['recommendation'] = f"Version {version_a} performs better"
            else:
                comparison_results['recommendation'] = "Performance is comparable"
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Failed to compare model performance: {e}")
            return {'error': str(e)}
    
    async def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model lineage information"""
        try:
            # Get model version info
            model_version = self.client.get_model_version(model_name, version)
            
            # Get run info
            run = self.client.get_run(model_version.run_id)
            
            # Get experiment info
            experiment = self.client.get_experiment(run.info.experiment_id)
            
            lineage = {
                'model_name': model_name,
                'version': version,
                'run_info': {
                    'run_id': run.info.run_id,
                    'experiment_id': run.info.experiment_id,
                    'experiment_name': experiment.name,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status,
                    'user_id': run.info.user_id
                },
                'parameters': dict(run.data.params),
                'metrics': dict(run.data.metrics),
                'tags': dict(run.data.tags),
                'artifacts': []
            }
            
            # Get artifacts list
            try:
                artifacts = self.client.list_artifacts(run.info.run_id)
                lineage['artifacts'] = [artifact.path for artifact in artifacts]
            except Exception as e:
                self.logger.warning(f"Failed to get artifacts list: {e}")
            
            return lineage
            
        except Exception as e:
            self.logger.error(f"Failed to get model lineage: {e}")
            return {'error': str(e)}
    
    async def search_experiments(self,
                               filter_string: Optional[str] = None,
                               max_results: int = 100) -> List[Dict[str, Any]]:
        """Search experiments with optional filtering"""
        try:
            experiments = self.client.search_experiments(
                filter_string=filter_string,
                max_results=max_results
            )
            
            experiment_list = []
            for exp in experiments:
                experiment_list.append({
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'creation_time': exp.creation_time,
                    'last_update_time': exp.last_update_time,
                    'lifecycle_stage': exp.lifecycle_stage,
                    'artifact_location': exp.artifact_location,
                    'tags': dict(exp.tags) if exp.tags else {}
                })
            
            return experiment_list
            
        except Exception as e:
            self.logger.error(f"Failed to search experiments: {e}")
            return []
    
    async def search_runs(self,
                        experiment_ids: List[str],
                        filter_string: Optional[str] = None,
                        max_results: int = 100) -> List[Dict[str, Any]]:
        """Search runs across experiments"""
        try:
            runs = self.client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results
            )
            
            run_list = []
            for run in runs:
                run_list.append({
                    'run_id': run.info.run_id,
                    'experiment_id': run.info.experiment_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'user_id': run.info.user_id,
                    'parameters': dict(run.data.params),
                    'metrics': dict(run.data.metrics),
                    'tags': dict(run.data.tags)
                })
            
            return run_list
            
        except Exception as e:
            self.logger.error(f"Failed to search runs: {e}")
            return []
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment"""
        try:
            self.client.delete_experiment(experiment_id)
            self.logger.info(f"Deleted experiment {experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete experiment: {e}")
            return False
    
    async def export_experiment(self,
                              experiment_id: str,
                              output_path: str) -> bool:
        """Export experiment data"""
        try:
            # Get experiment info
            experiment = self.client.get_experiment(experiment_id)
            
            # Get all runs in experiment
            runs = self.client.search_runs([experiment_id])
            
            # Prepare export data
            export_data = {
                'experiment': {
                    'experiment_id': experiment.experiment_id,
                    'name': experiment.name,
                    'creation_time': experiment.creation_time,
                    'artifact_location': experiment.artifact_location,
                    'tags': dict(experiment.tags) if experiment.tags else {}
                },
                'runs': []
            }
            
            for run in runs:
                run_data = {
                    'run_id': run.info.run_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'parameters': dict(run.data.params),
                    'metrics': dict(run.data.metrics),
                    'tags': dict(run.data.tags)
                }
                export_data['runs'].append(run_data)
            
            # Save to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported experiment {experiment_id} to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export experiment: {e}")
            return False
    
    def _select_best_model(self, results: Dict[str, ModelResult]) -> Optional[ModelResult]:
        """Select best model from results"""
        if not results:
            return None
        
        best_model = None
        best_score = float('inf')
        
        for result in results.values():
            # Use RMSE as primary metric
            rmse = result.validation_metrics.get('rmse', float('inf'))
            if rmse < best_score:
                best_score = rmse
                best_model = result
        
        return best_model
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                # Handle lists of numbers
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    async def _log_system_info(self):
        """Log system information to MLflow"""
        try:
            import platform
            import psutil
            
            # System info
            mlflow.set_tag("system.platform", platform.platform())
            mlflow.set_tag("system.python_version", platform.python_version())
            mlflow.set_tag("system.cpu_count", psutil.cpu_count())
            
            # Memory info
            memory = psutil.virtual_memory()
            mlflow.log_metric("system.memory_total_gb", memory.total / (1024**3))
            mlflow.log_metric("system.memory_available_gb", memory.available / (1024**3))
            
            # GPU info
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                mlflow.log_metric("system.gpu_count", len(gpus))
                mlflow.set_tag("system.tensorflow_version", tf.__version__)
            except Exception:
                mlflow.log_metric("system.gpu_count", 0)
            
        except Exception as e:
            self.logger.warning(f"Failed to log system info: {e}")
    
    async def _log_feature_importance(self, feature_importance: Dict[str, float], model_type: str):
        """Log feature importance as artifact"""
        try:
            # Create DataFrame
            df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in feature_importance.items()
            ]).sort_values('importance', ascending=False)
            
            # Save as CSV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name
            
            try:
                mlflow.log_artifact(temp_path, f"{model_type}_feature_importance")
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            self.logger.warning(f"Failed to log feature importance: {e}")
    
    async def _archive_existing_production_models(self, model_name: str):
        """Archive existing production models"""
        try:
            production_versions = self.client.get_latest_versions(
                name=model_name,
                stages=["Production"]
            )
            
            for version in production_versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                
                self.logger.info(f"Archived model {model_name} v{version.version}")
                
        except Exception as e:
            self.logger.warning(f"Failed to archive existing models: {e}")
    
    async def cleanup_old_runs(self,
                             experiment_id: str,
                             keep_last_n: int = 10,
                             older_than_days: int = 30) -> int:
        """Clean up old runs in an experiment"""
        try:
            # Get all runs in experiment
            runs = self.client.search_runs([experiment_id])
            
            # Sort by start time (newest first)
            runs = sorted(runs, key=lambda r: r.info.start_time, reverse=True)
            
            # Keep last N runs
            runs_to_delete = runs[keep_last_n:]
            
            # Also filter by age
            cutoff_time = (datetime.utcnow() - timedelta(days=older_than_days)).timestamp() * 1000
            runs_to_delete = [r for r in runs_to_delete if r.info.start_time < cutoff_time]
            
            # Delete runs
            deleted_count = 0
            for run in runs_to_delete:
                try:
                    self.client.delete_run(run.info.run_id)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete run {run.info.run_id}: {e}")
            
            self.logger.info(f"Deleted {deleted_count} old runs from experiment {experiment_id}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old runs: {e}")
            return 0