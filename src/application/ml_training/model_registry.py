"""
Model Registry for ML Model Version Management

This module provides comprehensive model version management, metadata tracking,
deployment status management, and rollback capabilities for the rental ML system.

Features:
- Model version management with semantic versioning
- Model metadata tracking (performance metrics, training data, hyperparameters)
- Deployment status tracking across environments
- Model rollback capabilities
- A/B testing support
- Model lineage tracking
- Performance comparison across versions
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
from enum import Enum
from uuid import uuid4, UUID
import hashlib

import numpy as np
import pandas as pd

# Domain imports
from ...domain.repositories.model_repository import ModelRepository


class ModelStatus(Enum):
    """Model deployment status"""
    REGISTERED = "registered"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


@dataclass
class ModelVersion:
    """Model version information"""
    model_name: str
    version: str
    model_id: str
    status: ModelStatus
    model_path: str
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    created_by: str
    parent_version: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    model_size_bytes: Optional[int] = None
    model_hash: Optional[str] = None


@dataclass
class DeploymentInfo:
    """Model deployment information"""
    model_name: str
    version: str
    environment: ModelEnvironment
    deployed_at: datetime
    deployed_by: str
    deployment_config: Dict[str, Any]
    traffic_percentage: float = 100.0
    status: str = "active"
    endpoint_url: Optional[str] = None
    health_check_url: Optional[str] = None


@dataclass
class ModelComparison:
    """Model performance comparison"""
    baseline_version: str
    candidate_version: str
    metrics_comparison: Dict[str, Dict[str, float]]
    improvement_percentage: Dict[str, float]
    statistical_significance: Dict[str, bool]
    recommendation: str  # "deploy", "reject", "needs_more_data"


@dataclass
class ExperimentConfig:
    """A/B testing experiment configuration"""
    experiment_id: str
    model_name: str
    baseline_version: str
    candidate_version: str
    traffic_split: Dict[str, float]  # version -> percentage
    success_criteria: Dict[str, Any]
    duration_days: int
    start_date: datetime
    status: str = "planned"


class ModelRegistry:
    """
    Comprehensive model registry for version management and deployment tracking.
    
    This class provides:
    - Model version management with semantic versioning
    - Metadata and performance tracking
    - Deployment management across environments
    - A/B testing and experimentation support
    - Model lineage and comparison capabilities
    - Rollback and disaster recovery features
    """
    
    def __init__(self, 
                 model_repository: ModelRepository,
                 registry_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model registry.
        
        Args:
            model_repository: Repository for storing models and metadata
            registry_config: Registry configuration options
        """
        self.model_repository = model_repository
        self.config = registry_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Internal state
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.deployments: Dict[str, List[DeploymentInfo]] = {}
        self.experiments: Dict[str, ExperimentConfig] = {}
        
        # Configuration
        self.max_versions_per_model = self.config.get('max_versions_per_model', 50)
        self.auto_archive_days = self.config.get('auto_archive_days', 90)
        self.enable_model_signing = self.config.get('enable_model_signing', True)
        
    async def initialize(self):
        """Initialize the model registry"""
        try:
            self.logger.info("Initializing model registry...")
            
            # Load existing model versions and deployments
            await self._load_registry_state()
            
            # Setup automatic cleanup tasks
            await self._setup_cleanup_tasks()
            
            self.logger.info("Model registry initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model registry: {e}")
            raise
    
    async def register_model(self,
                           model_name: str,
                           version: str,
                           model_path: str,
                           metadata: Dict[str, Any],
                           performance_metrics: Dict[str, float],
                           parent_version: Optional[str] = None,
                           tags: Optional[List[str]] = None,
                           description: Optional[str] = None,
                           created_by: str = "system") -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            version: Version identifier
            model_path: Path to the model file
            metadata: Model metadata including training config
            performance_metrics: Model performance metrics
            parent_version: Parent version for lineage tracking
            tags: Optional tags for categorization
            description: Optional description
            created_by: User or system that created the model
            
        Returns:
            Model ID for the registered version
        """
        try:
            # Generate model ID
            model_id = f"{model_name}_{version}_{uuid4().hex[:8]}"
            
            # Calculate model hash and size
            model_hash, model_size = await self._calculate_model_properties(model_path)
            
            # Validate version doesn't exist
            if await self._version_exists(model_name, version):
                raise ValueError(f"Model version already exists: {model_name}:{version}")
            
            # Create model version
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                model_id=model_id,
                status=ModelStatus.REGISTERED,
                model_path=model_path,
                metadata=metadata,
                performance_metrics=performance_metrics,
                created_at=datetime.utcnow(),
                created_by=created_by,
                parent_version=parent_version,
                tags=tags or [],
                description=description,
                model_size_bytes=model_size,
                model_hash=model_hash
            )
            
            # Store in repository
            await self.model_repository.save_model(
                model_name=model_id,
                model_data=self._serialize_model_version(model_version),
                version=version
            )
            
            # Store metadata
            await self.model_repository.save_training_metrics(
                model_name=model_name,
                version=version,
                metrics={
                    **performance_metrics,
                    'registry_metadata': metadata
                }
            )
            
            # Update internal state
            if model_name not in self.model_versions:
                self.model_versions[model_name] = []
            self.model_versions[model_name].append(model_version)
            
            # Cleanup old versions if needed
            await self._cleanup_old_versions(model_name)
            
            self.logger.info(f"Model registered: {model_name}:{version} ({model_id})")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    async def get_model_version(self, 
                              model_name: str, 
                              version: str = "latest") -> Optional[ModelVersion]:
        """
        Get a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version identifier or "latest"
            
        Returns:
            Model version information if found
        """
        try:
            # Load from cache or repository
            if model_name not in self.model_versions:
                await self._load_model_versions(model_name)
            
            if model_name not in self.model_versions:
                return None
            
            versions = self.model_versions[model_name]
            
            if version == "latest":
                # Return the latest version
                if versions:
                    return max(versions, key=lambda v: v.created_at)
                return None
            else:
                # Find specific version
                for model_version in versions:
                    if model_version.version == version:
                        return model_version
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get model version: {e}")
            return None
    
    async def list_model_versions(self, 
                                model_name: str,
                                status_filter: Optional[ModelStatus] = None,
                                limit: int = 20) -> List[ModelVersion]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            status_filter: Optional status filter
            limit: Maximum number of versions to return
            
        Returns:
            List of model versions
        """
        try:
            # Load versions if not cached
            if model_name not in self.model_versions:
                await self._load_model_versions(model_name)
            
            if model_name not in self.model_versions:
                return []
            
            versions = self.model_versions[model_name]
            
            # Apply status filter
            if status_filter:
                versions = [v for v in versions if v.status == status_filter]
            
            # Sort by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            
            return versions[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to list model versions: {e}")
            return []
    
    async def update_model_status(self,
                                model_name: str,
                                version: str,
                                new_status: ModelStatus) -> bool:
        """
        Update the status of a model version.
        
        Args:
            model_name: Name of the model
            version: Version identifier
            new_status: New status to set
            
        Returns:
            True if successful
        """
        try:
            model_version = await self.get_model_version(model_name, version)
            if not model_version:
                raise ValueError(f"Model version not found: {model_name}:{version}")
            
            # Update status
            model_version.status = new_status
            
            # Save updated version
            await self._save_model_version(model_version)
            
            self.logger.info(f"Model status updated: {model_name}:{version} -> {new_status.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            return False
    
    async def deploy_to_staging(self,
                              model_name: str,
                              version: str,
                              deployment_config: Optional[Dict[str, Any]] = None,
                              deployed_by: str = "system") -> bool:
        """
        Deploy model to staging environment.
        
        Args:
            model_name: Name of the model
            version: Version to deploy
            deployment_config: Deployment configuration
            deployed_by: User deploying the model
            
        Returns:
            True if successful
        """
        return await self._deploy_model(
            model_name=model_name,
            version=version,
            environment=ModelEnvironment.STAGING,
            deployment_config=deployment_config or {},
            deployed_by=deployed_by
        )
    
    async def deploy_to_production(self,
                                 model_name: str,
                                 version: str,
                                 deployment_config: Optional[Dict[str, Any]] = None,
                                 deployed_by: str = "system") -> bool:
        """
        Deploy model to production environment.
        
        Args:
            model_name: Name of the model
            version: Version to deploy
            deployment_config: Deployment configuration
            deployed_by: User deploying the model
            
        Returns:
            True if successful
        """
        return await self._deploy_model(
            model_name=model_name,
            version=version,
            environment=ModelEnvironment.PRODUCTION,
            deployment_config=deployment_config or {},
            deployed_by=deployed_by
        )
    
    async def deploy_canary(self,
                          model_name: str,
                          version: str,
                          traffic_percentage: float = 10.0,
                          deployment_config: Optional[Dict[str, Any]] = None,
                          deployed_by: str = "system") -> bool:
        """
        Deploy model as canary with limited traffic.
        
        Args:
            model_name: Name of the model
            version: Version to deploy
            traffic_percentage: Percentage of traffic to route to canary
            deployment_config: Deployment configuration
            deployed_by: User deploying the model
            
        Returns:
            True if successful
        """
        config = deployment_config or {}
        config['traffic_percentage'] = traffic_percentage
        
        return await self._deploy_model(
            model_name=model_name,
            version=version,
            environment=ModelEnvironment.CANARY,
            deployment_config=config,
            deployed_by=deployed_by,
            traffic_percentage=traffic_percentage
        )
    
    async def _deploy_model(self,
                          model_name: str,
                          version: str,
                          environment: ModelEnvironment,
                          deployment_config: Dict[str, Any],
                          deployed_by: str,
                          traffic_percentage: float = 100.0) -> bool:
        """Internal method to deploy model to environment"""
        try:
            # Validate model exists
            model_version = await self.get_model_version(model_name, version)
            if not model_version:
                raise ValueError(f"Model version not found: {model_name}:{version}")
            
            # Create deployment info
            deployment = DeploymentInfo(
                model_name=model_name,
                version=version,
                environment=environment,
                deployed_at=datetime.utcnow(),
                deployed_by=deployed_by,
                deployment_config=deployment_config,
                traffic_percentage=traffic_percentage
            )
            
            # Store deployment info
            deployment_key = f"{model_name}_{environment.value}"
            if deployment_key not in self.deployments:
                self.deployments[deployment_key] = []
            self.deployments[deployment_key].append(deployment)
            
            # Update model status
            if environment == ModelEnvironment.STAGING:
                await self.update_model_status(model_name, version, ModelStatus.STAGING)
            elif environment == ModelEnvironment.PRODUCTION:
                await self.update_model_status(model_name, version, ModelStatus.PRODUCTION)
            elif environment == ModelEnvironment.CANARY:
                await self.update_model_status(model_name, version, ModelStatus.CANARY)
            
            self.logger.info(f"Model deployed: {model_name}:{version} to {environment.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            return False
    
    async def rollback_deployment(self,
                                model_name: str,
                                environment: ModelEnvironment,
                                target_version: Optional[str] = None) -> bool:
        """
        Rollback deployment to previous version.
        
        Args:
            model_name: Name of the model
            environment: Environment to rollback
            target_version: Specific version to rollback to (defaults to previous)
            
        Returns:
            True if successful
        """
        try:
            deployment_key = f"{model_name}_{environment.value}"
            
            if deployment_key not in self.deployments or not self.deployments[deployment_key]:
                raise ValueError(f"No deployments found for {model_name} in {environment.value}")
            
            deployments = self.deployments[deployment_key]
            deployments.sort(key=lambda d: d.deployed_at, reverse=True)
            
            if target_version:
                # Find specific version
                target_deployment = None
                for deployment in deployments:
                    if deployment.version == target_version:
                        target_deployment = deployment
                        break
                
                if not target_deployment:
                    raise ValueError(f"Target version not found in deployments: {target_version}")
            else:
                # Use previous version
                if len(deployments) < 2:
                    raise ValueError("No previous version available for rollback")
                target_deployment = deployments[1]  # Second most recent
            
            # Deploy target version
            success = await self._deploy_model(
                model_name=model_name,
                version=target_deployment.version,
                environment=environment,
                deployment_config=target_deployment.deployment_config,
                deployed_by="system_rollback",
                traffic_percentage=target_deployment.traffic_percentage
            )
            
            if success:
                self.logger.info(f"Rollback successful: {model_name} in {environment.value} to {target_deployment.version}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to rollback deployment: {e}")
            return False
    
    async def compare_models(self,
                           model_name: str,
                           baseline_version: str,
                           candidate_version: str) -> ModelComparison:
        """
        Compare performance between two model versions.
        
        Args:
            model_name: Name of the model
            baseline_version: Baseline version for comparison
            candidate_version: Candidate version to compare
            
        Returns:
            Model comparison results
        """
        try:
            # Get model versions
            baseline = await self.get_model_version(model_name, baseline_version)
            candidate = await self.get_model_version(model_name, candidate_version)
            
            if not baseline or not candidate:
                raise ValueError("One or both model versions not found")
            
            # Compare metrics
            baseline_metrics = baseline.performance_metrics
            candidate_metrics = candidate.performance_metrics
            
            metrics_comparison = {}
            improvement_percentage = {}
            statistical_significance = {}
            
            for metric in set(baseline_metrics.keys()) | set(candidate_metrics.keys()):
                baseline_value = baseline_metrics.get(metric, 0.0)
                candidate_value = candidate_metrics.get(metric, 0.0)
                
                metrics_comparison[metric] = {
                    'baseline': baseline_value,
                    'candidate': candidate_value,
                    'difference': candidate_value - baseline_value
                }
                
                # Calculate improvement percentage
                if baseline_value != 0:
                    improvement = ((candidate_value - baseline_value) / baseline_value) * 100
                    improvement_percentage[metric] = improvement
                else:
                    improvement_percentage[metric] = float('inf') if candidate_value > 0 else 0.0
                
                # Simple significance test (would use proper statistical tests in production)
                statistical_significance[metric] = abs(improvement_percentage[metric]) > 5.0
            
            # Generate recommendation
            significant_improvements = sum(1 for sig in statistical_significance.values() if sig)
            total_metrics = len(statistical_significance)
            
            if significant_improvements >= total_metrics * 0.6:
                recommendation = "deploy"
            elif significant_improvements == 0:
                recommendation = "reject"
            else:
                recommendation = "needs_more_data"
            
            return ModelComparison(
                baseline_version=baseline_version,
                candidate_version=candidate_version,
                metrics_comparison=metrics_comparison,
                improvement_percentage=improvement_percentage,
                statistical_significance=statistical_significance,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            raise
    
    async def create_experiment(self,
                              model_name: str,
                              baseline_version: str,
                              candidate_version: str,
                              traffic_split: Dict[str, float],
                              success_criteria: Dict[str, Any],
                              duration_days: int = 7) -> str:
        """
        Create A/B testing experiment.
        
        Args:
            model_name: Name of the model
            baseline_version: Baseline version
            candidate_version: Candidate version to test
            traffic_split: Traffic split between versions
            success_criteria: Criteria for experiment success
            duration_days: Experiment duration in days
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = f"exp_{model_name}_{uuid4().hex[:8]}"
            
            # Validate traffic split
            total_traffic = sum(traffic_split.values())
            if abs(total_traffic - 100.0) > 0.1:
                raise ValueError(f"Traffic split must sum to 100%, got {total_traffic}%")
            
            # Create experiment
            experiment = ExperimentConfig(
                experiment_id=experiment_id,
                model_name=model_name,
                baseline_version=baseline_version,
                candidate_version=candidate_version,
                traffic_split=traffic_split,
                success_criteria=success_criteria,
                duration_days=duration_days,
                start_date=datetime.utcnow(),
                status="active"
            )
            
            self.experiments[experiment_id] = experiment
            
            self.logger.info(f"Experiment created: {experiment_id}")
            
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Get model lineage and ancestry information.
        
        Args:
            model_name: Name of the model
            version: Version to trace
            
        Returns:
            Lineage information
        """
        try:
            lineage = {
                'model_name': model_name,
                'version': version,
                'ancestors': [],
                'descendants': []
            }
            
            # Get all versions for lineage tracing
            all_versions = await self.list_model_versions(model_name, limit=1000)
            
            # Build ancestry tree
            current_version = await self.get_model_version(model_name, version)
            if not current_version:
                return lineage
            
            # Trace ancestors
            ancestors = []
            current = current_version
            while current and current.parent_version:
                parent = await self.get_model_version(model_name, current.parent_version)
                if parent:
                    ancestors.append({
                        'version': parent.version,
                        'created_at': parent.created_at.isoformat(),
                        'performance_metrics': parent.performance_metrics
                    })
                    current = parent
                else:
                    break
            
            lineage['ancestors'] = ancestors
            
            # Find descendants
            descendants = []
            for ver in all_versions:
                if ver.parent_version == version:
                    descendants.append({
                        'version': ver.version,
                        'created_at': ver.created_at.isoformat(),
                        'performance_metrics': ver.performance_metrics
                    })
            
            lineage['descendants'] = descendants
            
            return lineage
            
        except Exception as e:
            self.logger.error(f"Failed to get model lineage: {e}")
            return {}
    
    async def archive_old_versions(self, 
                                 model_name: str,
                                 keep_versions: int = 10) -> int:
        """
        Archive old model versions.
        
        Args:
            model_name: Name of the model
            keep_versions: Number of recent versions to keep
            
        Returns:
            Number of versions archived
        """
        try:
            versions = await self.list_model_versions(model_name, limit=1000)
            
            if len(versions) <= keep_versions:
                return 0
            
            # Sort by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            
            # Archive old versions
            archived_count = 0
            for version in versions[keep_versions:]:
                if version.status not in [ModelStatus.PRODUCTION, ModelStatus.STAGING]:
                    await self.update_model_status(model_name, version.version, ModelStatus.ARCHIVED)
                    archived_count += 1
            
            self.logger.info(f"Archived {archived_count} old versions for {model_name}")
            
            return archived_count
            
        except Exception as e:
            self.logger.error(f"Failed to archive old versions: {e}")
            return 0
    
    async def get_deployment_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get current deployment status across all environments.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Deployment status information
        """
        try:
            status = {
                'model_name': model_name,
                'environments': {}
            }
            
            for env in ModelEnvironment:
                deployment_key = f"{model_name}_{env.value}"
                if deployment_key in self.deployments and self.deployments[deployment_key]:
                    # Get latest deployment
                    latest_deployment = max(
                        self.deployments[deployment_key],
                        key=lambda d: d.deployed_at
                    )
                    
                    status['environments'][env.value] = {
                        'version': latest_deployment.version,
                        'deployed_at': latest_deployment.deployed_at.isoformat(),
                        'deployed_by': latest_deployment.deployed_by,
                        'traffic_percentage': latest_deployment.traffic_percentage,
                        'status': latest_deployment.status
                    }
                else:
                    status['environments'][env.value] = None
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    async def _load_registry_state(self):
        """Load existing registry state from repository"""
        try:
            # This would load existing models and deployments from the repository
            # For now, we'll start with empty state
            pass
        except Exception as e:
            self.logger.warning(f"Failed to load registry state: {e}")
    
    async def _setup_cleanup_tasks(self):
        """Setup automatic cleanup tasks"""
        try:
            # This would setup periodic tasks for:
            # - Archiving old versions
            # - Cleaning up unused models
            # - Updating deployment status
            pass
        except Exception as e:
            self.logger.warning(f"Failed to setup cleanup tasks: {e}")
    
    async def _calculate_model_properties(self, model_path: str) -> Tuple[str, int]:
        """Calculate model hash and size"""
        try:
            if not os.path.exists(model_path):
                return "", 0
            
            # Calculate file size
            size = os.path.getsize(model_path)
            
            # Calculate hash
            hash_md5 = hashlib.md5()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            return hash_md5.hexdigest(), size
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate model properties: {e}")
            return "", 0
    
    async def _version_exists(self, model_name: str, version: str) -> bool:
        """Check if a version already exists"""
        existing_version = await self.get_model_version(model_name, version)
        return existing_version is not None
    
    def _serialize_model_version(self, model_version: ModelVersion) -> bytes:
        """Serialize model version for storage"""
        try:
            version_dict = asdict(model_version)
            # Convert datetime to string
            version_dict['created_at'] = model_version.created_at.isoformat()
            return pickle.dumps(version_dict)
        except Exception as e:
            self.logger.error(f"Failed to serialize model version: {e}")
            raise
    
    async def _save_model_version(self, model_version: ModelVersion):
        """Save model version to repository"""
        try:
            serialized_data = self._serialize_model_version(model_version)
            await self.model_repository.save_model(
                model_name=model_version.model_id,
                model_data=serialized_data,
                version=model_version.version
            )
        except Exception as e:
            self.logger.error(f"Failed to save model version: {e}")
            raise
    
    async def _load_model_versions(self, model_name: str):
        """Load model versions from repository"""
        try:
            # Load versions from the repository
            repository_versions = await self.model_repository.get_model_versions(model_name)
            
            if model_name not in self.model_versions:
                self.model_versions[model_name] = []
            
            # Convert repository data to ModelVersion objects
            for version_str in repository_versions:
                try:
                    # Load the model data to get full version info
                    model_data = await self.model_repository.load_model(model_name, version_str)
                    if model_data:
                        # Deserialize model version if it's stored as version data
                        if isinstance(model_data, bytes):
                            version_dict = pickle.loads(model_data)
                            if isinstance(version_dict, dict) and 'created_at' in version_dict:
                                # Restore datetime
                                version_dict['created_at'] = datetime.fromisoformat(version_dict['created_at'])
                                model_version = ModelVersion(**version_dict)
                                self.model_versions[model_name].append(model_version)
                except Exception as e:
                    self.logger.warning(f"Failed to load version {version_str} for {model_name}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load model versions for {model_name}: {e}")
    
    async def _cleanup_old_versions(self, model_name: str):
        """Cleanup old versions if limit exceeded"""
        try:
            if model_name in self.model_versions:
                versions = self.model_versions[model_name]
                if len(versions) > self.max_versions_per_model:
                    # Auto-archive old versions
                    await self.archive_old_versions(
                        model_name, 
                        keep_versions=self.max_versions_per_model
                    )
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old versions: {e}")