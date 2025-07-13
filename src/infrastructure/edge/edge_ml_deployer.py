"""
Edge ML model deployment and management system.

This module handles deployment of ML models to edge nodes for faster inference,
model versioning, A/B testing, and health monitoring.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

import aiohttp
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class ModelFormat(str, Enum):
    """Supported model formats for edge deployment"""
    TENSORFLOW_LITE = "tflite"
    TENSORFLOW_JS = "tfjs"
    ONNX = "onnx"
    SCIKIT_LEARN = "sklearn"
    PYTORCH_MOBILE = "pytorch_mobile"


class EdgeNodeStatus(str, Enum):
    """Edge node status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ModelStatus(str, Enum):
    """Model deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    UPDATING = "updating"


class EdgeRegion(str, Enum):
    """Edge deployment regions"""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    ASIA_PACIFIC = "asia-pacific"
    GLOBAL = "global"


class EdgeModelConfig(BaseModel):
    """Edge model configuration"""
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_format: ModelFormat = Field(..., description="Model format")
    model_type: str = Field(..., description="Model type (recommender, classifier, etc.)")
    
    # Deployment configuration
    target_regions: List[EdgeRegion] = Field(..., description="Target deployment regions")
    resource_requirements: Dict[str, Any] = Field(default={}, description="Resource requirements")
    scaling_config: Dict[str, Any] = Field(default={}, description="Auto-scaling configuration")
    
    # Model metadata
    input_schema: Dict[str, Any] = Field(..., description="Input data schema")
    output_schema: Dict[str, Any] = Field(..., description="Output data schema")
    preprocessing_config: Optional[Dict[str, Any]] = Field(None, description="Preprocessing configuration")
    
    # Performance settings
    max_batch_size: int = Field(default=32, description="Maximum batch size")
    timeout_ms: int = Field(default=5000, description="Inference timeout in milliseconds")
    cache_predictions: bool = Field(default=True, description="Cache prediction results")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EdgeNode(BaseModel):
    """Edge node information"""
    node_id: str = Field(..., description="Unique node identifier")
    region: EdgeRegion = Field(..., description="Node region")
    endpoint_url: str = Field(..., description="Node endpoint URL")
    capabilities: List[str] = Field(default=[], description="Node capabilities")
    
    # Resource information
    cpu_cores: int = Field(..., description="Available CPU cores")
    memory_gb: float = Field(..., description="Available memory in GB")
    storage_gb: float = Field(..., description="Available storage in GB")
    
    # Status and health
    status: EdgeNodeStatus = Field(default=EdgeNodeStatus.ACTIVE)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    deployed_models: List[str] = Field(default=[], description="Currently deployed model IDs")
    
    # Performance metrics
    avg_response_time_ms: float = Field(default=0.0, description="Average response time")
    success_rate: float = Field(default=1.0, description="Success rate (0-1)")
    current_load: float = Field(default=0.0, description="Current load (0-1)")


class ModelDeployment(BaseModel):
    """Model deployment information"""
    deployment_id: str = Field(..., description="Unique deployment identifier")
    model_config: EdgeModelConfig = Field(..., description="Model configuration")
    node_id: str = Field(..., description="Target node ID")
    status: ModelStatus = Field(default=ModelStatus.PENDING)
    
    # Deployment details
    model_url: str = Field(..., description="Model file URL")
    model_checksum: str = Field(..., description="Model file checksum")
    deployment_time: Optional[datetime] = Field(None, description="Deployment timestamp")
    
    # Runtime information
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    prediction_count: int = Field(default=0, description="Total predictions served")
    error_count: int = Field(default=0, description="Total errors")
    
    # A/B testing
    traffic_percentage: float = Field(default=100.0, description="Traffic percentage for A/B testing")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EdgePredictionRequest(BaseModel):
    """Edge prediction request"""
    model_id: str = Field(..., description="Model identifier")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    request_id: Optional[str] = Field(None, description="Request identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    region: Optional[EdgeRegion] = Field(None, description="Preferred region")
    timeout_ms: Optional[int] = Field(None, description="Request timeout")


class EdgePredictionResponse(BaseModel):
    """Edge prediction response"""
    prediction_id: str = Field(..., description="Prediction identifier")
    model_id: str = Field(..., description="Model identifier")
    node_id: str = Field(..., description="Serving node ID")
    predictions: Dict[str, Any] = Field(..., description="Prediction results")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    
    # Performance metrics
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    total_time_ms: float = Field(..., description="Total processing time")
    cached: bool = Field(default=False, description="Result served from cache")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EdgeMLDeployer:
    """Edge ML model deployment and management system"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.deployments: Dict[str, ModelDeployment] = {}
        self.model_registry: Dict[str, EdgeModelConfig] = {}
        
        # Configuration
        self.config = {
            "deployment_timeout": 300,  # 5 minutes
            "health_check_interval": 30,  # 30 seconds
            "max_retries": 3,
            "rollback_on_failure": True
        }
    
    async def initialize(
        self,
        redis_url: str = "redis://localhost:6379",
        edge_nodes_config: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize the edge ML deployer"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            
            # Load edge nodes configuration
            if edge_nodes_config:
                for node_config in edge_nodes_config:
                    node = EdgeNode(**node_config)
                    self.edge_nodes[node.node_id] = node
                    await self.register_edge_node(node)
            
            # Start background tasks
            asyncio.create_task(self.health_check_loop())
            asyncio.create_task(self.cleanup_old_deployments())
            
            logger.info("Edge ML deployer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize edge ML deployer: {e}")
            raise
    
    async def register_edge_node(self, node: EdgeNode) -> bool:
        """Register a new edge node"""
        try:
            # Store node information
            node_key = f"edge_node:{node.node_id}"
            await self.redis_client.setex(
                node_key,
                3600,  # 1 hour TTL
                node.model_dump_json()
            )
            
            # Add to active nodes set
            await self.redis_client.sadd("active_edge_nodes", node.node_id)
            
            # Update local registry
            self.edge_nodes[node.node_id] = node
            
            logger.info(f"Edge node registered: {node.node_id} in {node.region}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register edge node: {e}")
            return False
    
    async def deploy_model(
        self,
        model_config: EdgeModelConfig,
        model_file_path: str,
        target_nodes: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Deploy model to edge nodes"""
        try:
            deployment_results = {}
            
            # Select target nodes
            if target_nodes:
                nodes = [self.edge_nodes[nid] for nid in target_nodes if nid in self.edge_nodes]
            else:
                nodes = await self.select_optimal_nodes(model_config)
            
            if not nodes:
                raise ValueError("No suitable edge nodes available")
            
            # Upload model to storage
            model_url = await self.upload_model_file(model_file_path, model_config)
            model_checksum = await self.calculate_file_checksum(model_file_path)
            
            # Deploy to each node
            deployment_tasks = []
            for node in nodes:
                task = self.deploy_to_node(
                    model_config, 
                    node, 
                    model_url, 
                    model_checksum
                )
                deployment_tasks.append(task)
            
            # Execute deployments concurrently
            results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                node_id = nodes[i].node_id
                if isinstance(result, Exception):
                    deployment_results[node_id] = f"failed: {str(result)}"
                else:
                    deployment_results[node_id] = result
            
            # Update model registry
            self.model_registry[model_config.model_id] = model_config
            await self.store_model_config(model_config)
            
            logger.info(f"Model deployment completed: {model_config.model_id}")
            return deployment_results
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    async def deploy_to_node(
        self,
        model_config: EdgeModelConfig,
        node: EdgeNode,
        model_url: str,
        model_checksum: str
    ) -> str:
        """Deploy model to a specific edge node"""
        try:
            deployment_id = f"deploy_{model_config.model_id}_{node.node_id}_{datetime.utcnow().timestamp()}"
            
            # Create deployment record
            deployment = ModelDeployment(
                deployment_id=deployment_id,
                model_config=model_config,
                node_id=node.node_id,
                model_url=model_url,
                model_checksum=model_checksum,
                status=ModelStatus.DEPLOYING
            )
            
            # Store deployment info
            await self.store_deployment(deployment)
            
            # Send deployment request to edge node
            deployment_payload = {
                "model_config": model_config.model_dump(),
                "model_url": model_url,
                "model_checksum": model_checksum,
                "deployment_id": deployment_id
            }
            
            async with aiohttp.ClientSession() as session:
                deploy_url = f"{node.endpoint_url}/deploy"
                
                async with session.post(
                    deploy_url,
                    json=deployment_payload,
                    timeout=aiohttp.ClientTimeout(total=self.config["deployment_timeout"])
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update deployment status
                        deployment.status = ModelStatus.ACTIVE
                        deployment.deployment_time = datetime.utcnow()
                        await self.store_deployment(deployment)
                        
                        # Update node's deployed models
                        if model_config.model_id not in node.deployed_models:
                            node.deployed_models.append(model_config.model_id)
                            await self.update_edge_node(node)
                        
                        logger.info(f"Model deployed successfully to {node.node_id}")
                        return "success"
                    else:
                        error_msg = f"Deployment failed with status {response.status}"
                        deployment.status = ModelStatus.FAILED
                        await self.store_deployment(deployment)
                        
                        logger.error(f"Deployment failed: {error_msg}")
                        return error_msg
            
        except Exception as e:
            # Mark deployment as failed
            deployment.status = ModelStatus.FAILED
            await self.store_deployment(deployment)
            
            logger.error(f"Deployment to {node.node_id} failed: {e}")
            raise
    
    async def predict(self, request: EdgePredictionRequest) -> EdgePredictionResponse:
        """Make prediction using edge models"""
        try:
            start_time = datetime.utcnow()
            
            # Select best edge node for prediction
            node = await self.select_prediction_node(request.model_id, request.region)
            
            if not node:
                raise ValueError(f"No active edge node available for model {request.model_id}")
            
            # Check cache first
            cache_key = self.generate_cache_key(request)
            cached_result = await self.get_cached_prediction(cache_key)
            
            if cached_result:
                return EdgePredictionResponse(
                    prediction_id=f"cached_{datetime.utcnow().timestamp()}",
                    model_id=request.model_id,
                    node_id=node.node_id,
                    predictions=cached_result["predictions"],
                    confidence=cached_result.get("confidence"),
                    inference_time_ms=0.0,
                    total_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                    cached=True
                )
            
            # Make prediction request to edge node
            prediction_payload = {
                "model_id": request.model_id,
                "input_data": request.input_data,
                "request_id": request.request_id
            }
            
            async with aiohttp.ClientSession() as session:
                predict_url = f"{node.endpoint_url}/predict"
                timeout = aiohttp.ClientTimeout(total=(request.timeout_ms or 5000) / 1000)
                
                async with session.post(
                    predict_url,
                    json=prediction_payload,
                    timeout=timeout
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Create response
                        prediction_response = EdgePredictionResponse(
                            prediction_id=result.get("prediction_id", f"pred_{datetime.utcnow().timestamp()}"),
                            model_id=request.model_id,
                            node_id=node.node_id,
                            predictions=result["predictions"],
                            confidence=result.get("confidence"),
                            inference_time_ms=result.get("inference_time_ms", 0.0),
                            total_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                        )
                        
                        # Cache result if configured
                        model_config = self.model_registry.get(request.model_id)
                        if model_config and model_config.cache_predictions:
                            await self.cache_prediction(cache_key, result, ttl=300)  # 5 minutes
                        
                        # Update node metrics
                        await self.update_node_metrics(node.node_id, True, prediction_response.inference_time_ms)
                        
                        return prediction_response
                    else:
                        error_msg = f"Prediction failed with status {response.status}"
                        await self.update_node_metrics(node.node_id, False, 0.0)
                        raise RuntimeError(error_msg)
            
        except Exception as e:
            logger.error(f"Edge prediction failed: {e}")
            raise
    
    async def select_optimal_nodes(self, model_config: EdgeModelConfig) -> List[EdgeNode]:
        """Select optimal edge nodes for model deployment"""
        try:
            suitable_nodes = []
            
            for node in self.edge_nodes.values():
                # Check if node is active
                if node.status != EdgeNodeStatus.ACTIVE:
                    continue
                
                # Check region targeting
                if model_config.target_regions and node.region not in model_config.target_regions:
                    continue
                
                # Check resource requirements
                if not self.check_resource_requirements(node, model_config):
                    continue
                
                # Check capabilities
                required_capabilities = model_config.model_format.value
                if required_capabilities not in node.capabilities:
                    continue
                
                suitable_nodes.append(node)
            
            # Sort by load and performance
            suitable_nodes.sort(key=lambda n: (n.current_load, -n.success_rate))
            
            return suitable_nodes[:3]  # Deploy to top 3 nodes
            
        except Exception as e:
            logger.error(f"Node selection failed: {e}")
            return []
    
    async def select_prediction_node(
        self,
        model_id: str,
        preferred_region: Optional[EdgeRegion] = None
    ) -> Optional[EdgeNode]:
        """Select best edge node for prediction"""
        try:
            candidate_nodes = []
            
            for node in self.edge_nodes.values():
                # Check if node has the model deployed
                if model_id not in node.deployed_models:
                    continue
                
                # Check if node is active
                if node.status != EdgeNodeStatus.ACTIVE:
                    continue
                
                # Check recent heartbeat
                if datetime.utcnow() - node.last_heartbeat > timedelta(minutes=5):
                    continue
                
                candidate_nodes.append(node)
            
            if not candidate_nodes:
                return None
            
            # Prefer nodes in the requested region
            if preferred_region:
                region_nodes = [n for n in candidate_nodes if n.region == preferred_region]
                if region_nodes:
                    candidate_nodes = region_nodes
            
            # Select node with best performance
            best_node = min(
                candidate_nodes,
                key=lambda n: (n.current_load, n.avg_response_time_ms, -n.success_rate)
            )
            
            return best_node
            
        except Exception as e:
            logger.error(f"Prediction node selection failed: {e}")
            return None
    
    def check_resource_requirements(self, node: EdgeNode, model_config: EdgeModelConfig) -> bool:
        """Check if node meets model resource requirements"""
        requirements = model_config.resource_requirements
        
        if requirements.get("min_cpu_cores", 0) > node.cpu_cores:
            return False
        
        if requirements.get("min_memory_gb", 0) > node.memory_gb:
            return False
        
        if requirements.get("min_storage_gb", 0) > node.storage_gb:
            return False
        
        return True
    
    async def upload_model_file(self, file_path: str, model_config: EdgeModelConfig) -> str:
        """Upload model file to distributed storage"""
        try:
            # In a real implementation, this would upload to S3, GCS, or similar
            # For now, we'll simulate with a local file server
            
            filename = f"{model_config.model_id}_{model_config.model_version}.{model_config.model_format.value}"
            
            # Simulate upload (replace with actual cloud storage upload)
            storage_url = f"https://edge-models.rental-ml.com/{filename}"
            
            logger.info(f"Model uploaded: {storage_url}")
            return storage_url
            
        except Exception as e:
            logger.error(f"Model upload failed: {e}")
            raise
    
    async def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of model file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            raise
    
    def generate_cache_key(self, request: EdgePredictionRequest) -> str:
        """Generate cache key for prediction request"""
        input_hash = hashlib.md5(
            json.dumps(request.input_data, sort_keys=True).encode()
        ).hexdigest()
        
        return f"edge_pred:{request.model_id}:{input_hash}"
    
    async def get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction result"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception:
            return None
    
    async def cache_prediction(
        self,
        cache_key: str,
        prediction_result: Dict[str, Any],
        ttl: int = 300
    ) -> None:
        """Cache prediction result"""
        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(prediction_result)
            )
        except Exception as e:
            logger.error(f"Prediction caching failed: {e}")
    
    async def store_deployment(self, deployment: ModelDeployment) -> None:
        """Store deployment information"""
        try:
            deployment_key = f"deployment:{deployment.deployment_id}"
            await self.redis_client.setex(
                deployment_key,
                86400 * 30,  # 30 days
                deployment.model_dump_json()
            )
            
            # Add to deployments index
            model_deployments_key = f"model_deployments:{deployment.model_config.model_id}"
            await self.redis_client.sadd(model_deployments_key, deployment.deployment_id)
            
        except Exception as e:
            logger.error(f"Failed to store deployment: {e}")
    
    async def store_model_config(self, model_config: EdgeModelConfig) -> None:
        """Store model configuration"""
        try:
            config_key = f"model_config:{model_config.model_id}"
            await self.redis_client.setex(
                config_key,
                86400 * 365,  # 1 year
                model_config.model_dump_json()
            )
        except Exception as e:
            logger.error(f"Failed to store model config: {e}")
    
    async def update_edge_node(self, node: EdgeNode) -> None:
        """Update edge node information"""
        try:
            node.last_heartbeat = datetime.utcnow()
            node_key = f"edge_node:{node.node_id}"
            await self.redis_client.setex(
                node_key,
                3600,  # 1 hour
                node.model_dump_json()
            )
            
            self.edge_nodes[node.node_id] = node
            
        except Exception as e:
            logger.error(f"Failed to update edge node: {e}")
    
    async def update_node_metrics(
        self,
        node_id: str,
        success: bool,
        response_time_ms: float
    ) -> None:
        """Update node performance metrics"""
        try:
            node = self.edge_nodes.get(node_id)
            if not node:
                return
            
            # Update metrics (simplified moving average)
            alpha = 0.1  # Smoothing factor
            
            if success:
                node.avg_response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * node.avg_response_time_ms
                )
            
            # Update success rate
            total_requests = await self.redis_client.incr(f"node_requests:{node_id}")
            if success:
                successful_requests = await self.redis_client.incr(f"node_success:{node_id}")
            else:
                successful_requests = await self.redis_client.get(f"node_success:{node_id}") or 0
                successful_requests = int(successful_requests)
            
            node.success_rate = successful_requests / total_requests if total_requests > 0 else 1.0
            
            await self.update_edge_node(node)
            
        except Exception as e:
            logger.error(f"Failed to update node metrics: {e}")
    
    async def health_check_loop(self) -> None:
        """Background health check for edge nodes"""
        while True:
            try:
                await asyncio.sleep(self.config["health_check_interval"])
                
                for node in list(self.edge_nodes.values()):
                    try:
                        # Check node health
                        async with aiohttp.ClientSession() as session:
                            health_url = f"{node.endpoint_url}/health"
                            
                            async with session.get(
                                health_url,
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as response:
                                
                                if response.status == 200:
                                    health_data = await response.json()
                                    
                                    # Update node status
                                    node.status = EdgeNodeStatus.ACTIVE
                                    node.current_load = health_data.get("cpu_usage", 0.0)
                                    node.last_heartbeat = datetime.utcnow()
                                    
                                    await self.update_edge_node(node)
                                else:
                                    # Mark node as inactive
                                    node.status = EdgeNodeStatus.INACTIVE
                                    await self.update_edge_node(node)
                                    
                    except Exception as e:
                        logger.warning(f"Health check failed for {node.node_id}: {e}")
                        node.status = EdgeNodeStatus.ERROR
                        await self.update_edge_node(node)
                        
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def cleanup_old_deployments(self) -> None:
        """Clean up old deployment records"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                
                # Get all deployment keys
                pattern = "deployment:*"
                keys = await self.redis_client.keys(pattern)
                
                for key in keys:
                    deployment_data = await self.redis_client.get(key)
                    if deployment_data:
                        try:
                            deployment = ModelDeployment.model_validate_json(deployment_data)
                            if deployment.created_at < cutoff_time:
                                await self.redis_client.delete(key)
                                logger.info(f"Cleaned up old deployment: {deployment.deployment_id}")
                        except Exception:
                            # Delete invalid deployment data
                            await self.redis_client.delete(key)
                            
            except Exception as e:
                logger.error(f"Deployment cleanup error: {e}")
    
    async def get_deployment_status(self, model_id: str) -> Dict[str, Any]:
        """Get deployment status for a model"""
        try:
            deployments_key = f"model_deployments:{model_id}"
            deployment_ids = await self.redis_client.smembers(deployments_key)
            
            deployments = []
            for deployment_id in deployment_ids:
                deployment_key = f"deployment:{deployment_id}"
                deployment_data = await self.redis_client.get(deployment_key)
                
                if deployment_data:
                    deployment = ModelDeployment.model_validate_json(deployment_data)
                    deployments.append(deployment)
            
            # Aggregate status
            active_deployments = [d for d in deployments if d.status == ModelStatus.ACTIVE]
            failed_deployments = [d for d in deployments if d.status == ModelStatus.FAILED]
            
            return {
                "model_id": model_id,
                "total_deployments": len(deployments),
                "active_deployments": len(active_deployments),
                "failed_deployments": len(failed_deployments),
                "deployments": [d.model_dump() for d in deployments]
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    async def rollback_deployment(self, model_id: str, target_version: str) -> bool:
        """Rollback model to previous version"""
        try:
            # Get current deployments
            status = await self.get_deployment_status(model_id)
            active_deployments = [
                d for d in status["deployments"] 
                if d["status"] == ModelStatus.ACTIVE.value
            ]
            
            # Find target version deployments
            target_deployments = [
                d for d in status["deployments"]
                if d["model_config"]["model_version"] == target_version
            ]
            
            if not target_deployments:
                logger.error(f"Target version {target_version} not found for model {model_id}")
                return False
            
            # Deactivate current deployments and activate target version
            rollback_tasks = []
            
            for deployment in active_deployments:
                task = self.deactivate_deployment(deployment["deployment_id"])
                rollback_tasks.append(task)
            
            for deployment in target_deployments:
                task = self.activate_deployment(deployment["deployment_id"])
                rollback_tasks.append(task)
            
            results = await asyncio.gather(*rollback_tasks, return_exceptions=True)
            
            success = all(not isinstance(r, Exception) and r for r in results)
            
            if success:
                logger.info(f"Successfully rolled back {model_id} to version {target_version}")
            else:
                logger.error(f"Rollback failed for {model_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def deactivate_deployment(self, deployment_id: str) -> bool:
        """Deactivate a specific deployment"""
        try:
            deployment_key = f"deployment:{deployment_id}"
            deployment_data = await self.redis_client.get(deployment_key)
            
            if not deployment_data:
                return False
            
            deployment = ModelDeployment.model_validate_json(deployment_data)
            deployment.status = ModelStatus.INACTIVE
            
            await self.redis_client.setex(
                deployment_key,
                86400 * 30,
                deployment.model_dump_json()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate deployment: {e}")
            return False
    
    async def activate_deployment(self, deployment_id: str) -> bool:
        """Activate a specific deployment"""
        try:
            deployment_key = f"deployment:{deployment_id}"
            deployment_data = await self.redis_client.get(deployment_key)
            
            if not deployment_data:
                return False
            
            deployment = ModelDeployment.model_validate_json(deployment_data)
            deployment.status = ModelStatus.ACTIVE
            
            await self.redis_client.setex(
                deployment_key,
                86400 * 30,
                deployment.model_dump_json()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate deployment: {e}")
            return False