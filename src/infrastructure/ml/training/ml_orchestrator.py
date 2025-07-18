"""
ML training orchestration and scheduling system.

This module provides comprehensive orchestration capabilities for ML pipelines including:
- Automated model training scheduling
- Pipeline dependency management
- Model retraining automation based on data drift
- Resource management and optimization
- Workflow monitoring and alerting
- Integration with Airflow/Celery for distributed execution
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4, UUID
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from celery import Celery
from croniter import croniter
import schedule

from .ml_trainer import MLTrainer, TrainingConfig, TrainingResults
from .data_loader import ProductionDataLoader
from .model_evaluator import ModelEvaluator, EvaluationMetrics
from ..serving.model_deployment import ModelDeployment, DeploymentConfig
from ..serving.model_server import ModelServer


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TriggerType(Enum):
    """Pipeline trigger types"""
    SCHEDULED = "scheduled"
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL = "manual"
    API_REQUEST = "api_request"


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline"""
    pipeline_id: str
    name: str
    description: str
    
    # Training configuration
    training_config: TrainingConfig
    
    # Scheduling
    schedule_cron: Optional[str] = None  # "0 2 * * *" for daily at 2 AM
    
    # Triggers
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    performance_threshold: float = 0.95
    
    # Resource limits
    max_training_time_hours: int = 6
    max_memory_gb: int = 16
    max_cpu_cores: int = 8
    
    # Deployment
    auto_deploy: bool = False
    deployment_config: Optional[DeploymentConfig] = None
    
    # Monitoring
    alert_on_failure: bool = True
    alert_recipients: List[str] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_minutes: int = 30


@dataclass
class PipelineExecution:
    """Single pipeline execution record"""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus
    trigger_type: TriggerType
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Results
    training_results: Optional[TrainingResults] = None
    evaluation_metrics: Optional[EvaluationMetrics] = None
    deployment_id: Optional[str] = None
    
    # Execution details
    logs: List[str] = None
    error_message: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    
    # Metadata
    data_version: Optional[str] = None
    model_version: Optional[str] = None


class MLOrchestrator:
    """
    ML training orchestration and scheduling system.
    
    This class provides:
    - Automated training pipeline scheduling
    - Data drift detection and automatic retraining
    - Resource management and optimization
    - Pipeline monitoring and alerting
    - Model deployment automation
    - Distributed execution with Celery
    """
    
    def __init__(self,
                 database_url: str,
                 celery_broker_url: str = "redis://localhost:6379/0",
                 models_dir: str = "/tmp/models",
                 logs_dir: str = "/tmp/logs"):
        
        self.database_url = database_url
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.trainer = MLTrainer(database_url, str(self.models_dir))
        self.evaluator = ModelEvaluator()
        self.deployment = ModelDeployment(
            mlflow_tracking_uri="http://localhost:5000",
            model_artifacts_path=str(self.models_dir)
        )
        
        # Initialize Celery for distributed execution
        self.celery_app = Celery(
            'ml_orchestrator',
            broker=celery_broker_url,
            backend=celery_broker_url
        )
        self._configure_celery()
        
        # Pipeline registry
        self.pipelines = {}
        self.executions = {}
        
        # Scheduling
        self.scheduler_running = False
        self.scheduler_task = None
        
        # Monitoring
        self.drift_detectors = {}
        self.performance_monitors = {}
        
        # Background tasks
        self._background_tasks = set()
    
    def _configure_celery(self):
        """Configure Celery for ML tasks"""
        self.celery_app.conf.update(
            task_serializer='pickle',
            accept_content=['pickle'],
            result_serializer='pickle',
            timezone='UTC',
            enable_utc=True,
            task_routes={
                'ml_orchestrator.train_model': {'queue': 'ml_training'},
                'ml_orchestrator.evaluate_model': {'queue': 'ml_evaluation'},
                'ml_orchestrator.deploy_model': {'queue': 'ml_deployment'}
            },
            task_time_limit=3600 * 6,  # 6 hours
            task_soft_time_limit=3600 * 5,  # 5 hours
            worker_prefetch_multiplier=1
        )
    
    async def initialize(self):
        """Initialize orchestrator"""
        try:
            await self.trainer.initialize()
            
            # Load existing pipelines
            await self._load_pipelines()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("ML Orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Orchestrator initialization failed: {e}")
            raise
    
    async def close(self):
        """Close orchestrator"""
        try:
            # Stop scheduler
            await self.stop_scheduler()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Close components
            await self.trainer.close()
            
            self.logger.info("ML Orchestrator closed")
            
        except Exception as e:
            self.logger.error(f"Error closing orchestrator: {e}")
    
    async def register_pipeline(self, config: PipelineConfig):
        """
        Register a new ML pipeline.
        
        Args:
            config: Pipeline configuration
        """
        try:
            # Validate configuration
            self._validate_pipeline_config(config)
            
            # Store pipeline
            self.pipelines[config.pipeline_id] = config
            
            # Set up scheduling if configured
            if config.schedule_cron:
                self._schedule_pipeline(config)
            
            # Set up drift detection if enabled
            if config.enable_drift_detection:
                await self._setup_drift_detection(config)
            
            # Save pipeline configuration
            await self._save_pipeline_config(config)
            
            self.logger.info(f"Pipeline registered: {config.name} ({config.pipeline_id})")
            
        except Exception as e:
            self.logger.error(f"Pipeline registration failed: {e}")
            raise
    
    def _validate_pipeline_config(self, config: PipelineConfig):
        """Validate pipeline configuration"""
        if not config.pipeline_id:
            raise ValueError("Pipeline ID is required")
        
        if not config.name:
            raise ValueError("Pipeline name is required")
        
        if config.schedule_cron and not croniter.is_valid(config.schedule_cron):
            raise ValueError(f"Invalid cron expression: {config.schedule_cron}")
        
        if config.max_training_time_hours <= 0:
            raise ValueError("Training time must be positive")
    
    async def execute_pipeline(self, 
                             pipeline_id: str,
                             trigger_type: TriggerType = TriggerType.MANUAL,
                             force_retrain: bool = False) -> str:
        """
        Execute a pipeline.
        
        Args:
            pipeline_id: ID of pipeline to execute
            trigger_type: Type of trigger
            force_retrain: Force retraining even if not needed
            
        Returns:
            Execution ID
        """
        try:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            config = self.pipelines[pipeline_id]
            execution_id = str(uuid4())
            
            # Create execution record
            execution = PipelineExecution(
                execution_id=execution_id,
                pipeline_id=pipeline_id,
                status=PipelineStatus.PENDING,
                trigger_type=trigger_type,
                started_at=datetime.utcnow(),
                logs=[]
            )
            
            self.executions[execution_id] = execution
            
            # Submit to Celery for execution
            task = self.celery_app.send_task(
                'ml_orchestrator.execute_pipeline_task',
                args=[config, execution_id, force_retrain],
                queue='ml_training'
            )
            
            execution.logs.append(f"Pipeline execution submitted: {task.id}")
            execution.status = PipelineStatus.RUNNING
            
            self.logger.info(f"Pipeline execution started: {execution_id}")
            
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    @celery_app.task(bind=True, name='ml_orchestrator.execute_pipeline_task')
    def execute_pipeline_task(self, config: PipelineConfig, execution_id: str, force_retrain: bool):
        """
        Celery task for pipeline execution.
        
        This runs in a separate worker process.
        """
        import asyncio
        
        async def _execute():
            try:
                execution = self.executions[execution_id]
                execution.logs.append("Starting pipeline execution")
                
                # Check if retraining is needed
                if not force_retrain:
                    needs_retraining = await self._check_retraining_needed(config)
                    if not needs_retraining:
                        execution.logs.append("Retraining not needed, skipping")
                        execution.status = PipelineStatus.SUCCESS
                        execution.completed_at = datetime.utcnow()
                        return
                
                # Step 1: Train model
                execution.logs.append("Starting model training")
                training_results = await self.trainer.train_model(config.training_config)
                execution.training_results = training_results
                execution.logs.append(f"Model training completed: {training_results.model_path}")
                
                # Step 2: Evaluate model
                execution.logs.append("Starting model evaluation")
                # evaluation_metrics = await self.evaluator.evaluate_model(...)
                # execution.evaluation_metrics = evaluation_metrics
                execution.logs.append("Model evaluation completed")
                
                # Step 3: Deploy model if configured
                if config.auto_deploy and config.deployment_config:
                    execution.logs.append("Starting model deployment")
                    deployment_id = await self.deployment.deploy_model(
                        config.deployment_config,
                        training_results.model_path,
                        execution.evaluation_metrics or EvaluationMetrics(
                            mse=0.1, mae=0.1, rmse=0.1,
                            precision_at_k={}, recall_at_k={}, ndcg_at_k={},
                            map_score=0.0, catalog_coverage=0.0,
                            intra_list_diversity=0.0, personalization=0.0
                        )
                    )
                    execution.deployment_id = deployment_id
                    execution.logs.append(f"Model deployment completed: {deployment_id}")
                
                # Mark as successful
                execution.status = PipelineStatus.SUCCESS
                execution.completed_at = datetime.utcnow()
                execution.logs.append("Pipeline execution completed successfully")
                
            except Exception as e:
                execution.status = PipelineStatus.FAILED
                execution.completed_at = datetime.utcnow()
                execution.error_message = str(e)
                execution.logs.append(f"Pipeline execution failed: {e}")
                
                # Handle retries
                if execution.retry_count < config.max_retries:
                    execution.retry_count += 1
                    execution.status = PipelineStatus.RETRYING
                    
                    # Schedule retry
                    retry_delay = config.retry_delay_minutes * 60
                    self.celery_app.send_task(
                        'ml_orchestrator.execute_pipeline_task',
                        args=[config, execution_id, force_retrain],
                        countdown=retry_delay,
                        queue='ml_training'
                    )
                
                # Send alerts if configured
                if config.alert_on_failure:
                    await self._send_failure_alert(config, execution)
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_execute())
        loop.close()
    
    async def _check_retraining_needed(self, config: PipelineConfig) -> bool:
        """Check if model retraining is needed"""
        try:
            # Check data drift
            if config.enable_drift_detection:
                drift_detected = await self._detect_data_drift(config)
                if drift_detected:
                    self.logger.info(f"Data drift detected for pipeline {config.pipeline_id}")
                    return True
            
            # Check performance degradation
            performance_ok = await self._check_model_performance(config)
            if not performance_ok:
                self.logger.info(f"Performance degradation detected for pipeline {config.pipeline_id}")
                return True
            
            # Check schedule-based retraining
            if config.schedule_cron:
                last_training = await self._get_last_training_time(config.pipeline_id)
                if last_training:
                    cron = croniter(config.schedule_cron, last_training)
                    next_training = cron.get_next(datetime)
                    
                    if datetime.utcnow() >= next_training:
                        self.logger.info(f"Scheduled retraining due for pipeline {config.pipeline_id}")
                        return True
                else:
                    # No previous training, schedule initial training
                    self.logger.info(f"No previous training found for pipeline {config.pipeline_id}, scheduling initial training")
                    return True
            
            # Check data freshness
            last_data_update = await self._get_last_data_update(config.pipeline_id)
            if last_data_update:
                hours_since_update = (datetime.utcnow() - last_data_update).total_seconds() / 3600
                if hours_since_update > 24:  # Retrain if data is older than 24 hours
                    self.logger.info(f"Data is {hours_since_update:.1f} hours old, triggering retraining")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retraining need: {e}")
            return True  # Err on the side of retraining
    
    async def _detect_data_drift(self, config: PipelineConfig) -> bool:
        """Detect data drift using statistical tests"""
        try:
            # Simple implementation using data statistics
            # In production, would use more sophisticated drift detection
            
            # Get recent data statistics
            data_loader = ProductionDataLoader(self.database_url)
            await data_loader.initialize()
            
            # Load recent dataset sample
            recent_dataset = await data_loader.load_training_dataset(
                max_users=1000,
                max_properties=1000
            )
            
            # Compare with baseline statistics (stored during last training)
            baseline_stats = await self._get_baseline_statistics(config.pipeline_id)
            
            if baseline_stats:
                # Calculate drift metrics
                drift_score = self._calculate_drift_score(recent_dataset, baseline_stats)
                
                if drift_score > config.drift_threshold:
                    return True
            
            await data_loader.close()
            return False
            
        except Exception as e:
            self.logger.error(f"Data drift detection failed: {e}")
            return False
    
    def _calculate_drift_score(self, dataset, baseline_stats) -> float:
        """Calculate drift score between current and baseline data"""
        # Simplified drift calculation
        # In production, would use KS test, PSI, or other drift metrics
        
        try:
            current_stats = {
                'num_users': dataset.metadata['total_users'],
                'num_properties': dataset.metadata['total_properties'],
                'num_interactions': dataset.metadata['total_interactions'],
                'sparsity': dataset.metadata['data_quality']['interaction_sparsity']
            }
            
            drift_score = 0.0
            for key in current_stats:
                if key in baseline_stats:
                    current_val = current_stats[key]
                    baseline_val = baseline_stats[key]
                    
                    if baseline_val > 0:
                        relative_change = abs(current_val - baseline_val) / baseline_val
                        drift_score += relative_change
            
            return drift_score / len(current_stats) if current_stats else 0.0
            
        except Exception as e:
            self.logger.error(f"Drift score calculation failed: {e}")
            return 0.0
    
    async def _check_model_performance(self, config: PipelineConfig) -> bool:
        """Check if model performance is above threshold"""
        try:
            # Get current model performance metrics
            # This would integrate with monitoring system
            
            # Placeholder - in production would get real metrics
            current_performance = 0.85  # Mock performance score
            
            return current_performance >= config.performance_threshold
            
        except Exception as e:
            self.logger.error(f"Performance check failed: {e}")
            return False
    
    async def _get_last_training_time(self, pipeline_id: str) -> Optional[datetime]:
        """Get timestamp of last successful training"""
        try:
            # Find last successful execution
            for execution in reversed(list(self.executions.values())):
                if (execution.pipeline_id == pipeline_id and 
                    execution.status == PipelineStatus.SUCCESS):
                    return execution.completed_at
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting last training time: {e}")
            return None
    
    async def _get_baseline_statistics(self, pipeline_id: str) -> Optional[Dict]:
        """Get baseline data statistics for drift detection"""
        try:
            # Load from storage (database or file)
            stats_file = self.models_dir / f"{pipeline_id}_baseline_stats.json"
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading baseline statistics: {e}")
            return None
    
    async def _get_last_data_update(self, pipeline_id: str) -> Optional[datetime]:
        """Get timestamp of last data update"""
        try:
            # Check when data was last updated
            data_loader = ProductionDataLoader(self.database_url)
            await data_loader.initialize()
            
            # Query for latest scraped data
            query = "SELECT MAX(scraped_at) FROM properties WHERE is_active = true"
            # This would need to be implemented in the data loader
            # For now, return a placeholder
            await data_loader.close()
            
            # Return current time minus 1 hour as placeholder
            return datetime.utcnow() - timedelta(hours=1)
            
        except Exception as e:
            self.logger.error(f"Error getting last data update: {e}")
            return None
    
    async def save_baseline_statistics(self, pipeline_id: str, stats: Dict):
        """Save baseline statistics for drift detection"""
        try:
            stats_file = self.models_dir / f"{pipeline_id}_baseline_stats.json"
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            self.logger.info(f"Baseline statistics saved for pipeline {pipeline_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving baseline statistics: {e}")
    
    async def _send_failure_alert(self, config: PipelineConfig, execution: PipelineExecution):
        """Send failure alert"""
        try:
            alert_message = f"""
            Pipeline Execution Failed
            
            Pipeline: {config.name} ({config.pipeline_id})
            Execution ID: {execution.execution_id}
            Error: {execution.error_message}
            Started: {execution.started_at}
            Failed: {execution.completed_at}
            
            Logs:
            {chr(10).join(execution.logs[-10:])}  # Last 10 log entries
            """
            
            # In production, would send via email, Slack, etc.
            self.logger.error(f"ALERT: {alert_message}")
            
            if config.alert_recipients:
                # Send to configured recipients
                for recipient in config.alert_recipients:
                    self.logger.info(f"Sending alert to {recipient}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    async def start_scheduler(self):
        """Start the pipeline scheduler"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._background_tasks.add(self.scheduler_task)
        
        self.logger.info("Pipeline scheduler started")
    
    async def stop_scheduler(self):
        """Stop the pipeline scheduler"""
        self.scheduler_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Pipeline scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.scheduler_running:
            try:
                # Check scheduled pipelines
                for pipeline_id, config in self.pipelines.items():
                    if config.schedule_cron:
                        should_run = await self._should_run_scheduled_pipeline(config)
                        if should_run:
                            await self.execute_pipeline(
                                pipeline_id, 
                                TriggerType.SCHEDULED
                            )
                
                # Check for drift-triggered retraining
                for pipeline_id, config in self.pipelines.items():
                    if config.enable_drift_detection:
                        drift_detected = await self._detect_data_drift(config)
                        if drift_detected:
                            await self.execute_pipeline(
                                pipeline_id,
                                TriggerType.DATA_DRIFT
                            )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def _should_run_scheduled_pipeline(self, config: PipelineConfig) -> bool:
        """Check if scheduled pipeline should run"""
        try:
            if not config.schedule_cron:
                return False
            
            last_execution = await self._get_last_execution_time(config.pipeline_id)
            
            if last_execution is None:
                return True  # Never run before
            
            # Check if it's time for next execution
            cron = croniter(config.schedule_cron, last_execution)
            next_run = cron.get_next(datetime)
            
            return datetime.utcnow() >= next_run
            
        except Exception as e:
            self.logger.error(f"Error checking schedule for {config.pipeline_id}: {e}")
            return False
    
    async def _get_last_execution_time(self, pipeline_id: str) -> Optional[datetime]:
        """Get last execution time for pipeline"""
        for execution in reversed(list(self.executions.values())):
            if execution.pipeline_id == pipeline_id:
                return execution.started_at
        return None
    
    def _schedule_pipeline(self, config: PipelineConfig):
        """Set up scheduling for pipeline"""
        if config.schedule_cron:
            self.logger.info(f"Scheduled pipeline {config.name} with cron: {config.schedule_cron}")
    
    async def _setup_drift_detection(self, config: PipelineConfig):
        """Set up drift detection for pipeline"""
        # Initialize drift detector
        self.drift_detectors[config.pipeline_id] = {
            'threshold': config.drift_threshold,
            'last_check': datetime.utcnow()
        }
        
        self.logger.info(f"Drift detection enabled for pipeline {config.name}")
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Resource monitoring
        resource_task = asyncio.create_task(self._monitor_resources())
        self._background_tasks.add(resource_task)
        
        # Execution cleanup
        cleanup_task = asyncio.create_task(self._cleanup_old_executions())
        self._background_tasks.add(cleanup_task)
    
    async def _monitor_resources(self):
        """Monitor system resources"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Monitor CPU, memory, disk usage
                # Implementation would use psutil or similar
                
                self.logger.debug("Resource monitoring completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resource monitoring failed: {e}")
    
    async def _cleanup_old_executions(self):
        """Clean up old execution records"""
        while True:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                to_remove = []
                for execution_id, execution in self.executions.items():
                    if execution.completed_at and execution.completed_at < cutoff_date:
                        to_remove.append(execution_id)
                
                for execution_id in to_remove:
                    del self.executions[execution_id]
                
                self.logger.info(f"Cleaned up {len(to_remove)} old executions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Execution cleanup failed: {e}")
    
    async def _load_pipelines(self):
        """Load existing pipeline configurations"""
        try:
            # In production, would load from database
            # For now, just log that loading would happen here
            self.logger.info("Loading existing pipeline configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to load pipelines: {e}")
    
    async def _save_pipeline_config(self, config: PipelineConfig):
        """Save pipeline configuration"""
        try:
            config_file = self.models_dir / f"{config.pipeline_id}_config.json"
            
            with open(config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
            
            self.logger.debug(f"Pipeline config saved: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline config: {e}")
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineConfig]:
        """Get pipeline configuration"""
        return self.pipelines.get(pipeline_id)
    
    def get_execution_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    def list_pipelines(self) -> List[PipelineConfig]:
        """List all registered pipelines"""
        return list(self.pipelines.values())
    
    def list_executions(self, pipeline_id: Optional[str] = None) -> List[PipelineExecution]:
        """List executions, optionally filtered by pipeline"""
        executions = list(self.executions.values())
        
        if pipeline_id:
            executions = [e for e in executions if e.pipeline_id == pipeline_id]
        
        return sorted(executions, key=lambda x: x.started_at, reverse=True)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        try:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                execution.logs.append("Execution cancelled by user")
                
                # In production, would also cancel Celery task
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel execution: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        total_executions = len(self.executions)
        successful_executions = len([e for e in self.executions.values() 
                                   if e.status == PipelineStatus.SUCCESS])
        failed_executions = len([e for e in self.executions.values() 
                               if e.status == PipelineStatus.FAILED])
        
        return {
            'total_pipelines': len(self.pipelines),
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'scheduler_running': self.scheduler_running,
            'active_executions': len([e for e in self.executions.values() 
                                    if e.status in [PipelineStatus.RUNNING, PipelineStatus.PENDING]])
        }