"""
Model Performance Monitoring for Production ML Systems

This module provides comprehensive monitoring capabilities for deployed ML models
including real-time prediction quality monitoring, data drift detection,
model degradation alerts, A/B testing framework, and performance dashboards.

Features:
- Real-time prediction quality monitoring
- Data drift detection using statistical methods
- Model degradation alerts and notifications
- A/B testing framework for model comparisons
- Performance dashboards and reporting
- Automated model retraining triggers
- Integration with monitoring systems (Prometheus, Grafana)
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
from enum import Enum
from uuid import uuid4, UUID
import statistics
from collections import deque, defaultdict
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Domain imports
from ...domain.repositories.model_repository import ModelRepository
from .model_registry import ModelRegistry


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringMetric(Enum):
    """Types of monitoring metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    AUC = "auc"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftDetectionMethod(Enum):
    """Data drift detection methods"""
    KOLMOGOROV_SMIRNOV = "ks_test"
    POPULATION_STABILITY_INDEX = "psi"
    JENSEN_SHANNON_DIVERGENCE = "js_divergence"
    WASSERSTEIN_DISTANCE = "wasserstein"
    CHI_SQUARE = "chi_square"


@dataclass
class MonitoringAlert:
    """Monitoring alert information"""
    alert_id: str
    model_name: str
    model_version: str
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    additional_data: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""
    feature_name: str
    drift_score: float
    p_value: float
    is_drift_detected: bool
    method_used: DriftDetectionMethod
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]
    timestamp: datetime


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance at a point in time"""
    model_name: str
    model_version: str
    timestamp: datetime
    metrics: Dict[str, float]
    predictions_count: int
    errors_count: int
    latency_p95: float
    latency_p99: float
    drift_scores: Dict[str, float]
    alerts_count: int


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    model_name: str
    control_version: str
    treatment_version: str
    traffic_split: Dict[str, float]  # version -> percentage
    success_metrics: List[str]
    duration_days: int
    start_date: datetime
    end_date: datetime
    min_sample_size: int = 1000
    confidence_level: float = 0.95


@dataclass
class ABTestResult:
    """A/B test result analysis"""
    test_id: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    statistical_significance: Dict[str, bool]
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendation: str  # "promote", "continue", "stop"
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]


class ModelMonitoringService:
    """
    Comprehensive model monitoring service for production ML systems.
    
    This service provides:
    - Real-time monitoring of model performance metrics
    - Data drift detection and alerting
    - Model degradation monitoring
    - A/B testing framework
    - Performance dashboards and reporting
    - Automated remediation triggers
    """
    
    def __init__(self,
                 model_registry: ModelRegistry,
                 model_repository: ModelRepository,
                 monitoring_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model monitoring service.
        
        Args:
            model_registry: Model registry for version management
            model_repository: Repository for storing monitoring data
            monitoring_config: Monitoring configuration options
        """
        self.model_registry = model_registry
        self.model_repository = model_repository
        self.config = monitoring_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[MonitoringAlert] = []
        self.performance_history: Dict[str, List[ModelPerformanceSnapshot]] = {}
        self.drift_baselines: Dict[str, Dict[str, Any]] = {}
        
        # A/B testing
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, ABTestResult] = {}
        
        # Prediction tracking
        self.prediction_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.feature_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.latency_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Configuration
        self.max_alerts = self.config.get('max_alerts', 1000)
        self.alert_retention_days = self.config.get('alert_retention_days', 30)
        self.performance_retention_days = self.config.get('performance_retention_days', 90)
        self.drift_check_frequency = self.config.get('drift_check_frequency', 3600)  # seconds
        self.performance_check_frequency = self.config.get('performance_check_frequency', 300)  # seconds
        
        # Thresholds
        self.default_thresholds = {
            'accuracy_drop': 0.05,
            'latency_increase': 2.0,
            'error_rate_increase': 0.1,
            'drift_threshold': 0.1,
            'min_predictions_for_analysis': 100
        }
        
    async def initialize(self):
        """Initialize the monitoring service"""
        try:
            self.logger.info("Initializing model monitoring service...")
            
            # Load existing monitoring state
            await self._load_monitoring_state()
            
            # Setup monitoring tasks
            await self._setup_monitoring_tasks()
            
            # Initialize drift detection baselines
            await self._initialize_drift_baselines()
            
            self.logger.info("Model monitoring service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring service: {e}")
            raise
    
    async def setup_model_monitoring(self,
                                   model_name: str,
                                   version: str,
                                   model_type: str,
                                   monitoring_config: Dict[str, Any]):
        """
        Setup monitoring for a deployed model.
        
        Args:
            model_name: Name of the model to monitor
            version: Version of the model
            model_type: Type of the model (collaborative_filtering, content_based, etc.)
            monitoring_config: Monitoring configuration
        """
        try:
            monitor_key = f"{model_name}:{version}"
            
            # Create monitoring configuration
            monitor_config = {
                'model_name': model_name,
                'version': version,
                'model_type': model_type,
                'setup_time': datetime.utcnow(),
                'config': monitoring_config,
                'is_active': True,
                'thresholds': {**self.default_thresholds, **monitoring_config.get('thresholds', {})},
                'check_frequency': monitoring_config.get('check_frequency', 'hourly')
            }
            
            self.active_monitors[monitor_key] = monitor_config
            
            # Initialize performance history
            if monitor_key not in self.performance_history:
                self.performance_history[monitor_key] = []
            
            # Setup drift baseline
            await self._setup_drift_baseline(monitor_key, monitoring_config)
            
            self.logger.info(f"Monitoring setup for model: {monitor_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup model monitoring: {e}")
            raise
    
    async def log_prediction(self,
                           model_name: str,
                           version: str,
                           input_features: np.ndarray,
                           prediction: Union[float, np.ndarray],
                           actual_value: Optional[float] = None,
                           latency_ms: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Log a model prediction for monitoring.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            input_features: Input features used for prediction
            prediction: Model prediction
            actual_value: Actual value (if available for accuracy tracking)
            latency_ms: Prediction latency in milliseconds
            metadata: Additional metadata
        """
        try:
            monitor_key = f"{model_name}:{version}"
            
            # Store prediction data
            prediction_data = {
                'timestamp': datetime.utcnow(),
                'prediction': prediction,
                'actual_value': actual_value,
                'metadata': metadata or {}
            }
            
            self.prediction_buffer[monitor_key].append(prediction_data)
            
            # Store feature data for drift detection
            if input_features is not None:
                feature_data = {
                    'timestamp': datetime.utcnow(),
                    'features': input_features.flatten() if hasattr(input_features, 'flatten') else input_features
                }
                self.feature_buffer[monitor_key].append(feature_data)
            
            # Store latency data
            if latency_ms is not None:
                self.latency_buffer[monitor_key].append({
                    'timestamp': datetime.utcnow(),
                    'latency_ms': latency_ms
                })
            
        except Exception as e:
            self.logger.warning(f"Failed to log prediction: {e}")
    
    async def check_model_performance(self, model_name: str, version: str) -> ModelPerformanceSnapshot:
        """
        Check current model performance and create snapshot.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Performance snapshot
        """
        try:
            monitor_key = f"{model_name}:{version}"
            
            if monitor_key not in self.active_monitors:
                raise ValueError(f"Model not being monitored: {monitor_key}")
            
            # Calculate performance metrics
            metrics = await self._calculate_performance_metrics(monitor_key)
            
            # Calculate latency metrics
            latency_metrics = await self._calculate_latency_metrics(monitor_key)
            
            # Check for drift
            drift_scores = await self._calculate_drift_scores(monitor_key)
            
            # Count recent alerts
            recent_alerts = await self._count_recent_alerts(monitor_key)
            
            # Create performance snapshot
            snapshot = ModelPerformanceSnapshot(
                model_name=model_name,
                model_version=version,
                timestamp=datetime.utcnow(),
                metrics=metrics,
                predictions_count=len(self.prediction_buffer[monitor_key]),
                errors_count=metrics.get('error_count', 0),
                latency_p95=latency_metrics.get('p95', 0.0),
                latency_p99=latency_metrics.get('p99', 0.0),
                drift_scores=drift_scores,
                alerts_count=recent_alerts
            )
            
            # Store snapshot
            self.performance_history[monitor_key].append(snapshot)
            
            # Check for alerts
            await self._check_performance_alerts(monitor_key, snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to check model performance: {e}")
            raise
    
    async def detect_data_drift(self,
                              model_name: str,
                              version: str,
                              method: DriftDetectionMethod = DriftDetectionMethod.KOLMOGOROV_SMIRNOV) -> List[DriftDetectionResult]:
        """
        Detect data drift for a monitored model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            method: Drift detection method to use
            
        Returns:
            List of drift detection results for each feature
        """
        try:
            monitor_key = f"{model_name}:{version}"
            
            if monitor_key not in self.active_monitors:
                raise ValueError(f"Model not being monitored: {monitor_key}")
            
            # Get baseline and current feature data
            baseline_data = self.drift_baselines.get(monitor_key, {})
            current_features = self._get_recent_features(monitor_key)
            
            if not baseline_data or len(current_features) < self.default_thresholds['min_predictions_for_analysis']:
                return []
            
            drift_results = []
            
            # Analyze each feature for drift
            baseline_features = baseline_data.get('features', np.array([]))
            if len(baseline_features) > 0 and len(current_features) > 0:
                
                # Ensure feature dimensions match
                min_features = min(baseline_features.shape[1] if baseline_features.ndim > 1 else 1,
                                 current_features.shape[1] if current_features.ndim > 1 else 1)
                
                for feature_idx in range(min_features):
                    baseline_feature = baseline_features[:, feature_idx] if baseline_features.ndim > 1 else baseline_features
                    current_feature = current_features[:, feature_idx] if current_features.ndim > 1 else current_features
                    
                    # Perform drift detection
                    drift_result = await self._detect_feature_drift(
                        baseline_feature, current_feature, feature_idx, method
                    )
                    
                    drift_results.append(drift_result)
                    
                    # Check for drift alert
                    if drift_result.is_drift_detected:
                        await self._create_drift_alert(monitor_key, drift_result)
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Failed to detect data drift: {e}")
            return []
    
    async def start_ab_test(self,
                          model_name: str,
                          control_version: str,
                          treatment_version: str,
                          traffic_split: Dict[str, float],
                          success_metrics: List[str],
                          duration_days: int = 7) -> str:
        """
        Start an A/B test between two model versions.
        
        Args:
            model_name: Name of the model
            control_version: Control (baseline) version
            treatment_version: Treatment (candidate) version
            traffic_split: Traffic split between versions
            success_metrics: Metrics to evaluate
            duration_days: Test duration in days
            
        Returns:
            A/B test ID
        """
        try:
            test_id = f"ab_test_{model_name}_{uuid4().hex[:8]}"
            
            # Validate traffic split
            if abs(sum(traffic_split.values()) - 100.0) > 0.1:
                raise ValueError("Traffic split must sum to 100%")
            
            # Create test configuration
            test_config = ABTestConfig(
                test_id=test_id,
                model_name=model_name,
                control_version=control_version,
                treatment_version=treatment_version,
                traffic_split=traffic_split,
                success_metrics=success_metrics,
                duration_days=duration_days,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=duration_days)
            )
            
            self.active_tests[test_id] = test_config
            
            self.logger.info(f"A/B test started: {test_id}")
            
            return test_id
            
        except Exception as e:
            self.logger.error(f"Failed to start A/B test: {e}")
            raise
    
    async def analyze_ab_test(self, test_id: str) -> ABTestResult:
        """
        Analyze A/B test results.
        
        Args:
            test_id: A/B test identifier
            
        Returns:
            A/B test analysis result
        """
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"A/B test not found: {test_id}")
            
            test_config = self.active_tests[test_id]
            
            # Get performance data for both versions
            control_key = f"{test_config.model_name}:{test_config.control_version}"
            treatment_key = f"{test_config.model_name}:{test_config.treatment_version}"
            
            control_metrics = await self._calculate_performance_metrics(control_key)
            treatment_metrics = await self._calculate_performance_metrics(treatment_key)
            
            # Perform statistical analysis
            statistical_significance = {}
            confidence_intervals = {}
            p_values = {}
            effect_sizes = {}
            
            for metric in test_config.success_metrics:
                if metric in control_metrics and metric in treatment_metrics:
                    # Perform t-test (simplified analysis)
                    control_values = self._get_metric_samples(control_key, metric)
                    treatment_values = self._get_metric_samples(treatment_key, metric)
                    
                    if len(control_values) > 0 and len(treatment_values) > 0:
                        stat_result = stats.ttest_ind(control_values, treatment_values)
                        
                        statistical_significance[metric] = stat_result.pvalue < 0.05
                        p_values[metric] = stat_result.pvalue
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values) +
                                            (len(treatment_values) - 1) * np.var(treatment_values)) /
                                           (len(control_values) + len(treatment_values) - 2))
                        
                        effect_size = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
                        effect_sizes[metric] = effect_size
                        
                        # Calculate confidence interval
                        ci_lower = np.mean(treatment_values) - 1.96 * np.std(treatment_values) / np.sqrt(len(treatment_values))
                        ci_upper = np.mean(treatment_values) + 1.96 * np.std(treatment_values) / np.sqrt(len(treatment_values))
                        confidence_intervals[metric] = (ci_lower, ci_upper)
            
            # Generate recommendation
            significant_improvements = sum(1 for metric in test_config.success_metrics
                                         if statistical_significance.get(metric, False) and
                                         treatment_metrics.get(metric, 0) > control_metrics.get(metric, 0))
            
            if significant_improvements >= len(test_config.success_metrics) * 0.6:
                recommendation = "promote"
            elif significant_improvements == 0:
                recommendation = "stop"
            else:
                recommendation = "continue"
            
            # Create result
            result = ABTestResult(
                test_id=test_id,
                control_metrics=control_metrics,
                treatment_metrics=treatment_metrics,
                statistical_significance=statistical_significance,
                confidence_intervals=confidence_intervals,
                recommendation=recommendation,
                p_values=p_values,
                effect_sizes=effect_sizes
            )
            
            self.test_results[test_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze A/B test: {e}")
            raise
    
    async def get_model_dashboard_data(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Get dashboard data for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Dashboard data
        """
        try:
            monitor_key = f"{model_name}:{version}"
            
            # Get recent performance snapshots
            recent_snapshots = self.performance_history.get(monitor_key, [])[-24:]  # Last 24 snapshots
            
            # Get recent alerts
            recent_alerts = [alert for alert in self.alerts 
                           if alert.model_name == model_name and alert.model_version == version
                           and (datetime.utcnow() - alert.timestamp).days <= 7]
            
            # Calculate summary statistics
            if recent_snapshots:
                latest_snapshot = recent_snapshots[-1]
                
                # Performance trends
                performance_trend = {}
                for metric in latest_snapshot.metrics:
                    values = [snapshot.metrics.get(metric, 0) for snapshot in recent_snapshots]
                    if len(values) > 1:
                        trend = (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
                        performance_trend[metric] = trend
                
                # Latency trends
                latency_values = [snapshot.latency_p95 for snapshot in recent_snapshots]
                latency_trend = (latency_values[-1] - latency_values[0]) / latency_values[0] * 100 if latency_values[0] != 0 else 0
                
                dashboard_data = {
                    'model_name': model_name,
                    'version': version,
                    'status': 'healthy' if len([a for a in recent_alerts if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]) == 0 else 'degraded',
                    'last_updated': latest_snapshot.timestamp.isoformat(),
                    'current_metrics': latest_snapshot.metrics,
                    'performance_trends': performance_trend,
                    'latency_p95': latest_snapshot.latency_p95,
                    'latency_p99': latest_snapshot.latency_p99,
                    'latency_trend': latency_trend,
                    'predictions_per_hour': len(self.prediction_buffer[monitor_key]),
                    'error_rate': latest_snapshot.errors_count / max(latest_snapshot.predictions_count, 1),
                    'drift_scores': latest_snapshot.drift_scores,
                    'active_alerts': len([a for a in recent_alerts if not a.resolved]),
                    'recent_alerts': [asdict(alert) for alert in recent_alerts[:10]],
                    'performance_history': [asdict(snapshot) for snapshot in recent_snapshots]
                }
            else:
                dashboard_data = {
                    'model_name': model_name,
                    'version': version,
                    'status': 'no_data',
                    'last_updated': None,
                    'message': 'No monitoring data available'
                }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    async def close(self):
        """Close monitoring service and cleanup resources"""
        try:
            # Save monitoring state
            await self._save_monitoring_state()
            
            # Cancel any running tasks
            # (Implementation would cancel background monitoring tasks)
            
            self.logger.info("Model monitoring service closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing monitoring service: {e}")
    
    async def _load_monitoring_state(self):
        """Load existing monitoring state"""
        try:
            # This would load state from persistent storage
            # For now, we start with empty state
            pass
        except Exception as e:
            self.logger.warning(f"Failed to load monitoring state: {e}")
    
    async def _save_monitoring_state(self):
        """Save current monitoring state"""
        try:
            # This would save state to persistent storage
            # For now, we just log
            self.logger.info("Monitoring state saved")
        except Exception as e:
            self.logger.warning(f"Failed to save monitoring state: {e}")
    
    async def _setup_monitoring_tasks(self):
        """Setup background monitoring tasks"""
        try:
            # This would setup periodic tasks for:
            # - Performance checking
            # - Drift detection
            # - Alert cleanup
            # - A/B test analysis
            pass
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring tasks: {e}")
    
    async def _initialize_drift_baselines(self):
        """Initialize drift detection baselines"""
        try:
            # This would load historical data to establish baselines
            pass
        except Exception as e:
            self.logger.warning(f"Failed to initialize drift baselines: {e}")
    
    async def _setup_drift_baseline(self, monitor_key: str, config: Dict[str, Any]):
        """Setup drift baseline for a specific model"""
        try:
            # This would establish baseline feature distributions
            self.drift_baselines[monitor_key] = {
                'setup_time': datetime.utcnow(),
                'features': np.array([]),
                'feature_stats': {}
            }
        except Exception as e:
            self.logger.warning(f"Failed to setup drift baseline: {e}")
    
    async def _calculate_performance_metrics(self, monitor_key: str) -> Dict[str, float]:
        """Calculate performance metrics from recent predictions"""
        try:
            predictions = list(self.prediction_buffer[monitor_key])
            
            if not predictions:
                return {}
            
            metrics = {}
            
            # Calculate accuracy metrics (if actual values available)
            predictions_with_actual = [p for p in predictions if p['actual_value'] is not None]
            
            if predictions_with_actual:
                pred_values = [p['prediction'] for p in predictions_with_actual]
                actual_values = [p['actual_value'] for p in predictions_with_actual]
                
                # Handle different prediction types
                if all(isinstance(p, (int, float)) for p in pred_values):
                    # Regression metrics
                    metrics['mse'] = mean_squared_error(actual_values, pred_values)
                    metrics['mae'] = mean_absolute_error(actual_values, pred_values)
                    metrics['rmse'] = np.sqrt(metrics['mse'])
                else:
                    # Classification metrics (simplified)
                    try:
                        metrics['accuracy'] = accuracy_score(actual_values, pred_values)
                    except:
                        pass
            
            # Calculate prediction statistics
            all_predictions = [p['prediction'] for p in predictions if isinstance(p['prediction'], (int, float))]
            if all_predictions:
                metrics['prediction_mean'] = np.mean(all_predictions)
                metrics['prediction_std'] = np.std(all_predictions)
                metrics['prediction_min'] = np.min(all_predictions)
                metrics['prediction_max'] = np.max(all_predictions)
            
            # Error count
            metrics['error_count'] = len([p for p in predictions if p.get('metadata', {}).get('error', False)])
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate performance metrics: {e}")
            return {}
    
    async def _calculate_latency_metrics(self, monitor_key: str) -> Dict[str, float]:
        """Calculate latency metrics"""
        try:
            latencies = [entry['latency_ms'] for entry in self.latency_buffer[monitor_key]]
            
            if not latencies:
                return {}
            
            return {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': np.min(latencies),
                'max': np.max(latencies)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate latency metrics: {e}")
            return {}
    
    async def _calculate_drift_scores(self, monitor_key: str) -> Dict[str, float]:
        """Calculate drift scores for all features"""
        try:
            drift_results = await self.detect_data_drift(
                monitor_key.split(':')[0], 
                monitor_key.split(':')[1]
            )
            
            return {f"feature_{result.feature_name}": result.drift_score 
                   for result in drift_results}
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate drift scores: {e}")
            return {}
    
    async def _count_recent_alerts(self, monitor_key: str, hours: int = 24) -> int:
        """Count recent alerts for a model"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        model_name, version = monitor_key.split(':')
        
        return len([alert for alert in self.alerts
                   if alert.model_name == model_name and alert.model_version == version
                   and alert.timestamp >= cutoff_time and not alert.resolved])
    
    def _get_recent_features(self, monitor_key: str, hours: int = 24) -> np.ndarray:
        """Get recent feature data for drift detection"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_features = [entry['features'] for entry in self.feature_buffer[monitor_key]
                             if entry['timestamp'] >= cutoff_time]
            
            if recent_features:
                return np.array(recent_features)
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.warning(f"Failed to get recent features: {e}")
            return np.array([])
    
    async def _detect_feature_drift(self,
                                  baseline_feature: np.ndarray,
                                  current_feature: np.ndarray,
                                  feature_idx: int,
                                  method: DriftDetectionMethod) -> DriftDetectionResult:
        """Detect drift for a single feature"""
        try:
            if method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV:
                statistic, p_value = stats.ks_2samp(baseline_feature, current_feature)
                drift_score = statistic
                
            elif method == DriftDetectionMethod.WASSERSTEIN_DISTANCE:
                drift_score = stats.wasserstein_distance(baseline_feature, current_feature)
                p_value = 0.0  # Wasserstein doesn't provide p-value directly
                
            else:
                # Default to KS test
                statistic, p_value = stats.ks_2samp(baseline_feature, current_feature)
                drift_score = statistic
            
            # Determine if drift is detected
            is_drift_detected = drift_score > self.default_thresholds['drift_threshold']
            
            # Calculate feature statistics
            baseline_stats = {
                'mean': float(np.mean(baseline_feature)),
                'std': float(np.std(baseline_feature)),
                'min': float(np.min(baseline_feature)),
                'max': float(np.max(baseline_feature))
            }
            
            current_stats = {
                'mean': float(np.mean(current_feature)),
                'std': float(np.std(current_feature)),
                'min': float(np.min(current_feature)),
                'max': float(np.max(current_feature))
            }
            
            return DriftDetectionResult(
                feature_name=str(feature_idx),
                drift_score=float(drift_score),
                p_value=float(p_value),
                is_drift_detected=is_drift_detected,
                method_used=method,
                baseline_stats=baseline_stats,
                current_stats=current_stats,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to detect feature drift: {e}")
            # Return safe default
            return DriftDetectionResult(
                feature_name=str(feature_idx),
                drift_score=0.0,
                p_value=1.0,
                is_drift_detected=False,
                method_used=method,
                baseline_stats={},
                current_stats={},
                timestamp=datetime.utcnow()
            )
    
    async def _check_performance_alerts(self, monitor_key: str, snapshot: ModelPerformanceSnapshot):
        """Check for performance-based alerts"""
        try:
            model_config = self.active_monitors.get(monitor_key, {})
            thresholds = model_config.get('thresholds', self.default_thresholds)
            
            # Get previous snapshot for comparison
            history = self.performance_history.get(monitor_key, [])
            if len(history) < 2:
                return  # Need at least 2 snapshots for comparison
            
            previous_snapshot = history[-2]
            
            # Check accuracy drop
            for metric in ['accuracy', 'mse', 'mae']:
                if metric in snapshot.metrics and metric in previous_snapshot.metrics:
                    current_value = snapshot.metrics[metric]
                    previous_value = previous_snapshot.metrics[metric]
                    
                    # For accuracy, we want to detect drops
                    if metric == 'accuracy':
                        if current_value < previous_value - thresholds.get('accuracy_drop', 0.05):
                            await self._create_alert(
                                monitor_key, AlertLevel.WARNING, metric,
                                current_value, previous_value,
                                f"Accuracy dropped from {previous_value:.3f} to {current_value:.3f}"
                            )
                    
                    # For error metrics, we want to detect increases
                    elif metric in ['mse', 'mae']:
                        if current_value > previous_value * (1 + thresholds.get('error_increase', 0.2)):
                            await self._create_alert(
                                monitor_key, AlertLevel.WARNING, metric,
                                current_value, previous_value,
                                f"{metric.upper()} increased from {previous_value:.3f} to {current_value:.3f}"
                            )
            
            # Check latency increase
            if snapshot.latency_p95 > previous_snapshot.latency_p95 * thresholds.get('latency_increase', 2.0):
                await self._create_alert(
                    monitor_key, AlertLevel.WARNING, 'latency_p95',
                    snapshot.latency_p95, previous_snapshot.latency_p95,
                    f"P95 latency increased from {previous_snapshot.latency_p95:.1f}ms to {snapshot.latency_p95:.1f}ms"
                )
            
            # Check error rate
            current_error_rate = snapshot.errors_count / max(snapshot.predictions_count, 1)
            previous_error_rate = previous_snapshot.errors_count / max(previous_snapshot.predictions_count, 1)
            
            if current_error_rate > previous_error_rate + thresholds.get('error_rate_increase', 0.1):
                await self._create_alert(
                    monitor_key, AlertLevel.CRITICAL, 'error_rate',
                    current_error_rate, previous_error_rate,
                    f"Error rate increased from {previous_error_rate:.3f} to {current_error_rate:.3f}"
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to check performance alerts: {e}")
    
    async def _create_drift_alert(self, monitor_key: str, drift_result: DriftDetectionResult):
        """Create an alert for detected drift"""
        model_name, version = monitor_key.split(':')
        
        alert_level = AlertLevel.CRITICAL if drift_result.drift_score > 0.2 else AlertLevel.WARNING
        
        await self._create_alert(
            monitor_key, alert_level, 'data_drift',
            drift_result.drift_score, self.default_thresholds['drift_threshold'],
            f"Data drift detected for feature {drift_result.feature_name} (score: {drift_result.drift_score:.3f})",
            additional_data=asdict(drift_result)
        )
    
    async def _create_alert(self,
                          monitor_key: str,
                          level: AlertLevel,
                          metric_name: str,
                          current_value: float,
                          threshold_value: float,
                          message: str,
                          additional_data: Optional[Dict[str, Any]] = None):
        """Create a monitoring alert"""
        try:
            model_name, version = monitor_key.split(':')
            
            alert = MonitoringAlert(
                alert_id=uuid4().hex,
                model_name=model_name,
                model_version=version,
                alert_level=level,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                message=message,
                timestamp=datetime.utcnow(),
                additional_data=additional_data or {}
            )
            
            self.alerts.append(alert)
            
            # Cleanup old alerts
            await self._cleanup_old_alerts()
            
            self.logger.warning(f"Alert created: {alert.alert_level.value.upper()} - {alert.message}")
            
            # Here you would integrate with notification systems
            # await self._send_notification(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.alert_retention_days)
            
            # Remove old alerts
            self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
            
            # Limit total number of alerts
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old alerts: {e}")
    
    def _get_metric_samples(self, monitor_key: str, metric: str, hours: int = 24) -> List[float]:
        """Get metric samples for statistical analysis"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # This is a simplified implementation
            # In a real system, you'd extract metric values from the monitoring data
            predictions = list(self.prediction_buffer[monitor_key])
            recent_predictions = [p for p in predictions if p['timestamp'] >= cutoff_time]
            
            # Extract metric values based on the metric type
            if metric == 'accuracy' and recent_predictions:
                values = []
                for p in recent_predictions:
                    if p.get('actual_value') is not None:
                        # Simple accuracy calculation (1 if correct, 0 if wrong)
                        accuracy = 1.0 if abs(p['prediction'] - p['actual_value']) < 0.1 else 0.0
                        values.append(accuracy)
                return values
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Failed to get metric samples: {e}")
            return []