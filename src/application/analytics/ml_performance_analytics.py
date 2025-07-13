"""
ML Performance Analytics module for monitoring and analyzing machine learning model performance.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis


class ModelType(Enum):
    """Types of ML models in the system."""
    COLLABORATIVE_FILTER = "collaborative_filter"
    CONTENT_RECOMMENDER = "content_recommender"
    HYBRID_RECOMMENDER = "hybrid_recommender"
    SEARCH_RANKER = "search_ranker"
    PRICE_PREDICTOR = "price_predictor"
    DEMAND_FORECASTER = "demand_forecaster"


class MetricType(Enum):
    """Types of ML metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    RMSE = "rmse"
    MAE = "mae"
    NDCG = "ndcg"
    MAP = "map"
    CTR = "ctr"
    CONVERSION_RATE = "conversion_rate"


class DriftType(Enum):
    """Types of model drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a specific model."""
    model_name: str
    model_type: ModelType
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    ndcg: Optional[float]
    map_score: Optional[float]
    ctr: Optional[float]
    conversion_rate: Optional[float]
    inference_time_ms: float
    throughput_rps: float
    memory_usage_mb: float
    timestamp: datetime


@dataclass
class DriftDetection:
    """Model drift detection results."""
    model_name: str
    drift_type: DriftType
    drift_score: float
    threshold: float
    is_drifting: bool
    affected_features: List[str]
    severity: str  # "low", "medium", "high"
    recommended_action: str
    timestamp: datetime


@dataclass
class ModelComparison:
    """Comparison between different model versions or types."""
    baseline_model: str
    comparison_model: str
    metric_comparisons: Dict[str, Dict[str, float]]  # metric -> {baseline, comparison, improvement}
    statistical_significance: Dict[str, bool]
    recommendation: str
    confidence_level: float


@dataclass
class ABTestResult:
    """A/B test results for model comparison."""
    test_id: str
    model_a: str
    model_b: str
    traffic_split: Dict[str, float]
    sample_size: int
    test_duration_days: int
    metrics: Dict[str, Dict[str, float]]  # metric -> {model_a, model_b}
    winner: Optional[str]
    statistical_significance: bool
    confidence_level: float
    lift: Dict[str, float]  # improvement by metric


class MLPerformanceAnalytics:
    """
    Monitors and analyzes ML model performance, drift, and provides insights
    for model optimization and maintenance.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML performance summary."""
        cache_key = "ml_performance_summary"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        # Gather performance data concurrently
        tasks = [
            self._get_model_performance_overview(),
            self._get_drift_alerts(),
            self._get_model_health_status(),
            self._get_performance_trends(),
            self._get_feature_importance_analysis(),
            self._get_prediction_distribution_analysis()
        ]
        
        (performance_overview, drift_alerts, health_status,
         performance_trends, feature_importance, prediction_distribution) = await asyncio.gather(*tasks)
        
        summary = {
            "performance_overview": performance_overview,
            "drift_alerts": drift_alerts,
            "health_status": health_status,
            "performance_trends": performance_trends,
            "feature_importance": feature_importance,
            "prediction_distribution": prediction_distribution,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Cache the result
        await self.redis_client.setex(cache_key, self.cache_ttl, str(summary))
        
        return summary
    
    async def get_current_model_accuracy(self) -> float:
        """Get current overall model accuracy."""
        cache_key = "current_model_accuracy"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return float(cached_result)
        
        # Get latest accuracy from main recommender model
        query = text("""
            SELECT accuracy
            FROM ml_model_metrics
            WHERE model_name = 'hybrid_recommender'
            AND created_at > NOW() - INTERVAL '1 hour'
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        result = await self.db_session.execute(query)
        row = result.fetchone()
        
        accuracy = float(row.accuracy) if row else 0.85  # Default fallback
        
        # Cache the result
        await self.redis_client.setex(cache_key, 300, str(accuracy))  # 5 minutes cache
        
        return accuracy
    
    async def get_recent_accuracy(self) -> float:
        """Get recent model accuracy for real-time monitoring."""
        cache_key = "recent_model_accuracy"
        
        # Try to get from cache with short TTL
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return float(cached_result)
        
        # Get accuracy from last 5 minutes
        query = text("""
            SELECT AVG(accuracy) as avg_accuracy
            FROM ml_model_metrics
            WHERE created_at > NOW() - INTERVAL '5 minutes'
        """)
        
        result = await self.db_session.execute(query)
        row = result.fetchone()
        
        accuracy = float(row.avg_accuracy) if row and row.avg_accuracy else 0.85
        
        # Cache with short TTL for real-time data
        await self.redis_client.setex(cache_key, 60, str(accuracy))  # 1 minute cache
        
        return accuracy
    
    async def analyze_model_performance(
        self,
        model_name: str,
        time_range: str = "24h"
    ) -> ModelPerformanceMetrics:
        """Analyze performance for a specific model."""
        cache_key = f"model_performance:{model_name}:{time_range}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return ModelPerformanceMetrics(**eval(cached_result))
        
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Get model performance metrics
        query = text("""
            SELECT 
                model_name,
                model_type,
                model_version,
                AVG(accuracy) as accuracy,
                AVG(precision_score) as precision,
                AVG(recall) as recall,
                AVG(f1_score) as f1_score,
                AVG(auc_roc) as auc_roc,
                AVG(rmse) as rmse,
                AVG(mae) as mae,
                AVG(ndcg) as ndcg,
                AVG(map_score) as map_score,
                AVG(ctr) as ctr,
                AVG(conversion_rate) as conversion_rate,
                AVG(inference_time_ms) as inference_time_ms,
                AVG(throughput_rps) as throughput_rps,
                AVG(memory_usage_mb) as memory_usage_mb
            FROM ml_model_metrics
            WHERE model_name = :model_name
            AND created_at BETWEEN :start_time AND :end_time
            GROUP BY model_name, model_type, model_version
            ORDER BY MAX(created_at) DESC
            LIMIT 1
        """)
        
        result = await self.db_session.execute(
            query,
            {"model_name": model_name, "start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        if not row:
            # Return default metrics if no data found
            return ModelPerformanceMetrics(
                model_name=model_name,
                model_type=ModelType.HYBRID_RECOMMENDER,
                version="unknown",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_roc=None,
                rmse=None,
                mae=None,
                ndcg=None,
                map_score=None,
                ctr=None,
                conversion_rate=None,
                inference_time_ms=0.0,
                throughput_rps=0.0,
                memory_usage_mb=0.0,
                timestamp=datetime.utcnow()
            )
        
        metrics = ModelPerformanceMetrics(
            model_name=row.model_name,
            model_type=ModelType(row.model_type),
            version=row.model_version or "unknown",
            accuracy=float(row.accuracy or 0),
            precision=float(row.precision or 0),
            recall=float(row.recall or 0),
            f1_score=float(row.f1_score or 0),
            auc_roc=float(row.auc_roc) if row.auc_roc else None,
            rmse=float(row.rmse) if row.rmse else None,
            mae=float(row.mae) if row.mae else None,
            ndcg=float(row.ndcg) if row.ndcg else None,
            map_score=float(row.map_score) if row.map_score else None,
            ctr=float(row.ctr) if row.ctr else None,
            conversion_rate=float(row.conversion_rate) if row.conversion_rate else None,
            inference_time_ms=float(row.inference_time_ms or 0),
            throughput_rps=float(row.throughput_rps or 0),
            memory_usage_mb=float(row.memory_usage_mb or 0),
            timestamp=datetime.utcnow()
        )
        
        # Cache the result
        await self.redis_client.setex(cache_key, self.cache_ttl, str(metrics.__dict__))
        
        return metrics
    
    async def detect_model_drift(
        self,
        model_name: str,
        lookback_days: int = 7
    ) -> List[DriftDetection]:
        """Detect various types of drift for a model."""
        cache_key = f"model_drift:{model_name}:{lookback_days}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return [DriftDetection(**d) for d in eval(cached_result)]
        
        drift_detections = []
        
        # Data drift detection
        data_drift = await self._detect_data_drift(model_name, lookback_days)
        if data_drift:
            drift_detections.append(data_drift)
        
        # Concept drift detection
        concept_drift = await self._detect_concept_drift(model_name, lookback_days)
        if concept_drift:
            drift_detections.append(concept_drift)
        
        # Prediction drift detection
        prediction_drift = await self._detect_prediction_drift(model_name, lookback_days)
        if prediction_drift:
            drift_detections.append(prediction_drift)
        
        # Feature drift detection
        feature_drift = await self._detect_feature_drift(model_name, lookback_days)
        if feature_drift:
            drift_detections.extend(feature_drift)
        
        # Cache the result
        cache_data = [d.__dict__ for d in drift_detections]
        await self.redis_client.setex(cache_key, 1800, str(cache_data))  # 30 minutes cache
        
        return drift_detections
    
    async def compare_models(
        self,
        baseline_model: str,
        comparison_model: str,
        metrics: Optional[List[MetricType]] = None
    ) -> ModelComparison:
        """Compare performance between two models."""
        if not metrics:
            metrics = [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, 
                      MetricType.F1_SCORE, MetricType.CTR, MetricType.CONVERSION_RATE]
        
        cache_key = f"model_comparison:{baseline_model}:{comparison_model}:{hash(str(metrics))}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return ModelComparison(**eval(cached_result))
        
        # Get performance metrics for both models
        baseline_metrics = await self.analyze_model_performance(baseline_model)
        comparison_metrics = await self.analyze_model_performance(comparison_model)
        
        metric_comparisons = {}
        statistical_significance = {}
        
        for metric in metrics:
            baseline_value = getattr(baseline_metrics, metric.value, 0) or 0
            comparison_value = getattr(comparison_metrics, metric.value, 0) or 0
            
            improvement = ((comparison_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
            
            metric_comparisons[metric.value] = {
                "baseline": baseline_value,
                "comparison": comparison_value,
                "improvement": improvement
            }
            
            # Simple statistical significance check (would use proper statistical tests in production)
            statistical_significance[metric.value] = abs(improvement) > 5.0  # 5% threshold
        
        # Generate recommendation
        significant_improvements = sum(1 for metric, is_sig in statistical_significance.items() 
                                     if is_sig and metric_comparisons[metric]["improvement"] > 0)
        
        if significant_improvements >= len(metrics) / 2:
            recommendation = f"Switch to {comparison_model}"
            confidence = 0.8
        elif significant_improvements > 0:
            recommendation = f"Consider A/B testing {comparison_model}"
            confidence = 0.6
        else:
            recommendation = f"Continue with {baseline_model}"
            confidence = 0.7
        
        comparison = ModelComparison(
            baseline_model=baseline_model,
            comparison_model=comparison_model,
            metric_comparisons=metric_comparisons,
            statistical_significance=statistical_significance,
            recommendation=recommendation,
            confidence_level=confidence
        )
        
        # Cache the result
        await self.redis_client.setex(cache_key, 1800, str(comparison.__dict__))  # 30 minutes cache
        
        return comparison
    
    async def analyze_ab_test_results(
        self,
        test_id: str
    ) -> ABTestResult:
        """Analyze A/B test results for model comparison."""
        cache_key = f"ab_test_results:{test_id}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return ABTestResult(**eval(cached_result))
        
        # Get A/B test data
        query = text("""
            SELECT 
                test_id,
                model_a,
                model_b,
                traffic_split_a,
                traffic_split_b,
                sample_size,
                test_start_date,
                test_end_date,
                metrics_a,
                metrics_b,
                winner,
                statistical_significance,
                confidence_level
            FROM ab_test_results
            WHERE test_id = :test_id
        """)
        
        result = await self.db_session.execute(query, {"test_id": test_id})
        row = result.fetchone()
        
        if not row:
            raise ValueError(f"A/B test {test_id} not found")
        
        # Parse metrics (assuming JSON format in database)
        metrics_a = eval(row.metrics_a) if row.metrics_a else {}
        metrics_b = eval(row.metrics_b) if row.metrics_b else {}
        
        # Calculate test duration
        test_duration = (row.test_end_date - row.test_start_date).days
        
        # Combine metrics
        metrics = {}
        lift = {}
        
        for metric_name in set(list(metrics_a.keys()) + list(metrics_b.keys())):
            value_a = metrics_a.get(metric_name, 0)
            value_b = metrics_b.get(metric_name, 0)
            
            metrics[metric_name] = {
                "model_a": value_a,
                "model_b": value_b
            }
            
            # Calculate lift (B vs A)
            lift[metric_name] = ((value_b - value_a) / value_a * 100) if value_a > 0 else 0
        
        ab_test_result = ABTestResult(
            test_id=test_id,
            model_a=row.model_a,
            model_b=row.model_b,
            traffic_split={
                "model_a": float(row.traffic_split_a or 0.5),
                "model_b": float(row.traffic_split_b or 0.5)
            },
            sample_size=row.sample_size or 0,
            test_duration_days=test_duration,
            metrics=metrics,
            winner=row.winner,
            statistical_significance=row.statistical_significance or False,
            confidence_level=float(row.confidence_level or 0.95),
            lift=lift
        )
        
        # Cache the result
        await self.redis_client.setex(cache_key, 3600, str(ab_test_result.__dict__))  # 1 hour cache
        
        return ab_test_result
    
    async def get_model_feature_importance(
        self,
        model_name: str
    ) -> Dict[str, float]:
        """Get feature importance for a model."""
        cache_key = f"feature_importance:{model_name}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        # Get feature importance data
        query = text("""
            SELECT feature_importance_json
            FROM ml_models
            WHERE model_name = :model_name
            AND is_active = true
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        result = await self.db_session.execute(query, {"model_name": model_name})
        row = result.fetchone()
        
        if row and row.feature_importance_json:
            feature_importance = eval(row.feature_importance_json)
        else:
            # Default feature importance if not available
            feature_importance = {
                "price": 0.25,
                "location": 0.20,
                "property_type": 0.15,
                "size": 0.12,
                "amenities": 0.10,
                "rating": 0.08,
                "availability": 0.05,
                "photos": 0.05
            }
        
        # Cache the result
        await self.redis_client.setex(cache_key, 3600, str(feature_importance))  # 1 hour cache
        
        return feature_importance
    
    async def analyze_prediction_quality(
        self,
        model_name: str,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Analyze the quality of model predictions."""
        cache_key = f"prediction_quality:{model_name}:{time_range}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Get prediction quality metrics
        query = text("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(confidence_score) as avg_confidence,
                STDDEV(confidence_score) as confidence_std,
                AVG(CASE WHEN actual_outcome = predicted_outcome THEN 1.0 ELSE 0.0 END) as accuracy,
                COUNT(CASE WHEN confidence_score > 0.8 THEN 1 END) as high_confidence_predictions,
                COUNT(CASE WHEN confidence_score < 0.5 THEN 1 END) as low_confidence_predictions
            FROM model_predictions
            WHERE model_name = :model_name
            AND created_at BETWEEN :start_time AND :end_time
        """)
        
        result = await self.db_session.execute(
            query,
            {"model_name": model_name, "start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        if not row or row.total_predictions == 0:
            return {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "confidence_std": 0.0,
                "accuracy": 0.0,
                "high_confidence_rate": 0.0,
                "low_confidence_rate": 0.0,
                "prediction_distribution": {}
            }
        
        # Get prediction distribution
        distribution_query = text("""
            SELECT 
                CASE 
                    WHEN confidence_score >= 0.8 THEN 'high'
                    WHEN confidence_score >= 0.6 THEN 'medium'
                    ELSE 'low'
                END as confidence_bucket,
                COUNT(*) as count
            FROM model_predictions
            WHERE model_name = :model_name
            AND created_at BETWEEN :start_time AND :end_time
            GROUP BY confidence_bucket
        """)
        
        dist_result = await self.db_session.execute(
            distribution_query,
            {"model_name": model_name, "start_time": start_time, "end_time": end_time}
        )
        
        distribution = {row.confidence_bucket: row.count for row in dist_result.fetchall()}
        
        analysis = {
            "total_predictions": row.total_predictions,
            "avg_confidence": float(row.avg_confidence or 0),
            "confidence_std": float(row.confidence_std or 0),
            "accuracy": float(row.accuracy or 0),
            "high_confidence_rate": (row.high_confidence_predictions / row.total_predictions) * 100,
            "low_confidence_rate": (row.low_confidence_predictions / row.total_predictions) * 100,
            "prediction_distribution": distribution
        }
        
        # Cache the result
        await self.redis_client.setex(cache_key, 1800, str(analysis))  # 30 minutes cache
        
        return analysis
    
    # Private helper methods
    def _get_start_time(self, time_range: str, end_time: datetime) -> datetime:
        """Calculate start time based on time range."""
        if time_range == "1h":
            return end_time - timedelta(hours=1)
        elif time_range == "24h":
            return end_time - timedelta(days=1)
        elif time_range == "7d":
            return end_time - timedelta(days=7)
        elif time_range == "30d":
            return end_time - timedelta(days=30)
        else:
            return end_time - timedelta(days=1)  # Default to 24 hours
    
    async def _get_model_performance_overview(self) -> Dict[str, Any]:
        """Get high-level model performance overview."""
        query = text("""
            SELECT 
                model_name,
                model_type,
                AVG(accuracy) as avg_accuracy,
                AVG(inference_time_ms) as avg_inference_time
            FROM ml_model_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY model_name, model_type
            ORDER BY avg_accuracy DESC
        """)
        
        result = await self.db_session.execute(query)
        models = result.fetchall()
        
        return {
            "total_models": len(models),
            "avg_accuracy": np.mean([m.avg_accuracy for m in models]) if models else 0,
            "avg_inference_time": np.mean([m.avg_inference_time for m in models]) if models else 0,
            "model_performance": [
                {
                    "model_name": m.model_name,
                    "model_type": m.model_type,
                    "accuracy": float(m.avg_accuracy or 0),
                    "inference_time": float(m.avg_inference_time or 0)
                }
                for m in models
            ]
        }
    
    async def _get_drift_alerts(self) -> List[Dict[str, Any]]:
        """Get current drift alerts."""
        # This would check for active drift alerts
        return []
    
    async def _get_model_health_status(self) -> Dict[str, str]:
        """Get health status for all models."""
        # This would implement model health checks
        return {
            "hybrid_recommender": "healthy",
            "search_ranker": "healthy",
            "collaborative_filter": "warning",
            "content_recommender": "healthy"
        }
    
    async def _get_performance_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends over time."""
        # This would implement trend analysis
        return {}
    
    async def _get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Get feature importance analysis across models."""
        # This would analyze feature importance
        return {}
    
    async def _get_prediction_distribution_analysis(self) -> Dict[str, Any]:
        """Get prediction distribution analysis."""
        # This would analyze prediction distributions
        return {}
    
    async def _detect_data_drift(self, model_name: str, lookback_days: int) -> Optional[DriftDetection]:
        """Detect data drift for a model."""
        # This would implement data drift detection using statistical tests
        # For now, return None (no drift detected)
        return None
    
    async def _detect_concept_drift(self, model_name: str, lookback_days: int) -> Optional[DriftDetection]:
        """Detect concept drift for a model."""
        # This would implement concept drift detection
        # For now, return None (no drift detected)
        return None
    
    async def _detect_prediction_drift(self, model_name: str, lookback_days: int) -> Optional[DriftDetection]:
        """Detect prediction drift for a model."""
        # This would implement prediction drift detection
        # For now, return None (no drift detected)
        return None
    
    async def _detect_feature_drift(self, model_name: str, lookback_days: int) -> List[DriftDetection]:
        """Detect feature drift for a model."""
        # This would implement feature-level drift detection
        # For now, return empty list (no drift detected)
        return []