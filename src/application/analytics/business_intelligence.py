"""
Business Intelligence Dashboard for comprehensive analytics and insights.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis

from .kpi_tracker import KPITracker
from .user_behavior_analytics import UserBehaviorAnalytics
from .market_analysis import MarketAnalysis
from .revenue_analytics import RevenueAnalytics
from .ml_performance_analytics import MLPerformanceAnalytics
from ..reporting.report_generator import ReportGenerator
from ..predictive.price_forecaster import PriceForecaster
from ...infrastructure.streaming.stream_analytics_engine import StreamAnalyticsEngine
from ...infrastructure.data_warehouse.analytics_warehouse import AnalyticsWarehouse


class TimeRange(Enum):
    """Time range options for analytics queries."""
    LAST_HOUR = "1h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    LAST_YEAR = "1y"
    CUSTOM = "custom"


@dataclass
class DashboardMetrics:
    """Container for dashboard metrics."""
    total_users: int
    active_users: int
    total_properties: int
    recommendations_served: int
    conversion_rate: float
    revenue: float
    avg_response_time: float
    ml_model_accuracy: float
    timestamp: datetime


@dataclass
class AnalyticsFilter:
    """Filter options for analytics queries."""
    time_range: TimeRange
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_segments: Optional[List[str]] = None
    property_types: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    channels: Optional[List[str]] = None


class BusinessIntelligenceDashboard:
    """
    Main Business Intelligence Dashboard that aggregates insights from all analytics modules.
    
    Provides real-time business metrics, KPI tracking, and comprehensive analytics
    for data-driven decision making.
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
        
        # Initialize analytics modules
        self.kpi_tracker = KPITracker(db_session, redis_client)
        self.user_analytics = UserBehaviorAnalytics(db_session, redis_client)
        self.market_analysis = MarketAnalysis(db_session, redis_client)
        self.revenue_analytics = RevenueAnalytics(db_session, redis_client)
        self.ml_analytics = MLPerformanceAnalytics(db_session, redis_client)
        self.report_generator = ReportGenerator(db_session, redis_client)
        self.price_forecaster = PriceForecaster(db_session, redis_client)
        self.stream_analytics = StreamAnalyticsEngine(db_session, redis_client)
        self.data_warehouse = AnalyticsWarehouse(None, db_session, redis_client)  # Config would be passed
    
    async def get_dashboard_overview(
        self,
        time_range: TimeRange = TimeRange.LAST_24_HOURS,
        filters: Optional[AnalyticsFilter] = None
    ) -> DashboardMetrics:
        """Get comprehensive dashboard overview metrics."""
        cache_key = f"dashboard_overview:{time_range.value}:{hash(str(filters))}"
        
        # Try to get from cache first
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return DashboardMetrics(**eval(cached_result))
        
        # Calculate time bounds
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Gather metrics from all modules concurrently
        tasks = [
            self._get_user_metrics(start_time, end_time, filters),
            self._get_property_metrics(start_time, end_time, filters),
            self._get_recommendation_metrics(start_time, end_time, filters),
            self._get_performance_metrics(start_time, end_time, filters),
        ]
        
        user_metrics, property_metrics, rec_metrics, perf_metrics = await asyncio.gather(*tasks)
        
        dashboard_metrics = DashboardMetrics(
            total_users=user_metrics["total_users"],
            active_users=user_metrics["active_users"],
            total_properties=property_metrics["total_properties"],
            recommendations_served=rec_metrics["recommendations_served"],
            conversion_rate=rec_metrics["conversion_rate"],
            revenue=await self.revenue_analytics.get_total_revenue(start_time, end_time),
            avg_response_time=perf_metrics["avg_response_time"],
            ml_model_accuracy=await self.ml_analytics.get_current_model_accuracy(),
            timestamp=datetime.utcnow()
        )
        
        # Cache the result
        await self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            str(dashboard_metrics.__dict__)
        )
        
        return dashboard_metrics
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for live dashboard updates."""
        cache_key = "real_time_metrics"
        
        # Try to get from cache with shorter TTL (30 seconds)
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        # Current timestamp for real-time data
        now = datetime.utcnow()
        last_minute = now - timedelta(minutes=1)
        last_hour = now - timedelta(hours=1)
        
        # Get real-time metrics
        metrics = {
            "current_active_users": await self._get_current_active_users(),
            "recommendations_last_minute": await self._get_recommendations_count(last_minute, now),
            "recommendations_last_hour": await self._get_recommendations_count(last_hour, now),
            "avg_response_time_last_minute": await self._get_avg_response_time(last_minute, now),
            "error_rate_last_minute": await self._get_error_rate(last_minute, now),
            "cache_hit_rate": await self._get_cache_hit_rate(),
            "ml_prediction_accuracy": await self.ml_analytics.get_recent_accuracy(),
            "system_load": await self._get_system_load(),
            "timestamp": now.isoformat()
        }
        
        # Cache with shorter TTL for real-time data
        await self.redis_client.setex(cache_key, 30, str(metrics))
        
        return metrics
    
    async def get_kpi_trends(
        self,
        kpi_names: List[str],
        time_range: TimeRange = TimeRange.LAST_7_DAYS,
        granularity: str = "hour"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get KPI trends over time with specified granularity."""
        return await self.kpi_tracker.get_kpi_trends(kpi_names, time_range, granularity)
    
    async def get_user_behavior_insights(
        self,
        time_range: TimeRange = TimeRange.LAST_30_DAYS
    ) -> Dict[str, Any]:
        """Get comprehensive user behavior insights."""
        return await self.user_analytics.get_behavior_insights(time_range)
    
    async def get_market_insights(
        self,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
        regions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get market analysis and trend insights."""
        return await self.market_analysis.get_market_insights(time_range, regions)
    
    async def get_revenue_analytics(
        self,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
        breakdown: str = "daily"
    ) -> Dict[str, Any]:
        """Get revenue analytics and forecasting."""
        return await self.revenue_analytics.get_revenue_analytics(time_range, breakdown)
    
    async def get_ml_performance_summary(self) -> Dict[str, Any]:
        """Get ML model performance summary and drift analysis."""
        return await self.ml_analytics.get_performance_summary()
    
    async def get_predictive_insights(self, property_type: str = None, region: str = None) -> Dict[str, Any]:
        """Get predictive analytics insights including price forecasts."""
        try:
            insights = {}
            
            # Price forecasting insights
            if property_type and region:
                price_forecast = await self.price_forecaster.forecast_prices(
                    property_type, region
                )
                insights["price_forecast"] = {
                    "predicted_prices": price_forecast.predicted_prices[:10],  # Next 10 periods
                    "trend_direction": price_forecast.trend_direction,
                    "volatility_score": price_forecast.volatility_score,
                    "market_momentum": price_forecast.market_momentum
                }
            
            # Market comparison
            if property_type:
                market_comparison = await self.price_forecaster.compare_markets(
                    property_type, ["downtown", "suburbs", "waterfront"]  # Example regions
                )
                insights["market_comparison"] = market_comparison.get("ranked_markets", [])
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get predictive insights: {e}")
            return {"error": str(e)}
    
    async def get_advanced_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive advanced analytics summary."""
        try:
            summary = {}
            
            # Real-time metrics
            summary["real_time_metrics"] = await self.get_real_time_metrics()
            
            # Streaming analytics status
            summary["streaming_status"] = await self.stream_analytics.get_real_time_metrics()
            
            # Data warehouse statistics
            summary["warehouse_stats"] = await self.data_warehouse.get_warehouse_statistics()
            
            # Predictive model performance
            summary["forecaster_performance"] = await self.price_forecaster.get_forecaster_performance()
            
            # Report generation statistics
            summary["reporting_stats"] = await self.report_generator.get_generation_statistics()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get advanced analytics summary: {e}")
            return {"error": str(e)}
    
    async def generate_executive_report(self, time_range: TimeRange = TimeRange.LAST_30_DAYS) -> Dict[str, Any]:
        """Generate comprehensive executive report."""
        try:
            # Calculate time bounds
            end_time = datetime.utcnow()
            start_time = self._get_start_time(time_range, end_time)
            
            # Generate executive summary report
            executive_report = await self.report_generator.generate_executive_summary(
                start_time, end_time, include_forecasts=True
            )
            
            return {
                "report_id": executive_report.report_id,
                "generated_at": executive_report.generated_at.isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "key_metrics": executive_report.data,
                "insights": executive_report.insights,
                "data_quality_score": executive_report.data_quality_score,
                "execution_time": executive_report.execution_time_seconds
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive report: {e}")
            return {"error": str(e)}
    
    async def get_custom_report_data(
        self,
        report_config: Dict[str, Any],
        filters: Optional[AnalyticsFilter] = None
    ) -> Dict[str, Any]:
        """Generate custom report data based on configuration."""
        report_type = report_config.get("type")
        metrics = report_config.get("metrics", [])
        dimensions = report_config.get("dimensions", [])
        
        if report_type == "executive_summary":
            return await self._generate_executive_summary(filters)
        elif report_type == "property_performance":
            return await self._generate_property_performance_report(metrics, dimensions, filters)
        elif report_type == "user_engagement":
            return await self._generate_user_engagement_report(metrics, dimensions, filters)
        elif report_type == "ml_model_performance":
            return await self._generate_ml_performance_report(metrics, dimensions, filters)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
    
    async def generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on anomaly detection and thresholds."""
        alerts = []
        
        # Check for performance anomalies
        current_metrics = await self.get_real_time_metrics()
        
        # High response time alert
        if current_metrics["avg_response_time_last_minute"] > 2000:  # 2 seconds
            alerts.append({
                "type": "performance",
                "severity": "high",
                "message": f"High response time: {current_metrics['avg_response_time_last_minute']}ms",
                "timestamp": datetime.utcnow(),
                "metric": "response_time",
                "value": current_metrics["avg_response_time_last_minute"]
            })
        
        # High error rate alert
        if current_metrics["error_rate_last_minute"] > 0.05:  # 5%
            alerts.append({
                "type": "reliability",
                "severity": "high",
                "message": f"High error rate: {current_metrics['error_rate_last_minute']:.2%}",
                "timestamp": datetime.utcnow(),
                "metric": "error_rate",
                "value": current_metrics["error_rate_last_minute"]
            })
        
        # Low ML accuracy alert
        if current_metrics["ml_prediction_accuracy"] < 0.85:  # 85%
            alerts.append({
                "type": "ml_performance",
                "severity": "medium",
                "message": f"ML accuracy below threshold: {current_metrics['ml_prediction_accuracy']:.2%}",
                "timestamp": datetime.utcnow(),
                "metric": "ml_accuracy",
                "value": current_metrics["ml_prediction_accuracy"]
            })
        
        # Add KPI-based alerts
        kpi_alerts = await self.kpi_tracker.check_kpi_thresholds()
        alerts.extend(kpi_alerts)
        
        return alerts
    
    async def get_data_lineage_insights(self) -> Dict[str, Any]:
        """Get data lineage and governance insights."""
        try:
            query = text("""
                SELECT 
                    source_table,
                    target_table,
                    transformation_type,
                    business_criticality,
                    data_quality_score,
                    COUNT(*) as transformation_count
                FROM data_lineage
                GROUP BY source_table, target_table, transformation_type, business_criticality, data_quality_score
                ORDER BY transformation_count DESC
                LIMIT 20
            """)
            
            result = await self.db_session.execute(query)
            lineage_data = [dict(row) for row in result.fetchall()]
            
            # Calculate summary statistics
            total_transformations = sum(row["transformation_count"] for row in lineage_data)
            avg_quality_score = np.mean([row["data_quality_score"] for row in lineage_data if row["data_quality_score"]])
            
            critical_data_sources = [
                row for row in lineage_data 
                if row["business_criticality"] == "critical"
            ]
            
            return {
                "data_lineage": lineage_data,
                "summary": {
                    "total_transformations": total_transformations,
                    "average_quality_score": avg_quality_score,
                    "critical_sources_count": len(critical_data_sources)
                },
                "critical_data_sources": critical_data_sources[:5]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get data lineage insights: {e}")
            return {"error": str(e)}
    
    async def get_anomaly_detection_results(self) -> Dict[str, Any]:
        """Get recent anomaly detection results."""
        try:
            query = text("""
                SELECT 
                    metric_name,
                    timestamp,
                    value,
                    anomaly_score,
                    is_anomaly
                FROM kpi_values
                WHERE is_anomaly = true
                AND timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY anomaly_score DESC, timestamp DESC
                LIMIT 50
            """)
            
            result = await self.db_session.execute(query)
            anomalies = [dict(row) for row in result.fetchall()]
            
            # Group by metric
            anomalies_by_metric = {}
            for anomaly in anomalies:
                metric = anomaly["metric_name"]
                if metric not in anomalies_by_metric:
                    anomalies_by_metric[metric] = []
                anomalies_by_metric[metric].append(anomaly)
            
            return {
                "recent_anomalies": anomalies,
                "anomalies_by_metric": anomalies_by_metric,
                "summary": {
                    "total_anomalies": len(anomalies),
                    "affected_metrics": len(anomalies_by_metric),
                    "avg_anomaly_score": np.mean([a["anomaly_score"] for a in anomalies if a["anomaly_score"]])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get anomaly detection results: {e}")
            return {"error": str(e)}
    
    async def initialize_advanced_components(self) -> None:
        """Initialize advanced analytics components."""
        try:
            # Initialize all advanced components
            await self.report_generator.initialize()
            await self.price_forecaster.initialize()
            await self.stream_analytics.initialize()
            await self.data_warehouse.initialize()
            
            self.logger.info("Advanced analytics components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced components: {e}")
            raise
    
    # Private helper methods
    def _get_start_time(self, time_range: TimeRange, end_time: datetime) -> datetime:
        """Calculate start time based on time range."""
        if time_range == TimeRange.LAST_HOUR:
            return end_time - timedelta(hours=1)
        elif time_range == TimeRange.LAST_24_HOURS:
            return end_time - timedelta(days=1)
        elif time_range == TimeRange.LAST_7_DAYS:
            return end_time - timedelta(days=7)
        elif time_range == TimeRange.LAST_30_DAYS:
            return end_time - timedelta(days=30)
        elif time_range == TimeRange.LAST_90_DAYS:
            return end_time - timedelta(days=90)
        elif time_range == TimeRange.LAST_YEAR:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)  # Default to 24 hours
    
    async def _get_user_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, int]:
        """Get user-related metrics."""
        query = text("""
            SELECT 
                COUNT(DISTINCT id) as total_users,
                COUNT(DISTINCT CASE WHEN last_activity > :start_time THEN id END) as active_users
            FROM users
            WHERE created_at <= :end_time
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return {
            "total_users": row.total_users or 0,
            "active_users": row.active_users or 0
        }
    
    async def _get_property_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, int]:
        """Get property-related metrics."""
        query = text("""
            SELECT COUNT(*) as total_properties
            FROM properties
            WHERE created_at <= :end_time
            AND is_active = true
        """)
        
        result = await self.db_session.execute(
            query,
            {"end_time": end_time}
        )
        row = result.fetchone()
        
        return {"total_properties": row.total_properties or 0}
    
    async def _get_recommendation_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, Union[int, float]]:
        """Get recommendation-related metrics."""
        query = text("""
            SELECT 
                COUNT(*) as recommendations_served,
                AVG(CASE WHEN clicked = true THEN 1.0 ELSE 0.0 END) as conversion_rate
            FROM recommendation_events
            WHERE created_at BETWEEN :start_time AND :end_time
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return {
            "recommendations_served": row.recommendations_served or 0,
            "conversion_rate": row.conversion_rate or 0.0
        }
    
    async def _get_performance_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, float]:
        """Get system performance metrics."""
        query = text("""
            SELECT AVG(response_time_ms) as avg_response_time
            FROM api_metrics
            WHERE created_at BETWEEN :start_time AND :end_time
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return {"avg_response_time": row.avg_response_time or 0.0}
    
    async def _get_current_active_users(self) -> int:
        """Get count of currently active users (last 5 minutes)."""
        five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
        
        query = text("""
            SELECT COUNT(DISTINCT user_id) as active_users
            FROM user_sessions
            WHERE last_activity > :five_minutes_ago
        """)
        
        result = await self.db_session.execute(
            query,
            {"five_minutes_ago": five_minutes_ago}
        )
        row = result.fetchone()
        
        return row.active_users or 0
    
    async def _get_recommendations_count(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Get count of recommendations in time period."""
        query = text("""
            SELECT COUNT(*) as count
            FROM recommendation_events
            WHERE created_at BETWEEN :start_time AND :end_time
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return row.count or 0
    
    async def _get_avg_response_time(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """Get average response time in time period."""
        query = text("""
            SELECT AVG(response_time_ms) as avg_time
            FROM api_metrics
            WHERE created_at BETWEEN :start_time AND :end_time
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return row.avg_time or 0.0
    
    async def _get_error_rate(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """Get error rate in time period."""
        query = text("""
            SELECT 
                COUNT(CASE WHEN status_code >= 400 THEN 1 END)::float / 
                COUNT(*)::float as error_rate
            FROM api_metrics
            WHERE created_at BETWEEN :start_time AND :end_time
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return row.error_rate or 0.0
    
    async def _get_cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        # This would typically come from Redis INFO stats
        info = await self.redis_client.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    async def _get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        # This would typically come from system monitoring
        # For now, return placeholder values
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0
        }
    
    async def _generate_executive_summary(
        self,
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, Any]:
        """Generate executive summary report."""
        # Implementation for executive summary
        pass
    
    async def _generate_property_performance_report(
        self,
        metrics: List[str],
        dimensions: List[str],
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, Any]:
        """Generate property performance report."""
        # Implementation for property performance report
        pass
    
    async def _generate_user_engagement_report(
        self,
        metrics: List[str],
        dimensions: List[str],
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, Any]:
        """Generate user engagement report."""
        # Implementation for user engagement report
        pass
    
    async def _generate_ml_performance_report(
        self,
        metrics: List[str],
        dimensions: List[str],
        filters: Optional[AnalyticsFilter]
    ) -> Dict[str, Any]:
        """Generate ML model performance report."""
        # Implementation for ML performance report
        pass