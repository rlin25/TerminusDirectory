"""
Advanced Report Generator for comprehensive analytics reporting.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import logging
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64


class ReportType(Enum):
    """Types of reports that can be generated."""
    EXECUTIVE_SUMMARY = "executive_summary"
    PROPERTY_PERFORMANCE = "property_performance"
    USER_ENGAGEMENT = "user_engagement"
    ML_MODEL_PERFORMANCE = "ml_model_performance"
    REVENUE_ANALYSIS = "revenue_analysis"
    MARKET_TRENDS = "market_trends"
    OPERATIONAL_METRICS = "operational_metrics"
    CUSTOM_QUERY = "custom_query"


class OutputFormat(Enum):
    """Output formats for reports."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"
    DASHBOARD = "dashboard"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    report_id: str
    report_name: str
    report_type: ReportType
    description: str
    
    # Time range configuration
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    time_range_preset: Optional[str] = None  # "last_7_days", "last_30_days", etc.
    
    # Data filtering and grouping
    filters: Dict[str, Any] = None
    group_by: List[str] = None
    metrics: List[str] = None
    dimensions: List[str] = None
    
    # Output configuration
    output_format: OutputFormat = OutputFormat.JSON
    include_visualizations: bool = True
    include_insights: bool = True
    include_raw_data: bool = False
    
    # Report customization
    template_id: Optional[str] = None
    custom_sql: Optional[str] = None
    visualization_config: Dict[str, Any] = None
    
    # Execution settings
    max_rows: int = 10000
    timeout_seconds: int = 300
    cache_ttl: int = 3600
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.group_by is None:
            self.group_by = []
        if self.metrics is None:
            self.metrics = []
        if self.dimensions is None:
            self.dimensions = []
        if self.visualization_config is None:
            self.visualization_config = {}


@dataclass
class ReportResult:
    """Result of report generation."""
    report_id: str
    report_name: str
    generated_at: datetime
    execution_time_seconds: float
    
    # Data and insights
    data: Dict[str, Any]
    insights: List[Dict[str, Any]]
    visualizations: List[Dict[str, Any]]
    
    # Metadata
    row_count: int
    data_sources: List[str]
    filters_applied: Dict[str, Any]
    
    # Output
    output_format: str
    output_content: Optional[str] = None
    output_file_path: Optional[str] = None
    
    # Quality metrics
    data_quality_score: float = 1.0
    completeness_score: float = 1.0


class ReportGenerator:
    """
    Advanced Report Generator for comprehensive analytics reporting.
    
    Provides automated report generation with support for multiple output formats,
    interactive visualizations, automated insights, and custom templates.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        output_directory: str = "/tmp/reports"
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Report templates and configurations
        self.report_templates: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.generation_stats = {
            "total_reports_generated": 0,
            "avg_generation_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the report generator."""
        try:
            # Load report templates
            await self._load_report_templates()
            
            # Initialize visualization components
            await self._initialize_visualization_engine()
            
            self.logger.info("Report generator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize report generator: {e}")
            raise
    
    async def generate_report(self, config: ReportConfig) -> ReportResult:
        """Generate a report based on the provided configuration."""
        start_time = datetime.utcnow()
        
        try:
            # Validate configuration
            await self._validate_report_config(config)
            
            # Check cache first
            cached_result = await self._get_cached_report(config)
            if cached_result:
                self.logger.info(f"Returning cached report: {config.report_id}")
                return cached_result
            
            # Generate report data
            report_data = await self._generate_report_data(config)
            
            # Generate insights
            insights = []
            if config.include_insights:
                insights = await self._generate_insights(report_data, config)
            
            # Generate visualizations
            visualizations = []
            if config.include_visualizations:
                visualizations = await self._generate_visualizations(report_data, config)
            
            # Create output content
            output_content = None
            output_file_path = None
            if config.output_format != OutputFormat.JSON:
                output_content, output_file_path = await self._create_output(
                    report_data, visualizations, insights, config
                )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create report result
            result = ReportResult(
                report_id=config.report_id,
                report_name=config.report_name,
                generated_at=datetime.utcnow(),
                execution_time_seconds=execution_time,
                data=report_data,
                insights=insights,
                visualizations=visualizations,
                row_count=self._count_rows(report_data),
                data_sources=self._extract_data_sources(config),
                filters_applied=config.filters,
                output_format=config.output_format.value,
                output_content=output_content,
                output_file_path=output_file_path,
                data_quality_score=await self._calculate_data_quality_score(report_data),
                completeness_score=await self._calculate_completeness_score(report_data)
            )
            
            # Cache the result
            await self._cache_report(config, result)
            
            # Update statistics
            await self._update_generation_stats(execution_time)
            
            self.logger.info(
                f"Generated report {config.report_id} in {execution_time:.2f}s "
                f"with {result.row_count} rows"
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Failed to generate report {config.report_id}: {e}")
            
            # Return error result
            return ReportResult(
                report_id=config.report_id,
                report_name=config.report_name,
                generated_at=datetime.utcnow(),
                execution_time_seconds=execution_time,
                data={"error": str(e)},
                insights=[],
                visualizations=[],
                row_count=0,
                data_sources=[],
                filters_applied=config.filters,
                output_format=config.output_format.value
            )
    
    async def generate_executive_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        include_forecasts: bool = True
    ) -> ReportResult:
        """Generate executive summary report."""
        config = ReportConfig(
            report_id=f"executive_summary_{int(datetime.utcnow().timestamp())}",
            report_name="Executive Summary",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            description="High-level business metrics and KPIs for executive leadership",
            start_date=start_date,
            end_date=end_date,
            output_format=OutputFormat.HTML,
            include_visualizations=True,
            include_insights=True
        )
        
        return await self.generate_report(config)
    
    async def generate_property_performance_report(
        self,
        property_ids: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        time_period: str = "last_30_days"
    ) -> ReportResult:
        """Generate property performance analysis report."""
        filters = {}
        if property_ids:
            filters["property_id"] = property_ids
        if regions:
            filters["region"] = regions
        
        config = ReportConfig(
            report_id=f"property_performance_{int(datetime.utcnow().timestamp())}",
            report_name="Property Performance Analysis",
            report_type=ReportType.PROPERTY_PERFORMANCE,
            description="Detailed analysis of property performance metrics",
            time_range_preset=time_period,
            filters=filters,
            metrics=["view_count", "inquiry_rate", "conversion_rate", "revenue"],
            dimensions=["property_type", "location", "price_range"],
            group_by=["property_id", "property_type"],
            output_format=OutputFormat.HTML,
            include_visualizations=True,
            include_insights=True
        )
        
        return await self.generate_report(config)
    
    async def generate_user_engagement_report(
        self,
        user_segments: Optional[List[str]] = None,
        time_period: str = "last_30_days"
    ) -> ReportResult:
        """Generate user engagement analysis report."""
        filters = {}
        if user_segments:
            filters["user_segment"] = user_segments
        
        config = ReportConfig(
            report_id=f"user_engagement_{int(datetime.utcnow().timestamp())}",
            report_name="User Engagement Analysis",
            report_type=ReportType.USER_ENGAGEMENT,
            description="Analysis of user behavior and engagement patterns",
            time_range_preset=time_period,
            filters=filters,
            metrics=["session_duration", "page_views", "interaction_rate", "conversion_rate"],
            dimensions=["user_segment", "traffic_source", "device_type"],
            group_by=["user_segment", "day"],
            output_format=OutputFormat.HTML,
            include_visualizations=True,
            include_insights=True
        )
        
        return await self.generate_report(config)
    
    async def generate_ml_performance_report(
        self,
        model_names: Optional[List[str]] = None,
        time_period: str = "last_7_days"
    ) -> ReportResult:
        """Generate ML model performance report."""
        filters = {}
        if model_names:
            filters["model_name"] = model_names
        
        config = ReportConfig(
            report_id=f"ml_performance_{int(datetime.utcnow().timestamp())}",
            report_name="ML Model Performance",
            report_type=ReportType.ML_MODEL_PERFORMANCE,
            description="Performance metrics and drift analysis for ML models",
            time_range_preset=time_period,
            filters=filters,
            metrics=["accuracy", "precision", "recall", "inference_time", "drift_score"],
            dimensions=["model_name", "model_version"],
            group_by=["model_name", "hour"],
            output_format=OutputFormat.HTML,
            include_visualizations=True,
            include_insights=True
        )
        
        return await self.generate_report(config)
    
    async def generate_custom_report(
        self,
        sql_query: str,
        report_name: str,
        description: str = "",
        output_format: OutputFormat = OutputFormat.JSON
    ) -> ReportResult:
        """Generate a custom report using SQL query."""
        config = ReportConfig(
            report_id=f"custom_{int(datetime.utcnow().timestamp())}",
            report_name=report_name,
            report_type=ReportType.CUSTOM_QUERY,
            description=description or "Custom SQL query report",
            custom_sql=sql_query,
            output_format=output_format,
            include_visualizations=False,
            include_insights=False
        )
        
        return await self.generate_report(config)
    
    async def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available report templates."""
        templates = []
        for template_id, template_config in self.report_templates.items():
            templates.append({
                "template_id": template_id,
                "name": template_config.get("name", template_id),
                "description": template_config.get("description", ""),
                "report_type": template_config.get("report_type", ""),
                "supported_formats": template_config.get("supported_formats", [])
            })
        return templates
    
    async def get_generation_statistics(self) -> Dict[str, Any]:
        """Get report generation statistics."""
        return {
            "statistics": self.generation_stats,
            "cached_reports_count": await self._get_cached_reports_count(),
            "available_templates": len(self.report_templates),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Private methods
    async def _validate_report_config(self, config: ReportConfig) -> None:
        """Validate report configuration."""
        if not config.report_id:
            raise ValueError("Report ID is required")
        
        if not config.report_name:
            raise ValueError("Report name is required")
        
        # Validate time range
        if config.start_date and config.end_date:
            if config.start_date >= config.end_date:
                raise ValueError("Start date must be before end date")
        
        # Validate custom SQL if provided
        if config.custom_sql and config.report_type != ReportType.CUSTOM_QUERY:
            raise ValueError("Custom SQL only allowed for custom query reports")
    
    async def _get_cached_report(self, config: ReportConfig) -> Optional[ReportResult]:
        """Get cached report if available."""
        try:
            cache_key = self._generate_cache_key(config)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                try:
                    result_dict = json.loads(cached_data)
                    # Convert datetime strings back to datetime objects
                    result_dict["generated_at"] = datetime.fromisoformat(result_dict["generated_at"])
                    return ReportResult(**result_dict)
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize cached report: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error accessing cached report: {e}")
            return None
    
    def _generate_cache_key(self, config: ReportConfig) -> str:
        """Generate cache key for report configuration."""
        # Create a hash of the configuration to use as cache key
        config_str = json.dumps(asdict(config), sort_keys=True, default=str)
        import hashlib
        return f"report_cache:{hashlib.md5(config_str.encode()).hexdigest()}"
    
    async def _generate_report_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate the core data for the report."""
        if config.custom_sql:
            return await self._execute_custom_query(config.custom_sql, config)
        elif config.report_type == ReportType.EXECUTIVE_SUMMARY:
            return await self._generate_executive_summary_data(config)
        elif config.report_type == ReportType.PROPERTY_PERFORMANCE:
            return await self._generate_property_performance_data(config)
        elif config.report_type == ReportType.USER_ENGAGEMENT:
            return await self._generate_user_engagement_data(config)
        elif config.report_type == ReportType.ML_MODEL_PERFORMANCE:
            return await self._generate_ml_performance_data(config)
        elif config.report_type == ReportType.REVENUE_ANALYSIS:
            return await self._generate_revenue_analysis_data(config)
        elif config.report_type == ReportType.MARKET_TRENDS:
            return await self._generate_market_trends_data(config)
        else:
            raise ValueError(f"Unsupported report type: {config.report_type}")
    
    async def _execute_custom_query(
        self,
        sql_query: str,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Execute custom SQL query."""
        try:
            # Security: Basic SQL injection protection (in production, use more sophisticated validation)
            dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
            query_upper = sql_query.upper()
            
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
            
            # Add LIMIT clause if not present and max_rows is set
            if config.max_rows and "LIMIT" not in query_upper:
                sql_query += f" LIMIT {config.max_rows}"
            
            # Execute query
            result = await self.db_session.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            
            # Convert to list of dictionaries
            data = [dict(zip(columns, row)) for row in rows]
            
            return {
                "query_results": data,
                "column_names": list(columns),
                "row_count": len(data),
                "sql_query": sql_query
            }
            
        except Exception as e:
            self.logger.error(f"Error executing custom query: {e}")
            raise
    
    async def _generate_executive_summary_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate executive summary report data."""
        # Get time range
        start_date, end_date = self._resolve_time_range(config)
        
        # Execute multiple queries to gather executive metrics
        summary_data = {}
        
        # Business metrics
        business_metrics_query = text("""
            SELECT 
                COUNT(DISTINCT user_id) as total_users,
                COUNT(DISTINCT CASE WHEN last_activity >= :start_date THEN user_id END) as active_users,
                COUNT(DISTINCT property_id) as total_properties,
                AVG(price) as avg_property_price,
                SUM(CASE WHEN event_name = 'property_view' THEN 1 ELSE 0 END) as total_views,
                SUM(CASE WHEN event_name = 'inquiry' THEN 1 ELSE 0 END) as total_inquiries
            FROM analytics_events 
            WHERE timestamp BETWEEN :start_date AND :end_date
        """)
        
        result = await self.db_session.execute(business_metrics_query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        business_metrics = dict(result.fetchone())
        summary_data["business_metrics"] = business_metrics
        
        # Revenue metrics (if available)
        try:
            revenue_query = text("""
                SELECT 
                    SUM(total_revenue) as total_revenue,
                    AVG(total_revenue) as avg_daily_revenue,
                    SUM(new_customers) as new_customers,
                    SUM(churned_customers) as churned_customers
                FROM revenue_analytics
                WHERE timestamp BETWEEN :start_date AND :end_date
            """)
            
            result = await self.db_session.execute(revenue_query, {
                "start_date": start_date,
                "end_date": end_date
            })
            
            revenue_metrics = dict(result.fetchone())
            summary_data["revenue_metrics"] = revenue_metrics
            
        except Exception as e:
            self.logger.warning(f"Could not fetch revenue metrics: {e}")
            summary_data["revenue_metrics"] = {}
        
        # Performance metrics
        performance_query = text("""
            SELECT 
                AVG(CASE WHEN properties->>'response_time_ms' IS NOT NULL 
                    THEN (properties->>'response_time_ms')::numeric END) as avg_response_time,
                COUNT(*) FILTER (WHERE event_type = 'system_event' AND properties->>'error' IS NOT NULL) as error_count,
                COUNT(*) as total_events
            FROM analytics_events
            WHERE timestamp BETWEEN :start_date AND :end_date
        """)
        
        result = await self.db_session.execute(performance_query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        performance_metrics = dict(result.fetchone())
        summary_data["performance_metrics"] = performance_metrics
        
        # Calculate derived metrics
        summary_data["kpis"] = self._calculate_executive_kpis(summary_data)
        
        return summary_data
    
    async def _generate_property_performance_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate property performance report data."""
        start_date, end_date = self._resolve_time_range(config)
        
        # Build WHERE clause from filters
        where_conditions = ["timestamp BETWEEN :start_date AND :end_date"]
        query_params = {"start_date": start_date, "end_date": end_date}
        
        if config.filters.get("property_id"):
            where_conditions.append("property_id = ANY(:property_ids)")
            query_params["property_ids"] = config.filters["property_id"]
        
        where_clause = " AND ".join(where_conditions)
        
        # Property performance query
        query = text(f"""
            SELECT 
                property_id,
                SUM(view_count) as total_views,
                SUM(unique_viewers) as total_unique_viewers,
                SUM(inquiries_count) as total_inquiries,
                AVG(conversion_rate) as avg_conversion_rate,
                AVG(avg_view_duration) as avg_view_duration,
                MAX(timestamp) as last_activity
            FROM property_analytics
            WHERE {where_clause}
            GROUP BY property_id
            ORDER BY total_views DESC
            LIMIT :max_rows
        """)
        
        query_params["max_rows"] = config.max_rows
        
        result = await self.db_session.execute(query, query_params)
        property_data = [dict(row) for row in result.fetchall()]
        
        # Get property details
        if property_data:
            property_ids = [p["property_id"] for p in property_data]
            
            details_query = text("""
                SELECT id, title, location, price, property_type, bedrooms, bathrooms
                FROM properties
                WHERE id = ANY(:property_ids)
            """)
            
            result = await self.db_session.execute(details_query, {"property_ids": property_ids})
            property_details = {str(row.id): dict(row) for row in result.fetchall()}
            
            # Merge performance data with property details
            for prop in property_data:
                prop_id = prop["property_id"]
                if prop_id in property_details:
                    prop.update(property_details[prop_id])
        
        return {
            "property_performance": property_data,
            "summary_stats": self._calculate_property_summary_stats(property_data),
            "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()}
        }
    
    async def _generate_user_engagement_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate user engagement report data."""
        start_date, end_date = self._resolve_time_range(config)
        
        # User engagement metrics
        engagement_query = text("""
            SELECT 
                user_segment,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(session_duration_seconds) as avg_session_duration,
                AVG(page_views) as avg_page_views,
                AVG(engagement_score) as avg_engagement_score,
                AVG(conversion_score) as avg_conversion_score,
                SUM(property_views) as total_property_views
            FROM user_behavior_analytics
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY user_segment
            ORDER BY unique_users DESC
        """)
        
        result = await self.db_session.execute(engagement_query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        engagement_data = [dict(row) for row in result.fetchall()]
        
        # Time series data for trends
        trend_query = text("""
            SELECT 
                DATE_TRUNC('day', timestamp) as date,
                COUNT(DISTINCT user_id) as daily_active_users,
                AVG(session_duration_seconds) as avg_session_duration,
                AVG(engagement_score) as avg_engagement_score
            FROM user_behavior_analytics
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY DATE_TRUNC('day', timestamp)
            ORDER BY date
        """)
        
        result = await self.db_session.execute(trend_query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        trend_data = [dict(row) for row in result.fetchall()]
        
        return {
            "engagement_by_segment": engagement_data,
            "engagement_trends": trend_data,
            "summary_stats": self._calculate_engagement_summary_stats(engagement_data),
            "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()}
        }
    
    async def _generate_ml_performance_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate ML model performance report data."""
        start_date, end_date = self._resolve_time_range(config)
        
        # ML performance metrics
        performance_query = text("""
            SELECT 
                model_name,
                model_version,
                AVG(accuracy) as avg_accuracy,
                AVG(precision_score) as avg_precision,
                AVG(recall_score) as avg_recall,
                AVG(f1_score) as avg_f1_score,
                AVG(avg_inference_time_ms) as avg_inference_time,
                AVG(feature_drift_score) as avg_feature_drift,
                AVG(prediction_drift_score) as avg_prediction_drift,
                COUNT(*) as measurement_count
            FROM ml_model_performance
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY model_name, model_version
            ORDER BY model_name, model_version
        """)
        
        result = await self.db_session.execute(performance_query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        performance_data = [dict(row) for row in result.fetchall()]
        
        # Time series for model performance trends
        trend_query = text("""
            SELECT 
                model_name,
                DATE_TRUNC('hour', timestamp) as hour,
                AVG(accuracy) as avg_accuracy,
                AVG(avg_inference_time_ms) as avg_inference_time
            FROM ml_model_performance
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY model_name, DATE_TRUNC('hour', timestamp)
            ORDER BY model_name, hour
        """)
        
        result = await self.db_session.execute(trend_query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        trend_data = [dict(row) for row in result.fetchall()]
        
        return {
            "model_performance": performance_data,
            "performance_trends": trend_data,
            "summary_stats": self._calculate_ml_summary_stats(performance_data),
            "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()}
        }
    
    async def _generate_revenue_analysis_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate revenue analysis report data."""
        # Placeholder implementation
        return {"revenue_analysis": "Not implemented yet"}
    
    async def _generate_market_trends_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate market trends report data."""
        # Placeholder implementation
        return {"market_trends": "Not implemented yet"}
    
    def _resolve_time_range(self, config: ReportConfig) -> Tuple[datetime, datetime]:
        """Resolve time range from configuration."""
        if config.start_date and config.end_date:
            return config.start_date, config.end_date
        
        end_date = datetime.utcnow()
        
        if config.time_range_preset == "last_7_days":
            start_date = end_date - timedelta(days=7)
        elif config.time_range_preset == "last_30_days":
            start_date = end_date - timedelta(days=30)
        elif config.time_range_preset == "last_90_days":
            start_date = end_date - timedelta(days=90)
        elif config.time_range_preset == "last_year":
            start_date = end_date - timedelta(days=365)
        else:
            # Default to last 30 days
            start_date = end_date - timedelta(days=30)
        
        return start_date, end_date
    
    def _calculate_executive_kpis(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate executive KPIs from summary data."""
        business = summary_data.get("business_metrics", {})
        
        kpis = {}
        
        # User engagement KPIs
        if business.get("total_users") and business.get("active_users"):
            kpis["user_activation_rate"] = (business["active_users"] / business["total_users"]) * 100
        
        # Conversion KPIs
        if business.get("total_views") and business.get("total_inquiries"):
            kpis["inquiry_conversion_rate"] = (business["total_inquiries"] / business["total_views"]) * 100
        
        return kpis
    
    def _calculate_property_summary_stats(self, property_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for property data."""
        if not property_data:
            return {}
        
        df = pd.DataFrame(property_data)
        
        return {
            "total_properties": len(property_data),
            "total_views": df["total_views"].sum(),
            "avg_conversion_rate": df["avg_conversion_rate"].mean(),
            "top_performing_property": property_data[0]["property_id"] if property_data else None
        }
    
    def _calculate_engagement_summary_stats(self, engagement_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for engagement data."""
        if not engagement_data:
            return {}
        
        df = pd.DataFrame(engagement_data)
        
        return {
            "total_users": df["unique_users"].sum(),
            "avg_session_duration": df["avg_session_duration"].mean(),
            "avg_engagement_score": df["avg_engagement_score"].mean(),
            "most_engaged_segment": df.loc[df["avg_engagement_score"].idxmax(), "user_segment"] if len(df) > 0 else None
        }
    
    def _calculate_ml_summary_stats(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for ML performance data."""
        if not performance_data:
            return {}
        
        df = pd.DataFrame(performance_data)
        
        return {
            "total_models": len(performance_data),
            "avg_accuracy": df["avg_accuracy"].mean(),
            "avg_inference_time": df["avg_inference_time"].mean(),
            "best_performing_model": df.loc[df["avg_accuracy"].idxmax(), "model_name"] if len(df) > 0 else None
        }
    
    # Additional helper methods would be implemented here
    async def _generate_insights(
        self,
        report_data: Dict[str, Any],
        config: ReportConfig
    ) -> List[Dict[str, Any]]:
        """Generate automated insights from report data."""
        # Placeholder for insight generation logic
        return []
    
    async def _generate_visualizations(
        self,
        report_data: Dict[str, Any],
        config: ReportConfig
    ) -> List[Dict[str, Any]]:
        """Generate visualizations for report data."""
        # Placeholder for visualization generation logic
        return []
    
    async def _create_output(
        self,
        report_data: Dict[str, Any],
        visualizations: List[Dict[str, Any]],
        insights: List[Dict[str, Any]],
        config: ReportConfig
    ) -> Tuple[Optional[str], Optional[str]]:
        """Create output content in specified format."""
        # Placeholder for output generation logic
        return None, None
    
    def _count_rows(self, report_data: Dict[str, Any]) -> int:
        """Count total rows in report data."""
        total_rows = 0
        for key, value in report_data.items():
            if isinstance(value, list):
                total_rows += len(value)
        return total_rows
    
    def _extract_data_sources(self, config: ReportConfig) -> List[str]:
        """Extract data sources used in report."""
        sources = []
        if config.report_type == ReportType.EXECUTIVE_SUMMARY:
            sources = ["analytics_events", "revenue_analytics", "business_metrics"]
        elif config.report_type == ReportType.PROPERTY_PERFORMANCE:
            sources = ["property_analytics", "properties"]
        elif config.report_type == ReportType.USER_ENGAGEMENT:
            sources = ["user_behavior_analytics"]
        elif config.report_type == ReportType.ML_MODEL_PERFORMANCE:
            sources = ["ml_model_performance"]
        return sources
    
    async def _calculate_data_quality_score(self, report_data: Dict[str, Any]) -> float:
        """Calculate data quality score for report."""
        # Placeholder implementation
        return 1.0
    
    async def _calculate_completeness_score(self, report_data: Dict[str, Any]) -> float:
        """Calculate data completeness score for report."""
        # Placeholder implementation
        return 1.0
    
    async def _cache_report(self, config: ReportConfig, result: ReportResult) -> None:
        """Cache report result."""
        try:
            cache_key = self._generate_cache_key(config)
            
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            result_dict["generated_at"] = result.generated_at.isoformat()
            
            cache_value = json.dumps(result_dict, default=str)
            await self.redis_client.setex(cache_key, config.cache_ttl, cache_value)
            
        except Exception as e:
            self.logger.error(f"Failed to cache report: {e}")
    
    async def _load_report_templates(self) -> None:
        """Load report templates from configuration."""
        # Placeholder for template loading
        pass
    
    async def _initialize_visualization_engine(self) -> None:
        """Initialize visualization components."""
        # Placeholder for visualization engine initialization
        pass
    
    async def _update_generation_stats(self, execution_time: float) -> None:
        """Update report generation statistics."""
        self.generation_stats["total_reports_generated"] += 1
        
        # Update average generation time
        total_reports = self.generation_stats["total_reports_generated"]
        current_avg = self.generation_stats["avg_generation_time"]
        self.generation_stats["avg_generation_time"] = (
            (current_avg * (total_reports - 1) + execution_time) / total_reports
        )
    
    async def _get_cached_reports_count(self) -> int:
        """Get count of cached reports."""
        try:
            keys = await self.redis_client.keys("report_cache:*")
            return len(keys)
        except:
            return 0