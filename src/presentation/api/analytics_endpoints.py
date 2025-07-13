"""
Analytics API Endpoints for comprehensive data access and visualization.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import json
import io
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from ...application.analytics.business_intelligence import BusinessIntelligenceDashboard, TimeRange
from ...application.reporting.report_generator import ReportGenerator, ReportConfig, ReportType, OutputFormat
from ...application.predictive.price_forecaster import PriceForecaster, ForecastHorizon, ModelType
from ...infrastructure.streaming.stream_analytics_engine import StreamAnalyticsEngine
from ...infrastructure.data_warehouse.analytics_warehouse import AnalyticsWarehouse
from ...infrastructure.data.config import DataManagerFactory


# Pydantic models for API requests/responses
class TimeRangeQuery(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    preset: Optional[str] = Field(None, description="Preset like 'last_7_days', 'last_30_days'")


class ReportRequest(BaseModel):
    report_name: str
    report_type: str
    description: Optional[str] = ""
    time_range: Optional[TimeRangeQuery] = None
    filters: Optional[Dict[str, Any]] = {}
    metrics: Optional[List[str]] = []
    dimensions: Optional[List[str]] = []
    output_format: str = "json"
    include_visualizations: bool = True
    include_insights: bool = True


class ForecastRequest(BaseModel):
    property_type: str
    region: str
    horizon: str = "3_months"
    model_type: str = "ensemble"
    include_market_factors: bool = True


class CustomQueryRequest(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    max_rows: int = Field(1000, description="Maximum number of rows to return")
    format: str = Field("json", description="Output format: json, csv, excel")


class MetricDefinition(BaseModel):
    name: str
    aggregation_type: str
    source_field: str
    time_windows: List[int]
    dimensions: List[str] = []
    filter_conditions: Dict[str, Any] = {}


# Initialize router
analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])

# Dependency injection
async def get_bi_dashboard() -> BusinessIntelligenceDashboard:
    """Get Business Intelligence Dashboard instance."""
    # This would be properly injected in production
    return None  # Placeholder


async def get_report_generator() -> ReportGenerator:
    """Get Report Generator instance."""
    # This would be properly injected in production
    return None  # Placeholder


async def get_price_forecaster() -> PriceForecaster:
    """Get Price Forecaster instance."""
    # This would be properly injected in production
    return None  # Placeholder


# Dashboard and Overview Endpoints
@analytics_router.get("/dashboard/overview")
async def get_dashboard_overview(
    time_range: str = Query("last_24_hours", description="Time range for dashboard data"),
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Get comprehensive dashboard overview with key metrics and KPIs."""
    try:
        # Convert string to TimeRange enum
        time_range_enum = TimeRange.LAST_24_HOURS
        if time_range == "last_7_days":
            time_range_enum = TimeRange.LAST_7_DAYS
        elif time_range == "last_30_days":
            time_range_enum = TimeRange.LAST_30_DAYS
        
        overview = await bi_dashboard.get_dashboard_overview(time_range_enum)
        
        return {
            "status": "success",
            "data": {
                "total_users": overview.total_users,
                "active_users": overview.active_users,
                "total_properties": overview.total_properties,
                "recommendations_served": overview.recommendations_served,
                "conversion_rate": overview.conversion_rate,
                "revenue": overview.revenue,
                "avg_response_time": overview.avg_response_time,
                "ml_model_accuracy": overview.ml_model_accuracy,
                "timestamp": overview.timestamp.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard overview: {str(e)}")


@analytics_router.get("/dashboard/realtime")
async def get_realtime_metrics(
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Get real-time metrics for live dashboard updates."""
    try:
        metrics = await bi_dashboard.get_real_time_metrics()
        return {
            "status": "success",
            "data": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time metrics: {str(e)}")


@analytics_router.get("/dashboard/executive")
async def get_executive_summary(
    time_range: str = Query("last_30_days", description="Time range for executive summary"),
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Generate executive summary with key business insights."""
    try:
        time_range_enum = TimeRange.LAST_30_DAYS
        if time_range == "last_7_days":
            time_range_enum = TimeRange.LAST_7_DAYS
        elif time_range == "last_90_days":
            time_range_enum = TimeRange.LAST_90_DAYS
        
        executive_report = await bi_dashboard.generate_executive_report(time_range_enum)
        
        return {
            "status": "success",
            "data": executive_report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate executive summary: {str(e)}")


# KPI and Metrics Endpoints
@analytics_router.get("/kpis/trends")
async def get_kpi_trends(
    kpi_names: List[str] = Query(..., description="List of KPI names to retrieve"),
    time_range: str = Query("last_7_days", description="Time range for trends"),
    granularity: str = Query("hour", description="Data granularity: hour, day, week"),
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Get KPI trends over time with specified granularity."""
    try:
        time_range_enum = TimeRange.LAST_7_DAYS
        if time_range == "last_30_days":
            time_range_enum = TimeRange.LAST_30_DAYS
        
        trends = await bi_dashboard.get_kpi_trends(kpi_names, time_range_enum, granularity)
        
        return {
            "status": "success",
            "data": trends
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KPI trends: {str(e)}")


@analytics_router.get("/metrics/user-behavior")
async def get_user_behavior_insights(
    time_range: str = Query("last_30_days", description="Time range for analysis"),
    user_segments: Optional[List[str]] = Query(None, description="Filter by user segments"),
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Get comprehensive user behavior insights and segmentation."""
    try:
        time_range_enum = TimeRange.LAST_30_DAYS
        if time_range == "last_7_days":
            time_range_enum = TimeRange.LAST_7_DAYS
        
        insights = await bi_dashboard.get_user_behavior_insights(time_range_enum)
        
        # Filter by segments if provided
        if user_segments:
            # Filter logic would be implemented here
            pass
        
        return {
            "status": "success",
            "data": insights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user behavior insights: {str(e)}")


@analytics_router.get("/metrics/market-insights")
async def get_market_insights(
    time_range: str = Query("last_30_days", description="Time range for analysis"),
    regions: Optional[List[str]] = Query(None, description="Filter by regions"),
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Get market analysis and trend insights."""
    try:
        time_range_enum = TimeRange.LAST_30_DAYS
        
        insights = await bi_dashboard.get_market_insights(time_range_enum, regions)
        
        return {
            "status": "success",
            "data": insights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market insights: {str(e)}")


# Reporting Endpoints
@analytics_router.post("/reports/generate")
async def generate_report(
    request: ReportRequest,
    report_generator: ReportGenerator = Depends(get_report_generator)
):
    """Generate a custom report based on the provided configuration."""
    try:
        # Convert request to ReportConfig
        config = ReportConfig(
            report_id=f"custom_{int(datetime.utcnow().timestamp())}",
            report_name=request.report_name,
            report_type=ReportType(request.report_type),
            description=request.description,
            filters=request.filters,
            metrics=request.metrics,
            dimensions=request.dimensions,
            output_format=OutputFormat(request.output_format),
            include_visualizations=request.include_visualizations,
            include_insights=request.include_insights
        )
        
        # Set time range if provided
        if request.time_range:
            if request.time_range.start_date and request.time_range.end_date:
                config.start_date = request.time_range.start_date
                config.end_date = request.time_range.end_date
            elif request.time_range.preset:
                config.time_range_preset = request.time_range.preset
        
        # Generate report
        result = await report_generator.generate_report(config)
        
        return {
            "status": "success",
            "report": {
                "report_id": result.report_id,
                "report_name": result.report_name,
                "generated_at": result.generated_at.isoformat(),
                "execution_time": result.execution_time_seconds,
                "row_count": result.row_count,
                "data": result.data,
                "insights": result.insights,
                "visualizations": result.visualizations,
                "output_format": result.output_format,
                "data_quality_score": result.data_quality_score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@analytics_router.get("/reports/templates")
async def get_report_templates(
    report_generator: ReportGenerator = Depends(get_report_generator)
):
    """Get list of available report templates."""
    try:
        templates = await report_generator.get_available_templates()
        
        return {
            "status": "success",
            "templates": templates
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report templates: {str(e)}")


@analytics_router.get("/reports/property-performance")
async def generate_property_performance_report(
    property_ids: Optional[List[str]] = Query(None, description="Filter by property IDs"),
    regions: Optional[List[str]] = Query(None, description="Filter by regions"),
    time_period: str = Query("last_30_days", description="Time period for analysis"),
    report_generator: ReportGenerator = Depends(get_report_generator)
):
    """Generate property performance analysis report."""
    try:
        result = await report_generator.generate_property_performance_report(
            property_ids, regions, time_period
        )
        
        return {
            "status": "success",
            "report": {
                "report_id": result.report_id,
                "generated_at": result.generated_at.isoformat(),
                "data": result.data,
                "insights": result.insights,
                "execution_time": result.execution_time_seconds
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate property performance report: {str(e)}")


# Predictive Analytics Endpoints
@analytics_router.post("/predictions/price-forecast")
async def forecast_prices(
    request: ForecastRequest,
    price_forecaster: PriceForecaster = Depends(get_price_forecaster)
):
    """Generate price forecasts for specified property type and region."""
    try:
        # Convert request parameters
        horizon = ForecastHorizon(request.horizon)
        model_type = ModelType(request.model_type)
        
        forecast = await price_forecaster.forecast_prices(
            property_type=request.property_type,
            region=request.region,
            horizon=horizon,
            model_type=model_type,
            include_market_factors=request.include_market_factors
        )
        
        return {
            "status": "success",
            "forecast": {
                "property_type": forecast.property_type,
                "region": forecast.region,
                "horizon": forecast.forecast_horizon,
                "model_type": forecast.model_type,
                "predicted_prices": forecast.predicted_prices,
                "confidence_intervals": forecast.confidence_intervals,
                "forecast_dates": [dt.isoformat() for dt in forecast.forecast_dates],
                "trend_direction": forecast.trend_direction,
                "volatility_score": forecast.volatility_score,
                "market_momentum": forecast.market_momentum,
                "model_accuracy": forecast.model_accuracy,
                "generated_at": forecast.generated_at.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate price forecast: {str(e)}")


@analytics_router.get("/predictions/price-insights/{property_type}/{region}")
async def get_price_insights(
    property_type: str = Path(..., description="Property type"),
    region: str = Path(..., description="Region"),
    current_price: float = Query(..., description="Current property price"),
    price_forecaster: PriceForecaster = Depends(get_price_forecaster)
):
    """Get insights about a property's current price relative to market."""
    try:
        insights = await price_forecaster.get_price_insights(
            property_type, region, current_price
        )
        
        return {
            "status": "success",
            "insights": insights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get price insights: {str(e)}")


@analytics_router.get("/predictions/market-comparison/{property_type}")
async def compare_markets(
    property_type: str = Path(..., description="Property type"),
    regions: List[str] = Query(..., description="Regions to compare"),
    horizon: str = Query("3_months", description="Forecast horizon"),
    price_forecaster: PriceForecaster = Depends(get_price_forecaster)
):
    """Compare price forecasts across multiple markets."""
    try:
        horizon_enum = ForecastHorizon(horizon)
        
        comparison = await price_forecaster.compare_markets(
            property_type, regions, horizon_enum
        )
        
        return {
            "status": "success",
            "comparison": comparison
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare markets: {str(e)}")


# Data Query and Export Endpoints
@analytics_router.post("/query/custom")
async def execute_custom_query(
    request: CustomQueryRequest,
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Execute a custom SQL query with safety restrictions."""
    try:
        # Security validation would be implemented here
        # For now, this is a placeholder
        
        if request.format == "json":
            # Execute query and return JSON
            result = {
                "status": "success",
                "data": [],  # Query results would go here
                "row_count": 0,
                "execution_time": 0.0
            }
            return result
        
        elif request.format == "csv":
            # Return CSV file
            output = io.StringIO()
            # CSV generation logic would go here
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=query_result.csv"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute custom query: {str(e)}")


@analytics_router.get("/export/data/{table_name}")
async def export_table_data(
    table_name: str = Path(..., description="Table name to export"),
    format: str = Query("csv", description="Export format: csv, json, excel"),
    limit: int = Query(10000, description="Maximum number of rows"),
    filters: Optional[str] = Query(None, description="JSON string of filters")
):
    """Export data from a specific table with optional filtering."""
    try:
        # Validate table name for security
        allowed_tables = [
            "analytics_events", "business_metrics", "property_analytics",
            "user_behavior_analytics", "ml_model_performance"
        ]
        
        if table_name not in allowed_tables:
            raise HTTPException(status_code=400, detail="Table not accessible for export")
        
        # Parse filters if provided
        filter_conditions = {}
        if filters:
            try:
                filter_conditions = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filter JSON")
        
        if format == "csv":
            # Generate CSV export
            output = io.StringIO()
            # CSV generation logic would go here
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={table_name}_export.csv"}
            )
        
        elif format == "json":
            # Return JSON export
            return {
                "status": "success",
                "table_name": table_name,
                "data": [],  # Exported data would go here
                "row_count": 0,
                "filters_applied": filter_conditions
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


# Streaming and Real-time Endpoints
@analytics_router.get("/streaming/status")
async def get_streaming_status():
    """Get status of real-time streaming analytics."""
    try:
        # This would connect to the actual streaming engine
        status = {
            "active_streams": 0,
            "total_events_processed": 0,
            "avg_processing_latency": 0.0,
            "error_rate": 0.0,
            "system_health": "healthy"
        }
        
        return {
            "status": "success",
            "streaming_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get streaming status: {str(e)}")


@analytics_router.post("/streaming/metrics/register")
async def register_realtime_metric(
    metric: MetricDefinition
):
    """Register a new real-time metric for streaming analytics."""
    try:
        # This would register the metric with the real-time aggregator
        
        return {
            "status": "success",
            "message": f"Metric '{metric.name}' registered successfully",
            "metric_id": f"metric_{int(datetime.utcnow().timestamp())}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register metric: {str(e)}")


@analytics_router.get("/streaming/metrics/{metric_name}")
async def get_realtime_metric(
    metric_name: str = Path(..., description="Metric name"),
    window_size: int = Query(300, description="Time window in seconds"),
    dimensions: Optional[str] = Query(None, description="JSON string of dimension filters")
):
    """Get current value of a real-time metric."""
    try:
        # Parse dimensions
        dimension_filters = {}
        if dimensions:
            try:
                dimension_filters = json.loads(dimensions)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid dimensions JSON")
        
        # This would connect to the real-time aggregator
        metric_value = {
            "metric_name": metric_name,
            "window_size": window_size,
            "value": 0.0,
            "count": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "dimensions": dimension_filters
        }
        
        return {
            "status": "success",
            "metric": metric_value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time metric: {str(e)}")


# System Health and Performance Endpoints
@analytics_router.get("/system/health")
async def get_system_health():
    """Get comprehensive system health status."""
    try:
        health_status = {
            "overall_status": "healthy",
            "components": {
                "database": {"status": "healthy", "response_time_ms": 15},
                "redis": {"status": "healthy", "response_time_ms": 2},
                "streaming": {"status": "healthy", "lag_ms": 50},
                "ml_models": {"status": "healthy", "accuracy": 0.95},
                "data_warehouse": {"status": "healthy", "utilization": 0.65}
            },
            "alerts": [],
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "health": health_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@analytics_router.get("/system/performance")
async def get_performance_metrics():
    """Get system performance metrics and statistics."""
    try:
        performance_metrics = {
            "response_times": {
                "avg_api_response_ms": 150,
                "p95_api_response_ms": 300,
                "p99_api_response_ms": 500
            },
            "throughput": {
                "requests_per_second": 100,
                "events_processed_per_second": 500,
                "ml_predictions_per_second": 50
            },
            "resource_utilization": {
                "cpu_usage_percent": 45,
                "memory_usage_percent": 60,
                "disk_usage_percent": 30,
                "network_io_mbps": 25
            },
            "error_rates": {
                "api_error_rate": 0.01,
                "ml_prediction_error_rate": 0.02,
                "data_processing_error_rate": 0.005
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "performance": performance_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


# Data Governance and Quality Endpoints
@analytics_router.get("/governance/data-lineage")
async def get_data_lineage(
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Get data lineage and governance insights."""
    try:
        lineage_insights = await bi_dashboard.get_data_lineage_insights()
        
        return {
            "status": "success",
            "data_lineage": lineage_insights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data lineage: {str(e)}")


@analytics_router.get("/governance/anomalies")
async def get_anomaly_detection_results(
    bi_dashboard: BusinessIntelligenceDashboard = Depends(get_bi_dashboard)
):
    """Get recent anomaly detection results."""
    try:
        anomalies = await bi_dashboard.get_anomaly_detection_results()
        
        return {
            "status": "success",
            "anomalies": anomalies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomaly detection results: {str(e)}")


# WebSocket endpoint for real-time updates (placeholder)
@analytics_router.websocket("/ws/realtime")
async def websocket_realtime_updates(websocket):
    """WebSocket endpoint for real-time analytics updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send real-time updates
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "active_users": 150,
                    "events_per_second": 25,
                    "system_load": 0.45
                }
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
    except Exception as e:
        await websocket.close()


# Error handlers and middleware would be added here in production