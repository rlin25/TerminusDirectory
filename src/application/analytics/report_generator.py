"""
Report Generator for creating automated and custom reports from analytics data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis
import json
import base64
from io import BytesIO


class ReportType(Enum):
    """Types of reports that can be generated."""
    EXECUTIVE_SUMMARY = "executive_summary"
    PROPERTY_PERFORMANCE = "property_performance"
    USER_ENGAGEMENT = "user_engagement"
    REVENUE_REPORT = "revenue_report"
    ML_MODEL_PERFORMANCE = "ml_model_performance"
    MARKET_ANALYSIS = "market_analysis"
    OPERATIONAL_METRICS = "operational_metrics"
    CUSTOM_ANALYTICS = "custom_analytics"


class ReportFormat(Enum):
    """Output formats for reports."""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"


class ReportFrequency(Enum):
    """Report generation frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    ON_DEMAND = "on_demand"


@dataclass
class ReportConfiguration:
    """Configuration for report generation."""
    report_id: str
    report_type: ReportType
    name: str
    description: str
    frequency: ReportFrequency
    output_format: ReportFormat
    recipients: List[str]
    metrics: List[str]
    dimensions: List[str]
    filters: Dict[str, Any]
    template_id: Optional[str] = None
    is_active: bool = True
    created_by: str = "system"
    created_at: datetime = None


@dataclass
class ReportData:
    """Container for report data and metadata."""
    report_id: str
    report_type: ReportType
    title: str
    description: str
    generated_at: datetime
    time_period: Dict[str, datetime]
    executive_summary: Dict[str, Any]
    key_metrics: Dict[str, Any]
    detailed_data: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    recommendations: List[str]
    appendices: Dict[str, Any]


@dataclass
class ScheduledReport:
    """Scheduled report execution details."""
    report_config_id: str
    next_execution: datetime
    last_execution: Optional[datetime]
    execution_count: int
    success_count: int
    failure_count: int
    last_error: Optional[str]


class ReportGenerator:
    """
    Generates automated and custom reports from analytics data.
    
    Supports various report types, formats, and automated scheduling.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        analytics_modules: Dict[str, Any]
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.analytics_modules = analytics_modules
        
        # Report templates
        self.report_templates = self._initialize_report_templates()
    
    def _initialize_report_templates(self) -> Dict[ReportType, Dict[str, Any]]:
        """Initialize default report templates."""
        return {
            ReportType.EXECUTIVE_SUMMARY: {
                "sections": [
                    "business_overview",
                    "key_metrics",
                    "performance_highlights",
                    "market_insights",
                    "recommendations"
                ],
                "key_metrics": [
                    "total_revenue",
                    "active_users",
                    "conversion_rate",
                    "ml_model_accuracy",
                    "customer_satisfaction"
                ]
            },
            
            ReportType.PROPERTY_PERFORMANCE: {
                "sections": [
                    "inventory_overview",
                    "listing_performance",
                    "market_trends",
                    "pricing_analysis",
                    "regional_breakdown"
                ],
                "key_metrics": [
                    "total_properties",
                    "new_listings",
                    "avg_days_on_market",
                    "occupancy_rate",
                    "avg_price_per_sqft"
                ]
            },
            
            ReportType.USER_ENGAGEMENT: {
                "sections": [
                    "user_base_overview",
                    "engagement_metrics",
                    "behavior_patterns",
                    "conversion_funnel",
                    "user_segmentation"
                ],
                "key_metrics": [
                    "daily_active_users",
                    "session_duration",
                    "page_views",
                    "bounce_rate",
                    "conversion_rate"
                ]
            },
            
            ReportType.REVENUE_REPORT: {
                "sections": [
                    "revenue_overview",
                    "revenue_streams",
                    "subscription_metrics",
                    "customer_lifetime_value",
                    "revenue_forecast"
                ],
                "key_metrics": [
                    "total_revenue",
                    "mrr",
                    "arpu",
                    "churn_rate",
                    "customer_acquisition_cost"
                ]
            },
            
            ReportType.ML_MODEL_PERFORMANCE: {
                "sections": [
                    "model_overview",
                    "performance_metrics",
                    "drift_analysis",
                    "model_comparison",
                    "optimization_recommendations"
                ],
                "key_metrics": [
                    "accuracy",
                    "precision",
                    "recall",
                    "inference_time",
                    "model_drift_score"
                ]
            }
        }
    
    async def generate_report(
        self,
        report_config: ReportConfiguration,
        time_period: Optional[Dict[str, datetime]] = None
    ) -> ReportData:
        """Generate a report based on configuration."""
        if not time_period:
            time_period = self._get_default_time_period(report_config.frequency)
        
        # Get report template
        template = self.report_templates.get(report_config.report_type)
        if not template:
            raise ValueError(f"No template found for report type: {report_config.report_type}")
        
        # Generate report data based on type
        if report_config.report_type == ReportType.EXECUTIVE_SUMMARY:
            report_data = await self._generate_executive_summary(report_config, time_period)
        elif report_config.report_type == ReportType.PROPERTY_PERFORMANCE:
            report_data = await self._generate_property_performance_report(report_config, time_period)
        elif report_config.report_type == ReportType.USER_ENGAGEMENT:
            report_data = await self._generate_user_engagement_report(report_config, time_period)
        elif report_config.report_type == ReportType.REVENUE_REPORT:
            report_data = await self._generate_revenue_report(report_config, time_period)
        elif report_config.report_type == ReportType.ML_MODEL_PERFORMANCE:
            report_data = await self._generate_ml_performance_report(report_config, time_period)
        elif report_config.report_type == ReportType.MARKET_ANALYSIS:
            report_data = await self._generate_market_analysis_report(report_config, time_period)
        elif report_config.report_type == ReportType.OPERATIONAL_METRICS:
            report_data = await self._generate_operational_metrics_report(report_config, time_period)
        elif report_config.report_type == ReportType.CUSTOM_ANALYTICS:
            report_data = await self._generate_custom_analytics_report(report_config, time_period)
        else:
            raise ValueError(f"Unsupported report type: {report_config.report_type}")
        
        # Store report in database
        await self._store_report(report_data)
        
        return report_data
    
    async def schedule_report(
        self,
        report_config: ReportConfiguration
    ) -> ScheduledReport:
        """Schedule a report for automated generation."""
        next_execution = self._calculate_next_execution(
            report_config.frequency,
            datetime.utcnow()
        )
        
        scheduled_report = ScheduledReport(
            report_config_id=report_config.report_id,
            next_execution=next_execution,
            last_execution=None,
            execution_count=0,
            success_count=0,
            failure_count=0,
            last_error=None
        )
        
        # Store scheduled report
        await self._store_scheduled_report(scheduled_report)
        
        # Store report configuration
        await self._store_report_configuration(report_config)
        
        return scheduled_report
    
    async def execute_scheduled_reports(self) -> List[Dict[str, Any]]:
        """Execute all scheduled reports that are due."""
        current_time = datetime.utcnow()
        
        # Get due reports
        due_reports = await self._get_due_reports(current_time)
        
        execution_results = []
        
        for scheduled_report in due_reports:
            try:
                # Get report configuration
                report_config = await self._get_report_configuration(
                    scheduled_report.report_config_id
                )
                
                if report_config and report_config.is_active:
                    # Generate report
                    report_data = await self.generate_report(report_config)
                    
                    # Send report to recipients
                    await self._send_report(report_data, report_config)
                    
                    # Update scheduled report
                    scheduled_report.last_execution = current_time
                    scheduled_report.execution_count += 1
                    scheduled_report.success_count += 1
                    scheduled_report.next_execution = self._calculate_next_execution(
                        report_config.frequency,
                        current_time
                    )
                    scheduled_report.last_error = None
                    
                    execution_results.append({
                        "report_id": report_config.report_id,
                        "status": "success",
                        "generated_at": report_data.generated_at
                    })
                
            except Exception as e:
                # Handle execution error
                scheduled_report.failure_count += 1
                scheduled_report.last_error = str(e)
                
                execution_results.append({
                    "report_id": scheduled_report.report_config_id,
                    "status": "error",
                    "error": str(e)
                })
            
            # Update scheduled report
            await self._update_scheduled_report(scheduled_report)
        
        return execution_results
    
    async def export_report(
        self,
        report_data: ReportData,
        format: ReportFormat
    ) -> bytes:
        """Export report data to specified format."""
        if format == ReportFormat.JSON:
            return self._export_to_json(report_data)
        elif format == ReportFormat.CSV:
            return self._export_to_csv(report_data)
        elif format == ReportFormat.HTML:
            return self._export_to_html(report_data)
        elif format == ReportFormat.PDF:
            return await self._export_to_pdf(report_data)
        elif format == ReportFormat.EXCEL:
            return self._export_to_excel(report_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def create_custom_query_report(
        self,
        query: str,
        parameters: Dict[str, Any],
        title: str,
        description: str
    ) -> ReportData:
        """Create a custom report from a SQL query."""
        try:
            # Execute custom query
            result = await self.db_session.execute(text(query), parameters)
            data = result.fetchall()
            
            # Convert to list of dictionaries
            columns = result.keys()
            detailed_data = [dict(zip(columns, row)) for row in data]
            
            # Create basic analytics
            key_metrics = {
                "total_records": len(detailed_data),
                "query_execution_time": "< 1s",  # Would measure actual time
                "data_freshness": datetime.utcnow().isoformat()
            }
            
            # Generate summary
            executive_summary = {
                "overview": f"Custom query returned {len(detailed_data)} records",
                "query": query,
                "parameters": parameters
            }
            
            report_data = ReportData(
                report_id=f"custom_{int(datetime.utcnow().timestamp())}",
                report_type=ReportType.CUSTOM_ANALYTICS,
                title=title,
                description=description,
                generated_at=datetime.utcnow(),
                time_period={
                    "start": datetime.utcnow() - timedelta(days=30),
                    "end": datetime.utcnow()
                },
                executive_summary=executive_summary,
                key_metrics=key_metrics,
                detailed_data={"query_results": detailed_data},
                visualizations=[],
                recommendations=[],
                appendices={"query_metadata": {"query": query, "parameters": parameters}}
            )
            
            return report_data
            
        except Exception as e:
            raise ValueError(f"Error executing custom query: {str(e)}")
    
    # Report generation methods for different types
    async def _generate_executive_summary(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate executive summary report."""
        # Get data from analytics modules
        business_intelligence = self.analytics_modules.get("business_intelligence")
        
        if business_intelligence:
            # Get dashboard overview
            overview = await business_intelligence.get_dashboard_overview()
            
            # Get KPI summary
            kpi_summary = await business_intelligence.kpi_tracker.get_kpi_summary()
            
            # Get real-time metrics
            real_time = await business_intelligence.get_real_time_metrics()
            
            executive_summary = {
                "period_summary": f"Performance summary for {time_period['start'].strftime('%Y-%m-%d')} to {time_period['end'].strftime('%Y-%m-%d')}",
                "key_highlights": [
                    f"Total users: {overview.total_users:,}",
                    f"Active users: {overview.active_users:,}",
                    f"Revenue: ${overview.revenue:,.2f}",
                    f"ML model accuracy: {overview.ml_model_accuracy:.1%}"
                ],
                "business_health": "Good" if kpi_summary["kpis_on_target"] > kpi_summary["kpis_at_critical"] else "Needs Attention"
            }
            
            key_metrics = {
                "total_revenue": overview.revenue,
                "active_users": overview.active_users,
                "conversion_rate": overview.conversion_rate,
                "ml_model_accuracy": overview.ml_model_accuracy,
                "kpi_health_score": kpi_summary["kpis_on_target"] / kpi_summary["total_kpis"] * 100
            }
            
            detailed_data = {
                "dashboard_overview": asdict(overview),
                "kpi_summary": kpi_summary,
                "real_time_metrics": real_time
            }
            
            recommendations = [
                "Continue monitoring user engagement trends",
                "Optimize ML model performance",
                "Focus on revenue growth initiatives"
            ]
        else:
            # Fallback data
            executive_summary = {"message": "Business Intelligence module not available"}
            key_metrics = {}
            detailed_data = {}
            recommendations = []
        
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="Executive Summary Report",
            description="High-level business performance overview",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary=executive_summary,
            key_metrics=key_metrics,
            detailed_data=detailed_data,
            visualizations=[],
            recommendations=recommendations,
            appendices={}
        )
    
    async def _generate_property_performance_report(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate property performance report."""
        # Get property performance data
        market_analysis = self.analytics_modules.get("market_analysis")
        
        if market_analysis:
            market_insights = await market_analysis.get_market_insights()
            regional_metrics = await market_analysis.get_regional_market_metrics()
            
            executive_summary = {
                "total_properties": sum(m.inventory_count for m in regional_metrics.values()),
                "avg_days_on_market": np.mean([m.days_on_market for m in regional_metrics.values()]),
                "market_health": "Stable"
            }
            
            key_metrics = {
                "total_inventory": sum(m.inventory_count for m in regional_metrics.values()),
                "new_listings": sum(m.new_listings for m in regional_metrics.values()),
                "avg_price": np.mean([m.avg_price for m in regional_metrics.values()]),
                "absorption_rate": np.mean([m.absorption_rate for m in regional_metrics.values()])
            }
            
            detailed_data = {
                "market_insights": market_insights,
                "regional_metrics": {k: asdict(v) for k, v in regional_metrics.items()}
            }
        else:
            executive_summary = {"message": "Market Analysis module not available"}
            key_metrics = {}
            detailed_data = {}
        
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="Property Performance Report",
            description="Comprehensive property market performance analysis",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary=executive_summary,
            key_metrics=key_metrics,
            detailed_data=detailed_data,
            visualizations=[],
            recommendations=[],
            appendices={}
        )
    
    async def _generate_user_engagement_report(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate user engagement report."""
        # Implementation for user engagement report
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="User Engagement Report",
            description="User behavior and engagement analysis",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary={},
            key_metrics={},
            detailed_data={},
            visualizations=[],
            recommendations=[],
            appendices={}
        )
    
    async def _generate_revenue_report(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate revenue report."""
        # Implementation for revenue report
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="Revenue Report",
            description="Revenue analysis and forecasting",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary={},
            key_metrics={},
            detailed_data={},
            visualizations=[],
            recommendations=[],
            appendices={}
        )
    
    async def _generate_ml_performance_report(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate ML model performance report."""
        # Implementation for ML performance report
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="ML Model Performance Report",
            description="Machine learning model performance analysis",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary={},
            key_metrics={},
            detailed_data={},
            visualizations=[],
            recommendations=[],
            appendices={}
        )
    
    async def _generate_market_analysis_report(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate market analysis report."""
        # Implementation for market analysis report
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="Market Analysis Report",
            description="Property market trends and opportunities",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary={},
            key_metrics={},
            detailed_data={},
            visualizations=[],
            recommendations=[],
            appendices={}
        )
    
    async def _generate_operational_metrics_report(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate operational metrics report."""
        # Implementation for operational metrics report
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="Operational Metrics Report",
            description="System performance and operational metrics",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary={},
            key_metrics={},
            detailed_data={},
            visualizations=[],
            recommendations=[],
            appendices={}
        )
    
    async def _generate_custom_analytics_report(
        self,
        config: ReportConfiguration,
        time_period: Dict[str, datetime]
    ) -> ReportData:
        """Generate custom analytics report."""
        # Implementation for custom analytics report
        return ReportData(
            report_id=config.report_id,
            report_type=config.report_type,
            title="Custom Analytics Report",
            description="Custom data analysis",
            generated_at=datetime.utcnow(),
            time_period=time_period,
            executive_summary={},
            key_metrics={},
            detailed_data={},
            visualizations=[],
            recommendations=[],
            appendices={}
        )
    
    # Helper methods
    def _get_default_time_period(self, frequency: ReportFrequency) -> Dict[str, datetime]:
        """Get default time period based on frequency."""
        end_time = datetime.utcnow()
        
        if frequency == ReportFrequency.DAILY:
            start_time = end_time - timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            start_time = end_time - timedelta(days=7)
        elif frequency == ReportFrequency.MONTHLY:
            start_time = end_time - timedelta(days=30)
        elif frequency == ReportFrequency.QUARTERLY:
            start_time = end_time - timedelta(days=90)
        elif frequency == ReportFrequency.ANNUALLY:
            start_time = end_time - timedelta(days=365)
        else:
            start_time = end_time - timedelta(days=7)  # Default to weekly
        
        return {"start": start_time, "end": end_time}
    
    def _calculate_next_execution(
        self,
        frequency: ReportFrequency,
        current_time: datetime
    ) -> datetime:
        """Calculate next execution time based on frequency."""
        if frequency == ReportFrequency.DAILY:
            return current_time + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            return current_time + timedelta(days=7)
        elif frequency == ReportFrequency.MONTHLY:
            return current_time + timedelta(days=30)
        elif frequency == ReportFrequency.QUARTERLY:
            return current_time + timedelta(days=90)
        elif frequency == ReportFrequency.ANNUALLY:
            return current_time + timedelta(days=365)
        else:
            return current_time + timedelta(days=1)  # Default to daily
    
    # Export methods
    def _export_to_json(self, report_data: ReportData) -> bytes:
        """Export report to JSON format."""
        data = asdict(report_data)
        # Convert datetime objects to strings
        data = self._serialize_datetime(data)
        return json.dumps(data, indent=2).encode('utf-8')
    
    def _export_to_csv(self, report_data: ReportData) -> bytes:
        """Export report to CSV format."""
        # Create CSV from key metrics and summary data
        csv_data = []
        
        # Add key metrics
        for metric, value in report_data.key_metrics.items():
            csv_data.append([metric, value])
        
        # Convert to DataFrame and then CSV
        df = pd.DataFrame(csv_data, columns=['Metric', 'Value'])
        return df.to_csv(index=False).encode('utf-8')
    
    def _export_to_html(self, report_data: ReportData) -> bytes:
        """Export report to HTML format."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
                .metric {{ margin: 10px 0; }}
                .section {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_data.title}</h1>
                <p>{report_data.description}</p>
                <p>Generated: {report_data.generated_at}</p>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                {''.join([f'<div class="metric"><strong>{k}:</strong> {v}</div>' for k, v in report_data.key_metrics.items()])}
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <pre>{json.dumps(report_data.executive_summary, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        return html_template.encode('utf-8')
    
    async def _export_to_pdf(self, report_data: ReportData) -> bytes:
        """Export report to PDF format."""
        # This would use a library like weasyprint or reportlab
        # For now, return HTML content as placeholder
        return self._export_to_html(report_data)
    
    def _export_to_excel(self, report_data: ReportData) -> bytes:
        """Export report to Excel format."""
        # Create Excel file with multiple sheets
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Key metrics sheet
            if report_data.key_metrics:
                metrics_df = pd.DataFrame(
                    list(report_data.key_metrics.items()),
                    columns=['Metric', 'Value']
                )
                metrics_df.to_excel(writer, sheet_name='Key Metrics', index=False)
            
            # Detailed data sheets
            for sheet_name, data in report_data.detailed_data.items():
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
        
        buffer.seek(0)
        return buffer.read()
    
    def _serialize_datetime(self, obj):
        """Recursively serialize datetime objects to strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        else:
            return obj
    
    # Database operations (placeholder implementations)
    async def _store_report(self, report_data: ReportData) -> None:
        """Store report data in database."""
        # Implementation would store report in database
        pass
    
    async def _store_scheduled_report(self, scheduled_report: ScheduledReport) -> None:
        """Store scheduled report in database."""
        # Implementation would store scheduled report
        pass
    
    async def _store_report_configuration(self, config: ReportConfiguration) -> None:
        """Store report configuration in database."""
        # Implementation would store report configuration
        pass
    
    async def _get_due_reports(self, current_time: datetime) -> List[ScheduledReport]:
        """Get reports that are due for execution."""
        # Implementation would query database for due reports
        return []
    
    async def _get_report_configuration(self, config_id: str) -> Optional[ReportConfiguration]:
        """Get report configuration by ID."""
        # Implementation would retrieve configuration from database
        return None
    
    async def _update_scheduled_report(self, scheduled_report: ScheduledReport) -> None:
        """Update scheduled report status."""
        # Implementation would update scheduled report in database
        pass
    
    async def _send_report(self, report_data: ReportData, config: ReportConfiguration) -> None:
        """Send report to recipients."""
        # Implementation would send report via email, webhook, etc.
        pass