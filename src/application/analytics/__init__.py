"""
Analytics package for the Rental ML System.

This package provides comprehensive business intelligence and analytics capabilities
including real-time metrics, KPI tracking, user behavior analysis, and ML model performance.
"""

from .business_intelligence import BusinessIntelligenceDashboard
from .kpi_tracker import KPITracker
from .user_behavior_analytics import UserBehaviorAnalytics
from .market_analysis import MarketAnalysis
from .revenue_analytics import RevenueAnalytics
from .ml_performance_analytics import MLPerformanceAnalytics
from .report_generator import ReportGenerator

__all__ = [
    "BusinessIntelligenceDashboard",
    "KPITracker",
    "UserBehaviorAnalytics", 
    "MarketAnalysis",
    "RevenueAnalytics",
    "MLPerformanceAnalytics",
    "ReportGenerator"
]