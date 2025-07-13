"""
KPI Tracker for monitoring key performance indicators and business metrics.
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


class KPIType(Enum):
    """Types of KPIs that can be tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    RATE = "rate"
    PERCENTAGE = "percentage"
    AVERAGE = "average"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KPIDefinition:
    """Definition of a KPI including calculation and thresholds."""
    name: str
    description: str
    kpi_type: KPIType
    calculation_query: str
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    unit: str = ""
    category: str = "general"
    is_active: bool = True


@dataclass
class KPIValue:
    """A KPI value at a specific point in time."""
    kpi_name: str
    value: float
    timestamp: datetime
    target_value: Optional[float] = None
    variance_from_target: Optional[float] = None
    trend: Optional[str] = None  # "up", "down", "stable"


@dataclass
class KPIAlert:
    """Alert for KPI threshold breach."""
    kpi_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    timestamp: datetime


class KPITracker:
    """
    Tracks and monitors key performance indicators for the rental ML system.
    
    Provides real-time KPI calculation, trend analysis, and threshold-based alerting.
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
        self.kpi_definitions = self._initialize_kpi_definitions()
    
    def _initialize_kpi_definitions(self) -> Dict[str, KPIDefinition]:
        """Initialize default KPI definitions."""
        return {
            "daily_active_users": KPIDefinition(
                name="daily_active_users",
                description="Number of unique users active in the last 24 hours",
                kpi_type=KPIType.GAUGE,
                calculation_query="""
                    SELECT COUNT(DISTINCT user_id) as value
                    FROM user_sessions
                    WHERE last_activity > NOW() - INTERVAL '24 hours'
                """,
                target_value=1000.0,
                warning_threshold=800.0,
                critical_threshold=500.0,
                unit="users",
                category="user_engagement"
            ),
            
            "conversion_rate": KPIDefinition(
                name="conversion_rate",
                description="Percentage of recommendations that lead to property views",
                kpi_type=KPIType.PERCENTAGE,
                calculation_query="""
                    SELECT 
                        (COUNT(CASE WHEN clicked = true THEN 1 END)::float / 
                         COUNT(*)::float * 100) as value
                    FROM recommendation_events
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """,
                target_value=15.0,
                warning_threshold=10.0,
                critical_threshold=5.0,
                unit="%",
                category="conversion"
            ),
            
            "avg_response_time": KPIDefinition(
                name="avg_response_time",
                description="Average API response time in milliseconds",
                kpi_type=KPIType.AVERAGE,
                calculation_query="""
                    SELECT AVG(response_time_ms) as value
                    FROM api_metrics
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """,
                target_value=500.0,
                warning_threshold=1000.0,
                critical_threshold=2000.0,
                unit="ms",
                category="performance"
            ),
            
            "ml_model_accuracy": KPIDefinition(
                name="ml_model_accuracy",
                description="Current ML model prediction accuracy",
                kpi_type=KPIType.PERCENTAGE,
                calculation_query="""
                    SELECT accuracy * 100 as value
                    FROM ml_model_metrics
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                target_value=90.0,
                warning_threshold=85.0,
                critical_threshold=80.0,
                unit="%",
                category="ml_performance"
            ),
            
            "property_inventory": KPIDefinition(
                name="property_inventory",
                description="Total number of active properties",
                kpi_type=KPIType.GAUGE,
                calculation_query="""
                    SELECT COUNT(*) as value
                    FROM properties
                    WHERE is_active = true
                """,
                target_value=10000.0,
                warning_threshold=8000.0,
                critical_threshold=5000.0,
                unit="properties",
                category="inventory"
            ),
            
            "error_rate": KPIDefinition(
                name="error_rate",
                description="Percentage of API requests resulting in errors",
                kpi_type=KPIType.PERCENTAGE,
                calculation_query="""
                    SELECT 
                        (COUNT(CASE WHEN status_code >= 400 THEN 1 END)::float / 
                         COUNT(*)::float * 100) as value
                    FROM api_metrics
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """,
                target_value=0.0,
                warning_threshold=1.0,
                critical_threshold=5.0,
                unit="%",
                category="reliability"
            ),
            
            "cache_hit_rate": KPIDefinition(
                name="cache_hit_rate",
                description="Cache hit rate percentage",
                kpi_type=KPIType.PERCENTAGE,
                calculation_query="",  # Calculated from Redis stats
                target_value=85.0,
                warning_threshold=70.0,
                critical_threshold=50.0,
                unit="%",
                category="performance"
            ),
            
            "revenue_per_day": KPIDefinition(
                name="revenue_per_day",
                description="Daily revenue from subscriptions and ads",
                kpi_type=KPIType.GAUGE,
                calculation_query="""
                    SELECT COALESCE(SUM(amount), 0) as value
                    FROM revenue_events
                    WHERE created_at > CURRENT_DATE
                """,
                target_value=1000.0,
                warning_threshold=800.0,
                critical_threshold=500.0,
                unit="$",
                category="revenue"
            )
        }
    
    async def calculate_kpi(self, kpi_name: str) -> Optional[KPIValue]:
        """Calculate current value for a specific KPI."""
        if kpi_name not in self.kpi_definitions:
            return None
        
        kpi_def = self.kpi_definitions[kpi_name]
        cache_key = f"kpi:{kpi_name}:current"
        
        # Try to get from cache first
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            data = eval(cached_result)
            return KPIValue(**data)
        
        # Calculate the KPI value
        try:
            if kpi_name == "cache_hit_rate":
                value = await self._calculate_cache_hit_rate()
            else:
                result = await self.db_session.execute(text(kpi_def.calculation_query))
                row = result.fetchone()
                value = float(row.value) if row and row.value is not None else 0.0
            
            # Calculate variance from target
            variance_from_target = None
            if kpi_def.target_value is not None:
                variance_from_target = ((value - kpi_def.target_value) / kpi_def.target_value) * 100
            
            kpi_value = KPIValue(
                kpi_name=kpi_name,
                value=value,
                timestamp=datetime.utcnow(),
                target_value=kpi_def.target_value,
                variance_from_target=variance_from_target
            )
            
            # Cache the result
            await self.redis_client.setex(
                cache_key,
                60,  # 1 minute cache for current KPI values
                str(kpi_value.__dict__)
            )
            
            return kpi_value
            
        except Exception as e:
            print(f"Error calculating KPI {kpi_name}: {e}")
            return None
    
    async def calculate_all_kpis(self) -> Dict[str, KPIValue]:
        """Calculate all active KPIs concurrently."""
        active_kpis = [name for name, kpi_def in self.kpi_definitions.items() if kpi_def.is_active]
        
        # Calculate all KPIs concurrently
        tasks = [self.calculate_kpi(kpi_name) for kpi_name in active_kpis]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        kpi_values = {}
        for kpi_name, result in zip(active_kpis, results):
            if isinstance(result, KPIValue):
                kpi_values[kpi_name] = result
            elif isinstance(result, Exception):
                print(f"Error calculating KPI {kpi_name}: {result}")
        
        return kpi_values
    
    async def get_kpi_trends(
        self,
        kpi_names: List[str],
        time_range: str = "24h",
        granularity: str = "hour"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical trends for specified KPIs."""
        trends = {}
        
        for kpi_name in kpi_names:
            if kpi_name not in self.kpi_definitions:
                continue
            
            cache_key = f"kpi_trend:{kpi_name}:{time_range}:{granularity}"
            
            # Try to get from cache
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                trends[kpi_name] = eval(cached_result)
                continue
            
            # Calculate trend data
            trend_data = await self._calculate_kpi_trend(kpi_name, time_range, granularity)
            trends[kpi_name] = trend_data
            
            # Cache the result
            await self.redis_client.setex(cache_key, self.cache_ttl, str(trend_data))
        
        return trends
    
    async def check_kpi_thresholds(self) -> List[KPIAlert]:
        """Check all KPIs against their thresholds and generate alerts."""
        alerts = []
        kpi_values = await self.calculate_all_kpis()
        
        for kpi_name, kpi_value in kpi_values.items():
            kpi_def = self.kpi_definitions[kpi_name]
            alerts.extend(self._check_kpi_thresholds(kpi_def, kpi_value))
        
        return alerts
    
    async def get_kpi_summary(self) -> Dict[str, Any]:
        """Get a summary of all KPI statuses."""
        kpi_values = await self.calculate_all_kpis()
        
        summary = {
            "total_kpis": len(self.kpi_definitions),
            "active_kpis": len([k for k in self.kpi_definitions.values() if k.is_active]),
            "kpis_on_target": 0,
            "kpis_at_warning": 0,
            "kpis_at_critical": 0,
            "categories": {},
            "kpi_details": {}
        }
        
        for kpi_name, kpi_value in kpi_values.items():
            kpi_def = self.kpi_definitions[kpi_name]
            status = self._get_kpi_status(kpi_def, kpi_value)
            
            # Count statuses
            if status == "on_target":
                summary["kpis_on_target"] += 1
            elif status == "warning":
                summary["kpis_at_warning"] += 1
            elif status == "critical":
                summary["kpis_at_critical"] += 1
            
            # Group by category
            category = kpi_def.category
            if category not in summary["categories"]:
                summary["categories"][category] = {
                    "total": 0,
                    "on_target": 0,
                    "warning": 0,
                    "critical": 0
                }
            
            summary["categories"][category]["total"] += 1
            summary["categories"][category][status] += 1
            
            # Add detailed info
            summary["kpi_details"][kpi_name] = {
                "value": kpi_value.value,
                "target": kpi_value.target_value,
                "variance": kpi_value.variance_from_target,
                "status": status,
                "unit": kpi_def.unit,
                "category": kpi_def.category
            }
        
        return summary
    
    async def add_kpi_definition(self, kpi_def: KPIDefinition) -> bool:
        """Add a new KPI definition."""
        try:
            self.kpi_definitions[kpi_def.name] = kpi_def
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                "kpi_definitions",
                kpi_def.name,
                str(kpi_def.__dict__)
            )
            
            return True
        except Exception as e:
            print(f"Error adding KPI definition: {e}")
            return False
    
    async def update_kpi_definition(self, kpi_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing KPI definition."""
        if kpi_name not in self.kpi_definitions:
            return False
        
        try:
            kpi_def = self.kpi_definitions[kpi_name]
            
            # Update the definition
            for key, value in updates.items():
                if hasattr(kpi_def, key):
                    setattr(kpi_def, key, value)
            
            # Store updated definition
            await self.redis_client.hset(
                "kpi_definitions",
                kpi_name,
                str(kpi_def.__dict__)
            )
            
            # Clear cache for this KPI
            await self._clear_kpi_cache(kpi_name)
            
            return True
        except Exception as e:
            print(f"Error updating KPI definition: {e}")
            return False
    
    async def get_kpi_forecast(
        self,
        kpi_name: str,
        forecast_days: int = 7
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate forecast for a KPI using historical data."""
        if kpi_name not in self.kpi_definitions:
            return None
        
        # Get historical data for the last 30 days
        historical_data = await self._get_kpi_historical_data(kpi_name, days=30)
        
        if len(historical_data) < 7:  # Need at least a week of data
            return None
        
        # Simple linear trend forecast
        # In production, you might use more sophisticated forecasting models
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Calculate trend
        df['days'] = (df.index - df.index[0]).days
        correlation = np.corrcoef(df['days'], df['value'])[0, 1]
        
        if abs(correlation) < 0.1:  # No clear trend
            # Use average for forecast
            avg_value = df['value'].mean()
            forecast = [{"date": (datetime.utcnow() + timedelta(days=i)).date().isoformat(), 
                        "predicted_value": avg_value} for i in range(1, forecast_days + 1)]
        else:
            # Linear trend forecast
            slope = np.polyfit(df['days'], df['value'], 1)[0]
            last_value = df['value'].iloc[-1]
            last_day = df['days'].iloc[-1]
            
            forecast = []
            for i in range(1, forecast_days + 1):
                predicted_value = last_value + slope * i
                forecast.append({
                    "date": (datetime.utcnow() + timedelta(days=i)).date().isoformat(),
                    "predicted_value": max(0, predicted_value)  # Ensure non-negative
                })
        
        return forecast
    
    # Private helper methods
    async def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from Redis stats."""
        try:
            info = await self.redis_client.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            
            total = hits + misses
            return (hits / total * 100) if total > 0 else 0.0
        except Exception:
            return 0.0
    
    async def _calculate_kpi_trend(
        self,
        kpi_name: str,
        time_range: str,
        granularity: str
    ) -> List[Dict[str, Any]]:
        """Calculate trend data for a KPI."""
        kpi_def = self.kpi_definitions[kpi_name]
        
        # Modify query to include time grouping
        if granularity == "hour":
            time_format = "YYYY-MM-DD HH24:00:00"
            interval = "1 hour"
        elif granularity == "day":
            time_format = "YYYY-MM-DD"
            interval = "1 day"
        else:
            time_format = "YYYY-MM-DD HH24:00:00"
            interval = "1 hour"
        
        # Parse time range
        if time_range == "24h":
            time_condition = "created_at > NOW() - INTERVAL '24 hours'"
        elif time_range == "7d":
            time_condition = "created_at > NOW() - INTERVAL '7 days'"
        elif time_range == "30d":
            time_condition = "created_at > NOW() - INTERVAL '30 days'"
        else:
            time_condition = "created_at > NOW() - INTERVAL '24 hours'"
        
        # This is a simplified implementation
        # In practice, you'd need to modify each KPI's query to support time grouping
        trend_data = []
        
        return trend_data
    
    def _check_kpi_thresholds(
        self,
        kpi_def: KPIDefinition,
        kpi_value: KPIValue
    ) -> List[KPIAlert]:
        """Check KPI value against thresholds and generate alerts."""
        alerts = []
        
        if kpi_def.critical_threshold is not None:
            if ((kpi_def.kpi_type in [KPIType.COUNTER, KPIType.GAUGE, KPIType.AVERAGE] and 
                 kpi_value.value >= kpi_def.critical_threshold) or
                (kpi_def.kpi_type == KPIType.PERCENTAGE and 
                 kpi_value.value >= kpi_def.critical_threshold)):
                
                alerts.append(KPIAlert(
                    kpi_name=kpi_def.name,
                    current_value=kpi_value.value,
                    threshold_value=kpi_def.critical_threshold,
                    severity=AlertSeverity.CRITICAL,
                    message=f"{kpi_def.name} is at critical level: {kpi_value.value}{kpi_def.unit}",
                    timestamp=datetime.utcnow()
                ))
        
        elif kpi_def.warning_threshold is not None:
            if ((kpi_def.kpi_type in [KPIType.COUNTER, KPIType.GAUGE, KPIType.AVERAGE] and 
                 kpi_value.value >= kpi_def.warning_threshold) or
                (kpi_def.kpi_type == KPIType.PERCENTAGE and 
                 kpi_value.value >= kpi_def.warning_threshold)):
                
                alerts.append(KPIAlert(
                    kpi_name=kpi_def.name,
                    current_value=kpi_value.value,
                    threshold_value=kpi_def.warning_threshold,
                    severity=AlertSeverity.MEDIUM,
                    message=f"{kpi_def.name} is above warning threshold: {kpi_value.value}{kpi_def.unit}",
                    timestamp=datetime.utcnow()
                ))
        
        return alerts
    
    def _get_kpi_status(self, kpi_def: KPIDefinition, kpi_value: KPIValue) -> str:
        """Determine the status of a KPI based on its value and thresholds."""
        if kpi_def.critical_threshold is not None and kpi_value.value >= kpi_def.critical_threshold:
            return "critical"
        elif kpi_def.warning_threshold is not None and kpi_value.value >= kpi_def.warning_threshold:
            return "warning"
        else:
            return "on_target"
    
    async def _clear_kpi_cache(self, kpi_name: str) -> None:
        """Clear all cached data for a KPI."""
        pattern = f"kpi:{kpi_name}:*"
        keys = await self.redis_client.keys(pattern)
        if keys:
            await self.redis_client.delete(*keys)
    
    async def _get_kpi_historical_data(
        self,
        kpi_name: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical data for KPI forecasting."""
        # This would query historical KPI values from a time-series database
        # For now, return empty list
        return []