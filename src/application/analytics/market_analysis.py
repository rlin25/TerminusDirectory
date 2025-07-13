"""
Market Analysis module for property market insights and trend detection.
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


class MarketTrend(Enum):
    """Market trend directions."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    STABLE = "stable"
    VOLATILE = "volatile"


class PropertyType(Enum):
    """Property types for analysis."""
    APARTMENT = "apartment"
    HOUSE = "house"
    CONDO = "condo"
    STUDIO = "studio"
    TOWNHOUSE = "townhouse"


@dataclass
class MarketMetrics:
    """Market metrics for a specific region/property type."""
    region: str
    property_type: Optional[PropertyType]
    avg_price: float
    median_price: float
    price_per_sqft: float
    inventory_count: int
    new_listings: int
    days_on_market: float
    absorption_rate: float
    price_trend: MarketTrend
    demand_score: float
    supply_score: float
    timestamp: datetime


@dataclass
class PriceAnalysis:
    """Price analysis for market segments."""
    segment: str
    current_price: float
    price_change_1m: float
    price_change_3m: float
    price_change_6m: float
    price_change_1y: float
    volatility: float
    price_forecast_30d: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class MarketOpportunity:
    """Identified market opportunity."""
    opportunity_id: str
    region: str
    property_type: PropertyType
    opportunity_type: str  # "undervalued", "high_demand", "emerging_market"
    score: float
    description: str
    metrics: Dict[str, float]
    risk_level: str
    potential_roi: Optional[float] = None


class MarketAnalysis:
    """
    Analyzes property market trends, pricing patterns, and identifies opportunities.
    
    Provides comprehensive market insights for data-driven investment decisions.
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
    
    async def get_market_insights(
        self,
        time_range: str = "30d",
        regions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive market insights and analysis."""
        cache_key = f"market_insights:{time_range}:{hash(str(regions))}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        # Calculate time bounds
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Gather insights concurrently
        tasks = [
            self._get_market_overview(start_time, end_time, regions),
            self._get_price_trends(start_time, end_time, regions),
            self._get_inventory_analysis(start_time, end_time, regions),
            self._get_demand_supply_metrics(start_time, end_time, regions),
            self._get_regional_comparison(start_time, end_time, regions),
            self._identify_market_opportunities(start_time, end_time, regions)
        ]
        
        (market_overview, price_trends, inventory_analysis, 
         demand_supply, regional_comparison, opportunities) = await asyncio.gather(*tasks)
        
        insights = {
            "time_range": time_range,
            "regions": regions or "all",
            "market_overview": market_overview,
            "price_trends": price_trends,
            "inventory_analysis": inventory_analysis,
            "demand_supply_metrics": demand_supply,
            "regional_comparison": regional_comparison,
            "market_opportunities": opportunities,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Cache the result
        await self.redis_client.setex(cache_key, self.cache_ttl, str(insights))
        
        return insights
    
    async def analyze_price_trends(
        self,
        property_type: Optional[PropertyType] = None,
        regions: Optional[List[str]] = None,
        time_range: str = "1y"
    ) -> Dict[str, PriceAnalysis]:
        """Analyze price trends for properties."""
        cache_key = f"price_trends:{property_type}:{hash(str(regions))}:{time_range}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return {k: PriceAnalysis(**v) for k, v in eval(cached_result).items()}
        
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Build filters
        property_filter = ""
        if property_type:
            property_filter = "AND property_type = :property_type"
        
        region_filter = ""
        if regions:
            region_filter = f"AND region IN ({','.join(['%s'] * len(regions))})"
        
        # Get price history data
        query = text(f"""
            SELECT 
                region,
                property_type,
                DATE_TRUNC('month', listed_date) as month,
                AVG(price) as avg_price,
                COUNT(*) as property_count
            FROM property_listings
            WHERE listed_date BETWEEN :start_time AND :end_time
            {property_filter}
            {region_filter}
            GROUP BY region, property_type, DATE_TRUNC('month', listed_date)
            ORDER BY region, property_type, month
        """)
        
        params = {"start_time": start_time, "end_time": end_time}
        if property_type:
            params["property_type"] = property_type.value
        
        result = await self.db_session.execute(query, params)
        price_data = result.fetchall()
        
        # Process price trends
        price_analyses = {}
        
        # Group by region and property type
        grouped_data = {}
        for row in price_data:
            key = f"{row.region}_{row.property_type}"
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append({
                "month": row.month,
                "avg_price": float(row.avg_price),
                "count": row.property_count
            })
        
        for segment_key, data in grouped_data.items():
            if len(data) < 3:  # Need at least 3 months of data
                continue
            
            # Sort by month
            data.sort(key=lambda x: x["month"])
            
            # Calculate price changes
            current_price = data[-1]["avg_price"]
            
            # Find prices for different periods
            price_1m = self._find_price_for_period(data, 1)
            price_3m = self._find_price_for_period(data, 3)
            price_6m = self._find_price_for_period(data, 6)
            price_1y = self._find_price_for_period(data, 12)
            
            # Calculate changes
            change_1m = ((current_price - price_1m) / price_1m * 100) if price_1m else 0
            change_3m = ((current_price - price_3m) / price_3m * 100) if price_3m else 0
            change_6m = ((current_price - price_6m) / price_6m * 100) if price_6m else 0
            change_1y = ((current_price - price_1y) / price_1y * 100) if price_1y else 0
            
            # Calculate volatility (standard deviation of month-over-month changes)
            prices = [d["avg_price"] for d in data]
            if len(prices) > 1:
                pct_changes = [(prices[i] - prices[i-1]) / prices[i-1] * 100 
                              for i in range(1, len(prices))]
                volatility = np.std(pct_changes)
            else:
                volatility = 0.0
            
            # Generate forecast (simple linear trend)
            forecast_30d = self._forecast_price(data, 1)
            
            price_analysis = PriceAnalysis(
                segment=segment_key,
                current_price=current_price,
                price_change_1m=change_1m,
                price_change_3m=change_3m,
                price_change_6m=change_6m,
                price_change_1y=change_1y,
                volatility=volatility,
                price_forecast_30d=forecast_30d
            )
            
            price_analyses[segment_key] = price_analysis
        
        # Cache the result
        cache_data = {k: analysis.__dict__ for k, analysis in price_analyses.items()}
        await self.redis_client.setex(cache_key, 3600, str(cache_data))  # 1 hour cache
        
        return price_analyses
    
    async def get_regional_market_metrics(
        self,
        regions: Optional[List[str]] = None
    ) -> Dict[str, MarketMetrics]:
        """Get market metrics for specific regions."""
        cache_key = f"regional_metrics:{hash(str(regions))}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return {k: MarketMetrics(**v) for k, v in eval(cached_result).items()}
        
        region_filter = ""
        if regions:
            region_filter = f"WHERE region IN ({','.join(['%s'] * len(regions))})"
        
        # Get current market metrics
        query = text(f"""
            SELECT 
                region,
                property_type,
                AVG(price) as avg_price,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
                AVG(price / square_feet) as price_per_sqft,
                COUNT(*) as inventory_count,
                COUNT(CASE WHEN listed_date > NOW() - INTERVAL '30 days' THEN 1 END) as new_listings,
                AVG(EXTRACT(days FROM NOW() - listed_date)) as days_on_market
            FROM property_listings
            {region_filter}
            AND is_active = true
            GROUP BY region, property_type
        """)
        
        result = await self.db_session.execute(query)
        metrics_data = result.fetchall()
        
        regional_metrics = {}
        
        for row in metrics_data:
            # Calculate additional metrics
            absorption_rate = await self._calculate_absorption_rate(row.region, row.property_type)
            demand_score = await self._calculate_demand_score(row.region, row.property_type)
            supply_score = await self._calculate_supply_score(row.region, row.property_type)
            price_trend = await self._determine_price_trend(row.region, row.property_type)
            
            metrics = MarketMetrics(
                region=row.region,
                property_type=PropertyType(row.property_type) if row.property_type else None,
                avg_price=float(row.avg_price),
                median_price=float(row.median_price),
                price_per_sqft=float(row.price_per_sqft or 0),
                inventory_count=row.inventory_count,
                new_listings=row.new_listings,
                days_on_market=float(row.days_on_market or 0),
                absorption_rate=absorption_rate,
                price_trend=price_trend,
                demand_score=demand_score,
                supply_score=supply_score,
                timestamp=datetime.utcnow()
            )
            
            key = f"{row.region}_{row.property_type or 'all'}"
            regional_metrics[key] = metrics
        
        # Cache the result
        cache_data = {k: metrics.__dict__ for k, metrics in regional_metrics.items()}
        await self.redis_client.setex(cache_key, 1800, str(cache_data))  # 30 minutes cache
        
        return regional_metrics
    
    async def identify_investment_opportunities(
        self,
        min_score: float = 0.7,
        max_risk: str = "medium"
    ) -> List[MarketOpportunity]:
        """Identify potential investment opportunities in the market."""
        cache_key = f"investment_opportunities:{min_score}:{max_risk}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return [MarketOpportunity(**op) for op in eval(cached_result)]
        
        opportunities = []
        
        # Get market metrics for analysis
        regional_metrics = await self.get_regional_market_metrics()
        
        for key, metrics in regional_metrics.items():
            # Analyze for different opportunity types
            
            # 1. Undervalued markets (low price but high demand)
            if (metrics.demand_score > 0.7 and 
                metrics.price_per_sqft < await self._get_regional_price_median(metrics.region) * 0.9):
                
                opportunity = MarketOpportunity(
                    opportunity_id=f"undervalued_{key}_{int(datetime.utcnow().timestamp())}",
                    region=metrics.region,
                    property_type=metrics.property_type or PropertyType.APARTMENT,
                    opportunity_type="undervalued",
                    score=metrics.demand_score,
                    description=f"Undervalued market in {metrics.region} with high demand",
                    metrics={
                        "demand_score": metrics.demand_score,
                        "price_per_sqft": metrics.price_per_sqft,
                        "days_on_market": metrics.days_on_market
                    },
                    risk_level="low",
                    potential_roi=15.0  # Estimated
                )
                opportunities.append(opportunity)
            
            # 2. High demand markets
            if (metrics.demand_score > 0.8 and 
                metrics.days_on_market < 30 and
                metrics.absorption_rate > 0.7):
                
                opportunity = MarketOpportunity(
                    opportunity_id=f"high_demand_{key}_{int(datetime.utcnow().timestamp())}",
                    region=metrics.region,
                    property_type=metrics.property_type or PropertyType.APARTMENT,
                    opportunity_type="high_demand",
                    score=metrics.demand_score,
                    description=f"High demand market in {metrics.region} with fast absorption",
                    metrics={
                        "demand_score": metrics.demand_score,
                        "absorption_rate": metrics.absorption_rate,
                        "days_on_market": metrics.days_on_market
                    },
                    risk_level="medium",
                    potential_roi=12.0  # Estimated
                )
                opportunities.append(opportunity)
            
            # 3. Emerging markets (increasing prices with good fundamentals)
            if (metrics.price_trend == MarketTrend.BULLISH and
                metrics.new_listings > metrics.inventory_count * 0.1 and  # 10% new listings
                metrics.supply_score < 0.6):  # Limited supply
                
                opportunity = MarketOpportunity(
                    opportunity_id=f"emerging_{key}_{int(datetime.utcnow().timestamp())}",
                    region=metrics.region,
                    property_type=metrics.property_type or PropertyType.APARTMENT,
                    opportunity_type="emerging_market",
                    score=0.8,  # Fixed score for emerging markets
                    description=f"Emerging market in {metrics.region} with bullish trends",
                    metrics={
                        "price_trend": metrics.price_trend.value,
                        "new_listings": metrics.new_listings,
                        "supply_score": metrics.supply_score
                    },
                    risk_level="high",
                    potential_roi=20.0  # Estimated
                )
                opportunities.append(opportunity)
        
        # Filter by criteria
        filtered_opportunities = [
            op for op in opportunities 
            if op.score >= min_score and self._risk_level_meets_criteria(op.risk_level, max_risk)
        ]
        
        # Sort by score
        filtered_opportunities.sort(key=lambda x: x.score, reverse=True)
        
        # Cache the result
        cache_data = [op.__dict__ for op in filtered_opportunities]
        await self.redis_client.setex(cache_key, 3600, str(cache_data))  # 1 hour cache
        
        return filtered_opportunities[:20]  # Return top 20 opportunities
    
    async def get_market_sentiment(
        self,
        regions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze market sentiment based on various indicators."""
        cache_key = f"market_sentiment:{hash(str(regions))}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        # Calculate sentiment indicators
        sentiment_data = {
            "overall_sentiment": "neutral",
            "confidence_score": 0.5,
            "key_indicators": {},
            "regional_sentiment": {},
            "trend_indicators": {}
        }
        
        # Price momentum indicator
        price_trends = await self.analyze_price_trends(time_range="3m", regions=regions)
        positive_trends = sum(1 for analysis in price_trends.values() if analysis.price_change_3m > 0)
        total_trends = len(price_trends)
        
        if total_trends > 0:
            price_momentum = positive_trends / total_trends
            sentiment_data["key_indicators"]["price_momentum"] = price_momentum
        
        # Inventory indicator
        regional_metrics = await self.get_regional_market_metrics(regions)
        avg_inventory_change = np.mean([
            metrics.new_listings / max(metrics.inventory_count, 1) 
            for metrics in regional_metrics.values()
        ])
        sentiment_data["key_indicators"]["inventory_growth"] = avg_inventory_change
        
        # Market activity indicator
        avg_days_on_market = np.mean([
            metrics.days_on_market 
            for metrics in regional_metrics.values()
        ])
        
        # Market activity score (lower days on market = higher activity = positive sentiment)
        activity_score = max(0, min(1, (60 - avg_days_on_market) / 60))
        sentiment_data["key_indicators"]["market_activity"] = activity_score
        
        # Calculate overall sentiment
        indicators = sentiment_data["key_indicators"]
        confidence_score = np.mean([
            indicators.get("price_momentum", 0.5),
            min(1, indicators.get("inventory_growth", 0.5) * 2),  # Cap at 1
            indicators.get("market_activity", 0.5)
        ])
        
        sentiment_data["confidence_score"] = confidence_score
        
        if confidence_score > 0.7:
            sentiment_data["overall_sentiment"] = "bullish"
        elif confidence_score < 0.3:
            sentiment_data["overall_sentiment"] = "bearish"
        else:
            sentiment_data["overall_sentiment"] = "neutral"
        
        # Regional sentiment breakdown
        for key, metrics in regional_metrics.items():
            region_sentiment = "neutral"
            if metrics.price_trend == MarketTrend.BULLISH and metrics.demand_score > 0.6:
                region_sentiment = "bullish"
            elif metrics.price_trend == MarketTrend.BEARISH or metrics.demand_score < 0.4:
                region_sentiment = "bearish"
            
            sentiment_data["regional_sentiment"][metrics.region] = {
                "sentiment": region_sentiment,
                "demand_score": metrics.demand_score,
                "price_trend": metrics.price_trend.value
            }
        
        # Cache the result
        await self.redis_client.setex(cache_key, 1800, str(sentiment_data))  # 30 minutes cache
        
        return sentiment_data
    
    # Private helper methods
    def _get_start_time(self, time_range: str, end_time: datetime) -> datetime:
        """Calculate start time based on time range."""
        if time_range == "1m":
            return end_time - timedelta(days=30)
        elif time_range == "3m":
            return end_time - timedelta(days=90)
        elif time_range == "6m":
            return end_time - timedelta(days=180)
        elif time_range == "1y":
            return end_time - timedelta(days=365)
        elif time_range == "30d":
            return end_time - timedelta(days=30)
        else:
            return end_time - timedelta(days=30)  # Default to 30 days
    
    def _find_price_for_period(self, data: List[Dict], months_back: int) -> Optional[float]:
        """Find price for a specific number of months back."""
        target_date = datetime.utcnow() - timedelta(days=months_back * 30)
        
        # Find closest data point
        closest_data = None
        min_diff = float('inf')
        
        for point in data:
            diff = abs((point["month"] - target_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_data = point
        
        return closest_data["avg_price"] if closest_data else None
    
    def _forecast_price(self, data: List[Dict], months_ahead: int) -> Optional[float]:
        """Simple linear forecast for price."""
        if len(data) < 3:
            return None
        
        # Extract prices and create time series
        prices = [d["avg_price"] for d in data]
        time_points = list(range(len(prices)))
        
        # Linear regression
        coeffs = np.polyfit(time_points, prices, 1)
        
        # Forecast
        future_point = len(prices) + months_ahead - 1
        forecast = coeffs[0] * future_point + coeffs[1]
        
        return max(0, forecast)  # Ensure non-negative
    
    async def _get_market_overview(
        self,
        start_time: datetime,
        end_time: datetime,
        regions: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get high-level market overview."""
        region_filter = ""
        if regions:
            region_filter = f"AND region IN ({','.join(['%s'] * len(regions))})"
        
        query = text(f"""
            SELECT 
                COUNT(*) as total_properties,
                AVG(price) as avg_price,
                COUNT(CASE WHEN listed_date > :start_time THEN 1 END) as new_listings,
                COUNT(CASE WHEN rented_date BETWEEN :start_time AND :end_time THEN 1 END) as rented_properties
            FROM property_listings
            WHERE is_active = true
            {region_filter}
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return {
            "total_properties": row.total_properties or 0,
            "avg_price": float(row.avg_price or 0),
            "new_listings": row.new_listings or 0,
            "rented_properties": row.rented_properties or 0,
            "absorption_rate": (row.rented_properties / max(row.new_listings, 1)) * 100 if row.new_listings else 0
        }
    
    async def _get_price_trends(
        self,
        start_time: datetime,
        end_time: datetime,
        regions: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get price trend analysis."""
        # This would implement detailed price trend analysis
        return {
            "trend_direction": "stable",
            "price_change_percentage": 0.0,
            "volatility": 0.0
        }
    
    async def _get_inventory_analysis(
        self,
        start_time: datetime,
        end_time: datetime,
        regions: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get inventory analysis."""
        # This would implement inventory analysis
        return {
            "inventory_levels": "normal",
            "months_of_supply": 3.0,
            "new_construction": 0
        }
    
    async def _get_demand_supply_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        regions: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get demand and supply metrics."""
        # This would implement demand/supply analysis
        return {
            "demand_index": 0.5,
            "supply_index": 0.5,
            "market_balance": "balanced"
        }
    
    async def _get_regional_comparison(
        self,
        start_time: datetime,
        end_time: datetime,
        regions: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get regional comparison data."""
        # This would implement regional comparison
        return {
            "top_performing_regions": [],
            "bottom_performing_regions": [],
            "regional_rankings": {}
        }
    
    async def _identify_market_opportunities(
        self,
        start_time: datetime,
        end_time: datetime,
        regions: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Identify market opportunities."""
        opportunities = await self.identify_investment_opportunities()
        return [op.__dict__ for op in opportunities[:5]]  # Top 5 opportunities
    
    async def _calculate_absorption_rate(self, region: str, property_type: str) -> float:
        """Calculate absorption rate for a region/property type."""
        # This would calculate the actual absorption rate
        return 0.65  # Placeholder
    
    async def _calculate_demand_score(self, region: str, property_type: str) -> float:
        """Calculate demand score for a region/property type."""
        # This would calculate demand based on searches, views, applications
        return 0.7  # Placeholder
    
    async def _calculate_supply_score(self, region: str, property_type: str) -> float:
        """Calculate supply score for a region/property type."""
        # This would calculate supply based on inventory levels and new listings
        return 0.5  # Placeholder
    
    async def _determine_price_trend(self, region: str, property_type: str) -> MarketTrend:
        """Determine price trend for a region/property type."""
        # This would analyze recent price movements
        return MarketTrend.STABLE  # Placeholder
    
    async def _get_regional_price_median(self, region: str) -> float:
        """Get median price for a region."""
        query = text("""
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price
            FROM property_listings
            WHERE region = :region AND is_active = true
        """)
        
        result = await self.db_session.execute(query, {"region": region})
        row = result.fetchone()
        
        return float(row.median_price or 0)
    
    def _risk_level_meets_criteria(self, risk_level: str, max_risk: str) -> bool:
        """Check if risk level meets maximum criteria."""
        risk_hierarchy = {"low": 1, "medium": 2, "high": 3}
        return risk_hierarchy.get(risk_level, 3) <= risk_hierarchy.get(max_risk, 3)