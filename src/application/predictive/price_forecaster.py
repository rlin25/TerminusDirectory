"""
Price Forecasting for rental property market analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import logging
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')


class ForecastHorizon(Enum):
    """Forecast time horizons."""
    SHORT_TERM = "1_month"      # 1 month
    MEDIUM_TERM = "3_months"    # 3 months
    LONG_TERM = "12_months"     # 1 year


class ModelType(Enum):
    """Types of forecasting models."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ENSEMBLE = "ensemble"


@dataclass
class ForecastResult:
    """Result of price forecasting."""
    property_type: str
    region: str
    forecast_horizon: str
    model_type: str
    
    # Forecast values
    predicted_prices: List[float]
    confidence_intervals: List[Tuple[float, float]]
    forecast_dates: List[datetime]
    
    # Model performance
    model_accuracy: float
    mae: float
    rmse: float
    r2_score: float
    
    # Additional insights
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    volatility_score: float
    market_momentum: float
    
    # Metadata
    features_used: List[str]
    data_points_count: int
    generated_at: datetime


@dataclass
class MarketFactors:
    """Market factors affecting price predictions."""
    seasonality_factor: float
    supply_demand_ratio: float
    economic_indicators: Dict[str, float]
    local_market_conditions: Dict[str, float]
    competitor_pricing: Dict[str, float]


class PriceForecaster:
    """
    Advanced Price Forecasting for rental property market analytics.
    
    Provides comprehensive price prediction using multiple ML models,
    time series analysis, and market factor integration.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        model_cache_ttl: int = 86400  # 24 hours
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.model_cache_ttl = model_cache_ttl
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        
        # Feature engineering components
        self.feature_columns = [
            'bedrooms', 'bathrooms', 'square_feet', 'property_type_encoded',
            'location_encoded', 'amenity_count', 'price_per_sqft',
            'days_on_market', 'season', 'month', 'year',
            'local_avg_price', 'supply_demand_ratio', 'market_trend'
        ]
        
        # Model performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the price forecaster."""
        try:
            # Load pre-trained models if available
            await self._load_cached_models()
            
            # Initialize encoders and scalers
            await self._initialize_feature_processors()
            
            self.logger.info("Price forecaster initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize price forecaster: {e}")
            raise
    
    async def train_models(
        self,
        property_type: Optional[str] = None,
        region: Optional[str] = None,
        retrain: bool = False
    ) -> Dict[str, Any]:
        """Train or retrain price forecasting models."""
        try:
            # Load training data
            training_data = await self._load_training_data(property_type, region)
            
            if len(training_data) < 100:
                raise ValueError(f"Insufficient training data: {len(training_data)} records")
            
            # Prepare features and target
            X, y = await self._prepare_features(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train multiple models
            model_results = {}
            
            # Linear Regression
            lr_results = await self._train_linear_regression(X_train, X_test, y_train, y_test)
            model_results[ModelType.LINEAR_REGRESSION.value] = lr_results
            
            # Random Forest
            rf_results = await self._train_random_forest(X_train, X_test, y_train, y_test)
            model_results[ModelType.RANDOM_FOREST.value] = rf_results
            
            # Gradient Boosting
            gb_results = await self._train_gradient_boosting(X_train, X_test, y_train, y_test)
            model_results[ModelType.GRADIENT_BOOSTING.value] = gb_results
            
            # Time series models (ARIMA, Exponential Smoothing)
            ts_data = training_data[['price', 'scraped_at']].sort_values('scraped_at')
            
            arima_results = await self._train_arima(ts_data)
            model_results[ModelType.ARIMA.value] = arima_results
            
            exp_smoothing_results = await self._train_exponential_smoothing(ts_data)
            model_results[ModelType.EXPONENTIAL_SMOOTHING.value] = exp_smoothing_results
            
            # Create ensemble model
            ensemble_results = await self._create_ensemble_model(model_results, X_test, y_test)
            model_results[ModelType.ENSEMBLE.value] = ensemble_results
            
            # Store models and performance metrics
            await self._save_models(property_type, region, model_results)
            
            # Update performance tracking
            best_model = max(model_results.items(), key=lambda x: x[1]["r2_score"])
            
            self.logger.info(
                f"Training completed. Best model: {best_model[0]} "
                f"(R² = {best_model[1]['r2_score']:.4f})"
            )
            
            return {
                "status": "success",
                "property_type": property_type,
                "region": region,
                "training_samples": len(training_data),
                "model_results": model_results,
                "best_model": best_model[0],
                "best_r2_score": best_model[1]["r2_score"]
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    async def forecast_prices(
        self,
        property_type: str,
        region: str,
        horizon: ForecastHorizon = ForecastHorizon.MEDIUM_TERM,
        model_type: ModelType = ModelType.ENSEMBLE,
        include_market_factors: bool = True
    ) -> ForecastResult:
        """Generate price forecasts for specified criteria."""
        try:
            # Load appropriate model
            model_key = f"{property_type}_{region}_{model_type.value}"
            model = await self._load_model(model_key)
            
            if not model:
                # Train model if not available
                training_result = await self.train_models(property_type, region)
                if training_result["status"] != "success":
                    raise ValueError("Failed to train model for forecasting")
                model = await self._load_model(model_key)
            
            # Get current market data
            current_data = await self._get_current_market_data(property_type, region)
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(horizon)
            
            # Prepare features for forecasting
            forecast_features = await self._prepare_forecast_features(
                current_data, forecast_dates, property_type, region
            )
            
            # Apply market factors if requested
            if include_market_factors:
                market_factors = await self._get_market_factors(property_type, region)
                forecast_features = self._apply_market_factors(forecast_features, market_factors)
            
            # Generate predictions
            if model_type in [ModelType.ARIMA, ModelType.EXPONENTIAL_SMOOTHING]:
                predictions, confidence_intervals = await self._forecast_time_series(
                    model, len(forecast_dates)
                )
            else:
                predictions, confidence_intervals = await self._forecast_ml_model(
                    model, forecast_features
                )
            
            # Calculate additional insights
            trend_direction = self._analyze_trend_direction(predictions)
            volatility_score = self._calculate_volatility(predictions)
            market_momentum = self._calculate_market_momentum(current_data, predictions)
            
            # Get model performance metrics
            performance = self.model_performance.get(model_key, {})
            
            result = ForecastResult(
                property_type=property_type,
                region=region,
                forecast_horizon=horizon.value,
                model_type=model_type.value,
                predicted_prices=predictions,
                confidence_intervals=confidence_intervals,
                forecast_dates=forecast_dates,
                model_accuracy=performance.get("accuracy", 0.0),
                mae=performance.get("mae", 0.0),
                rmse=performance.get("rmse", 0.0),
                r2_score=performance.get("r2_score", 0.0),
                trend_direction=trend_direction,
                volatility_score=volatility_score,
                market_momentum=market_momentum,
                features_used=self.feature_columns,
                data_points_count=len(current_data),
                generated_at=datetime.utcnow()
            )
            
            # Cache the result
            await self._cache_forecast_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Price forecasting failed: {e}")
            raise
    
    async def get_price_insights(
        self,
        property_type: str,
        region: str,
        current_price: float
    ) -> Dict[str, Any]:
        """Get insights about a property's current price."""
        try:
            # Get market data
            market_data = await self._get_current_market_data(property_type, region)
            
            if market_data.empty:
                return {"error": "No market data available"}
            
            # Calculate market statistics
            market_stats = {
                "avg_price": market_data["price"].mean(),
                "median_price": market_data["price"].median(),
                "min_price": market_data["price"].min(),
                "max_price": market_data["price"].max(),
                "std_price": market_data["price"].std()
            }
            
            # Calculate price position
            percentile = (market_data["price"] <= current_price).mean() * 100
            
            # Determine price category
            if percentile <= 25:
                price_category = "below_market"
            elif percentile <= 75:
                price_category = "market_rate"
            else:
                price_category = "above_market"
            
            # Calculate recommended price range
            recommended_min = market_stats["median_price"] * 0.95
            recommended_max = market_stats["median_price"] * 1.05
            
            # Get recent price trends
            trend_data = await self._get_price_trend_data(property_type, region)
            recent_trend = self._calculate_recent_trend(trend_data)
            
            return {
                "current_price": current_price,
                "market_position": {
                    "percentile": percentile,
                    "category": price_category,
                    "vs_average": ((current_price - market_stats["avg_price"]) / market_stats["avg_price"]) * 100,
                    "vs_median": ((current_price - market_stats["median_price"]) / market_stats["median_price"]) * 100
                },
                "market_statistics": market_stats,
                "recommended_range": {
                    "min": recommended_min,
                    "max": recommended_max,
                    "optimal": market_stats["median_price"]
                },
                "price_trends": {
                    "recent_trend": recent_trend,
                    "trend_direction": "increasing" if recent_trend > 0 else "decreasing" if recent_trend < 0 else "stable"
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get price insights: {e}")
            return {"error": str(e)}
    
    async def compare_markets(
        self,
        property_type: str,
        regions: List[str],
        horizon: ForecastHorizon = ForecastHorizon.MEDIUM_TERM
    ) -> Dict[str, Any]:
        """Compare price forecasts across multiple markets."""
        try:
            comparisons = {}
            
            for region in regions:
                try:
                    forecast = await self.forecast_prices(
                        property_type, region, horizon, ModelType.ENSEMBLE
                    )
                    
                    # Calculate summary metrics
                    avg_predicted_price = np.mean(forecast.predicted_prices)
                    price_change = ((forecast.predicted_prices[-1] - forecast.predicted_prices[0]) / 
                                  forecast.predicted_prices[0]) * 100
                    
                    comparisons[region] = {
                        "avg_predicted_price": avg_predicted_price,
                        "price_change_percent": price_change,
                        "trend_direction": forecast.trend_direction,
                        "volatility_score": forecast.volatility_score,
                        "market_momentum": forecast.market_momentum,
                        "model_accuracy": forecast.model_accuracy
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to forecast for region {region}: {e}")
                    comparisons[region] = {"error": str(e)}
            
            # Rank markets by attractiveness
            ranked_markets = self._rank_markets_by_attractiveness(comparisons)
            
            return {
                "property_type": property_type,
                "forecast_horizon": horizon.value,
                "market_comparisons": comparisons,
                "ranked_markets": ranked_markets,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Market comparison failed: {e}")
            return {"error": str(e)}
    
    async def get_forecaster_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all forecasting models."""
        try:
            # Aggregate performance across all models
            total_models = len(self.model_performance)
            avg_r2 = np.mean([perf["r2_score"] for perf in self.model_performance.values()])
            avg_mae = np.mean([perf["mae"] for perf in self.model_performance.values()])
            
            # Get best performing models
            best_models = sorted(
                self.model_performance.items(),
                key=lambda x: x[1]["r2_score"],
                reverse=True
            )[:5]
            
            # Calculate forecast accuracy over time
            forecast_accuracy = await self._calculate_forecast_accuracy()
            
            return {
                "summary": {
                    "total_models": total_models,
                    "average_r2_score": avg_r2,
                    "average_mae": avg_mae,
                    "forecast_accuracy": forecast_accuracy
                },
                "best_performing_models": [
                    {"model": model, "r2_score": perf["r2_score"], "mae": perf["mae"]}
                    for model, perf in best_models
                ],
                "model_details": self.model_performance,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get forecaster performance: {e}")
            return {"error": str(e)}
    
    # Private methods for model training
    async def _load_training_data(
        self,
        property_type: Optional[str] = None,
        region: Optional[str] = None
    ) -> pd.DataFrame:
        """Load historical data for model training."""
        where_conditions = ["scraped_at >= NOW() - INTERVAL '2 years'"]
        params = {}
        
        if property_type:
            where_conditions.append("property_type = :property_type")
            params["property_type"] = property_type
            
        if region:
            where_conditions.append("location ILIKE :region")
            params["region"] = f"%{region}%"
        
        where_clause = " AND ".join(where_conditions)
        
        query = text(f"""
            SELECT 
                price,
                bedrooms,
                bathrooms,
                square_feet,
                property_type,
                location,
                amenities,
                scraped_at,
                array_length(amenities, 1) as amenity_count,
                price / NULLIF(square_feet, 0) as price_per_sqft
            FROM properties 
            WHERE {where_clause}
            AND price > 0 
            AND price < 50000  -- Reasonable upper limit
            ORDER BY scraped_at
        """)
        
        result = await self.db_session.execute(query, params)
        data = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        return data
    
    async def _prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        # Feature engineering
        data = data.copy()
        
        # Handle missing values
        data["square_feet"] = data["square_feet"].fillna(data["square_feet"].median())
        data["amenity_count"] = data["amenity_count"].fillna(0)
        data["price_per_sqft"] = data["price_per_sqft"].fillna(
            data["price"] / data["square_feet"]
        )
        
        # Encode categorical variables
        if "property_type" not in self.encoders:
            self.encoders["property_type"] = LabelEncoder()
            data["property_type_encoded"] = self.encoders["property_type"].fit_transform(
                data["property_type"].fillna("unknown")
            )
        else:
            data["property_type_encoded"] = self.encoders["property_type"].transform(
                data["property_type"].fillna("unknown")
            )
        
        if "location" not in self.encoders:
            self.encoders["location"] = LabelEncoder()
            data["location_encoded"] = self.encoders["location"].fit_transform(
                data["location"].fillna("unknown")
            )
        else:
            data["location_encoded"] = self.encoders["location"].transform(
                data["location"].fillna("unknown")
            )
        
        # Time features
        data["scraped_at"] = pd.to_datetime(data["scraped_at"])
        data["year"] = data["scraped_at"].dt.year
        data["month"] = data["scraped_at"].dt.month
        data["season"] = data["month"].apply(self._get_season)
        
        # Calculate days on market (simplified)
        data["days_on_market"] = (datetime.utcnow() - data["scraped_at"]).dt.days
        
        # Market features (simplified - would be enhanced with real market data)
        data["local_avg_price"] = data.groupby("location_encoded")["price"].transform("mean")
        data["supply_demand_ratio"] = 1.0  # Placeholder
        data["market_trend"] = 0.0  # Placeholder
        
        # Select features
        feature_cols = [col for col in self.feature_columns if col in data.columns]
        X = data[feature_cols]
        y = data["price"]
        
        # Scale features
        scaler_key = "main_scaler"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scalers[scaler_key].fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scalers[scaler_key].transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled, y
    
    def _get_season(self, month: int) -> int:
        """Convert month to season."""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    async def _train_linear_regression(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Train linear regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Store model
        model_key = "linear_regression"
        self.models[model_key] = model
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "accuracy": max(0, r2)  # Use R² as accuracy measure
        }
    
    async def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Train random forest model."""
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Store model
        model_key = "random_forest"
        self.models[model_key] = model
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "accuracy": max(0, r2),
            "feature_importance": dict(zip(X_train.columns, model.feature_importances_))
        }
    
    async def _train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Train gradient boosting model."""
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Store model
        model_key = "gradient_boosting"
        self.models[model_key] = model
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "accuracy": max(0, r2),
            "feature_importance": dict(zip(X_train.columns, model.feature_importances_))
        }
    
    async def _train_arima(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Train ARIMA time series model."""
        try:
            # Prepare time series data
            ts_data = ts_data.set_index("scraped_at")["price"].resample("D").mean().dropna()
            
            if len(ts_data) < 30:
                return {"error": "Insufficient data for ARIMA model"}
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Store model
            model_key = "arima"
            self.models[model_key] = fitted_model
            
            # Calculate in-sample metrics
            forecast = fitted_model.forecast(steps=len(ts_data)//4)  # Forecast 25% of data length
            
            return {
                "model": fitted_model,
                "aic": fitted_model.aic,
                "bic": fitted_model.bic,
                "mae": 0.0,  # Would calculate with out-of-sample validation
                "rmse": 0.0,
                "r2_score": 0.0,
                "accuracy": 0.7  # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            return {"error": str(e)}
    
    async def _train_exponential_smoothing(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Exponential Smoothing time series model."""
        try:
            # Prepare time series data
            ts_data = ts_data.set_index("scraped_at")["price"].resample("D").mean().dropna()
            
            if len(ts_data) < 30:
                return {"error": "Insufficient data for Exponential Smoothing model"}
            
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                ts_data,
                trend="add",
                seasonal="add",
                seasonal_periods=7  # Weekly seasonality
            )
            fitted_model = model.fit()
            
            # Store model
            model_key = "exponential_smoothing"
            self.models[model_key] = fitted_model
            
            return {
                "model": fitted_model,
                "sse": fitted_model.sse,
                "mae": 0.0,  # Would calculate with out-of-sample validation
                "rmse": 0.0,
                "r2_score": 0.0,
                "accuracy": 0.75  # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Exponential Smoothing training failed: {e}")
            return {"error": str(e)}
    
    async def _create_ensemble_model(
        self,
        model_results: Dict[str, Dict[str, Any]],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Create ensemble model from individual models."""
        try:
            # Collect predictions from all successful models
            predictions = []
            weights = []
            
            for model_type, results in model_results.items():
                if "model" in results and "r2_score" in results:
                    model = results["model"]
                    
                    # Get predictions based on model type
                    if model_type in ["arima", "exponential_smoothing"]:
                        # For time series models, we'd need different handling
                        continue
                    else:
                        pred = model.predict(X_test)
                        predictions.append(pred)
                        weights.append(max(0, results["r2_score"]))  # Use R² as weight
            
            if not predictions:
                return {"error": "No valid models for ensemble"}
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Create weighted ensemble prediction
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            # Calculate ensemble metrics
            mae = mean_absolute_error(y_test, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            r2 = r2_score(y_test, ensemble_pred)
            
            # Store ensemble configuration
            ensemble_config = {
                "models": list(model_results.keys()),
                "weights": weights.tolist(),
                "predictions": predictions
            }
            
            self.models["ensemble"] = ensemble_config
            
            return {
                "model": ensemble_config,
                "mae": mae,
                "rmse": rmse,
                "r2_score": r2,
                "accuracy": max(0, r2),
                "component_models": len(predictions),
                "weights": weights.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            return {"error": str(e)}
    
    # Additional helper methods would be implemented here
    def _generate_forecast_dates(self, horizon: ForecastHorizon) -> List[datetime]:
        """Generate forecast dates based on horizon."""
        current_date = datetime.utcnow()
        dates = []
        
        if horizon == ForecastHorizon.SHORT_TERM:
            days = 30
        elif horizon == ForecastHorizon.MEDIUM_TERM:
            days = 90
        else:  # LONG_TERM
            days = 365
        
        for i in range(1, days + 1):
            dates.append(current_date + timedelta(days=i))
        
        return dates
    
    def _analyze_trend_direction(self, predictions: List[float]) -> str:
        """Analyze trend direction from predictions."""
        if len(predictions) < 2:
            return "stable"
        
        start_price = predictions[0]
        end_price = predictions[-1]
        change_percent = ((end_price - start_price) / start_price) * 100
        
        if change_percent > 2:
            return "increasing"
        elif change_percent < -2:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_volatility(self, predictions: List[float]) -> float:
        """Calculate volatility score from predictions."""
        if len(predictions) < 2:
            return 0.0
        
        changes = [abs(predictions[i] - predictions[i-1]) for i in range(1, len(predictions))]
        avg_change = np.mean(changes)
        avg_price = np.mean(predictions)
        
        return (avg_change / avg_price) * 100 if avg_price > 0 else 0.0
    
    def _calculate_market_momentum(
        self,
        current_data: pd.DataFrame,
        predictions: List[float]
    ) -> float:
        """Calculate market momentum indicator."""
        if current_data.empty or not predictions:
            return 0.0
        
        current_avg = current_data["price"].mean()
        predicted_avg = np.mean(predictions)
        
        return ((predicted_avg - current_avg) / current_avg) * 100 if current_avg > 0 else 0.0
    
    # Placeholder implementations for additional methods
    async def _load_cached_models(self) -> None:
        """Load cached models from Redis."""
        pass
    
    async def _initialize_feature_processors(self) -> None:
        """Initialize feature processors."""
        pass
    
    async def _load_model(self, model_key: str) -> Optional[Any]:
        """Load specific model."""
        return self.models.get(model_key)
    
    async def _get_current_market_data(self, property_type: str, region: str) -> pd.DataFrame:
        """Get current market data."""
        return pd.DataFrame()
    
    async def _prepare_forecast_features(
        self,
        current_data: pd.DataFrame,
        forecast_dates: List[datetime],
        property_type: str,
        region: str
    ) -> pd.DataFrame:
        """Prepare features for forecasting."""
        return pd.DataFrame()
    
    async def _get_market_factors(self, property_type: str, region: str) -> MarketFactors:
        """Get market factors."""
        return MarketFactors(
            seasonality_factor=1.0,
            supply_demand_ratio=1.0,
            economic_indicators={},
            local_market_conditions={},
            competitor_pricing={}
        )
    
    def _apply_market_factors(
        self,
        features: pd.DataFrame,
        factors: MarketFactors
    ) -> pd.DataFrame:
        """Apply market factors to features."""
        return features
    
    async def _forecast_time_series(
        self,
        model: Any,
        steps: int
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate time series forecasts."""
        return [], []
    
    async def _forecast_ml_model(
        self,
        model: Any,
        features: pd.DataFrame
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate ML model forecasts."""
        return [], []
    
    async def _cache_forecast_result(self, result: ForecastResult) -> None:
        """Cache forecast result."""
        pass
    
    async def _get_price_trend_data(self, property_type: str, region: str) -> pd.DataFrame:
        """Get price trend data."""
        return pd.DataFrame()
    
    def _calculate_recent_trend(self, trend_data: pd.DataFrame) -> float:
        """Calculate recent price trend."""
        return 0.0
    
    def _rank_markets_by_attractiveness(
        self,
        comparisons: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank markets by investment attractiveness."""
        return []
    
    async def _calculate_forecast_accuracy(self) -> float:
        """Calculate historical forecast accuracy."""
        return 0.85  # Placeholder
    
    async def _save_models(
        self,
        property_type: Optional[str],
        region: Optional[str],
        model_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """Save trained models."""
        for model_type, results in model_results.items():
            if "model" in results:
                model_key = f"{property_type}_{region}_{model_type}"
                self.model_performance[model_key] = {
                    "mae": results.get("mae", 0.0),
                    "rmse": results.get("rmse", 0.0),
                    "r2_score": results.get("r2_score", 0.0),
                    "accuracy": results.get("accuracy", 0.0)
                }