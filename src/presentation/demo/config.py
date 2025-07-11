"""
Configuration settings for Rental ML System Demo

This module contains configuration constants and settings for the demo application:
- UI configuration
- Demo data settings
- API endpoints
- Visualization parameters
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class UIConfig:
    """UI configuration settings"""
    page_title: str = "Rental ML System Demo"
    page_icon: str = "🏠"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Color scheme
    primary_color: str = "#667eea"
    secondary_color: str = "#764ba2"
    background_color: str = "#ffffff"
    text_color: str = "#262730"
    
    # Chart settings
    chart_height: int = 400
    map_height: int = 600
    table_page_size: int = 20


@dataclass
class DemoDataConfig:
    """Demo data generation settings"""
    default_property_count: int = 100
    default_user_count: int = 50
    default_interaction_count: int = 500
    
    # Property generation parameters
    price_range: tuple = (800, 8000)
    bedroom_range: tuple = (0, 5)
    bathroom_range: tuple = (1.0, 4.0)
    sqft_range: tuple = (300, 3000)
    
    # Random seed for consistent demo data
    random_seed: int = 42
    
    # Data file paths
    sample_data_file: str = "demo_sample_data.json"
    export_directory: str = "exports"


@dataclass
class MLConfig:
    """ML model configuration for demo"""
    # Model types
    available_models: List[str] = None
    default_model: str = "hybrid"
    
    # Recommendation parameters
    default_num_recommendations: int = 10
    max_recommendations: int = 50
    min_confidence_threshold: float = 0.1
    
    # Model performance simulation
    simulate_performance: bool = True
    base_accuracy: Dict[str, float] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = ["collaborative_filtering", "content_based", "hybrid"]
        
        if self.base_accuracy is None:
            self.base_accuracy = {
                "collaborative_filtering": 0.85,
                "content_based": 0.80,
                "hybrid": 0.90
            }


@dataclass
class APIConfig:
    """API configuration settings"""
    # Base URLs (for demo purposes, these would be real endpoints)
    base_url: str = "http://localhost:8000"
    api_version: str = "v1"
    
    # Endpoints
    properties_endpoint: str = "/api/v1/properties"
    users_endpoint: str = "/api/v1/users"
    recommendations_endpoint: str = "/api/v1/recommendations"
    search_endpoint: str = "/api/v1/search"
    analytics_endpoint: str = "/api/v1/analytics"
    health_endpoint: str = "/api/v1/health"
    
    # Request settings
    timeout: int = 30
    max_retries: int = 3


class LocationData:
    """Location data and coordinates for demo"""
    
    # Default coordinates (New York City)
    DEFAULT_CENTER = {"lat": 40.7128, "lon": -74.0060}
    
    # Location coordinate offsets for demo
    LOCATION_COORDINATES = {
        "Downtown": {"lat": 40.7128, "lon": -74.0060, "offset": (0.002, -0.001)},
        "Midtown": {"lat": 40.7589, "lon": -73.9851, "offset": (0.008, 0.003)},
        "Uptown": {"lat": 40.7831, "lon": -73.9712, "offset": (0.015, 0.002)},
        "Financial District": {"lat": 40.7074, "lon": -74.0113, "offset": (-0.003, -0.008)},
        "Arts District": {"lat": 40.7505, "lon": -73.9934, "offset": (0.005, -0.012)},
        "Suburban Heights": {"lat": 40.7282, "lon": -73.7949, "offset": (0.020, 0.015)},
        "Riverside": {"lat": 40.8176, "lon": -73.7781, "offset": (-0.010, 0.008)},
        "Hillside": {"lat": 40.7282, "lon": -73.6776, "offset": (0.012, -0.005)},
        "University Area": {"lat": 40.8075, "lon": -73.9626, "offset": (0.007, 0.010)},
        "Old Town": {"lat": 40.7505, "lon": -73.9780, "offset": (-0.005, -0.003)},
        "Tech Quarter": {"lat": 40.7831, "lon": -73.9440, "offset": (0.010, -0.015)},
        "Green Valley": {"lat": 40.6892, "lon": -73.9442, "offset": (0.025, 0.008)},
        "Sunset Park": {"lat": 40.6562, "lon": -74.0105, "offset": (-0.015, 0.012)},
        "Brookside": {"lat": 40.6892, "lon": -73.9442, "offset": (0.018, -0.010)},
        "Central Plaza": {"lat": 40.7505, "lon": -73.9934, "offset": (0.001, 0.001)},
        "Marina District": {"lat": 40.7282, "lon": -74.0776, "offset": (-0.008, 0.005)}
    }


class VisualizationConfig:
    """Configuration for charts and visualizations"""
    
    # Color palettes
    PROPERTY_TYPE_COLORS = {
        "apartment": "#1f77b4",
        "house": "#ff7f0e", 
        "condo": "#2ca02c",
        "studio": "#d62728",
        "townhouse": "#9467bd"
    }
    
    LOCATION_COLORS = {
        "Downtown": "#e41a1c",
        "Midtown": "#377eb8",
        "Uptown": "#4daf4a",
        "Financial District": "#984ea3",
        "Arts District": "#ff7f00",
        "Suburban Heights": "#ffff33",
        "Riverside": "#a65628",
        "Hillside": "#f781bf"
    }
    
    # Chart templates
    CHART_TEMPLATE = "plotly_white"
    
    # Default chart parameters
    DEFAULT_CHART_CONFIG = {
        "displayModeBar": False,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "pan2d", "lasso2d", "select2d", "autoScale2d",
            "hoverClosestCartesian", "hoverCompareCartesian"
        ]
    }


class DatabaseConfig:
    """Database configuration for demo (simulated)"""
    
    # Simulated database settings
    SIMULATED_DB = True
    
    # If using real database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/rental_ml_demo")
    DATABASE_POOL_SIZE = 10
    DATABASE_TIMEOUT = 30
    
    # Redis cache settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL = 3600  # 1 hour


class PerformanceConfig:
    """Performance and monitoring configuration"""
    
    # Simulated metrics
    SIMULATE_REAL_TIME_METRICS = True
    METRIC_UPDATE_INTERVAL = 60  # seconds
    
    # Performance thresholds
    RESPONSE_TIME_THRESHOLD = 100  # ms
    CPU_USAGE_THRESHOLD = 80  # percent
    MEMORY_USAGE_THRESHOLD = 85  # percent
    
    # System health status
    HEALTH_CHECK_INTERVAL = 30  # seconds


class SecurityConfig:
    """Security configuration for demo"""
    
    # Demo authentication (simplified)
    DEMO_MODE = True
    REQUIRE_AUTH = False
    
    # API keys (demo values)
    DEMO_API_KEY = "demo_api_key_12345"
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # 1 hour


# Global configuration instance
class Config:
    """Main configuration class combining all settings"""
    
    def __init__(self):
        self.ui = UIConfig()
        self.demo_data = DemoDataConfig()
        self.ml = MLConfig()
        self.api = APIConfig()
        self.locations = LocationData()
        self.visualization = VisualizationConfig()
        self.database = DatabaseConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        
        # Environment-specific overrides
        self._load_environment_config()
    
    def _load_environment_config(self):
        """Load configuration from environment variables"""
        
        # Override with environment variables if present
        if os.getenv("DEMO_PROPERTY_COUNT"):
            self.demo_data.default_property_count = int(os.getenv("DEMO_PROPERTY_COUNT"))
        
        if os.getenv("DEMO_USER_COUNT"):
            self.demo_data.default_user_count = int(os.getenv("DEMO_USER_COUNT"))
        
        if os.getenv("API_BASE_URL"):
            self.api.base_url = os.getenv("API_BASE_URL")
        
        if os.getenv("RANDOM_SEED"):
            self.demo_data.random_seed = int(os.getenv("RANDOM_SEED"))
    
    def get_property_types(self) -> List[str]:
        """Get list of available property types"""
        return ["apartment", "house", "condo", "studio", "townhouse"]
    
    def get_locations(self) -> List[str]:
        """Get list of available locations"""
        return list(self.locations.LOCATION_COORDINATES.keys())
    
    def get_amenities(self) -> List[str]:
        """Get list of available amenities"""
        return [
            "parking", "gym", "pool", "laundry", "pet-friendly", "balcony",
            "air-conditioning", "heating", "dishwasher", "hardwood-floors",
            "in-unit-laundry", "walk-in-closet", "fireplace", "garden",
            "rooftop-access", "concierge", "security", "storage"
        ]
    
    def get_interaction_types(self) -> List[str]:
        """Get list of user interaction types"""
        return ["view", "like", "inquiry", "save"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "ui": self.ui.__dict__,
            "demo_data": self.demo_data.__dict__,
            "ml": self.ml.__dict__,
            "api": self.api.__dict__,
            "visualization": self.visualization.__dict__,
            "performance": self.performance.__dict__,
            "security": self.security.__dict__
        }


# Create global config instance
config = Config()


# Feature flags for demo
class FeatureFlags:
    """Feature flags for enabling/disabling demo features"""
    
    # Main features
    ENABLE_RECOMMENDATIONS = True
    ENABLE_ANALYTICS = True
    ENABLE_MONITORING = True
    ENABLE_COMPARISON = True
    ENABLE_MAP_VIEW = True
    ENABLE_EXPORT = True
    
    # Advanced features
    ENABLE_ML_EXPLANATIONS = True
    ENABLE_REAL_TIME_UPDATES = False
    ENABLE_USER_AUTHENTICATION = False
    ENABLE_API_INTEGRATION = False
    
    # Demo-specific features
    ENABLE_SAMPLE_DATA_GENERATION = True
    ENABLE_SIMULATED_METRICS = True
    ENABLE_TUTORIAL_MODE = True
    
    @classmethod
    def is_enabled(cls, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return getattr(cls, feature_name, False)


# Constants for demo
class Constants:
    """Demo application constants"""
    
    # Application metadata
    APP_NAME = "Rental ML System Demo"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Interactive demonstration of ML-powered rental property platform"
    
    # Demo limits
    MAX_PROPERTIES_DISPLAY = 100
    MAX_RECOMMENDATIONS = 20
    MAX_COMPARISON_PROPERTIES = 4
    MAX_EXPORT_RECORDS = 10000
    
    # UI constants
    SIDEBAR_WIDTH = 300
    MAIN_CONTENT_WIDTH = 800
    CHART_HEIGHT = 400
    MAP_HEIGHT = 600
    
    # Date formats
    DATE_FORMAT = "%Y-%m-%d"
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    DISPLAY_DATE_FORMAT = "%B %d, %Y"
    
    # File formats
    SUPPORTED_EXPORT_FORMATS = ["CSV", "JSON", "Excel"]
    MAX_FILE_SIZE_MB = 50
    
    # Performance thresholds
    SLOW_QUERY_THRESHOLD_MS = 500
    MEMORY_WARNING_THRESHOLD_MB = 1000
    
    # Demo messages
    WELCOME_MESSAGE = """
    Welcome to the Rental ML System Demo! 
    
    This interactive demonstration showcases the capabilities of our 
    machine learning-powered rental property platform.
    
    Features include:
    • Intelligent property search with ML-powered ranking
    • Personalized recommendations based on user preferences  
    • Advanced analytics and market insights
    • Real-time system monitoring and performance metrics
    • Interactive property comparison tools
    """
    
    TUTORIAL_STEPS = [
        "Start by exploring properties in the Search tab",
        "Configure your preferences in the User Preferences tab", 
        "View personalized recommendations in the Recommendations tab",
        "Analyze market trends in the Analytics Dashboard",
        "Monitor system performance in the ML Performance tab",
        "Compare properties side-by-side in the Comparison tab"
    ]