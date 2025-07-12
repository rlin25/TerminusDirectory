"""
Production scraping configuration and settings management.

This module provides comprehensive configuration management for production-ready
property scraping operations with support for multiple environments and sources.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: float = 0.5
    requests_per_minute: int = 30
    requests_per_hour: int = 1800
    burst_size: int = 5
    backoff_factor: float = 2.0
    max_backoff_seconds: int = 300
    retry_after_respect: bool = True


@dataclass
class RobotsTxtConfig:
    """Robots.txt compliance configuration"""
    respect_robots_txt: bool = True
    cache_robots_txt: bool = True
    robots_txt_timeout: int = 10
    robots_txt_retry_attempts: int = 3
    default_crawl_delay: float = 1.0
    user_agent_override: Optional[str] = None


@dataclass
class DataQualityConfig:
    """Data quality and validation configuration"""
    min_required_fields: int = 3
    max_price_threshold: float = 50000.0
    min_price_threshold: float = 100.0
    max_bedrooms: int = 10
    max_bathrooms: float = 10.0
    min_title_length: int = 10
    max_title_length: int = 500
    min_description_length: int = 20
    max_description_length: int = 10000
    required_fields: List[str] = field(default_factory=lambda: ['title', 'price', 'location'])
    validate_coordinates: bool = True
    validate_phone_numbers: bool = True
    validate_email_addresses: bool = True


@dataclass
class ScrapingConfig:
    """Core scraping configuration"""
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay_base: float = 1.0
    max_pages_per_source: int = 100
    max_properties_per_session: int = 10000
    session_timeout_hours: int = 6
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ])
    rotate_user_agents: bool = True
    use_proxy_rotation: bool = False
    proxy_list: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_metrics: bool = True
    enable_alerts: bool = True
    alert_on_error_rate: float = 0.1  # 10% error rate
    alert_on_slow_response: float = 30.0  # 30 seconds
    metrics_retention_days: int = 30
    log_level: LogLevel = LogLevel.INFO
    structured_logging: bool = True
    log_to_file: bool = True
    log_file_path: str = "/var/log/rental-ml/scraping.log"
    log_rotation_size: str = "100MB"
    log_retention_days: int = 30


@dataclass
class CacheConfig:
    """Caching configuration"""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_max_size: int = 10000
    cache_key_prefix: str = "scraping"
    redis_url: Optional[str] = None
    redis_cluster_mode: bool = False


@dataclass
class DatabaseConfig:
    """Database configuration for scraped data"""
    batch_size: int = 100
    connection_pool_size: int = 20
    connection_timeout: int = 30
    enable_transactions: bool = True
    auto_commit: bool = False
    duplicate_check_enabled: bool = True
    data_retention_days: int = 365


@dataclass
class GeocodingConfig:
    """Geocoding service configuration"""
    enable_geocoding: bool = True
    provider: str = "nominatim"  # nominatim, google, mapbox
    api_key: Optional[str] = None
    cache_results: bool = True
    timeout_seconds: int = 5
    max_retries: int = 2
    rate_limit_per_second: float = 1.0


@dataclass
class ScraperSourceConfig:
    """Configuration for a specific scraper source"""
    name: str
    enabled: bool = True
    base_url: str = ""
    search_locations: List[str] = field(default_factory=list)
    max_pages: int = 50
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    robots_txt: RobotsTxtConfig = field(default_factory=RobotsTxtConfig)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    selectors: Dict[str, str] = field(default_factory=dict)
    priority: int = 1  # Higher number = higher priority


class ProductionScrapingConfig:
    """Production scraping configuration manager"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from environment variables and defaults"""
        
        # Core scraping configuration
        self.scraping = ScrapingConfig(
            max_concurrent_requests=int(os.getenv('SCRAPING_MAX_CONCURRENT', '5')),
            request_timeout=int(os.getenv('SCRAPING_TIMEOUT', '30')),
            max_retries=int(os.getenv('SCRAPING_MAX_RETRIES', '3')),
            max_pages_per_source=int(os.getenv('SCRAPING_MAX_PAGES', '100')),
            max_properties_per_session=int(os.getenv('SCRAPING_MAX_PROPERTIES', '10000')),
            use_proxy_rotation=os.getenv('SCRAPING_USE_PROXIES', 'false').lower() == 'true'
        )
        
        # Rate limiting
        self.rate_limit = RateLimitConfig(
            requests_per_second=float(os.getenv('RATE_LIMIT_PER_SECOND', '0.5')),
            requests_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '30')),
            requests_per_hour=int(os.getenv('RATE_LIMIT_PER_HOUR', '1800')),
            backoff_factor=float(os.getenv('RATE_LIMIT_BACKOFF', '2.0'))
        )
        
        # Data quality
        self.data_quality = DataQualityConfig(
            min_required_fields=int(os.getenv('DATA_QUALITY_MIN_FIELDS', '3')),
            max_price_threshold=float(os.getenv('DATA_QUALITY_MAX_PRICE', '50000')),
            min_price_threshold=float(os.getenv('DATA_QUALITY_MIN_PRICE', '100'))
        )
        
        # Monitoring
        self.monitoring = MonitoringConfig(
            enable_metrics=os.getenv('MONITORING_ENABLE_METRICS', 'true').lower() == 'true',
            enable_alerts=os.getenv('MONITORING_ENABLE_ALERTS', 'true').lower() == 'true',
            log_level=LogLevel(os.getenv('LOG_LEVEL', 'INFO')),
            log_file_path=os.getenv('LOG_FILE_PATH', '/var/log/rental-ml/scraping.log')
        )
        
        # Caching
        self.cache = CacheConfig(
            enable_caching=os.getenv('CACHE_ENABLE', 'true').lower() == 'true',
            cache_ttl_seconds=int(os.getenv('CACHE_TTL', '3600')),
            redis_url=os.getenv('REDIS_URL')
        )
        
        # Database
        self.database = DatabaseConfig(
            batch_size=int(os.getenv('DB_BATCH_SIZE', '100')),
            connection_pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
            duplicate_check_enabled=os.getenv('DB_DUPLICATE_CHECK', 'true').lower() == 'true'
        )
        
        # Geocoding
        self.geocoding = GeocodingConfig(
            enable_geocoding=os.getenv('GEOCODING_ENABLE', 'true').lower() == 'true',
            provider=os.getenv('GEOCODING_PROVIDER', 'nominatim'),
            api_key=os.getenv('GEOCODING_API_KEY')
        )
        
        # Source configurations
        self.sources = self._load_source_configurations()
        
        # Environment-specific adjustments
        self._apply_environment_overrides()
    
    def _load_source_configurations(self) -> Dict[str, ScraperSourceConfig]:
        """Load configurations for all scraper sources"""
        sources = {}
        
        # Apartments.com configuration
        sources['apartments_com'] = ScraperSourceConfig(
            name='apartments_com',
            enabled=os.getenv('APARTMENTS_COM_ENABLED', 'true').lower() == 'true',
            base_url='https://www.apartments.com',
            search_locations=[
                'new-york-ny', 'los-angeles-ca', 'chicago-il', 'houston-tx',
                'phoenix-az', 'philadelphia-pa', 'san-antonio-tx', 'san-diego-ca',
                'dallas-tx', 'san-jose-ca', 'austin-tx', 'jacksonville-fl',
                'fort-worth-tx', 'columbus-oh', 'charlotte-nc', 'san-francisco-ca',
                'indianapolis-in', 'seattle-wa', 'denver-co', 'washington-dc'
            ],
            max_pages=int(os.getenv('APARTMENTS_COM_MAX_PAGES', '50')),
            rate_limit=RateLimitConfig(
                requests_per_second=0.3,
                requests_per_minute=18,
                requests_per_hour=1000
            ),
            priority=1
        )
        
        # Rentals.com configuration
        sources['rentals_com'] = ScraperSourceConfig(
            name='rentals_com',
            enabled=os.getenv('RENTALS_COM_ENABLED', 'false').lower() == 'true',
            base_url='https://www.rentals.com',
            search_locations=[
                'new-york-ny', 'los-angeles-ca', 'chicago-il', 'houston-tx',
                'phoenix-az', 'philadelphia-pa', 'san-diego-ca', 'dallas-tx'
            ],
            max_pages=int(os.getenv('RENTALS_COM_MAX_PAGES', '30')),
            rate_limit=RateLimitConfig(
                requests_per_second=0.4,
                requests_per_minute=24,
                requests_per_hour=1200
            ),
            priority=2
        )
        
        # Zillow configuration (more restrictive)
        sources['zillow'] = ScraperSourceConfig(
            name='zillow',
            enabled=os.getenv('ZILLOW_ENABLED', 'false').lower() == 'true',
            base_url='https://www.zillow.com',
            search_locations=[
                'new-york-ny', 'los-angeles-ca', 'chicago-il', 'houston-tx'
            ],
            max_pages=int(os.getenv('ZILLOW_MAX_PAGES', '20')),
            rate_limit=RateLimitConfig(
                requests_per_second=0.2,
                requests_per_minute=12,
                requests_per_hour=600
            ),
            priority=3
        )
        
        # Rent.com configuration
        sources['rent_com'] = ScraperSourceConfig(
            name='rent_com',
            enabled=os.getenv('RENT_COM_ENABLED', 'false').lower() == 'true',
            base_url='https://www.rent.com',
            search_locations=[
                'new-york-ny', 'los-angeles-ca', 'chicago-il', 'houston-tx',
                'phoenix-az', 'philadelphia-pa'
            ],
            max_pages=int(os.getenv('RENT_COM_MAX_PAGES', '40')),
            rate_limit=RateLimitConfig(
                requests_per_second=0.5,
                requests_per_minute=30,
                requests_per_hour=1500
            ),
            priority=2
        )
        
        return sources
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.PRODUCTION:
            # Production: More conservative settings
            self.scraping.max_concurrent_requests = min(self.scraping.max_concurrent_requests, 3)
            self.rate_limit.requests_per_second = min(self.rate_limit.requests_per_second, 0.3)
            self.monitoring.enable_alerts = True
            self.monitoring.log_level = LogLevel.INFO
            
        elif self.environment == Environment.STAGING:
            # Staging: Balanced settings
            self.scraping.max_concurrent_requests = min(self.scraping.max_concurrent_requests, 5)
            self.rate_limit.requests_per_second = min(self.rate_limit.requests_per_second, 0.5)
            self.monitoring.log_level = LogLevel.DEBUG
            
        elif self.environment == Environment.DEVELOPMENT:
            # Development: More permissive settings but still respectful
            self.scraping.max_concurrent_requests = min(self.scraping.max_concurrent_requests, 2)
            self.rate_limit.requests_per_second = min(self.rate_limit.requests_per_second, 0.2)
            self.monitoring.log_level = LogLevel.DEBUG
            
            # Limit pages in development
            for source in self.sources.values():
                source.max_pages = min(source.max_pages, 5)
    
    def get_source_config(self, source_name: str) -> Optional[ScraperSourceConfig]:
        """Get configuration for a specific source"""
        return self.sources.get(source_name)
    
    def get_enabled_sources(self) -> List[ScraperSourceConfig]:
        """Get list of enabled scraper sources"""
        return [config for config in self.sources.values() if config.enabled]
    
    def get_sources_by_priority(self) -> List[ScraperSourceConfig]:
        """Get enabled sources sorted by priority"""
        enabled_sources = self.get_enabled_sources()
        return sorted(enabled_sources, key=lambda x: x.priority)
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate rate limits
        if self.rate_limit.requests_per_second <= 0:
            issues.append("Rate limit requests_per_second must be positive")
        
        if self.rate_limit.requests_per_minute <= 0:
            issues.append("Rate limit requests_per_minute must be positive")
        
        # Validate data quality thresholds
        if self.data_quality.min_price_threshold <= 0:
            issues.append("Minimum price threshold must be positive")
        
        if self.data_quality.max_price_threshold <= self.data_quality.min_price_threshold:
            issues.append("Maximum price threshold must be greater than minimum")
        
        # Validate database configuration
        if self.database.batch_size <= 0:
            issues.append("Database batch size must be positive")
        
        if self.database.connection_pool_size <= 0:
            issues.append("Database connection pool size must be positive")
        
        # Validate source configurations
        enabled_sources = self.get_enabled_sources()
        if not enabled_sources:
            issues.append("At least one scraper source must be enabled")
        
        for source in enabled_sources:
            if not source.base_url:
                issues.append(f"Source {source.name} missing base_url")
            
            if not source.search_locations:
                issues.append(f"Source {source.name} missing search_locations")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment.value,
            'scraping': self.scraping.__dict__,
            'rate_limit': self.rate_limit.__dict__,
            'data_quality': self.data_quality.__dict__,
            'monitoring': {
                **self.monitoring.__dict__,
                'log_level': self.monitoring.log_level.value
            },
            'cache': self.cache.__dict__,
            'database': self.database.__dict__,
            'geocoding': self.geocoding.__dict__,
            'sources': {
                name: {
                    **config.__dict__,
                    'rate_limit': config.rate_limit.__dict__,
                    'robots_txt': config.robots_txt.__dict__
                }
                for name, config in self.sources.items()
            }
        }
    
    @classmethod
    def from_file(cls, config_file: Path, environment: Environment = Environment.DEVELOPMENT):
        """Load configuration from file"""
        # This would implement loading from YAML/JSON config file
        # For now, we'll use environment variables and defaults
        return cls(environment)


# Global configuration instance
_config = None

def get_config(environment: Environment = None) -> ProductionScrapingConfig:
    """Get global configuration instance"""
    global _config
    
    if _config is None or (environment and _config.environment != environment):
        if environment is None:
            env_name = os.getenv('ENVIRONMENT', 'development').lower()
            environment = Environment(env_name)
        
        _config = ProductionScrapingConfig(environment)
        
        # Validate configuration on load
        issues = _config.validate_configuration()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
    
    return _config


def reload_config(environment: Environment = None):
    """Reload configuration"""
    global _config
    _config = None
    return get_config(environment)