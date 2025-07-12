# Production Rental Property Scraping System

A comprehensive, production-ready scraping system for collecting rental property data from multiple sources with advanced monitoring, compliance, and data quality features.

## Overview

This system provides a complete solution for scraping rental property data at scale with:

- **Multi-source support**: Apartments.com, Rentals.com, Zillow, Rent.com, and custom sources
- **Production-grade reliability**: Circuit breakers, rate limiting, error handling, and retries
- **GDPR compliance**: Automatic PII detection, anonymization, and data retention policies
- **Real-time monitoring**: Comprehensive metrics, alerts, and health checks
- **Data quality assurance**: Validation, cleaning, normalization, and duplicate detection
- **Scalable architecture**: Async processing, database integration, and job orchestration

## Key Features

### 🏗️ Production Infrastructure
- **Rate Limiting**: Advanced token bucket algorithm with multiple limits (per second/minute/hour)
- **Circuit Breakers**: Automatic failure detection and recovery
- **Robots.txt Compliance**: Respects site crawling policies with caching
- **Error Handling**: Comprehensive retry mechanisms with exponential backoff
- **Session Management**: Connection pooling and proper cleanup

### 📊 Data Quality & Processing
- **PII Detection**: Automatic identification of personally identifiable information
- **Data Validation**: Multi-level validation with configurable rules
- **Duplicate Detection**: Advanced similarity algorithms for property matching
- **Data Normalization**: Text cleaning, location standardization, and price extraction
- **Geocoding Integration**: Location enrichment with latitude/longitude

### 📈 Monitoring & Alerting
- **Real-time Metrics**: Request rates, success rates, data quality scores
- **Health Checks**: Database, Redis, external services, and system resources
- **Alert Management**: Configurable thresholds with multiple notification channels
- **Performance Monitoring**: Response times, throughput, and resource usage

### 🔒 Compliance & Security
- **GDPR Compliance**: Data classification, retention policies, and audit trails
- **Data Anonymization**: Multiple anonymization methods (hashing, masking, pseudonymization)
- **Processing Records**: Complete audit trail of all data processing activities
- **Data Subject Rights**: Support for access, erasure, and portability requests

### 🚀 Orchestration & Scheduling
- **Job Management**: Queue-based job execution with priority handling
- **Session Management**: Batch processing across multiple sources
- **Automated Scheduling**: Daily/weekly scraping with configurable parameters
- **Real-time Ingestion**: Direct database integration with batch processing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Scraping System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   Orchestrator  │  │    Monitoring    │  │   Scheduler     │ │
│  │                 │  │                  │  │                 │ │
│  │ • Job Queue     │  │ • Metrics        │  │ • Daily Jobs    │ │
│  │ • Session Mgmt  │  │ • Health Checks  │  │ • Weekly Jobs   │ │
│  │ • Event System  │  │ • Alerts         │  │ • Custom Jobs   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   Scrapers      │  │  Data Quality    │  │   Database      │ │
│  │                 │  │                  │  │                 │ │
│  │ • Apartments    │  │ • Validation     │  │ • PostgreSQL    │ │
│  │ • Rentals       │  │ • Cleaning       │  │ • Redis Cache   │ │
│  │ • Zillow        │  │ • Deduplication  │  │ • Real-time     │ │
│  │ • Generic       │  │ • Geocoding      │  │   Ingestion     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐                     │
│  │ GDPR Compliance │  │ Generic Framework│                     │
│  │                 │  │                  │                     │
│  │ • PII Detection │  │ • Config-driven  │                     │
│  │ • Anonymization │  │ • Easy Addition  │                     │
│  │ • Audit Trails  │  │ • YAML Config    │                     │
│  │ • Retention     │  │ • Validation     │                     │
│  └─────────────────┘  └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements/base.txt

# Setup database
python migrations/run_migrations.py

# Configure environment
cp .env.example .env
# Edit .env with your database and API credentials
```

### 2. Basic Usage

```python
from src.infrastructure.scrapers.main_production_system import ProductionScrapingSystem
from src.infrastructure.scrapers.config import Environment

# Initialize system
system = ProductionScrapingSystem(Environment.DEVELOPMENT)
await system.initialize()

# Run manual scraping
results = await system.run_manual_scraping(
    sources=['apartments_com', 'rentals_com'],
    max_pages_per_source=5,
    test_mode=True
)

print(f"Collected {results['total_properties']} properties")
```

### 3. Command Line Interface

```bash
# Run full system with scheduler
python -m src.infrastructure.scrapers.main_production_system --environment production --command run

# Test all scrapers
python -m src.infrastructure.scrapers.main_production_system --command test

# Manual scraping
python -m src.infrastructure.scrapers.main_production_system --command scrape --sources apartments_com --max-pages 5

# System status
python -m src.infrastructure.scrapers.main_production_system --command status
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rental_ml
DB_USER=postgres
DB_PASSWORD=password

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Scraping Configuration
SCRAPING_MAX_CONCURRENT=5
SCRAPING_TIMEOUT=30
RATE_LIMIT_PER_SECOND=0.5
RATE_LIMIT_PER_MINUTE=30

# Monitoring Configuration
LOG_LEVEL=INFO
MONITORING_ENABLE_METRICS=true
MONITORING_ENABLE_ALERTS=true

# GDPR Configuration
DATA_QUALITY_MIN_FIELDS=3
GEOCODING_ENABLE=true
GEOCODING_PROVIDER=nominatim
```

### Source Configuration

Enable/disable specific sources in your configuration:

```python
# Enable apartments.com scraper
APARTMENTS_COM_ENABLED=true
APARTMENTS_COM_MAX_PAGES=50

# Enable rentals.com scraper
RENTALS_COM_ENABLED=true
RENTALS_COM_MAX_PAGES=30

# Disable Zillow (requires careful consideration)
ZILLOW_ENABLED=false
```

## Adding New Sources

### Method 1: Generic Framework (Recommended)

Create a YAML configuration file:

```yaml
# new_source.yaml
source_name: 'example_rentals'
base_url: 'https://example-rentals.com'
search_locations: ['new-york-ny', 'los-angeles-ca']
search_url_pattern: '{base_url}/rentals/{location}'
pagination_pattern: '{search_url}?page={page}'
max_pages: 20

listing_links:
  field: 'listing_urls'
  selectors: ['.property-card a', '.listing a[href*="/property/"]']
  type: 'attribute'
  attribute: 'href'
  multiple: true

property_extraction:
  title:
    field: 'title'
    selectors: ['h1.property-title', '.listing-title']
    type: 'text'
    required: true
  
  price:
    field: 'price'
    selectors: ['.price', '.rent-amount']
    type: 'regex'
    regex: '\$[\d,]+'
    required: true
  
  # ... additional fields
```

Load and use:

```python
from src.infrastructure.scrapers.generic_scraping_framework import GenericScrapingFramework

framework = GenericScrapingFramework()
framework.load_config_from_yaml('example_rentals', 'new_source.yaml')
scraper = framework.create_scraper('example_rentals')
```

### Method 2: Custom Scraper Class

```python
from src.infrastructure.scrapers.production_base_scraper import ProductionBaseScraper

class NewSourceScraper(ProductionBaseScraper):
    def __init__(self, config=None):
        super().__init__('new_source', config)
        self.base_url = "https://newsource.com"
    
    def get_search_urls(self) -> List[str]:
        # Implement search URL generation
        pass
    
    async def get_listing_urls(self, base_url: str, max_pages: int = None):
        # Implement listing URL extraction
        pass
    
    async def extract_property_data(self, listing_url: str, html_content: str):
        # Implement property data extraction
        pass
```

## Monitoring & Dashboards

### Health Check Endpoint

```python
system = ProductionScrapingSystem()
await system.initialize()

# Get system health
health = await system.monitor.health_checker.check_database_health(system.database_connector)
print(f"Database status: {health.status}")

# Get comprehensive dashboard
dashboard = system.get_monitoring_dashboard()
```

### Metrics Available

- **System Metrics**: CPU, memory, disk usage
- **Scraping Metrics**: Request rates, success rates, response times
- **Data Quality Metrics**: Validation scores, duplicate rates
- **Business Metrics**: Properties scraped, sources active

### Alerts

Configurable alerts for:
- High error rates (>10% by default)
- Slow response times (>30s by default)
- Low property collection rates
- System resource exhaustion
- Database connectivity issues

## Data Quality & Validation

### Validation Pipeline

```python
from src.infrastructure.scrapers.data_quality_pipeline import EnhancedDataQualityPipeline

pipeline = EnhancedDataQualityPipeline()

# Process property data
result = await pipeline.process_property_with_enrichment(
    property_data={'title': 'Nice Apartment', 'price': '$2000', 'location': 'New York, NY'},
    existing_properties=[]
)

print(f"Validation score: {result.score}")
print(f"Issues found: {result.issues}")
print(f"Enhanced data: {result.cleaned_data}")
```

### Duplicate Detection

```python
from src.infrastructure.scrapers.data_quality_pipeline import DuplicateDetector

detector = DuplicateDetector(similarity_threshold=0.8)

# Check if properties are duplicates
is_duplicate = detector.is_duplicate(property1_data, property2_data)
similarity_score = detector.calculate_similarity(property1_data, property2_data)
```

## GDPR Compliance

### Automatic PII Detection

```python
from src.infrastructure.scrapers.gdpr_compliance import GDPRCompliance

gdpr = GDPRCompliance()

# Process property data for compliance
processed_data = gdpr.process_property_for_gdpr(
    property_data={'title': 'Apartment', 'contact_info': {'phone': '555-1234'}},
    anonymize=True
)

# Handle data subject requests
result = gdpr.handle_data_subject_request('access', 'user_id_123')
```

### Data Retention

- **Property listings**: 1 year retention (configurable)
- **Market analysis data**: 3 years retention
- **Analytics data**: 2 years retention
- **Audit logs**: 7 years retention

## Database Schema

### Properties Table

```sql
CREATE TABLE properties (
    id UUID PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    location VARCHAR(500) NOT NULL,
    bedrooms INTEGER DEFAULT 0,
    bathrooms DECIMAL(3,1) DEFAULT 0,
    square_feet INTEGER,
    property_type VARCHAR(50) DEFAULT 'apartment',
    amenities JSONB DEFAULT '[]',
    contact_info JSONB DEFAULT '{}',
    images JSONB DEFAULT '[]',
    
    -- Geographic data
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    formatted_address VARCHAR(500),
    
    -- Data quality
    data_quality_score DECIMAL(3,2) DEFAULT 0.0,
    validation_issues JSONB DEFAULT '[]',
    validation_warnings JSONB DEFAULT '[]',
    
    -- Metadata
    source_name VARCHAR(50) NOT NULL,
    source_url TEXT,
    source_id VARCHAR(200),
    is_active BOOLEAN DEFAULT true,
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## Performance & Scaling

### Optimization Tips

1. **Rate Limiting**: Adjust based on site requirements
2. **Concurrent Requests**: Start with 3-5, monitor performance
3. **Database Batching**: Use batch inserts for better performance
4. **Caching**: Enable Redis for improved duplicate detection
5. **Monitoring**: Regular monitoring prevents issues

### Scaling Considerations

- **Horizontal Scaling**: Run multiple instances with different sources
- **Database Sharding**: Partition by location or date
- **Queue Systems**: Use Redis/RabbitMQ for job distribution
- **Caching Layers**: CDN for static content, Redis for dynamic data

## Troubleshooting

### Common Issues

1. **Rate Limiting Errors**
   - Increase delay between requests
   - Reduce concurrent request limit
   - Check robots.txt compliance

2. **Database Connection Errors**
   - Verify connection pool settings
   - Check database credentials
   - Monitor connection count

3. **Memory Issues**
   - Reduce batch sizes
   - Implement pagination
   - Monitor memory usage

4. **Data Quality Issues**
   - Review extraction selectors
   - Check validation rules
   - Monitor data quality scores

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m src.infrastructure.scrapers.main_production_system --command test --environment development
```

### Logs Analysis

```bash
# Search for errors
grep "ERROR" /var/log/rental-ml/scraping.log

# Monitor rate limiting
grep "rate limited" /var/log/rental-ml/scraping.log

# Check data quality issues
grep "validation" /var/log/rental-ml/scraping.log
```

## Contributing

### Adding New Features

1. **Create Feature Branch**: `git checkout -b feature/new-scraper`
2. **Implement Changes**: Follow existing patterns
3. **Add Tests**: Include unit and integration tests
4. **Update Documentation**: Keep README and docstrings current
5. **Submit PR**: Include description and test results

### Code Standards

- **Type Hints**: Use comprehensive type annotations
- **Docstrings**: Document all public methods
- **Error Handling**: Comprehensive exception handling
- **Logging**: Appropriate log levels and messages
- **Testing**: Minimum 80% code coverage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- **GitHub Issues**: Use for bug reports and feature requests
- **Documentation**: Check inline docstrings and code comments
- **Email**: contact@rental-ml-system.com

---

**⚠️ Important**: Always respect website terms of service and robots.txt files when scraping. This system is designed for educational and research purposes. Ensure compliance with all applicable laws and regulations in your jurisdiction.