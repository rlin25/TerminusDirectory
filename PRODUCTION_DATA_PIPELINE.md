# Production Data Pipeline Setup Guide

This guide covers the complete setup and operation of the production data ingestion pipeline for the Rental ML System.

## 📋 Overview

The production data pipeline includes:
- **Data Seeding**: Populate database with realistic property and user data
- **Data Ingestion**: Import property data from various sources
- **Data Validation**: Ensure data quality and consistency  
- **Data Operations Testing**: Verify search, recommendations, and repository operations
- **Data Management**: Quality checks, cleaning, and export/import utilities

## 🚀 Quick Start

### 1. Complete Pipeline Setup

Run the comprehensive setup script to initialize everything:

```bash
python setup_production_data_pipeline.py
```

This will:
- ✅ Test database connectivity
- ✅ Seed initial production data (50 users, 200 properties, 1000 interactions)
- ✅ Test all data operations (search, recommendations, repositories)
- ✅ Run data quality analysis
- ✅ Test data ingestion pipeline

### 2. Individual Script Usage

Each component can be run independently:

#### Production Data Seeding
```bash
# Seed with default amounts
python scripts/production_data_seeder.py

# The script will create:
# - 100 realistic users with preferences
# - 500 properties with varied characteristics
# - 2000 user interactions
# - 1000 search queries
# - Sample ML model records
```

#### Data Ingestion Pipeline
```bash
# Check database status
python scripts/data_ingestion_pipeline.py --command status

# Generate and ingest sample properties
python scripts/data_ingestion_pipeline.py --command sample --count 100

# Ingest from JSON file
python scripts/data_ingestion_pipeline.py --command ingest --file data.json

# Clean up data
python scripts/data_ingestion_pipeline.py --command cleanup
```

#### Test Data Operations
```bash
# Run comprehensive tests
python scripts/test_data_operations.py
```

#### Data Management Utilities
```bash
# Run data quality check
python scripts/data_management_utilities.py --command quality

# Clean up data issues
python scripts/data_management_utilities.py --command cleanup

# Export data
python scripts/data_management_utilities.py --command export --output-dir ./exports

# Run all utilities
python scripts/data_management_utilities.py --command all
```

## 📊 Data Schema

### Properties Table
```sql
CREATE TABLE properties (
    id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    location VARCHAR(255) NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms DECIMAL(3,1) NOT NULL,
    square_feet INTEGER,
    amenities TEXT[],
    contact_info JSONB,
    images TEXT[],
    scraped_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    property_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    slug VARCHAR(255),
    external_id VARCHAR(255),
    external_url VARCHAR(500),
    data_quality_score DECIMAL(3,2),
    last_verified TIMESTAMP
);
```

### Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active',
    min_price DECIMAL(10,2),
    max_price DECIMAL(10,2),
    min_bedrooms INTEGER,
    max_bedrooms INTEGER,
    min_bathrooms DECIMAL(3,1),
    max_bathrooms DECIMAL(3,1),
    preferred_locations TEXT[],
    required_amenities TEXT[],
    property_types TEXT[],
    last_login TIMESTAMP,
    login_count INTEGER DEFAULT 0,
    preference_updated_at TIMESTAMP
);
```

### User Interactions Table
```sql
CREATE TABLE user_interactions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    property_id UUID REFERENCES properties(id),
    interaction_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    duration_seconds INTEGER,
    session_id UUID,
    user_agent TEXT,
    ip_address INET,
    referrer VARCHAR(500),
    interaction_strength DECIMAL(3,2)
);
```

## 🔧 Configuration

### Environment Variables

Create `.env.production` with:

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rental_ml
REDIS_URL=redis://localhost:6379

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Performance
CONNECTION_POOL_SIZE=10
QUERY_TIMEOUT=30
CACHE_TTL=3600
```

### Database Setup

Ensure PostgreSQL is running and the database exists:

```bash
# Create database
createdb rental_ml

# Run migrations
python scripts/run_migrations.py
```

## 📈 Data Quality Monitoring

### Quality Metrics Tracked

1. **Property Data Quality**:
   - Missing required fields (title, price, location)
   - Data completeness (description, amenities, images)
   - Price outliers and validation
   - Location standardization
   - Date consistency

2. **User Data Quality**:
   - Email validation and duplicates
   - Preference completeness
   - User activity levels

3. **Interaction Data Quality**:
   - Orphaned records
   - Temporal patterns
   - Data integrity

### Quality Check Commands

```bash
# Run quality analysis
python scripts/data_management_utilities.py --command quality

# Clean up data issues
python scripts/data_management_utilities.py --command cleanup

# Export quality reports
python scripts/data_management_utilities.py --command export
```

## 🛠 Data Ingestion Sources

### Supported Formats

1. **JSON Files**: Property data in JSON format
2. **CSV Files**: Bulk property imports
3. **API Integration**: Real-time data from scraping services
4. **Sample Data**: Generated realistic test data

### JSON Format Example

```json
[
  {
    "title": "Modern Downtown Apartment",
    "description": "Beautiful 2-bedroom apartment in downtown",
    "price": 2500,
    "location": "Downtown",
    "bedrooms": 2,
    "bathrooms": 1.5,
    "square_feet": 900,
    "property_type": "apartment",
    "amenities": ["parking", "gym", "pool"],
    "contact_info": {
      "phone": "(555) 123-4567",
      "email": "contact@example.com"
    },
    "images": ["https://example.com/image1.jpg"]
  }
]
```

## 📊 Testing and Validation

### Test Coverage

The test suite validates:

1. **Repository Operations**:
   - Create, read, update, delete operations
   - User and property repositories
   - Model repository functionality

2. **Search Operations**:
   - Basic property search
   - Price range filtering
   - Location and amenity filtering
   - Combined filter searches

3. **Recommendation System**:
   - Data availability for recommendations
   - Content-based similarity
   - User preference matching

4. **Data Integrity**:
   - Foreign key constraints
   - Data type validation
   - Business rule compliance

### Running Tests

```bash
# Run all data operation tests
python scripts/test_data_operations.py

# Check specific functionality
python scripts/data_ingestion_pipeline.py --command status
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   ```bash
   # Check database is running
   pg_isready -h localhost -p 5432
   
   # Verify connection string
   echo $DATABASE_URL
   ```

2. **Missing Tables**:
   ```bash
   # Run migrations
   python scripts/run_migrations.py
   ```

3. **Data Quality Issues**:
   ```bash
   # Run cleanup utilities
   python scripts/data_management_utilities.py --command cleanup
   ```

4. **Performance Issues**:
   ```bash
   # Check connection pool settings
   # Verify database indexes
   # Monitor query performance
   ```

### Debugging Steps

1. Check logs in the console output
2. Verify environment variables are set correctly
3. Test database connectivity independently
4. Run individual components to isolate issues
5. Check data quality reports for inconsistencies

## 📈 Performance Optimization

### Database Optimization

1. **Indexes**: Ensure proper indexes on frequently queried fields
2. **Connection Pooling**: Configure appropriate pool sizes
3. **Query Optimization**: Monitor slow queries
4. **Partitioning**: Consider table partitioning for large datasets

### Data Pipeline Optimization

1. **Batch Processing**: Use batch inserts for large datasets
2. **Parallel Processing**: Utilize concurrent operations where possible
3. **Caching**: Implement Redis caching for frequently accessed data
4. **Data Validation**: Pre-validate data to avoid database errors

## 🔄 Maintenance

### Regular Tasks

1. **Daily**:
   - Monitor data quality metrics
   - Check for new properties and users
   - Review system logs

2. **Weekly**:
   - Run data cleanup utilities
   - Export data statistics
   - Performance analysis

3. **Monthly**:
   - Database maintenance and optimization
   - Archive old interaction data
   - Review and update data quality rules

### Automation

Set up cron jobs for regular maintenance:

```bash
# Daily data quality check
0 2 * * * cd /path/to/rental-ml-system && python scripts/data_management_utilities.py --command quality

# Weekly cleanup
0 1 * * 0 cd /path/to/rental-ml-system && python scripts/data_management_utilities.py --command cleanup

# Monthly export
0 0 1 * * cd /path/to/rental-ml-system && python scripts/data_management_utilities.py --command export
```

## 📞 Support

For issues or questions:

1. Check this documentation
2. Review system logs
3. Run diagnostic scripts
4. Contact development team

## 🎯 Next Steps

After setting up the data pipeline:

1. **Start Production API**: `python main_production.py`
2. **Configure Monitoring**: Set up alerts for data quality
3. **Schedule Data Updates**: Implement regular data refresh
4. **Performance Tuning**: Optimize based on usage patterns
5. **Scale Planning**: Prepare for data growth and increased load