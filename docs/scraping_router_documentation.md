# Scraping Router Documentation

## Overview

The Scraping Router (`scraping_router.py`) provides a comprehensive FastAPI interface for managing web scraping operations in the Rental ML System. It includes endpoints for manual scraping execution, scheduled job management, system monitoring, and scraper information.

## Endpoints

### Manual Scraping

#### `POST /api/v1/scraping/run`
Execute an immediate scraping session with configurable parameters.

**Request Body:**
```json
{
  "scrapers": ["apartments_com"],
  "max_properties": 1000,
  "config_override": {
    "max_concurrent_scrapers": 2,
    "global_rate_limit": 1.0,
    "deduplication_enabled": true
  }
}
```

**Response:**
- Comprehensive scraping statistics including success rates, duration, and errors
- Properties found and saved counts
- Performance metrics per scraper

### Scheduled Job Management

#### `GET /api/v1/scraping/jobs`
List all scheduled scraping jobs with their status and configuration.

**Query Parameters:**
- `enabled_only`: Show only enabled jobs (boolean)

#### `POST /api/v1/scraping/jobs`
Create a new scheduled scraping job.

**Request Body:**
```json
{
  "name": "Daily Property Scraping",
  "schedule_type": "daily",
  "interval_hours": 24.0,
  "scrapers": ["apartments_com"],
  "max_properties": 1000,
  "enabled": true
}
```

#### `GET /api/v1/scraping/jobs/{job_id}`
Get detailed information about a specific scheduled job.

#### `PUT /api/v1/scraping/jobs/{job_id}`
Update an existing scheduled job configuration.

#### `DELETE /api/v1/scraping/jobs/{job_id}`
Delete a scheduled job permanently.

#### `PATCH /api/v1/scraping/jobs/{job_id}/enable`
Enable a disabled scheduled job.

#### `PATCH /api/v1/scraping/jobs/{job_id}/disable`
Disable a scheduled job without deleting it.

### Job Execution

#### `POST /api/v1/scraping/jobs/{job_id}/run`
Execute a scheduled job immediately, outside of its normal schedule.

**Response:**
- Job execution results with timestamps
- Success/failure status
- Detailed scraping statistics if successful
- Error messages if failed

### System Monitoring

#### `GET /api/v1/scraping/status`
Get comprehensive scraping system status including:
- Scheduler initialization status
- All scheduled jobs and their states
- Recent activity metrics
- System health indicators

#### `GET /api/v1/scraping/jobs/{job_id}/history`
Get execution history for a specific job.

**Query Parameters:**
- `limit`: Number of history entries to return (default: 50, max: 200)

**Response:**
- Detailed execution history
- Success rates and performance trends
- Error patterns and debugging information

### Scraper Information

#### `GET /api/v1/scraping/scrapers`
Get list of available scrapers and their capabilities.

**Response:**
- Scraper names and descriptions
- Supported locations
- Rate limits and performance characteristics
- Current status and availability

## Data Models

### ScheduleType Enum
- `hourly`: Execute every N hours
- `daily`: Execute daily
- `weekly`: Execute weekly
- `custom`: Custom interval in hours

### Configuration Override
```json
{
  "max_concurrent_scrapers": 2,
  "max_properties_per_scraper": 1000,
  "deduplication_enabled": true,
  "cache_results": true,
  "save_to_database": true,
  "global_rate_limit": 1.0,
  "scraper_delay": 10.0
}
```

### Job Response Structure
```json
{
  "job_id": "job_1234567890_abc123",
  "name": "Daily Property Scraping",
  "schedule_type": "daily",
  "interval_hours": 24.0,
  "scrapers": ["apartments_com"],
  "max_properties": 1000,
  "enabled": true,
  "next_run": "2023-12-07T10:00:00Z",
  "last_run": "2023-12-06T10:00:00Z",
  "last_success": true,
  "total_runs": 42,
  "created_at": "2023-11-15T08:30:00Z"
}
```

## Error Handling

The router implements comprehensive error handling:

- **404 Not Found**: When requesting non-existent jobs
- **400 Bad Request**: For invalid input parameters or validation errors
- **500 Internal Server Error**: For system failures during execution
- **503 Service Unavailable**: When scheduler is not initialized

All errors include detailed messages and relevant context for debugging.

## Integration with ScrapingUseCase

The router uses dependency injection to access the `ScrapingUseCase` class, which provides:
- Manual scraping execution
- Scheduled job management
- System status monitoring
- Job history tracking
- Scraper capability information

## Security Considerations

- All endpoints require proper repository factory access
- Input validation prevents malicious configurations
- Rate limiting protects against abuse
- Error messages don't expose sensitive system information

## Usage Examples

### Create a Weekly Comprehensive Scraping Job
```bash
curl -X POST "http://localhost:8000/api/v1/scraping/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Weekly Comprehensive Scraping",
    "schedule_type": "weekly",
    "interval_hours": 168.0,
    "scrapers": ["apartments_com"],
    "max_properties": 5000,
    "config_override": {
      "max_concurrent_scrapers": 1,
      "global_rate_limit": 0.5
    },
    "enabled": true
  }'
```

### Execute Manual Scraping
```bash
curl -X POST "http://localhost:8000/api/v1/scraping/run" \
  -H "Content-Type: application/json" \
  -d '{
    "scrapers": ["apartments_com"],
    "max_properties": 100,
    "config_override": {
      "deduplication_enabled": true,
      "cache_results": true
    }
  }'
```

### Get System Status
```bash
curl "http://localhost:8000/api/v1/scraping/status"
```