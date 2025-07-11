from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class ScheduleTypeEnum(str, Enum):
    """Enum for schedule types"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"


class ScrapingConfigRequest(BaseModel):
    """Request model for scraping configuration overrides"""
    max_concurrent_scrapers: Optional[int] = Field(None, ge=1, le=10, description="Maximum concurrent scrapers")
    max_properties_per_scraper: Optional[int] = Field(None, ge=1, le=10000, description="Maximum properties per scraper")
    deduplication_enabled: Optional[bool] = Field(None, description="Enable deduplication")
    cache_results: Optional[bool] = Field(None, description="Cache scraping results")
    save_to_database: Optional[bool] = Field(None, description="Save results to database")
    global_rate_limit: Optional[float] = Field(None, ge=0.1, le=10.0, description="Global rate limit in requests/second")
    scraper_delay: Optional[float] = Field(None, ge=0.0, le=60.0, description="Delay between scrapers in seconds")


class ManualScrapingRequest(BaseModel):
    """Request model for manual scraping execution"""
    scrapers: Optional[List[str]] = Field(default_factory=list, description="List of scraper names to use")
    max_properties: Optional[int] = Field(None, ge=1, le=10000, description="Maximum total properties to scrape")
    config_override: Optional[ScrapingConfigRequest] = Field(None, description="Configuration overrides")

    @validator('scrapers')
    def validate_scrapers(cls, v):
        if v and not all(isinstance(name, str) and name.strip() for name in v):
            raise ValueError('All scraper names must be non-empty strings')
        return v


class ScrapingStatsResponse(BaseModel):
    """Response model for scraping statistics"""
    total_properties_found: int = Field(..., description="Total properties found during scraping")
    total_properties_saved: int = Field(..., description="Total properties saved to database")
    total_duplicates_filtered: int = Field(..., description="Total duplicate properties filtered out")
    total_errors: int = Field(..., description="Total errors encountered")
    duration_seconds: float = Field(..., description="Total scraping duration in seconds")
    scrapers_used: List[str] = Field(..., description="List of scrapers that were executed")
    success_rate: Optional[float] = Field(None, description="Success rate as percentage")
    properties_per_scraper: Optional[Dict[str, int]] = Field(None, description="Properties found per scraper")


class ScheduledJobRequest(BaseModel):
    """Request model for creating/updating scheduled jobs"""
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable job name")
    schedule_type: ScheduleTypeEnum = Field(..., description="Type of schedule")
    interval_hours: float = Field(..., gt=0, le=8760, description="Interval in hours")
    scrapers: List[str] = Field(..., min_items=1, description="List of scraper names to use")
    max_properties: Optional[int] = Field(None, ge=1, le=10000, description="Maximum properties to scrape per run")
    config_override: Optional[ScrapingConfigRequest] = Field(None, description="Configuration overrides")
    enabled: bool = Field(default=True, description="Whether the job should be enabled")

    @validator('scrapers')
    def validate_scrapers(cls, v):
        if not all(isinstance(name, str) and name.strip() for name in v):
            raise ValueError('All scraper names must be non-empty strings')
        return v

    @validator('interval_hours')
    def validate_interval_hours(cls, v, values):
        schedule_type = values.get('schedule_type')
        if schedule_type == ScheduleTypeEnum.HOURLY and v < 1:
            raise ValueError('Hourly jobs must have at least 1 hour interval')
        elif schedule_type == ScheduleTypeEnum.DAILY and v < 24:
            raise ValueError('Daily jobs must have at least 24 hour interval')
        elif schedule_type == ScheduleTypeEnum.WEEKLY and v < 168:
            raise ValueError('Weekly jobs must have at least 168 hour interval')
        return v


class ScheduledJobUpdateRequest(BaseModel):
    """Request model for updating scheduled jobs"""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Human-readable job name")
    schedule_type: Optional[ScheduleTypeEnum] = Field(None, description="Type of schedule")
    interval_hours: Optional[float] = Field(None, gt=0, le=8760, description="Interval in hours")
    scrapers: Optional[List[str]] = Field(None, min_items=1, description="List of scraper names to use")
    max_properties: Optional[int] = Field(None, ge=1, le=10000, description="Maximum properties to scrape per run")
    config_override: Optional[ScrapingConfigRequest] = Field(None, description="Configuration overrides")
    enabled: Optional[bool] = Field(None, description="Whether the job should be enabled")

    @validator('scrapers')
    def validate_scrapers(cls, v):
        if v is not None and not all(isinstance(name, str) and name.strip() for name in v):
            raise ValueError('All scraper names must be non-empty strings')
        return v


class ScheduledJobResponse(BaseModel):
    """Response model for scheduled job information"""
    job_id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Human-readable job name")
    schedule_type: ScheduleTypeEnum = Field(..., description="Type of schedule")
    interval_hours: float = Field(..., description="Interval in hours")
    scrapers: List[str] = Field(..., description="List of scraper names")
    max_properties: Optional[int] = Field(None, description="Maximum properties to scrape per run")
    enabled: bool = Field(..., description="Whether the job is enabled")
    next_run: Optional[datetime] = Field(None, description="Next scheduled run time")
    last_run: Optional[datetime] = Field(None, description="Last execution time")
    last_success: Optional[bool] = Field(None, description="Success status of last run")
    total_runs: int = Field(default=0, description="Total number of executions")
    created_at: datetime = Field(..., description="Job creation time")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobExecutionResponse(BaseModel):
    """Response model for job execution results"""
    job_id: str = Field(..., description="Job identifier")
    success: bool = Field(..., description="Whether execution was successful")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: datetime = Field(..., description="Execution completion time")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    stats: Optional[ScrapingStatsResponse] = Field(None, description="Scraping statistics")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ScrapingStatusResponse(BaseModel):
    """Response model for scraping system status"""
    scheduler_initialized: bool = Field(..., description="Whether the scheduler is initialized")
    scheduled_jobs: Dict[str, Any] = Field(..., description="Status of all scheduled jobs")
    recent_activity: Dict[str, Any] = Field(..., description="Recent scraping activity")
    system_health: Dict[str, Any] = Field(default_factory=dict, description="System health metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobHistoryResponse(BaseModel):
    """Response model for job execution history"""
    job_id: str = Field(..., description="Job identifier")
    history: List[JobExecutionResponse] = Field(..., description="List of execution history entries")
    total_executions: int = Field(..., description="Total number of executions")
    success_rate: float = Field(..., description="Success rate as percentage")
    average_duration: float = Field(..., description="Average execution duration in seconds")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ScraperInfoResponse(BaseModel):
    """Response model for scraper information"""
    name: str = Field(..., description="Scraper name/identifier")
    display_name: str = Field(..., description="Human-readable display name")
    description: str = Field(..., description="Scraper description")
    supported_locations: List[str] = Field(..., description="List of supported locations")
    rate_limit: str = Field(..., description="Rate limit description")
    estimated_properties_per_run: str = Field(..., description="Estimated properties per run")
    last_run: Optional[datetime] = Field(None, description="Last execution time")
    status: str = Field(default="available", description="Current scraper status")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ScrapersListResponse(BaseModel):
    """Response model for list of available scrapers"""
    scrapers: List[ScraperInfoResponse] = Field(..., description="List of available scrapers")
    total_count: int = Field(..., description="Total number of scrapers")
    active_count: int = Field(..., description="Number of active scrapers")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ScrapingErrorResponse(BaseModel):
    """Error response model for scraping operations"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    job_id: Optional[str] = Field(None, description="Related job ID if applicable")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }