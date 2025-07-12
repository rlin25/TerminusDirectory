"""
Production-ready scraping orchestrator for managing multiple rental data sources.

This module provides comprehensive orchestration of all property scrapers with
job management, scheduling, monitoring, and real-time data ingestion.
"""

import asyncio
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import pickle
import time
import traceback

from .production_base_scraper import ProductionBaseScraper
from .production_apartments_com_scraper import ProductionApartmentsComScraper
from .production_rentals_com_scraper import ProductionRentalsComScraper
from .production_zillow_scraper import ProductionZillowScraper
from .production_rent_com_scraper import ProductionRentComScraper
from .data_quality_pipeline import EnhancedDataQualityPipeline, ValidationResult
from .config import get_config, ProductionScrapingConfig
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScrapingJob:
    """Represents a scraping job"""
    id: str
    source_name: str
    job_type: str  # 'full_scrape', 'incremental', 'test'
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    properties_found: int = 0
    properties_valid: int = 0
    properties_stored: int = 0
    max_pages: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def duration(self) -> Optional[timedelta]:
        """Get job duration"""
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return end_time - self.started_at
        return None


@dataclass
class ScrapingSession:
    """Represents a scraping session across multiple sources"""
    id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    jobs: List[ScrapingJob] = field(default_factory=list)
    total_properties: int = 0
    total_duplicates: int = 0
    session_type: str = "scheduled"  # 'scheduled', 'manual', 'test'


class ProductionScrapingOrchestrator:
    """Production orchestrator for managing all rental property scrapers"""
    
    def __init__(self, config: ProductionScrapingConfig = None, database_connector=None):
        self.config = config or get_config()
        self.database_connector = database_connector
        self.data_quality_pipeline = EnhancedDataQualityPipeline(self.config)
        
        # Job management
        self.active_jobs: Dict[str, ScrapingJob] = {}
        self.job_queue: List[ScrapingJob] = []
        self.completed_jobs: List[ScrapingJob] = []
        self.current_session: Optional[ScrapingSession] = None
        
        # Scraper registry
        self.scraper_classes: Dict[str, Type[ProductionBaseScraper]] = {
            'apartments_com': ProductionApartmentsComScraper,
            'rentals_com': ProductionRentalsComScraper,
            'zillow': ProductionZillowScraper,
            'rent_com': ProductionRentComScraper
        }
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'job_started': [],
            'job_completed': [],
            'job_failed': [],
            'property_found': [],
            'session_started': [],
            'session_completed': []
        }
        
        # Performance metrics
        self.session_metrics: Dict[str, Any] = {}
        
        # Concurrency control
        self.max_concurrent_jobs = min(self.config.scraping.max_concurrent_requests, 3)
        self.job_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        logger.info(f"Initialized scraping orchestrator with {len(self.scraper_classes)} scrapers")
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for scraping events"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    async def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event callbacks"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback {callback}: {e}")
    
    def create_job(
        self,
        source_name: str,
        job_type: str = 'full_scrape',
        priority: JobPriority = JobPriority.MEDIUM,
        max_pages: Optional[int] = None,
        parameters: Dict[str, Any] = None
    ) -> ScrapingJob:
        """Create a new scraping job"""
        
        if source_name not in self.scraper_classes:
            raise ValueError(f"Unknown scraper source: {source_name}")
        
        job = ScrapingJob(
            id=str(uuid4()),
            source_name=source_name,
            job_type=job_type,
            priority=priority,
            max_pages=max_pages,
            parameters=parameters or {}
        )
        
        # Add to queue
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda j: j.priority.value, reverse=True)
        
        logger.info(f"Created job {job.id} for {source_name} with priority {priority}")
        return job
    
    def create_session(
        self,
        sources: List[str] = None,
        session_type: str = "manual",
        job_type: str = 'full_scrape',
        max_pages_per_source: Optional[int] = None
    ) -> ScrapingSession:
        """Create a new scraping session"""
        
        if sources is None:
            sources = [source.name for source in self.config.get_enabled_sources()]
        
        session = ScrapingSession(
            id=str(uuid4()),
            session_type=session_type
        )
        
        # Create jobs for each source
        for source_name in sources:
            if source_name in self.scraper_classes:
                job = self.create_job(
                    source_name=source_name,
                    job_type=job_type,
                    max_pages=max_pages_per_source
                )
                session.jobs.append(job)
        
        self.current_session = session
        logger.info(f"Created session {session.id} with {len(session.jobs)} jobs")
        return session
    
    async def execute_job(self, job: ScrapingJob) -> bool:
        """Execute a single scraping job"""
        
        async with self.job_semaphore:
            try:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                self.active_jobs[job.id] = job
                
                await self._trigger_event('job_started', {
                    'job': job,
                    'session': self.current_session
                })
                
                logger.info(f"Starting job {job.id} for {job.source_name}")
                
                # Create and initialize scraper
                scraper_class = self.scraper_classes[job.source_name]
                scraper = scraper_class(self.config)
                
                properties_processed = []
                
                async with scraper:
                    # Get search URLs
                    search_urls = scraper.get_search_urls()
                    total_urls = len(search_urls)
                    
                    for url_index, search_url in enumerate(search_urls):
                        try:
                            logger.info(f"Processing search URL {url_index + 1}/{total_urls}: {search_url}")
                            
                            # Get listing URLs
                            listing_count = 0
                            async for listing_url in scraper.get_listing_urls(search_url, job.max_pages):
                                try:
                                    # Fetch and process property
                                    html_content = await scraper.fetch_page(listing_url)
                                    if html_content:
                                        property_data = await scraper.extract_property_data(listing_url, html_content)
                                        
                                        if property_data:
                                            # Process through data quality pipeline
                                            validation_result = await self.data_quality_pipeline.process_property_with_enrichment(
                                                property_data.__dict__,
                                                existing_properties=[p.__dict__ for p in properties_processed]
                                            )
                                            
                                            if validation_result.is_valid:
                                                # Create enhanced property
                                                enhanced_property = self._create_enhanced_property(
                                                    property_data, validation_result
                                                )
                                                properties_processed.append(enhanced_property)
                                                job.properties_valid += 1
                                                
                                                # Store in database if configured
                                                if self.database_connector:
                                                    await self._store_property(enhanced_property)
                                                    job.properties_stored += 1
                                                
                                                await self._trigger_event('property_found', {
                                                    'property': enhanced_property,
                                                    'job': job,
                                                    'validation_result': validation_result
                                                })
                                            else:
                                                logger.debug(f"Property failed validation: {validation_result.issues}")
                                        
                                        job.properties_found += 1
                                        listing_count += 1
                                        
                                        # Update progress
                                        job.progress = ((url_index + 1) / total_urls) * 100
                                        
                                        # Check for cancellation
                                        if job.status == JobStatus.CANCELLED:
                                            logger.info(f"Job {job.id} cancelled")
                                            return False
                                        
                                except Exception as e:
                                    logger.error(f"Error processing listing {listing_url}: {e}")
                                    continue
                            
                            logger.info(f"Processed {listing_count} listings from {search_url}")
                            
                        except Exception as e:
                            logger.error(f"Error processing search URL {search_url}: {e}")
                            continue
                
                # Job completed successfully
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.progress = 100.0
                
                # Update session metrics
                if self.current_session:
                    self.current_session.total_properties += job.properties_valid
                
                logger.info(
                    f"Job {job.id} completed - Found: {job.properties_found}, "
                    f"Valid: {job.properties_valid}, Stored: {job.properties_stored}"
                )
                
                await self._trigger_event('job_completed', {
                    'job': job,
                    'properties': properties_processed,
                    'session': self.current_session
                })
                
                return True
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error_message = str(e)
                
                logger.error(f"Job {job.id} failed: {e}")
                logger.debug(traceback.format_exc())
                
                await self._trigger_event('job_failed', {
                    'job': job,
                    'error': e,
                    'session': self.current_session
                })
                
                return False
                
            finally:
                # Cleanup
                if job.id in self.active_jobs:
                    del self.active_jobs[job.id]
                self.completed_jobs.append(job)
    
    async def execute_session(self, session: ScrapingSession = None) -> ScrapingSession:
        """Execute all jobs in a session"""
        
        if session is None:
            session = self.current_session
        
        if not session:
            raise ValueError("No session provided or active")
        
        session.started_at = datetime.utcnow()
        
        await self._trigger_event('session_started', {
            'session': session
        })
        
        logger.info(f"Starting session {session.id} with {len(session.jobs)} jobs")
        
        # Execute jobs concurrently
        tasks = []
        for job in session.jobs:
            if job.status == JobStatus.PENDING:
                task = asyncio.create_task(self.execute_job(job))
                tasks.append(task)
        
        # Wait for all jobs to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            successful_jobs = sum(1 for result in results if result is True)
            failed_jobs = len(results) - successful_jobs
            
            logger.info(f"Session {session.id} completed - {successful_jobs} successful, {failed_jobs} failed")
        
        session.completed_at = datetime.utcnow()
        
        await self._trigger_event('session_completed', {
            'session': session
        })
        
        return session
    
    def _create_enhanced_property(
        self, 
        property_data: Property, 
        validation_result: ValidationResult
    ) -> Property:
        """Create enhanced property with validation results"""
        
        # Update property with cleaned data
        for key, value in validation_result.cleaned_data.items():
            if hasattr(property_data, key):
                setattr(property_data, key, value)
        
        # Add quality metadata
        property_data.data_quality_score = validation_result.score
        property_data.validation_issues = validation_result.issues
        property_data.validation_warnings = validation_result.warnings
        
        return property_data
    
    async def _store_property(self, property_data: Property):
        """Store property in database"""
        try:
            if self.database_connector:
                await self.database_connector.create_property(property_data)
        except Exception as e:
            logger.error(f"Error storing property {property_data.id}: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[ScrapingJob]:
        """Get status of a specific job"""
        
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check completed jobs
        for job in self.completed_jobs:
            if job.id == job_id:
                return job
        
        # Check queued jobs
        for job in self.job_queue:
            if job.id == job_id:
                return job
        
        return None
    
    def get_session_status(self, session_id: str = None) -> Optional[ScrapingSession]:
        """Get status of a session"""
        if session_id is None:
            return self.current_session
        
        if self.current_session and self.current_session.id == session_id:
            return self.current_session
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or queued job"""
        job = self.get_job_status(job_id)
        if job:
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                job.status = JobStatus.CANCELLED
                logger.info(f"Cancelled job {job_id}")
                return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        
        active_count = len(self.active_jobs)
        queued_count = len(self.job_queue)
        completed_count = len(self.completed_jobs)
        
        # Calculate success rate
        if completed_count > 0:
            successful_jobs = sum(1 for job in self.completed_jobs if job.status == JobStatus.COMPLETED)
            success_rate = (successful_jobs / completed_count) * 100
        else:
            success_rate = 0.0
        
        # Current session metrics
        session_metrics = {}
        if self.current_session:
            session_metrics = {
                'id': self.current_session.id,
                'started_at': self.current_session.started_at,
                'total_jobs': len(self.current_session.jobs),
                'completed_jobs': sum(1 for job in self.current_session.jobs if job.status == JobStatus.COMPLETED),
                'total_properties': self.current_session.total_properties
            }
        
        return {
            'jobs': {
                'active': active_count,
                'queued': queued_count,
                'completed': completed_count,
                'success_rate': success_rate
            },
            'current_session': session_metrics,
            'scrapers': {
                'available': list(self.scraper_classes.keys()),
                'enabled': [source.name for source in self.config.get_enabled_sources()]
            },
            'performance': {
                'avg_job_duration': self._calculate_avg_job_duration(),
                'properties_per_hour': self._calculate_properties_per_hour()
            }
        }
    
    def _calculate_avg_job_duration(self) -> float:
        """Calculate average job duration in seconds"""
        completed_jobs = [job for job in self.completed_jobs if job.duration()]
        if not completed_jobs:
            return 0.0
        
        total_duration = sum(job.duration().total_seconds() for job in completed_jobs)
        return total_duration / len(completed_jobs)
    
    def _calculate_properties_per_hour(self) -> float:
        """Calculate properties scraped per hour"""
        if not self.completed_jobs:
            return 0.0
        
        total_properties = sum(job.properties_valid for job in self.completed_jobs)
        total_hours = sum(job.duration().total_seconds() for job in self.completed_jobs if job.duration()) / 3600
        
        return total_properties / total_hours if total_hours > 0 else 0.0
    
    async def test_scrapers(self) -> Dict[str, Any]:
        """Test all available scrapers"""
        results = {}
        
        for source_name in self.scraper_classes.keys():
            try:
                logger.info(f"Testing {source_name} scraper")
                
                job = self.create_job(
                    source_name=source_name,
                    job_type='test',
                    priority=JobPriority.LOW,
                    max_pages=1
                )
                
                success = await self.execute_job(job)
                
                results[source_name] = {
                    'success': success,
                    'properties_found': job.properties_found,
                    'properties_valid': job.properties_valid,
                    'duration': job.duration().total_seconds() if job.duration() else 0,
                    'error': job.error_message
                }
                
            except Exception as e:
                results[source_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the orchestrator"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'configuration': {
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'enabled_sources': [s.name for s in self.config.get_enabled_sources()],
                'data_quality_enabled': True,
                'database_connected': self.database_connector is not None
            },
            'metrics': self.get_metrics()
        }


# Scheduler for automated scraping
class ScrapingScheduler:
    """Scheduler for automated scraping sessions"""
    
    def __init__(self, orchestrator: ProductionScrapingOrchestrator):
        self.orchestrator = orchestrator
        self.scheduled_jobs: List[Dict[str, Any]] = []
        self.running = False
    
    def schedule_daily_scraping(
        self,
        hour: int = 2,
        minute: int = 0,
        sources: List[str] = None,
        max_pages_per_source: int = 50
    ):
        """Schedule daily scraping at specified time"""
        
        schedule_config = {
            'type': 'daily',
            'hour': hour,
            'minute': minute,
            'sources': sources,
            'max_pages_per_source': max_pages_per_source,
            'last_run': None
        }
        
        self.scheduled_jobs.append(schedule_config)
        logger.info(f"Scheduled daily scraping at {hour:02d}:{minute:02d}")
    
    def schedule_weekly_scraping(
        self,
        weekday: int,  # 0=Monday, 6=Sunday
        hour: int = 1,
        minute: int = 0,
        sources: List[str] = None,
        max_pages_per_source: int = 100
    ):
        """Schedule weekly scraping"""
        
        schedule_config = {
            'type': 'weekly',
            'weekday': weekday,
            'hour': hour,
            'minute': minute,
            'sources': sources,
            'max_pages_per_source': max_pages_per_source,
            'last_run': None
        }
        
        self.scheduled_jobs.append(schedule_config)
        logger.info(f"Scheduled weekly scraping on weekday {weekday} at {hour:02d}:{minute:02d}")
    
    async def run_scheduler(self):
        """Run the scheduler loop"""
        self.running = True
        logger.info("Starting scraping scheduler")
        
        while self.running:
            try:
                now = datetime.now()
                
                for schedule in self.scheduled_jobs:
                    if self._should_run_schedule(schedule, now):
                        logger.info(f"Triggering scheduled scraping: {schedule['type']}")
                        
                        session = self.orchestrator.create_session(
                            sources=schedule['sources'],
                            session_type='scheduled',
                            max_pages_per_source=schedule['max_pages_per_source']
                        )
                        
                        # Run session in background
                        asyncio.create_task(self.orchestrator.execute_session(session))
                        
                        schedule['last_run'] = now
                
                # Sleep for 1 minute before checking again
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                await asyncio.sleep(60)
    
    def _should_run_schedule(self, schedule: Dict[str, Any], now: datetime) -> bool:
        """Check if schedule should run now"""
        
        # Check if already ran today/this week
        if schedule['last_run']:
            last_run = schedule['last_run']
            
            if schedule['type'] == 'daily':
                if last_run.date() == now.date():
                    return False
            elif schedule['type'] == 'weekly':
                # Check if we're in the same week
                if (now - last_run).days < 7:
                    return False
        
        # Check time match
        if schedule['type'] == 'daily':
            return now.hour == schedule['hour'] and now.minute == schedule['minute']
        
        elif schedule['type'] == 'weekly':
            return (now.weekday() == schedule['weekday'] and 
                   now.hour == schedule['hour'] and 
                   now.minute == schedule['minute'])
        
        return False
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Stopping scraping scheduler")