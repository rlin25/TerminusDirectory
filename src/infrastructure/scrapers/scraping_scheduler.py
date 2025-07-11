"""
Scraping scheduler for automated property data collection.

This module provides scheduling capabilities for running scrapers at regular intervals,
with support for different scheduling strategies and job management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from .scraping_orchestrator import ScrapingOrchestrator, ScrapingStats, ScrapingConfig
from ..data.repositories.postgres_property_repository import PostgresPropertyRepository
from ..data.repositories.redis_cache_repository import RedisCacheRepository

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduling intervals"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"


@dataclass
class ScheduledJob:
    """Configuration for a scheduled scraping job"""
    job_id: str
    name: str
    schedule_type: ScheduleType
    interval_hours: float
    scrapers: List[str]
    max_properties: Optional[int] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    total_runs: int = 0
    successful_runs: int = 0
    config_override: Optional[ScrapingConfig] = None


@dataclass
class JobResult:
    """Result of a scheduled job execution"""
    job_id: str
    started_at: datetime
    completed_at: datetime
    success: bool
    stats: Optional[ScrapingStats] = None
    error_message: Optional[str] = None


class ScrapingScheduler:
    """Scheduler for automated scraping jobs"""
    
    def __init__(
        self,
        property_repository: PostgresPropertyRepository,
        cache_repository: RedisCacheRepository,
        default_config: ScrapingConfig = None
    ):
        self.property_repository = property_repository
        self.cache_repository = cache_repository
        self.default_config = default_config or ScrapingConfig()
        
        # Job management
        self.jobs: Dict[str, ScheduledJob] = {}
        self.job_history: Dict[str, List[JobResult]] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        
        # Scheduler state
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()
        
        # Callbacks
        self.job_started_callbacks: List[Callable[[str], None]] = []
        self.job_completed_callbacks: List[Callable[[str, JobResult], None]] = []
        
    async def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start main scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Scraping scheduler started")
    
    async def stop(self):
        """Stop the scheduler and all running jobs"""
        if not self.is_running:
            return
        
        logger.info("Stopping scraping scheduler")
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop all running jobs
        for job_id, task in list(self.running_jobs.items()):
            logger.info(f"Cancelling running job: {job_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.running_jobs.clear()
        
        # Stop scheduler task
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Scraping scheduler stopped")
    
    def add_job(
        self,
        job_id: str,
        name: str,
        schedule_type: ScheduleType,
        interval_hours: float,
        scrapers: List[str],
        max_properties: Optional[int] = None,
        config_override: Optional[ScrapingConfig] = None,
        enabled: bool = True
    ) -> ScheduledJob:
        """Add a new scheduled job"""
        if job_id in self.jobs:
            raise ValueError(f"Job with ID {job_id} already exists")
        
        # Calculate next run time
        now = datetime.utcnow()
        next_run = self._calculate_next_run(now, schedule_type, interval_hours)
        
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            schedule_type=schedule_type,
            interval_hours=interval_hours,
            scrapers=scrapers,
            max_properties=max_properties,
            enabled=enabled,
            next_run=next_run,
            config_override=config_override
        )
        
        self.jobs[job_id] = job
        self.job_history[job_id] = []
        
        logger.info(f"Added scheduled job: {name} ({job_id}) - next run: {next_run}")
        return job
    
    def remove_job(self, job_id: str):
        """Remove a scheduled job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        
        # Cancel if currently running
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            del self.running_jobs[job_id]
        
        job_name = self.jobs[job_id].name
        del self.jobs[job_id]
        del self.job_history[job_id]
        
        logger.info(f"Removed scheduled job: {job_name} ({job_id})")
    
    def enable_job(self, job_id: str):
        """Enable a scheduled job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        
        self.jobs[job_id].enabled = True
        
        # Recalculate next run if not already set
        if not self.jobs[job_id].next_run:
            job = self.jobs[job_id]
            now = datetime.utcnow()
            job.next_run = self._calculate_next_run(now, job.schedule_type, job.interval_hours)
        
        logger.info(f"Enabled job: {job_id}")
    
    def disable_job(self, job_id: str):
        """Disable a scheduled job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        
        self.jobs[job_id].enabled = False
        self.jobs[job_id].next_run = None
        
        # Cancel if currently running
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            del self.running_jobs[job_id]
        
        logger.info(f"Disabled job: {job_id}")
    
    async def run_job_now(self, job_id: str) -> JobResult:
        """Run a job immediately, regardless of schedule"""
        if job_id not in self.jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        
        if job_id in self.running_jobs:
            raise ValueError(f"Job {job_id} is already running")
        
        job = self.jobs[job_id]
        logger.info(f"Running job immediately: {job.name} ({job_id})")
        
        # Create task for job execution
        task = asyncio.create_task(self._execute_job(job))
        self.running_jobs[job_id] = task
        
        try:
            result = await task
            return result
        finally:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status information for a job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        
        job = self.jobs[job_id]
        history = self.job_history[job_id]
        
        # Calculate success rate
        if job.total_runs > 0:
            success_rate = job.successful_runs / job.total_runs
        else:
            success_rate = 0.0
        
        # Get last result
        last_result = history[-1] if history else None
        
        return {
            'job_id': job_id,
            'name': job.name,
            'enabled': job.enabled,
            'schedule_type': job.schedule_type.value,
            'interval_hours': job.interval_hours,
            'scrapers': job.scrapers,
            'max_properties': job.max_properties,
            'last_run': job.last_run.isoformat() if job.last_run else None,
            'next_run': job.next_run.isoformat() if job.next_run else None,
            'total_runs': job.total_runs,
            'successful_runs': job.successful_runs,
            'success_rate': success_rate,
            'is_running': job_id in self.running_jobs,
            'last_result': {
                'success': last_result.success,
                'started_at': last_result.started_at.isoformat(),
                'completed_at': last_result.completed_at.isoformat(),
                'properties_found': last_result.stats.total_properties_found if last_result.stats else 0,
                'error_message': last_result.error_message
            } if last_result else None
        }
    
    def get_all_jobs_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all jobs"""
        return {
            job_id: self.get_job_status(job_id)
            for job_id in self.jobs
        }
    
    def add_job_callback(
        self,
        on_started: Optional[Callable[[str], None]] = None,
        on_completed: Optional[Callable[[str, JobResult], None]] = None
    ):
        """Add callbacks for job events"""
        if on_started:
            self.job_started_callbacks.append(on_started)
        if on_completed:
            self.job_completed_callbacks.append(on_completed)
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.is_running:
            try:
                # Check for jobs that need to run
                now = datetime.utcnow()
                
                for job_id, job in list(self.jobs.items()):
                    if (job.enabled and 
                        job.next_run and 
                        now >= job.next_run and 
                        job_id not in self.running_jobs):
                        
                        # Start job
                        logger.info(f"Starting scheduled job: {job.name} ({job_id})")
                        task = asyncio.create_task(self._execute_job(job))
                        self.running_jobs[job_id] = task
                        
                        # Don't await here - let it run in background
                        asyncio.create_task(self._handle_job_completion(job_id, task))
                
                # Wait before next check (check every minute)
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=60.0)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Continue checking for jobs
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
        
        logger.info("Scheduler loop stopped")
    
    async def _execute_job(self, job: ScheduledJob) -> JobResult:
        """Execute a single scraping job"""
        started_at = datetime.utcnow()
        
        # Notify callbacks
        for callback in self.job_started_callbacks:
            try:
                callback(job.job_id)
            except Exception as e:
                logger.warning(f"Job started callback failed: {e}")
        
        try:
            # Create orchestrator with job-specific config
            config = job.config_override or self.default_config
            
            async with ScrapingOrchestrator(
                self.property_repository,
                self.cache_repository,
                config
            ) as orchestrator:
                
                # Run scraping session
                stats = await orchestrator.run_full_scraping_session(
                    scraper_names=job.scrapers,
                    max_properties=job.max_properties
                )
                
                # Create successful result
                result = JobResult(
                    job_id=job.job_id,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    success=True,
                    stats=stats
                )
                
                # Update job tracking
                job.last_run = started_at
                job.total_runs += 1
                job.successful_runs += 1
                
                logger.info(
                    f"Job {job.job_id} completed successfully: "
                    f"{stats.total_properties_found} properties found"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            
            # Create failed result
            result = JobResult(
                job_id=job.job_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
            
            # Update job tracking
            job.last_run = started_at
            job.total_runs += 1
            
            return result
        
        finally:
            # Schedule next run
            if job.enabled:
                job.next_run = self._calculate_next_run(
                    datetime.utcnow(), 
                    job.schedule_type, 
                    job.interval_hours
                )
    
    async def _handle_job_completion(self, job_id: str, task: asyncio.Task):
        """Handle completion of a job task"""
        try:
            result = await task
            
            # Store result in history
            if job_id in self.job_history:
                self.job_history[job_id].append(result)
                
                # Keep only last 100 results
                if len(self.job_history[job_id]) > 100:
                    self.job_history[job_id] = self.job_history[job_id][-100:]
            
            # Notify callbacks
            for callback in self.job_completed_callbacks:
                try:
                    callback(job_id, result)
                except Exception as e:
                    logger.warning(f"Job completed callback failed: {e}")
                    
        except asyncio.CancelledError:
            logger.info(f"Job {job_id} was cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in job {job_id}: {e}")
        finally:
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def _calculate_next_run(
        self, 
        current_time: datetime, 
        schedule_type: ScheduleType, 
        interval_hours: float
    ) -> datetime:
        """Calculate the next run time for a job"""
        if schedule_type == ScheduleType.HOURLY:
            return current_time + timedelta(hours=interval_hours)
        
        elif schedule_type == ScheduleType.DAILY:
            # Run at the same time tomorrow
            next_day = current_time + timedelta(days=1)
            return next_day.replace(
                hour=current_time.hour,
                minute=current_time.minute,
                second=0,
                microsecond=0
            )
        
        elif schedule_type == ScheduleType.WEEKLY:
            # Run at the same time next week
            next_week = current_time + timedelta(weeks=1)
            return next_week.replace(
                hour=current_time.hour,
                minute=current_time.minute,
                second=0,
                microsecond=0
            )
        
        elif schedule_type == ScheduleType.CUSTOM:
            return current_time + timedelta(hours=interval_hours)
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def export_job_config(self, job_id: str) -> Dict[str, Any]:
        """Export job configuration as JSON-serializable dict"""
        if job_id not in self.jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        
        job = self.jobs[job_id]
        
        config_dict = {
            'job_id': job.job_id,
            'name': job.name,
            'schedule_type': job.schedule_type.value,
            'interval_hours': job.interval_hours,
            'scrapers': job.scrapers,
            'max_properties': job.max_properties,
            'enabled': job.enabled
        }
        
        if job.config_override:
            config_dict['config_override'] = {
                'max_concurrent_scrapers': job.config_override.max_concurrent_scrapers,
                'max_properties_per_scraper': job.config_override.max_properties_per_scraper,
                'deduplication_enabled': job.config_override.deduplication_enabled,
                'save_to_database': job.config_override.save_to_database,
                'global_rate_limit': job.config_override.global_rate_limit
            }
        
        return config_dict
    
    def import_job_config(self, config_dict: Dict[str, Any]) -> ScheduledJob:
        """Import job configuration from dict"""
        # Extract config override if present
        config_override = None
        if 'config_override' in config_dict:
            override_dict = config_dict['config_override']
            config_override = ScrapingConfig(
                max_concurrent_scrapers=override_dict.get('max_concurrent_scrapers', 3),
                max_properties_per_scraper=override_dict.get('max_properties_per_scraper', 1000),
                deduplication_enabled=override_dict.get('deduplication_enabled', True),
                save_to_database=override_dict.get('save_to_database', True),
                global_rate_limit=override_dict.get('global_rate_limit', 2.0)
            )
        
        return self.add_job(
            job_id=config_dict['job_id'],
            name=config_dict['name'],
            schedule_type=ScheduleType(config_dict['schedule_type']),
            interval_hours=config_dict['interval_hours'],
            scrapers=config_dict['scrapers'],
            max_properties=config_dict.get('max_properties'),
            config_override=config_override,
            enabled=config_dict.get('enabled', True)
        )