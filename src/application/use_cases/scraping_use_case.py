"""
Use cases for property scraping operations.

This module provides the application layer interface for scraping operations,
including manual scraping, scheduled scraping, and scraping management.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...domain.entities.property import Property
from ...infrastructure.scrapers.scraping_orchestrator import (
    ScrapingOrchestrator, ScrapingStats, ScrapingConfig
)
from ...infrastructure.scrapers.scraping_scheduler import (
    ScrapingScheduler, ScheduledJob, ScheduleType, JobResult
)
from ...infrastructure.data.repositories.postgres_property_repository import PostgresPropertyRepository
from ...infrastructure.data.repositories.redis_cache_repository import RedisCacheRepository

logger = logging.getLogger(__name__)


class ScrapingUseCase:
    """Use case for property scraping operations"""
    
    def __init__(
        self,
        property_repository: PostgresPropertyRepository,
        cache_repository: RedisCacheRepository
    ):
        self.property_repository = property_repository
        self.cache_repository = cache_repository
        
        # Initialize components
        self.default_config = ScrapingConfig(
            max_concurrent_scrapers=2,
            max_properties_per_scraper=500,
            deduplication_enabled=True,
            cache_results=True,
            save_to_database=True,
            global_rate_limit=1.0,  # Conservative rate limiting
            scraper_delay=10.0  # 10 second delay between scrapers
        )
        
        self.scheduler: Optional[ScrapingScheduler] = None
        self._scheduler_initialized = False
    
    async def run_manual_scraping(
        self,
        scrapers: Optional[List[str]] = None,
        max_properties: Optional[int] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> ScrapingStats:
        """
        Run a manual scraping session.
        
        Args:
            scrapers: List of scraper names to use (defaults to all available)
            max_properties: Maximum total properties to scrape
            config_override: Dictionary of configuration overrides
            
        Returns:
            ScrapingStats: Statistics from the scraping session
        """
        logger.info("Starting manual scraping session")
        
        # Apply configuration overrides
        config = self._create_config_with_overrides(config_override)
        
        try:
            async with ScrapingOrchestrator(
                self.property_repository,
                self.cache_repository,
                config
            ) as orchestrator:
                
                stats = await orchestrator.run_full_scraping_session(
                    scraper_names=scrapers,
                    max_properties=max_properties
                )
                
                logger.info(
                    f"Manual scraping completed: "
                    f"{stats.total_properties_found} properties found, "
                    f"{stats.total_properties_saved} saved"
                )
                
                return stats
                
        except Exception as e:
            logger.error(f"Manual scraping failed: {e}")
            raise
    
    async def initialize_scheduler(self) -> bool:
        """
        Initialize the scraping scheduler.
        
        Returns:
            bool: True if successfully initialized
        """
        if self._scheduler_initialized:
            logger.warning("Scheduler already initialized")
            return True
        
        try:
            self.scheduler = ScrapingScheduler(
                self.property_repository,
                self.cache_repository,
                self.default_config
            )
            
            # Add default jobs
            await self._setup_default_scheduled_jobs()
            
            # Start scheduler
            await self.scheduler.start()
            
            self._scheduler_initialized = True
            logger.info("Scraping scheduler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            return False
    
    async def shutdown_scheduler(self):
        """Shutdown the scraping scheduler"""
        if self.scheduler and self._scheduler_initialized:
            await self.scheduler.stop()
            self.scheduler = None
            self._scheduler_initialized = False
            logger.info("Scraping scheduler shutdown")
    
    async def add_scheduled_job(
        self,
        job_id: str,
        name: str,
        schedule_type: str,
        interval_hours: float,
        scrapers: List[str],
        max_properties: Optional[int] = None,
        config_override: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Add a new scheduled scraping job.
        
        Args:
            job_id: Unique identifier for the job
            name: Human-readable name for the job
            schedule_type: Type of schedule ('hourly', 'daily', 'weekly', 'custom')
            interval_hours: Interval in hours for custom schedules
            scrapers: List of scraper names to use
            max_properties: Maximum properties to scrape per run
            config_override: Configuration overrides for this job
            enabled: Whether the job should be enabled
            
        Returns:
            Dict containing job status information
        """
        await self._ensure_scheduler_initialized()
        
        try:
            # Convert string to enum
            schedule_enum = ScheduleType(schedule_type.lower())
            
            # Create scraping config if overrides provided
            scraping_config = None
            if config_override:
                scraping_config = self._create_config_with_overrides(config_override)
            
            job = self.scheduler.add_job(
                job_id=job_id,
                name=name,
                schedule_type=schedule_enum,
                interval_hours=interval_hours,
                scrapers=scrapers,
                max_properties=max_properties,
                config_override=scraping_config,
                enabled=enabled
            )
            
            logger.info(f"Added scheduled job: {name} ({job_id})")
            return self.scheduler.get_job_status(job_id)
            
        except Exception as e:
            logger.error(f"Failed to add scheduled job {job_id}: {e}")
            raise
    
    async def remove_scheduled_job(self, job_id: str) -> bool:
        """
        Remove a scheduled scraping job.
        
        Args:
            job_id: ID of the job to remove
            
        Returns:
            bool: True if successfully removed
        """
        await self._ensure_scheduler_initialized()
        
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed scheduled job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove scheduled job {job_id}: {e}")
            return False
    
    async def enable_scheduled_job(self, job_id: str) -> bool:
        """
        Enable a scheduled scraping job.
        
        Args:
            job_id: ID of the job to enable
            
        Returns:
            bool: True if successfully enabled
        """
        await self._ensure_scheduler_initialized()
        
        try:
            self.scheduler.enable_job(job_id)
            logger.info(f"Enabled scheduled job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable scheduled job {job_id}: {e}")
            return False
    
    async def disable_scheduled_job(self, job_id: str) -> bool:
        """
        Disable a scheduled scraping job.
        
        Args:
            job_id: ID of the job to disable
            
        Returns:
            bool: True if successfully disabled
        """
        await self._ensure_scheduler_initialized()
        
        try:
            self.scheduler.disable_job(job_id)
            logger.info(f"Disabled scheduled job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable scheduled job {job_id}: {e}")
            return False
    
    async def run_scheduled_job_now(self, job_id: str) -> Dict[str, Any]:
        """
        Run a scheduled job immediately.
        
        Args:
            job_id: ID of the job to run
            
        Returns:
            Dict containing job result information
        """
        await self._ensure_scheduler_initialized()
        
        try:
            result = await self.scheduler.run_job_now(job_id)
            
            logger.info(f"Executed scheduled job {job_id}: success={result.success}")
            
            return {
                'job_id': result.job_id,
                'success': result.success,
                'started_at': result.started_at.isoformat(),
                'completed_at': result.completed_at.isoformat(),
                'error_message': result.error_message,
                'stats': {
                    'properties_found': result.stats.total_properties_found,
                    'properties_saved': result.stats.total_properties_saved,
                    'duplicates_filtered': result.stats.total_duplicates_filtered,
                    'errors': result.stats.total_errors,
                    'duration_seconds': result.stats.duration_seconds,
                    'scrapers_used': result.stats.scrapers_used
                } if result.stats else None
            }
            
        except Exception as e:
            logger.error(f"Failed to run scheduled job {job_id}: {e}")
            raise
    
    async def get_scraping_status(self) -> Dict[str, Any]:
        """
        Get comprehensive scraping system status.
        
        Returns:
            Dict containing system status information
        """
        status = {
            'scheduler_initialized': self._scheduler_initialized,
            'scheduled_jobs': {},
            'recent_activity': {}
        }
        
        if self._scheduler_initialized and self.scheduler:
            # Get all jobs status
            status['scheduled_jobs'] = self.scheduler.get_all_jobs_status()
            
            # Get recent scraping activity from cache
            try:
                recent_sessions = await self._get_recent_scraping_sessions()
                status['recent_activity'] = recent_sessions
            except Exception as e:
                logger.warning(f"Failed to get recent activity: {e}")
                status['recent_activity'] = {}
        
        return status
    
    async def get_job_history(self, job_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get execution history for a scheduled job.
        
        Args:
            job_id: ID of the job
            limit: Maximum number of history entries to return
            
        Returns:
            List of job execution history entries
        """
        await self._ensure_scheduler_initialized()
        
        if job_id not in self.scheduler.job_history:
            return []
        
        history = self.scheduler.job_history[job_id][-limit:]
        
        return [
            {
                'started_at': result.started_at.isoformat(),
                'completed_at': result.completed_at.isoformat(),
                'success': result.success,
                'error_message': result.error_message,
                'stats': {
                    'properties_found': result.stats.total_properties_found,
                    'properties_saved': result.stats.total_properties_saved,
                    'duration_seconds': result.stats.duration_seconds
                } if result.stats else None
            }
            for result in history
        ]
    
    async def get_available_scrapers(self) -> List[Dict[str, Any]]:
        """
        Get list of available scrapers and their capabilities.
        
        Returns:
            List of scraper information
        """
        # This would typically come from a registry or configuration
        return [
            {
                'name': 'apartments_com',
                'display_name': 'Apartments.com',
                'description': 'Scrapes apartment listings from apartments.com',
                'supported_locations': [
                    'new-york-ny', 'los-angeles-ca', 'chicago-il', 'houston-tx',
                    'phoenix-az', 'philadelphia-pa', 'san-antonio-tx', 'san-diego-ca',
                    'dallas-tx', 'san-jose-ca'
                ],
                'rate_limit': '0.5 requests/second',
                'estimated_properties_per_run': '100-500'
            }
        ]
    
    def _create_config_with_overrides(self, overrides: Optional[Dict[str, Any]]) -> ScrapingConfig:
        """Create a scraping config with overrides applied"""
        if not overrides:
            return self.default_config
        
        return ScrapingConfig(
            max_concurrent_scrapers=overrides.get(
                'max_concurrent_scrapers', 
                self.default_config.max_concurrent_scrapers
            ),
            max_properties_per_scraper=overrides.get(
                'max_properties_per_scraper',
                self.default_config.max_properties_per_scraper
            ),
            deduplication_enabled=overrides.get(
                'deduplication_enabled',
                self.default_config.deduplication_enabled
            ),
            cache_results=overrides.get(
                'cache_results',
                self.default_config.cache_results
            ),
            save_to_database=overrides.get(
                'save_to_database',
                self.default_config.save_to_database
            ),
            global_rate_limit=overrides.get(
                'global_rate_limit',
                self.default_config.global_rate_limit
            ),
            scraper_delay=overrides.get(
                'scraper_delay',
                self.default_config.scraper_delay
            )
        )
    
    async def _ensure_scheduler_initialized(self):
        """Ensure the scheduler is initialized"""
        if not self._scheduler_initialized:
            success = await self.initialize_scheduler()
            if not success:
                raise RuntimeError("Failed to initialize scraping scheduler")
    
    async def _setup_default_scheduled_jobs(self):
        """Set up default scheduled scraping jobs"""
        try:
            # Daily scraping job
            self.scheduler.add_job(
                job_id="daily_scraping",
                name="Daily Property Scraping",
                schedule_type=ScheduleType.DAILY,
                interval_hours=24.0,
                scrapers=["apartments_com"],
                max_properties=1000,
                enabled=True
            )
            
            # Weekly comprehensive scraping
            self.scheduler.add_job(
                job_id="weekly_comprehensive",
                name="Weekly Comprehensive Scraping",
                schedule_type=ScheduleType.WEEKLY,
                interval_hours=168.0,  # 7 days
                scrapers=["apartments_com"],
                max_properties=5000,
                config_override=ScrapingConfig(
                    max_concurrent_scrapers=1,
                    max_properties_per_scraper=5000,
                    global_rate_limit=0.5  # Slower rate for comprehensive scraping
                ),
                enabled=False  # Disabled by default
            )
            
            logger.info("Default scheduled jobs created")
            
        except Exception as e:
            logger.warning(f"Failed to create some default jobs: {e}")
    
    async def _get_recent_scraping_sessions(self) -> Dict[str, Any]:
        """Get recent scraping session data from cache"""
        try:
            # This would typically query the cache for recent session data
            # For now, return a placeholder
            return {
                'sessions_last_24h': 0,
                'properties_scraped_last_24h': 0,
                'last_session_time': None,
                'average_session_duration': 0.0
            }
            
        except Exception as e:
            logger.warning(f"Failed to get recent sessions: {e}")
            return {}