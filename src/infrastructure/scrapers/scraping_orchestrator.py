"""
Scraping orchestrator for coordinating multiple scrapers.

This module provides orchestration for running multiple scrapers concurrently,
managing the data pipeline, and handling error recovery.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .base_scraper import BaseScraper, ScrapingResult, ScrapingError
from .apartments_com_scraper import ApartmentsComScraper
from ...domain.entities.property import Property
from ..data.repositories.postgres_property_repository import PostgresPropertyRepository
from ..data.repositories.redis_cache_repository import RedisCacheRepository

logger = logging.getLogger(__name__)


@dataclass
class ScrapingStats:
    """Statistics for a scraping session"""
    total_properties_found: int
    total_properties_saved: int
    total_duplicates_filtered: int
    total_errors: int
    scrapers_used: List[str]
    duration_seconds: float
    started_at: datetime
    completed_at: datetime


@dataclass
class ScrapingConfig:
    """Configuration for scraping orchestration"""
    max_concurrent_scrapers: int = 3
    max_properties_per_scraper: int = 1000
    deduplication_enabled: bool = True
    cache_results: bool = True
    retry_failed_scrapers: bool = True
    save_to_database: bool = True
    
    # Rate limiting
    global_rate_limit: float = 2.0  # requests per second across all scrapers
    scraper_delay: float = 5.0  # delay between starting scrapers
    
    # Data quality
    min_property_fields: int = 3  # minimum required fields for valid property
    max_price_threshold: float = 50000.0  # maximum reasonable rent price
    min_price_threshold: float = 100.0  # minimum reasonable rent price


class ScrapingOrchestrator:
    """Orchestrates multiple property scrapers"""
    
    def __init__(
        self,
        property_repository: PostgresPropertyRepository,
        cache_repository: RedisCacheRepository,
        config: ScrapingConfig = None
    ):
        self.property_repository = property_repository
        self.cache_repository = cache_repository
        self.config = config or ScrapingConfig()
        
        # Initialize scrapers
        self.scrapers: Dict[str, BaseScraper] = {
            'apartments_com': ApartmentsComScraper()
        }
        
        # Tracking
        self.session_stats = {}
        self.active_scrapers = set()
        self.rate_limiter = asyncio.Semaphore(1)  # Global rate limiting
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize all scrapers
        for name, scraper in self.scrapers.items():
            await scraper.initialize()
        
        logger.info(f"Initialized {len(self.scrapers)} scrapers")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cleanup all scrapers
        for name, scraper in self.scrapers.items():
            await scraper.cleanup()
        
        logger.info("Cleaned up all scrapers")
    
    async def run_full_scraping_session(
        self,
        scraper_names: Optional[List[str]] = None,
        max_properties: Optional[int] = None
    ) -> ScrapingStats:
        """Run a complete scraping session with all specified scrapers"""
        start_time = time.time()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting scraping session {session_id}")
        
        # Default to all scrapers if none specified
        if scraper_names is None:
            scraper_names = list(self.scrapers.keys())
        
        # Validate scraper names
        invalid_scrapers = set(scraper_names) - set(self.scrapers.keys())
        if invalid_scrapers:
            raise ValueError(f"Invalid scrapers: {invalid_scrapers}")
        
        # Calculate max properties per scraper
        if max_properties:
            max_per_scraper = max_properties // len(scraper_names)
        else:
            max_per_scraper = self.config.max_properties_per_scraper
        
        try:
            # Run scrapers concurrently
            scraping_tasks = []
            
            for i, scraper_name in enumerate(scraper_names):
                # Stagger scraper starts to avoid overwhelming sources
                start_delay = i * self.config.scraper_delay
                
                task = asyncio.create_task(
                    self._run_single_scraper_with_delay(
                        scraper_name, max_per_scraper, start_delay, session_id
                    )
                )
                scraping_tasks.append(task)
            
            # Wait for all scrapers to complete
            scraper_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            # Process results and calculate stats
            all_properties = []
            all_errors = []
            scrapers_used = []
            
            for i, result in enumerate(scraper_results):
                scraper_name = scraper_names[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Scraper {scraper_name} failed: {result}")
                    all_errors.append(f"{scraper_name}: {result}")
                elif result:
                    all_properties.extend(result.properties)
                    all_errors.extend(result.errors)
                    scrapers_used.append(scraper_name)
                    
                    logger.info(
                        f"Scraper {scraper_name} completed: "
                        f"{len(result.properties)} properties, "
                        f"{len(result.errors)} errors"
                    )
            
            # Deduplicate properties
            if self.config.deduplication_enabled:
                unique_properties = await self._deduplicate_properties(all_properties)
                duplicates_filtered = len(all_properties) - len(unique_properties)
                all_properties = unique_properties
            else:
                duplicates_filtered = 0
            
            # Save to database
            properties_saved = 0
            if self.config.save_to_database and all_properties:
                properties_saved = await self._save_properties_to_database(all_properties)
            
            # Cache results
            if self.config.cache_results and all_properties:
                await self._cache_scraping_results(all_properties, session_id)
            
            # Calculate final stats
            end_time = time.time()
            stats = ScrapingStats(
                total_properties_found=len(all_properties),
                total_properties_saved=properties_saved,
                total_duplicates_filtered=duplicates_filtered,
                total_errors=len(all_errors),
                scrapers_used=scrapers_used,
                duration_seconds=end_time - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.fromtimestamp(end_time)
            )
            
            logger.info(
                f"Scraping session {session_id} completed: "
                f"{stats.total_properties_found} properties found, "
                f"{stats.total_properties_saved} saved, "
                f"{stats.total_duplicates_filtered} duplicates filtered, "
                f"{stats.total_errors} errors in {stats.duration_seconds:.2f}s"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Scraping session {session_id} failed: {e}")
            raise ScrapingError(f"Orchestration failed: {e}")
    
    async def _run_single_scraper_with_delay(
        self,
        scraper_name: str,
        max_properties: int,
        start_delay: float,
        session_id: str
    ) -> Optional[ScrapingResult]:
        """Run a single scraper with start delay"""
        if start_delay > 0:
            logger.info(f"Delaying {scraper_name} start by {start_delay}s")
            await asyncio.sleep(start_delay)
        
        scraper = self.scrapers[scraper_name]
        self.active_scrapers.add(scraper_name)
        
        try:
            logger.info(f"Starting scraper {scraper_name} (max {max_properties} properties)")
            
            # Calculate max pages based on estimated properties per page
            estimated_props_per_page = 25  # Conservative estimate
            max_pages = max(1, max_properties // estimated_props_per_page)
            
            result = await scraper.scrape_properties(max_pages=max_pages)
            
            # Store session stats
            self.session_stats[scraper_name] = {
                'session_id': session_id,
                'result': result,
                'completed_at': datetime.utcnow()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Scraper {scraper_name} failed: {e}")
            
            # Retry logic
            if self.config.retry_failed_scrapers:
                logger.info(f"Retrying scraper {scraper_name}")
                try:
                    await asyncio.sleep(10)  # Wait before retry
                    result = await scraper.scrape_properties(max_pages=max_pages)
                    return result
                except Exception as retry_error:
                    logger.error(f"Retry of {scraper_name} also failed: {retry_error}")
            
            raise
            
        finally:
            self.active_scrapers.discard(scraper_name)
    
    async def _deduplicate_properties(self, properties: List[Property]) -> List[Property]:
        """Remove duplicate properties based on multiple criteria"""
        if not properties:
            return []
        
        logger.info(f"Deduplicating {len(properties)} properties")
        
        seen_properties = {}
        unique_properties = []
        
        for prop in properties:
            # Create deduplication key based on multiple fields
            dedup_key = self._create_deduplication_key(prop)
            
            if dedup_key not in seen_properties:
                seen_properties[dedup_key] = prop
                unique_properties.append(prop)
            else:
                # Keep the property with more complete data
                existing_prop = seen_properties[dedup_key]
                if self._property_completeness_score(prop) > self._property_completeness_score(existing_prop):
                    # Replace with more complete property
                    unique_properties = [p for p in unique_properties if p.id != existing_prop.id]
                    unique_properties.append(prop)
                    seen_properties[dedup_key] = prop
        
        logger.info(f"Deduplication result: {len(unique_properties)} unique properties")
        return unique_properties
    
    def _create_deduplication_key(self, property: Property) -> str:
        """Create a key for deduplication based on property characteristics"""
        # Normalize location and title for comparison
        location = property.location.lower().strip() if property.location else ""
        title = property.title.lower().strip() if property.title else ""
        
        # Remove common variations
        location = location.replace(" apartment", "").replace(" apt", "")
        title = title.replace(" apartment", "").replace(" apt", "")
        
        # Create composite key
        key_parts = [
            location,
            str(property.bedrooms),
            str(property.bathrooms),
            str(int(property.price)) if property.price else "0"
        ]
        
        return "|".join(key_parts)
    
    def _property_completeness_score(self, property: Property) -> int:
        """Calculate a completeness score for a property"""
        score = 0
        
        # Required fields
        if property.title: score += 3
        if property.price: score += 3
        if property.location: score += 3
        
        # Optional but valuable fields
        if property.description: score += 2
        if property.bedrooms > 0: score += 1
        if property.bathrooms > 0: score += 1
        if property.square_feet: score += 1
        if property.amenities: score += len(property.amenities)
        if property.images: score += min(len(property.images), 5)
        if property.contact_info: score += len(property.contact_info)
        
        return score
    
    async def _save_properties_to_database(self, properties: List[Property]) -> int:
        """Save properties to the database"""
        if not properties:
            return 0
        
        logger.info(f"Saving {len(properties)} properties to database")
        
        saved_count = 0
        failed_count = 0
        
        # Process in batches to avoid overwhelming the database
        batch_size = 50
        
        for i in range(0, len(properties), batch_size):
            batch = properties[i:i + batch_size]
            
            try:
                # Filter out invalid properties
                valid_properties = [
                    prop for prop in batch 
                    if self._is_valid_property(prop)
                ]
                
                if valid_properties:
                    # Use batch create method
                    await self.property_repository.batch_create(valid_properties)
                    saved_count += len(valid_properties)
                    
                    logger.debug(f"Saved batch {i//batch_size + 1}: {len(valid_properties)} properties")
                
                failed_count += len(batch) - len(valid_properties)
                
            except Exception as e:
                logger.error(f"Failed to save property batch {i//batch_size + 1}: {e}")
                failed_count += len(batch)
        
        logger.info(f"Database save completed: {saved_count} saved, {failed_count} failed")
        return saved_count
    
    def _is_valid_property(self, property: Property) -> bool:
        """Validate a property meets minimum quality standards"""
        # Check required fields
        if not property.title or not property.price or not property.location:
            return False
        
        # Check price reasonableness
        if (property.price < self.config.min_price_threshold or 
            property.price > self.config.max_price_threshold):
            return False
        
        # Check minimum number of fields
        field_count = sum([
            bool(property.title),
            bool(property.description),
            bool(property.price),
            bool(property.location),
            bool(property.bedrooms),
            bool(property.bathrooms),
            bool(property.square_feet),
            bool(property.amenities),
            bool(property.contact_info),
            bool(property.images)
        ])
        
        return field_count >= self.config.min_property_fields
    
    async def _cache_scraping_results(self, properties: List[Property], session_id: str):
        """Cache scraping results for quick access"""
        if not properties:
            return
        
        try:
            # Cache property summaries
            property_summaries = [
                {
                    'id': str(prop.id),
                    'title': prop.title,
                    'price': prop.price,
                    'location': prop.location,
                    'bedrooms': prop.bedrooms,
                    'bathrooms': prop.bathrooms,
                    'scraped_at': prop.scraped_at.isoformat() if prop.scraped_at else None
                }
                for prop in properties
            ]
            
            # Cache with TTL
            cache_key = f"scraping_session:{session_id}"
            await self.cache_repository.cache_search_results(
                cache_key, property_summaries, ttl_hours=24
            )
            
            logger.info(f"Cached scraping results for session {session_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cache scraping results: {e}")
    
    async def get_scraping_status(self) -> Dict[str, Any]:
        """Get current scraping status"""
        return {
            'active_scrapers': list(self.active_scrapers),
            'total_scrapers': len(self.scrapers),
            'session_stats': {
                name: {
                    'session_id': stats['session_id'],
                    'properties_found': stats['result'].total_found,
                    'completed_at': stats['completed_at'].isoformat()
                }
                for name, stats in self.session_stats.items()
            },
            'config': {
                'max_concurrent_scrapers': self.config.max_concurrent_scrapers,
                'max_properties_per_scraper': self.config.max_properties_per_scraper,
                'deduplication_enabled': self.config.deduplication_enabled,
                'save_to_database': self.config.save_to_database
            }
        }
    
    async def stop_all_scrapers(self):
        """Emergency stop for all active scrapers"""
        logger.warning("Emergency stop requested for all scrapers")
        
        # Note: This is a basic implementation
        # In a production system, you'd want more sophisticated cancellation
        for scraper_name in list(self.active_scrapers):
            logger.warning(f"Stopping scraper {scraper_name}")
            # Individual scrapers would need cancellation support
        
        self.active_scrapers.clear()