"""
Base scraper class for property data extraction.

This module provides the abstract base class for all property scrapers,
defining the common interface and shared functionality.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from urllib.parse import urljoin, urlparse

from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Configuration for scraping operations"""
    max_concurrent_requests: int = 10
    request_delay: float = 1.0
    max_retries: int = 3
    timeout_seconds: int = 30
    user_agent: str = "RentalMLSystem/1.0 (+https://github.com/rental-ml)"
    respect_robots_txt: bool = True
    max_pages_per_source: int = 100
    rate_limit_per_second: float = 1.0


@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    properties: List[Property]
    total_found: int
    pages_scraped: int
    errors: List[str]
    duration_seconds: float
    source_url: str
    scraped_at: datetime


class ScrapingError(Exception):
    """Custom exception for scraping operations"""
    pass


class RateLimitError(ScrapingError):
    """Exception raised when rate limits are exceeded"""
    pass


class BaseScraper(ABC):
    """Abstract base class for property scrapers"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(config.max_concurrent_requests)
        self.last_request_time = 0.0
        self.request_count = 0
        self.start_time = time.time()
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the scraper session"""
        timeout = ClientTimeout(total=self.config.timeout_seconds)
        headers = {
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        )
        
        logger.info(f"Initialized {self.__class__.__name__} scraper")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        logger.info(f"Cleaned up {self.__class__.__name__} scraper")
    
    async def rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.request_delay:
            sleep_time = self.config.request_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Check rate limit per second
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            current_rate = self.request_count / elapsed
            if current_rate > self.config.rate_limit_per_second:
                sleep_time = 1.0 / self.config.rate_limit_per_second
                await asyncio.sleep(sleep_time)
    
    async def fetch_page(self, url: str, retries: int = None) -> Optional[str]:
        """Fetch a single page with retries and rate limiting"""
        if retries is None:
            retries = self.config.max_retries
        
        async with self.rate_limiter:
            await self.rate_limit()
            
            for attempt in range(retries + 1):
                try:
                    logger.debug(f"Fetching {url} (attempt {attempt + 1})")
                    
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            logger.debug(f"Successfully fetched {url}")
                            return content
                        elif response.status == 429:  # Rate limited
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status >= 400:
                            logger.warning(f"HTTP {response.status} for {url}")
                            if attempt == retries:
                                return None
                            continue
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                except Exception as e:
                    logger.error(f"Error fetching {url}: {e}")
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
            
            logger.error(f"Failed to fetch {url} after {retries + 1} attempts")
            return None
    
    @abstractmethod
    async def get_listing_urls(self, base_url: str, max_pages: int = None) -> AsyncGenerator[str, None]:
        """Get URLs of individual property listings"""
        pass
    
    @abstractmethod
    async def extract_property_data(self, listing_url: str, html_content: str) -> Optional[Property]:
        """Extract property data from a listing page"""
        pass
    
    @abstractmethod
    def get_search_urls(self) -> List[str]:
        """Get list of search URLs to scrape"""
        pass
    
    async def scrape_properties(self, max_pages: int = None) -> ScrapingResult:
        """Main scraping method"""
        start_time = time.time()
        all_properties = []
        all_errors = []
        total_pages_scraped = 0
        
        try:
            search_urls = self.get_search_urls()
            if max_pages:
                max_pages_per_url = max(1, max_pages // len(search_urls))
            else:
                max_pages_per_url = self.config.max_pages_per_source
            
            logger.info(f"Starting scraping for {len(search_urls)} search URLs")
            
            for search_url in search_urls:
                try:
                    logger.info(f"Scraping search URL: {search_url}")
                    
                    # Get listing URLs
                    listing_urls = []
                    pages_for_this_url = 0
                    
                    async for listing_url in self.get_listing_urls(search_url, max_pages_per_url):
                        listing_urls.append(listing_url)
                        pages_for_this_url += 1
                        
                        if pages_for_this_url >= max_pages_per_url:
                            break
                    
                    logger.info(f"Found {len(listing_urls)} listings for {search_url}")
                    total_pages_scraped += pages_for_this_url
                    
                    # Process listings concurrently
                    semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
                    tasks = [
                        self._scrape_single_property(listing_url, semaphore)
                        for listing_url in listing_urls
                    ]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            error_msg = f"Error scraping {listing_urls[i]}: {result}"
                            logger.error(error_msg)
                            all_errors.append(error_msg)
                        elif result:
                            all_properties.append(result)
                
                except Exception as e:
                    error_msg = f"Error processing search URL {search_url}: {e}"
                    logger.error(error_msg)
                    all_errors.append(error_msg)
            
            duration = time.time() - start_time
            
            result = ScrapingResult(
                properties=all_properties,
                total_found=len(all_properties),
                pages_scraped=total_pages_scraped,
                errors=all_errors,
                duration_seconds=duration,
                source_url=search_urls[0] if search_urls else "",
                scraped_at=datetime.utcnow()
            )
            
            logger.info(
                f"Scraping completed: {len(all_properties)} properties found "
                f"in {duration:.2f}s with {len(all_errors)} errors"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in scraping: {e}")
            raise ScrapingError(f"Scraping failed: {e}")
    
    async def _scrape_single_property(self, listing_url: str, semaphore: asyncio.Semaphore) -> Optional[Property]:
        """Scrape a single property with concurrency control"""
        async with semaphore:
            try:
                html_content = await self.fetch_page(listing_url)
                if not html_content:
                    return None
                
                property_data = await self.extract_property_data(listing_url, html_content)
                if property_data:
                    logger.debug(f"Successfully extracted property from {listing_url}")
                    return property_data
                else:
                    logger.warning(f"No property data extracted from {listing_url}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error scraping single property {listing_url}: {e}")
                raise
    
    def validate_property_data(self, property_data: Dict[str, Any]) -> bool:
        """Validate extracted property data"""
        required_fields = ['title', 'price', 'location']
        
        for field in required_fields:
            if field not in property_data or not property_data[field]:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate price
        try:
            price = float(property_data['price'])
            if price <= 0 or price > 100000:  # Reasonable bounds
                logger.warning(f"Invalid price: {price}")
                return False
        except (ValueError, TypeError):
            logger.warning(f"Invalid price format: {property_data['price']}")
            return False
        
        return True
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common unwanted characters
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\r', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        return text.strip()
    
    def extract_price(self, price_text: str) -> Optional[float]:
        """Extract price from text"""
        if not price_text:
            return None
        
        import re
        
        # Remove currency symbols and common text
        price_text = re.sub(r'[^\d,.]', '', price_text)
        price_text = price_text.replace(',', '')
        
        try:
            return float(price_text)
        except ValueError:
            return None
    
    def normalize_location(self, location: str) -> str:
        """Normalize location text"""
        if not location:
            return ""
        
        # Clean up common location formatting
        location = self.clean_text(location)
        location = location.replace(' - ', ', ')
        location = location.replace(' | ', ', ')
        
        return location
    
    async def check_robots_txt(self, base_url: str) -> bool:
        """Check robots.txt compliance"""
        if not self.config.respect_robots_txt:
            return True
        
        try:
            parsed_url = urlparse(base_url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            robots_content = await self.fetch_page(robots_url)
            if not robots_content:
                return True  # No robots.txt found, assume allowed
            
            # Simple robots.txt parsing (basic implementation)
            user_agent_section = False
            for line in robots_content.split('\n'):
                line = line.strip().lower()
                
                if line.startswith('user-agent:'):
                    agent = line.split(':', 1)[1].strip()
                    user_agent_section = agent == '*' or self.config.user_agent.lower().startswith(agent)
                
                elif user_agent_section and line.startswith('disallow:'):
                    disallow_path = line.split(':', 1)[1].strip()
                    if disallow_path and parsed_url.path.startswith(disallow_path):
                        logger.warning(f"Robots.txt disallows scraping {base_url}")
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt: {e}")
            return True  # Assume allowed if we can't check