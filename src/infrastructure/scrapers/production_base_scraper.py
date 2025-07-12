"""
Production-ready base scraper with enhanced error handling, rate limiting, and compliance.

This module provides an enhanced base scraper class with comprehensive production features
including robots.txt compliance, advanced rate limiting, circuit breakers, and monitoring.
"""

import asyncio
import aiohttp
import logging
import time
import random
import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, quote_plus
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET
from pathlib import Path
import re

from .config import ProductionScrapingConfig, get_config
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


@dataclass
class ScrapingMetrics:
    """Metrics for scraping performance tracking"""
    requests_made: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    requests_rate_limited: int = 0
    requests_blocked: int = 0
    avg_response_time: float = 0.0
    properties_extracted: int = 0
    properties_valid: int = 0
    properties_invalid: int = 0
    session_start_time: datetime = field(default_factory=datetime.now)
    last_request_time: Optional[datetime] = None
    
    def add_request(self, success: bool, response_time: float, rate_limited: bool = False, blocked: bool = False):
        """Add request metrics"""
        self.requests_made += 1
        self.last_request_time = datetime.now()
        
        if success:
            self.requests_successful += 1
        else:
            self.requests_failed += 1
        
        if rate_limited:
            self.requests_rate_limited += 1
        
        if blocked:
            self.requests_blocked += 1
        
        # Update average response time
        if self.requests_made > 1:
            self.avg_response_time = (
                (self.avg_response_time * (self.requests_made - 1) + response_time) / 
                self.requests_made
            )
        else:
            self.avg_response_time = response_time
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        if self.requests_made == 0:
            return 0.0
        return (self.requests_successful / self.requests_made) * 100
    
    def get_session_duration(self) -> timedelta:
        """Get session duration"""
        return datetime.now() - self.session_start_time


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for handling failures"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: int = 300  # 5 minutes
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def can_attempt_request(self) -> bool:
        """Check if request can be attempted"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = "HALF_OPEN"
                return True
            return False
        
        # HALF_OPEN state
        return True


class RobotsTxtParser:
    """Enhanced robots.txt parser with caching"""
    
    def __init__(self, config: ProductionScrapingConfig):
        self.config = config
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def can_fetch(self, url: str, user_agent: str, session: aiohttp.ClientSession) -> tuple[bool, float]:
        """
        Check if URL can be fetched according to robots.txt
        Returns (can_fetch, crawl_delay)
        """
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        # Check cache first
        cache_key = f"{robots_url}:{user_agent}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['can_fetch'], cached['crawl_delay']
        
        try:
            can_fetch, crawl_delay = await self._fetch_and_parse_robots(
                robots_url, url, user_agent, session
            )
            
            # Cache result
            self.cache[cache_key] = {
                'can_fetch': can_fetch,
                'crawl_delay': crawl_delay,
                'timestamp': time.time()
            }
            
            return can_fetch, crawl_delay
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {robots_url}: {e}")
            # Default to allowing with conservative crawl delay
            return True, self.config.rate_limit.default_crawl_delay
    
    async def _fetch_and_parse_robots(
        self, 
        robots_url: str, 
        target_url: str, 
        user_agent: str,
        session: aiohttp.ClientSession
    ) -> tuple[bool, float]:
        """Fetch and parse robots.txt"""
        
        try:
            async with session.get(
                robots_url, 
                timeout=aiohttp.ClientTimeout(total=self.config.robots_txt.robots_txt_timeout)
            ) as response:
                
                if response.status != 200:
                    # No robots.txt or error - allow with default delay
                    return True, self.config.rate_limit.default_crawl_delay
                
                robots_content = await response.text()
                
                # Parse robots.txt content
                return self._parse_robots_content(robots_content, target_url, user_agent)
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching robots.txt from {robots_url}")
            return True, self.config.rate_limit.default_crawl_delay
        except Exception as e:
            logger.warning(f"Error fetching robots.txt from {robots_url}: {e}")
            return True, self.config.rate_limit.default_crawl_delay
    
    def _parse_robots_content(self, content: str, target_url: str, user_agent: str) -> tuple[bool, float]:
        """Parse robots.txt content"""
        
        # Use Python's robotparser for comprehensive parsing
        rp = RobotFileParser()
        rp.set_url("dummy")  # URL not used for read_lines
        rp.read_lines(content.splitlines())
        
        can_fetch = rp.can_fetch(user_agent, target_url)
        
        # Extract crawl delay
        crawl_delay = self.config.rate_limit.default_crawl_delay
        
        # Manual parsing for crawl-delay (not supported by robotparser)
        current_user_agent = None
        for line in content.splitlines():
            line = line.strip()
            
            if line.lower().startswith('user-agent:'):
                current_user_agent = line.split(':', 1)[1].strip().lower()
            
            elif (line.lower().startswith('crawl-delay:') and 
                  current_user_agent in ['*', user_agent.lower()]):
                try:
                    crawl_delay = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
        
        return can_fetch, crawl_delay


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(self, config: ProductionScrapingConfig):
        self.config = config
        self.request_times: List[float] = []
        self.bucket_tokens = config.rate_limit.burst_size
        self.last_refill = time.time()
        self.backoff_until = 0.0
        
    async def acquire(self) -> bool:
        """Acquire permission to make a request"""
        current_time = time.time()
        
        # Check if we're in backoff period
        if current_time < self.backoff_until:
            wait_time = self.backoff_until - current_time
            await asyncio.sleep(wait_time)
        
        # Token bucket algorithm
        await self._refill_bucket()
        
        if self.bucket_tokens > 0:
            self.bucket_tokens -= 1
            
            # Sliding window rate limiting
            self._cleanup_old_requests()
            self.request_times.append(current_time)
            
            # Check if we exceed rate limits
            if self._exceeds_rate_limits():
                await self._apply_backoff()
                return False
            
            return True
        else:
            # No tokens available, wait
            wait_time = 1.0 / self.config.rate_limit.requests_per_second
            await asyncio.sleep(wait_time)
            return await self.acquire()
    
    async def _refill_bucket(self):
        """Refill token bucket"""
        current_time = time.time()
        time_passed = current_time - self.last_refill
        
        tokens_to_add = time_passed * self.config.rate_limit.requests_per_second
        self.bucket_tokens = min(
            self.config.rate_limit.burst_size,
            self.bucket_tokens + tokens_to_add
        )
        
        self.last_refill = current_time
    
    def _cleanup_old_requests(self):
        """Remove old request timestamps"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep last hour
        
        self.request_times = [t for t in self.request_times if t > cutoff_time]
    
    def _exceeds_rate_limits(self) -> bool:
        """Check if current request rate exceeds limits"""
        current_time = time.time()
        
        # Check requests per second
        recent_requests = [t for t in self.request_times if t > current_time - 1]
        if len(recent_requests) > self.config.rate_limit.requests_per_second:
            return True
        
        # Check requests per minute
        minute_requests = [t for t in self.request_times if t > current_time - 60]
        if len(minute_requests) > self.config.rate_limit.requests_per_minute:
            return True
        
        # Check requests per hour
        hour_requests = [t for t in self.request_times if t > current_time - 3600]
        if len(hour_requests) > self.config.rate_limit.requests_per_hour:
            return True
        
        return False
    
    async def _apply_backoff(self):
        """Apply exponential backoff"""
        backoff_time = min(
            self.config.rate_limit.max_backoff_seconds,
            1.0 * (self.config.rate_limit.backoff_factor ** len([
                t for t in self.request_times if t > time.time() - 300
            ]))
        )
        
        self.backoff_until = time.time() + backoff_time
        logger.warning(f"Rate limit exceeded, backing off for {backoff_time:.2f} seconds")


class ProductionBaseScraper(ABC):
    """Production-ready base scraper with comprehensive features"""
    
    def __init__(self, source_name: str, config: ProductionScrapingConfig = None):
        self.source_name = source_name
        self.config = config or get_config()
        self.source_config = self.config.get_source_config(source_name)
        
        if not self.source_config:
            raise ValueError(f"No configuration found for source: {source_name}")
        
        # Initialize components
        self.metrics = ScrapingMetrics()
        self.circuit_breaker = CircuitBreakerState()
        self.rate_limiter = AdvancedRateLimiter(self.config)
        self.robots_parser = RobotsTxtParser(self.config)
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.session_id = self._generate_session_id()
        self.user_agent_index = 0
        
        # Request queue for managing concurrent requests
        self.request_semaphore = asyncio.Semaphore(self.config.scraping.max_concurrent_requests)
        
        logger.info(f"Initialized {source_name} scraper with session {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(int(time.time()))
        random_part = str(random.randint(1000, 9999))
        return f"{self.source_name}_{timestamp}_{random_part}"
    
    def _get_user_agent(self) -> str:
        """Get user agent with rotation if enabled"""
        if self.config.scraping.rotate_user_agents:
            user_agents = self.config.scraping.user_agents
            self.user_agent_index = (self.user_agent_index + 1) % len(user_agents)
            return user_agents[self.user_agent_index]
        else:
            return self.config.scraping.user_agents[0]
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the scraper session"""
        timeout = aiohttp.ClientTimeout(total=self.config.scraping.request_timeout)
        
        # Build headers
        headers = {
            'User-Agent': self._get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Add custom headers from source config
        headers.update(self.source_config.custom_headers)
        
        # Configure connector
        connector = aiohttp.TCPConnector(
            limit=self.config.scraping.max_concurrent_requests,
            limit_per_host=self.config.scraping.max_concurrent_requests,
            ttl_dns_cache=300,  # 5 minutes DNS cache
            use_dns_cache=True,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=connector,
            trust_env=True  # Respect proxy environment variables
        )
        
        logger.info(f"Initialized session for {self.source_name}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        # Log final metrics
        logger.info(
            f"Session {self.session_id} completed - "
            f"Requests: {self.metrics.requests_made}, "
            f"Success rate: {self.metrics.get_success_rate():.1f}%, "
            f"Properties: {self.metrics.properties_valid}, "
            f"Duration: {self.metrics.get_session_duration()}"
        )
    
    async def fetch_page(self, url: str, retries: int = None) -> Optional[str]:
        """Fetch a single page with comprehensive error handling"""
        if retries is None:
            retries = self.config.scraping.max_retries
        
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt_request():
            logger.warning(f"Circuit breaker open for {self.source_name}, skipping {url}")
            return None
        
        async with self.request_semaphore:
            start_time = time.time()
            
            try:
                # Rate limiting
                await self.rate_limiter.acquire()
                
                # Check robots.txt
                user_agent = self.session.headers.get('User-Agent', '')
                can_fetch, crawl_delay = await self.robots_parser.can_fetch(url, user_agent, self.session)
                
                if not can_fetch:
                    logger.warning(f"Robots.txt disallows fetching {url}")
                    self.metrics.add_request(False, 0, blocked=True)
                    return None
                
                # Apply additional crawl delay if specified
                if crawl_delay > 0:
                    await asyncio.sleep(crawl_delay)
                
                # Attempt to fetch with retries
                for attempt in range(retries + 1):
                    try:
                        response_time = time.time()
                        
                        async with self.session.get(url) as response:
                            request_time = time.time() - response_time
                            
                            if response.status == 200:
                                content = await response.text()
                                self.metrics.add_request(True, request_time)
                                self.circuit_breaker.record_success()
                                
                                logger.debug(f"Successfully fetched {url} in {request_time:.2f}s")
                                return content
                            
                            elif response.status == 429:  # Rate limited
                                self.metrics.add_request(False, request_time, rate_limited=True)
                                
                                # Respect Retry-After header
                                retry_after = response.headers.get('Retry-After')
                                if retry_after and self.config.rate_limit.retry_after_respect:
                                    try:
                                        wait_time = int(retry_after)
                                        logger.warning(f"Rate limited, waiting {wait_time}s as requested")
                                        await asyncio.sleep(wait_time)
                                    except ValueError:
                                        # Retry-After might be a date
                                        pass
                                else:
                                    # Exponential backoff
                                    wait_time = (2 ** attempt) * self.config.scraping.retry_delay_base
                                    await asyncio.sleep(wait_time)
                                
                                continue
                            
                            elif 400 <= response.status < 500:
                                # Client error - don't retry
                                self.metrics.add_request(False, request_time)
                                logger.warning(f"Client error {response.status} for {url}")
                                return None
                            
                            elif response.status >= 500:
                                # Server error - retry
                                self.metrics.add_request(False, request_time)
                                if attempt < retries:
                                    wait_time = (2 ** attempt) * self.config.scraping.retry_delay_base
                                    logger.warning(f"Server error {response.status}, retrying in {wait_time}s")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    logger.error(f"Server error {response.status} for {url}, max retries exceeded")
                                    return None
                    
                    except asyncio.TimeoutError:
                        self.metrics.add_request(False, time.time() - start_time)
                        if attempt < retries:
                            wait_time = (2 ** attempt) * self.config.scraping.retry_delay_base
                            logger.warning(f"Timeout fetching {url}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Timeout fetching {url}, max retries exceeded")
                            break
                    
                    except Exception as e:
                        self.metrics.add_request(False, time.time() - start_time)
                        if attempt < retries:
                            wait_time = (2 ** attempt) * self.config.scraping.retry_delay_base
                            logger.warning(f"Error fetching {url}: {e}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Error fetching {url}: {e}, max retries exceeded")
                            break
                
                # All retries failed
                self.circuit_breaker.record_failure()
                return None
                
            except Exception as e:
                self.metrics.add_request(False, time.time() - start_time)
                self.circuit_breaker.record_failure()
                logger.error(f"Critical error fetching {url}: {e}")
                return None
    
    def validate_property_data(self, property_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate extracted property data against quality standards
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        for field in self.config.data_quality.required_fields:
            if field not in property_data or not property_data[field]:
                issues.append(f"Missing required field: {field}")
        
        # Validate price
        try:
            price = float(property_data.get('price', 0))
            if price < self.config.data_quality.min_price_threshold:
                issues.append(f"Price too low: ${price}")
            elif price > self.config.data_quality.max_price_threshold:
                issues.append(f"Price too high: ${price}")
        except (ValueError, TypeError):
            issues.append(f"Invalid price format: {property_data.get('price')}")
        
        # Validate title
        title = property_data.get('title', '')
        if len(title) < self.config.data_quality.min_title_length:
            issues.append(f"Title too short: {len(title)} chars")
        elif len(title) > self.config.data_quality.max_title_length:
            issues.append(f"Title too long: {len(title)} chars")
        
        # Validate description
        description = property_data.get('description', '')
        if len(description) < self.config.data_quality.min_description_length:
            issues.append(f"Description too short: {len(description)} chars")
        elif len(description) > self.config.data_quality.max_description_length:
            issues.append(f"Description too long: {len(description)} chars")
        
        # Validate bedrooms/bathrooms
        bedrooms = property_data.get('bedrooms', 0)
        if isinstance(bedrooms, (int, float)) and (bedrooms < 0 or bedrooms > self.config.data_quality.max_bedrooms):
            issues.append(f"Invalid bedrooms: {bedrooms}")
        
        bathrooms = property_data.get('bathrooms', 0)
        if isinstance(bathrooms, (int, float)) and (bathrooms < 0 or bathrooms > self.config.data_quality.max_bathrooms):
            issues.append(f"Invalid bathrooms: {bathrooms}")
        
        # Validate coordinates if present
        if self.config.data_quality.validate_coordinates:
            lat = property_data.get('latitude')
            lng = property_data.get('longitude')
            if lat is not None and lng is not None:
                try:
                    lat_f, lng_f = float(lat), float(lng)
                    if not (-90 <= lat_f <= 90) or not (-180 <= lng_f <= 180):
                        issues.append(f"Invalid coordinates: {lat}, {lng}")
                except (ValueError, TypeError):
                    issues.append(f"Invalid coordinate format: {lat}, {lng}")
        
        # Count valid fields for minimum threshold check
        valid_field_count = sum(1 for field in [
            'title', 'description', 'price', 'location', 'bedrooms', 'bathrooms',
            'square_feet', 'amenities', 'contact_info', 'images'
        ] if property_data.get(field))
        
        if valid_field_count < self.config.data_quality.min_required_fields:
            issues.append(f"Insufficient data fields: {valid_field_count} < {self.config.data_quality.min_required_fields}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove unwanted characters
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\r', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        # Remove suspicious patterns (potential bot detection evasion)
        text = re.sub(r'\b[A-Z]{10,}\b', '', text)  # Remove long uppercase strings
        
        return text.strip()
    
    def extract_price(self, price_text: str) -> Optional[float]:
        """Enhanced price extraction"""
        if not price_text:
            return None
        
        # Remove currency symbols and normalize
        price_text = re.sub(r'[^\d,.]', '', price_text)
        price_text = price_text.replace(',', '')
        
        # Handle ranges (take the lower value)
        if '-' in price_text:
            parts = price_text.split('-')
            if parts:
                price_text = parts[0].strip()
        
        try:
            price = float(price_text)
            # Sanity check
            if self.config.data_quality.min_price_threshold <= price <= self.config.data_quality.max_price_threshold:
                return price
        except ValueError:
            pass
        
        return None
    
    def calculate_data_quality_score(self, property_data: Dict[str, Any]) -> float:
        """Calculate a data quality score (0.0 to 1.0)"""
        score = 0.0
        max_score = 0.0
        
        # Required fields (high weight)
        for field in self.config.data_quality.required_fields:
            max_score += 3
            if property_data.get(field):
                score += 3
        
        # Optional fields (medium weight)
        optional_fields = ['description', 'square_feet', 'amenities', 'images', 'contact_info']
        for field in optional_fields:
            max_score += 2
            value = property_data.get(field)
            if value:
                if isinstance(value, (list, dict)) and len(value) > 0:
                    score += 2
                elif isinstance(value, str) and len(value) > 10:
                    score += 2
                elif isinstance(value, (int, float)) and value > 0:
                    score += 2
        
        # Quality indicators (low weight)
        max_score += 3
        
        # Title quality
        title = property_data.get('title', '')
        if len(title) >= self.config.data_quality.min_title_length:
            score += 1
        
        # Description quality
        description = property_data.get('description', '')
        if len(description) >= self.config.data_quality.min_description_length:
            score += 1
        
        # Contact info presence
        contact_info = property_data.get('contact_info', {})
        if isinstance(contact_info, dict) and len(contact_info) > 0:
            score += 1
        
        return min(1.0, score / max_score) if max_score > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current scraping metrics"""
        return {
            'session_id': self.session_id,
            'source_name': self.source_name,
            'requests_made': self.metrics.requests_made,
            'requests_successful': self.metrics.requests_successful,
            'requests_failed': self.metrics.requests_failed,
            'requests_rate_limited': self.metrics.requests_rate_limited,
            'requests_blocked': self.metrics.requests_blocked,
            'success_rate': self.metrics.get_success_rate(),
            'avg_response_time': self.metrics.avg_response_time,
            'properties_extracted': self.metrics.properties_extracted,
            'properties_valid': self.metrics.properties_valid,
            'properties_invalid': self.metrics.properties_invalid,
            'session_duration': str(self.metrics.get_session_duration()),
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failures': self.circuit_breaker.failure_count
        }
    
    # Abstract methods that subclasses must implement
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