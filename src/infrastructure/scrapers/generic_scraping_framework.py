"""
Generic scraping framework for easy addition of new rental property sources.

This module provides a configuration-driven framework that allows adding new
property sources without writing custom scraper code, using selectors and rules.
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from urllib.parse import urljoin, urlparse
from uuid import uuid4
import yaml

from bs4 import BeautifulSoup
from .production_base_scraper import ProductionBaseScraper
from .config import get_config, ScraperSourceConfig
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


class ExtractionRule:
    """Represents a data extraction rule"""
    
    def __init__(self, config: Dict[str, Any]):
        self.field_name = config['field']
        self.selectors = config['selectors'] if isinstance(config['selectors'], list) else [config['selectors']]
        self.extraction_type = config.get('type', 'text')  # text, attribute, regex, json
        self.attribute = config.get('attribute')
        self.regex_pattern = config.get('regex')
        self.multiple = config.get('multiple', False)
        self.required = config.get('required', False)
        self.default_value = config.get('default')
        self.transformation = config.get('transformation')  # clean, lower, upper, strip, etc.
        self.validation = config.get('validation')  # regex pattern for validation
    
    def extract(self, soup: BeautifulSoup, base_url: str = None) -> Any:
        """Extract data using this rule"""
        
        extracted_values = []
        
        for selector in self.selectors:
            elements = soup.select(selector)
            
            for element in elements:
                value = self._extract_from_element(element, base_url)
                
                if value:
                    if self.multiple:
                        if isinstance(value, list):
                            extracted_values.extend(value)
                        else:
                            extracted_values.append(value)
                    else:
                        # For non-multiple, return first valid value
                        return self._transform_and_validate(value)
            
            # If we found values with this selector and it's not multiple, break
            if not self.multiple and extracted_values:
                break
        
        if self.multiple:
            return self._transform_and_validate(extracted_values)
        
        if not extracted_values:
            return self.default_value
        
        return self._transform_and_validate(extracted_values[0])
    
    def _extract_from_element(self, element, base_url: str = None) -> Any:
        """Extract value from a BeautifulSoup element"""
        
        if self.extraction_type == 'text':
            return element.get_text().strip()
        
        elif self.extraction_type == 'attribute':
            if self.attribute:
                value = element.get(self.attribute)
                
                # Handle URLs
                if self.attribute in ['src', 'href'] and base_url and value:
                    if value.startswith('//'):
                        value = 'https:' + value
                    elif value.startswith('/'):
                        value = urljoin(base_url, value)
                
                return value
        
        elif self.extraction_type == 'regex':
            if self.regex_pattern:
                text = element.get_text()
                matches = re.findall(self.regex_pattern, text)
                return matches if self.multiple else (matches[0] if matches else None)
        
        elif self.extraction_type == 'json':
            try:
                json_text = element.get_text()
                return json.loads(json_text)
            except (json.JSONDecodeError, TypeError):
                return None
        
        return None
    
    def _transform_and_validate(self, value: Any) -> Any:
        """Apply transformations and validation"""
        
        if value is None:
            return self.default_value
        
        # Apply transformations
        if self.transformation and isinstance(value, str):
            if self.transformation == 'clean':
                value = ' '.join(value.split())
            elif self.transformation == 'lower':
                value = value.lower()
            elif self.transformation == 'upper':
                value = value.upper()
            elif self.transformation == 'strip':
                value = value.strip()
            elif self.transformation == 'extract_numbers':
                numbers = re.findall(r'\d+\.?\d*', value)
                if numbers:
                    try:
                        value = float(numbers[0])
                    except ValueError:
                        value = None
        
        # Apply validation
        if self.validation and isinstance(value, str):
            if not re.match(self.validation, value):
                return self.default_value
        
        return value


class GenericScrapingConfig:
    """Configuration for generic scraper"""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.source_name = config_data['source_name']
        self.base_url = config_data['base_url']
        self.search_locations = config_data.get('search_locations', [])
        
        # URL patterns
        self.search_url_pattern = config_data.get('search_url_pattern', '{base_url}/search/{location}')
        self.pagination_pattern = config_data.get('pagination_pattern', '{search_url}?page={page}')
        
        # Extraction rules
        self.listing_links_rule = ExtractionRule(config_data['listing_links'])
        self.property_rules = {
            field: ExtractionRule(rule_config)
            for field, rule_config in config_data['property_extraction'].items()
        }
        
        # Pagination detection
        self.pagination_config = config_data.get('pagination', {})
        self.max_pages = config_data.get('max_pages', 50)
        
        # Optional configurations
        self.headers = config_data.get('headers', {})
        self.cookies = config_data.get('cookies', {})
        self.delay_between_requests = config_data.get('delay', 1.0)


class GenericPropertyScraper(ProductionBaseScraper):
    """Generic scraper that uses configuration to extract property data"""
    
    def __init__(self, scraping_config: GenericScrapingConfig, base_config=None):
        super().__init__(scraping_config.source_name, base_config)
        self.scraping_config = scraping_config
    
    async def initialize(self):
        """Initialize with custom headers"""
        await super().initialize()
        
        # Add custom headers
        if self.session and self.scraping_config.headers:
            self.session.headers.update(self.scraping_config.headers)
    
    def get_search_urls(self) -> List[str]:
        """Generate search URLs from configuration"""
        
        search_urls = []
        
        for location in self.scraping_config.search_locations:
            search_url = self.scraping_config.search_url_pattern.format(
                base_url=self.scraping_config.base_url,
                location=location
            )
            search_urls.append(search_url)
        
        logger.info(f"Generated {len(search_urls)} search URLs for {self.scraping_config.source_name}")
        return search_urls
    
    async def get_listing_urls(self, base_url: str, max_pages: int = None) -> AsyncGenerator[str, None]:
        """Extract listing URLs using configuration"""
        
        if max_pages is None:
            max_pages = self.scraping_config.max_pages
        
        page = 1
        
        while page <= max_pages:
            try:
                # Generate page URL
                if page == 1:
                    page_url = base_url
                else:
                    page_url = self.scraping_config.pagination_pattern.format(
                        search_url=base_url,
                        page=page
                    )
                
                logger.debug(f"Fetching {self.scraping_config.source_name} page {page}: {page_url}")
                
                html_content = await self.fetch_page(page_url)
                if not html_content:
                    logger.warning(f"No content retrieved for page {page}")
                    break
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract listing links using configuration
                listing_links = self._extract_listing_links(soup)
                
                if not listing_links:
                    logger.info(f"No more listings found on page {page}")
                    break
                
                logger.debug(f"Found {len(listing_links)} listings on page {page}")
                
                for link in listing_links:
                    full_url = urljoin(self.scraping_config.base_url, link)
                    yield full_url
                
                page += 1
                
                # Apply configured delay
                await asyncio.sleep(self.scraping_config.delay_between_requests)
                
            except Exception as e:
                logger.error(f"Error processing search page {page}: {e}")
                break
    
    def _extract_listing_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract listing links using configuration"""
        
        try:
            # Use the listing links extraction rule
            links = self.scraping_config.listing_links_rule.extract(
                soup, self.scraping_config.base_url
            )
            
            # Ensure we have a list
            if not isinstance(links, list):
                links = [links] if links else []
            
            # Filter and clean links
            valid_links = []
            for link in links:
                if link and isinstance(link, str):
                    # Clean up relative URLs
                    if link.startswith('/'):
                        valid_links.append(link)
                    elif link.startswith('http'):
                        parsed = urlparse(link)
                        valid_links.append(parsed.path + (f"?{parsed.query}" if parsed.query else ""))
            
            return list(set(valid_links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting listing links: {e}")
            return []
    
    async def extract_property_data(self, listing_url: str, html_content: str) -> Optional[Property]:
        """Extract property data using configuration"""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract all configured fields
            extracted_data = {}
            
            for field_name, extraction_rule in self.scraping_config.property_rules.items():
                try:
                    value = extraction_rule.extract(soup, self.scraping_config.base_url)
                    if value is not None:
                        extracted_data[field_name] = value
                except Exception as e:
                    logger.debug(f"Error extracting {field_name}: {e}")
                    if extraction_rule.required:
                        logger.warning(f"Required field {field_name} extraction failed for {listing_url}")
                        return None
            
            # Validate required fields
            required_fields = ['title', 'price', 'location']
            for field in required_fields:
                if field not in extracted_data or not extracted_data[field]:
                    logger.warning(f"Missing required field {field} for {listing_url}")
                    return None
            
            # Convert to Property entity
            property_entity = self._create_property_entity(extracted_data, listing_url)
            
            if property_entity:
                self.metrics.properties_extracted += 1
                self.metrics.properties_valid += 1
                logger.debug(f"Successfully extracted property: {property_entity.title}")
                return property_entity
            
        except Exception as e:
            self.metrics.properties_invalid += 1
            logger.error(f"Error extracting property data from {listing_url}: {e}")
        
        return None
    
    def _create_property_entity(self, data: Dict[str, Any], listing_url: str) -> Optional[Property]:
        """Create Property entity from extracted data"""
        
        try:
            # Handle price conversion
            price = data.get('price')
            if isinstance(price, str):
                price = self.extract_price(price)
            
            if not price:
                return None
            
            # Create property entity
            property_entity = Property(
                id=uuid4(),
                title=str(data['title']),
                description=str(data.get('description', '')),
                price=price,
                location=str(data['location']),
                bedrooms=int(data.get('bedrooms', 0)),
                bathrooms=float(data.get('bathrooms', 0.0)),
                square_feet=int(data['square_feet']) if data.get('square_feet') else None,
                amenities=data.get('amenities', []) if isinstance(data.get('amenities'), list) else [],
                contact_info=data.get('contact_info', {}) if isinstance(data.get('contact_info'), dict) else {},
                images=data.get('images', []) if isinstance(data.get('images'), list) else [],
                scraped_at=datetime.utcnow(),
                is_active=True,
                property_type=str(data.get('property_type', 'apartment')),
                source_url=listing_url,
                source_name=self.scraping_config.source_name
            )
            
            return property_entity
            
        except Exception as e:
            logger.error(f"Error creating property entity: {e}")
            return None


class GenericScrapingFramework:
    """Framework for managing generic scrapers"""
    
    def __init__(self, base_config=None):
        self.base_config = base_config or get_config()
        self.scraper_configs: Dict[str, GenericScrapingConfig] = {}
        self.active_scrapers: Dict[str, GenericPropertyScraper] = {}
    
    def load_config_from_dict(self, source_name: str, config_data: Dict[str, Any]):
        """Load scraper configuration from dictionary"""
        
        try:
            scraping_config = GenericScrapingConfig(config_data)
            self.scraper_configs[source_name] = scraping_config
            logger.info(f"Loaded configuration for {source_name}")
        except Exception as e:
            logger.error(f"Error loading configuration for {source_name}: {e}")
    
    def load_config_from_yaml(self, source_name: str, yaml_file: str):
        """Load scraper configuration from YAML file"""
        
        try:
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self.load_config_from_dict(source_name, config_data)
            
        except Exception as e:
            logger.error(f"Error loading YAML configuration for {source_name}: {e}")
    
    def create_scraper(self, source_name: str) -> Optional[GenericPropertyScraper]:
        """Create a scraper instance for the given source"""
        
        if source_name not in self.scraper_configs:
            logger.error(f"No configuration found for source: {source_name}")
            return None
        
        try:
            scraping_config = self.scraper_configs[source_name]
            scraper = GenericPropertyScraper(scraping_config, self.base_config)
            
            self.active_scrapers[source_name] = scraper
            logger.info(f"Created scraper for {source_name}")
            
            return scraper
            
        except Exception as e:
            logger.error(f"Error creating scraper for {source_name}: {e}")
            return None
    
    def get_available_sources(self) -> List[str]:
        """Get list of available configured sources"""
        return list(self.scraper_configs.keys())
    
    def validate_config(self, source_name: str) -> Dict[str, Any]:
        """Validate a scraper configuration"""
        
        if source_name not in self.scraper_configs:
            return {'valid': False, 'error': 'Configuration not found'}
        
        config = self.scraper_configs[source_name]
        issues = []
        
        # Check required fields
        if not config.base_url:
            issues.append('Missing base_url')
        
        if not config.search_locations:
            issues.append('Missing search_locations')
        
        if not config.listing_links_rule:
            issues.append('Missing listing_links configuration')
        
        # Check required property extraction rules
        required_fields = ['title', 'price', 'location']
        for field in required_fields:
            if field not in config.property_rules:
                issues.append(f'Missing extraction rule for required field: {field}')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'configured_fields': list(config.property_rules.keys())
        }


# Example configuration for demonstration
EXAMPLE_GENERIC_CONFIG = {
    'source_name': 'example_rental_site',
    'base_url': 'https://example-rentals.com',
    'search_locations': ['new-york-ny', 'los-angeles-ca'],
    'search_url_pattern': '{base_url}/rentals/{location}',
    'pagination_pattern': '{search_url}?page={page}',
    'max_pages': 20,
    'delay': 2.0,
    
    'headers': {
        'User-Agent': 'Mozilla/5.0 (compatible; PropertyBot/1.0)'
    },
    
    'listing_links': {
        'field': 'listing_urls',
        'selectors': ['.property-card a', '.listing-item a[href*="/property/"]'],
        'type': 'attribute',
        'attribute': 'href',
        'multiple': True
    },
    
    'property_extraction': {
        'title': {
            'field': 'title',
            'selectors': ['h1.property-title', '.listing-title', 'h1'],
            'type': 'text',
            'transformation': 'clean',
            'required': True
        },
        
        'price': {
            'field': 'price',
            'selectors': ['.price', '.rent-amount', '[data-testid="price"]'],
            'type': 'regex',
            'regex': r'\$[\d,]+',
            'transformation': 'extract_numbers',
            'required': True
        },
        
        'location': {
            'field': 'location',
            'selectors': ['.address', '.property-location', '[data-testid="address"]'],
            'type': 'text',
            'transformation': 'clean',
            'required': True
        },
        
        'description': {
            'field': 'description',
            'selectors': ['.description', '.property-details', '.listing-description'],
            'type': 'text',
            'transformation': 'clean'
        },
        
        'bedrooms': {
            'field': 'bedrooms',
            'selectors': ['.bedrooms', '.bed-bath'],
            'type': 'regex',
            'regex': r'(\d+)\s*(?:bed|br)',
            'transformation': 'extract_numbers',
            'default': 0
        },
        
        'bathrooms': {
            'field': 'bathrooms',
            'selectors': ['.bathrooms', '.bed-bath'],
            'type': 'regex',
            'regex': r'(\d+(?:\.\d+)?)\s*(?:bath|ba)',
            'transformation': 'extract_numbers',
            'default': 0
        },
        
        'amenities': {
            'field': 'amenities',
            'selectors': ['.amenities li', '.features li'],
            'type': 'text',
            'multiple': True
        },
        
        'images': {
            'field': 'images',
            'selectors': ['.property-images img', '.gallery img'],
            'type': 'attribute',
            'attribute': 'src',
            'multiple': True
        }
    }
}