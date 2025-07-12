"""
Generic scraper framework for easy addition of new rental property sources.

This module provides a configurable framework that allows easy creation of new
scrapers through configuration rather than writing new code.
"""

import asyncio
import logging
import re
import json
import yaml
from typing import List, Optional, AsyncGenerator, Dict, Any, Union
from datetime import datetime
from urllib.parse import urljoin, urlparse
from uuid import uuid4
from pathlib import Path

from bs4 import BeautifulSoup
from .production_base_scraper import ProductionBaseScraper
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


class ScraperTemplate:
    """Template configuration for a scraper source"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.name = config_dict['name']
        self.base_url = config_dict['base_url']
        self.search_url_pattern = config_dict.get('search_url_pattern', '{base_url}/{location}/')
        
        # Selectors for different elements
        self.selectors = config_dict.get('selectors', {})
        
        # URL patterns and extraction rules
        self.url_patterns = config_dict.get('url_patterns', {})
        
        # Data extraction rules
        self.extraction_rules = config_dict.get('extraction_rules', {})
        
        # Pagination configuration
        self.pagination = config_dict.get('pagination', {})
        
        # Location mapping for URL generation
        self.location_mapping = config_dict.get('location_mapping', {})
        
        # Data normalization rules
        self.normalization = config_dict.get('normalization', {})
        
        # Rate limiting overrides
        self.rate_limiting = config_dict.get('rate_limiting', {})


class GenericScraper(ProductionBaseScraper):
    """Generic configurable scraper that works with template configurations"""
    
    def __init__(self, template: ScraperTemplate, config=None):
        super().__init__(template.name, config)
        self.template = template
        
        # Override source config if template provides rate limiting
        if self.template.rate_limiting:
            for key, value in self.template.rate_limiting.items():
                if hasattr(self.source_config.rate_limit, key):
                    setattr(self.source_config.rate_limit, key, value)
    
    def get_search_urls(self) -> List[str]:
        """Generate search URLs using template configuration"""
        search_urls = []
        
        for location in self.source_config.search_locations:
            # Map location if mapping is provided
            mapped_location = self.template.location_mapping.get(location, location)
            
            # Generate search URL using template pattern
            search_url = self.template.search_url_pattern.format(
                base_url=self.template.base_url,
                location=mapped_location
            )
            search_urls.append(search_url)
            
            # Add filtered searches if configured
            filters = self.template.url_patterns.get('filters', [])
            for filter_config in filters:
                if self._should_apply_filter(location, filter_config):
                    filtered_url = self._apply_url_filter(search_url, filter_config)
                    search_urls.append(filtered_url)
        
        logger.info(f"Generated {len(search_urls)} search URLs for {self.template.name}")
        return search_urls
    
    def _should_apply_filter(self, location: str, filter_config: Dict) -> bool:
        """Check if filter should be applied to this location"""
        target_locations = filter_config.get('locations', [])
        return not target_locations or location in target_locations
    
    def _apply_url_filter(self, base_url: str, filter_config: Dict) -> str:
        """Apply URL filter configuration"""
        filter_pattern = filter_config.get('pattern', '')
        filter_params = filter_config.get('params', {})
        
        if filter_pattern:
            return filter_pattern.format(base_url=base_url, **filter_params)
        
        # Add query parameters
        separator = '&' if '?' in base_url else '?'
        param_string = '&'.join([f"{k}={v}" for k, v in filter_params.items()])
        return f"{base_url}{separator}{param_string}"
    
    async def get_listing_urls(self, base_url: str, max_pages: int = None) -> AsyncGenerator[str, None]:
        """Extract listing URLs using template configuration"""
        if max_pages is None:
            max_pages = self.source_config.max_pages
        
        page = 1
        seen_urls = set()
        consecutive_empty_pages = 0
        
        while page <= max_pages and consecutive_empty_pages < 3:
            try:
                search_url = self._build_paginated_url(base_url, page)
                
                logger.debug(f"Fetching {self.template.name} page {page}: {search_url}")
                
                html_content = await self.fetch_page(search_url)
                if not html_content:
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                listing_links = self._extract_listing_links_from_template(soup)
                
                if not listing_links:
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                consecutive_empty_pages = 0
                new_urls_count = 0
                
                for link in listing_links:
                    full_url = urljoin(self.template.base_url, link)
                    
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        new_urls_count += 1
                        yield full_url
                
                logger.debug(f"Found {new_urls_count} new listings on page {page}")
                
                if self._is_last_page_from_template(soup) or new_urls_count == 0:
                    break
                
                page += 1
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {self.template.name} page {page}: {e}")
                consecutive_empty_pages += 1
                page += 1
    
    def _build_paginated_url(self, base_url: str, page: int) -> str:
        """Build paginated URL using template configuration"""
        pagination_config = self.template.pagination
        
        if page == 1:
            return base_url
        
        pattern = pagination_config.get('pattern', '{base_url}?page={page}')
        return pattern.format(base_url=base_url, page=page)
    
    def _extract_listing_links_from_template(self, soup: BeautifulSoup) -> List[str]:
        """Extract listing links using template selectors"""
        links = []
        
        selectors = self.template.selectors.get('listing_links', [])
        url_patterns = self.template.url_patterns.get('listing_patterns', [])
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href and self._matches_listing_pattern(href, url_patterns):
                    if href.startswith('/'):
                        links.append(href)
                    elif href.startswith('http') and self.template.base_url in href:
                        parsed = urlparse(href)
                        links.append(parsed.path)
        
        return list(set(links))
    
    def _matches_listing_pattern(self, href: str, patterns: List[str]) -> bool:
        """Check if URL matches any of the listing patterns"""
        if not patterns:
            return True  # No patterns means accept all
        
        for pattern in patterns:
            if pattern in href:
                return True
        return False
    
    def _is_last_page_from_template(self, soup: BeautifulSoup) -> bool:
        """Check if this is the last page using template configuration"""
        last_page_selectors = self.template.pagination.get('last_page_indicators', [])
        
        for selector in last_page_selectors:
            if soup.select(selector):
                return True
        
        return False
    
    async def extract_property_data(self, listing_url: str, html_content: str) -> Optional[Property]:
        """Extract property data using template configuration"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            property_data = {
                'source_url': listing_url,
                'scraped_at': datetime.utcnow()
            }
            
            # Extract data using template rules
            extraction_rules = self.template.extraction_rules
            
            for field, rules in extraction_rules.items():
                value = self._extract_field_from_template(soup, field, rules)
                if value is not None:
                    property_data[field] = value
            
            # Apply normalization rules
            property_data = self._apply_normalization_rules(property_data)
            
            # Validate data
            is_valid, issues = self.validate_property_data(property_data)
            if not is_valid:
                logger.warning(f"Invalid property data for {listing_url}: {issues}")
                self.metrics.properties_invalid += 1
                return None
            
            # Create Property entity
            property_entity = Property(
                id=uuid4(),
                title=property_data.get('title', ''),
                description=property_data.get('description', ''),
                price=property_data.get('price', 0),
                location=property_data.get('location', ''),
                bedrooms=property_data.get('bedrooms', 0),
                bathrooms=property_data.get('bathrooms', 0.0),
                square_feet=property_data.get('square_feet'),
                amenities=property_data.get('amenities', []),
                contact_info=property_data.get('contact_info', {}),
                images=property_data.get('images', []),
                scraped_at=property_data['scraped_at'],
                is_active=True,
                property_type=property_data.get('property_type', 'apartment')
            )
            
            self.metrics.properties_extracted += 1
            self.metrics.properties_valid += 1
            
            return property_entity
            
        except Exception as e:
            logger.error(f"Error extracting {self.template.name} property data from {listing_url}: {e}")
            self.metrics.properties_invalid += 1
            return None
    
    def _extract_field_from_template(self, soup: BeautifulSoup, field: str, rules: Dict) -> Any:
        """Extract a field using template rules"""
        selectors = rules.get('selectors', [])
        extraction_type = rules.get('type', 'text')
        patterns = rules.get('patterns', [])
        fallbacks = rules.get('fallbacks', [])
        
        # Try main selectors first
        value = self._extract_with_selectors_and_patterns(soup, selectors, patterns, extraction_type)
        
        # Try fallbacks if no value found
        if value is None:
            for fallback in fallbacks:
                fallback_selectors = fallback.get('selectors', [])
                fallback_patterns = fallback.get('patterns', [])
                fallback_type = fallback.get('type', extraction_type)
                
                value = self._extract_with_selectors_and_patterns(
                    soup, fallback_selectors, fallback_patterns, fallback_type
                )
                if value is not None:
                    break
        
        return value
    
    def _extract_with_selectors_and_patterns(
        self, 
        soup: BeautifulSoup, 
        selectors: List[str], 
        patterns: List[str],
        extraction_type: str
    ) -> Any:
        """Extract value using selectors and patterns"""
        
        for selector in selectors:
            try:
                if extraction_type == 'text':
                    elements = soup.select(selector)
                    for element in elements:
                        text = self.clean_text(element.get_text())
                        if patterns:
                            for pattern in patterns:
                                match = re.search(pattern, text, re.IGNORECASE)
                                if match:
                                    return match.group(1) if match.groups() else match.group(0)
                        elif text:
                            return text
                
                elif extraction_type == 'attribute':
                    elements = soup.select(selector)
                    for element in elements:
                        for attr in ['src', 'href', 'data-src', 'data-href']:
                            value = element.get(attr)
                            if value:
                                return value
                
                elif extraction_type == 'price':
                    elements = soup.select(selector)
                    for element in elements:
                        text = self.clean_text(element.get_text())
                        price = self.extract_price(text)
                        if price:
                            return price
                
                elif extraction_type == 'number':
                    elements = soup.select(selector)
                    for element in elements:
                        text = self.clean_text(element.get_text())
                        if patterns:
                            for pattern in patterns:
                                match = re.search(pattern, text, re.IGNORECASE)
                                if match:
                                    try:
                                        return int(match.group(1)) if match.groups() else int(match.group(0))
                                    except ValueError:
                                        continue
                        else:
                            # Try to extract any number
                            numbers = re.findall(r'\d+', text)
                            if numbers:
                                return int(numbers[0])
                
                elif extraction_type == 'float':
                    elements = soup.select(selector)
                    for element in elements:
                        text = self.clean_text(element.get_text())
                        if patterns:
                            for pattern in patterns:
                                match = re.search(pattern, text, re.IGNORECASE)
                                if match:
                                    try:
                                        return float(match.group(1)) if match.groups() else float(match.group(0))
                                    except ValueError:
                                        continue
                
                elif extraction_type == 'list':
                    elements = soup.select(selector)
                    items = []
                    for element in elements:
                        text = self.clean_text(element.get_text())
                        if text and len(text) > 2:
                            items.append(text.lower())
                    return items
                
            except Exception as e:
                logger.debug(f"Error with selector '{selector}': {e}")
                continue
        
        return None
    
    def _apply_normalization_rules(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization rules from template"""
        normalization_rules = self.template.normalization
        
        for field, rules in normalization_rules.items():
            if field in property_data:
                value = property_data[field]
                
                # Apply text replacements
                replacements = rules.get('replacements', [])
                if isinstance(value, str):
                    for replacement in replacements:
                        pattern = replacement.get('pattern', '')
                        replace_with = replacement.get('replace_with', '')
                        value = re.sub(pattern, replace_with, value, flags=re.IGNORECASE)
                
                # Apply value mappings
                mappings = rules.get('mappings', {})
                if str(value).lower() in mappings:
                    value = mappings[str(value).lower()]
                
                # Apply validation rules
                validation = rules.get('validation', {})
                if validation:
                    min_value = validation.get('min')
                    max_value = validation.get('max')
                    
                    if isinstance(value, (int, float)):
                        if min_value is not None and value < min_value:
                            value = None
                        elif max_value is not None and value > max_value:
                            value = None
                
                property_data[field] = value
        
        return property_data


class ScraperTemplateManager:
    """Manager for scraper templates"""
    
    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"
        self.templates: Dict[str, ScraperTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all scraper templates from directory"""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_dir}")
            return
        
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_config = yaml.safe_load(f)
                
                template = ScraperTemplate(template_config)
                self.templates[template.name] = template
                
                logger.info(f"Loaded scraper template: {template.name}")
                
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
    
    def get_template(self, name: str) -> Optional[ScraperTemplate]:
        """Get a template by name"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())
    
    def create_scraper(self, template_name: str, config=None) -> Optional[GenericScraper]:
        """Create a scraper instance from template"""
        template = self.get_template(template_name)
        if template:
            return GenericScraper(template, config)
        return None
    
    def add_template_from_dict(self, template_dict: Dict[str, Any]):
        """Add a template from dictionary configuration"""
        template = ScraperTemplate(template_dict)
        self.templates[template.name] = template
        logger.info(f"Added scraper template: {template.name}")
    
    def validate_template(self, template_dict: Dict[str, Any]) -> List[str]:
        """Validate template configuration and return issues"""
        issues = []
        
        required_fields = ['name', 'base_url', 'selectors']
        for field in required_fields:
            if field not in template_dict:
                issues.append(f"Missing required field: {field}")
        
        # Validate selectors
        selectors = template_dict.get('selectors', {})
        required_selectors = ['listing_links']
        for selector in required_selectors:
            if selector not in selectors:
                issues.append(f"Missing required selector: {selector}")
        
        # Validate extraction rules
        extraction_rules = template_dict.get('extraction_rules', {})
        required_extraction_rules = ['title', 'price', 'location']
        for rule in required_extraction_rules:
            if rule not in extraction_rules:
                issues.append(f"Missing required extraction rule: {rule}")
        
        return issues


# Global template manager instance
_template_manager = None

def get_template_manager() -> ScraperTemplateManager:
    """Get global template manager instance"""
    global _template_manager
    if _template_manager is None:
        _template_manager = ScraperTemplateManager()
    return _template_manager