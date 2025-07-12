"""
Production-ready Zillow scraper implementation.

This module provides a comprehensive scraper for Zillow rental listings
with enhanced error handling, rate limiting, and strict compliance features.
Note: Zillow has strict anti-scraping measures - use with extreme caution.
"""

import asyncio
import logging
import re
import json
from typing import List, Optional, AsyncGenerator, Dict, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse, quote_plus
from uuid import uuid4

from bs4 import BeautifulSoup, Tag
from .production_base_scraper import ProductionBaseScraper
from .config import get_config
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


class ProductionZillowScraper(ProductionBaseScraper):
    """Production-ready scraper for Zillow rental listings with strict compliance"""
    
    def __init__(self, config=None):
        super().__init__('zillow', config)
        self.base_url = "https://www.zillow.com"
        
        # Zillow-specific headers to appear more browser-like
        self.extra_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
    async def initialize(self):
        """Initialize with Zillow-specific settings"""
        await super().initialize()
        
        # Add Zillow-specific headers
        if self.session:
            self.session.headers.update(self.extra_headers)
        
    def get_search_urls(self) -> List[str]:
        """Get search URLs for different locations"""
        search_urls = []
        
        for location in self.source_config.search_locations:
            # Zillow URL structure for rentals
            location_formatted = location.replace('-', '-')
            search_url = f"{self.base_url}/homes/for_rent/{location_formatted}/"
            search_urls.append(search_url)
        
        logger.info(f"Generated {len(search_urls)} search URLs for Zillow")
        return search_urls
    
    async def get_listing_urls(self, base_url: str, max_pages: int = None) -> AsyncGenerator[str, None]:
        """Get URLs of individual property listings from search results"""
        if max_pages is None:
            max_pages = min(self.source_config.max_pages, 10)  # Very conservative for Zillow
        
        page = 1
        
        while page <= max_pages:
            try:
                # Construct search results URL for Zillow
                if page == 1:
                    search_url = base_url
                else:
                    search_url = f"{base_url}{page}_p/"
                
                logger.debug(f"Fetching Zillow search results page {page}: {search_url}")
                
                # Extra delay for Zillow
                if page > 1:
                    await asyncio.sleep(self.source_config.rate_limit.default_crawl_delay * 2)
                
                html_content = await self.fetch_page(search_url)
                if not html_content:
                    logger.warning(f"No content retrieved for page {page}")
                    break
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Check for CAPTCHA or blocking
                if self._detect_blocking(soup, html_content):
                    logger.warning("Zillow blocking detected, stopping scraping")
                    break
                
                # Extract property listing links
                listing_links = self._extract_listing_links(soup)
                
                if not listing_links:
                    logger.info(f"No more listings found on page {page}")
                    break
                
                logger.debug(f"Found {len(listing_links)} listings on page {page}")
                
                for link in listing_links:
                    full_url = urljoin(self.base_url, link)
                    yield full_url
                
                page += 1
                
                # Longer delay between pages for Zillow
                await asyncio.sleep(self.source_config.rate_limit.default_crawl_delay * 3)
                
            except Exception as e:
                logger.error(f"Error processing search page {page}: {e}")
                break
    
    def _detect_blocking(self, soup: BeautifulSoup, html_content: str) -> bool:
        """Detect if we're being blocked by Zillow"""
        
        # Check for CAPTCHA
        if soup.find('div', class_='g-recaptcha') or 'captcha' in html_content.lower():
            return True
        
        # Check for access denied messages
        access_denied_indicators = [
            'access denied',
            'blocked',
            'robot',
            'automated',
            'suspicious activity',
            'rate limit'
        ]
        
        page_text = html_content.lower()
        for indicator in access_denied_indicators:
            if indicator in page_text:
                return True
        
        # Check for unusual redirects
        if 'location.href' in html_content and 'security' in html_content:
            return True
        
        return False
    
    def _extract_listing_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract property listing links from search results page"""
        links = []
        
        try:
            # Zillow-specific selectors (these may change frequently)
            listing_selectors = [
                'article[data-test="property-card"] a',
                '.list-card-link',
                'a[data-test="property-card-link"]',
                '.property-card-link',
                'a[href*="/homedetails/"]',
                'a[href*="/b/"]'  # Zillow rental format
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                
                for element in elements:
                    href = element.get('href')
                    
                    if href and ('/homedetails/' in href or '/b/' in href):
                        # Clean and validate the URL
                        if href.startswith('/'):
                            href = href
                        elif href.startswith('http'):
                            parsed = urlparse(href)
                            href = parsed.path + (f"?{parsed.query}" if parsed.query else "")
                        
                        if href not in links:
                            links.append(href)
                
                if links:  # If we found links with this selector, use them
                    break
            
            # Also try to extract from JSON-LD data
            json_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'url' in data:
                        url = data['url']
                        if '/homedetails/' in url or '/b/' in url:
                            parsed = urlparse(url)
                            href = parsed.path
                            if href not in links:
                                links.append(href)
                except:
                    continue
            
            logger.debug(f"Extracted {len(links)} unique listing links")
            return links
            
        except Exception as e:
            logger.error(f"Error extracting listing links: {e}")
            return []
    
    async def extract_property_data(self, listing_url: str, html_content: str) -> Optional[Property]:
        """Extract property data from a listing page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Check for blocking first
            if self._detect_blocking(soup, html_content):
                logger.warning(f"Blocking detected on {listing_url}")
                return None
            
            # Extract basic property information
            property_data = {}
            
            # Try to extract from JSON-LD first (more reliable)
            json_data = self._extract_json_ld_data(soup)
            if json_data:
                property_data.update(json_data)
            
            # Fallback to HTML extraction
            if not property_data.get('title'):
                property_data['title'] = self._extract_title(soup)
            
            if not property_data.get('price'):
                property_data['price'] = self._extract_price(soup)
            
            if not property_data.get('location'):
                property_data['location'] = self._extract_location(soup)
            
            # Check if we have minimum required data
            if not all(property_data.get(field) for field in ['title', 'price', 'location']):
                logger.warning(f"Missing required data for {listing_url}")
                return None
            
            # Additional fields
            if not property_data.get('description'):
                property_data['description'] = self._extract_description(soup)
            
            if not property_data.get('bedrooms'):
                property_data['bedrooms'] = self._extract_bedrooms(soup)
            
            if not property_data.get('bathrooms'):
                property_data['bathrooms'] = self._extract_bathrooms(soup)
            
            if not property_data.get('square_feet'):
                property_data['square_feet'] = self._extract_square_feet(soup)
            
            property_data['amenities'] = self._extract_amenities(soup)
            property_data['images'] = self._extract_images(soup, listing_url)
            property_data['contact_info'] = self._extract_contact_info(soup)
            property_data['property_type'] = self._extract_property_type(soup)
            
            # Validate the extracted data
            is_valid, issues = self.validate_property_data(property_data)
            if not is_valid:
                logger.warning(f"Invalid property data for {listing_url}: {issues}")
                return None
            
            self.metrics.properties_extracted += 1
            
            # Create Property entity
            property_entity = Property(
                id=uuid4(),
                title=property_data['title'],
                description=property_data.get('description', ''),
                price=property_data['price'],
                location=property_data['location'],
                bedrooms=property_data.get('bedrooms', 0),
                bathrooms=property_data.get('bathrooms', 0.0),
                square_feet=property_data.get('square_feet'),
                amenities=property_data.get('amenities', []),
                contact_info=property_data.get('contact_info', {}),
                images=property_data.get('images', []),
                scraped_at=datetime.utcnow(),
                is_active=True,
                property_type=property_data.get('property_type', 'apartment'),
                source_url=listing_url,
                source_name='zillow'
            )
            
            self.metrics.properties_valid += 1
            logger.debug(f"Successfully extracted property: {property_entity.title}")
            return property_entity
            
        except Exception as e:
            self.metrics.properties_invalid += 1
            logger.error(f"Error extracting property data from {listing_url}: {e}")
            return None
    
    def _extract_json_ld_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract property data from JSON-LD structured data"""
        property_data = {}
        
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                
                if isinstance(data, list):
                    data = data[0] if data else {}
                
                if isinstance(data, dict):
                    # Look for property-related data
                    if data.get('@type') in ['Product', 'RealEstateListing', 'Place']:
                        
                        # Title
                        if data.get('name'):
                            property_data['title'] = data['name']
                        
                        # Price
                        if data.get('offers') and isinstance(data['offers'], dict):
                            price_text = data['offers'].get('price', '')
                            if price_text:
                                price = self.extract_price(str(price_text))
                                if price:
                                    property_data['price'] = price
                        
                        # Location
                        if data.get('address'):
                            address = data['address']
                            if isinstance(address, dict):
                                parts = []
                                if address.get('streetAddress'):
                                    parts.append(address['streetAddress'])
                                if address.get('addressLocality'):
                                    parts.append(address['addressLocality'])
                                if address.get('addressRegion'):
                                    parts.append(address['addressRegion'])
                                
                                if parts:
                                    property_data['location'] = ', '.join(parts)
                            elif isinstance(address, str):
                                property_data['location'] = address
                        
                        # Description
                        if data.get('description'):
                            property_data['description'] = data['description']
                        
                        break
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        return property_data
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property title"""
        title_selectors = [
            'h1[data-testid="property-overview-address"]',
            '.property-overview-address',
            'h1.notranslate',
            '[data-test="property-title"]',
            'h1'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = self.clean_text(element.get_text())
                if title and len(title) > 5:
                    return title
        
        return None
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract property price"""
        price_selectors = [
            '[data-testid="price"]',
            '.property-overview-price',
            '.notranslate',
            '.price-display',
            '[data-test="property-price"]'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for element in elements:
                price_text = self.clean_text(element.get_text())
                
                # Extract price using regex
                price_matches = re.findall(r'\$[\d,]+', price_text)
                if price_matches:
                    price = self.extract_price(price_matches[0])
                    if price:
                        return price
        
        return None
    
    def _extract_location(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property location"""
        location_selectors = [
            '[data-testid="property-overview-address"]',
            '.property-overview-address',
            '[data-test="property-address"]',
            '.address-info'
        ]
        
        for selector in location_selectors:
            element = soup.select_one(selector)
            if element:
                location = self.clean_text(element.get_text())
                if location and len(location) > 5:
                    return location
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract property description"""
        desc_selectors = [
            '[data-testid="property-description"]',
            '.property-description',
            '.listing-description',
            '.description-text'
        ]
        
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                description = self.clean_text(element.get_text())
                if description and len(description) > 20:
                    return description
        
        return ""
    
    def _extract_bedrooms(self, soup: BeautifulSoup) -> int:
        """Extract number of bedrooms"""
        bedroom_patterns = [
            r'(\d+)\s*(?:bed|bedroom|br)\b',
            r'\b(\d+)br\b'
        ]
        
        text_selectors = [
            '[data-testid="bed-bath-sqft"]',
            '.property-overview-meta',
            '.property-details'
        ]
        
        for selector in text_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().lower()
                for pattern in bedroom_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            return int(match.group(1))
                        except ValueError:
                            continue
        
        return 0
    
    def _extract_bathrooms(self, soup: BeautifulSoup) -> float:
        """Extract number of bathrooms"""
        bathroom_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)\b',
            r'\b(\d+(?:\.\d+)?)ba\b'
        ]
        
        text_selectors = [
            '[data-testid="bed-bath-sqft"]',
            '.property-overview-meta',
            '.property-details'
        ]
        
        for selector in text_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().lower()
                for pattern in bathroom_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            return float(match.group(1))
                        except ValueError:
                            continue
        
        return 0.0
    
    def _extract_square_feet(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract square footage"""
        sqft_patterns = [
            r'(\d{1,4}(?:,\d{3})?)\s*(?:sq\.?\s*ft|square\s*feet|sqft)',
            r'(\d{1,4}(?:,\d{3})?)\s*sf\b'
        ]
        
        text_selectors = [
            '[data-testid="bed-bath-sqft"]',
            '.property-overview-meta',
            '.property-details'
        ]
        
        for selector in text_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().lower()
                for pattern in sqft_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            sqft_str = match.group(1).replace(',', '')
                            return int(sqft_str)
                        except ValueError:
                            continue
        
        return None
    
    def _extract_amenities(self, soup: BeautifulSoup) -> List[str]:
        """Extract property amenities"""
        amenities = []
        
        amenity_selectors = [
            '.amenities-list li',
            '.features-list li',
            '.property-features li',
            '[data-testid="amenity"]'
        ]
        
        for selector in amenity_selectors:
            elements = soup.select(selector)
            for element in elements:
                amenity = self.clean_text(element.get_text())
                if amenity and len(amenity) > 2 and amenity not in amenities:
                    amenities.append(amenity.lower())
        
        return amenities[:10]  # Limit to 10 amenities
    
    def _extract_images(self, soup: BeautifulSoup, listing_url: str) -> List[str]:
        """Extract property images"""
        images = []
        
        img_selectors = [
            '.media-stream img',
            '.media-carousel img',
            '.property-photos img',
            'picture img'
        ]
        
        for selector in img_selectors:
            img_elements = soup.select(selector)
            for img in img_elements:
                src = img.get('src') or img.get('data-src')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(self.base_url, src)
                    
                    if src not in images and src.startswith('http'):
                        images.append(src)
        
        return images[:5]  # Limit to 5 images
    
    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {}
        
        # Phone number (often protected on Zillow)
        phone_selectors = [
            '[data-testid="phone"]',
            '.phone-number',
            '.contact-phone'
        ]
        
        for selector in phone_selectors:
            element = soup.select_one(selector)
            if element:
                phone = self.clean_text(element.get_text())
                if phone:
                    contact_info['phone'] = phone
                    break
        
        return contact_info
    
    def _extract_property_type(self, soup: BeautifulSoup) -> str:
        """Extract property type"""
        page_text = soup.get_text().lower()
        
        if any(word in page_text for word in ['apartment', 'apt']):
            return 'apartment'
        elif any(word in page_text for word in ['house', 'home', 'single family']):
            return 'house'
        elif any(word in page_text for word in ['condo', 'condominium']):
            return 'condo'
        elif 'studio' in page_text:
            return 'studio'
        elif any(word in page_text for word in ['townhouse', 'townhome']):
            return 'townhouse'
        else:
            return 'apartment'  # Default