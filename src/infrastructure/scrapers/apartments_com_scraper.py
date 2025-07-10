"""
Apartments.com scraper implementation.

This module provides a concrete scraper for apartments.com property listings.
Note: This is for educational purposes only. Always respect robots.txt and terms of service.
"""

import asyncio
import logging
import re
from typing import List, Optional, AsyncGenerator
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
from uuid import uuid4

from bs4 import BeautifulSoup, Tag
from .base_scraper import BaseScraper, ScrapingConfig
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


class ApartmentsComScraper(BaseScraper):
    """Scraper for apartments.com listings"""
    
    def __init__(self, config: ScrapingConfig = None):
        if config is None:
            config = ScrapingConfig(
                max_concurrent_requests=5,  # Be conservative with apartments.com
                request_delay=2.0,          # 2 second delay between requests
                rate_limit_per_second=0.5,  # 0.5 requests per second
                respect_robots_txt=True
            )
        super().__init__(config)
        
        self.base_url = "https://www.apartments.com"
        self.search_locations = [
            "new-york-ny",
            "los-angeles-ca", 
            "chicago-il",
            "houston-tx",
            "phoenix-az",
            "philadelphia-pa",
            "san-antonio-tx",
            "san-diego-ca",
            "dallas-tx",
            "san-jose-ca"
        ]
    
    def get_search_urls(self) -> List[str]:
        """Get search URLs for different locations"""
        search_urls = []
        
        for location in self.search_locations:
            # Basic search URL structure for apartments.com
            search_url = f"{self.base_url}/{location}/"
            search_urls.append(search_url)
        
        logger.info(f"Generated {len(search_urls)} search URLs")
        return search_urls
    
    async def get_listing_urls(self, base_url: str, max_pages: int = None) -> AsyncGenerator[str, None]:
        """Get URLs of individual property listings from search results"""
        if max_pages is None:
            max_pages = self.config.max_pages_per_source
        
        page = 1
        
        while page <= max_pages:
            try:
                # Construct search results URL
                if page == 1:
                    search_url = base_url
                else:
                    search_url = f"{base_url}{page}/"
                
                logger.debug(f"Fetching search results page {page}: {search_url}")
                
                # Check robots.txt compliance
                if not await self.check_robots_txt(search_url):
                    logger.warning(f"Robots.txt disallows scraping {search_url}")
                    break
                
                html_content = await self.fetch_page(search_url)
                if not html_content:
                    logger.warning(f"No content retrieved for page {page}")
                    break
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find property listing links
                listing_links = self._extract_listing_links(soup)
                
                if not listing_links:
                    logger.info(f"No more listings found on page {page}")
                    break
                
                logger.debug(f"Found {len(listing_links)} listings on page {page}")
                
                for link in listing_links:
                    full_url = urljoin(self.base_url, link)
                    yield full_url
                
                page += 1
                
                # Add delay between pages
                await asyncio.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"Error processing search page {page}: {e}")
                break
    
    def _extract_listing_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract property listing links from search results page"""
        links = []
        
        try:
            # Look for property listing containers
            # Note: These selectors are examples and may need updating based on site structure
            listing_selectors = [
                'article.listing',
                '.property-listing',
                '.property-card',
                'a[href*="/apartments/"]',
                '.property-link'
            ]
            
            for selector in listing_selectors:
                elements = soup.select(selector)
                
                for element in elements:
                    href = None
                    
                    if element.name == 'a':
                        href = element.get('href')
                    else:
                        # Look for link within the element
                        link_elem = element.find('a', href=True)
                        if link_elem:
                            href = link_elem.get('href')
                    
                    if href and '/apartments/' in href:
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
            
            logger.debug(f"Extracted {len(links)} unique listing links")
            return links
            
        except Exception as e:
            logger.error(f"Error extracting listing links: {e}")
            return []
    
    async def extract_property_data(self, listing_url: str, html_content: str) -> Optional[Property]:
        """Extract property data from a listing page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract basic property information
            property_data = {}
            
            # Title
            title = self._extract_title(soup)
            if not title:
                logger.warning(f"No title found for {listing_url}")
                return None
            property_data['title'] = title
            
            # Price
            price = self._extract_price(soup)
            if not price:
                logger.warning(f"No price found for {listing_url}")
                return None
            property_data['price'] = price
            
            # Location
            location = self._extract_location(soup)
            if not location:
                logger.warning(f"No location found for {listing_url}")
                return None
            property_data['location'] = location
            
            # Description
            property_data['description'] = self._extract_description(soup)
            
            # Bedrooms and bathrooms
            property_data['bedrooms'] = self._extract_bedrooms(soup)
            property_data['bathrooms'] = self._extract_bathrooms(soup)
            
            # Square feet
            property_data['square_feet'] = self._extract_square_feet(soup)
            
            # Amenities
            property_data['amenities'] = self._extract_amenities(soup)
            
            # Images
            property_data['images'] = self._extract_images(soup, listing_url)
            
            # Contact info
            property_data['contact_info'] = self._extract_contact_info(soup)
            
            # Property type
            property_data['property_type'] = self._extract_property_type(soup)
            
            # Validate the extracted data
            if not self.validate_property_data(property_data):
                logger.warning(f"Invalid property data for {listing_url}")
                return None
            
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
                property_type=property_data.get('property_type', 'apartment')
            )
            
            logger.debug(f"Successfully extracted property: {property_entity.title}")
            return property_entity
            
        except Exception as e:
            logger.error(f"Error extracting property data from {listing_url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property title"""
        title_selectors = [
            'h1.property-title',
            'h1[data-test="property-name"]',
            '.property-name h1',
            'h1.listing-title',
            'h1'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = self.clean_text(element.get_text())
                if title:
                    return title
        
        return None
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract property price"""
        price_selectors = [
            '.property-pricing',
            '.price-range',
            '[data-test="property-pricing"]',
            '.rent-price',
            '.pricing-info'
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
            '.property-address',
            '[data-test="property-address"]',
            '.listing-address',
            '.address-info',
            '.location'
        ]
        
        for selector in location_selectors:
            element = soup.select_one(selector)
            if element:
                location = self.clean_text(element.get_text())
                if location:
                    return self.normalize_location(location)
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract property description"""
        desc_selectors = [
            '.property-description',
            '[data-test="property-description"]',
            '.listing-description',
            '.description-text',
            '.about-section'
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
        # Look for bedroom information in various places
        bedroom_patterns = [
            r'(\d+)\s*(?:bed|bedroom|br)\b',
            r'\b(\d+)br\b',
            r'(\d+)\s*(?:bed|bedroom)'
        ]
        
        # Check multiple selectors
        text_selectors = [
            '.bed-bath-info',
            '.property-specs',
            '.unit-info',
            '.floorplan-info'
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
            r'\b(\d+(?:\.\d+)?)ba\b',
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom)'
        ]
        
        text_selectors = [
            '.bed-bath-info',
            '.property-specs',
            '.unit-info',
            '.floorplan-info'
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
            '.property-specs',
            '.unit-info', 
            '.floorplan-info',
            '.size-info'
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
            '[data-test="amenity"]'
        ]
        
        for selector in amenity_selectors:
            elements = soup.select(selector)
            for element in elements:
                amenity = self.clean_text(element.get_text())
                if amenity and amenity not in amenities:
                    amenities.append(amenity.lower())
        
        # Normalize common amenities
        normalized_amenities = []
        amenity_mapping = {
            'pool': ['pool', 'swimming pool'],
            'gym': ['gym', 'fitness', 'fitness center', 'workout'],
            'parking': ['parking', 'garage', 'covered parking'],
            'pet-friendly': ['pet', 'pets allowed', 'pet friendly'],
            'laundry': ['laundry', 'washer', 'dryer', 'in-unit laundry'],
            'ac': ['air conditioning', 'ac', 'central air'],
            'dishwasher': ['dishwasher'],
            'balcony': ['balcony', 'patio', 'terrace']
        }
        
        for amenity in amenities:
            for normalized, variants in amenity_mapping.items():
                if any(variant in amenity for variant in variants):
                    if normalized not in normalized_amenities:
                        normalized_amenities.append(normalized)
                    break
        
        return normalized_amenities
    
    def _extract_images(self, soup: BeautifulSoup, listing_url: str) -> List[str]:
        """Extract property images"""
        images = []
        
        img_selectors = [
            '.property-photos img',
            '.gallery img',
            '.photo-gallery img',
            '.listing-images img'
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
        
        return images[:10]  # Limit to 10 images
    
    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {}
        
        # Phone number
        phone_selectors = [
            '[data-test="phone"]',
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
        
        # Email (less common on listing pages)
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        page_text = soup.get_text()
        email_match = re.search(email_pattern, page_text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        return contact_info
    
    def _extract_property_type(self, soup: BeautifulSoup) -> str:
        """Extract property type"""
        # Look for property type indicators
        page_text = soup.get_text().lower()
        
        if any(word in page_text for word in ['apartment', 'apt']):
            return 'apartment'
        elif any(word in page_text for word in ['house', 'home']):
            return 'house'
        elif any(word in page_text for word in ['condo', 'condominium']):
            return 'condo'
        elif 'studio' in page_text:
            return 'studio'
        elif any(word in page_text for word in ['townhouse', 'townhome']):
            return 'townhouse'
        else:
            return 'apartment'  # Default