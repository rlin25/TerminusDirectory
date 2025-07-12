"""
Production-ready Rent.com scraper.

This module provides a robust scraper for rent.com with comprehensive 
handling of their specific layout and data patterns.
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
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


class RentComScraper(ProductionBaseScraper):
    """Production scraper for rent.com listings"""
    
    def __init__(self, config=None):
        super().__init__('rent_com', config)
        
        # Rent.com specific selectors
        self.selectors = {
            'listing_links': [
                'a[href*="/listing/"]',
                '.property-link a',
                '.listing-item a',
                'a[data-testid="listing-link"]',
                '.rental-card a'
            ],
            'title': [
                'h1.property-title',
                'h1.listing-name',
                '.property-header h1',
                'h1[data-testid="property-name"]',
                '.title h1'
            ],
            'price': [
                '.rent-price',
                '.price-section',
                '.pricing-display',
                '[data-testid="rent-price"]',
                '.monthly-rent'
            ],
            'location': [
                '.property-address',
                '.address-display',
                '.location-section',
                '[data-testid="property-address"]',
                '.listing-address'
            ],
            'description': [
                '.property-overview',
                '.description-content',
                '.listing-description',
                '.property-details',
                '.overview-section'
            ],
            'amenities': [
                '.amenities-section li',
                '.features-list li',
                '.amenity-tag',
                '.property-amenities li',
                '.feature-item'
            ],
            'images': [
                '.property-photos img',
                '.gallery-images img',
                '.photo-slider img',
                '.listing-gallery img',
                '.property-gallery img'
            ],
            'contact': [
                '.contact-number',
                '.phone-display',
                '[data-testid="contact-phone"]',
                '.leasing-number'
            ]
        }
    
    def get_search_urls(self) -> List[str]:
        """Generate search URLs for Rent.com"""
        search_urls = []
        base_url = self.source_config.base_url
        
        # Rent.com uses state-city format
        location_mapping = {
            'new-york-ny': 'new-york/new-york-city',
            'los-angeles-ca': 'california/los-angeles',
            'chicago-il': 'illinois/chicago',
            'houston-tx': 'texas/houston',
            'phoenix-az': 'arizona/phoenix',
            'philadelphia-pa': 'pennsylvania/philadelphia',
            'san-diego-ca': 'california/san-diego',
            'dallas-tx': 'texas/dallas'
        }
        
        for location in self.source_config.search_locations:
            rent_location = location_mapping.get(location)
            if rent_location:
                search_urls.append(f"{base_url}/{rent_location}")
                
                # Add filtered searches
                if location in ['new-york-ny', 'los-angeles-ca', 'chicago-il']:
                    search_urls.append(f"{base_url}/{rent_location}?property_type=apartment")
                    search_urls.append(f"{base_url}/{rent_location}?min_rent=2000")
        
        logger.info(f"Generated {len(search_urls)} search URLs for rent.com")
        return search_urls
    
    async def get_listing_urls(self, base_url: str, max_pages: int = None) -> AsyncGenerator[str, None]:
        """Extract listing URLs from Rent.com search results"""
        if max_pages is None:
            max_pages = self.source_config.max_pages
        
        page = 1
        seen_urls = set()
        consecutive_empty_pages = 0
        
        while page <= max_pages and consecutive_empty_pages < 3:
            try:
                # Build search URL for current page
                if page == 1:
                    search_url = base_url
                else:
                    separator = '&' if '?' in base_url else '?'
                    search_url = f"{base_url}{separator}page={page}"
                
                logger.debug(f"Fetching Rent.com page {page}: {search_url}")
                
                html_content = await self.fetch_page(search_url)
                if not html_content:
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract listing links
                listing_links = self._extract_listing_links(soup)
                
                if not listing_links:
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                consecutive_empty_pages = 0
                new_urls_count = 0
                
                for link in listing_links:
                    full_url = urljoin(self.source_config.base_url, link)
                    
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        new_urls_count += 1
                        yield full_url
                
                logger.debug(f"Found {new_urls_count} new listings on page {page}")
                
                # Check for end of results
                if self._is_last_page(soup) or new_urls_count == 0:
                    break
                
                page += 1
                await asyncio.sleep(1)  # Be respectful
                
            except Exception as e:
                logger.error(f"Error processing Rent.com page {page}: {e}")
                consecutive_empty_pages += 1
                page += 1
    
    def _extract_listing_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract listing links from search results"""
        links = []
        
        for selector in self.selectors['listing_links']:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href and ('/listing/' in href or '/property/' in href):
                    # Clean the URL
                    if href.startswith('/'):
                        links.append(href)
                    elif href.startswith('http') and 'rent.com' in href:
                        parsed = urlparse(href)
                        links.append(parsed.path)
        
        return list(set(links))  # Remove duplicates
    
    def _is_last_page(self, soup: BeautifulSoup) -> bool:
        """Check if this is the last page"""
        # Look for pagination indicators
        pagination_indicators = [
            '.pagination .next.disabled',
            '.pager .next[disabled]',
            '.pagination .current:last-child',
            '[data-testid="next-page"][disabled]'
        ]
        
        for indicator in pagination_indicators:
            if soup.select(indicator):
                return True
        
        return False
    
    async def extract_property_data(self, listing_url: str, html_content: str) -> Optional[Property]:
        """Extract property data from Rent.com listing"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            property_data = {
                'source_url': listing_url,
                'scraped_at': datetime.utcnow()
            }
            
            # Extract basic information
            property_data['title'] = self._extract_title(soup)
            property_data['price'] = self._extract_price(soup)
            property_data['location'] = self._extract_location(soup)
            property_data['description'] = self._extract_description(soup)
            
            # Extract specifications
            property_data['bedrooms'] = self._extract_bedrooms(soup)
            property_data['bathrooms'] = self._extract_bathrooms(soup)
            property_data['square_feet'] = self._extract_square_feet(soup)
            
            # Extract additional details
            property_data['amenities'] = self._extract_amenities(soup)
            property_data['images'] = self._extract_images(soup)
            property_data['contact_info'] = self._extract_contact_info(soup)
            property_data['property_type'] = self._extract_property_type(soup)
            
            # Validate data
            is_valid, issues = self.validate_property_data(property_data)
            if not is_valid:
                logger.warning(f"Invalid property data for {listing_url}: {issues}")
                self.metrics.properties_invalid += 1
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
                scraped_at=property_data['scraped_at'],
                is_active=True,
                property_type=property_data.get('property_type', 'apartment')
            )
            
            self.metrics.properties_extracted += 1
            self.metrics.properties_valid += 1
            
            return property_entity
            
        except Exception as e:
            logger.error(f"Error extracting Rent.com property data from {listing_url}: {e}")
            self.metrics.properties_invalid += 1
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property title"""
        for selector in self.selectors['title']:
            element = soup.select_one(selector)
            if element:
                title = self.clean_text(element.get_text())
                if title:
                    return title
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            title = self.clean_text(title_tag.get_text())
            title = re.sub(r'\s*\|\s*Rent\.com.*$', '', title, flags=re.IGNORECASE)
            if title:
                return title
        
        return None
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract rental price"""
        for selector in self.selectors['price']:
            elements = soup.select(selector)
            for element in elements:
                price_text = self.clean_text(element.get_text())
                
                # Price patterns for Rent.com
                price_patterns = [
                    r'\$(\d{1,2},?\d{3,4})',
                    r'(\d{1,2},?\d{3,4})\s*/\s*month',
                    r'(\d{1,2},?\d{3,4})\s*/\s*mo',
                    r'rent:\s*\$(\d{1,2},?\d{3,4})'
                ]
                
                for pattern in price_patterns:
                    matches = re.findall(pattern, price_text, re.IGNORECASE)
                    if matches:
                        price_str = matches[0].replace(',', '')
                        try:
                            price = float(price_str)
                            if (self.config.data_quality.min_price_threshold <= 
                                price <= self.config.data_quality.max_price_threshold):
                                return price
                        except ValueError:
                            continue
        
        return None
    
    def _extract_location(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property location"""
        for selector in self.selectors['location']:
            element = soup.select_one(selector)
            if element:
                location = self.clean_text(element.get_text())
                if location:
                    # Normalize location
                    location = re.sub(r'\s*,\s*United States$', '', location, flags=re.IGNORECASE)
                    return location
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract property description"""
        for selector in self.selectors['description']:
            element = soup.select_one(selector)
            if element:
                description = self.clean_text(element.get_text())
                if description and len(description) > 50:
                    return description
        
        return ""
    
    def _extract_bedrooms(self, soup: BeautifulSoup) -> int:
        """Extract number of bedrooms"""
        # Look for bedroom info in various sections
        text_to_search = soup.get_text().lower()
        
        bedroom_patterns = [
            r'(\d+)\s*(?:bed|bedroom|br)\b',
            r'\b(\d+)br\b',
            r'bedrooms?\s*:\s*(\d+)',
            r'(\d+)\s*bed\s*/',
            r'(\d+)\s*bedroom\s*apartment'
        ]
        
        for pattern in bedroom_patterns:
            match = re.search(pattern, text_to_search)
            if match:
                try:
                    bedrooms = int(match.group(1))
                    if 0 <= bedrooms <= self.config.data_quality.max_bedrooms:
                        return bedrooms
                except ValueError:
                    continue
        
        # Check for studio
        if 'studio' in text_to_search:
            return 0
        
        return 0
    
    def _extract_bathrooms(self, soup: BeautifulSoup) -> float:
        """Extract number of bathrooms"""
        text_to_search = soup.get_text().lower()
        
        bathroom_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)\b',
            r'\b(\d+(?:\.\d+)?)ba\b',
            r'bathrooms?\s*:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*bath\s*/',
            r'(\d+(?:\.\d+)?)\s*bathroom\s*apartment'
        ]
        
        for pattern in bathroom_patterns:
            match = re.search(pattern, text_to_search)
            if match:
                try:
                    bathrooms = float(match.group(1))
                    if 0 <= bathrooms <= self.config.data_quality.max_bathrooms:
                        return bathrooms
                except ValueError:
                    continue
        
        return 0.0
    
    def _extract_square_feet(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract square footage"""
        text_to_search = soup.get_text().lower()
        
        sqft_patterns = [
            r'(\d{1,4}(?:,\d{3})?)\s*(?:sq\.?\s*ft|square\s*feet|sqft)',
            r'(\d{1,4}(?:,\d{3})?)\s*sf\b',
            r'size:\s*(\d{1,4}(?:,\d{3})?)\s*sq'
        ]
        
        for pattern in sqft_patterns:
            match = re.search(pattern, text_to_search)
            if match:
                try:
                    sqft_str = match.group(1).replace(',', '')
                    sqft = int(sqft_str)
                    if 200 <= sqft <= 10000:
                        return sqft
                except ValueError:
                    continue
        
        return None
    
    def _extract_amenities(self, soup: BeautifulSoup) -> List[str]:
        """Extract amenities"""
        amenities = []
        
        for selector in self.selectors['amenities']:
            elements = soup.select(selector)
            for element in elements:
                amenity = self.clean_text(element.get_text())
                if amenity and len(amenity) > 2:
                    amenities.append(amenity.lower())
        
        # Look for amenities in description sections too
        amenity_sections = soup.select('.amenities-section, .features-section')
        for section in amenity_sections:
            text = section.get_text().lower()
            # Common amenity keywords
            amenity_keywords = [
                'pool', 'gym', 'fitness', 'parking', 'garage', 'laundry',
                'dishwasher', 'balcony', 'patio', 'air conditioning',
                'heating', 'hardwood', 'carpet', 'tile'
            ]
            
            for keyword in amenity_keywords:
                if keyword in text and keyword not in amenities:
                    amenities.append(keyword)
        
        # Normalize and limit amenities
        normalized_amenities = []
        for amenity in amenities:
            if 3 <= len(amenity) <= 30:
                normalized_amenities.append(amenity)
        
        return normalized_amenities[:15]
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract property images"""
        images = []
        
        for selector in self.selectors['images']:
            img_elements = soup.select(selector)
            for img in img_elements:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(self.source_config.base_url, src)
                    
                    if (src.startswith('http') and 
                        not any(skip in src for skip in ['placeholder', 'loading']) and
                        src not in images):
                        images.append(src)
        
        return images[:10]
    
    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {}
        
        # Phone number
        for selector in self.selectors['contact']:
            element = soup.select_one(selector)
            if element:
                phone = self.clean_text(element.get_text())
                phone_pattern = r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
                match = re.search(phone_pattern, phone)
                if match:
                    contact_info['phone'] = match.group(1)
                    break
        
        # Look for email in the page
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        page_text = soup.get_text()
        email_match = re.search(email_pattern, page_text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        return contact_info
    
    def _extract_property_type(self, soup: BeautifulSoup) -> str:
        """Extract property type"""
        page_text = soup.get_text().lower()
        
        if 'apartment' in page_text or 'apt' in page_text:
            return 'apartment'
        elif 'house' in page_text or 'home' in page_text:
            return 'house'
        elif 'condo' in page_text:
            return 'condo'
        elif 'studio' in page_text:
            return 'studio'
        elif 'townhouse' in page_text or 'townhome' in page_text:
            return 'townhouse'
        else:
            return 'apartment'