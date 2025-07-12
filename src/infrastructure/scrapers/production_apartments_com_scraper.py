"""
Production-ready Apartments.com scraper with comprehensive real-world handling.

This module provides a robust scraper for apartments.com that handles edge cases,
data variations, and production requirements for reliable data collection.
"""

import asyncio
import logging
import re
import json
from typing import List, Optional, AsyncGenerator, Dict, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs, quote_plus
from uuid import uuid4

from bs4 import BeautifulSoup, Tag
from .production_base_scraper import ProductionBaseScraper
from .config import get_config
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


class ProductionApartmentsComScraper(ProductionBaseScraper):
    """Production-ready scraper for apartments.com listings"""
    
    def __init__(self, config=None):
        super().__init__('apartments_com', config)
        
        # Enhanced selectors with fallbacks for layout changes
        self.selectors = {
            'listing_links': [
                'article.listing a[href*="/apartments/"]',
                '.property-link a[href*="/apartments/"]',
                '.property-card a[href*="/apartments/"]',
                '.listing-item a[href*="/apartments/"]',
                'a[data-testid="property-link"]',
                'a.property-url'
            ],
            'title': [
                'h1[data-testid="property-name"]',
                'h1.property-title',
                'h1.property-name',
                '.property-header h1',
                'h1.listing-title',
                '.title h1',
                'h1'
            ],
            'price': [
                '[data-testid="property-pricing"]',
                '.property-pricing',
                '.price-range',
                '.rent-price',
                '.pricing-info',
                '.property-price',
                '.listing-price'
            ],
            'location': [
                '[data-testid="property-address"]',
                '.property-address',
                '.listing-address',
                '.address-info',
                '.location',
                '.property-location'
            ],
            'description': [
                '[data-testid="property-description"]',
                '.property-description',
                '.listing-description',
                '.description-text',
                '.about-section',
                '.property-about'
            ],
            'amenities': [
                '.amenities-list li',
                '.features-list li',
                '.property-features li',
                '[data-testid="amenity"]',
                '.amenity-item',
                '.feature-item'
            ],
            'images': [
                '.property-photos img',
                '.gallery img',
                '.photo-gallery img',
                '.listing-images img',
                '.media-gallery img',
                '[data-testid="property-image"]'
            ],
            'contact': [
                '[data-testid="phone"]',
                '.phone-number',
                '.contact-phone',
                '.leasing-office-phone'
            ]
        }
        
        # Common apartment amenities mapping for normalization
        self.amenity_mapping = {
            'pool': ['pool', 'swimming pool', 'swimming', 'resort-style pool'],
            'gym': ['gym', 'fitness', 'fitness center', 'workout', 'exercise room'],
            'parking': ['parking', 'garage', 'covered parking', 'assigned parking', 'carport'],
            'pet-friendly': ['pet', 'pets allowed', 'pet friendly', 'dog park', 'pet play area'],
            'laundry': ['laundry', 'washer', 'dryer', 'in-unit laundry', 'laundry room'],
            'ac': ['air conditioning', 'ac', 'central air', 'climate control'],
            'dishwasher': ['dishwasher', 'dishwasher included'],
            'balcony': ['balcony', 'patio', 'terrace', 'deck'],
            'security': ['security', 'gated', 'secure entry', 'controlled access'],
            'elevator': ['elevator', 'elevators available'],
            'storage': ['storage', 'storage unit', 'extra storage'],
            'concierge': ['concierge', 'front desk', '24-hour staff'],
            'internet': ['internet', 'wifi', 'high-speed internet', 'fiber'],
            'hardwood': ['hardwood', 'hardwood floors', 'wood flooring'],
            'updated': ['updated', 'renovated', 'newly renovated', 'modern'],
            'utilities': ['utilities included', 'water included', 'heat included']
        }
    
    def get_search_urls(self) -> List[str]:
        """Get search URLs for different locations and property types"""
        search_urls = []
        base_url = self.source_config.base_url
        
        for location in self.source_config.search_locations:
            # Basic location search
            search_urls.append(f"{base_url}/{location}/")
            
            # Add specific property type searches for major cities
            if location in ['new-york-ny', 'los-angeles-ca', 'chicago-il', 'houston-tx']:
                # Apartment-specific search
                search_urls.append(f"{base_url}/{location}/?type=apartment")
                # Different price ranges
                search_urls.append(f"{base_url}/{location}/?min-price=1000&max-price=3000")
                search_urls.append(f"{base_url}/{location}/?min-price=3000&max-price=5000")
        
        logger.info(f"Generated {len(search_urls)} search URLs for apartments.com")
        return search_urls
    
    async def get_listing_urls(self, base_url: str, max_pages: int = None) -> AsyncGenerator[str, None]:
        """Get URLs of individual property listings from search results"""
        if max_pages is None:
            max_pages = self.source_config.max_pages
        
        page = 1
        seen_urls = set()
        consecutive_empty_pages = 0
        
        while page <= max_pages and consecutive_empty_pages < 3:
            try:
                # Construct search results URL
                search_url = self._build_search_url(base_url, page)
                
                logger.debug(f"Fetching search results page {page}: {search_url}")
                
                html_content = await self.fetch_page(search_url)
                if not html_content:
                    logger.warning(f"No content retrieved for page {page}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract listing links using multiple selectors
                listing_links = self._extract_listing_links(soup)
                
                if not listing_links:
                    logger.info(f"No listings found on page {page}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                consecutive_empty_pages = 0
                new_urls_found = 0
                
                for link in listing_links:
                    full_url = urljoin(self.source_config.base_url, link)
                    
                    # Normalize URL (remove query parameters that don't affect content)
                    normalized_url = self._normalize_listing_url(full_url)
                    
                    if normalized_url not in seen_urls:
                        seen_urls.add(normalized_url)
                        new_urls_found += 1
                        yield normalized_url
                
                logger.debug(f"Found {new_urls_found} new listings on page {page}")
                
                # Check if we've reached the end of results
                if self._is_last_page(soup) or new_urls_found == 0:
                    logger.info(f"Reached last page or no new listings at page {page}")
                    break
                
                page += 1
                
                # Add delay between pages to be respectful
                await asyncio.sleep(self.source_config.rate_limit.requests_per_second)
                
            except Exception as e:
                logger.error(f"Error processing search page {page}: {e}")
                consecutive_empty_pages += 1
                page += 1
    
    def _build_search_url(self, base_url: str, page: int) -> str:
        """Build search URL for specific page"""
        if page == 1:
            return base_url
        
        # Handle different URL patterns
        if '?' in base_url:
            return f"{base_url}&page={page}"
        else:
            if base_url.endswith('/'):
                return f"{base_url}{page}/"
            else:
                return f"{base_url}/{page}/"
    
    def _normalize_listing_url(self, url: str) -> str:
        """Normalize listing URL by removing tracking parameters"""
        parsed = urlparse(url)
        
        # Remove common tracking parameters
        query_params = parse_qs(parsed.query)
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'ref', 'source']
        
        for param in tracking_params:
            query_params.pop(param, None)
        
        # Rebuild query string
        clean_query = '&'.join([f"{k}={v[0]}" for k, v in query_params.items()])
        
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" + (f"?{clean_query}" if clean_query else "")
    
    def _extract_listing_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract property listing links using multiple selectors"""
        links = []
        
        for selector in self.selectors['listing_links']:
            try:
                elements = soup.select(selector)
                
                for element in elements:
                    href = element.get('href')
                    if href and '/apartments/' in href:
                        # Clean and validate the URL
                        if href.startswith('/'):
                            links.append(href)
                        elif href.startswith('http'):
                            parsed = urlparse(href)
                            if 'apartments.com' in parsed.netloc:
                                links.append(parsed.path + (f"?{parsed.query}" if parsed.query else ""))
                
                if links:  # If we found links with this selector, use them
                    break
                    
            except Exception as e:
                logger.warning(f"Error with selector '{selector}': {e}")
                continue
        
        # Remove duplicates while preserving order
        unique_links = []
        seen = set()
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        logger.debug(f"Extracted {len(unique_links)} unique listing links")
        return unique_links
    
    def _is_last_page(self, soup: BeautifulSoup) -> bool:
        """Check if this is the last page of results"""
        # Look for pagination indicators
        pagination_selectors = [
            '.pagination .next[disabled]',
            '.pagination .next.disabled',
            '.pagination .current:last-child',
            '.paging .next[disabled]'
        ]
        
        for selector in pagination_selectors:
            if soup.select(selector):
                return True
        
        # Look for "no more results" messages
        no_results_selectors = [
            '.no-results',
            '.no-listings',
            '[data-testid="no-results"]'
        ]
        
        for selector in no_results_selectors:
            if soup.select(selector):
                return True
        
        return False
    
    async def extract_property_data(self, listing_url: str, html_content: str) -> Optional[Property]:
        """Extract comprehensive property data from a listing page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize property data
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
            property_data['images'] = self._extract_images(soup, listing_url)
            property_data['contact_info'] = self._extract_contact_info(soup)
            property_data['property_type'] = self._extract_property_type(soup)
            
            # Extract structured data if available
            structured_data = self._extract_structured_data(soup)
            if structured_data:
                property_data.update(structured_data)
            
            # Extract coordinates if available
            coordinates = self._extract_coordinates(soup)
            if coordinates:
                property_data['latitude'], property_data['longitude'] = coordinates
            
            # Validate the extracted data
            is_valid, issues = self.validate_property_data(property_data)
            
            if not is_valid:
                logger.warning(f"Invalid property data for {listing_url}: {issues}")
                self.metrics.properties_invalid += 1
                return None
            
            # Calculate data quality score
            quality_score = self.calculate_data_quality_score(property_data)
            property_data['data_quality_score'] = quality_score
            
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
            
            # Add metadata fields if available
            if hasattr(property_entity, 'external_url'):
                property_entity.external_url = listing_url
            if hasattr(property_entity, 'data_quality_score'):
                property_entity.data_quality_score = quality_score
            if 'latitude' in property_data:
                if hasattr(property_entity, 'latitude'):
                    property_entity.latitude = property_data['latitude']
                    property_entity.longitude = property_data['longitude']
            
            self.metrics.properties_extracted += 1
            self.metrics.properties_valid += 1
            
            logger.debug(f"Successfully extracted property: {property_entity.title}")
            return property_entity
            
        except Exception as e:
            logger.error(f"Error extracting property data from {listing_url}: {e}")
            self.metrics.properties_invalid += 1
            return None
    
    def _extract_with_selectors(self, soup: BeautifulSoup, selector_list: List[str]) -> Optional[str]:
        """Extract text using multiple selectors as fallbacks"""
        for selector in selector_list:
            try:
                element = soup.select_one(selector)
                if element:
                    text = self.clean_text(element.get_text())
                    if text:
                        return text
            except Exception as e:
                logger.debug(f"Error with selector '{selector}': {e}")
                continue
        return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property title with enhanced fallbacks"""
        title = self._extract_with_selectors(soup, self.selectors['title'])
        
        # If no title found, try to construct from other elements
        if not title:
            # Try to get from meta tags
            meta_title = soup.find('meta', {'property': 'og:title'})
            if meta_title:
                title = self.clean_text(meta_title.get('content', ''))
            
            # Try page title as last resort
            if not title:
                page_title = soup.find('title')
                if page_title:
                    title = self.clean_text(page_title.get_text())
                    # Remove site name from title
                    title = re.sub(r'\s*\|\s*Apartments\.com.*$', '', title, flags=re.IGNORECASE)
        
        return title
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract property price with comprehensive parsing"""
        # Try main price selectors
        for selector in self.selectors['price']:
            elements = soup.select(selector)
            for element in elements:
                price_text = self.clean_text(element.get_text())
                
                # Multiple price patterns
                price_patterns = [
                    r'\$(\d{1,2},?\d{3,4})',  # $1,234 or $1234
                    r'(\d{1,2},?\d{3,4})\s*\/?\s*mo',  # 1234/mo or 1234 mo
                    r'starting\s+at\s+\$(\d{1,2},?\d{3,4})',  # starting at $1234
                    r'from\s+\$(\d{1,2},?\d{3,4})'  # from $1234
                ]
                
                for pattern in price_patterns:
                    matches = re.findall(pattern, price_text, re.IGNORECASE)
                    if matches:
                        price_str = matches[0].replace(',', '')
                        try:
                            price = float(price_str)
                            # Validate price range
                            if (self.config.data_quality.min_price_threshold <= 
                                price <= self.config.data_quality.max_price_threshold):
                                return price
                        except ValueError:
                            continue
        
        # Try structured data
        structured_price = self._extract_price_from_structured_data(soup)
        if structured_price:
            return structured_price
        
        return None
    
    def _extract_price_from_structured_data(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract price from structured data (JSON-LD)"""
        try:
            script_tags = soup.find_all('script', {'type': 'application/ld+json'})
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        # Look for price in various schema.org formats
                        price_fields = ['price', 'priceRange', 'offers']
                        for field in price_fields:
                            if field in data:
                                price_data = data[field]
                                if isinstance(price_data, (int, float)):
                                    return float(price_data)
                                elif isinstance(price_data, str):
                                    price = self.extract_price(price_data)
                                    if price:
                                        return price
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Error extracting price from structured data: {e}")
        
        return None
    
    def _extract_location(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract property location with normalization"""
        location = self._extract_with_selectors(soup, self.selectors['location'])
        
        if not location:
            # Try meta tags
            meta_location = soup.find('meta', {'property': 'og:street-address'})
            if meta_location:
                location = self.clean_text(meta_location.get('content', ''))
        
        if location:
            # Normalize location format
            location = self.clean_text(location)
            # Remove redundant text
            location = re.sub(r'\s*,\s*United States$', '', location, flags=re.IGNORECASE)
            location = re.sub(r'\s*,\s*USA$', '', location, flags=re.IGNORECASE)
            return location
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract property description with content aggregation"""
        description = self._extract_with_selectors(soup, self.selectors['description'])
        
        if not description or len(description) < 50:
            # Try to aggregate description from multiple sections
            description_parts = []
            
            # Look for common description sections
            desc_sections = [
                '.about-section',
                '.description-section',
                '.property-details',
                '.overview-section'
            ]
            
            for selector in desc_sections:
                elements = soup.select(selector)
                for element in elements:
                    text = self.clean_text(element.get_text())
                    if text and len(text) > 20:
                        description_parts.append(text)
            
            if description_parts:
                description = ' '.join(description_parts)
        
        # Clean up description
        if description:
            # Remove excessive whitespace
            description = re.sub(r'\s+', ' ', description)
            # Remove promotional text patterns
            description = re.sub(r'call today|contact us|schedule a tour', '', description, flags=re.IGNORECASE)
            description = description.strip()
        
        return description or ""
    
    def _extract_bedrooms(self, soup: BeautifulSoup) -> int:
        """Extract number of bedrooms with multiple parsing strategies"""
        # Look in structured areas first
        spec_selectors = [
            '.bed-bath-info',
            '.property-specs',
            '.unit-info',
            '.floorplan-info',
            '.specs-section'
        ]
        
        bedroom_patterns = [
            r'(\d+)\s*(?:bed|bedroom|br)\b',
            r'\b(\d+)br\b',
            r'(\d+)\s*(?:bed|bedroom)',
            r'beds?\s*:\s*(\d+)',
            r'bedroom\s*:\s*(\d+)'
        ]
        
        for selector in spec_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().lower()
                for pattern in bedroom_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            bedrooms = int(match.group(1))
                            if 0 <= bedrooms <= self.config.data_quality.max_bedrooms:
                                return bedrooms
                        except ValueError:
                            continue
        
        # Try structured data
        bedrooms = self._extract_bedrooms_from_structured_data(soup)
        if bedrooms is not None:
            return bedrooms
        
        # Special handling for studio apartments
        page_text = soup.get_text().lower()
        if 'studio' in page_text:
            return 0
        
        return 0
    
    def _extract_bedrooms_from_structured_data(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract bedrooms from structured data"""
        try:
            script_tags = soup.find_all('script', {'type': 'application/ld+json'})
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        bedroom_fields = ['numberOfRooms', 'numberOfBedrooms', 'bedrooms']
                        for field in bedroom_fields:
                            if field in data:
                                try:
                                    return int(data[field])
                                except (ValueError, TypeError):
                                    continue
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        return None
    
    def _extract_bathrooms(self, soup: BeautifulSoup) -> float:
        """Extract number of bathrooms"""
        bathroom_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba)\b',
            r'\b(\d+(?:\.\d+)?)ba\b',
            r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom)',
            r'bath\s*:\s*(\d+(?:\.\d+)?)',
            r'bathroom\s*:\s*(\d+(?:\.\d+)?)'
        ]
        
        spec_selectors = [
            '.bed-bath-info',
            '.property-specs',
            '.unit-info',
            '.floorplan-info'
        ]
        
        for selector in spec_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().lower()
                for pattern in bathroom_patterns:
                    match = re.search(pattern, text)
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
        sqft_patterns = [
            r'(\d{1,4}(?:,\d{3})?)\s*(?:sq\.?\s*ft|square\s*feet|sqft)',
            r'(\d{1,4}(?:,\d{3})?)\s*sf\b',
            r'size\s*:\s*(\d{1,4}(?:,\d{3})?)\s*sq',
            r'(\d{1,4}(?:,\d{3})?) square feet'
        ]
        
        spec_selectors = [
            '.property-specs',
            '.unit-info',
            '.floorplan-info',
            '.size-info',
            '.square-feet'
        ]
        
        for selector in spec_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().lower()
                for pattern in sqft_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            sqft_str = match.group(1).replace(',', '')
                            sqft = int(sqft_str)
                            if 200 <= sqft <= 10000:  # Reasonable bounds
                                return sqft
                        except ValueError:
                            continue
        
        return None
    
    def _extract_amenities(self, soup: BeautifulSoup) -> List[str]:
        """Extract and normalize amenities"""
        amenities = []
        
        for selector in self.selectors['amenities']:
            elements = soup.select(selector)
            for element in elements:
                amenity = self.clean_text(element.get_text())
                if amenity and len(amenity) > 2:
                    amenities.append(amenity.lower())
        
        # Also check for amenities in description text
        desc_elements = soup.select('.description, .amenities-section, .features-section')
        for element in desc_elements:
            text = element.get_text().lower()
            # Look for common amenity keywords
            for normalized, variants in self.amenity_mapping.items():
                for variant in variants:
                    if variant in text and normalized not in amenities:
                        amenities.append(normalized)
                        break
        
        # Normalize amenities
        normalized_amenities = []
        for amenity in amenities:
            normalized = self._normalize_amenity(amenity)
            if normalized and normalized not in normalized_amenities:
                normalized_amenities.append(normalized)
        
        return normalized_amenities[:20]  # Limit to 20 amenities
    
    def _normalize_amenity(self, amenity: str) -> Optional[str]:
        """Normalize amenity text"""
        amenity = amenity.lower().strip()
        
        # Map to standard amenities
        for normalized, variants in self.amenity_mapping.items():
            for variant in variants:
                if variant in amenity:
                    return normalized
        
        # Return original if it's a reasonable length
        if 3 <= len(amenity) <= 30:
            return amenity
        
        return None
    
    def _extract_images(self, soup: BeautifulSoup, listing_url: str) -> List[str]:
        """Extract property images"""
        images = []
        
        for selector in self.selectors['images']:
            img_elements = soup.select(selector)
            for img in img_elements:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(self.source_config.base_url, src)
                    
                    # Only include high-quality images
                    if (src.startswith('http') and 
                        not any(skip in src for skip in ['placeholder', 'loading', 'icon', 'logo']) and
                        src not in images):
                        images.append(src)
        
        return images[:15]  # Limit to 15 images
    
    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {}
        
        # Phone number
        for selector in self.selectors['contact']:
            element = soup.select_one(selector)
            if element:
                phone = self.clean_text(element.get_text())
                # Validate phone number format
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
        
        type_indicators = {
            'apartment': ['apartment', 'apt'],
            'house': ['house', 'home', 'single family'],
            'condo': ['condo', 'condominium'],
            'studio': ['studio'],
            'townhouse': ['townhouse', 'townhome', 'town home'],
            'loft': ['loft']
        }
        
        for prop_type, indicators in type_indicators.items():
            if any(indicator in page_text for indicator in indicators):
                return prop_type
        
        return 'apartment'  # Default
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data (JSON-LD) for additional property information"""
        structured_data = {}
        
        try:
            script_tags = soup.find_all('script', {'type': 'application/ld+json'})
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get('@type') in ['Apartment', 'House', 'Residence']:
                        # Extract additional fields
                        field_mapping = {
                            'name': 'title',
                            'description': 'description',
                            'address': 'location',
                            'geo': 'coordinates'
                        }
                        
                        for json_field, prop_field in field_mapping.items():
                            if json_field in data and not structured_data.get(prop_field):
                                structured_data[prop_field] = data[json_field]
                        
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Error extracting structured data: {e}")
        
        return structured_data
    
    def _extract_coordinates(self, soup: BeautifulSoup) -> Optional[tuple[float, float]]:
        """Extract latitude and longitude coordinates"""
        try:
            # Look in structured data first
            script_tags = soup.find_all('script', {'type': 'application/ld+json'})
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'geo' in data:
                        geo = data['geo']
                        if isinstance(geo, dict):
                            lat = geo.get('latitude')
                            lng = geo.get('longitude')
                            if lat and lng:
                                return float(lat), float(lng)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
            
            # Look for coordinates in script tags or data attributes
            coord_patterns = [
                r'latitude["\']?\s*:\s*([+-]?\d+\.?\d*)',
                r'lat["\']?\s*:\s*([+-]?\d+\.?\d*)',
                r'longitude["\']?\s*:\s*([+-]?\d+\.?\d*)',
                r'lng["\']?\s*:\s*([+-]?\d+\.?\d*)'
            ]
            
            page_text = soup.get_text()
            lat_match = re.search(coord_patterns[0], page_text) or re.search(coord_patterns[1], page_text)
            lng_match = re.search(coord_patterns[2], page_text) or re.search(coord_patterns[3], page_text)
            
            if lat_match and lng_match:
                try:
                    lat = float(lat_match.group(1))
                    lng = float(lng_match.group(1))
                    # Validate coordinates are reasonable
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        return lat, lng
                except ValueError:
                    pass
            
        except Exception as e:
            logger.debug(f"Error extracting coordinates: {e}")
        
        return None