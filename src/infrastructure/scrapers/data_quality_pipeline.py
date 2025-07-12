"""
Data quality and normalization pipeline for scraped property data.

This module provides comprehensive data cleaning, validation, and normalization
for property data from various sources to ensure consistency and quality.
"""

import re
import logging
import unicodedata
import phonenumbers
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urlparse
from email_validator import validate_email, EmailNotValidError
import geopy.geocoders as geocoders
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from .config import get_config
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    warnings: List[str]
    cleaned_data: Dict[str, Any]


@dataclass
class NormalizationRule:
    """Rule for data normalization"""
    field: str
    rule_type: str  # 'replace', 'regex', 'mapping', 'format'
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    mapping: Optional[Dict[str, str]] = None
    options: Optional[Dict[str, Any]] = None


class TextNormalizer:
    """Text normalization utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove unwanted characters
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '-')  # Em dash
        text = text.replace('\u2018', "'")  # Left single quote
        text = text.replace('\u2019', "'")  # Right single quote
        text = text.replace('\u201c', '"')  # Left double quote
        text = text.replace('\u201d', '"')  # Right double quote
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return text.strip()
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize property title"""
        if not title:
            return ""
        
        title = TextNormalizer.clean_text(title)
        
        # Remove excessive punctuation
        title = re.sub(r'[!]{2,}', '!', title)
        title = re.sub(r'[?]{2,}', '?', title)
        title = re.sub(r'\.{3,}', '...', title)
        
        # Remove promotional text
        promotional_patterns = [
            r'\b(call today|contact us|schedule a tour|move in special)\b',
            r'\b(special offer|limited time|act now|don\'t miss)\b',
            r'\$\d+\s*(off|discount|special)',
            r'\b(luxury|premium|upscale)\s*(?=apartment|home|condo)',
        ]
        
        for pattern in promotional_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Clean up resulting text
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    @staticmethod
    def normalize_description(description: str) -> str:
        """Normalize property description"""
        if not description:
            return ""
        
        description = TextNormalizer.clean_text(description)
        
        # Remove excessive repetition
        description = re.sub(r'\b(\w+)\s+\1\b', r'\1', description, flags=re.IGNORECASE)
        
        # Remove contact instructions
        contact_patterns = [
            r'call\s+\d{3}[-.]?\d{3}[-.]?\d{4}',
            r'contact\s+us\s+at\s+\d{3}[-.]?\d{3}[-.]?\d{4}',
            r'for\s+more\s+information\s+call',
            r'schedule\s+a\s+tour\s+today',
            r'visit\s+our\s+website'
        ]
        
        for pattern in contact_patterns:
            description = re.sub(pattern, '', description, flags=re.IGNORECASE)
        
        # Normalize common abbreviations
        abbreviations = {
            r'\bw/d\b': 'washer/dryer',
            r'\ba/c\b': 'air conditioning',
            r'\bhw\s*fls\b': 'hardwood floors',
            r'\bss\s*appl\b': 'stainless steel appliances',
            r'\bgr\s*flr\b': 'ground floor',
            r'\buplvl\b': 'upper level'
        }
        
        for pattern, replacement in abbreviations.items():
            description = re.sub(pattern, replacement, description, flags=re.IGNORECASE)
        
        return description
    
    @staticmethod
    def normalize_location(location: str) -> str:
        """Normalize location string"""
        if not location:
            return ""
        
        location = TextNormalizer.clean_text(location)
        
        # Standardize state abbreviations
        state_abbreviations = {
            'new york': 'NY', 'california': 'CA', 'texas': 'TX', 'florida': 'FL',
            'illinois': 'IL', 'pennsylvania': 'PA', 'ohio': 'OH', 'georgia': 'GA',
            'north carolina': 'NC', 'michigan': 'MI', 'new jersey': 'NJ',
            'virginia': 'VA', 'washington': 'WA', 'arizona': 'AZ', 'massachusetts': 'MA',
            'tennessee': 'TN', 'indiana': 'IN', 'missouri': 'MO', 'maryland': 'MD',
            'wisconsin': 'WI', 'colorado': 'CO', 'minnesota': 'MN', 'south carolina': 'SC',
            'alabama': 'AL', 'louisiana': 'LA', 'kentucky': 'KY', 'oregon': 'OR',
            'oklahoma': 'OK', 'connecticut': 'CT', 'utah': 'UT', 'iowa': 'IA',
            'nevada': 'NV', 'arkansas': 'AR', 'mississippi': 'MS', 'kansas': 'KS',
            'new mexico': 'NM', 'nebraska': 'NE', 'west virginia': 'WV',
            'idaho': 'ID', 'hawaii': 'HI', 'new hampshire': 'NH', 'maine': 'ME',
            'montana': 'MT', 'rhode island': 'RI', 'delaware': 'DE',
            'south dakota': 'SD', 'north dakota': 'ND', 'alaska': 'AK',
            'vermont': 'VT', 'wyoming': 'WY'
        }
        
        # Replace full state names with abbreviations
        for full_name, abbrev in state_abbreviations.items():
            pattern = rf'\b{re.escape(full_name)}\b'
            location = re.sub(pattern, abbrev, location, flags=re.IGNORECASE)
        
        # Remove redundant country information
        location = re.sub(r',?\s*United States?$', '', location, flags=re.IGNORECASE)
        location = re.sub(r',?\s*USA?$', '', location, flags=re.IGNORECASE)
        
        # Standardize formatting
        location = re.sub(r'\s*,\s*', ', ', location)
        
        return location


class PriceNormalizer:
    """Price normalization utilities"""
    
    @staticmethod
    def extract_and_normalize_price(price_text: str) -> Optional[float]:
        """Extract and normalize price from text"""
        if not price_text:
            return None
        
        # Remove currency symbols and normalize
        price_text = re.sub(r'[^\d,.-]', '', price_text)
        
        # Handle ranges (take the lower value)
        if '-' in price_text:
            parts = price_text.split('-')
            if parts:
                price_text = parts[0].strip()
        
        # Remove commas
        price_text = price_text.replace(',', '')
        
        try:
            price = float(price_text)
            
            # Validate reasonable price range
            if 50 <= price <= 100000:
                return price
                
        except ValueError:
            pass
        
        return None
    
    @staticmethod
    def validate_price_consistency(price: float, bedrooms: int, location: str) -> List[str]:
        """Validate price consistency against market expectations"""
        issues = []
        
        # Market-based price validation (simplified)
        market_ranges = {
            'NY': {'studio': (1500, 4000), '1br': (2000, 5000), '2br': (3000, 8000), '3br': (4000, 12000)},
            'CA': {'studio': (1200, 3500), '1br': (1800, 4500), '2br': (2500, 6500), '3br': (3500, 9000)},
            'TX': {'studio': (800, 2000), '1br': (1000, 2500), '2br': (1400, 3500), '3br': (1800, 4500)},
            'FL': {'studio': (900, 2200), '1br': (1200, 2800), '2br': (1600, 3800), '3br': (2000, 5000)}
        }
        
        # Extract state from location
        state_match = re.search(r'\b([A-Z]{2})\b', location.upper())
        if not state_match:
            return issues
        
        state = state_match.group(1)
        if state not in market_ranges:
            return issues
        
        # Determine bedroom category
        if bedrooms == 0:
            bedroom_cat = 'studio'
        elif bedrooms == 1:
            bedroom_cat = '1br'
        elif bedrooms == 2:
            bedroom_cat = '2br'
        elif bedrooms >= 3:
            bedroom_cat = '3br'
        else:
            return issues
        
        # Check price range
        if bedroom_cat in market_ranges[state]:
            min_price, max_price = market_ranges[state][bedroom_cat]
            if price < min_price * 0.5:
                issues.append(f"Price ${price} unusually low for {bedrooms}br in {state}")
            elif price > max_price * 2:
                issues.append(f"Price ${price} unusually high for {bedrooms}br in {state}")
        
        return issues


class AmenityNormalizer:
    """Amenity normalization utilities"""
    
    def __init__(self):
        self.standard_amenities = {
            'pool': ['pool', 'swimming pool', 'resort-style pool', 'outdoor pool', 'indoor pool'],
            'gym': ['gym', 'fitness', 'fitness center', 'workout room', 'exercise room'],
            'parking': ['parking', 'garage', 'covered parking', 'assigned parking', 'carport'],
            'pet-friendly': ['pet friendly', 'pets allowed', 'dog park', 'pet play area'],
            'laundry': ['laundry', 'washer/dryer', 'in-unit laundry', 'laundry room', 'w/d'],
            'ac': ['air conditioning', 'a/c', 'central air', 'climate control'],
            'dishwasher': ['dishwasher', 'dishwasher included'],
            'balcony': ['balcony', 'patio', 'terrace', 'deck', 'outdoor space'],
            'security': ['security', 'gated', 'secure entry', 'controlled access', 'doorman'],
            'elevator': ['elevator', 'elevators', 'lift'],
            'storage': ['storage', 'storage unit', 'extra storage', 'closet space'],
            'hardwood': ['hardwood floors', 'hardwood', 'wood flooring', 'hardwood flooring'],
            'carpet': ['carpet', 'carpeted', 'carpeting'],
            'tile': ['tile', 'tile flooring', 'ceramic tile'],
            'stainless': ['stainless steel appliances', 'stainless appliances', 'ss appliances'],
            'updated': ['updated', 'renovated', 'newly renovated', 'modern', 'remodeled'],
            'utilities': ['utilities included', 'heat included', 'water included', 'electric included'],
            'internet': ['internet', 'wifi', 'high-speed internet', 'fiber', 'cable ready'],
            'concierge': ['concierge', 'front desk', '24-hour staff', 'doorman'],
            'rooftop': ['rooftop', 'roof deck', 'rooftop terrace', 'rooftop access']
        }
    
    def normalize_amenities(self, amenities: List[str]) -> List[str]:
        """Normalize amenity list to standard terms"""
        if not amenities:
            return []
        
        normalized = []
        seen = set()
        
        for amenity in amenities:
            if not amenity or len(amenity) < 3:
                continue
            
            amenity_lower = amenity.lower().strip()
            
            # Find matching standard amenity
            matched = False
            for standard, variants in self.standard_amenities.items():
                for variant in variants:
                    if variant in amenity_lower:
                        if standard not in seen:
                            normalized.append(standard)
                            seen.add(standard)
                        matched = True
                        break
                if matched:
                    break
            
            # If no match found, keep original if it's reasonable
            if not matched and 3 <= len(amenity_lower) <= 50:
                # Remove common junk words
                junk_words = ['available', 'included', 'feature', 'option']
                clean_amenity = amenity_lower
                for junk in junk_words:
                    clean_amenity = clean_amenity.replace(junk, '').strip()
                
                if len(clean_amenity) >= 3 and clean_amenity not in seen:
                    normalized.append(clean_amenity)
                    seen.add(clean_amenity)
        
        return normalized[:20]  # Limit to 20 amenities


class ContactInfoValidator:
    """Contact information validation utilities"""
    
    @staticmethod
    def validate_phone_number(phone: str) -> Optional[str]:
        """Validate and normalize phone number"""
        if not phone:
            return None
        
        try:
            # Parse US phone number
            parsed = phonenumbers.parse(phone, "US")
            if phonenumbers.is_valid_number(parsed):
                # Format as standard US format
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)
        except:
            pass
        
        # Fallback: basic regex validation
        phone_clean = re.sub(r'[^\d]', '', phone)
        if len(phone_clean) == 10:
            return f"({phone_clean[:3]}) {phone_clean[3:6]}-{phone_clean[6:]}"
        elif len(phone_clean) == 11 and phone_clean[0] == '1':
            return f"({phone_clean[1:4]}) {phone_clean[4:7]}-{phone_clean[7:]}"
        
        return None
    
    @staticmethod
    def validate_email(email: str) -> Optional[str]:
        """Validate email address"""
        if not email:
            return None
        
        try:
            # Use email-validator library
            valid = validate_email(email)
            return valid.email
        except EmailNotValidError:
            return None


class DataQualityPipeline:
    """Comprehensive data quality and normalization pipeline"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.text_normalizer = TextNormalizer()
        self.price_normalizer = PriceNormalizer()
        self.amenity_normalizer = AmenityNormalizer()
        self.contact_validator = ContactInfoValidator()
    
    def process_property(self, property_data: Dict[str, Any]) -> ValidationResult:
        """Process and validate property data through the complete pipeline"""
        
        issues = []
        warnings = []
        cleaned_data = property_data.copy()
        
        # Text normalization
        cleaned_data = self._normalize_text_fields(cleaned_data)
        
        # Price validation and normalization
        price_issues, cleaned_data = self._process_price(cleaned_data)
        issues.extend(price_issues)
        
        # Location normalization
        cleaned_data = self._normalize_location_field(cleaned_data)
        
        # Amenity normalization
        cleaned_data = self._normalize_amenities_field(cleaned_data)
        
        # Contact info validation
        contact_warnings, cleaned_data = self._validate_contact_info(cleaned_data)
        warnings.extend(contact_warnings)
        
        # Specification validation
        spec_issues, cleaned_data = self._validate_specifications(cleaned_data)
        issues.extend(spec_issues)
        
        # Cross-field validation
        cross_issues = self._cross_validate_fields(cleaned_data)
        issues.extend(cross_issues)
        
        # Calculate quality score
        score = self._calculate_quality_score(cleaned_data, issues, warnings)
        
        # Final validation
        is_valid = self._is_data_valid(cleaned_data, issues)
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            cleaned_data=cleaned_data
        )
    
    def _normalize_text_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize text fields"""
        if 'title' in data:
            data['title'] = self.text_normalizer.normalize_title(data['title'])
        
        if 'description' in data:
            data['description'] = self.text_normalizer.normalize_description(data['description'])
        
        return data
    
    def _process_price(self, data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Process and validate price"""
        issues = []
        
        if 'price' in data:
            if isinstance(data['price'], str):
                normalized_price = self.price_normalizer.extract_and_normalize_price(data['price'])
                if normalized_price:
                    data['price'] = normalized_price
                else:
                    issues.append("Could not parse price")
            
            if isinstance(data['price'], (int, float)):
                # Validate price consistency
                bedrooms = data.get('bedrooms', 0)
                location = data.get('location', '')
                price_issues = self.price_normalizer.validate_price_consistency(
                    data['price'], bedrooms, location
                )
                issues.extend(price_issues)
        
        return issues, data
    
    def _normalize_location_field(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize location field"""
        if 'location' in data:
            data['location'] = self.text_normalizer.normalize_location(data['location'])
        
        return data
    
    def _normalize_amenities_field(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize amenities field"""
        if 'amenities' in data and isinstance(data['amenities'], list):
            data['amenities'] = self.amenity_normalizer.normalize_amenities(data['amenities'])
        
        return data
    
    def _validate_contact_info(self, data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Validate contact information"""
        warnings = []
        
        if 'contact_info' in data and isinstance(data['contact_info'], dict):
            contact_info = data['contact_info'].copy()
            
            # Validate phone
            if 'phone' in contact_info:
                validated_phone = self.contact_validator.validate_phone_number(contact_info['phone'])
                if validated_phone:
                    contact_info['phone'] = validated_phone
                else:
                    warnings.append("Invalid phone number format")
                    del contact_info['phone']
            
            # Validate email
            if 'email' in contact_info:
                validated_email = self.contact_validator.validate_email(contact_info['email'])
                if validated_email:
                    contact_info['email'] = validated_email
                else:
                    warnings.append("Invalid email address")
                    del contact_info['email']
            
            data['contact_info'] = contact_info
        
        return warnings, data
    
    def _validate_specifications(self, data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Validate property specifications"""
        issues = []
        
        # Validate bedrooms
        if 'bedrooms' in data:
            bedrooms = data['bedrooms']
            if not isinstance(bedrooms, int) or bedrooms < 0 or bedrooms > 10:
                issues.append(f"Invalid bedrooms value: {bedrooms}")
        
        # Validate bathrooms
        if 'bathrooms' in data:
            bathrooms = data['bathrooms']
            if not isinstance(bathrooms, (int, float)) or bathrooms < 0 or bathrooms > 10:
                issues.append(f"Invalid bathrooms value: {bathrooms}")
        
        # Validate square feet
        if 'square_feet' in data and data['square_feet'] is not None:
            sqft = data['square_feet']
            if not isinstance(sqft, int) or sqft < 100 or sqft > 20000:
                issues.append(f"Invalid square feet value: {sqft}")
        
        return issues, data
    
    def _cross_validate_fields(self, data: Dict[str, Any]) -> List[str]:
        """Cross-validate fields for consistency"""
        issues = []
        
        # Check price per square foot reasonableness
        if 'price' in data and 'square_feet' in data and data['square_feet']:
            price_per_sqft = data['price'] / data['square_feet']
            if price_per_sqft < 0.5 or price_per_sqft > 20:
                issues.append(f"Unusual price per sq ft: ${price_per_sqft:.2f}")
        
        # Check bedroom/bathroom ratio
        bedrooms = data.get('bedrooms', 0)
        bathrooms = data.get('bathrooms', 0)
        if bedrooms > 0 and bathrooms > 0:
            if bathrooms > bedrooms * 2:
                issues.append("Unusual bathroom to bedroom ratio")
        
        return issues
    
    def _calculate_quality_score(
        self, 
        data: Dict[str, Any], 
        issues: List[str], 
        warnings: List[str]
    ) -> float:
        """Calculate data quality score"""
        score = 1.0
        
        # Deduct for missing required fields
        required_fields = ['title', 'price', 'location']
        for field in required_fields:
            if not data.get(field):
                score -= 0.2
        
        # Deduct for issues and warnings
        score -= len(issues) * 0.1
        score -= len(warnings) * 0.05
        
        # Add points for optional fields
        optional_fields = ['description', 'square_feet', 'amenities', 'contact_info', 'images']
        for field in optional_fields:
            if data.get(field):
                score += 0.05
        
        # Bonus for complete contact info
        if (data.get('contact_info', {}).get('phone') and 
            data.get('contact_info', {}).get('email')):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _is_data_valid(self, data: Dict[str, Any], issues: List[str]) -> bool:
        """Determine if data meets minimum quality standards"""
        
        # Must have required fields
        required_fields = ['title', 'price', 'location']
        for field in required_fields:
            if not data.get(field):
                return False
        
        # Must not have critical issues
        critical_keywords = ['invalid price', 'could not parse', 'missing required']
        for issue in issues:
            if any(keyword in issue.lower() for keyword in critical_keywords):
                return False
        
        # Price must be reasonable
        price = data.get('price', 0)
        if not (50 <= price <= 100000):
            return False
        
        return True