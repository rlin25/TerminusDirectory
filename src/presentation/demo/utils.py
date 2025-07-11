"""
Utility functions for Rental ML System Demo

This module provides helper functions for the demo application:
- Data formatting and transformation
- Calculation utilities
- Visualization helpers
- File processing functions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import math
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
import streamlit as st

from src.domain.entities.property import Property
from src.domain.entities.user import User


def format_price(price: float, include_symbol: bool = True) -> str:
    """
    Format price for display
    
    Args:
        price: Price value
        include_symbol: Whether to include $ symbol
        
    Returns:
        Formatted price string
    """
    if price >= 1000000:
        formatted = f"{price / 1000000:.1f}M"
    elif price >= 1000:
        formatted = f"{price / 1000:.0f}K"
    else:
        formatted = f"{price:.0f}"
    
    return f"${formatted}" if include_symbol else formatted


def format_number(number: float, decimal_places: int = 0) -> str:
    """
    Format large numbers with appropriate suffixes
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if number >= 1000000000:
        return f"{number / 1000000000:.{decimal_places}f}B"
    elif number >= 1000000:
        return f"{number / 1000000:.{decimal_places}f}M"
    elif number >= 1000:
        return f"{number / 1000:.{decimal_places}f}K"
    else:
        return f"{number:.{decimal_places}f}"


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two coordinates using Haversine formula
    
    Args:
        lat1, lon1: First coordinate
        lat2, lon2: Second coordinate
        
    Returns:
        Distance in miles
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    
    return c * r


def generate_map_data(properties: List[Property], 
                     center_lat: float = 40.7128, 
                     center_lon: float = -74.0060) -> List[Dict[str, Any]]:
    """
    Generate map data for properties with simulated coordinates
    
    Args:
        properties: List of properties
        center_lat: Center latitude for coordinate generation
        center_lon: Center longitude for coordinate generation
        
    Returns:
        List of property map data
    """
    map_data = []
    
    # Location-based coordinate offsets for consistent positioning
    location_offsets = {
        "Downtown": (0.002, -0.001),
        "Midtown": (0.008, 0.003),
        "Uptown": (0.015, 0.002),
        "Financial District": (-0.003, -0.008),
        "Arts District": (0.005, -0.012),
        "Suburban Heights": (0.020, 0.015),
        "Riverside": (-0.010, 0.008),
        "Hillside": (0.012, -0.005),
        "University Area": (0.007, 0.010),
        "Old Town": (-0.005, -0.003),
        "Tech Quarter": (0.010, -0.015),
        "Green Valley": (0.025, 0.008),
        "Sunset Park": (-0.015, 0.012),
        "Brookside": (0.018, -0.010),
        "Central Plaza": (0.001, 0.001),
        "Marina District": (-0.008, 0.005)
    }
    
    for prop in properties:
        # Get base offset for location
        base_offset = location_offsets.get(prop.location, (0, 0))
        
        # Add some randomness around the base location
        lat_offset = base_offset[0] + np.random.normal(0, 0.002)
        lon_offset = base_offset[1] + np.random.normal(0, 0.002)
        
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        
        map_data.append({
            'property_id': str(prop.id),
            'lat': lat,
            'lon': lon,
            'title': prop.title,
            'price': prop.price,
            'location': prop.location,
            'bedrooms': prop.bedrooms,
            'bathrooms': prop.bathrooms,
            'square_feet': prop.square_feet,
            'property_type': prop.property_type,
            'amenities_count': len(prop.amenities),
            'is_active': prop.is_active,
            'price_per_sqft': prop.get_price_per_sqft()
        })
    
    return map_data


def calculate_property_score(property_obj: Property, user_preferences: Optional[Any] = None) -> float:
    """
    Calculate a simple property score based on various factors
    
    Args:
        property_obj: Property to score
        user_preferences: Optional user preferences for personalized scoring
        
    Returns:
        Score between 0 and 1
    """
    score = 0.5  # Base score
    
    # Price scoring (lower prices get higher scores)
    if property_obj.price <= 2000:
        score += 0.2
    elif property_obj.price <= 3000:
        score += 0.1
    elif property_obj.price >= 5000:
        score -= 0.1
    
    # Size scoring
    if property_obj.square_feet:
        if property_obj.square_feet >= 1000:
            score += 0.15
        elif property_obj.square_feet >= 800:
            score += 0.1
        elif property_obj.square_feet < 500:
            score -= 0.1
    
    # Amenities scoring
    amenity_count = len(property_obj.amenities)
    if amenity_count >= 5:
        score += 0.2
    elif amenity_count >= 3:
        score += 0.1
    
    # Bedroom/bathroom balance
    if property_obj.bedrooms > 0:
        bathroom_ratio = property_obj.bathrooms / property_obj.bedrooms
        if 0.5 <= bathroom_ratio <= 1.5:
            score += 0.1
    
    # Active status
    if not property_obj.is_active:
        score -= 0.3
    
    # User preference matching
    if user_preferences:
        if hasattr(user_preferences, 'min_price') and hasattr(user_preferences, 'max_price'):
            if (user_preferences.min_price and property_obj.price < user_preferences.min_price) or \
               (user_preferences.max_price and property_obj.price > user_preferences.max_price):
                score -= 0.2
        
        if hasattr(user_preferences, 'preferred_locations'):
            if user_preferences.preferred_locations and property_obj.location in user_preferences.preferred_locations:
                score += 0.15
    
    # Ensure score is between 0 and 1
    return max(0, min(1, score))


def analyze_market_trends(properties: List[Property]) -> Dict[str, Any]:
    """
    Analyze market trends from property data
    
    Args:
        properties: List of properties to analyze
        
    Returns:
        Dictionary containing market analysis
    """
    if not properties:
        return {}
    
    # Price analysis
    prices = [p.price for p in properties]
    price_stats = {
        'mean': np.mean(prices),
        'median': np.median(prices),
        'std': np.std(prices),
        'min': np.min(prices),
        'max': np.max(prices),
        'q25': np.percentile(prices, 25),
        'q75': np.percentile(prices, 75)
    }
    
    # Location analysis
    location_counts = {}
    location_prices = {}
    for prop in properties:
        location = prop.location
        if location not in location_counts:
            location_counts[location] = 0
            location_prices[location] = []
        
        location_counts[location] += 1
        location_prices[location].append(prop.price)
    
    location_analysis = {}
    for location in location_counts:
        location_analysis[location] = {
            'count': location_counts[location],
            'avg_price': np.mean(location_prices[location]),
            'median_price': np.median(location_prices[location]),
            'price_range': np.max(location_prices[location]) - np.min(location_prices[location])
        }
    
    # Property type analysis
    type_counts = {}
    type_prices = {}
    for prop in properties:
        prop_type = prop.property_type
        if prop_type not in type_counts:
            type_counts[prop_type] = 0
            type_prices[prop_type] = []
        
        type_counts[prop_type] += 1
        type_prices[prop_type].append(prop.price)
    
    type_analysis = {}
    for prop_type in type_counts:
        type_analysis[prop_type] = {
            'count': type_counts[prop_type],
            'avg_price': np.mean(type_prices[prop_type]),
            'market_share': type_counts[prop_type] / len(properties)
        }
    
    # Size analysis
    sizes = [p.square_feet for p in properties if p.square_feet]
    if sizes:
        size_stats = {
            'mean': np.mean(sizes),
            'median': np.median(sizes),
            'min': np.min(sizes),
            'max': np.max(sizes)
        }
    else:
        size_stats = {}
    
    # Price per square foot analysis
    price_per_sqft = [p.get_price_per_sqft() for p in properties if p.get_price_per_sqft()]
    if price_per_sqft:
        price_per_sqft_stats = {
            'mean': np.mean(price_per_sqft),
            'median': np.median(price_per_sqft),
            'min': np.min(price_per_sqft),
            'max': np.max(price_per_sqft)
        }
    else:
        price_per_sqft_stats = {}
    
    return {
        'price_stats': price_stats,
        'location_analysis': location_analysis,
        'type_analysis': type_analysis,
        'size_stats': size_stats,
        'price_per_sqft_stats': price_per_sqft_stats,
        'total_properties': len(properties),
        'active_properties': len([p for p in properties if p.is_active])
    }


def generate_recommendation_explanation(property_obj: Property, 
                                      user: User, 
                                      score: float) -> str:
    """
    Generate human-readable explanation for why a property was recommended
    
    Args:
        property_obj: Recommended property
        user: User receiving recommendation
        score: Recommendation score
        
    Returns:
        Explanation string
    """
    explanations = []
    user_prefs = user.preferences
    
    # Price matching
    if user_prefs.min_price and user_prefs.max_price:
        if user_prefs.min_price <= property_obj.price <= user_prefs.max_price:
            explanations.append("matches your budget range")
    
    # Location preference
    if user_prefs.preferred_locations and property_obj.location in user_prefs.preferred_locations:
        explanations.append(f"located in your preferred area ({property_obj.location})")
    
    # Bedroom preference
    if user_prefs.min_bedrooms and user_prefs.max_bedrooms:
        if user_prefs.min_bedrooms <= property_obj.bedrooms <= user_prefs.max_bedrooms:
            explanations.append("has your preferred number of bedrooms")
    
    # Bathroom preference
    if user_prefs.min_bathrooms and property_obj.bathrooms >= user_prefs.min_bathrooms:
        explanations.append("meets your bathroom requirements")
    
    # Amenity matching
    if user_prefs.required_amenities:
        matching_amenities = set(property_obj.amenities) & set(user_prefs.required_amenities)
        if matching_amenities:
            explanations.append(f"includes {len(matching_amenities)} of your required amenities")
    
    # Property type preference
    if user_prefs.property_types and property_obj.property_type in user_prefs.property_types:
        explanations.append("matches your preferred property type")
    
    # Score-based explanations
    if score > 0.8:
        explanations.append("highly rated based on your preferences")
    elif score > 0.6:
        explanations.append("good match for your criteria")
    
    # Default explanation if no specific matches
    if not explanations:
        explanations.append("similar to properties you've shown interest in")
    
    return "This property " + ", ".join(explanations) + "."


def export_data_to_csv(data: List[Dict[str, Any]], filename: str) -> str:
    """
    Export data to CSV format and return as downloadable content
    
    Args:
        data: Data to export
        filename: Filename for download
        
    Returns:
        CSV content as string
    """
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def create_download_link(data: str, filename: str, link_text: str = "Download") -> str:
    """
    Create a download link for data
    
    Args:
        data: Data to download
        filename: Filename for download
        link_text: Text for the download link
        
    Returns:
        HTML download link
    """
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" target="_blank">{link_text}</a>'
    return href


def calculate_commute_score(property_location: str, user_work_location: str = None) -> float:
    """
    Calculate a commute convenience score (simplified for demo)
    
    Args:
        property_location: Property location
        user_work_location: User's work location
        
    Returns:
        Commute score between 0 and 1
    """
    if not user_work_location:
        return 0.5  # Neutral score if no work location
    
    # Simplified scoring based on location names
    downtown_locations = ["Downtown", "Financial District", "Central Plaza"]
    tech_locations = ["Tech Quarter", "University Area"]
    suburban_locations = ["Suburban Heights", "Green Valley", "Brookside"]
    
    property_category = None
    work_category = None
    
    # Categorize locations
    for category, locations in [
        ("downtown", downtown_locations),
        ("tech", tech_locations), 
        ("suburban", suburban_locations)
    ]:
        if property_location in locations:
            property_category = category
        if user_work_location in locations:
            work_category = category
    
    # Score based on category match
    if property_category == work_category:
        return 0.9  # Same area - excellent commute
    elif property_category == "downtown" or work_category == "downtown":
        return 0.7  # Downtown has good transit
    else:
        return 0.5  # Average commute


def analyze_user_behavior(user: User) -> Dict[str, Any]:
    """
    Analyze user behavior patterns from interaction history
    
    Args:
        user: User to analyze
        
    Returns:
        Dictionary containing behavior analysis
    """
    interactions = user.interactions
    
    if not interactions:
        return {
            'total_interactions': 0,
            'behavior_summary': 'New user with no activity yet'
        }
    
    # Interaction type distribution
    interaction_counts = {}
    for interaction in interactions:
        interaction_type = interaction.interaction_type
        interaction_counts[interaction_type] = interaction_counts.get(interaction_type, 0) + 1
    
    total_interactions = len(interactions)
    
    # Activity timeline
    recent_interactions = [i for i in interactions if 
                          (datetime.now() - i.timestamp).days <= 7]
    
    # Viewing behavior analysis
    view_interactions = [i for i in interactions if i.interaction_type == "view"]
    avg_view_duration = None
    if view_interactions:
        durations = [i.duration_seconds for i in view_interactions if i.duration_seconds]
        if durations:
            avg_view_duration = np.mean(durations)
    
    # Engagement level
    engagement_score = 0
    if interaction_counts.get('view', 0) > 10:
        engagement_score += 0.3
    if interaction_counts.get('like', 0) > 0:
        engagement_score += 0.3
    if interaction_counts.get('inquiry', 0) > 0:
        engagement_score += 0.4
    
    # User type classification
    user_type = "Browser"
    if interaction_counts.get('inquiry', 0) > 0:
        user_type = "Serious Buyer"
    elif interaction_counts.get('like', 0) > 3:
        user_type = "Active Searcher"
    elif interaction_counts.get('view', 0) > 20:
        user_type = "Casual Browser"
    
    return {
        'total_interactions': total_interactions,
        'interaction_breakdown': interaction_counts,
        'recent_activity': len(recent_interactions),
        'avg_view_duration': avg_view_duration,
        'engagement_score': engagement_score,
        'user_type': user_type,
        'days_active': (datetime.now() - user.created_at).days,
        'behavior_summary': f"{user_type} with {total_interactions} total interactions"
    }


def create_time_series_data(start_date: datetime, 
                           end_date: datetime, 
                           base_value: float = 100,
                           trend: float = 0.01,
                           seasonality: float = 0.1,
                           noise: float = 0.05) -> Tuple[List[datetime], List[float]]:
    """
    Generate time series data for visualization
    
    Args:
        start_date: Start date for time series
        end_date: End date for time series
        base_value: Base value for the series
        trend: Linear trend coefficient
        seasonality: Seasonal variation amplitude
        noise: Random noise amplitude
        
    Returns:
        Tuple of (dates, values)
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    values = []
    for i, date in enumerate(dates):
        # Linear trend
        trend_value = base_value + (trend * i)
        
        # Seasonal component (annual cycle)
        seasonal_value = seasonality * base_value * np.sin(2 * np.pi * i / 365)
        
        # Random noise
        noise_value = np.random.normal(0, noise * base_value)
        
        # Combine components
        final_value = trend_value + seasonal_value + noise_value
        values.append(max(0, final_value))  # Ensure non-negative
    
    return list(dates), values


def validate_property_data(property_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate property data for completeness and correctness
    
    Args:
        property_data: Property data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ['title', 'price', 'location', 'bedrooms', 'bathrooms', 'property_type']
    for field in required_fields:
        if field not in property_data or property_data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate data types and ranges
    if 'price' in property_data:
        try:
            price = float(property_data['price'])
            if price <= 0:
                errors.append("Price must be greater than 0")
            elif price > 50000:
                errors.append("Price seems unreasonably high (>$50,000)")
        except (ValueError, TypeError):
            errors.append("Price must be a valid number")
    
    if 'bedrooms' in property_data:
        try:
            bedrooms = int(property_data['bedrooms'])
            if bedrooms < 0 or bedrooms > 10:
                errors.append("Bedrooms must be between 0 and 10")
        except (ValueError, TypeError):
            errors.append("Bedrooms must be a valid integer")
    
    if 'bathrooms' in property_data:
        try:
            bathrooms = float(property_data['bathrooms'])
            if bathrooms <= 0 or bathrooms > 10:
                errors.append("Bathrooms must be between 0 and 10")
        except (ValueError, TypeError):
            errors.append("Bathrooms must be a valid number")
    
    if 'square_feet' in property_data and property_data['square_feet'] is not None:
        try:
            sqft = int(property_data['square_feet'])
            if sqft <= 0:
                errors.append("Square feet must be greater than 0")
            elif sqft > 20000:
                errors.append("Square feet seems unreasonably large (>20,000)")
        except (ValueError, TypeError):
            errors.append("Square feet must be a valid integer")
    
    # Validate property type
    valid_types = ['apartment', 'house', 'condo', 'studio', 'townhouse']
    if 'property_type' in property_data and property_data['property_type'] not in valid_types:
        errors.append(f"Property type must be one of: {', '.join(valid_types)}")
    
    return len(errors) == 0, errors


def calculate_roi_estimate(property_obj: Property, 
                          estimated_expenses_ratio: float = 0.3,
                          market_appreciation: float = 0.03) -> Dict[str, float]:
    """
    Calculate estimated ROI for investment properties
    
    Args:
        property_obj: Property to analyze
        estimated_expenses_ratio: Estimated expenses as ratio of rent
        market_appreciation: Annual market appreciation rate
        
    Returns:
        Dictionary containing ROI calculations
    """
    monthly_rent = property_obj.price
    annual_rent = monthly_rent * 12
    
    # Estimate property value (simplified calculation)
    # Assuming 1% monthly rule: property value = monthly_rent * 100
    estimated_property_value = monthly_rent * 100
    
    # Calculate expenses
    annual_expenses = annual_rent * estimated_expenses_ratio
    net_operating_income = annual_rent - annual_expenses
    
    # Cash flow yield
    cash_flow_yield = net_operating_income / estimated_property_value
    
    # Cap rate (Net Operating Income / Property Value)
    cap_rate = net_operating_income / estimated_property_value
    
    # Total return (cash flow + appreciation)
    total_return = cash_flow_yield + market_appreciation
    
    return {
        'monthly_rent': monthly_rent,
        'annual_rent': annual_rent,
        'estimated_property_value': estimated_property_value,
        'annual_expenses': annual_expenses,
        'net_operating_income': net_operating_income,
        'cash_flow_yield': cash_flow_yield,
        'cap_rate': cap_rate,
        'total_return_estimate': total_return
    }