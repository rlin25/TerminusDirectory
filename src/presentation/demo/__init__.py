"""
Rental ML System - Streamlit Demo Package

This package contains the complete Streamlit demo application for showcasing
the Rental ML System's capabilities.

Modules:
    app: Main Streamlit application
    components: Reusable UI components
    sample_data: Sample data generation utilities
    utils: Utility functions for data processing and formatting
    config: Configuration settings and constants
"""

__version__ = "1.0.0"
__author__ = "Rental ML System Team"
__description__ = "Interactive demo application for the Rental ML System"

# Import main classes for easy access
from .sample_data import SampleDataGenerator
from .config import config, FeatureFlags, Constants
from .utils import (
    format_price, 
    format_number, 
    calculate_distance,
    generate_map_data,
    analyze_market_trends
)

# Package metadata
__all__ = [
    "SampleDataGenerator",
    "config", 
    "FeatureFlags",
    "Constants",
    "format_price",
    "format_number", 
    "calculate_distance",
    "generate_map_data",
    "analyze_market_trends"
]