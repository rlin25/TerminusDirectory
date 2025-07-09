from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator


class SearchFiltersRequest(BaseModel):
    """Request model for search filters"""
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    min_bedrooms: Optional[int] = Field(None, ge=0, description="Minimum number of bedrooms")
    max_bedrooms: Optional[int] = Field(None, ge=0, description="Maximum number of bedrooms")
    min_bathrooms: Optional[float] = Field(None, ge=0, description="Minimum number of bathrooms")
    max_bathrooms: Optional[float] = Field(None, ge=0, description="Maximum number of bathrooms")
    locations: Optional[List[str]] = Field(default_factory=list, description="List of preferred locations")
    amenities: Optional[List[str]] = Field(default_factory=list, description="List of required amenities")
    property_types: Optional[List[str]] = Field(default_factory=list, description="List of property types")
    min_square_feet: Optional[int] = Field(None, ge=0, description="Minimum square feet")
    max_square_feet: Optional[int] = Field(None, ge=0, description="Maximum square feet")

    @validator('max_price')
    def validate_max_price(cls, v, values):
        if v is not None and 'min_price' in values and values['min_price'] is not None:
            if v < values['min_price']:
                raise ValueError('max_price must be greater than or equal to min_price')
        return v

    @validator('max_bedrooms')
    def validate_max_bedrooms(cls, v, values):
        if v is not None and 'min_bedrooms' in values and values['min_bedrooms'] is not None:
            if v < values['min_bedrooms']:
                raise ValueError('max_bedrooms must be greater than or equal to min_bedrooms')
        return v

    @validator('max_bathrooms')
    def validate_max_bathrooms(cls, v, values):
        if v is not None and 'min_bathrooms' in values and values['min_bathrooms'] is not None:
            if v < values['min_bathrooms']:
                raise ValueError('max_bathrooms must be greater than or equal to min_bathrooms')
        return v

    @validator('max_square_feet')
    def validate_max_square_feet(cls, v, values):
        if v is not None and 'min_square_feet' in values and values['min_square_feet'] is not None:
            if v < values['min_square_feet']:
                raise ValueError('max_square_feet must be greater than or equal to min_square_feet')
        return v


class SearchRequest(BaseModel):
    """Request model for property search"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    filters: Optional[SearchFiltersRequest] = Field(default_factory=SearchFiltersRequest, description="Search filters")
    limit: int = Field(default=20, ge=1, le=100, description="Number of results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    sort_by: str = Field(default="relevance", description="Sort order for results")
    user_id: Optional[UUID] = Field(None, description="User ID for personalized results")

    @validator('sort_by')
    def validate_sort_by(cls, v):
        allowed_values = ["relevance", "price_asc", "price_desc", "date_new", "date_old"]
        if v not in allowed_values:
            raise ValueError(f'sort_by must be one of: {", ".join(allowed_values)}')
        return v


class PropertyResponse(BaseModel):
    """Response model for property information"""
    id: UUID
    title: str
    description: str
    price: float
    location: str
    bedrooms: int
    bathrooms: float
    square_feet: Optional[int]
    amenities: List[str]
    contact_info: Dict[str, str]
    images: List[str]
    property_type: str
    scraped_at: datetime
    is_active: bool
    price_per_sqft: Optional[float] = None
    relevance_score: Optional[float] = None

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class SearchResponse(BaseModel):
    """Response model for search results"""
    properties: List[PropertyResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    query: str
    filters: Optional[SearchFiltersRequest]
    sort_by: str
    search_time_ms: float
    suggestions: Optional[List[str]] = None

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class SearchSuggestionRequest(BaseModel):
    """Request model for search suggestions"""
    query: str = Field(..., min_length=1, max_length=100, description="Partial search query")
    limit: int = Field(default=10, ge=1, le=20, description="Number of suggestions to return")
    user_id: Optional[UUID] = Field(None, description="User ID for personalized suggestions")


class SearchSuggestionResponse(BaseModel):
    """Response model for search suggestions"""
    suggestions: List[str]
    query: str
    response_time_ms: float

    class Config:
        json_encoders = {
            UUID: str
        }


class PopularSearchRequest(BaseModel):
    """Request model for popular search terms"""
    limit: int = Field(default=10, ge=1, le=50, description="Number of popular terms to return")
    time_range: str = Field(default="24h", description="Time range for popularity calculation")
    category: Optional[str] = Field(None, description="Category filter for popular terms")

    @validator('time_range')
    def validate_time_range(cls, v):
        allowed_values = ["1h", "24h", "7d", "30d", "all"]
        if v not in allowed_values:
            raise ValueError(f'time_range must be one of: {", ".join(allowed_values)}')
        return v


class PopularSearchResponse(BaseModel):
    """Response model for popular search terms"""
    terms: List[Dict[str, Any]]  # Each term has 'term', 'count', 'category', etc.
    time_range: str
    category: Optional[str]
    response_time_ms: float

    class Config:
        json_encoders = {
            UUID: str
        }


class SearchErrorResponse(BaseModel):
    """Error response model for search operations"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }