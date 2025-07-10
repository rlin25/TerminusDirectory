from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator, EmailStr


class UserPreferencesRequest(BaseModel):
    """Request model for user preferences"""
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price preference")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price preference")
    min_bedrooms: Optional[int] = Field(None, ge=0, description="Minimum bedrooms preference")
    max_bedrooms: Optional[int] = Field(None, ge=0, description="Maximum bedrooms preference")
    min_bathrooms: Optional[float] = Field(None, ge=0, description="Minimum bathrooms preference")
    max_bathrooms: Optional[float] = Field(None, ge=0, description="Maximum bathrooms preference")
    preferred_locations: List[str] = Field(default_factory=list, description="List of preferred locations")
    required_amenities: List[str] = Field(default_factory=list, description="List of required amenities")
    property_types: List[str] = Field(default_factory=list, description="List of preferred property types")

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


class UserCreateRequest(BaseModel):
    """Request model for creating a new user"""
    email: EmailStr = Field(..., description="User email address")
    preferences: Optional[UserPreferencesRequest] = Field(None, description="User preferences")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional user metadata")

    class Config:
        json_encoders = {
            UUID: str
        }


class UserUpdateRequest(BaseModel):
    """Request model for updating user information"""
    email: Optional[EmailStr] = Field(None, description="Updated email address")
    preferences: Optional[UserPreferencesRequest] = Field(None, description="Updated user preferences")
    is_active: Optional[bool] = Field(None, description="User active status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional user metadata")

    class Config:
        json_encoders = {
            UUID: str
        }


class UserPreferencesResponse(BaseModel):
    """Response model for user preferences"""
    user_id: UUID
    preferences: UserPreferencesRequest
    derived_insights: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserInteractionSummary(BaseModel):
    """Summary model for user interactions"""
    interaction_type: str
    count: int
    last_interaction: datetime
    most_recent_property_id: Optional[UUID]

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserInteractionHistoryRequest(BaseModel):
    """Request model for user interaction history"""
    user_id: UUID = Field(..., description="User ID")
    interaction_type: Optional[str] = Field(None, description="Filter by interaction type")
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    limit: int = Field(default=50, ge=1, le=200, description="Number of interactions to return")
    offset: int = Field(default=0, ge=0, description="Number of interactions to skip")

    @validator('interaction_type')
    def validate_interaction_type(cls, v):
        if v is not None:
            allowed_types = ["view", "like", "dislike", "inquiry", "save", "unsave", "contact", "share"]
            if v not in allowed_types:
                raise ValueError(f'interaction_type must be one of: {", ".join(allowed_types)}')
        return v

    @validator('end_date')
    def validate_end_date(cls, v, values):
        if v is not None and 'start_date' in values and values['start_date'] is not None:
            if v < values['start_date']:
                raise ValueError('end_date must be after start_date')
        return v

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserInteractionDetail(BaseModel):
    """Detailed model for user interactions"""
    property_id: UUID
    interaction_type: str
    timestamp: datetime
    duration_seconds: Optional[int]
    metadata: Optional[Dict[str, Any]]
    
    # Property details for context
    property_title: Optional[str]
    property_price: Optional[float]
    property_location: Optional[str]

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserResponse(BaseModel):
    """Response model for user information"""
    id: UUID
    email: str
    preferences: UserPreferencesRequest
    created_at: datetime
    is_active: bool
    total_interactions: int
    last_activity: Optional[datetime]

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserInteractionHistoryResponse(BaseModel):
    """Response model for user interaction history"""
    user_id: UUID
    interactions: List[UserInteractionDetail]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    interaction_type: Optional[str]
    date_range: Optional[Dict[str, datetime]]

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserListRequest(BaseModel):
    """Request model for listing users"""
    limit: int = Field(default=20, ge=1, le=100, description="Number of users to return")
    offset: int = Field(default=0, ge=0, description="Number of users to skip")
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    email_filter: Optional[str] = Field(None, description="Filter by email pattern")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")

    @validator('created_before')
    def validate_created_before(cls, v, values):
        if v is not None and 'created_after' in values and values['created_after'] is not None:
            if v < values['created_after']:
                raise ValueError('created_before must be after created_after')
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserListResponse(BaseModel):
    """Response model for user listing"""
    users: List[UserResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    filters: Optional[Dict[str, Any]]

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserStatsResponse(BaseModel):
    """Response model for user statistics"""
    user_id: UUID
    total_searches: int
    total_views: int
    total_likes: int
    total_saves: int
    total_inquiries: int
    avg_session_duration: float
    favorite_locations: List[str]
    favorite_price_range: Dict[str, Optional[float]]
    most_viewed_property_types: List[str]
    activity_score: float
    last_active: datetime
    join_date: datetime

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserErrorResponse(BaseModel):
    """Error response model for user operations"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }