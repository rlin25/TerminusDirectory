from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator


class RecommendationRequest(BaseModel):
    """Request model for personalized recommendations"""
    user_id: UUID = Field(..., description="User ID for personalized recommendations")
    limit: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    exclude_viewed: bool = Field(default=True, description="Exclude already viewed properties")
    include_explanations: bool = Field(default=False, description="Include explanation for each recommendation")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters for recommendations")

    class Config:
        json_encoders = {
            UUID: str
        }


class SimilarPropertiesRequest(BaseModel):
    """Request model for similar properties"""
    property_id: UUID = Field(..., description="Property ID to find similar properties for")
    limit: int = Field(default=10, ge=1, le=50, description="Number of similar properties to return")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_explanations: bool = Field(default=False, description="Include explanation for similarity")

    class Config:
        json_encoders = {
            UUID: str
        }


class UserInteractionRequest(BaseModel):
    """Request model for tracking user interactions"""
    user_id: UUID = Field(..., description="User ID")
    property_id: UUID = Field(..., description="Property ID")
    interaction_type: str = Field(..., description="Type of interaction")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Duration of interaction in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional interaction metadata")

    @validator('interaction_type')
    def validate_interaction_type(cls, v):
        allowed_types = ["view", "like", "dislike", "inquiry", "save", "unsave", "contact", "share"]
        if v not in allowed_types:
            raise ValueError(f'interaction_type must be one of: {", ".join(allowed_types)}')
        return v

    class Config:
        json_encoders = {
            UUID: str
        }


class RecommendationExplanationRequest(BaseModel):
    """Request model for recommendation explanations"""
    user_id: UUID = Field(..., description="User ID")
    property_id: UUID = Field(..., description="Property ID")
    recommendation_type: str = Field(default="personalized", description="Type of recommendation")

    @validator('recommendation_type')
    def validate_recommendation_type(cls, v):
        allowed_types = ["personalized", "similar", "popular", "trending"]
        if v not in allowed_types:
            raise ValueError(f'recommendation_type must be one of: {", ".join(allowed_types)}')
        return v

    class Config:
        json_encoders = {
            UUID: str
        }


class RecommendationScore(BaseModel):
    """Model for recommendation scoring details"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall recommendation score")
    content_score: float = Field(..., ge=0.0, le=1.0, description="Content-based similarity score")
    collaborative_score: float = Field(..., ge=0.0, le=1.0, description="Collaborative filtering score")
    popularity_score: float = Field(..., ge=0.0, le=1.0, description="Popularity score")
    recency_score: float = Field(..., ge=0.0, le=1.0, description="Recency score")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Diversity score")


class RecommendationExplanation(BaseModel):
    """Model for recommendation explanation"""
    reason: str = Field(..., description="Main reason for recommendation")
    factors: List[str] = Field(..., description="List of contributing factors")
    similar_properties: List[UUID] = Field(default_factory=list, description="Similar properties user liked")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="Matching user preferences")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")

    class Config:
        json_encoders = {
            UUID: str
        }


class RecommendedProperty(BaseModel):
    """Model for recommended property with scoring"""
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
    
    # Recommendation-specific fields
    recommendation_score: RecommendationScore
    explanation: Optional[RecommendationExplanation] = None
    rank: int = Field(..., ge=1, description="Rank in recommendation list")
    recommendation_type: str = Field(..., description="Type of recommendation")

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[RecommendedProperty]
    user_id: UUID
    total_count: int
    page: int
    page_size: int
    recommendation_type: str
    generated_at: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(default="1.0", description="ML model version used")
    response_time_ms: float

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class SimilarPropertiesResponse(BaseModel):
    """Response model for similar properties"""
    similar_properties: List[RecommendedProperty]
    source_property_id: UUID
    total_count: int
    similarity_threshold: float
    generated_at: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(default="1.0", description="ML model version used")
    response_time_ms: float

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class UserInteractionResponse(BaseModel):
    """Response model for user interaction tracking"""
    success: bool
    message: str
    interaction_id: Optional[UUID] = None
    user_id: UUID
    property_id: UUID
    interaction_type: str
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class RecommendationExplanationResponse(BaseModel):
    """Response model for recommendation explanations"""
    user_id: UUID
    property_id: UUID
    explanation: RecommendationExplanation
    recommendation_type: str
    generated_at: datetime = Field(default_factory=datetime.now)
    response_time_ms: float

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class RecommendationMetrics(BaseModel):
    """Model for recommendation system metrics"""
    user_id: UUID
    total_recommendations: int
    click_through_rate: float
    conversion_rate: float
    diversity_score: float
    novelty_score: float
    coverage_score: float
    last_updated: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class RecommendationErrorResponse(BaseModel):
    """Error response model for recommendation operations"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }