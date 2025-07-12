"""
User Management Endpoints for the Rental ML System.

This module provides comprehensive user management capabilities including:
- User registration and authentication
- User preferences and profile management
- User recommendation history
- User search history and analytics
- User feedback and ratings
- User data export and privacy controls
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body, status, Cookie
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr, validator
from passlib.context import CryptContext
from jose import jwt, JWTError

from ...domain.entities.user import User, UserPreferences, UserInteraction
from ...domain.repositories.user_repository import UserRepository
from ...infrastructure.data.repositories.postgres_user_repository import PostgresUserRepository

logger = logging.getLogger(__name__)

user_router = APIRouter()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(str, Enum):
    """User role options"""
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    AGENT = "agent"


class InteractionType(str, Enum):
    """User interaction types"""
    VIEW = "view"
    LIKE = "like"
    SAVE = "save"
    INQUIRY = "inquiry"
    SHARE = "share"
    REPORT = "report"


class PrivacyLevel(str, Enum):
    """Privacy level options"""
    PUBLIC = "public"
    FRIENDS = "friends"
    PRIVATE = "private"


class NotificationPreference(str, Enum):
    """Notification preference options"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    NONE = "none"


class UserRegistrationRequest(BaseModel):
    """Request model for user registration"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    phone: Optional[str] = Field(None, description="Phone number")
    date_of_birth: Optional[datetime] = Field(None, description="Date of birth")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Initial user preferences")
    
    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLoginRequest(BaseModel):
    """Request model for user login"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(default=False, description="Remember login for extended period")


class UserUpdateRequest(BaseModel):
    """Request model for updating user profile"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = Field(None, max_length=500)
    privacy_level: Optional[PrivacyLevel] = None
    notification_preferences: Optional[List[NotificationPreference]] = None


class UserPreferencesUpdateRequest(BaseModel):
    """Request model for updating user preferences"""
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price filter")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price filter")
    min_bedrooms: Optional[int] = Field(None, ge=0, le=20, description="Minimum bedrooms")
    max_bedrooms: Optional[int] = Field(None, ge=0, le=20, description="Maximum bedrooms")
    min_bathrooms: Optional[float] = Field(None, ge=0, le=20, description="Minimum bathrooms")
    max_bathrooms: Optional[float] = Field(None, ge=0, le=20, description="Maximum bathrooms")
    preferred_locations: Optional[List[str]] = Field(None, description="Preferred locations")
    required_amenities: Optional[List[str]] = Field(None, description="Required amenities")
    property_types: Optional[List[str]] = Field(None, description="Preferred property types")
    
    @validator("max_price")
    def validate_max_price(cls, v, values):
        if v is not None and "min_price" in values and values["min_price"] is not None:
            if v < values["min_price"]:
                raise ValueError("Maximum price must be greater than minimum price")
        return v


class UserInteractionRequest(BaseModel):
    """Request model for user interactions"""
    property_id: UUID = Field(..., description="Property ID")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Duration of interaction")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional interaction metadata")


class UserFeedbackRequest(BaseModel):
    """Request model for user feedback"""
    property_id: UUID = Field(..., description="Property ID")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    review: Optional[str] = Field(None, max_length=1000, description="Review text")
    tags: Optional[List[str]] = Field(None, description="Feedback tags")
    is_anonymous: bool = Field(default=False, description="Submit as anonymous feedback")


class UserSearchRequest(BaseModel):
    """Request model for user search history"""
    query: str = Field(..., min_length=1, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Applied filters")
    result_count: Optional[int] = Field(None, ge=0, description="Number of results returned")
    clicked_results: Optional[List[UUID]] = Field(None, description="Property IDs clicked")


class PasswordChangeRequest(BaseModel):
    """Request model for password change"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator("new_password")
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(BaseModel):
    """Response model for user data"""
    id: UUID
    email: str
    first_name: str
    last_name: str
    phone: Optional[str]
    date_of_birth: Optional[datetime]
    avatar_url: Optional[str]
    bio: Optional[str]
    role: str
    privacy_level: str
    notification_preferences: List[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    total_interactions: Optional[int] = None
    total_saved_properties: Optional[int] = None


class UserPreferencesResponse(BaseModel):
    """Response model for user preferences"""
    min_price: Optional[float]
    max_price: Optional[float]
    min_bedrooms: Optional[int]
    max_bedrooms: Optional[int]
    min_bathrooms: Optional[float]
    max_bathrooms: Optional[float]
    preferred_locations: List[str]
    required_amenities: List[str]
    property_types: List[str]
    updated_at: datetime


class UserInteractionResponse(BaseModel):
    """Response model for user interactions"""
    id: UUID
    property_id: UUID
    interaction_type: str
    timestamp: datetime
    duration_seconds: Optional[int]
    metadata: Optional[Dict[str, Any]]


class UserHistoryResponse(BaseModel):
    """Response model for user history"""
    interactions: List[UserInteractionResponse]
    search_history: List[Dict[str, Any]]
    saved_properties: List[UUID]
    recommendations_received: List[Dict[str, Any]]
    total_count: int
    time_period: Dict[str, datetime]


class AuthResponse(BaseModel):
    """Response model for authentication"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class UserAnalyticsResponse(BaseModel):
    """Response model for user analytics"""
    user_id: UUID
    activity_summary: Dict[str, Any]
    preferences_insights: Dict[str, Any]
    recommendation_performance: Dict[str, Any]
    search_patterns: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    time_period: Dict[str, datetime]


# Dependency injection
async def get_user_repository() -> UserRepository:
    """Get user repository instance"""
    return PostgresUserRepository()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    repository: UserRepository = Depends(get_user_repository)
) -> User:
    """Get current authenticated user"""
    try:
        # Decode JWT token (simplified - in production use proper JWT verification)
        token = credentials.credentials
        # Mock token validation - implement proper JWT verification
        user_id = UUID("123e4567-e89b-12d3-a456-426614174000")  # Mock user ID
        
        user = await repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
        
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# Helper functions
def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: UUID, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    
    to_encode = {"sub": str(user_id), "exp": expire}
    encoded_jwt = jwt.encode(to_encode, "secret", algorithm="HS256")
    return encoded_jwt


@user_router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: UserRegistrationRequest,
    background_tasks: BackgroundTasks,
    repository: UserRepository = Depends(get_user_repository)
):
    """
    Register a new user account.
    
    Creates a new user with email verification and initial preferences.
    """
    try:
        logger.info(f"Registering new user: {request.email}")
        
        # Check if user already exists
        existing_user = await repository.get_by_email(request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        # Hash password
        hashed_password = hash_password(request.password)
        
        # Create user preferences
        preferences = UserPreferences()
        if request.preferences:
            for key, value in request.preferences.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)
        
        # Create user entity
        user = User.create(
            email=request.email,
            preferences=preferences
        )
        
        # Add additional fields (in a real implementation, extend User entity)
        # user.first_name = request.first_name
        # user.last_name = request.last_name
        # user.phone = request.phone
        # user.date_of_birth = request.date_of_birth
        # user.password_hash = hashed_password
        
        # Save user
        created_user = await repository.create(user)
        
        # Create tokens
        access_token = create_access_token(created_user.id)
        refresh_token = create_access_token(
            created_user.id,
            expires_delta=timedelta(days=7)
        )
        
        # Schedule background tasks
        background_tasks.add_task(
            send_verification_email,
            email=request.email,
            user_id=created_user.id
        )
        background_tasks.add_task(
            create_user_onboarding,
            user_id=created_user.id
        )
        
        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=1800,  # 30 minutes
            user=UserResponse(
                id=created_user.id,
                email=created_user.email,
                first_name=request.first_name,
                last_name=request.last_name,
                phone=request.phone,
                date_of_birth=request.date_of_birth,
                avatar_url=None,
                bio=None,
                role=UserRole.USER.value,
                privacy_level=PrivacyLevel.PRIVATE.value,
                notification_preferences=[NotificationPreference.EMAIL.value],
                is_active=created_user.is_active,
                is_verified=False,
                created_at=created_user.created_at,
                last_login=None
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@user_router.post("/login", response_model=AuthResponse)
async def login_user(
    request: UserLoginRequest,
    background_tasks: BackgroundTasks,
    repository: UserRepository = Depends(get_user_repository)
):
    """
    Authenticate user and return access tokens.
    """
    try:
        logger.info(f"User login attempt: {request.email}")
        
        # Get user by email
        user = await repository.get_by_email(request.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password (mock verification for now)
        # In real implementation: verify_password(request.password, user.password_hash)
        password_valid = True  # Mock
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Create tokens
        expires_delta = timedelta(days=7) if request.remember_me else timedelta(minutes=30)
        access_token = create_access_token(user.id, expires_delta)
        refresh_token = create_access_token(
            user.id,
            expires_delta=timedelta(days=30)
        )
        
        # Update last login
        background_tasks.add_task(
            update_last_login,
            user_id=user.id
        )
        
        # Log security event
        background_tasks.add_task(
            log_security_event,
            user_id=user.id,
            event_type="login",
            ip_address="127.0.0.1"  # Get from request
        )
        
        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(expires_delta.total_seconds()),
            user=UserResponse(
                id=user.id,
                email=user.email,
                first_name="John",  # Mock data
                last_name="Doe",    # Mock data
                phone=None,
                date_of_birth=None,
                avatar_url=None,
                bio=None,
                role=UserRole.USER.value,
                privacy_level=PrivacyLevel.PRIVATE.value,
                notification_preferences=[NotificationPreference.EMAIL.value],
                is_active=user.is_active,
                is_verified=True,
                created_at=user.created_at,
                last_login=datetime.now()
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@user_router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks
):
    """Logout user and invalidate tokens."""
    try:
        logger.info(f"User logout: {current_user.id}")
        
        # In production: invalidate tokens in Redis/database
        background_tasks.add_task(
            invalidate_user_tokens,
            user_id=current_user.id
        )
        
        # Log security event
        background_tasks.add_task(
            log_security_event,
            user_id=current_user.id,
            event_type="logout",
            ip_address="127.0.0.1"
        )
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )


@user_router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    include_stats: bool = Query(default=False, description="Include user statistics"),
    current_user: User = Depends(get_current_user),
    repository: UserRepository = Depends(get_user_repository)
):
    """Get current user's profile information."""
    try:
        response = UserResponse(
            id=current_user.id,
            email=current_user.email,
            first_name="John",  # Mock data
            last_name="Doe",    # Mock data
            phone=None,
            date_of_birth=None,
            avatar_url=None,
            bio=None,
            role=UserRole.USER.value,
            privacy_level=PrivacyLevel.PRIVATE.value,
            notification_preferences=[NotificationPreference.EMAIL.value],
            is_active=current_user.is_active,
            is_verified=True,
            created_at=current_user.created_at,
            last_login=datetime.now()
        )
        
        if include_stats:
            stats = await repository.get_user_statistics(current_user.id)
            response.total_interactions = stats.get("total_interactions", 0)
            response.total_saved_properties = stats.get("total_saved_properties", 0)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile: {str(e)}"
        )


@user_router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    request: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    repository: UserRepository = Depends(get_user_repository)
):
    """Update user profile information."""
    try:
        logger.info(f"Updating profile for user {current_user.id}")
        
        # Update user fields
        update_data = request.dict(exclude_unset=True)
        
        # In a real implementation, update the user entity with new data
        # For now, just return mock updated response
        
        # Save updated user
        updated_user = await repository.update(current_user)
        
        return UserResponse(
            id=updated_user.id,
            email=updated_user.email,
            first_name=request.first_name or "John",
            last_name=request.last_name or "Doe",
            phone=request.phone,
            date_of_birth=request.date_of_birth,
            avatar_url=request.avatar_url,
            bio=request.bio,
            role=UserRole.USER.value,
            privacy_level=request.privacy_level.value if request.privacy_level else PrivacyLevel.PRIVATE.value,
            notification_preferences=[np.value for np in request.notification_preferences] if request.notification_preferences else [NotificationPreference.EMAIL.value],
            is_active=updated_user.is_active,
            is_verified=True,
            created_at=updated_user.created_at,
            last_login=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}"
        )


@user_router.get("/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_user)
):
    """Get user's search and recommendation preferences."""
    try:
        preferences = current_user.preferences
        
        return UserPreferencesResponse(
            min_price=preferences.min_price,
            max_price=preferences.max_price,
            min_bedrooms=preferences.min_bedrooms,
            max_bedrooms=preferences.max_bedrooms,
            min_bathrooms=preferences.min_bathrooms,
            max_bathrooms=preferences.max_bathrooms,
            preferred_locations=preferences.preferred_locations,
            required_amenities=preferences.required_amenities,
            property_types=preferences.property_types,
            updated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get preferences: {str(e)}"
        )


@user_router.put("/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    request: UserPreferencesUpdateRequest,
    current_user: User = Depends(get_current_user),
    repository: UserRepository = Depends(get_user_repository)
):
    """Update user's search and recommendation preferences."""
    try:
        logger.info(f"Updating preferences for user {current_user.id}")
        
        # Update preferences
        update_data = request.dict(exclude_unset=True)
        
        new_preferences = UserPreferences()
        for key, value in update_data.items():
            if hasattr(new_preferences, key):
                setattr(new_preferences, key, value)
        
        # Merge with existing preferences
        if current_user.preferences:
            for attr in ["min_price", "max_price", "min_bedrooms", "max_bedrooms", 
                        "min_bathrooms", "max_bathrooms", "preferred_locations", 
                        "required_amenities", "property_types"]:
                if hasattr(new_preferences, attr) and getattr(new_preferences, attr) is None:
                    setattr(new_preferences, attr, getattr(current_user.preferences, attr))
        
        current_user.update_preferences(new_preferences)
        await repository.update(current_user)
        
        return UserPreferencesResponse(
            min_price=new_preferences.min_price,
            max_price=new_preferences.max_price,
            min_bedrooms=new_preferences.min_bedrooms,
            max_bedrooms=new_preferences.max_bedrooms,
            min_bathrooms=new_preferences.min_bathrooms,
            max_bathrooms=new_preferences.max_bathrooms,
            preferred_locations=new_preferences.preferred_locations,
            required_amenities=new_preferences.required_amenities,
            property_types=new_preferences.property_types,
            updated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to update user preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preferences: {str(e)}"
        )


@user_router.post("/interactions", status_code=status.HTTP_201_CREATED)
async def record_user_interaction(
    request: UserInteractionRequest,
    current_user: User = Depends(get_current_user),
    repository: UserRepository = Depends(get_user_repository)
):
    """Record a user interaction with a property."""
    try:
        logger.info(f"Recording {request.interaction_type} interaction for user {current_user.id}")
        
        # Create interaction
        interaction = UserInteraction.create(
            property_id=request.property_id,
            interaction_type=request.interaction_type.value,
            duration_seconds=request.duration_seconds
        )
        
        # Add to user
        current_user.add_interaction(interaction)
        await repository.update(current_user)
        
        return {
            "message": "Interaction recorded successfully",
            "interaction_id": str(interaction.property_id),  # Mock ID
            "interaction_type": interaction.interaction_type,
            "timestamp": interaction.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to record interaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record interaction: {str(e)}"
        )


@user_router.get("/history", response_model=UserHistoryResponse)
async def get_user_history(
    interaction_type: Optional[InteractionType] = Query(None, description="Filter by interaction type"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum results"),
    current_user: User = Depends(get_current_user)
):
    """Get user's interaction history and activity."""
    try:
        # Get interaction history
        interactions = current_user.get_interaction_history(
            interaction_type.value if interaction_type else None
        )
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_interactions = []
            for interaction in interactions:
                if start_date and interaction.timestamp < start_date:
                    continue
                if end_date and interaction.timestamp > end_date:
                    continue
                filtered_interactions.append(interaction)
            interactions = filtered_interactions
        
        # Limit results
        interactions = interactions[:limit]
        
        # Convert to response format
        interaction_responses = [
            UserInteractionResponse(
                id=uuid4(),  # Mock ID
                property_id=interaction.property_id,
                interaction_type=interaction.interaction_type,
                timestamp=interaction.timestamp,
                duration_seconds=interaction.duration_seconds,
                metadata={}
            )
            for interaction in interactions
        ]
        
        return UserHistoryResponse(
            interactions=interaction_responses,
            search_history=[],  # Mock data
            saved_properties=current_user.get_liked_properties(),
            recommendations_received=[],  # Mock data
            total_count=len(interaction_responses),
            time_period={
                "start_date": start_date or (datetime.now() - timedelta(days=30)),
                "end_date": end_date or datetime.now()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get user history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}"
        )


@user_router.post("/feedback")
async def submit_user_feedback(
    request: UserFeedbackRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks
):
    """Submit user feedback and rating for a property."""
    try:
        logger.info(f"User {current_user.id} submitting feedback for property {request.property_id}")
        
        # Process feedback
        feedback_data = {
            "user_id": current_user.id,
            "property_id": request.property_id,
            "rating": request.rating,
            "review": request.review,
            "tags": request.tags or [],
            "is_anonymous": request.is_anonymous,
            "timestamp": datetime.now()
        }
        
        # Schedule background processing
        background_tasks.add_task(
            process_user_feedback,
            feedback_data=feedback_data
        )
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": str(uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@user_router.post("/search-history")
async def record_search_history(
    request: UserSearchRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks
):
    """Record user search activity for analytics."""
    try:
        logger.info(f"Recording search for user {current_user.id}: {request.query}")
        
        # Schedule background processing
        background_tasks.add_task(
            process_search_history,
            user_id=current_user.id,
            search_data=request.dict()
        )
        
        return {
            "message": "Search recorded successfully",
            "search_id": str(uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to record search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record search: {str(e)}"
        )


@user_router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks
):
    """Change user password."""
    try:
        logger.info(f"Password change request for user {current_user.id}")
        
        # Verify current password (mock verification)
        current_password_valid = True  # Mock
        
        if not current_password_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_password_hash = hash_password(request.new_password)
        
        # Update password (in real implementation)
        # current_user.password_hash = new_password_hash
        
        # Schedule security notifications
        background_tasks.add_task(
            send_password_change_notification,
            user_id=current_user.id,
            email=current_user.email
        )
        
        background_tasks.add_task(
            log_security_event,
            user_id=current_user.id,
            event_type="password_change",
            ip_address="127.0.0.1"
        )
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to change password: {str(e)}"
        )


@user_router.get("/analytics", response_model=UserAnalyticsResponse)
async def get_user_analytics(
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    current_user: User = Depends(get_current_user),
    repository: UserRepository = Depends(get_user_repository)
):
    """Get comprehensive user analytics and insights."""
    try:
        analytics = await repository.get_user_analytics(
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date
        )
        
        return UserAnalyticsResponse(
            user_id=current_user.id,
            activity_summary=analytics["activity_summary"],
            preferences_insights=analytics["preferences_insights"],
            recommendation_performance=analytics["recommendation_performance"],
            search_patterns=analytics["search_patterns"],
            engagement_metrics=analytics["engagement_metrics"],
            time_period={
                "start_date": start_date or (datetime.now() - timedelta(days=30)),
                "end_date": end_date or datetime.now()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get user analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@user_router.get("/export-data")
async def export_user_data(
    format: str = Query(default="json", description="Export format (json, csv)"),
    current_user: User = Depends(get_current_user),
    repository: UserRepository = Depends(get_user_repository)
):
    """Export user data for privacy compliance (GDPR)."""
    try:
        logger.info(f"Exporting data for user {current_user.id} in {format} format")
        
        # Get all user data
        user_data = await repository.export_user_data(current_user.id)
        
        if format.lower() == "json":
            return JSONResponse(
                content=user_data,
                headers={"Content-Disposition": f"attachment; filename=user_data_{current_user.id}.json"}
            )
        else:
            # CSV format (simplified)
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers and data
            for section, data in user_data.items():
                writer.writerow([section])
                if isinstance(data, dict):
                    for key, value in data.items():
                        writer.writerow([key, str(value)])
                writer.writerow([])  # Empty row
            
            output.seek(0)
            return FileResponse(
                path=output,
                media_type="text/csv",
                filename=f"user_data_{current_user.id}.csv"
            )
        
    except Exception as e:
        logger.error(f"Failed to export user data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export data: {str(e)}"
        )


@user_router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_account(
    password: str = Body(..., description="Current password for confirmation"),
    current_user: User = Depends(get_current_user),
    repository: UserRepository = Depends(get_user_repository),
    background_tasks: BackgroundTasks
):
    """Delete user account (GDPR compliance)."""
    try:
        logger.info(f"Account deletion request for user {current_user.id}")
        
        # Verify password (mock verification)
        password_valid = True  # Mock
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password verification failed"
            )
        
        # Schedule account deletion
        background_tasks.add_task(
            process_account_deletion,
            user_id=current_user.id,
            email=current_user.email
        )
        
        return {"message": "Account deletion initiated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete account: {str(e)}"
        )


# Background tasks
async def send_verification_email(email: str, user_id: UUID):
    """Send email verification"""
    logger.info(f"Sending verification email to {email}")


async def create_user_onboarding(user_id: UUID):
    """Create user onboarding flow"""
    logger.info(f"Creating onboarding for user {user_id}")


async def update_last_login(user_id: UUID):
    """Update user's last login timestamp"""
    logger.info(f"Updating last login for user {user_id}")


async def invalidate_user_tokens(user_id: UUID):
    """Invalidate user's authentication tokens"""
    logger.info(f"Invalidating tokens for user {user_id}")


async def log_security_event(user_id: UUID, event_type: str, ip_address: str):
    """Log security-related events"""
    logger.info(f"Security event: {event_type} for user {user_id} from {ip_address}")


async def process_user_feedback(feedback_data: Dict[str, Any]):
    """Process user feedback"""
    logger.info(f"Processing feedback: {feedback_data}")


async def process_search_history(user_id: UUID, search_data: Dict[str, Any]):
    """Process search history"""
    logger.info(f"Processing search for user {user_id}: {search_data}")


async def send_password_change_notification(user_id: UUID, email: str):
    """Send password change notification"""
    logger.info(f"Sending password change notification to {email}")


async def process_account_deletion(user_id: UUID, email: str):
    """Process account deletion"""
    logger.info(f"Processing account deletion for user {user_id}")