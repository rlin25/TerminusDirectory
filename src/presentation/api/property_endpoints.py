"""
Property Management Endpoints for the Rental ML System.

This module provides comprehensive property management capabilities including:
- CRUD operations for properties
- Property search and filtering
- Property analytics and statistics
- Property data quality endpoints
- Bulk property operations
- Property image and media handling
"""

import asyncio
import logging
import time
import csv
import io
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body, File, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field, validator
import pandas as pd

from ...domain.entities.property import Property
from ...domain.repositories.property_repository import PropertyRepository
from ...infrastructure.data.repositories.postgres_property_repository import PostgresPropertyRepository

logger = logging.getLogger(__name__)

property_router = APIRouter()


class PropertyStatus(str, Enum):
    """Property status options"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ARCHIVED = "archived"


class SortOrder(str, Enum):
    """Sort order options"""
    ASC = "asc"
    DESC = "desc"


class PropertyType(str, Enum):
    """Property type options"""
    APARTMENT = "apartment"
    HOUSE = "house"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    STUDIO = "studio"
    LOFT = "loft"


class DataQualityLevel(str, Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


class PropertyCreateRequest(BaseModel):
    """Request model for creating a new property"""
    title: str = Field(..., min_length=1, max_length=500, description="Property title")
    description: str = Field(..., min_length=1, description="Property description")
    price: float = Field(..., gt=0, description="Monthly rent price")
    location: str = Field(..., min_length=1, description="Property location")
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=20, description="Number of bathrooms")
    square_feet: Optional[int] = Field(None, gt=0, description="Property square footage")
    property_type: PropertyType = Field(default=PropertyType.APARTMENT, description="Type of property")
    amenities: List[str] = Field(default=[], description="List of amenities")
    contact_info: Dict[str, str] = Field(default={}, description="Contact information")
    images: List[str] = Field(default=[], description="List of image URLs")
    is_active: bool = Field(default=True, description="Property active status")
    
    @validator("price")
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        if v > 50000:  # Reasonable upper limit
            raise ValueError("Price seems unreasonably high")
        return v
    
    @validator("contact_info")
    def validate_contact_info(cls, v):
        allowed_keys = {"email", "phone", "website", "agent_name"}
        invalid_keys = set(v.keys()) - allowed_keys
        if invalid_keys:
            raise ValueError(f"Invalid contact info keys: {invalid_keys}")
        return v


class PropertyUpdateRequest(BaseModel):
    """Request model for updating a property"""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, min_length=1)
    price: Optional[float] = Field(None, gt=0)
    location: Optional[str] = Field(None, min_length=1)
    bedrooms: Optional[int] = Field(None, ge=0, le=20)
    bathrooms: Optional[float] = Field(None, ge=0, le=20)
    square_feet: Optional[int] = Field(None, gt=0)
    property_type: Optional[PropertyType] = None
    amenities: Optional[List[str]] = None
    contact_info: Optional[Dict[str, str]] = None
    images: Optional[List[str]] = None
    is_active: Optional[bool] = None


class PropertySearchRequest(BaseModel):
    """Request model for property search"""
    query: Optional[str] = Field(None, description="Search query")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price filter")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price filter")
    min_bedrooms: Optional[int] = Field(None, ge=0, description="Minimum bedrooms")
    max_bedrooms: Optional[int] = Field(None, ge=0, description="Maximum bedrooms")
    min_bathrooms: Optional[float] = Field(None, ge=0, description="Minimum bathrooms")
    max_bathrooms: Optional[float] = Field(None, ge=0, description="Maximum bathrooms")
    locations: Optional[List[str]] = Field(None, description="Filter by locations")
    property_types: Optional[List[PropertyType]] = Field(None, description="Filter by property types")
    amenities: Optional[List[str]] = Field(None, description="Required amenities")
    status: Optional[PropertyStatus] = Field(None, description="Property status filter")
    sort_by: Optional[str] = Field(default="created_at", description="Sort field")
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="Sort order")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")


class BulkOperationRequest(BaseModel):
    """Request model for bulk operations"""
    property_ids: List[UUID] = Field(..., min_items=1, description="List of property IDs")
    operation: str = Field(..., description="Operation to perform")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Operation parameters")


class PropertyAnalyticsRequest(BaseModel):
    """Request model for property analytics"""
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")
    location_filter: Optional[List[str]] = Field(None, description="Filter by locations")
    property_type_filter: Optional[List[PropertyType]] = Field(None, description="Filter by property types")
    group_by: Optional[str] = Field(None, description="Group analytics by field")


class PropertyResponse(BaseModel):
    """Response model for property data"""
    id: UUID
    title: str
    description: str
    price: float
    location: str
    bedrooms: int
    bathrooms: float
    square_feet: Optional[int]
    property_type: str
    amenities: List[str]
    contact_info: Dict[str, str]
    images: List[str]
    is_active: bool
    scraped_at: datetime
    created_at: datetime
    updated_at: Optional[datetime] = None
    data_quality_score: Optional[float] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None


class PropertyListResponse(BaseModel):
    """Response model for property lists"""
    properties: List[PropertyResponse]
    total_count: int
    has_more: bool
    page_info: Dict[str, Any]
    filters_applied: Dict[str, Any]
    processing_time_ms: float


class PropertyAnalyticsResponse(BaseModel):
    """Response model for property analytics"""
    summary: Dict[str, Any]
    trends: Dict[str, Any]
    distributions: Dict[str, Any]
    insights: List[str]
    time_period: Dict[str, datetime]
    data_quality: Dict[str, Any]


class BulkOperationResponse(BaseModel):
    """Response model for bulk operations"""
    operation: str
    total_processed: int
    successful: int
    failed: int
    errors: List[Dict[str, Any]]
    processing_time_ms: float
    job_id: Optional[str] = None


# Dependency injection
async def get_property_repository() -> PropertyRepository:
    """Get property repository instance"""
    # This would be injected from the app state in a real implementation
    return PostgresPropertyRepository()


@property_router.post("/", response_model=PropertyResponse, status_code=status.HTTP_201_CREATED)
async def create_property(
    request: PropertyCreateRequest,
    background_tasks: BackgroundTasks,
    repository: PropertyRepository = Depends(get_property_repository)
):
    """
    Create a new property listing.
    
    Creates a new property with comprehensive data validation and quality checks.
    """
    try:
        logger.info(f"Creating new property: {request.title}")
        
        # Create property entity
        property_entity = Property.create(
            title=request.title,
            description=request.description,
            price=request.price,
            location=request.location,
            bedrooms=request.bedrooms,
            bathrooms=request.bathrooms,
            square_feet=request.square_feet,
            amenities=request.amenities,
            contact_info=request.contact_info,
            images=request.images,
            property_type=request.property_type.value
        )
        
        if not request.is_active:
            property_entity.deactivate()
        
        # Save to repository
        created_property = await repository.create(property_entity)
        
        # Schedule background tasks
        background_tasks.add_task(
            analyze_property_data_quality,
            property_id=created_property.id
        )
        background_tasks.add_task(
            update_location_statistics,
            location=created_property.location
        )
        
        return PropertyResponse(
            id=created_property.id,
            title=created_property.title,
            description=created_property.description,
            price=created_property.price,
            location=created_property.location,
            bedrooms=created_property.bedrooms,
            bathrooms=created_property.bathrooms,
            square_feet=created_property.square_feet,
            property_type=created_property.property_type,
            amenities=created_property.amenities,
            contact_info=created_property.contact_info,
            images=created_property.images,
            is_active=created_property.is_active,
            scraped_at=created_property.scraped_at,
            created_at=created_property.scraped_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create property: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create property: {str(e)}"
        )


@property_router.get("/{property_id}", response_model=PropertyResponse)
async def get_property(
    property_id: UUID,
    include_stats: bool = Query(default=False, description="Include view/like statistics"),
    repository: PropertyRepository = Depends(get_property_repository)
):
    """Get a specific property by ID with optional statistics."""
    try:
        property_entity = await repository.get_by_id(property_id)
        
        if not property_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Property {property_id} not found"
            )
        
        response = PropertyResponse(
            id=property_entity.id,
            title=property_entity.title,
            description=property_entity.description,
            price=property_entity.price,
            location=property_entity.location,
            bedrooms=property_entity.bedrooms,
            bathrooms=property_entity.bathrooms,
            square_feet=property_entity.square_feet,
            property_type=property_entity.property_type,
            amenities=property_entity.amenities,
            contact_info=property_entity.contact_info,
            images=property_entity.images,
            is_active=property_entity.is_active,
            scraped_at=property_entity.scraped_at,
            created_at=property_entity.scraped_at
        )
        
        if include_stats:
            stats = await repository.get_property_statistics(property_id)
            response.view_count = stats.get("view_count", 0)
            response.like_count = stats.get("like_count", 0)
            response.data_quality_score = stats.get("data_quality_score", 0.0)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get property {property_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve property: {str(e)}"
        )


@property_router.put("/{property_id}", response_model=PropertyResponse)
async def update_property(
    property_id: UUID,
    request: PropertyUpdateRequest,
    background_tasks: BackgroundTasks,
    repository: PropertyRepository = Depends(get_property_repository)
):
    """Update an existing property."""
    try:
        logger.info(f"Updating property {property_id}")
        
        # Get existing property
        existing_property = await repository.get_by_id(property_id)
        if not existing_property:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Property {property_id} not found"
            )
        
        # Update fields that are provided
        update_data = request.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(existing_property, field):
                setattr(existing_property, field, value)
        
        # Save updated property
        updated_property = await repository.update(existing_property)
        
        # Schedule background tasks
        background_tasks.add_task(
            analyze_property_data_quality,
            property_id=updated_property.id
        )
        
        return PropertyResponse(
            id=updated_property.id,
            title=updated_property.title,
            description=updated_property.description,
            price=updated_property.price,
            location=updated_property.location,
            bedrooms=updated_property.bedrooms,
            bathrooms=updated_property.bathrooms,
            square_feet=updated_property.square_feet,
            property_type=updated_property.property_type,
            amenities=updated_property.amenities,
            contact_info=updated_property.contact_info,
            images=updated_property.images,
            is_active=updated_property.is_active,
            scraped_at=updated_property.scraped_at,
            created_at=updated_property.scraped_at,
            updated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update property {property_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update property: {str(e)}"
        )


@property_router.delete("/{property_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_property(
    property_id: UUID,
    hard_delete: bool = Query(default=False, description="Permanently delete vs soft delete"),
    repository: PropertyRepository = Depends(get_property_repository)
):
    """Delete a property (soft delete by default)."""
    try:
        logger.info(f"Deleting property {property_id} (hard_delete={hard_delete})")
        
        if hard_delete:
            success = await repository.delete(property_id)
        else:
            # Soft delete - just deactivate
            property_entity = await repository.get_by_id(property_id)
            if property_entity:
                property_entity.deactivate()
                await repository.update(property_entity)
                success = True
            else:
                success = False
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Property {property_id} not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete property {property_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete property: {str(e)}"
        )


@property_router.post("/search", response_model=PropertyListResponse)
async def search_properties(
    request: PropertySearchRequest,
    repository: PropertyRepository = Depends(get_property_repository)
):
    """
    Advanced property search with filtering and sorting.
    
    Supports complex queries with multiple filters and sorting options.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Searching properties with filters: {request.dict(exclude_unset=True)}")
        
        # Build search filters
        filters = {}
        
        if request.min_price is not None:
            filters["min_price"] = request.min_price
        if request.max_price is not None:
            filters["max_price"] = request.max_price
        if request.min_bedrooms is not None:
            filters["min_bedrooms"] = request.min_bedrooms
        if request.max_bedrooms is not None:
            filters["max_bedrooms"] = request.max_bedrooms
        if request.min_bathrooms is not None:
            filters["min_bathrooms"] = request.min_bathrooms
        if request.max_bathrooms is not None:
            filters["max_bathrooms"] = request.max_bathrooms
        if request.locations:
            filters["locations"] = request.locations
        if request.property_types:
            filters["property_types"] = [pt.value for pt in request.property_types]
        if request.amenities:
            filters["amenities"] = request.amenities
        if request.status:
            filters["status"] = request.status.value
        
        # Perform search
        properties, total_count = await repository.search_properties(
            query=request.query,
            filters=filters,
            sort_by=request.sort_by,
            sort_order=request.sort_order.value,
            limit=request.limit,
            offset=request.offset
        )
        
        # Convert to response format
        property_responses = [
            PropertyResponse(
                id=prop.id,
                title=prop.title,
                description=prop.description,
                price=prop.price,
                location=prop.location,
                bedrooms=prop.bedrooms,
                bathrooms=prop.bathrooms,
                square_feet=prop.square_feet,
                property_type=prop.property_type,
                amenities=prop.amenities,
                contact_info=prop.contact_info,
                images=prop.images,
                is_active=prop.is_active,
                scraped_at=prop.scraped_at,
                created_at=prop.scraped_at
            )
            for prop in properties
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return PropertyListResponse(
            properties=property_responses,
            total_count=total_count,
            has_more=(request.offset + len(properties)) < total_count,
            page_info={
                "limit": request.limit,
                "offset": request.offset,
                "current_page": (request.offset // request.limit) + 1,
                "total_pages": (total_count + request.limit - 1) // request.limit
            },
            filters_applied=filters,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Property search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Property search failed: {str(e)}"
        )


@property_router.post("/bulk-operation", response_model=BulkOperationResponse)
async def bulk_operation(
    request: BulkOperationRequest,
    background_tasks: BackgroundTasks,
    repository: PropertyRepository = Depends(get_property_repository)
):
    """
    Perform bulk operations on multiple properties.
    
    Supported operations:
    - activate: Activate properties
    - deactivate: Deactivate properties
    - delete: Delete properties
    - update_price: Update price (requires 'price' parameter)
    - add_amenity: Add amenity (requires 'amenity' parameter)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Performing bulk operation '{request.operation}' on {len(request.property_ids)} properties")
        
        successful = 0
        failed = 0
        errors = []
        
        for property_id in request.property_ids:
            try:
                property_entity = await repository.get_by_id(property_id)
                if not property_entity:
                    errors.append({
                        "property_id": str(property_id),
                        "error": "Property not found"
                    })
                    failed += 1
                    continue
                
                # Perform operation
                if request.operation == "activate":
                    property_entity.activate()
                    await repository.update(property_entity)
                elif request.operation == "deactivate":
                    property_entity.deactivate()
                    await repository.update(property_entity)
                elif request.operation == "delete":
                    await repository.delete(property_id)
                elif request.operation == "update_price":
                    if not request.parameters or "price" not in request.parameters:
                        raise ValueError("Price parameter required for update_price operation")
                    property_entity.price = request.parameters["price"]
                    await repository.update(property_entity)
                elif request.operation == "add_amenity":
                    if not request.parameters or "amenity" not in request.parameters:
                        raise ValueError("Amenity parameter required for add_amenity operation")
                    amenity = request.parameters["amenity"]
                    if amenity not in property_entity.amenities:
                        property_entity.amenities.append(amenity)
                        await repository.update(property_entity)
                else:
                    raise ValueError(f"Unknown operation: {request.operation}")
                
                successful += 1
                
            except Exception as e:
                errors.append({
                    "property_id": str(property_id),
                    "error": str(e)
                })
                failed += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        return BulkOperationResponse(
            operation=request.operation,
            total_processed=len(request.property_ids),
            successful=successful,
            failed=failed,
            errors=errors,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Bulk operation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk operation failed: {str(e)}"
        )


@property_router.post("/analytics", response_model=PropertyAnalyticsResponse)
async def get_property_analytics(
    request: PropertyAnalyticsRequest,
    repository: PropertyRepository = Depends(get_property_repository)
):
    """
    Get comprehensive property analytics and insights.
    
    Provides market analysis, pricing trends, and property insights.
    """
    try:
        logger.info("Generating property analytics")
        
        # Build filters for analytics
        filters = {}
        if request.location_filter:
            filters["locations"] = request.location_filter
        if request.property_type_filter:
            filters["property_types"] = [pt.value for pt in request.property_type_filter]
        
        # Get analytics data
        analytics = await repository.get_analytics(
            start_date=request.start_date,
            end_date=request.end_date,
            filters=filters,
            group_by=request.group_by
        )
        
        return PropertyAnalyticsResponse(
            summary=analytics["summary"],
            trends=analytics["trends"],
            distributions=analytics["distributions"],
            insights=analytics["insights"],
            time_period={
                "start_date": request.start_date or datetime.now() - timedelta(days=30),
                "end_date": request.end_date or datetime.now()
            },
            data_quality=analytics["data_quality"]
        )
        
    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics generation failed: {str(e)}"
        )


@property_router.get("/export/csv")
async def export_properties_csv(
    query: Optional[str] = Query(None),
    location_filter: Optional[str] = Query(None),
    repository: PropertyRepository = Depends(get_property_repository)
):
    """Export properties to CSV format."""
    try:
        logger.info("Exporting properties to CSV")
        
        # Get properties based on filters
        filters = {}
        if location_filter:
            filters["locations"] = [location_filter]
        
        properties, _ = await repository.search_properties(
            query=query,
            filters=filters,
            limit=10000  # Large limit for export
        )
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "ID", "Title", "Price", "Location", "Bedrooms", "Bathrooms",
            "Square Feet", "Property Type", "Amenities", "Created At"
        ])
        
        # Write data
        for prop in properties:
            writer.writerow([
                str(prop.id),
                prop.title,
                prop.price,
                prop.location,
                prop.bedrooms,
                prop.bathrooms,
                prop.square_feet or "",
                prop.property_type,
                ", ".join(prop.amenities),
                prop.scraped_at.isoformat()
            ])
        
        # Return as streaming response
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=properties.csv"}
        )
        
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV export failed: {str(e)}"
        )


@property_router.post("/upload/images/{property_id}")
async def upload_property_images(
    property_id: UUID,
    files: List[UploadFile] = File(...),
    repository: PropertyRepository = Depends(get_property_repository)
):
    """Upload images for a property."""
    try:
        logger.info(f"Uploading {len(files)} images for property {property_id}")
        
        # Validate property exists
        property_entity = await repository.get_by_id(property_id)
        if not property_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Property {property_id} not found"
            )
        
        uploaded_urls = []
        
        for file in files:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not an image"
                )
            
            # Save file (in a real implementation, this would upload to cloud storage)
            file_content = await file.read()
            filename = f"property_{property_id}_{uuid4()}_{file.filename}"
            
            # Mock URL - in reality, this would be the actual cloud storage URL
            image_url = f"https://storage.example.com/properties/{filename}"
            uploaded_urls.append(image_url)
        
        # Update property with new image URLs
        property_entity.images.extend(uploaded_urls)
        await repository.update(property_entity)
        
        return {
            "property_id": property_id,
            "uploaded_images": uploaded_urls,
            "total_images": len(property_entity.images),
            "message": f"Successfully uploaded {len(uploaded_urls)} images"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image upload failed: {str(e)}"
        )


@property_router.get("/data-quality/report")
async def get_data_quality_report(
    quality_level: Optional[DataQualityLevel] = Query(None),
    repository: PropertyRepository = Depends(get_property_repository)
):
    """Get comprehensive data quality report for properties."""
    try:
        logger.info("Generating data quality report")
        
        report = await repository.get_data_quality_report(
            quality_level=quality_level.value if quality_level else None
        )
        
        return {
            "summary": report["summary"],
            "quality_scores": report["quality_scores"],
            "common_issues": report["common_issues"],
            "recommendations": report["recommendations"],
            "properties_by_quality": report["properties_by_quality"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data quality report failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data quality report failed: {str(e)}"
        )


# Background tasks
async def analyze_property_data_quality(property_id: UUID):
    """Analyze data quality for a property."""
    logger.info(f"Analyzing data quality for property {property_id}")
    # Implementation would include data quality scoring


async def update_location_statistics(location: str):
    """Update statistics for a location."""
    logger.info(f"Updating statistics for location: {location}")
    # Implementation would update location-based analytics