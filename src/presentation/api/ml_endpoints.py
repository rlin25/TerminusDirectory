"""
ML Inference Endpoints for the Rental ML System.

This module provides comprehensive machine learning inference endpoints including:
- Property recommendation endpoints (collaborative, content-based, hybrid)
- Real-time property search with NLP ranking
- Batch prediction endpoints for large-scale inference
- Model performance metrics endpoints
- Feature engineering endpoints for debugging
- Model A/B testing endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Body, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

from ...domain.entities.user import User, UserPreferences
from ...domain.entities.property import Property
from ...domain.services.recommendation_service import RecommendationService
from ...domain.services.search_service import SearchService
from ...infrastructure.ml.models.hybrid_recommender import HybridRecommender
from ...infrastructure.ml.models.collaborative_filter import CollaborativeFilter
from ...infrastructure.ml.models.content_recommender import ContentBasedRecommender
from ...infrastructure.ml.models.search_ranker import SearchRanker
from ...infrastructure.ml.serving.model_server import ModelServer
from ...infrastructure.ml.serving.feature_store import FeatureStore

logger = logging.getLogger(__name__)

ml_router = APIRouter()


class RecommendationType(str, Enum):
    """Types of recommendations available"""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"


class ModelType(str, Enum):
    """Types of ML models"""
    RECOMMENDER = "recommender"
    SEARCH_RANKER = "search_ranker"
    CONTENT_FILTER = "content_filter"
    COLLABORATIVE_FILTER = "collaborative_filter"


class BatchProcessingStatus(str, Enum):
    """Batch processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RecommendationRequest(BaseModel):
    """Request model for property recommendations"""
    user_id: UUID = Field(..., description="User ID for personalized recommendations")
    recommendation_type: RecommendationType = Field(
        default=RecommendationType.HYBRID,
        description="Type of recommendation algorithm to use"
    )
    num_recommendations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return"
    )
    exclude_viewed: bool = Field(
        default=True,
        description="Exclude properties the user has already viewed"
    )
    exclude_liked: bool = Field(
        default=False,
        description="Exclude properties the user has already liked"
    )
    price_filter: Optional[Dict[str, float]] = Field(
        None,
        description="Price filtering: {'min': min_price, 'max': max_price}"
    )
    location_filter: Optional[List[str]] = Field(
        None,
        description="Filter by specific locations"
    )
    property_type_filter: Optional[List[str]] = Field(
        None,
        description="Filter by property types"
    )
    include_explanations: bool = Field(
        default=False,
        description="Include explanation for why properties were recommended"
    )


class SearchRequest(BaseModel):
    """Request model for property search with ML ranking"""
    query: str = Field(..., min_length=1, description="Search query")
    user_id: Optional[UUID] = Field(None, description="User ID for personalized ranking")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional filters (price, location, bedrooms, etc.)"
    )
    use_ml_ranking: bool = Field(
        default=True,
        description="Use ML-based ranking for search results"
    )
    include_scores: bool = Field(
        default=False,
        description="Include relevance scores in results"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    user_ids: List[UUID] = Field(..., description="List of user IDs")
    property_ids: Optional[List[UUID]] = Field(
        None,
        description="Optional list of property IDs to predict for"
    )
    model_type: ModelType = Field(
        default=ModelType.RECOMMENDER,
        description="Type of model to use for predictions"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for processing"
    )
    async_processing: bool = Field(
        default=True,
        description="Process asynchronously and return job ID"
    )


class ModelMetricsRequest(BaseModel):
    """Request model for model performance metrics"""
    model_type: ModelType = Field(..., description="Type of model to get metrics for")
    start_date: Optional[datetime] = Field(None, description="Start date for metrics")
    end_date: Optional[datetime] = Field(None, description="End date for metrics")
    user_segment: Optional[str] = Field(None, description="User segment to filter by")


class ABTestRequest(BaseModel):
    """Request model for A/B testing"""
    test_name: str = Field(..., description="Name of the A/B test")
    user_id: UUID = Field(..., description="User ID for test assignment")
    model_a: str = Field(..., description="Model A identifier")
    model_b: str = Field(..., description="Model B identifier")
    traffic_split: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Traffic split for model A (0.5 = 50/50 split)"
    )


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: UUID
    recommendations: List[Dict[str, Any]]
    recommendation_type: str
    total_count: int
    processing_time_ms: float
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    explanations: Optional[List[str]] = None


class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    results: List[Dict[str, Any]]
    total_count: int
    has_more: bool
    processing_time_ms: float
    ranking_model: str
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchJobResponse(BaseModel):
    """Response model for batch job status"""
    job_id: str
    status: BatchProcessingStatus
    progress: float = Field(ge=0.0, le=100.0)
    created_at: datetime
    updated_at: datetime
    result_count: Optional[int] = None
    error_message: Optional[str] = None


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics"""
    model_type: str
    metrics: Dict[str, float]
    performance_summary: Dict[str, Any]
    data_quality: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# Dependency injection
async def get_recommendation_service() -> RecommendationService:
    """Get recommendation service instance"""
    # This would be injected from the app state in a real implementation
    # For now, we'll create a mock service
    return RecommendationService()


async def get_search_service() -> SearchService:
    """Get search service instance"""
    return SearchService()


async def get_model_server() -> ModelServer:
    """Get model server instance"""
    return ModelServer()


async def get_feature_store() -> FeatureStore:
    """Get feature store instance"""
    return FeatureStore()


@ml_router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    recommendation_service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get personalized property recommendations for a user.
    
    Supports multiple recommendation algorithms:
    - Collaborative filtering: Based on similar users' preferences
    - Content-based: Based on property features and user preferences
    - Hybrid: Combines collaborative and content-based approaches
    """
    start_time = time.time()
    
    try:
        logger.info(f"Generating {request.recommendation_type} recommendations for user {request.user_id}")
        
        # Get user preferences and history
        user = await recommendation_service.get_user(request.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {request.user_id} not found"
            )
        
        # Apply filters based on request
        filters = {}
        if request.price_filter:
            filters["price_range"] = request.price_filter
        if request.location_filter:
            filters["locations"] = request.location_filter
        if request.property_type_filter:
            filters["property_types"] = request.property_type_filter
        
        # Get recommendations based on type
        if request.recommendation_type == RecommendationType.COLLABORATIVE:
            recommendations = await recommendation_service.get_collaborative_recommendations(
                user_id=request.user_id,
                num_recommendations=request.num_recommendations,
                filters=filters
            )
        elif request.recommendation_type == RecommendationType.CONTENT_BASED:
            recommendations = await recommendation_service.get_content_based_recommendations(
                user_id=request.user_id,
                num_recommendations=request.num_recommendations,
                filters=filters
            )
        else:  # HYBRID
            recommendations = await recommendation_service.get_hybrid_recommendations(
                user_id=request.user_id,
                num_recommendations=request.num_recommendations,
                filters=filters
            )
        
        # Filter out viewed/liked properties if requested
        if request.exclude_viewed:
            viewed_ids = user.get_viewed_properties()
            recommendations = [r for r in recommendations if r["property_id"] not in viewed_ids]
        
        if request.exclude_liked:
            liked_ids = user.get_liked_properties()
            recommendations = [r for r in recommendations if r["property_id"] not in liked_ids]
        
        # Generate explanations if requested
        explanations = None
        if request.include_explanations:
            explanations = await recommendation_service.generate_explanations(
                user_id=request.user_id,
                recommendations=recommendations,
                recommendation_type=request.recommendation_type
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log recommendation event for analytics
        background_tasks.add_task(
            log_recommendation_event,
            user_id=request.user_id,
            recommendation_type=request.recommendation_type,
            num_recommendations=len(recommendations),
            processing_time=processing_time
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            recommendation_type=request.recommendation_type.value,
            total_count=len(recommendations),
            processing_time_ms=processing_time,
            model_version="v2.0.0",
            explanations=explanations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@ml_router.post("/search", response_model=SearchResponse)
async def search_properties(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Search properties with ML-powered ranking.
    
    Features:
    - Natural language query processing
    - Personalized ranking based on user history
    - Advanced filtering capabilities
    - Real-time relevance scoring
    """
    start_time = time.time()
    
    try:
        logger.info(f"Searching properties for query: '{request.query}'")
        
        # Perform search with ML ranking
        if request.use_ml_ranking and request.user_id:
            results = await search_service.search_with_personalized_ranking(
                query=request.query,
                user_id=request.user_id,
                limit=request.limit,
                offset=request.offset,
                filters=request.filters
            )
        else:
            results = await search_service.search_properties(
                query=request.query,
                limit=request.limit,
                offset=request.offset,
                filters=request.filters
            )
        
        # Add relevance scores if requested
        if request.include_scores:
            for result in results:
                result["relevance_score"] = result.get("score", 0.0)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log search event
        background_tasks.add_task(
            log_search_event,
            query=request.query,
            user_id=request.user_id,
            result_count=len(results),
            processing_time=processing_time
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_count=len(results),
            has_more=len(results) == request.limit,
            processing_time_ms=processing_time,
            ranking_model="neural_ranking_v2"
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@ml_router.post("/batch-predictions", response_model=BatchJobResponse)
async def batch_predictions(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_server: ModelServer = Depends(get_model_server)
):
    """
    Submit batch prediction job for large-scale inference.
    
    Supports:
    - Batch recommendations for multiple users
    - Async processing for large datasets
    - Progress tracking and result retrieval
    """
    try:
        logger.info(f"Starting batch prediction for {len(request.user_ids)} users")
        
        # Generate job ID
        job_id = f"batch_{int(time.time())}_{len(request.user_ids)}"
        
        if request.async_processing:
            # Start async processing
            background_tasks.add_task(
                process_batch_predictions,
                job_id=job_id,
                request=request,
                model_server=model_server
            )
            
            return BatchJobResponse(
                job_id=job_id,
                status=BatchProcessingStatus.PENDING,
                progress=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        else:
            # Synchronous processing (for smaller batches)
            results = await model_server.batch_predict(
                user_ids=request.user_ids,
                property_ids=request.property_ids,
                model_type=request.model_type.value,
                batch_size=request.batch_size
            )
            
            return BatchJobResponse(
                job_id=job_id,
                status=BatchProcessingStatus.COMPLETED,
                progress=100.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                result_count=len(results)
            )
            
    except Exception as e:
        logger.error(f"Batch prediction submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit batch prediction: {str(e)}"
        )


@ml_router.get("/batch-predictions/{job_id}", response_model=BatchJobResponse)
async def get_batch_job_status(
    job_id: str,
    model_server: ModelServer = Depends(get_model_server)
):
    """Get status and results of a batch prediction job."""
    try:
        job_status = await model_server.get_batch_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job {job_id} not found"
            )
        
        return BatchJobResponse(**job_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@ml_router.post("/model-metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    request: ModelMetricsRequest,
    model_server: ModelServer = Depends(get_model_server)
):
    """
    Get comprehensive model performance metrics.
    
    Includes:
    - Prediction accuracy metrics
    - Model drift detection
    - Feature importance
    - Data quality metrics
    """
    try:
        logger.info(f"Getting metrics for {request.model_type} model")
        
        metrics = await model_server.get_model_metrics(
            model_type=request.model_type.value,
            start_date=request.start_date,
            end_date=request.end_date,
            user_segment=request.user_segment
        )
        
        return ModelMetricsResponse(
            model_type=request.model_type.value,
            metrics=metrics["metrics"],
            performance_summary=metrics["performance_summary"],
            data_quality=metrics["data_quality"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model metrics: {str(e)}"
        )


@ml_router.post("/ab-test", response_model=Dict[str, Any])
async def run_ab_test(
    request: ABTestRequest,
    background_tasks: BackgroundTasks,
    model_server: ModelServer = Depends(get_model_server)
):
    """
    Run A/B test between two models.
    
    Features:
    - Traffic splitting
    - Real-time metric comparison
    - Statistical significance testing
    """
    try:
        logger.info(f"Running A/B test '{request.test_name}' for user {request.user_id}")
        
        # Determine which model to use based on traffic split
        import random
        use_model_a = random.random() < request.traffic_split
        selected_model = request.model_a if use_model_a else request.model_b
        
        # Get prediction from selected model
        prediction = await model_server.predict_single(
            user_id=request.user_id,
            model_name=selected_model
        )
        
        # Log A/B test event
        background_tasks.add_task(
            log_ab_test_event,
            test_name=request.test_name,
            user_id=request.user_id,
            selected_model=selected_model,
            model_a=request.model_a,
            model_b=request.model_b
        )
        
        return {
            "test_name": request.test_name,
            "user_id": request.user_id,
            "selected_model": selected_model,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"A/B test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test failed: {str(e)}"
        )


@ml_router.get("/feature-importance/{model_type}")
async def get_feature_importance(
    model_type: ModelType,
    top_k: int = Query(default=20, ge=1, le=100),
    model_server: ModelServer = Depends(get_model_server)
):
    """Get feature importance for model debugging and interpretation."""
    try:
        importance = await model_server.get_feature_importance(
            model_type=model_type.value,
            top_k=top_k
        )
        
        return {
            "model_type": model_type.value,
            "feature_importance": importance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature importance: {str(e)}"
        )


@ml_router.post("/similar-properties/{property_id}")
async def get_similar_properties(
    property_id: UUID,
    num_similar: int = Query(default=10, ge=1, le=50),
    similarity_threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    model_server: ModelServer = Depends(get_model_server)
):
    """Find properties similar to a given property using content-based similarity."""
    try:
        similar_properties = await model_server.find_similar_properties(
            property_id=property_id,
            num_similar=num_similar,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "property_id": property_id,
            "similar_properties": similar_properties,
            "similarity_model": "content_similarity_v2",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to find similar properties: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar properties: {str(e)}"
        )


@ml_router.get("/model-health")
async def get_model_health(
    model_server: ModelServer = Depends(get_model_server)
):
    """Get health status of all ML models."""
    try:
        health_status = await model_server.check_model_health()
        
        return {
            "overall_status": health_status["overall_status"],
            "models": health_status["models"],
            "last_check": health_status["last_check"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to check model health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check model health: {str(e)}"
        )


# Background tasks
async def log_recommendation_event(
    user_id: UUID,
    recommendation_type: str,
    num_recommendations: int,
    processing_time: float
):
    """Log recommendation event for analytics."""
    logger.info(
        f"Recommendation event: user={user_id}, type={recommendation_type}, "
        f"count={num_recommendations}, time={processing_time:.2f}ms"
    )


async def log_search_event(
    query: str,
    user_id: Optional[UUID],
    result_count: int,
    processing_time: float
):
    """Log search event for analytics."""
    logger.info(
        f"Search event: query='{query}', user={user_id}, "
        f"results={result_count}, time={processing_time:.2f}ms"
    )


async def log_ab_test_event(
    test_name: str,
    user_id: UUID,
    selected_model: str,
    model_a: str,
    model_b: str
):
    """Log A/B test event for analysis."""
    logger.info(
        f"A/B test event: test={test_name}, user={user_id}, "
        f"selected={selected_model}, models={model_a}vs{model_b}"
    )


async def process_batch_predictions(
    job_id: str,
    request: BatchPredictionRequest,
    model_server: ModelServer
):
    """Process batch predictions asynchronously."""
    logger.info(f"Processing batch job {job_id}")
    
    try:
        # Update job status to processing
        await model_server.update_batch_job_status(
            job_id=job_id,
            status=BatchProcessingStatus.PROCESSING,
            progress=0.0
        )
        
        # Process in batches
        results = []
        total_users = len(request.user_ids)
        
        for i in range(0, total_users, request.batch_size):
            batch_users = request.user_ids[i:i + request.batch_size]
            
            # Process batch
            batch_results = await model_server.batch_predict(
                user_ids=batch_users,
                property_ids=request.property_ids,
                model_type=request.model_type.value,
                batch_size=request.batch_size
            )
            
            results.extend(batch_results)
            
            # Update progress
            progress = min(100.0, (i + len(batch_users)) / total_users * 100)
            await model_server.update_batch_job_status(
                job_id=job_id,
                status=BatchProcessingStatus.PROCESSING,
                progress=progress
            )
        
        # Mark as completed
        await model_server.update_batch_job_status(
            job_id=job_id,
            status=BatchProcessingStatus.COMPLETED,
            progress=100.0,
            result_count=len(results)
        )
        
        logger.info(f"Batch job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        await model_server.update_batch_job_status(
            job_id=job_id,
            status=BatchProcessingStatus.FAILED,
            error_message=str(e)
        )