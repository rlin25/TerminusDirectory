"""
Scraping API router for web scraping management and execution.

This module provides endpoints for manual scraping execution, scheduled job management,
job monitoring, and scraping system status.
"""

import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse

from ...dto.scraping_dto import (
    ManualScrapingRequest, ScrapingStatsResponse,
    ScheduledJobRequest, ScheduledJobUpdateRequest, ScheduledJobResponse,
    JobExecutionResponse, ScrapingStatusResponse, JobHistoryResponse,
    ScraperInfoResponse, ScrapersListResponse, ScrapingErrorResponse
)
from ...use_cases.scraping_use_case import ScrapingUseCase

logger = logging.getLogger(__name__)

router = APIRouter()


def get_repository_factory(request: Request):
    """Dependency to get repository factory from app state"""
    return request.app.state.repository_factory


def get_scraping_use_case(repository_factory = Depends(get_repository_factory)) -> ScrapingUseCase:
    """Dependency to get scraping use case"""
    property_repository = repository_factory.get_property_repository()
    cache_repository = repository_factory.get_cache_repository()
    return ScrapingUseCase(property_repository, cache_repository)


@router.post("/run", response_model=ScrapingStatsResponse, status_code=status.HTTP_200_OK)
async def run_manual_scraping(
    scraping_request: ManualScrapingRequest,
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Execute a manual scraping session.
    
    This endpoint triggers an immediate scraping operation with:
    - Configurable scraper selection
    - Property count limits
    - Custom configuration overrides
    - Real-time execution monitoring
    
    The operation runs synchronously and returns comprehensive statistics
    about the scraping session including success rates and performance metrics.
    """
    try:
        logger.info(f"Starting manual scraping with {len(scraping_request.scrapers)} scrapers")
        
        # Convert config override to dict if provided
        config_override = None
        if scraping_request.config_override:
            config_override = scraping_request.config_override.dict(exclude_unset=True)
        
        # Execute scraping
        stats = await scraping_use_case.run_manual_scraping(
            scrapers=scraping_request.scrapers if scraping_request.scrapers else None,
            max_properties=scraping_request.max_properties,
            config_override=config_override
        )
        
        # Calculate success rate
        success_rate = None
        if stats.total_properties_found > 0:
            success_rate = (stats.total_properties_saved / stats.total_properties_found) * 100
        
        response = ScrapingStatsResponse(
            total_properties_found=stats.total_properties_found,
            total_properties_saved=stats.total_properties_saved,
            total_duplicates_filtered=stats.total_duplicates_filtered,
            total_errors=stats.total_errors,
            duration_seconds=stats.duration_seconds,
            scrapers_used=stats.scrapers_used,
            success_rate=success_rate,
            properties_per_scraper=getattr(stats, 'properties_per_scraper', None)
        )
        
        logger.info(
            f"Manual scraping completed: {stats.total_properties_saved} properties saved "
            f"in {stats.duration_seconds:.2f} seconds"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Manual scraping failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scraping execution failed: {str(e)}"
        )


@router.get("/jobs", response_model=List[ScheduledJobResponse])
async def list_scheduled_jobs(
    enabled_only: bool = Query(default=False, description="Show only enabled jobs"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Get list of all scheduled scraping jobs.
    
    Returns comprehensive information about scheduled jobs including:
    - Job configuration and schedule details
    - Execution status and history
    - Next run times and frequency
    - Success rates and performance metrics
    """
    try:
        status_info = await scraping_use_case.get_scraping_status()
        scheduled_jobs = status_info.get('scheduled_jobs', {})
        
        job_responses = []
        for job_id, job_info in scheduled_jobs.items():
            # Skip disabled jobs if enabled_only is True
            if enabled_only and not job_info.get('enabled', True):
                continue
                
            job_response = ScheduledJobResponse(
                job_id=job_id,
                name=job_info.get('name', job_id),
                schedule_type=job_info.get('schedule_type', 'custom'),
                interval_hours=job_info.get('interval_hours', 24.0),
                scrapers=job_info.get('scrapers', []),
                max_properties=job_info.get('max_properties'),
                enabled=job_info.get('enabled', True),
                next_run=job_info.get('next_run'),
                last_run=job_info.get('last_run'),
                last_success=job_info.get('last_success'),
                total_runs=job_info.get('total_runs', 0),
                created_at=job_info.get('created_at', datetime.now())
            )
            job_responses.append(job_response)
        
        logger.info(f"Retrieved {len(job_responses)} scheduled jobs")
        return job_responses
        
    except Exception as e:
        logger.error(f"Failed to list scheduled jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scheduled jobs: {str(e)}"
        )


@router.post("/jobs", response_model=ScheduledJobResponse, status_code=status.HTTP_201_CREATED)
async def create_scheduled_job(
    job_request: ScheduledJobRequest,
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Create a new scheduled scraping job.
    
    Creates a new recurring scraping job with:
    - Flexible scheduling options (hourly, daily, weekly, custom)
    - Configurable scraper selection
    - Property limits and rate limiting
    - Custom configuration overrides
    
    The job will be automatically registered with the scheduler and begin
    executing according to its schedule if enabled.
    """
    try:
        # Generate unique job ID
        job_id = f"job_{int(time.time())}_{str(uuid4())[:8]}"
        
        # Convert config override to dict if provided
        config_override = None
        if job_request.config_override:
            config_override = job_request.config_override.dict(exclude_unset=True)
        
        # Create the scheduled job
        job_info = await scraping_use_case.add_scheduled_job(
            job_id=job_id,
            name=job_request.name,
            schedule_type=job_request.schedule_type.value,
            interval_hours=job_request.interval_hours,
            scrapers=job_request.scrapers,
            max_properties=job_request.max_properties,
            config_override=config_override,
            enabled=job_request.enabled
        )
        
        response = ScheduledJobResponse(
            job_id=job_id,
            name=job_request.name,
            schedule_type=job_request.schedule_type,
            interval_hours=job_request.interval_hours,
            scrapers=job_request.scrapers,
            max_properties=job_request.max_properties,
            enabled=job_request.enabled,
            next_run=job_info.get('next_run'),
            last_run=None,
            last_success=None,
            total_runs=0,
            created_at=job_info.get('created_at', datetime.now())
        )
        
        logger.info(f"Created scheduled job: {job_request.name} ({job_id})")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create scheduled job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scheduled job: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=ScheduledJobResponse)
async def get_scheduled_job(
    job_id: str = Path(..., description="Unique job identifier"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Get detailed information about a specific scheduled job.
    
    Returns comprehensive job details including:
    - Configuration and schedule information
    - Execution history and statistics
    - Current status and next run time
    - Performance metrics and success rates
    """
    try:
        status_info = await scraping_use_case.get_scraping_status()
        scheduled_jobs = status_info.get('scheduled_jobs', {})
        
        if job_id not in scheduled_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled job {job_id} not found"
            )
        
        job_info = scheduled_jobs[job_id]
        
        response = ScheduledJobResponse(
            job_id=job_id,
            name=job_info.get('name', job_id),
            schedule_type=job_info.get('schedule_type', 'custom'),
            interval_hours=job_info.get('interval_hours', 24.0),
            scrapers=job_info.get('scrapers', []),
            max_properties=job_info.get('max_properties'),
            enabled=job_info.get('enabled', True),
            next_run=job_info.get('next_run'),
            last_run=job_info.get('last_run'),
            last_success=job_info.get('last_success'),
            total_runs=job_info.get('total_runs', 0),
            created_at=job_info.get('created_at', datetime.now())
        )
        
        logger.info(f"Retrieved scheduled job: {job_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scheduled job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scheduled job: {str(e)}"
        )


@router.put("/jobs/{job_id}", response_model=ScheduledJobResponse)
async def update_scheduled_job(
    job_request: ScheduledJobUpdateRequest,
    job_id: str = Path(..., description="Unique job identifier"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Update an existing scheduled scraping job.
    
    Updates job configuration including:
    - Schedule timing and frequency
    - Scraper selection and limits
    - Enable/disable status
    - Configuration overrides
    
    Changes take effect on the next scheduled execution.
    """
    try:
        # First check if job exists
        status_info = await scraping_use_case.get_scraping_status()
        scheduled_jobs = status_info.get('scheduled_jobs', {})
        
        if job_id not in scheduled_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled job {job_id} not found"
            )
        
        # For this implementation, we'll need to remove and recreate the job
        # since the ScrapingUseCase doesn't have an update method
        current_job = scheduled_jobs[job_id]
        
        # Remove existing job
        await scraping_use_case.remove_scheduled_job(job_id)
        
        # Create updated job with same ID
        config_override = None
        if job_request.config_override:
            config_override = job_request.config_override.dict(exclude_unset=True)
        
        # Use existing values if not provided in update
        updated_job_info = await scraping_use_case.add_scheduled_job(
            job_id=job_id,
            name=job_request.name or current_job.get('name', job_id),
            schedule_type=(job_request.schedule_type.value if job_request.schedule_type 
                          else current_job.get('schedule_type', 'custom')),
            interval_hours=(job_request.interval_hours if job_request.interval_hours is not None 
                           else current_job.get('interval_hours', 24.0)),
            scrapers=job_request.scrapers or current_job.get('scrapers', []),
            max_properties=(job_request.max_properties if job_request.max_properties is not None 
                           else current_job.get('max_properties')),
            config_override=config_override,
            enabled=(job_request.enabled if job_request.enabled is not None 
                    else current_job.get('enabled', True))
        )
        
        response = ScheduledJobResponse(
            job_id=job_id,
            name=job_request.name or current_job.get('name', job_id),
            schedule_type=(job_request.schedule_type if job_request.schedule_type 
                          else current_job.get('schedule_type', 'custom')),
            interval_hours=(job_request.interval_hours if job_request.interval_hours is not None 
                           else current_job.get('interval_hours', 24.0)),
            scrapers=job_request.scrapers or current_job.get('scrapers', []),
            max_properties=(job_request.max_properties if job_request.max_properties is not None 
                           else current_job.get('max_properties')),
            enabled=(job_request.enabled if job_request.enabled is not None 
                    else current_job.get('enabled', True)),
            next_run=updated_job_info.get('next_run'),
            last_run=current_job.get('last_run'),
            last_success=current_job.get('last_success'),
            total_runs=current_job.get('total_runs', 0),
            created_at=current_job.get('created_at', datetime.now())
        )
        
        logger.info(f"Updated scheduled job: {job_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update scheduled job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update scheduled job: {str(e)}"
        )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scheduled_job(
    job_id: str = Path(..., description="Unique job identifier"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Delete a scheduled scraping job.
    
    Permanently removes the job from the scheduler. This action cannot be undone.
    The job will stop executing immediately and all associated configuration
    will be lost.
    
    Execution history may be preserved for analytics purposes.
    """
    try:
        success = await scraping_use_case.remove_scheduled_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled job {job_id} not found"
            )
        
        logger.info(f"Deleted scheduled job: {job_id}")
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete scheduled job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete scheduled job: {str(e)}"
        )


@router.post("/jobs/{job_id}/run", response_model=JobExecutionResponse)
async def run_scheduled_job_now(
    job_id: str = Path(..., description="Unique job identifier"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Execute a scheduled job immediately.
    
    Triggers an immediate execution of the specified job outside of its
    normal schedule. This is useful for:
    - Testing job configuration
    - Emergency data collection
    - Manual intervention scenarios
    
    The execution runs asynchronously and returns execution details
    including success status and performance metrics.
    """
    try:
        result = await scraping_use_case.run_scheduled_job_now(job_id)
        
        # Convert string timestamps to datetime objects if needed
        started_at = result['started_at']
        if isinstance(started_at, str):
            from datetime import datetime
            started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        
        completed_at = result['completed_at']
        if isinstance(completed_at, str):
            from datetime import datetime
            completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
        
        response = JobExecutionResponse(
            job_id=result['job_id'],
            success=result['success'],
            started_at=started_at,
            completed_at=completed_at,
            error_message=result.get('error_message'),
            stats=ScrapingStatsResponse(**result['stats']) if result.get('stats') else None
        )
        
        logger.info(f"Executed scheduled job {job_id}: success={result['success']}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to run scheduled job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute job: {str(e)}"
        )


@router.get("/status", response_model=ScrapingStatusResponse)
async def get_scraping_status(
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Get comprehensive scraping system status.
    
    Returns detailed system information including:
    - Scheduler initialization status
    - All scheduled jobs and their states
    - Recent scraping activity and metrics
    - System health indicators
    - Performance statistics
    
    This endpoint is useful for monitoring and debugging the scraping system.
    """
    try:
        status_info = await scraping_use_case.get_scraping_status()
        
        # Add system health metrics
        system_health = {
            "scheduler_running": status_info.get('scheduler_initialized', False),
            "active_jobs": len([
                job for job in status_info.get('scheduled_jobs', {}).values()
                if job.get('enabled', True)
            ]),
            "total_jobs": len(status_info.get('scheduled_jobs', {})),
            "last_successful_scraping": None,  # TODO: Implement
            "system_load": "normal",  # TODO: Implement actual system monitoring
            "memory_usage": "within_limits",  # TODO: Implement
            "error_rate": 0.0  # TODO: Calculate from recent activity
        }
        
        response = ScrapingStatusResponse(
            scheduler_initialized=status_info.get('scheduler_initialized', False),
            scheduled_jobs=status_info.get('scheduled_jobs', {}),
            recent_activity=status_info.get('recent_activity', {}),
            system_health=system_health
        )
        
        logger.info("Retrieved scraping system status")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get scraping status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system status: {str(e)}"
        )


@router.get("/jobs/{job_id}/history", response_model=JobHistoryResponse)
async def get_job_history(
    job_id: str = Path(..., description="Unique job identifier"),
    limit: int = Query(default=50, ge=1, le=200, description="Number of history entries to return"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Get execution history for a scheduled job.
    
    Returns detailed execution history including:
    - Execution timestamps and durations
    - Success/failure status for each run
    - Error messages and debugging information
    - Performance metrics and statistics
    - Trend analysis and patterns
    
    History is ordered by execution time (most recent first).
    """
    try:
        history_data = await scraping_use_case.get_job_history(job_id, limit)
        
        if not history_data:
            # Check if job exists
            status_info = await scraping_use_case.get_scraping_status()
            if job_id not in status_info.get('scheduled_jobs', {}):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scheduled job {job_id} not found"
                )
        
        # Convert history data to response format
        execution_history = []
        total_executions = len(history_data)
        successful_executions = 0
        total_duration = 0.0
        
        for entry in history_data:
            # Convert string timestamps to datetime objects if needed
            started_at = entry['started_at']
            if isinstance(started_at, str):
                from datetime import datetime
                started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            
            completed_at = entry['completed_at']
            if isinstance(completed_at, str):
                from datetime import datetime
                completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
            
            execution = JobExecutionResponse(
                job_id=job_id,
                success=entry['success'],
                started_at=started_at,
                completed_at=completed_at,
                error_message=entry.get('error_message'),
                stats=ScrapingStatsResponse(**entry['stats']) if entry.get('stats') else None
            )
            execution_history.append(execution)
            
            if entry['success']:
                successful_executions += 1
            
            if entry.get('stats', {}).get('duration_seconds'):
                total_duration += entry['stats']['duration_seconds']
        
        # Calculate metrics
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0.0
        average_duration = total_duration / total_executions if total_executions > 0 else 0.0
        
        response = JobHistoryResponse(
            job_id=job_id,
            history=execution_history,
            total_executions=total_executions,
            success_rate=success_rate,
            average_duration=average_duration
        )
        
        logger.info(f"Retrieved history for job {job_id}: {total_executions} executions")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job history for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job history: {str(e)}"
        )


@router.get("/scrapers", response_model=ScrapersListResponse)
async def get_available_scrapers(
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Get list of available scrapers and their capabilities.
    
    Returns detailed information about all registered scrapers including:
    - Scraper names and descriptions
    - Supported locations and property types
    - Rate limits and performance characteristics
    - Current status and availability
    - Configuration options and requirements
    
    This information is useful for configuring scraping jobs and
    understanding system capabilities.
    """
    try:
        scrapers_data = await scraping_use_case.get_available_scrapers()
        
        scrapers = []
        active_count = 0
        
        for scraper_info in scrapers_data:
            scraper = ScraperInfoResponse(
                name=scraper_info['name'],
                display_name=scraper_info['display_name'],
                description=scraper_info['description'],
                supported_locations=scraper_info['supported_locations'],
                rate_limit=scraper_info['rate_limit'],
                estimated_properties_per_run=scraper_info['estimated_properties_per_run'],
                last_run=None,  # TODO: Implement last run tracking
                status="available"
            )
            scrapers.append(scraper)
            
            if scraper.status == "available":
                active_count += 1
        
        response = ScrapersListResponse(
            scrapers=scrapers,
            total_count=len(scrapers),
            active_count=active_count
        )
        
        logger.info(f"Retrieved {len(scrapers)} available scrapers")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get available scrapers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scrapers: {str(e)}"
        )


@router.patch("/jobs/{job_id}/enable", response_model=ScheduledJobResponse)
async def enable_scheduled_job(
    job_id: str = Path(..., description="Unique job identifier"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Enable a scheduled scraping job.
    
    Activates a previously disabled job, allowing it to execute according
    to its schedule. The job will resume normal operation starting with
    the next scheduled run time.
    """
    try:
        success = await scraping_use_case.enable_scheduled_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled job {job_id} not found"
            )
        
        # Get updated job information
        status_info = await scraping_use_case.get_scraping_status()
        job_info = status_info.get('scheduled_jobs', {}).get(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled job {job_id} not found"
            )
        
        response = ScheduledJobResponse(
            job_id=job_id,
            name=job_info.get('name', job_id),
            schedule_type=job_info.get('schedule_type', 'custom'),
            interval_hours=job_info.get('interval_hours', 24.0),
            scrapers=job_info.get('scrapers', []),
            max_properties=job_info.get('max_properties'),
            enabled=True,
            next_run=job_info.get('next_run'),
            last_run=job_info.get('last_run'),
            last_success=job_info.get('last_success'),
            total_runs=job_info.get('total_runs', 0),
            created_at=job_info.get('created_at', datetime.now())
        )
        
        logger.info(f"Enabled scheduled job: {job_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable scheduled job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable job: {str(e)}"
        )


@router.patch("/jobs/{job_id}/disable", response_model=ScheduledJobResponse)
async def disable_scheduled_job(
    job_id: str = Path(..., description="Unique job identifier"),
    scraping_use_case: ScrapingUseCase = Depends(get_scraping_use_case)
):
    """
    Disable a scheduled scraping job.
    
    Temporarily deactivates a job without deleting it. The job configuration
    is preserved and can be re-enabled at any time. Currently running
    executions will complete normally.
    """
    try:
        success = await scraping_use_case.disable_scheduled_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled job {job_id} not found"
            )
        
        # Get updated job information
        status_info = await scraping_use_case.get_scraping_status()
        job_info = status_info.get('scheduled_jobs', {}).get(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled job {job_id} not found"
            )
        
        response = ScheduledJobResponse(
            job_id=job_id,
            name=job_info.get('name', job_id),
            schedule_type=job_info.get('schedule_type', 'custom'),
            interval_hours=job_info.get('interval_hours', 24.0),
            scrapers=job_info.get('scrapers', []),
            max_properties=job_info.get('max_properties'),
            enabled=False,
            next_run=None,  # No next run when disabled
            last_run=job_info.get('last_run'),
            last_success=job_info.get('last_success'),
            total_runs=job_info.get('total_runs', 0),
            created_at=job_info.get('created_at', datetime.now())
        )
        
        logger.info(f"Disabled scheduled job: {job_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable scheduled job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable job: {str(e)}"
        )