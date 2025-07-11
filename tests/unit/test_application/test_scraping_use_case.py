"""
Unit tests for the ScrapingUseCase application service.

Tests scraping orchestration, scheduling, configuration, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
from datetime import datetime, timedelta

from application.use_cases.scraping_use_case import ScrapingUseCase
from tests.utils.test_helpers import MockHelpers, PerformanceTestHelpers
from tests.utils.data_factories import PropertyFactory, FactoryConfig


class MockScrapingStats:
    """Mock scraping statistics for testing."""
    
    def __init__(self, **kwargs):
        self.total_properties_found = kwargs.get('total_properties_found', 100)
        self.total_properties_saved = kwargs.get('total_properties_saved', 95)
        self.total_duplicates_filtered = kwargs.get('total_duplicates_filtered', 5)
        self.total_errors = kwargs.get('total_errors', 0)
        self.duration_seconds = kwargs.get('duration_seconds', 60.0)
        self.scrapers_used = kwargs.get('scrapers_used', ['apartments_com'])


class MockJobResult:
    """Mock job result for testing."""
    
    def __init__(self, **kwargs):
        self.job_id = kwargs.get('job_id', 'test_job')
        self.success = kwargs.get('success', True)
        self.started_at = kwargs.get('started_at', datetime.now())
        self.completed_at = kwargs.get('completed_at', datetime.now() + timedelta(minutes=5))
        self.error_message = kwargs.get('error_message', None)
        self.stats = kwargs.get('stats', MockScrapingStats())


class TestScrapingUseCase:
    """Test cases for ScrapingUseCase."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.mock_property_repo = Mock()
        self.mock_cache_repo = Mock()
        
        self.use_case = ScrapingUseCase(
            property_repository=self.mock_property_repo,
            cache_repository=self.mock_cache_repo
        )
        
        # Mock the scheduler
        self.mock_scheduler = Mock()
        self.use_case.scheduler = self.mock_scheduler
        
        self.property_factory = PropertyFactory(FactoryConfig(seed=42))
    
    def test_initialization(self):
        """Test ScrapingUseCase initialization."""
        assert self.use_case.property_repository == self.mock_property_repo
        assert self.use_case.cache_repository == self.mock_cache_repo
        assert self.use_case.default_config is not None
        assert not self.use_case._scheduler_initialized
        assert self.use_case.scheduler is not None  # We set it in setup
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = self.use_case.default_config
        
        assert config.max_concurrent_scrapers == 2
        assert config.max_properties_per_scraper == 500
        assert config.deduplication_enabled is True
        assert config.cache_results is True
        assert config.save_to_database is True
        assert config.global_rate_limit == 1.0
        assert config.scraper_delay == 10.0
    
    @pytest.mark.asyncio
    async def test_run_manual_scraping_success(self):
        """Test successful manual scraping."""
        expected_stats = MockScrapingStats(
            total_properties_found=150,
            total_properties_saved=140,
            total_duplicates_filtered=10
        )
        
        with patch('application.use_cases.scraping_use_case.ScrapingOrchestrator') as MockOrchestrator:
            mock_orchestrator_instance = AsyncMock()
            mock_orchestrator_instance.run_full_scraping_session = AsyncMock(return_value=expected_stats)
            
            # Setup async context manager
            mock_orchestrator_instance.__aenter__ = AsyncMock(return_value=mock_orchestrator_instance)
            mock_orchestrator_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockOrchestrator.return_value = mock_orchestrator_instance
            
            # Run manual scraping
            result = await self.use_case.run_manual_scraping(
                scrapers=['apartments_com'],
                max_properties=150,
                config_override={'max_concurrent_scrapers': 1}
            )
            
            # Assertions
            assert result == expected_stats
            assert result.total_properties_found == 150
            assert result.total_properties_saved == 140
            mock_orchestrator_instance.run_full_scraping_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_manual_scraping_with_defaults(self):
        """Test manual scraping with default parameters."""
        expected_stats = MockScrapingStats()
        
        with patch('application.use_cases.scraping_use_case.ScrapingOrchestrator') as MockOrchestrator:
            mock_orchestrator_instance = AsyncMock()
            mock_orchestrator_instance.run_full_scraping_session = AsyncMock(return_value=expected_stats)
            mock_orchestrator_instance.__aenter__ = AsyncMock(return_value=mock_orchestrator_instance)
            mock_orchestrator_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockOrchestrator.return_value = mock_orchestrator_instance
            
            result = await self.use_case.run_manual_scraping()
            
            assert result == expected_stats
            mock_orchestrator_instance.run_full_scraping_session.assert_called_once_with(
                scraper_names=None,
                max_properties=None
            )
    
    @pytest.mark.asyncio
    async def test_run_manual_scraping_failure(self):
        """Test manual scraping failure handling."""
        with patch('application.use_cases.scraping_use_case.ScrapingOrchestrator') as MockOrchestrator:
            mock_orchestrator_instance = AsyncMock()
            mock_orchestrator_instance.run_full_scraping_session = AsyncMock(
                side_effect=Exception("Scraping failed")
            )
            mock_orchestrator_instance.__aenter__ = AsyncMock(return_value=mock_orchestrator_instance)
            mock_orchestrator_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockOrchestrator.return_value = mock_orchestrator_instance
            
            with pytest.raises(Exception, match="Scraping failed"):
                await self.use_case.run_manual_scraping()
    
    @pytest.mark.asyncio
    async def test_initialize_scheduler_success(self):
        """Test successful scheduler initialization."""
        # Reset scheduler state
        self.use_case.scheduler = None
        self.use_case._scheduler_initialized = False
        
        with patch('application.use_cases.scraping_use_case.ScrapingScheduler') as MockScheduler:
            mock_scheduler = AsyncMock()
            mock_scheduler.start = AsyncMock()
            MockScheduler.return_value = mock_scheduler
            
            # Mock the setup method
            with patch.object(self.use_case, '_setup_default_scheduled_jobs', new_callable=AsyncMock):
                result = await self.use_case.initialize_scheduler()
            
            assert result is True
            assert self.use_case._scheduler_initialized is True
            assert self.use_case.scheduler == mock_scheduler
            mock_scheduler.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_scheduler_already_initialized(self):
        """Test scheduler initialization when already initialized."""
        self.use_case._scheduler_initialized = True
        
        result = await self.use_case.initialize_scheduler()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_initialize_scheduler_failure(self):
        """Test scheduler initialization failure."""
        self.use_case.scheduler = None
        self.use_case._scheduler_initialized = False
        
        with patch('application.use_cases.scraping_use_case.ScrapingScheduler') as MockScheduler:
            MockScheduler.side_effect = Exception("Scheduler init failed")
            
            result = await self.use_case.initialize_scheduler()
            
            assert result is False
            assert not self.use_case._scheduler_initialized
    
    @pytest.mark.asyncio
    async def test_shutdown_scheduler(self):
        """Test scheduler shutdown."""
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.stop = AsyncMock()
        
        await self.use_case.shutdown_scheduler()
        
        assert self.use_case.scheduler is None
        assert not self.use_case._scheduler_initialized
        self.mock_scheduler.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_scheduled_job_success(self):
        """Test successful scheduled job addition."""
        # Setup scheduler mock
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.add_job = Mock()
        self.mock_scheduler.get_job_status = Mock(return_value={
            'job_id': 'test_job',
            'name': 'Test Job',
            'enabled': True,
            'next_run': datetime.now().isoformat()
        })
        
        result = await self.use_case.add_scheduled_job(
            job_id='test_job',
            name='Test Job',
            schedule_type='daily',
            interval_hours=24.0,
            scrapers=['apartments_com'],
            max_properties=100,
            enabled=True
        )
        
        assert result['job_id'] == 'test_job'
        assert result['name'] == 'Test Job'
        assert result['enabled'] is True
        self.mock_scheduler.add_job.assert_called_once()
        self.mock_scheduler.get_job_status.assert_called_once_with('test_job')
    
    @pytest.mark.asyncio
    async def test_add_scheduled_job_with_config_override(self):
        """Test adding scheduled job with configuration override."""
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.add_job = Mock()
        self.mock_scheduler.get_job_status = Mock(return_value={})
        
        config_override = {
            'max_concurrent_scrapers': 1,
            'global_rate_limit': 0.5
        }
        
        await self.use_case.add_scheduled_job(
            job_id='test_job',
            name='Test Job',
            schedule_type='hourly',
            interval_hours=1.0,
            scrapers=['apartments_com'],
            config_override=config_override
        )
        
        # Verify add_job was called with config override
        call_args = self.mock_scheduler.add_job.call_args
        assert call_args is not None
        assert 'config_override' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_add_scheduled_job_scheduler_not_initialized(self):
        """Test adding job when scheduler is not initialized."""
        self.use_case._scheduler_initialized = False
        
        with patch.object(self.use_case, 'initialize_scheduler', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            self.mock_scheduler.add_job = Mock()
            self.mock_scheduler.get_job_status = Mock(return_value={})
            
            await self.use_case.add_scheduled_job(
                job_id='test_job',
                name='Test Job',
                schedule_type='daily',
                interval_hours=24.0,
                scrapers=['apartments_com']
            )
            
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_scheduled_job_success(self):
        """Test successful job removal."""
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.remove_job = Mock()
        
        result = await self.use_case.remove_scheduled_job('test_job')
        
        assert result is True
        self.mock_scheduler.remove_job.assert_called_once_with('test_job')
    
    @pytest.mark.asyncio
    async def test_remove_scheduled_job_failure(self):
        """Test job removal failure."""
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.remove_job = Mock(side_effect=Exception("Job not found"))
        
        result = await self.use_case.remove_scheduled_job('nonexistent_job')
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_enable_scheduled_job_success(self):
        """Test successful job enabling."""
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.enable_job = Mock()
        
        result = await self.use_case.enable_scheduled_job('test_job')
        
        assert result is True
        self.mock_scheduler.enable_job.assert_called_once_with('test_job')
    
    @pytest.mark.asyncio
    async def test_disable_scheduled_job_success(self):
        """Test successful job disabling."""
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.disable_job = Mock()
        
        result = await self.use_case.disable_scheduled_job('test_job')
        
        assert result is True
        self.mock_scheduler.disable_job.assert_called_once_with('test_job')
    
    @pytest.mark.asyncio
    async def test_run_scheduled_job_now_success(self):
        """Test running scheduled job immediately."""
        self.use_case._scheduler_initialized = True
        
        mock_result = MockJobResult(
            job_id='test_job',
            success=True,
            stats=MockScrapingStats(total_properties_found=50, total_properties_saved=48)
        )
        
        self.mock_scheduler.run_job_now = AsyncMock(return_value=mock_result)
        
        result = await self.use_case.run_scheduled_job_now('test_job')
        
        assert result['job_id'] == 'test_job'
        assert result['success'] is True
        assert result['stats']['properties_found'] == 50
        assert result['stats']['properties_saved'] == 48
        self.mock_scheduler.run_job_now.assert_called_once_with('test_job')
    
    @pytest.mark.asyncio
    async def test_run_scheduled_job_now_failure(self):
        """Test running scheduled job with failure."""
        self.use_case._scheduler_initialized = True
        
        mock_result = MockJobResult(
            job_id='test_job',
            success=False,
            error_message="Scraper timeout",
            stats=None
        )
        
        self.mock_scheduler.run_job_now = AsyncMock(return_value=mock_result)
        
        result = await self.use_case.run_scheduled_job_now('test_job')
        
        assert result['success'] is False
        assert result['error_message'] == "Scraper timeout"
        assert result['stats'] is None
    
    @pytest.mark.asyncio
    async def test_get_scraping_status_scheduler_initialized(self):
        """Test getting scraping status when scheduler is initialized."""
        self.use_case._scheduler_initialized = True
        
        mock_jobs_status = {
            'daily_scraping': {'enabled': True, 'next_run': '2024-01-02T00:00:00'},
            'weekly_comprehensive': {'enabled': False, 'next_run': None}
        }
        
        self.mock_scheduler.get_all_jobs_status = Mock(return_value=mock_jobs_status)
        
        with patch.object(self.use_case, '_get_recent_scraping_sessions', new_callable=AsyncMock) as mock_recent:
            mock_recent.return_value = {'sessions_last_24h': 3}
            
            result = await self.use_case.get_scraping_status()
        
        assert result['scheduler_initialized'] is True
        assert result['scheduled_jobs'] == mock_jobs_status
        assert result['recent_activity']['sessions_last_24h'] == 3
    
    @pytest.mark.asyncio
    async def test_get_scraping_status_scheduler_not_initialized(self):
        """Test getting scraping status when scheduler is not initialized."""
        self.use_case._scheduler_initialized = False
        
        result = await self.use_case.get_scraping_status()
        
        assert result['scheduler_initialized'] is False
        assert result['scheduled_jobs'] == {}
        assert result['recent_activity'] == {}
    
    @pytest.mark.asyncio
    async def test_get_job_history_success(self):
        """Test getting job execution history."""
        self.use_case._scheduler_initialized = True
        
        mock_history = [
            MockJobResult(success=True, stats=MockScrapingStats(total_properties_found=100)),
            MockJobResult(success=False, error_message="Network error", stats=None),
            MockJobResult(success=True, stats=MockScrapingStats(total_properties_found=85))
        ]
        
        self.mock_scheduler.job_history = {'test_job': mock_history}
        
        result = await self.use_case.get_job_history('test_job', limit=10)
        
        assert len(result) == 3
        assert result[0]['success'] is True
        assert result[1]['success'] is False
        assert result[1]['error_message'] == "Network error"
        assert result[2]['stats']['properties_found'] == 85
    
    @pytest.mark.asyncio
    async def test_get_job_history_no_history(self):
        """Test getting job history when no history exists."""
        self.use_case._scheduler_initialized = True
        self.mock_scheduler.job_history = {}
        
        result = await self.use_case.get_job_history('nonexistent_job')
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_available_scrapers(self):
        """Test getting available scrapers information."""
        result = await self.use_case.get_available_scrapers()
        
        assert len(result) > 0
        assert all('name' in scraper for scraper in result)
        assert all('display_name' in scraper for scraper in result)
        assert all('description' in scraper for scraper in result)
        
        # Check specific scraper details
        apartments_scraper = next(s for s in result if s['name'] == 'apartments_com')
        assert apartments_scraper['display_name'] == 'Apartments.com'
        assert 'supported_locations' in apartments_scraper
        assert 'rate_limit' in apartments_scraper
    
    def test_create_config_with_overrides(self):
        """Test configuration creation with overrides."""
        overrides = {
            'max_concurrent_scrapers': 1,
            'global_rate_limit': 0.5,
            'cache_results': False
        }
        
        config = self.use_case._create_config_with_overrides(overrides)
        
        assert config.max_concurrent_scrapers == 1
        assert config.global_rate_limit == 0.5
        assert config.cache_results is False
        
        # Values not overridden should use defaults
        assert config.deduplication_enabled == self.use_case.default_config.deduplication_enabled
        assert config.save_to_database == self.use_case.default_config.save_to_database
    
    def test_create_config_with_no_overrides(self):
        """Test configuration creation without overrides."""
        config = self.use_case._create_config_with_overrides(None)
        
        assert config == self.use_case.default_config
    
    def test_create_config_with_empty_overrides(self):
        """Test configuration creation with empty overrides."""
        config = self.use_case._create_config_with_overrides({})
        
        # Should return default config values
        assert config.max_concurrent_scrapers == self.use_case.default_config.max_concurrent_scrapers
        assert config.global_rate_limit == self.use_case.default_config.global_rate_limit
    
    @pytest.mark.asyncio
    async def test_ensure_scheduler_initialized_success(self):
        """Test ensuring scheduler is initialized when it's not."""
        self.use_case._scheduler_initialized = False
        
        with patch.object(self.use_case, 'initialize_scheduler', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            
            await self.use_case._ensure_scheduler_initialized()
            
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_scheduler_initialized_failure(self):
        """Test ensuring scheduler initialization when it fails."""
        self.use_case._scheduler_initialized = False
        
        with patch.object(self.use_case, 'initialize_scheduler', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = False
            
            with pytest.raises(RuntimeError, match="Failed to initialize scraping scheduler"):
                await self.use_case._ensure_scheduler_initialized()
    
    @pytest.mark.asyncio
    async def test_ensure_scheduler_initialized_already_initialized(self):
        """Test ensuring scheduler initialization when already initialized."""
        self.use_case._scheduler_initialized = True
        
        with patch.object(self.use_case, 'initialize_scheduler', new_callable=AsyncMock) as mock_init:
            await self.use_case._ensure_scheduler_initialized()
            
            mock_init.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_setup_default_scheduled_jobs(self):
        """Test setting up default scheduled jobs."""
        mock_scheduler = Mock()
        self.use_case.scheduler = mock_scheduler
        
        await self.use_case._setup_default_scheduled_jobs()
        
        # Should create daily and weekly jobs
        assert mock_scheduler.add_job.call_count == 2
        
        # Check daily job
        daily_call = mock_scheduler.add_job.call_args_list[0]
        assert daily_call.kwargs['job_id'] == 'daily_scraping'
        assert daily_call.kwargs['enabled'] is True
        
        # Check weekly job
        weekly_call = mock_scheduler.add_job.call_args_list[1]
        assert weekly_call.kwargs['job_id'] == 'weekly_comprehensive'
        assert weekly_call.kwargs['enabled'] is False
    
    @pytest.mark.asyncio
    async def test_setup_default_scheduled_jobs_failure(self):
        """Test setup of default jobs with partial failure."""
        mock_scheduler = Mock()
        mock_scheduler.add_job.side_effect = [None, Exception("Failed to create weekly job")]
        self.use_case.scheduler = mock_scheduler
        
        # Should not raise exception, just log warning
        await self.use_case._setup_default_scheduled_jobs()
        
        assert mock_scheduler.add_job.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_recent_scraping_sessions(self):
        """Test getting recent scraping sessions data."""
        result = await self.use_case._get_recent_scraping_sessions()
        
        assert isinstance(result, dict)
        assert 'sessions_last_24h' in result
        assert 'properties_scraped_last_24h' in result
        assert 'last_session_time' in result
        assert 'average_session_duration' in result
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_manual_scraping_performance(self):
        """Test manual scraping performance."""
        expected_stats = MockScrapingStats(
            total_properties_found=1000,
            total_properties_saved=950,
            duration_seconds=30.0
        )
        
        with patch('application.use_cases.scraping_use_case.ScrapingOrchestrator') as MockOrchestrator:
            mock_orchestrator_instance = AsyncMock()
            mock_orchestrator_instance.run_full_scraping_session = AsyncMock(return_value=expected_stats)
            mock_orchestrator_instance.__aenter__ = AsyncMock(return_value=mock_orchestrator_instance)
            mock_orchestrator_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockOrchestrator.return_value = mock_orchestrator_instance
            
            with PerformanceTestHelpers.measure_time() as timer:
                result = await self.use_case.run_manual_scraping(
                    scrapers=['apartments_com'],
                    max_properties=1000
                )
            
            elapsed_time = timer()
            
            # Should complete quickly (mocked operation)
            PerformanceTestHelpers.assert_performance_threshold(
                elapsed_time, threshold=1.0, operation="Manual scraping use case"
            )
            
            assert result.total_properties_found == 1000
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_operations(self):
        """Test handling multiple concurrent operations."""
        self.use_case._scheduler_initialized = True
        
        # Setup mocks for concurrent operations
        self.mock_scheduler.get_all_jobs_status = Mock(return_value={})
        self.mock_scheduler.enable_job = Mock()
        self.mock_scheduler.disable_job = Mock()
        
        with patch.object(self.use_case, '_get_recent_scraping_sessions', new_callable=AsyncMock) as mock_recent:
            mock_recent.return_value = {}
            
            # Run multiple operations concurrently
            tasks = [
                self.use_case.get_scraping_status(),
                self.use_case.enable_scheduled_job('job1'),
                self.use_case.disable_scheduled_job('job2'),
                self.use_case.get_scraping_status()
            ]
            
            results = await asyncio.gather(*tasks)
        
        # All operations should complete successfully
        assert len(results) == 4
        assert results[1] is True  # enable_job result
        assert results[2] is True  # disable_job result
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Test scheduler initialization failure recovery
        self.use_case._scheduler_initialized = False
        
        with patch('application.use_cases.scraping_use_case.ScrapingScheduler') as MockScheduler:
            MockScheduler.side_effect = [Exception("First attempt failed"), Mock()]
            
            # First attempt should fail
            result1 = await self.use_case.initialize_scheduler()
            assert result1 is False
            
            # Reset for second attempt
            MockScheduler.side_effect = None
            mock_scheduler = AsyncMock()
            mock_scheduler.start = AsyncMock()
            MockScheduler.return_value = mock_scheduler
            
            with patch.object(self.use_case, '_setup_default_scheduled_jobs', new_callable=AsyncMock):
                result2 = await self.use_case.initialize_scheduler()
            
            assert result2 is True
    
    @pytest.mark.asyncio
    async def test_configuration_edge_cases(self):
        """Test configuration handling with edge cases."""
        # Test with extreme values
        extreme_overrides = {
            'max_concurrent_scrapers': 0,
            'max_properties_per_scraper': 1000000,
            'global_rate_limit': 0.01,
            'scraper_delay': 0.0
        }
        
        config = self.use_case._create_config_with_overrides(extreme_overrides)
        
        assert config.max_concurrent_scrapers == 0
        assert config.max_properties_per_scraper == 1000000
        assert config.global_rate_limit == 0.01
        assert config.scraper_delay == 0.0
    
    @pytest.mark.asyncio
    async def test_scheduler_lifecycle_management(self):
        """Test complete scheduler lifecycle."""
        # Start with uninitialized state
        self.use_case._scheduler_initialized = False
        self.use_case.scheduler = None
        
        # Initialize
        with patch('application.use_cases.scraping_use_case.ScrapingScheduler') as MockScheduler:
            mock_scheduler = AsyncMock()
            mock_scheduler.start = AsyncMock()
            mock_scheduler.stop = AsyncMock()
            MockScheduler.return_value = mock_scheduler
            
            with patch.object(self.use_case, '_setup_default_scheduled_jobs', new_callable=AsyncMock):
                init_result = await self.use_case.initialize_scheduler()
            
            assert init_result is True
            assert self.use_case._scheduler_initialized is True
            
            # Shutdown
            await self.use_case.shutdown_scheduler()
            
            assert self.use_case.scheduler is None
            assert self.use_case._scheduler_initialized is False
            mock_scheduler.stop.assert_called_once()