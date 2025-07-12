"""
Main production scraping system that integrates all components.

This module provides the main entry point for the production-ready
rental property scraping system with full monitoring, compliance, and management.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import argparse

from .production_scraping_orchestrator import ProductionScrapingOrchestrator, ScrapingScheduler
from .production_database_connector import ProductionDatabaseConnector
from .production_monitoring import ProductionMonitor
from .gdpr_compliance import GDPRCompliance
from .generic_scraping_framework import GenericScrapingFramework
from .config import get_config, Environment

logger = logging.getLogger(__name__)


class ProductionScrapingSystem:
    """Main production scraping system"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config = get_config(environment)
        
        # Core components
        self.database_connector: Optional[ProductionDatabaseConnector] = None
        self.orchestrator: Optional[ProductionScrapingOrchestrator] = None
        self.monitor: Optional[ProductionMonitor] = None
        self.scheduler: Optional[ScrapingScheduler] = None
        self.gdpr_compliance: Optional[GDPRCompliance] = None
        self.generic_framework: Optional[GenericScrapingFramework] = None
        
        # System state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized production scraping system for {environment.value}")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.monitoring.log_level.value)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if configured
        if self.config.monitoring.log_to_file:
            try:
                import os
                os.makedirs(os.path.dirname(self.config.monitoring.log_file_path), exist_ok=True)
                
                file_handler = logging.FileHandler(self.config.monitoring.log_file_path)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                
                logger.info(f"Logging to file: {self.config.monitoring.log_file_path}")
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
    
    async def initialize(self):
        """Initialize all system components"""
        
        logger.info("Initializing production scraping system components...")
        
        try:
            # Initialize database connector
            self.database_connector = ProductionDatabaseConnector(self.config)
            await self.database_connector.initialize()
            logger.info("Database connector initialized")
            
            # Initialize GDPR compliance
            self.gdpr_compliance = GDPRCompliance(self.config)
            logger.info("GDPR compliance initialized")
            
            # Initialize orchestrator
            self.orchestrator = ProductionScrapingOrchestrator(
                self.config, 
                self.database_connector
            )
            logger.info("Scraping orchestrator initialized")
            
            # Initialize monitoring
            self.monitor = ProductionMonitor(
                self.config,
                self.database_connector,
                None  # Redis client would go here
            )
            
            # Connect orchestrator events to monitoring
            self.orchestrator.register_event_callback('job_started', self.monitor.on_job_started)
            self.orchestrator.register_event_callback('job_completed', self.monitor.on_job_completed)
            self.orchestrator.register_event_callback('job_failed', self.monitor.on_job_failed)
            self.orchestrator.register_event_callback('property_found', self.monitor.on_property_found)
            
            logger.info("Monitoring system initialized")
            
            # Initialize scheduler
            self.scheduler = ScrapingScheduler(self.orchestrator)
            
            # Setup default schedules based on environment
            if self.environment == Environment.PRODUCTION:
                # Production: Conservative daily scraping
                self.scheduler.schedule_daily_scraping(
                    hour=2, minute=0, max_pages_per_source=20
                )
            elif self.environment == Environment.STAGING:
                # Staging: Less frequent scraping
                self.scheduler.schedule_weekly_scraping(
                    weekday=1, hour=1, minute=0, max_pages_per_source=10
                )
            
            logger.info("Scheduler initialized")
            
            # Initialize generic framework
            self.generic_framework = GenericScrapingFramework(self.config)
            logger.info("Generic scraping framework initialized")
            
            logger.info("All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system components: {e}")
            raise
    
    async def start(self):
        """Start the production scraping system"""
        
        if self.is_running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting production scraping system...")
        
        try:
            # Start monitoring
            if self.monitor:
                monitoring_task = asyncio.create_task(self.monitor.start_monitoring())
                logger.info("Monitoring started")
            
            # Start scheduler
            if self.scheduler:
                scheduler_task = asyncio.create_task(self.scheduler.run_scheduler())
                logger.info("Scheduler started")
            
            self.is_running = True
            logger.info("Production scraping system started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        
        if not self.is_running:
            return
        
        logger.info("Shutting down production scraping system...")
        
        try:
            # Stop scheduler
            if self.scheduler:
                self.scheduler.stop()
                logger.info("Scheduler stopped")
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
                logger.info("Monitoring stopped")
            
            # Cancel any running jobs
            if self.orchestrator:
                for job_id in list(self.orchestrator.active_jobs.keys()):
                    self.orchestrator.cancel_job(job_id)
                logger.info("Cancelled active scraping jobs")
            
            # Close database connections
            if self.database_connector:
                await self.database_connector.close()
                logger.info("Database connections closed")
            
            self.is_running = False
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def run_manual_scraping(
        self,
        sources: list = None,
        max_pages_per_source: int = 5,
        test_mode: bool = False
    ) -> Dict[str, Any]:
        """Run manual scraping session"""
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        job_type = 'test' if test_mode else 'manual'
        
        logger.info(f"Starting manual scraping session (test_mode={test_mode})")
        
        # Create session
        session = self.orchestrator.create_session(
            sources=sources,
            session_type='manual',
            job_type=job_type,
            max_pages_per_source=max_pages_per_source
        )
        
        # Execute session
        completed_session = await self.orchestrator.execute_session(session)
        
        # Generate results
        results = {
            'session_id': completed_session.id,
            'session_type': completed_session.session_type,
            'started_at': completed_session.started_at.isoformat() if completed_session.started_at else None,
            'completed_at': completed_session.completed_at.isoformat() if completed_session.completed_at else None,
            'duration_minutes': (
                (completed_session.completed_at - completed_session.started_at).total_seconds() / 60
                if completed_session.started_at and completed_session.completed_at else 0
            ),
            'total_properties': completed_session.total_properties,
            'jobs': [
                {
                    'source': job.source_name,
                    'status': job.status.value,
                    'properties_found': job.properties_found,
                    'properties_valid': job.properties_valid,
                    'properties_stored': job.properties_stored,
                    'error_message': job.error_message
                }
                for job in completed_session.jobs
            ]
        }
        
        logger.info(f"Manual scraping session completed: {results['total_properties']} properties collected")
        return results
    
    async def test_all_scrapers(self) -> Dict[str, Any]:
        """Test all available scrapers"""
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        logger.info("Testing all available scrapers...")
        
        results = await self.orchestrator.test_scrapers()
        
        logger.info(f"Scraper testing completed: {len(results)} scrapers tested")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment.value,
            'system_running': self.is_running,
            'components': {}
        }
        
        # Database status
        if self.database_connector:
            status['components']['database'] = {
                'initialized': True,
                'connected': self.database_connector.connection_pool is not None
            }
        
        # Orchestrator status
        if self.orchestrator:
            status['components']['orchestrator'] = self.orchestrator.get_metrics()
        
        # Monitoring status
        if self.monitor:
            status['components']['monitoring'] = {
                'active': self.monitor.monitoring_active,
                'health_checks': {
                    name: hc.status for name, hc in self.monitor.health_checker.health_checks.items()
                }
            }
        
        # Scheduler status
        if self.scheduler:
            status['components']['scheduler'] = {
                'running': self.scheduler.running,
                'scheduled_jobs': len(self.scheduler.scheduled_jobs)
            }
        
        return status
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        
        if not self.monitor:
            return {'error': 'Monitoring not initialized'}
        
        return self.monitor.get_monitoring_dashboard()


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Production Rental Property Scraping System')
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'staging', 'production'],
                       default='development',
                       help='Environment to run in')
    parser.add_argument('--command', '-c',
                       choices=['run', 'test', 'scrape', 'status'],
                       default='run',
                       help='Command to execute')
    parser.add_argument('--sources', '-s',
                       nargs='*',
                       help='Sources to scrape (for manual scraping)')
    parser.add_argument('--max-pages', '-p',
                       type=int,
                       default=5,
                       help='Max pages per source (for manual scraping)')
    parser.add_argument('--test-mode', '-t',
                       action='store_true',
                       help='Run in test mode (limited scraping)')
    
    args = parser.parse_args()
    
    # Create system
    environment = Environment(args.environment)
    system = ProductionScrapingSystem(environment)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)
    
    try:
        # Initialize system
        await system.initialize()
        
        if args.command == 'run':
            # Run the full system
            await system.start()
            
        elif args.command == 'test':
            # Test all scrapers
            results = await system.test_all_scrapers()
            print("Scraper Test Results:")
            for source, result in results.items():
                status = "✓" if result['success'] else "✗"
                print(f"  {status} {source}: {result.get('properties_found', 0)} properties")
                if not result['success']:
                    print(f"    Error: {result.get('error', 'Unknown error')}")
        
        elif args.command == 'scrape':
            # Run manual scraping
            results = await system.run_manual_scraping(
                sources=args.sources,
                max_pages_per_source=args.max_pages,
                test_mode=args.test_mode
            )
            print(f"Manual Scraping Results:")
            print(f"  Session ID: {results['session_id']}")
            print(f"  Total Properties: {results['total_properties']}")
            print(f"  Duration: {results['duration_minutes']:.1f} minutes")
            print(f"  Jobs:")
            for job in results['jobs']:
                print(f"    {job['source']}: {job['properties_valid']} properties ({job['status']})")
        
        elif args.command == 'status':
            # Show system status
            status = system.get_system_status()
            print("System Status:")
            print(f"  Environment: {status['environment']}")
            print(f"  Running: {status['system_running']}")
            print(f"  Components:")
            for component, info in status['components'].items():
                print(f"    {component}: {info}")
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())