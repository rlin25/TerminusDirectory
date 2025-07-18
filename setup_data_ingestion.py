#!/usr/bin/env python3
"""
Data Ingestion Pipeline Setup Script

This script sets up the data ingestion pipeline to populate the production database
with real property data using the existing scraping infrastructure and data generators.

Features:
- Uses existing production scraping orchestrator
- Integrates with production database seeder
- Provides data quality validation
- Supports both scraping and sample data generation
- Creates ongoing data pipeline with scheduling
"""

import asyncio
import logging
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.database.connection_manager import get_connection_manager, database_transaction
from database.seeds.production_data_seeder import ProductionDataSeeder, SeedingMode, DataQuality
from src.infrastructure.scrapers.production_scraping_orchestrator import (
    ProductionScrapingOrchestrator, ScrapingScheduler
)
from src.infrastructure.scrapers.config import get_config
from src.presentation.demo.sample_data import SampleDataGenerator
from src.domain.entities.property import Property
from src.domain.entities.user import User, UserInteraction
from src.infrastructure.data.repositories.postgres_property_repository import PostgresPropertyRepository
from src.infrastructure.data.repositories.postgres_user_repository import PostgresUserRepository

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionManager:
    """
    Manages the complete data ingestion pipeline for the rental ML system
    """
    
    def __init__(self):
        self.connection_manager = None
        self.data_seeder = None
        self.scraping_orchestrator = None
        self.scraping_scheduler = None
        self.sample_data_generator = SampleDataGenerator()
        
        # Repositories
        self.property_repo = None
        self.user_repo = None
        
        # Configuration
        self.config = {
            "use_real_scraping": True,  # Set to False for development/testing
            "enable_sample_data": True,  # Add sample data alongside scraped data
            "seeding_mode": SeedingMode.DEVELOPMENT,  # Change for different environments
            "enable_scheduling": True,  # Enable ongoing scraping
            "data_validation": True,
            "max_properties_per_source": 100,  # Limit for initial setup
            "properties_sample_count": 500,
            "users_sample_count": 100,
            "interactions_sample_count": 1000
        }
        
    async def initialize(self) -> None:
        """Initialize all components"""
        logger.info("Initializing data ingestion manager...")
        
        try:
            # Initialize database connection manager
            self.connection_manager = await get_connection_manager()
            logger.info("Database connection manager initialized")
            
            # Initialize data seeder
            self.data_seeder = ProductionDataSeeder(self.connection_manager)
            logger.info("Data seeder initialized")
            
            # Initialize repositories
            self.property_repo = PostgresPropertyRepository(self.connection_manager)
            self.user_repo = PostgresUserRepository(self.connection_manager)
            logger.info("Repositories initialized")
            
            # Initialize scraping components if enabled
            if self.config["use_real_scraping"]:
                await self._initialize_scraping_components()
                
            logger.info("Data ingestion manager initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize data ingestion manager: {e}")
            raise
            
    async def _initialize_scraping_components(self) -> None:
        """Initialize scraping orchestrator and scheduler"""
        try:
            # Get scraping configuration
            scraping_config = get_config()
            
            # Initialize scraping orchestrator
            self.scraping_orchestrator = ProductionScrapingOrchestrator(
                config=scraping_config,
                database_connector=self.property_repo
            )
            
            # Initialize scheduler
            self.scraping_scheduler = ScrapingScheduler(self.scraping_orchestrator)
            
            # Set up event callbacks for monitoring
            self.scraping_orchestrator.register_event_callback(
                'property_found', self._on_property_scraped
            )
            
            logger.info("Scraping components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize scraping components: {e}")
            raise
            
    async def _on_property_scraped(self, data: Dict[str, Any]) -> None:
        """Callback for when a property is scraped"""
        property_data = data.get('property')
        validation_result = data.get('validation_result')
        
        if property_data and validation_result:
            logger.info(
                f"Property scraped: {property_data.title} "
                f"(Quality Score: {validation_result.score:.2f})"
            )
            
    async def setup_initial_data(self) -> Dict[str, Any]:
        """
        Set up initial data in the database using multiple sources
        """
        logger.info("Starting initial data setup...")
        start_time = time.time()
        
        results = {
            "scraped_data": {},
            "sample_data": {},
            "validation": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # Step 1: Create base data using seeder (users, basic structure)
            if self.config["enable_sample_data"]:
                logger.info("Creating sample users and interaction data...")
                seeding_results = await self._create_sample_data()
                results["sample_data"] = seeding_results
                
            # Step 2: Scrape real property data if enabled
            if self.config["use_real_scraping"] and self.scraping_orchestrator:
                logger.info("Starting property data scraping...")
                scraping_results = await self._scrape_property_data()
                results["scraped_data"] = scraping_results
                
            # Step 3: Validate data integrity
            if self.config["data_validation"]:
                logger.info("Validating data integrity...")
                validation_results = await self._validate_data_integrity()
                results["validation"] = validation_results
                
            # Step 4: Set up ongoing data pipeline
            if self.config["enable_scheduling"] and self.scraping_scheduler:
                logger.info("Setting up ongoing data pipeline...")
                await self._setup_ongoing_pipeline()
                
            # Calculate performance metrics
            total_time = time.time() - start_time
            results["performance"] = {
                "total_time_seconds": round(total_time, 2),
                "setup_completed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Initial data setup completed in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Failed to setup initial data: {e}")
            results["errors"].append(str(e))
            return results
            
    async def _create_sample_data(self) -> Dict[str, Any]:
        """Create sample data using the data seeder"""
        try:
            # Use the production data seeder for realistic data
            seeding_results = await self.data_seeder.seed_database(
                mode=self.config["seeding_mode"],
                tables=['users', 'user_interactions', 'search_queries', 'ml_models'],
                incremental=False
            )
            
            # Also generate some additional sample properties using the demo generator
            logger.info("Generating additional sample properties...")
            
            sample_properties = self.sample_data_generator.generate_properties(
                count=self.config["properties_sample_count"]
            )
            
            # Store sample properties in database
            properties_stored = 0
            async with database_transaction() as conn:
                for prop in sample_properties:
                    try:
                        await self.property_repo.create(prop)
                        properties_stored += 1
                    except Exception as e:
                        logger.warning(f"Failed to store sample property: {e}")
                        
            seeding_results["sample_properties_created"] = properties_stored
            logger.info(f"Created {properties_stored} sample properties")
            
            return seeding_results
            
        except Exception as e:
            logger.error(f"Failed to create sample data: {e}")
            return {"error": str(e)}
            
    async def _scrape_property_data(self) -> Dict[str, Any]:
        """Scrape real property data using the production orchestrator"""
        try:
            # Create a scraping session for initial data population
            session = self.scraping_orchestrator.create_session(
                sources=['apartments_com', 'rentals_com', 'zillow'],  # Use available scrapers
                session_type='initial_setup',
                job_type='full_scrape',
                max_pages_per_source=self.config["max_properties_per_source"]
            )
            
            # Execute the scraping session
            completed_session = await self.scraping_orchestrator.execute_session(session)
            
            # Get results
            session_metrics = self.scraping_orchestrator.get_metrics()
            
            results = {
                "session_id": completed_session.id,
                "total_properties_scraped": completed_session.total_properties,
                "jobs_completed": len([j for j in completed_session.jobs if j.status.value == 'completed']),
                "jobs_failed": len([j for j in completed_session.jobs if j.status.value == 'failed']),
                "duration_seconds": (completed_session.completed_at - completed_session.started_at).total_seconds() if completed_session.completed_at else None,
                "metrics": session_metrics
            }
            
            logger.info(f"Scraping completed: {results['total_properties_scraped']} properties")
            return results
            
        except Exception as e:
            logger.error(f"Failed to scrape property data: {e}")
            return {"error": str(e)}
            
    async def _validate_data_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of ingested data"""
        try:
            validation_results = {
                "properties": {"count": 0, "active": 0, "with_images": 0},
                "users": {"count": 0, "with_preferences": 0},
                "interactions": {"count": 0},
                "orphaned_records": 0,
                "data_quality_scores": [],
                "validation_passed": True
            }
            
            async with database_transaction(read_only=True) as conn:
                # Check properties
                prop_count = await conn.fetchval("SELECT COUNT(*) FROM properties")
                active_prop_count = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE status = 'active'")
                props_with_images = await conn.fetchval("SELECT COUNT(*) FROM properties WHERE array_length(images, 1) > 0")
                
                validation_results["properties"].update({
                    "count": prop_count,
                    "active": active_prop_count,
                    "with_images": props_with_images
                })
                
                # Check users
                user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
                users_with_prefs = await conn.fetchval("SELECT COUNT(*) FROM users WHERE preferred_locations IS NOT NULL")
                
                validation_results["users"].update({
                    "count": user_count,
                    "with_preferences": users_with_prefs
                })
                
                # Check interactions
                interaction_count = await conn.fetchval("SELECT COUNT(*) FROM user_interactions")
                validation_results["interactions"]["count"] = interaction_count
                
                # Check for orphaned records
                orphaned_interactions = await conn.fetchval("""
                    SELECT COUNT(*) FROM user_interactions ui
                    WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.id = ui.user_id)
                    OR NOT EXISTS (SELECT 1 FROM properties p WHERE p.id = ui.property_id)
                """)
                validation_results["orphaned_records"] = orphaned_interactions
                
                # Get data quality scores
                quality_scores = await conn.fetch("""
                    SELECT data_quality_score FROM properties 
                    WHERE data_quality_score IS NOT NULL
                """)
                validation_results["data_quality_scores"] = [row['data_quality_score'] for row in quality_scores]
                
            # Determine if validation passed
            validation_results["validation_passed"] = (
                validation_results["properties"]["count"] > 0 and
                validation_results["users"]["count"] > 0 and
                validation_results["orphaned_records"] == 0
            )
            
            if validation_results["validation_passed"]:
                logger.info("Data integrity validation passed")
            else:
                logger.warning("Data integrity validation failed")
                
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate data integrity: {e}")
            return {"error": str(e), "validation_passed": False}
            
    async def _setup_ongoing_pipeline(self) -> None:
        """Set up ongoing data pipeline with scheduling"""
        try:
            # Schedule daily property updates
            self.scraping_scheduler.schedule_daily_scraping(
                hour=2,  # 2 AM daily
                minute=0,
                sources=['apartments_com', 'rentals_com'],
                max_pages_per_source=50
            )
            
            # Schedule weekly full scraping
            self.scraping_scheduler.schedule_weekly_scraping(
                weekday=0,  # Monday
                hour=1,
                minute=0,
                sources=['apartments_com', 'rentals_com', 'zillow'],
                max_pages_per_source=200
            )
            
            # Start the scheduler in background
            asyncio.create_task(self.scraping_scheduler.run_scheduler())
            
            logger.info("Ongoing data pipeline scheduled successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup ongoing pipeline: {e}")
            raise
            
    async def test_data_ingestion(self) -> Dict[str, Any]:
        """Test the data ingestion pipeline"""
        logger.info("Testing data ingestion pipeline...")
        
        test_results = {
            "database_connection": False,
            "sample_data_generation": False,
            "scraping_test": False,
            "property_search": False,
            "recommendations": False,
            "errors": []
        }
        
        try:
            # Test database connection
            async with database_transaction() as conn:
                result = await conn.fetchval("SELECT 1")
                test_results["database_connection"] = (result == 1)
                
            # Test sample data generation
            sample_props = self.sample_data_generator.generate_properties(count=5)
            sample_users = self.sample_data_generator.generate_users(count=3)
            test_results["sample_data_generation"] = len(sample_props) == 5 and len(sample_users) == 3
            
            # Test scraping (if enabled)
            if self.config["use_real_scraping"] and self.scraping_orchestrator:
                scraper_test_results = await self.scraping_orchestrator.test_scrapers()
                test_results["scraping_test"] = any(result.get('success', False) for result in scraper_test_results.values())
            else:
                test_results["scraping_test"] = True  # Skip if disabled
                
            # Test property search
            try:
                async with database_transaction(read_only=True) as conn:
                    properties = await conn.fetch("SELECT * FROM properties LIMIT 5")
                    test_results["property_search"] = len(properties) > 0
            except Exception as e:
                test_results["errors"].append(f"Property search test failed: {e}")
                
            # Test basic recommendation capability
            try:
                async with database_transaction(read_only=True) as conn:
                    user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
                    prop_count = await conn.fetchval("SELECT COUNT(*) FROM properties")
                    test_results["recommendations"] = user_count > 0 and prop_count > 0
            except Exception as e:
                test_results["errors"].append(f"Recommendations test failed: {e}")
                
            # Overall test result
            test_results["overall_success"] = all([
                test_results["database_connection"],
                test_results["sample_data_generation"],
                test_results["scraping_test"],
                test_results["property_search"],
                test_results["recommendations"]
            ])
            
            if test_results["overall_success"]:
                logger.info("All data ingestion tests passed!")
            else:
                logger.warning("Some data ingestion tests failed")
                
            return test_results
            
        except Exception as e:
            logger.error(f"Data ingestion test failed: {e}")
            test_results["errors"].append(str(e))
            return test_results
            
    async def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of current data in the system"""
        try:
            summary = {}
            
            async with database_transaction(read_only=True) as conn:
                # Properties summary
                prop_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'active') as active,
                        COUNT(*) FILTER (WHERE scraped_at > NOW() - INTERVAL '7 days') as recent,
                        AVG(price) as avg_price,
                        AVG(data_quality_score) as avg_quality
                    FROM properties
                """)
                
                # Users summary
                user_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE last_login > NOW() - INTERVAL '30 days') as active_users
                    FROM users
                """)
                
                # Interactions summary
                interaction_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT property_id) as unique_properties
                    FROM user_interactions
                """)
                
                # Top locations
                top_locations = await conn.fetch("""
                    SELECT location, COUNT(*) as property_count
                    FROM properties
                    GROUP BY location
                    ORDER BY property_count DESC
                    LIMIT 5
                """)
                
                summary = {
                    "properties": dict(prop_stats) if prop_stats else {},
                    "users": dict(user_stats) if user_stats else {},
                    "interactions": dict(interaction_stats) if interaction_stats else {},
                    "top_locations": [dict(row) for row in top_locations],
                    "generated_at": datetime.now().isoformat()
                }
                
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {"error": str(e)}
            
    async def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up data ingestion manager...")
        
        try:
            if self.scraping_scheduler:
                self.scraping_scheduler.stop()
                
            if self.connection_manager:
                await self.connection_manager.close()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main execution function"""
    print("🏠 Rental ML System - Data Ingestion Pipeline Setup")
    print("=" * 50)
    
    manager = DataIngestionManager()
    
    try:
        # Initialize the manager
        await manager.initialize()
        
        # Test the pipeline first
        print("\n📋 Testing data ingestion pipeline...")
        test_results = await manager.test_data_ingestion()
        
        if not test_results.get("overall_success", False):
            print("❌ Data ingestion tests failed. Please check configuration.")
            print(f"Errors: {test_results.get('errors', [])}")
            return
            
        print("✅ Data ingestion tests passed!")
        
        # Set up initial data
        print("\n🚀 Setting up initial data...")
        setup_results = await manager.setup_initial_data()
        
        # Display results
        print("\n📊 Setup Results:")
        print("-" * 30)
        
        if setup_results.get("sample_data"):
            sample_data = setup_results["sample_data"]
            print(f"Sample Data Created:")
            print(f"  - Total records: {sample_data.get('total_records_created', 0)}")
            print(f"  - Duration: {sample_data.get('total_duration_seconds', 0):.2f}s")
            
        if setup_results.get("scraped_data"):
            scraped_data = setup_results["scraped_data"]
            print(f"Scraped Data:")
            print(f"  - Properties: {scraped_data.get('total_properties_scraped', 0)}")
            print(f"  - Jobs completed: {scraped_data.get('jobs_completed', 0)}")
            print(f"  - Jobs failed: {scraped_data.get('jobs_failed', 0)}")
            
        if setup_results.get("validation"):
            validation = setup_results["validation"]
            status = "✅ PASSED" if validation.get("validation_passed") else "❌ FAILED"
            print(f"Data Validation: {status}")
            print(f"  - Properties: {validation.get('properties', {}).get('count', 0)}")
            print(f"  - Users: {validation.get('users', {}).get('count', 0)}")
            print(f"  - Interactions: {validation.get('interactions', {}).get('count', 0)}")
            
        # Get and display data summary
        print("\n📈 Data Summary:")
        print("-" * 30)
        summary = await manager.get_data_summary()
        
        if "error" not in summary:
            props = summary.get("properties", {})
            users = summary.get("users", {})
            interactions = summary.get("interactions", {})
            
            print(f"Properties: {props.get('total', 0)} total, {props.get('active', 0)} active")
            if props.get("avg_price"):
                print(f"  - Average price: ${props['avg_price']:,.2f}")
            if props.get("avg_quality"):
                print(f"  - Average quality score: {props['avg_quality']:.2f}")
                
            print(f"Users: {users.get('total', 0)} total")
            print(f"Interactions: {interactions.get('total', 0)} total")
            
            top_locations = summary.get("top_locations", [])
            if top_locations:
                print("Top locations:")
                for loc in top_locations[:3]:
                    print(f"  - {loc['location']}: {loc['property_count']} properties")
        
        print("\n🎉 Data ingestion pipeline setup completed successfully!")
        print("\nNext steps:")
        print("1. Verify data in your database")
        print("2. Test the property search functionality")
        print("3. Test the recommendation system")
        print("4. Monitor the ongoing scraping scheduler")
        
        # Keep the scheduler running if enabled
        if manager.config["enable_scheduling"]:
            print("\n⏰ Ongoing data pipeline is now active")
            print("Press Ctrl+C to stop...")
            
            try:
                # Keep the process running
                while True:
                    await asyncio.sleep(60)
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping data pipeline...")
                
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("Check the logs for more details.")
        
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    # Set environment variables for development if not set
    os.environ.setdefault("DB_HOST", "localhost")
    os.environ.setdefault("DB_PORT", "5432")
    os.environ.setdefault("DB_NAME", "rental_ml")
    os.environ.setdefault("DB_USER", "rental_ml_user")
    os.environ.setdefault("DB_PASSWORD", "your_password_here")
    
    # Run the setup
    asyncio.run(main())