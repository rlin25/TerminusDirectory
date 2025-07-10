#!/usr/bin/env python3
"""
Database initialization script for the Rental ML System.

This script creates the database, tables, and initial indexes.
Run this before starting the application for the first time.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.data.config import DataConfig
from src.infrastructure.data.repository_factory import RepositoryFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Initialize the database and create all tables"""
    try:
        logger.info("Starting database initialization...")
        
        # Load configuration
        config = DataConfig()
        logger.info(f"Database: {config.database.host}:{config.database.port}/{config.database.database}")
        logger.info(f"Redis: {config.redis.host}:{config.redis.port}/{config.redis.db}")
        
        # Create repository factory
        factory = RepositoryFactory(config)
        
        # Initialize all data connections
        await factory.initialize()
        logger.info("Data connections initialized successfully")
        
        # Test database connectivity
        user_repo = factory.get_user_repository()
        property_repo = factory.get_property_repository()
        model_repo = factory.get_model_repository()
        cache_repo = factory.get_cache_repository()
        
        # Test basic operations
        logger.info("Testing database operations...")
        
        # Test user repository
        active_users_count = await user_repo.get_active_users_count()
        logger.info(f"Active users count: {active_users_count}")
        
        # Test property repository
        total_properties = await property_repo.get_count()
        active_properties = await property_repo.get_active_count()
        logger.info(f"Properties: {active_properties}/{total_properties} active")
        
        # Test cache repository
        cache_health = await cache_repo.health_check()
        logger.info(f"Cache health: {'OK' if cache_health else 'FAILED'}")
        
        # Perform health check
        health_status = await factory.health_check()
        logger.info(f"Overall health check: {health_status}")
        
        if health_status.get("overall"):
            logger.info("✅ Database initialization completed successfully!")
        else:
            logger.error("❌ Database initialization completed with issues")
            return 1
        
        # Close connections
        await factory.close()
        logger.info("Database connections closed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return 1


def check_environment():
    """Check if required environment variables are set"""
    required_vars = [
        'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USERNAME', 'DB_PASSWORD',
        'REDIS_HOST', 'REDIS_PORT'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables or create a .env file")
        return False
    
    return True


if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run initialization
    exit_code = asyncio.run(main())
    sys.exit(exit_code)