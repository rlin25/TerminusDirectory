#!/usr/bin/env python3
"""
Test script for PostgreSQL User Repository Implementation
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from uuid import uuid4
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.domain.entities.user import User, UserPreferences, UserInteraction
from src.infrastructure.data.repositories.postgres_user_repository import PostgreSQLUserRepository
from src.infrastructure.data.config import DataConfig
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_user_repository():
    """Test the PostgreSQL User Repository implementation"""
    
    # Load configuration
    config = DataConfig()
    
    # Create connection pool
    pool = await asyncpg.create_pool(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        user=config.database.username,
        password=config.database.password,
        min_size=1,
        max_size=3
    )
    
    try:
        # Initialize repository
        repo = PostgreSQLUserRepository(pool)
        
        # Test health check
        logger.info("Testing health check...")
        health = await repo.health_check()
        logger.info(f"Health check result: {health}")
        
        # Test user creation
        logger.info("Testing user creation...")
        preferences = UserPreferences(
            min_price=1000.0,
            max_price=3000.0,
            min_bedrooms=1,
            max_bedrooms=3,
            preferred_locations=["San Francisco", "Oakland"],
            required_amenities=["parking", "laundry"],
            property_types=["apartment", "condo"]
        )
        
        user = User.create(
            email="test@example.com",
            preferences=preferences
        )
        
        created_user = await repo.create(user)
        logger.info(f"Created user: {created_user.id}")
        
        # Test get by ID
        logger.info("Testing get by ID...")
        retrieved_user = await repo.get_by_id(created_user.id)
        assert retrieved_user is not None
        assert retrieved_user.email == "test@example.com"
        logger.info(f"Retrieved user: {retrieved_user.id}")
        
        # Test get by email
        logger.info("Testing get by email...")
        retrieved_by_email = await repo.get_by_email("test@example.com")
        assert retrieved_by_email is not None
        assert retrieved_by_email.id == created_user.id
        logger.info(f"Retrieved by email: {retrieved_by_email.id}")
        
        # Test add interaction
        logger.info("Testing add interaction...")
        interaction = UserInteraction.create(
            property_id=uuid4(),
            interaction_type="view",
            duration_seconds=45
        )
        
        success = await repo.add_interaction(created_user.id, interaction)
        assert success
        logger.info("Added interaction successfully")
        
        # Test get interactions
        logger.info("Testing get interactions...")
        interactions = await repo.get_interactions(created_user.id)
        assert len(interactions) > 0
        logger.info(f"Retrieved {len(interactions)} interactions")
        
        # Test update user
        logger.info("Testing update user...")
        updated_preferences = UserPreferences(
            min_price=1500.0,
            max_price=4000.0,
            min_bedrooms=2,
            max_bedrooms=4,
            preferred_locations=["San Francisco", "Berkeley"],
            required_amenities=["parking", "laundry", "gym"],
            property_types=["apartment"]
        )
        
        user.preferences = updated_preferences
        updated_user = await repo.update(user)
        assert updated_user.preferences.min_price == 1500.0
        logger.info("Updated user successfully")
        
        # Test get all active users
        logger.info("Testing get all active users...")
        active_users = await repo.get_all_active(limit=10)
        assert len(active_users) > 0
        logger.info(f"Retrieved {len(active_users)} active users")
        
        # Test user statistics
        logger.info("Testing user statistics...")
        stats = await repo.get_user_statistics(created_user.id)
        assert "total_interactions" in stats
        logger.info(f"User statistics: {stats}")
        
        # Test get count
        logger.info("Testing get count...")
        total_count = await repo.get_count()
        active_count = await repo.get_active_count()
        assert total_count >= 1
        assert active_count >= 1
        logger.info(f"Total users: {total_count}, Active users: {active_count}")
        
        # Test behavior analytics
        logger.info("Testing behavior analytics...")
        analytics = await repo.get_user_behavior_analytics(days=30)
        assert "active_users" in analytics
        logger.info(f"Behavior analytics: {analytics}")
        
        # Test user segmentation
        logger.info("Testing user segmentation...")
        segmentation = await repo.get_user_segmentation()
        assert "activity_segments" in segmentation
        logger.info(f"User segmentation: {segmentation}")
        
        # Test delete user (soft delete)
        logger.info("Testing delete user...")
        deleted = await repo.delete(created_user.id)
        assert deleted
        
        # Verify user is soft deleted
        deleted_user = await repo.get_by_id(created_user.id)
        assert deleted_user is None  # Should not be found because we filter by active status
        logger.info("User deleted successfully")
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Close connection pool
        await pool.close()


if __name__ == "__main__":
    asyncio.run(test_user_repository())