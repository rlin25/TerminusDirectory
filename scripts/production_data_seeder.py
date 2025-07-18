#!/usr/bin/env python3
"""
Production Data Seeder for Rental ML System
Adapts SampleDataGenerator to populate PostgreSQL database with realistic data
"""

import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uuid

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load production environment
load_dotenv(project_root / ".env.production")

from src.presentation.demo.sample_data import SampleDataGenerator
from src.infrastructure.data.repository_factory import DatabaseConnectionManager
from src.domain.entities.property import Property
from src.domain.entities.user import User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDataSeeder:
    """Production data seeder that populates PostgreSQL with realistic data"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_manager = DatabaseConnectionManager(database_url)
        self.sample_generator = SampleDataGenerator()
        
    async def initialize(self):
        """Initialize database connection"""
        await self.connection_manager.initialize()
        logger.info("Database connection initialized")
        
    async def close(self):
        """Close database connections"""
        await self.connection_manager.close()
        logger.info("Database connections closed")
        
    async def clear_existing_data(self):
        """Clear existing data from tables"""
        logger.info("Clearing existing data...")
        
        async with self.connection_manager.get_connection() as conn:
            # Clear tables in dependency order
            tables = [
                'user_interactions',
                'search_queries', 
                'training_metrics',
                'ml_models',
                'embeddings',
                'properties',
                'users'
            ]
            
            for table in tables:
                try:
                    await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                    logger.info(f"Cleared table: {table}")
                except Exception as e:
                    logger.warning(f"Could not clear table {table}: {e}")
        
        logger.info("Data clearing completed")
        
    async def seed_users(self, count: int = 100) -> List[str]:
        """Seed users table with realistic data"""
        logger.info(f"Seeding {count} users...")
        
        # Generate users using sample data generator
        users = self.sample_generator.generate_users(count)
        user_ids = []
        
        async with self.connection_manager.get_connection() as conn:
            for user in users:
                # Convert user to database format
                user_data = {
                    'id': str(user.id),
                    'email': user.email,
                    'created_at': user.created_at,
                    'updated_at': user.created_at,
                    'status': 'active',
                    'min_price': user.preferences.min_price,
                    'max_price': user.preferences.max_price,
                    'min_bedrooms': user.preferences.min_bedrooms,
                    'max_bedrooms': user.preferences.max_bedrooms,
                    'min_bathrooms': user.preferences.min_bathrooms,
                    'max_bathrooms': user.preferences.max_bathrooms,
                    'preferred_locations': user.preferences.preferred_locations,
                    'required_amenities': user.preferences.required_amenities,
                    'property_types': user.preferences.property_types,
                    'last_login': None,
                    'login_count': 0,
                    'preference_updated_at': user.created_at
                }
                
                try:
                    await conn.execute("""
                        INSERT INTO users (
                            id, email, is_active, min_price, max_price, min_bedrooms, max_bedrooms,
                            min_bathrooms, max_bathrooms, preferred_locations,
                            required_amenities, property_types
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """, 
                        user_data['id'], user_data['email'], True,  # is_active
                        user_data['min_price'], user_data['max_price'], user_data['min_bedrooms'], 
                        user_data['max_bedrooms'], user_data['min_bathrooms'], user_data['max_bathrooms'], 
                        user_data['preferred_locations'], user_data['required_amenities'],
                        user_data['property_types']
                    )
                    user_ids.append(user_data['id'])
                except Exception as e:
                    logger.error(f"Failed to insert user {user.email}: {e}")
        
        logger.info(f"Successfully seeded {len(user_ids)} users")
        return user_ids
        
    async def seed_properties(self, count: int = 500) -> List[str]:
        """Seed properties table with realistic data"""
        logger.info(f"Seeding {count} properties...")
        
        # Generate properties using sample data generator
        properties = self.sample_generator.generate_properties(count)
        property_ids = []
        
        async with self.connection_manager.get_connection() as conn:
            for prop in properties:
                # Convert property to database format
                property_data = {
                    'id': str(prop.id),
                    'title': prop.title,
                    'description': prop.description,
                    'price': float(prop.price),
                    'location': prop.location,
                    'bedrooms': prop.bedrooms,
                    'bathrooms': float(prop.bathrooms),
                    'square_feet': prop.square_feet,
                    'amenities': prop.amenities,
                    'contact_info': prop.contact_info,
                    'images': prop.images,
                    'scraped_at': prop.scraped_at or datetime.now(),
                    'status': 'active' if prop.is_active else 'inactive',
                    'property_type': prop.property_type,
                    'created_at': prop.scraped_at or datetime.now(),
                    'updated_at': prop.scraped_at or datetime.now(),
                    'latitude': None,  # Could be added later
                    'longitude': None,  # Could be added later
                    'slug': prop.title.lower().replace(' ', '-').replace(',', ''),
                    'external_id': f"seed_{prop.id}",
                    'external_url': f"https://example.com/property/{prop.id}",
                    'data_quality_score': 0.85,
                    'last_verified': prop.scraped_at or datetime.now()
                }
                
                try:
                    await conn.execute("""
                        INSERT INTO properties (
                            id, title, description, price, location, bedrooms, bathrooms,
                            square_feet, amenities, contact_info, images, scraped_at,
                            is_active, property_type
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """, 
                        property_data['id'], property_data['title'], property_data['description'],
                        property_data['price'], property_data['location'], property_data['bedrooms'],
                        property_data['bathrooms'], property_data['square_feet'], property_data['amenities'],
                        json.dumps(property_data['contact_info']) if property_data['contact_info'] else '{}', 
                        property_data['images'], property_data['scraped_at'],
                        prop.is_active, property_data['property_type']
                    )
                    property_ids.append(property_data['id'])
                except Exception as e:
                    logger.error(f"Failed to insert property {prop.title}: {e}")
        
        logger.info(f"Successfully seeded {len(property_ids)} properties")
        return property_ids
        
    async def seed_interactions(self, user_ids: List[str], property_ids: List[str], count: int = 2000):
        """Seed user interactions table"""
        logger.info(f"Seeding {count} user interactions...")
        
        # Create dummy users and properties for interaction generation
        dummy_users = [User.create(email=f"user_{uid}@example.com") for uid in user_ids[:50]]  # Use subset
        dummy_properties = [Property.create(
            title=f"Property {pid}", 
            description="Test property", 
            price=1000, 
            location="Test Location",
            bedrooms=2,
            bathrooms=1,
            square_feet=800,
            amenities=[],
            contact_info={},
            images=[]
        ) for pid in property_ids[:100]]  # Use subset
        
        # Set IDs to match our seeded data
        for i, user in enumerate(dummy_users):
            user.id = uuid.UUID(user_ids[i])
        for i, prop in enumerate(dummy_properties):
            prop.id = uuid.UUID(property_ids[i])
        
        # Generate interactions
        interactions = self.sample_generator.generate_interactions(dummy_users, dummy_properties, count)
        
        async with self.connection_manager.get_connection() as conn:
            for interaction in interactions:
                try:
                    await conn.execute("""
                        INSERT INTO user_interactions (
                            id, user_id, property_id, interaction_type, timestamp,
                            duration_seconds, session_id, user_agent, ip_address,
                            referrer, interaction_strength
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, 
                        str(uuid.uuid4()), str(interaction.user_id), str(interaction.property_id),
                        interaction.interaction_type, interaction.timestamp, interaction.duration_seconds,
                        str(uuid.uuid4()), 'Mozilla/5.0 SampleDataBot', '127.0.0.1',
                        'direct', 0.5
                    )
                except Exception as e:
                    logger.error(f"Failed to insert interaction: {e}")
        
        logger.info(f"Successfully seeded {len(interactions)} user interactions")
        
    async def seed_search_queries(self, user_ids: List[str], count: int = 1000):
        """Seed search queries table"""
        logger.info(f"Seeding {count} search queries...")
        
        query_patterns = [
            "2 bedroom apartment downtown",
            "cheap studio apartment", 
            "luxury condo with pool",
            "pet friendly house",
            "apartment under $2000",
            "furnished apartment short term",
            "3 bedroom house with garage",
            "studio apartment near university"
        ]
        
        async with self.connection_manager.get_connection() as conn:
            for i in range(count):
                import random
                user_id = random.choice(user_ids) if random.random() > 0.3 else None
                query_text = random.choice(query_patterns)
                
                try:
                    await conn.execute("""
                        INSERT INTO search_queries (
                            id, user_id, query_text, filters, results_count,
                            execution_time_ms, created_at, session_id, page_number,
                            page_size, sort_by
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, 
                        str(uuid.uuid4()), user_id, query_text, {},
                        random.randint(0, 100), random.randint(50, 500), datetime.now(),
                        str(uuid.uuid4()), 1, 20, 'relevance'
                    )
                except Exception as e:
                    logger.error(f"Failed to insert search query: {e}")
        
        logger.info(f"Successfully seeded {count} search queries")
        
    async def create_sample_ml_models(self):
        """Create sample ML model records"""
        logger.info("Creating sample ML model records...")
        
        models = [
            {
                'id': str(uuid.uuid4()),
                'model_name': 'collaborative_filter',
                'version': 'v1.0.0',
                'model_file_path': '/models/collaborative_filter/v1.0.0/model.pkl',
                'metadata': {'algorithm': 'matrix_factorization', 'features': 50},
                'created_at': datetime.now(),
                'is_active': True,
                'training_accuracy': 0.85,
                'validation_accuracy': 0.82,
                'training_time_seconds': 1200,
                'model_size_bytes': 1024 * 1024,
                'parent_model_id': None
            },
            {
                'id': str(uuid.uuid4()),
                'model_name': 'content_recommender',
                'version': 'v1.0.0',
                'model_file_path': '/models/content_recommender/v1.0.0/model.pkl',
                'metadata': {'algorithm': 'tfidf_cosine', 'features': 100},
                'created_at': datetime.now(),
                'is_active': True,
                'training_accuracy': 0.80,
                'validation_accuracy': 0.78,
                'training_time_seconds': 800,
                'model_size_bytes': 512 * 1024,
                'parent_model_id': None
            }
        ]
        
        async with self.connection_manager.get_connection() as conn:
            for model in models:
                try:
                    await conn.execute("""
                        INSERT INTO ml_models (
                            id, model_name, version, model_file_path, metadata,
                            created_at, is_active, training_accuracy, validation_accuracy,
                            training_time_seconds, model_size_bytes, parent_model_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """, 
                        model['id'], model['model_name'], model['version'], model['model_file_path'],
                        model['metadata'], model['created_at'], model['is_active'],
                        model['training_accuracy'], model['validation_accuracy'],
                        model['training_time_seconds'], model['model_size_bytes'], model['parent_model_id']
                    )
                except Exception as e:
                    logger.error(f"Failed to insert ML model {model['model_name']}: {e}")
        
        logger.info(f"Successfully created {len(models)} ML model records")
        
    async def verify_data(self) -> Dict[str, int]:
        """Verify seeded data by counting records in each table"""
        logger.info("Verifying seeded data...")
        
        counts = {}
        tables = ['users', 'properties', 'user_interactions', 'search_queries', 'ml_models']
        
        async with self.connection_manager.get_connection() as conn:
            for table in tables:
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = count
                    logger.info(f"{table}: {count} records")
                except Exception as e:
                    logger.error(f"Failed to count {table}: {e}")
                    counts[table] = 0
        
        return counts
        
    async def seed_all(self, users_count: int = 100, properties_count: int = 500, 
                      interactions_count: int = 2000, queries_count: int = 1000, 
                      clear_existing: bool = True) -> Dict[str, Any]:
        """Seed all data tables"""
        start_time = datetime.now()
        results = {
            'started_at': start_time.isoformat(),
            'settings': {
                'users_count': users_count,
                'properties_count': properties_count,
                'interactions_count': interactions_count,
                'queries_count': queries_count,
                'clear_existing': clear_existing
            },
            'results': {},
            'errors': []
        }
        
        try:
            await self.initialize()
            
            if clear_existing:
                await self.clear_existing_data()
            
            # Seed core data
            user_ids = await self.seed_users(users_count)
            property_ids = await self.seed_properties(properties_count)
            
            # Seed derived data
            await self.seed_interactions(user_ids, property_ids, interactions_count)
            await self.seed_search_queries(user_ids, queries_count)
            await self.create_sample_ml_models()
            
            # Verify results
            counts = await self.verify_data()
            results['results'] = counts
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results['completed_at'] = end_time.isoformat()
            results['duration_seconds'] = duration
            results['success'] = True
            
            logger.info(f"Data seeding completed successfully in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Data seeding failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
            
        finally:
            await self.close()
            
        return results


async def main():
    """Main function for command-line usage"""
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rental_ml")
    
    seeder = ProductionDataSeeder(database_url)
    
    # Default seeding configuration
    results = await seeder.seed_all(
        users_count=10,
        properties_count=20,
        interactions_count=50,
        queries_count=25,
        clear_existing=True
    )
    
    print("\n" + "="*50)
    print("PRODUCTION DATA SEEDING RESULTS")
    print("="*50)
    print(f"Success: {results['success']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    print("\nRecord Counts:")
    for table, count in results.get('results', {}).items():
        print(f"  {table}: {count}")
    
    if results.get('errors'):
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())