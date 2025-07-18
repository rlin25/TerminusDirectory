"""
Production Data Seeding System
High-performance data generation and seeding for large-scale testing and development
"""

import asyncio
import logging
import random
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import asyncpg
from asyncpg import Connection
from faker import Faker
import numpy as np


class SeedingMode(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PERFORMANCE_TESTING = "performance_testing"
    LOAD_TESTING = "load_testing"


class DataQuality(Enum):
    MINIMAL = "minimal"
    BASIC = "basic"
    REALISTIC = "realistic"
    HIGH_FIDELITY = "high_fidelity"


@dataclass
class SeedingConfig:
    mode: SeedingMode
    data_quality: DataQuality
    batch_size: int
    max_workers: int
    validate_integrity: bool
    create_relationships: bool
    include_historical_data: bool
    data_distribution: Dict[str, Any]
    

@dataclass
class SeedingProgress:
    table_name: str
    total_records: int
    completed_records: int
    failed_records: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    current_batch: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
            
    @property
    def progress_percent(self) -> float:
        if self.total_records == 0:
            return 100.0
        return (self.completed_records / self.total_records) * 100


class ProductionDataSeeder:
    """
    Enterprise-grade data seeding system with:
    - Realistic data generation at scale
    - Referential integrity maintenance
    - Performance optimization for large datasets
    - Data quality validation
    - Incremental and parallel seeding
    """
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
        self.faker = Faker(['en_US'])
        
        # Seeding configurations for different environments
        self.seeding_configs = {
            SeedingMode.DEVELOPMENT: SeedingConfig(
                mode=SeedingMode.DEVELOPMENT,
                data_quality=DataQuality.BASIC,
                batch_size=1000,
                max_workers=2,
                validate_integrity=True,
                create_relationships=True,
                include_historical_data=False,
                data_distribution={
                    'users': 1000,
                    'properties': 5000,
                    'user_interactions': 25000,
                    'search_queries': 10000,
                    'ml_models': 10,
                    'training_metrics': 100
                }
            ),
            SeedingMode.STAGING: SeedingConfig(
                mode=SeedingMode.STAGING,
                data_quality=DataQuality.REALISTIC,
                batch_size=5000,
                max_workers=4,
                validate_integrity=True,
                create_relationships=True,
                include_historical_data=True,
                data_distribution={
                    'users': 10000,
                    'properties': 50000,
                    'user_interactions': 500000,
                    'search_queries': 100000,
                    'ml_models': 50,
                    'training_metrics': 1000
                }
            ),
            SeedingMode.PERFORMANCE_TESTING: SeedingConfig(
                mode=SeedingMode.PERFORMANCE_TESTING,
                data_quality=DataQuality.REALISTIC,
                batch_size=10000,
                max_workers=8,
                validate_integrity=False,
                create_relationships=True,
                include_historical_data=True,
                data_distribution={
                    'users': 100000,
                    'properties': 1000000,
                    'user_interactions': 10000000,
                    'search_queries': 2000000,
                    'ml_models': 100,
                    'training_metrics': 5000
                }
            ),
            SeedingMode.LOAD_TESTING: SeedingConfig(
                mode=SeedingMode.LOAD_TESTING,
                data_quality=DataQuality.MINIMAL,
                batch_size=50000,
                max_workers=16,
                validate_integrity=False,
                create_relationships=False,
                include_historical_data=True,
                data_distribution={
                    'users': 1000000,
                    'properties': 10000000,
                    'user_interactions': 100000000,
                    'search_queries': 20000000,
                    'ml_models': 200,
                    'training_metrics': 10000
                }
            )
        }
        
        # Data generation patterns
        self.property_types = ['apartment', 'house', 'condo', 'townhouse', 'studio', 'loft']
        self.interaction_types = ['view', 'like', 'inquiry', 'save', 'contact', 'favorite']
        self.amenities_pool = [
            'parking', 'pool', 'gym', 'balcony', 'dishwasher', 'laundry', 'ac', 'heating',
            'hardwood', 'carpet', 'tile', 'granite', 'stainless', 'elevator', 'doorman',
            'pets', 'furnished', 'utilities', 'internet', 'cable', 'storage', 'garden',
            'fireplace', 'walk_in_closet', 'in_unit_laundry', 'rooftop', 'concierge'
        ]
        
        # Location data (simplified - in production, use real geographic data)
        self.cities = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
            'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
            'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis',
            'Seattle', 'Denver', 'Washington DC', 'Boston', 'Nashville', 'Baltimore',
            'Oklahoma City', 'Louisville', 'Portland', 'Las Vegas', 'Milwaukee',
            'Albuquerque', 'Tucson', 'Fresno', 'Sacramento', 'Kansas City', 'Mesa'
        ]
        
        # Progress tracking
        self.seeding_progress: Dict[str, SeedingProgress] = {}
        
    async def seed_database(self, mode: SeedingMode, 
                          tables: Optional[List[str]] = None,
                          incremental: bool = False) -> Dict[str, Any]:
        """
        Seed database with realistic data for specified mode
        
        Args:
            mode: Seeding mode (development, staging, performance_testing, load_testing)
            tables: Specific tables to seed (None for all tables)
            incremental: Whether to add to existing data or replace
        """
        if mode not in self.seeding_configs:
            raise ValueError(f"Unknown seeding mode: {mode}")
            
        config = self.seeding_configs[mode]
        
        # Default tables in dependency order
        default_tables = ['users', 'properties', 'ml_models', 'user_interactions', 
                         'search_queries', 'training_metrics', 'embeddings', 'audit_log']
        
        tables_to_seed = tables if tables else default_tables
        
        # Initialize progress tracking
        for table in tables_to_seed:
            if table in config.data_distribution:
                self.seeding_progress[table] = SeedingProgress(
                    table_name=table,
                    total_records=config.data_distribution[table],
                    completed_records=0,
                    failed_records=0,
                    start_time=datetime.now()
                )
                
        start_time = time.time()
        results = {
            'mode': mode.value,
            'started_at': datetime.now().isoformat(),
            'tables_seeded': {},
            'total_records_created': 0,
            'errors': [],
            'performance_metrics': {}
        }
        
        try:
            self.logger.info(f"Starting database seeding in {mode.value} mode")
            
            # Clear existing data if not incremental
            if not incremental:
                await self._clear_existing_data(tables_to_seed)
                
            # Seed tables in dependency order
            for table in tables_to_seed:
                if table not in config.data_distribution:
                    continue
                    
                table_start_time = time.time()
                
                try:
                    self.logger.info(f"Seeding table: {table}")
                    records_created = await self._seed_table(table, config)
                    
                    table_duration = time.time() - table_start_time
                    results['tables_seeded'][table] = {
                        'records_created': records_created,
                        'duration_seconds': round(table_duration, 2),
                        'records_per_second': round(records_created / table_duration, 2) if table_duration > 0 else 0
                    }
                    results['total_records_created'] += records_created
                    
                    self.logger.info(f"Completed seeding {table}: {records_created} records in {table_duration:.2f}s")
                    
                except Exception as e:
                    error_msg = f"Failed to seed table {table}: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
            # Validate data integrity if configured
            if config.validate_integrity:
                validation_results = await self._validate_data_integrity(tables_to_seed)
                results['validation'] = validation_results
                
            # Create relationships if configured
            if config.create_relationships:
                relationship_results = await self._create_additional_relationships(config)
                results['relationships'] = relationship_results
                
            total_duration = time.time() - start_time
            results['total_duration_seconds'] = round(total_duration, 2)
            results['completed_at'] = datetime.now().isoformat()
            
            # Performance metrics
            results['performance_metrics'] = {
                'total_records_per_second': round(results['total_records_created'] / total_duration, 2) if total_duration > 0 else 0,
                'avg_batch_processing_time': self._calculate_avg_batch_time(),
                'memory_usage_peak_mb': self._get_peak_memory_usage(),
                'database_size_after_mb': await self._get_database_size()
            }
            
            self.logger.info(f"Database seeding completed: {results['total_records_created']} records in {total_duration:.2f}s")
            
        except Exception as e:
            error_msg = f"Database seeding failed: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
            
        return results
        
    async def _clear_existing_data(self, tables: List[str]):
        """Clear existing data from specified tables"""
        async with self.connection_manager.get_connection() as conn:
            # Disable triggers and constraints temporarily for faster deletion
            await conn.execute("SET session_replication_role = replica")
            
            try:
                # Delete in reverse dependency order
                for table in reversed(tables):
                    if table in ['users', 'properties', 'user_interactions', 'search_queries', 
                               'ml_models', 'training_metrics', 'embeddings', 'audit_log']:
                        await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                        self.logger.info(f"Cleared table: {table}")
                        
            finally:
                # Re-enable triggers and constraints
                await conn.execute("SET session_replication_role = DEFAULT")
                
    async def _seed_table(self, table_name: str, config: SeedingConfig) -> int:
        """Seed a specific table with data"""
        total_records = config.data_distribution[table_name]
        batch_size = config.batch_size
        
        if table_name == 'users':
            return await self._seed_users(total_records, batch_size, config)
        elif table_name == 'properties':
            return await self._seed_properties(total_records, batch_size, config)
        elif table_name == 'user_interactions':
            return await self._seed_user_interactions(total_records, batch_size, config)
        elif table_name == 'search_queries':
            return await self._seed_search_queries(total_records, batch_size, config)
        elif table_name == 'ml_models':
            return await self._seed_ml_models(total_records, batch_size, config)
        elif table_name == 'training_metrics':
            return await self._seed_training_metrics(total_records, batch_size, config)
        elif table_name == 'embeddings':
            return await self._seed_embeddings(total_records, batch_size, config)
        elif table_name == 'audit_log':
            return await self._seed_audit_log(total_records, batch_size, config)
        else:
            self.logger.warning(f"Unknown table for seeding: {table_name}")
            return 0
            
    async def _seed_users(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed users table"""
        records_created = 0
        progress = self.seeding_progress.get('users')
        
        async with self.connection_manager.get_connection() as conn:
            for batch_start in range(0, total_records, batch_size):
                batch_end = min(batch_start + batch_size, total_records)
                batch_records = []
                
                for i in range(batch_start, batch_end):
                    user_data = self._generate_user_data(config.data_quality)
                    batch_records.append(user_data)
                    
                # Insert batch
                try:
                    await conn.executemany("""
                        INSERT INTO users (
                            id, email, created_at, updated_at, status,
                            min_price, max_price, min_bedrooms, max_bedrooms,
                            min_bathrooms, max_bathrooms, preferred_locations,
                            required_amenities, property_types, last_login,
                            login_count, preference_updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                    """, batch_records)
                    
                    batch_count = len(batch_records)
                    records_created += batch_count
                    
                    if progress:
                        progress.completed_records = records_created
                        progress.current_batch += 1
                        
                    self.logger.debug(f"Inserted {batch_count} users (total: {records_created})")
                    
                except Exception as e:
                    if progress:
                        progress.failed_records += len(batch_records)
                        progress.errors.append(f"Batch {progress.current_batch}: {e}")
                    self.logger.error(f"Failed to insert user batch: {e}")
                    
        return records_created
        
    def _generate_user_data(self, quality: DataQuality) -> Tuple:
        """Generate realistic user data"""
        user_id = str(uuid.uuid4())
        email = self.faker.email()
        created_at = self.faker.date_time_between(start_date='-2y', end_date='now')
        updated_at = created_at
        status = random.choice(['active', 'inactive'])
        
        # Price preferences
        base_price = random.randint(500, 5000)
        min_price = base_price
        max_price = base_price + random.randint(500, 2000)
        
        # Bedroom/bathroom preferences
        min_bedrooms = random.randint(0, 3)
        max_bedrooms = min_bedrooms + random.randint(0, 3)
        min_bathrooms = random.choice([1.0, 1.5, 2.0])
        max_bathrooms = min_bathrooms + random.choice([0, 0.5, 1.0, 1.5])
        
        # Location preferences
        preferred_locations = random.sample(self.cities, random.randint(1, 5))
        
        # Amenity preferences
        required_amenities = random.sample(self.amenities_pool, random.randint(0, 8))
        
        # Property type preferences
        property_types = random.sample(self.property_types, random.randint(1, 4))
        
        # Activity data
        last_login = self.faker.date_time_between(start_date='-30d', end_date='now') if status == 'active' else None
        login_count = random.randint(1, 100) if status == 'active' else 0
        preference_updated_at = self.faker.date_time_between(start_date=created_at, end_date='now')
        
        return (
            user_id, email, created_at, updated_at, status,
            min_price, max_price, min_bedrooms, max_bedrooms,
            min_bathrooms, max_bathrooms, preferred_locations,
            required_amenities, property_types, last_login,
            login_count, preference_updated_at
        )
        
    async def _seed_properties(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed properties table"""
        records_created = 0
        progress = self.seeding_progress.get('properties')
        
        async with self.connection_manager.get_connection() as conn:
            for batch_start in range(0, total_records, batch_size):
                batch_end = min(batch_start + batch_size, total_records)
                batch_records = []
                
                for i in range(batch_start, batch_end):
                    property_data = self._generate_property_data(config.data_quality)
                    batch_records.append(property_data)
                    
                # Insert batch
                try:
                    await conn.executemany("""
                        INSERT INTO properties (
                            id, title, description, price, location, bedrooms, bathrooms,
                            square_feet, amenities, contact_info, images, scraped_at,
                            status, property_type, created_at, updated_at, latitude,
                            longitude, slug, external_id, external_url, data_quality_score,
                            last_verified
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
                    """, batch_records)
                    
                    batch_count = len(batch_records)
                    records_created += batch_count
                    
                    if progress:
                        progress.completed_records = records_created
                        progress.current_batch += 1
                        
                    self.logger.debug(f"Inserted {batch_count} properties (total: {records_created})")
                    
                except Exception as e:
                    if progress:
                        progress.failed_records += len(batch_records)
                        progress.errors.append(f"Batch {progress.current_batch}: {e}")
                    self.logger.error(f"Failed to insert property batch: {e}")
                    
        return records_created
        
    def _generate_property_data(self, quality: DataQuality) -> Tuple:
        """Generate realistic property data"""
        property_id = str(uuid.uuid4())
        
        # Property basics
        property_type = random.choice(self.property_types)
        bedrooms = random.randint(0, 5)
        bathrooms = random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        square_feet = random.randint(400, 3000)
        
        # Location
        city = random.choice(self.cities)
        location = f"{self.faker.street_address()}, {city}, {self.faker.state_abbr()} {self.faker.zipcode()}"
        
        # Coordinates (simplified - use real geocoding in production)
        latitude = round(self.faker.latitude(), 6)
        longitude = round(self.faker.longitude(), 6)
        
        # Pricing based on size and location
        base_price_per_sqft = random.uniform(1.5, 8.0)
        price = round(square_feet * base_price_per_sqft, 2)
        
        # Title and description
        title = f"{bedrooms} bed, {bathrooms} bath {property_type} in {city}"
        
        if quality == DataQuality.HIGH_FIDELITY:
            description = self._generate_detailed_property_description(property_type, bedrooms, bathrooms, location)
        else:
            description = f"Beautiful {property_type} with {bedrooms} bedrooms and {bathrooms} bathrooms."
            
        # Amenities
        num_amenities = random.randint(3, 12)
        amenities = random.sample(self.amenities_pool, num_amenities)
        
        # Contact info
        contact_info = {
            'phone': self.faker.phone_number(),
            'email': self.faker.email(),
            'agent_name': self.faker.name()
        }
        
        # Images (URLs)
        num_images = random.randint(1, 8)
        images = [f"https://example.com/images/{property_id}_{i}.jpg" for i in range(num_images)]
        
        # Metadata
        scraped_at = self.faker.date_time_between(start_date='-6m', end_date='now')
        status = random.choices(['active', 'inactive', 'rented'], weights=[70, 20, 10])[0]
        created_at = scraped_at
        updated_at = scraped_at
        
        # External data
        slug = title.lower().replace(' ', '-').replace(',', '')
        external_id = f"ext_{random.randint(100000, 999999)}"
        external_url = f"https://apartments.com/property/{external_id}"
        
        # Quality score
        data_quality_score = random.uniform(0.7, 1.0)
        last_verified = self.faker.date_time_between(start_date=scraped_at, end_date='now')
        
        return (
            property_id, title, description, price, location, bedrooms, bathrooms,
            square_feet, amenities, contact_info, images, scraped_at,
            status, property_type, created_at, updated_at, latitude,
            longitude, slug, external_id, external_url, data_quality_score,
            last_verified
        )
        
    def _generate_detailed_property_description(self, property_type: str, bedrooms: int, 
                                              bathrooms: float, location: str) -> str:
        """Generate detailed property description"""
        descriptions = [
            f"Stunning {property_type} featuring {bedrooms} spacious bedrooms and {bathrooms} beautifully appointed bathrooms.",
            f"This gorgeous {property_type} offers modern living with {bedrooms} bedrooms and {bathrooms} bathrooms.",
            f"Exceptional {property_type} in prime location with {bedrooms} bedrooms and {bathrooms} bathrooms.",
            f"Luxurious {property_type} boasting {bedrooms} comfortable bedrooms and {bathrooms} elegant bathrooms."
        ]
        
        features = [
            "Updated kitchen with stainless steel appliances",
            "Hardwood floors throughout",
            "In-unit washer and dryer",
            "Central air conditioning",
            "Walk-in closets",
            "Private balcony with city views",
            "Granite countertops",
            "Tile bathrooms with modern fixtures"
        ]
        
        base_description = random.choice(descriptions)
        selected_features = random.sample(features, random.randint(2, 5))
        feature_text = " Features include: " + ", ".join(selected_features) + "."
        
        return base_description + feature_text
        
    async def _seed_user_interactions(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed user interactions table"""
        # Get existing user and property IDs
        async with self.connection_manager.get_connection(analytics=True) as conn:
            user_ids = [row['id'] for row in await conn.fetch("SELECT id FROM users LIMIT 10000")]
            property_ids = [row['id'] for row in await conn.fetch("SELECT id FROM properties LIMIT 50000")]
            
        if not user_ids or not property_ids:
            self.logger.warning("No users or properties found for creating interactions")
            return 0
            
        records_created = 0
        progress = self.seeding_progress.get('user_interactions')
        
        async with self.connection_manager.get_connection() as conn:
            for batch_start in range(0, total_records, batch_size):
                batch_end = min(batch_start + batch_size, total_records)
                batch_records = []
                
                for i in range(batch_start, batch_end):
                    interaction_data = self._generate_interaction_data(user_ids, property_ids)
                    batch_records.append(interaction_data)
                    
                # Insert batch
                try:
                    await conn.executemany("""
                        INSERT INTO user_interactions (
                            id, user_id, property_id, interaction_type, timestamp,
                            duration_seconds, session_id, user_agent, ip_address,
                            referrer, interaction_strength
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, batch_records)
                    
                    batch_count = len(batch_records)
                    records_created += batch_count
                    
                    if progress:
                        progress.completed_records = records_created
                        progress.current_batch += 1
                        
                    self.logger.debug(f"Inserted {batch_count} interactions (total: {records_created})")
                    
                except Exception as e:
                    if progress:
                        progress.failed_records += len(batch_records)
                        progress.errors.append(f"Batch {progress.current_batch}: {e}")
                    self.logger.error(f"Failed to insert interaction batch: {e}")
                    
        return records_created
        
    def _generate_interaction_data(self, user_ids: List[str], property_ids: List[str]) -> Tuple:
        """Generate realistic user interaction data"""
        interaction_id = str(uuid.uuid4())
        user_id = random.choice(user_ids)
        property_id = random.choice(property_ids)
        
        # Interaction type with realistic distribution
        interaction_type = random.choices(
            self.interaction_types,
            weights=[50, 15, 5, 20, 3, 7]  # views are most common
        )[0]
        
        # Timestamp within last 6 months
        timestamp = self.faker.date_time_between(start_date='-6m', end_date='now')
        
        # Duration based on interaction type
        if interaction_type == 'view':
            duration_seconds = random.randint(5, 300)  # 5 seconds to 5 minutes
        elif interaction_type == 'inquiry':
            duration_seconds = random.randint(60, 600)  # 1-10 minutes
        else:
            duration_seconds = random.randint(10, 120)  # 10 seconds to 2 minutes
            
        # Session data
        session_id = str(uuid.uuid4())
        user_agent = self.faker.user_agent()
        ip_address = self.faker.ipv4()
        referrer = random.choice([
            'https://google.com/search',
            'https://apartments.com',
            'https://zillow.com',
            'direct',
            None
        ])
        
        # Interaction strength based on type
        strength_mapping = {
            'view': 0.1,
            'like': 0.3,
            'save': 0.5,
            'contact': 0.8,
            'inquiry': 0.9,
            'favorite': 0.6
        }
        interaction_strength = strength_mapping.get(interaction_type, 0.1)
        
        return (
            interaction_id, user_id, property_id, interaction_type, timestamp,
            duration_seconds, session_id, user_agent, ip_address,
            referrer, interaction_strength
        )
        
    async def _seed_search_queries(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed search queries table"""
        # Get existing user IDs
        async with self.connection_manager.get_connection(analytics=True) as conn:
            user_ids = [row['id'] for row in await conn.fetch("SELECT id FROM users LIMIT 10000")]
            
        if not user_ids:
            self.logger.warning("No users found for creating search queries")
            return 0
            
        records_created = 0
        progress = self.seeding_progress.get('search_queries')
        
        async with self.connection_manager.get_connection() as conn:
            for batch_start in range(0, total_records, batch_size):
                batch_end = min(batch_start + batch_size, total_records)
                batch_records = []
                
                for i in range(batch_start, batch_end):
                    query_data = self._generate_search_query_data(user_ids)
                    batch_records.append(query_data)
                    
                # Insert batch
                try:
                    await conn.executemany("""
                        INSERT INTO search_queries (
                            id, user_id, query_text, filters, results_count,
                            execution_time_ms, created_at, session_id, page_number,
                            page_size, sort_by
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, batch_records)
                    
                    batch_count = len(batch_records)
                    records_created += batch_count
                    
                    if progress:
                        progress.completed_records = records_created
                        progress.current_batch += 1
                        
                    self.logger.debug(f"Inserted {batch_count} search queries (total: {records_created})")
                    
                except Exception as e:
                    if progress:
                        progress.failed_records += len(batch_records)
                        progress.errors.append(f"Batch {progress.current_batch}: {e}")
                    self.logger.error(f"Failed to insert search query batch: {e}")
                    
        return records_created
        
    def _generate_search_query_data(self, user_ids: List[str]) -> Tuple:
        """Generate realistic search query data"""
        query_id = str(uuid.uuid4())
        user_id = random.choice(user_ids) if random.random() > 0.3 else None  # 30% anonymous
        
        # Generate realistic search queries
        query_patterns = [
            f"{random.randint(1, 4)} bedroom apartment",
            f"apartments in {random.choice(self.cities)}",
            f"under ${random.randint(1000, 5000)} rent",
            f"{random.choice(self.property_types)} with {random.choice(self.amenities_pool)}",
            f"pet friendly {random.choice(self.property_types)}",
            f"{random.choice(self.cities)} {random.choice(self.property_types)}",
            "luxury apartment downtown",
            "cheap studio apartment",
            "furnished apartment short term"
        ]
        
        query_text = random.choice(query_patterns)
        
        # Search filters
        filters = {}
        if random.random() > 0.5:
            filters['min_price'] = random.randint(500, 2000)
            filters['max_price'] = filters['min_price'] + random.randint(500, 2000)
        if random.random() > 0.6:
            filters['bedrooms'] = random.randint(1, 4)
        if random.random() > 0.7:
            filters['property_type'] = random.choice(self.property_types)
        if random.random() > 0.8:
            filters['amenities'] = random.sample(self.amenities_pool, random.randint(1, 3))
            
        # Search results
        results_count = random.randint(0, 500)
        execution_time_ms = random.randint(50, 2000)
        
        # Metadata
        created_at = self.faker.date_time_between(start_date='-3m', end_date='now')
        session_id = str(uuid.uuid4())
        page_number = random.randint(1, 5)
        page_size = random.choice([10, 20, 50])
        sort_by = random.choice(['price_asc', 'price_desc', 'newest', 'relevance'])
        
        return (
            query_id, user_id, query_text, filters, results_count,
            execution_time_ms, created_at, session_id, page_number,
            page_size, sort_by
        )
        
    async def _seed_ml_models(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed ML models table"""
        records_created = 0
        
        model_types = [
            'collaborative_filter', 'content_recommender', 'hybrid_recommender',
            'search_ranker', 'price_predictor', 'demand_forecaster'
        ]
        
        async with self.connection_manager.get_connection() as conn:
            for i in range(total_records):
                model_data = self._generate_ml_model_data(model_types)
                
                try:
                    await conn.execute("""
                        INSERT INTO ml_models (
                            id, model_name, version, model_file_path, metadata,
                            created_at, is_active, training_accuracy, validation_accuracy,
                            training_time_seconds, model_size_bytes, parent_model_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """, *model_data)
                    
                    records_created += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to insert ML model: {e}")
                    
        return records_created
        
    def _generate_ml_model_data(self, model_types: List[str]) -> Tuple:
        """Generate ML model data"""
        model_id = str(uuid.uuid4())
        model_name = random.choice(model_types)
        version = f"v{random.randint(1, 10)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        model_file_path = f"/models/{model_name}/{version}/model.pkl"
        
        metadata = {
            'algorithm': random.choice(['xgboost', 'neural_network', 'random_forest', 'svm']),
            'features': random.randint(10, 100),
            'hyperparameters': {
                'learning_rate': round(random.uniform(0.001, 0.1), 4),
                'max_depth': random.randint(3, 10),
                'n_estimators': random.randint(50, 500)
            }
        }
        
        created_at = self.faker.date_time_between(start_date='-1y', end_date='now')
        is_active = random.choice([True, False])
        training_accuracy = round(random.uniform(0.7, 0.95), 4)
        validation_accuracy = round(training_accuracy - random.uniform(0.01, 0.05), 4)
        training_time_seconds = random.randint(300, 7200)  # 5 minutes to 2 hours
        model_size_bytes = random.randint(1024*1024, 100*1024*1024)  # 1MB to 100MB
        parent_model_id = None  # Could reference previous version
        
        return (
            model_id, model_name, version, model_file_path, metadata,
            created_at, is_active, training_accuracy, validation_accuracy,
            training_time_seconds, model_size_bytes, parent_model_id
        )
        
    async def _seed_training_metrics(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed training metrics table"""
        # Get existing model names
        async with self.connection_manager.get_connection(analytics=True) as conn:
            models = await conn.fetch("SELECT model_name, version FROM ml_models")
            
        if not models:
            self.logger.warning("No ML models found for creating training metrics")
            return 0
            
        records_created = 0
        
        async with self.connection_manager.get_connection() as conn:
            for i in range(total_records):
                model = random.choice(models)
                metrics_data = self._generate_training_metrics_data(model['model_name'], model['version'])
                
                try:
                    await conn.execute("""
                        INSERT INTO training_metrics (
                            id, model_name, version, metrics, training_date,
                            job_id, training_duration_seconds, dataset_size,
                            hyperparameters, cpu_usage_percent, memory_usage_mb,
                            gpu_usage_percent
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """, *metrics_data)
                    
                    records_created += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to insert training metrics: {e}")
                    
        return records_created
        
    def _generate_training_metrics_data(self, model_name: str, version: str) -> Tuple:
        """Generate training metrics data"""
        metrics_id = str(uuid.uuid4())
        
        # Training metrics
        metrics = {
            'accuracy': round(random.uniform(0.7, 0.95), 4),
            'precision': round(random.uniform(0.65, 0.92), 4),
            'recall': round(random.uniform(0.68, 0.90), 4),
            'f1_score': round(random.uniform(0.66, 0.91), 4),
            'rmse': round(random.uniform(0.1, 0.5), 4),
            'mae': round(random.uniform(0.05, 0.3), 4),
            'loss': round(random.uniform(0.1, 1.0), 4)
        }
        
        training_date = self.faker.date_time_between(start_date='-6m', end_date='now')
        job_id = f"job_{random.randint(100000, 999999)}"
        training_duration_seconds = random.randint(600, 14400)  # 10 minutes to 4 hours
        dataset_size = random.randint(10000, 1000000)
        
        hyperparameters = {
            'learning_rate': round(random.uniform(0.001, 0.1), 4),
            'batch_size': random.choice([32, 64, 128, 256]),
            'epochs': random.randint(10, 100),
            'dropout': round(random.uniform(0.1, 0.5), 2)
        }
        
        # Resource usage
        cpu_usage_percent = round(random.uniform(60, 95), 2)
        memory_usage_mb = random.randint(1024, 8192)
        gpu_usage_percent = round(random.uniform(70, 100), 2) if random.random() > 0.3 else None
        
        return (
            metrics_id, model_name, version, metrics, training_date,
            job_id, training_duration_seconds, dataset_size,
            hyperparameters, cpu_usage_percent, memory_usage_mb,
            gpu_usage_percent
        )
        
    async def _seed_embeddings(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed embeddings table (simplified)"""
        # This is a simplified implementation - in production, you'd generate actual embeddings
        records_created = 0
        
        async with self.connection_manager.get_connection() as conn:
            # Get some entity IDs
            user_ids = [row['id'] for row in await conn.fetch("SELECT id FROM users LIMIT 1000")]
            property_ids = [row['id'] for row in await conn.fetch("SELECT id FROM properties LIMIT 5000")]
            
            entities = [(uid, 'user') for uid in user_ids] + [(pid, 'property') for pid in property_ids]
            
            for entity_id, entity_type in entities[:total_records]:
                embedding_data = self._generate_embedding_data(entity_id, entity_type)
                
                try:
                    await conn.execute("""
                        INSERT INTO embeddings (
                            id, entity_type, entity_id, embeddings, dimension,
                            model_version, created_at, updated_at, norm
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """, *embedding_data)
                    
                    records_created += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to insert embedding: {e}")
                    
        return records_created
        
    def _generate_embedding_data(self, entity_id: str, entity_type: str) -> Tuple:
        """Generate embedding data"""
        embedding_id = str(uuid.uuid4())
        
        # Generate random embedding vector (in production, use actual model)
        dimension = 128
        embedding_vector = np.random.rand(dimension).astype(np.float32)
        embeddings = embedding_vector.tobytes()  # Store as bytes
        
        model_version = "v1.0.0"
        created_at = self.faker.date_time_between(start_date='-3m', end_date='now')
        updated_at = created_at
        norm = float(np.linalg.norm(embedding_vector))
        
        return (
            embedding_id, entity_type, entity_id, embeddings, dimension,
            model_version, created_at, updated_at, norm
        )
        
    async def _seed_audit_log(self, total_records: int, batch_size: int, config: SeedingConfig) -> int:
        """Seed audit log table"""
        records_created = 0
        
        tables = ['users', 'properties', 'user_interactions', 'ml_models']
        operations = ['INSERT', 'UPDATE', 'DELETE']
        
        async with self.connection_manager.get_connection() as conn:
            for i in range(total_records):
                audit_data = self._generate_audit_log_data(tables, operations)
                
                try:
                    await conn.execute("""
                        INSERT INTO audit_log (
                            id, table_name, operation, row_id, old_values,
                            new_values, changed_by, changed_at, session_id,
                            ip_address, user_agent
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, *audit_data)
                    
                    records_created += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to insert audit log: {e}")
                    
        return records_created
        
    def _generate_audit_log_data(self, tables: List[str], operations: List[str]) -> Tuple:
        """Generate audit log data"""
        audit_id = str(uuid.uuid4())
        table_name = random.choice(tables)
        operation = random.choice(operations)
        row_id = str(uuid.uuid4())
        
        # Sample change data
        if operation == 'INSERT':
            old_values = None
            new_values = {'status': 'active', 'created_at': datetime.now().isoformat()}
        elif operation == 'UPDATE':
            old_values = {'status': 'inactive', 'updated_at': datetime.now().isoformat()}
            new_values = {'status': 'active', 'updated_at': datetime.now().isoformat()}
        else:  # DELETE
            old_values = {'status': 'active', 'deleted_at': datetime.now().isoformat()}
            new_values = None
            
        changed_by = str(uuid.uuid4())  # User ID
        changed_at = self.faker.date_time_between(start_date='-6m', end_date='now')
        session_id = str(uuid.uuid4())
        ip_address = self.faker.ipv4()
        user_agent = self.faker.user_agent()
        
        return (
            audit_id, table_name, operation, row_id, old_values,
            new_values, changed_by, changed_at, session_id,
            ip_address, user_agent
        )
        
    async def _validate_data_integrity(self, tables: List[str]) -> Dict[str, Any]:
        """Validate data integrity after seeding"""
        validation_results = {
            'passed': True,
            'table_counts': {},
            'integrity_checks': {},
            'errors': []
        }
        
        async with self.connection_manager.get_connection(analytics=True) as conn:
            # Check record counts
            for table in tables:
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    validation_results['table_counts'][table] = count
                except Exception as e:
                    validation_results['errors'].append(f"Failed to count {table}: {e}")
                    validation_results['passed'] = False
                    
            # Check foreign key integrity
            integrity_checks = [
                ("user_interactions", "SELECT COUNT(*) FROM user_interactions WHERE user_id NOT IN (SELECT id FROM users)"),
                ("user_interactions", "SELECT COUNT(*) FROM user_interactions WHERE property_id NOT IN (SELECT id FROM properties)"),
                ("search_queries", "SELECT COUNT(*) FROM search_queries WHERE user_id IS NOT NULL AND user_id NOT IN (SELECT id FROM users)")
            ]
            
            for table, check_sql in integrity_checks:
                try:
                    orphaned_count = await conn.fetchval(check_sql)
                    validation_results['integrity_checks'][f"{table}_orphaned"] = orphaned_count
                    
                    if orphaned_count > 0:
                        validation_results['passed'] = False
                        validation_results['errors'].append(f"Found {orphaned_count} orphaned records in {table}")
                        
                except Exception as e:
                    validation_results['errors'].append(f"Failed integrity check for {table}: {e}")
                    validation_results['passed'] = False
                    
        return validation_results
        
    async def _create_additional_relationships(self, config: SeedingConfig) -> Dict[str, Any]:
        """Create additional relationships between seeded data"""
        results = {
            'relationships_created': 0,
            'errors': []
        }
        
        # This could include creating more realistic user-property interaction patterns,
        # ensuring geographical consistency, etc.
        # For now, return empty results
        
        return results
        
    def _calculate_avg_batch_time(self) -> float:
        """Calculate average batch processing time"""
        # This would track actual batch times during seeding
        return 0.0
        
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage during seeding"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)  # MB
        except:
            return 0.0
            
    async def _get_database_size(self) -> float:
        """Get current database size in MB"""
        try:
            async with self.connection_manager.get_connection(analytics=True) as conn:
                size_bytes = await conn.fetchval("SELECT pg_database_size(current_database())")
                return size_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
            
    async def get_seeding_progress(self) -> Dict[str, Any]:
        """Get current seeding progress"""
        progress_data = {}
        
        for table_name, progress in self.seeding_progress.items():
            progress_data[table_name] = {
                'total_records': progress.total_records,
                'completed_records': progress.completed_records,
                'failed_records': progress.failed_records,
                'progress_percent': progress.progress_percent,
                'current_batch': progress.current_batch,
                'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None,
                'errors': progress.errors
            }
            
        return progress_data