"""
Production database connector for real-time property data ingestion.

This module provides comprehensive database integration for scraped property data
with real-time ingestion, duplicate detection, and data archival capabilities.
"""

import asyncio
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
import asyncpg
from asyncpg import Connection, Pool
import redis
from redis import Redis

from .config import get_config, ProductionScrapingConfig
from ...domain.entities.property import Property

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of property ingestion"""
    success: bool
    property_id: str
    operation: str  # 'created', 'updated', 'skipped', 'failed'
    duplicate_detected: bool = False
    duplicate_of: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class BatchIngestionResult:
    """Result of batch property ingestion"""
    total_processed: int
    created: int
    updated: int
    skipped: int
    failed: int
    duplicates_detected: int
    processing_time_ms: float
    errors: List[str] = field(default_factory=list)


class ProductionDatabaseConnector:
    """Production database connector with real-time ingestion capabilities"""
    
    def __init__(self, config: ProductionScrapingConfig = None):
        self.config = config or get_config()
        self.connection_pool: Optional[Pool] = None
        self.redis_client: Optional[Redis] = None
        
        # Connection settings
        self.db_config = {
            'host': self._get_env_var('DB_HOST', 'localhost'),
            'port': int(self._get_env_var('DB_PORT', '5432')),
            'database': self._get_env_var('DB_NAME', 'rental_ml'),
            'user': self._get_env_var('DB_USER', 'postgres'),
            'password': self._get_env_var('DB_PASSWORD', 'password'),
            'min_size': 5,
            'max_size': self.config.database.connection_pool_size,
            'command_timeout': self.config.database.connection_timeout
        }
        
        # Redis configuration for caching
        if self.config.cache.redis_url:
            self.redis_config = {
                'url': self.config.cache.redis_url,
                'decode_responses': True,
                'socket_connect_timeout': 5,
                'socket_timeout': 5
            }
        
        # Table names
        self.properties_table = 'properties'
        self.property_history_table = 'property_history'
        self.scraping_jobs_table = 'scraping_jobs'
        self.property_duplicates_table = 'property_duplicates'
        
        logger.info("Initialized production database connector")
    
    def _get_env_var(self, key: str, default: str) -> str:
        """Get environment variable with fallback"""
        import os
        return os.getenv(key, default)
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize PostgreSQL connection pool
            self.connection_pool = await asyncpg.create_pool(**self.db_config)
            logger.info("Connected to PostgreSQL database")
            
            # Initialize Redis if configured
            if hasattr(self, 'redis_config'):
                self.redis_client = redis.from_url(
                    self.redis_config['url'],
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test Redis connection
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.ping
                )
                logger.info("Connected to Redis cache")
            
            # Ensure database schema exists
            await self._ensure_schema()
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _ensure_schema(self):
        """Ensure database schema exists"""
        
        schema_sql = """
        -- Properties table with enhanced fields
        CREATE TABLE IF NOT EXISTS properties (
            id UUID PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            price DECIMAL(10,2) NOT NULL,
            location VARCHAR(500) NOT NULL,
            bedrooms INTEGER DEFAULT 0,
            bathrooms DECIMAL(3,1) DEFAULT 0,
            square_feet INTEGER,
            property_type VARCHAR(50) DEFAULT 'apartment',
            amenities JSONB DEFAULT '[]',
            contact_info JSONB DEFAULT '{}',
            images JSONB DEFAULT '[]',
            source_name VARCHAR(50) NOT NULL,
            source_url TEXT,
            source_id VARCHAR(200),
            
            -- Geographic information
            latitude DECIMAL(10,8),
            longitude DECIMAL(11,8),
            formatted_address VARCHAR(500),
            
            -- Data quality fields
            data_quality_score DECIMAL(3,2) DEFAULT 0.0,
            validation_issues JSONB DEFAULT '[]',
            validation_warnings JSONB DEFAULT '[]',
            
            -- Metadata
            is_active BOOLEAN DEFAULT true,
            scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            
            -- Indexing
            CONSTRAINT unique_source_property UNIQUE(source_name, source_id)
        );
        
        -- Property history for tracking changes
        CREATE TABLE IF NOT EXISTS property_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            property_id UUID REFERENCES properties(id),
            change_type VARCHAR(20) NOT NULL, -- 'created', 'updated', 'deactivated'
            old_data JSONB,
            new_data JSONB,
            changed_fields JSONB DEFAULT '[]',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Scraping jobs tracking
        CREATE TABLE IF NOT EXISTS scraping_jobs (
            id UUID PRIMARY KEY,
            source_name VARCHAR(50) NOT NULL,
            job_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            properties_found INTEGER DEFAULT 0,
            properties_valid INTEGER DEFAULT 0,
            properties_stored INTEGER DEFAULT 0,
            error_message TEXT,
            metadata JSONB DEFAULT '{}'
        );
        
        -- Property duplicates tracking
        CREATE TABLE IF NOT EXISTS property_duplicates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            property_id_1 UUID REFERENCES properties(id),
            property_id_2 UUID REFERENCES properties(id),
            similarity_score DECIMAL(3,2) NOT NULL,
            detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            verified BOOLEAN DEFAULT false,
            
            CONSTRAINT unique_duplicate_pair UNIQUE(property_id_1, property_id_2)
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_properties_source ON properties(source_name);
        CREATE INDEX IF NOT EXISTS idx_properties_location ON properties USING GIN(to_tsvector('english', location));
        CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price);
        CREATE INDEX IF NOT EXISTS idx_properties_bedrooms ON properties(bedrooms);
        CREATE INDEX IF NOT EXISTS idx_properties_scraped_at ON properties(scraped_at);
        CREATE INDEX IF NOT EXISTS idx_properties_geo ON properties(latitude, longitude);
        CREATE INDEX IF NOT EXISTS idx_properties_active ON properties(is_active);
        
        -- Full-text search index
        CREATE INDEX IF NOT EXISTS idx_properties_search ON properties 
        USING GIN(to_tsvector('english', title || ' ' || COALESCE(description, '')));
        
        -- Function to update updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        -- Trigger to automatically update updated_at
        DROP TRIGGER IF EXISTS update_properties_updated_at ON properties;
        CREATE TRIGGER update_properties_updated_at
            BEFORE UPDATE ON properties
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """
        
        async with self.connection_pool.acquire() as connection:
            await connection.execute(schema_sql)
        
        logger.info("Database schema ensured")
    
    async def create_property(
        self, 
        property_data: Property,
        check_duplicates: bool = True
    ) -> IngestionResult:
        """Create a new property with duplicate detection"""
        
        start_time = datetime.now()
        
        try:
            # Generate source ID if not present
            if not hasattr(property_data, 'source_id') or not property_data.source_id:
                property_data.source_id = self._generate_source_id(property_data)
            
            # Check for duplicates if enabled
            duplicate_info = None
            if check_duplicates and self.config.database.duplicate_check_enabled:
                duplicate_info = await self._check_duplicates(property_data)
                if duplicate_info:
                    # Record duplicate
                    await self._record_duplicate(property_data.id, duplicate_info['property_id'], duplicate_info['similarity'])
                    
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    return IngestionResult(
                        success=True,
                        property_id=str(property_data.id),
                        operation='skipped',
                        duplicate_detected=True,
                        duplicate_of=duplicate_info['property_id'],
                        processing_time_ms=processing_time
                    )
            
            # Insert property
            async with self.connection_pool.acquire() as connection:
                
                # Check if property already exists by source
                existing = await connection.fetchrow(
                    "SELECT id FROM properties WHERE source_name = $1 AND source_id = $2",
                    property_data.source_name, property_data.source_id
                )
                
                if existing:
                    # Update existing property
                    result = await self._update_property(connection, property_data, existing['id'])
                    operation = 'updated'
                else:
                    # Create new property
                    await self._insert_property(connection, property_data)
                    operation = 'created'
                    
                    # Record creation in history
                    await self._record_history(
                        connection, 
                        property_data.id, 
                        'created', 
                        None, 
                        property_data.__dict__
                    )
            
            # Cache property if Redis is available
            if self.redis_client:
                await self._cache_property(property_data)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IngestionResult(
                success=True,
                property_id=str(property_data.id),
                operation=operation,
                duplicate_detected=bool(duplicate_info),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error creating property {property_data.id}: {e}")
            
            return IngestionResult(
                success=False,
                property_id=str(property_data.id),
                operation='failed',
                error_message=str(e),
                processing_time_ms=processing_time
            )
    
    async def create_properties_batch(
        self, 
        properties: List[Property],
        check_duplicates: bool = True
    ) -> BatchIngestionResult:
        """Batch create properties for improved performance"""
        
        start_time = datetime.now()
        
        results = {
            'created': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'duplicates_detected': 0
        }
        errors = []
        
        # Process in batches
        batch_size = self.config.database.batch_size
        
        for i in range(0, len(properties), batch_size):
            batch = properties[i:i + batch_size]
            
            try:
                async with self.connection_pool.acquire() as connection:
                    async with connection.transaction():
                        
                        for property_data in batch:
                            try:
                                result = await self.create_property(property_data, check_duplicates)
                                
                                results[result.operation] += 1
                                if result.duplicate_detected:
                                    results['duplicates_detected'] += 1
                                
                                if not result.success:
                                    errors.append(result.error_message)
                                    
                            except Exception as e:
                                results['failed'] += 1
                                errors.append(f"Property {property_data.id}: {str(e)}")
                                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                results['failed'] += len(batch)
                errors.append(f"Batch error: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchIngestionResult(
            total_processed=len(properties),
            created=results['created'],
            updated=results['updated'],
            skipped=results['skipped'],
            failed=results['failed'],
            duplicates_detected=results['duplicates_detected'],
            processing_time_ms=processing_time,
            errors=errors
        )
    
    async def _insert_property(self, connection: Connection, property_data: Property):
        """Insert property into database"""
        
        sql = """
        INSERT INTO properties (
            id, title, description, price, location, bedrooms, bathrooms,
            square_feet, property_type, amenities, contact_info, images,
            source_name, source_url, source_id, latitude, longitude,
            formatted_address, data_quality_score, validation_issues,
            validation_warnings, is_active, scraped_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
            $16, $17, $18, $19, $20, $21, $22, $23
        )
        """
        
        await connection.execute(
            sql,
            property_data.id,
            property_data.title,
            property_data.description,
            property_data.price,
            property_data.location,
            property_data.bedrooms,
            property_data.bathrooms,
            property_data.square_feet,
            property_data.property_type,
            json.dumps(property_data.amenities),
            json.dumps(property_data.contact_info),
            json.dumps(property_data.images),
            property_data.source_name,
            getattr(property_data, 'source_url', None),
            getattr(property_data, 'source_id', None),
            getattr(property_data, 'latitude', None),
            getattr(property_data, 'longitude', None),
            getattr(property_data, 'formatted_address', None),
            getattr(property_data, 'data_quality_score', 0.0),
            json.dumps(getattr(property_data, 'validation_issues', [])),
            json.dumps(getattr(property_data, 'validation_warnings', [])),
            property_data.is_active,
            property_data.scraped_at
        )
    
    async def _update_property(
        self, 
        connection: Connection, 
        property_data: Property, 
        existing_id: str
    ) -> bool:
        """Update existing property"""
        
        # Get current data for history
        current_data = await connection.fetchrow(
            "SELECT * FROM properties WHERE id = $1", existing_id
        )
        
        # Update property
        sql = """
        UPDATE properties SET
            title = $2, description = $3, price = $4, location = $5,
            bedrooms = $6, bathrooms = $7, square_feet = $8,
            property_type = $9, amenities = $10, contact_info = $11,
            images = $12, latitude = $13, longitude = $14,
            formatted_address = $15, data_quality_score = $16,
            validation_issues = $17, validation_warnings = $18,
            is_active = $19, scraped_at = $20
        WHERE id = $1
        """
        
        await connection.execute(
            sql,
            existing_id,
            property_data.title,
            property_data.description,
            property_data.price,
            property_data.location,
            property_data.bedrooms,
            property_data.bathrooms,
            property_data.square_feet,
            property_data.property_type,
            json.dumps(property_data.amenities),
            json.dumps(property_data.contact_info),
            json.dumps(property_data.images),
            getattr(property_data, 'latitude', None),
            getattr(property_data, 'longitude', None),
            getattr(property_data, 'formatted_address', None),
            getattr(property_data, 'data_quality_score', 0.0),
            json.dumps(getattr(property_data, 'validation_issues', [])),
            json.dumps(getattr(property_data, 'validation_warnings', [])),
            property_data.is_active,
            property_data.scraped_at
        )
        
        # Record update in history
        await self._record_history(
            connection,
            existing_id,
            'updated',
            dict(current_data),
            property_data.__dict__
        )
        
        return True
    
    async def _check_duplicates(self, property_data: Property) -> Optional[Dict[str, Any]]:
        """Check for duplicate properties"""
        
        async with self.connection_pool.acquire() as connection:
            
            # Simple duplicate check based on title and location similarity
            similar_properties = await connection.fetch("""
                SELECT id, title, location, price
                FROM properties 
                WHERE source_name != $1 
                AND is_active = true
                AND similarity(title, $2) > 0.7
                AND similarity(location, $3) > 0.8
                AND ABS(price - $4) / $4 < 0.2
                LIMIT 5
            """, property_data.source_name, property_data.title, property_data.location, property_data.price)
            
            if similar_properties:
                # Return the most similar one
                best_match = similar_properties[0]
                return {
                    'property_id': str(best_match['id']),
                    'similarity': 0.8  # Simplified similarity score
                }
        
        return None
    
    async def _record_duplicate(self, property_id: str, duplicate_id: str, similarity: float):
        """Record duplicate relationship"""
        
        async with self.connection_pool.acquire() as connection:
            try:
                await connection.execute("""
                    INSERT INTO property_duplicates (property_id_1, property_id_2, similarity_score)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (property_id_1, property_id_2) DO NOTHING
                """, property_id, duplicate_id, similarity)
            except Exception as e:
                logger.warning(f"Error recording duplicate: {e}")
    
    async def _record_history(
        self,
        connection: Connection,
        property_id: str,
        change_type: str,
        old_data: Optional[Dict],
        new_data: Dict
    ):
        """Record property change history"""
        
        try:
            await connection.execute("""
                INSERT INTO property_history (property_id, change_type, old_data, new_data)
                VALUES ($1, $2, $3, $4)
            """, property_id, change_type, json.dumps(old_data), json.dumps(new_data))
        except Exception as e:
            logger.warning(f"Error recording history: {e}")
    
    async def _cache_property(self, property_data: Property):
        """Cache property in Redis"""
        
        if not self.redis_client:
            return
        
        try:
            cache_key = f"property:{property_data.source_name}:{property_data.source_id}"
            cache_data = {
                'id': str(property_data.id),
                'title': property_data.title,
                'price': property_data.price,
                'location': property_data.location,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.setex,
                cache_key,
                self.config.cache.cache_ttl_seconds,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.warning(f"Error caching property: {e}")
    
    def _generate_source_id(self, property_data: Property) -> str:
        """Generate unique source ID for property"""
        
        # Create hash from title, location, and price
        content = f"{property_data.title}|{property_data.location}|{property_data.price}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_property_stats(self) -> Dict[str, Any]:
        """Get property statistics"""
        
        async with self.connection_pool.acquire() as connection:
            
            # Basic stats
            total_properties = await connection.fetchval("SELECT COUNT(*) FROM properties WHERE is_active = true")
            
            # Stats by source
            source_stats = await connection.fetch("""
                SELECT source_name, COUNT(*) as count, AVG(price) as avg_price
                FROM properties 
                WHERE is_active = true
                GROUP BY source_name
                ORDER BY count DESC
            """)
            
            # Recent activity
            recent_properties = await connection.fetchval("""
                SELECT COUNT(*) FROM properties 
                WHERE scraped_at > $1 AND is_active = true
            """, datetime.utcnow() - timedelta(hours=24))
            
            # Duplicates
            total_duplicates = await connection.fetchval("SELECT COUNT(*) FROM property_duplicates")
            
            return {
                'total_properties': total_properties,
                'recent_properties_24h': recent_properties,
                'total_duplicates': total_duplicates,
                'source_stats': [dict(row) for row in source_stats]
            }
    
    async def cleanup_old_data(self, retention_days: int = None):
        """Clean up old property data"""
        
        if retention_days is None:
            retention_days = self.config.database.data_retention_days
        
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        async with self.connection_pool.acquire() as connection:
            
            # Archive old properties
            archived_count = await connection.fetchval("""
                UPDATE properties 
                SET is_active = false 
                WHERE scraped_at < $1 AND is_active = true
                RETURNING id
            """, cutoff_date)
            
            # Clean up old history records
            history_deleted = await connection.fetchval("""
                DELETE FROM property_history 
                WHERE created_at < $1
                RETURNING id
            """, cutoff_date)
            
            logger.info(f"Archived {archived_count} old properties and deleted {history_deleted} history records")
    
    async def close(self):
        """Close database connections"""
        
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Closed PostgreSQL connection pool")
        
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.close
            )
            logger.info("Closed Redis connection")