#!/usr/bin/env python3
"""
Database Setup and Testing Script

This script sets up the database schema and tests connections for the rental ML system.
It provides a simple way to initialize the database before running the data ingestion pipeline.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime
from pathlib import Path
import asyncpg
from asyncpg import Pool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseSetup:
    """Database setup and testing utilities"""
    
    def __init__(self):
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'rental_ml'),
            'user': os.getenv('DB_USER', 'rental_ml_user'),
            'password': os.getenv('DB_PASSWORD', ''),
        }
        self.pool = None
        
    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            logger.info("Testing database connection...")
            
            # Try to connect
            conn = await asyncpg.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            # Test query
            result = await conn.fetchval('SELECT version()')
            await conn.close()
            
            logger.info(f"✅ Database connection successful: {result[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
            
    async def create_connection_pool(self) -> bool:
        """Create connection pool"""
        try:
            logger.info("Creating connection pool...")
            
            self.pool = await asyncpg.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                min_size=2,
                max_size=10
            )
            
            # Test the pool
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
                
            logger.info("✅ Connection pool created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            return False
            
    async def check_schema(self) -> dict:
        """Check if required tables exist"""
        logger.info("Checking database schema...")
        
        required_tables = [
            'users', 'properties', 'user_interactions', 
            'search_queries', 'ml_models', 'training_metrics',
            'embeddings', 'audit_log'
        ]
        
        schema_status = {}
        
        try:
            async with self.pool.acquire() as conn:
                for table in required_tables:
                    try:
                        exists = await conn.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_schema = 'public' 
                                AND table_name = $1
                            )
                        """, table)
                        
                        if exists:
                            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                            schema_status[table] = {'exists': True, 'count': count}
                        else:
                            schema_status[table] = {'exists': False, 'count': 0}
                            
                    except Exception as e:
                        schema_status[table] = {'exists': False, 'error': str(e)}
                        
            # Summary
            existing_tables = sum(1 for status in schema_status.values() if status.get('exists'))
            logger.info(f"Schema check: {existing_tables}/{len(required_tables)} tables exist")
            
            return schema_status
            
        except Exception as e:
            logger.error(f"❌ Schema check failed: {e}")
            return {}
            
    async def run_migrations(self) -> bool:
        """Run database migrations"""
        logger.info("Running database migrations...")
        
        try:
            migrations_dir = Path(__file__).parent / "migrations"
            
            if not migrations_dir.exists():
                logger.warning("Migrations directory not found")
                return await self.create_basic_schema()
                
            # Get migration files
            migration_files = sorted([
                f for f in migrations_dir.glob("*.sql") 
                if not f.name.endswith("_rollback.sql")
            ])
            
            if not migration_files:
                logger.warning("No migration files found")
                return await self.create_basic_schema()
                
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for migration_file in migration_files:
                        logger.info(f"Running migration: {migration_file.name}")
                        
                        with open(migration_file, 'r') as f:
                            migration_sql = f.read()
                            
                        await conn.execute(migration_sql)
                        
            logger.info("✅ Migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
            
    async def create_basic_schema(self) -> bool:
        """Create basic schema if migrations are not available"""
        logger.info("Creating basic database schema...")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Users table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            email VARCHAR(255) UNIQUE NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status VARCHAR(50) DEFAULT 'active',
                            min_price DECIMAL(10,2),
                            max_price DECIMAL(10,2),
                            min_bedrooms INTEGER,
                            max_bedrooms INTEGER,
                            min_bathrooms DECIMAL(3,1),
                            max_bathrooms DECIMAL(3,1),
                            preferred_locations TEXT[],
                            required_amenities TEXT[],
                            property_types TEXT[],
                            last_login TIMESTAMP,
                            login_count INTEGER DEFAULT 0,
                            preference_updated_at TIMESTAMP
                        )
                    """)
                    
                    # Properties table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS properties (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            title VARCHAR(500) NOT NULL,
                            description TEXT,
                            price DECIMAL(10,2) NOT NULL,
                            location TEXT NOT NULL,
                            bedrooms INTEGER,
                            bathrooms DECIMAL(3,1),
                            square_feet INTEGER,
                            amenities TEXT[],
                            contact_info JSONB,
                            images TEXT[],
                            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status VARCHAR(50) DEFAULT 'active',
                            property_type VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            latitude DECIMAL(10,8),
                            longitude DECIMAL(11,8),
                            slug VARCHAR(500),
                            external_id VARCHAR(200),
                            external_url TEXT,
                            data_quality_score DECIMAL(3,2),
                            last_verified TIMESTAMP
                        )
                    """)
                    
                    # User interactions table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS user_interactions (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                            property_id UUID REFERENCES properties(id) ON DELETE CASCADE,
                            interaction_type VARCHAR(50) NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            duration_seconds INTEGER,
                            session_id UUID,
                            user_agent TEXT,
                            ip_address INET,
                            referrer TEXT,
                            interaction_strength DECIMAL(3,2)
                        )
                    """)
                    
                    # Search queries table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS search_queries (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                            query_text TEXT,
                            filters JSONB,
                            results_count INTEGER,
                            execution_time_ms INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            session_id UUID,
                            page_number INTEGER DEFAULT 1,
                            page_size INTEGER DEFAULT 20,
                            sort_by VARCHAR(100)
                        )
                    """)
                    
                    # ML models table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS ml_models (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            model_name VARCHAR(200) NOT NULL,
                            version VARCHAR(50) NOT NULL,
                            model_file_path TEXT,
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            is_active BOOLEAN DEFAULT FALSE,
                            training_accuracy DECIMAL(5,4),
                            validation_accuracy DECIMAL(5,4),
                            training_time_seconds INTEGER,
                            model_size_bytes BIGINT,
                            parent_model_id UUID REFERENCES ml_models(id)
                        )
                    """)
                    
                    # Training metrics table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS training_metrics (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            model_name VARCHAR(200) NOT NULL,
                            version VARCHAR(50) NOT NULL,
                            metrics JSONB,
                            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            job_id VARCHAR(200),
                            training_duration_seconds INTEGER,
                            dataset_size INTEGER,
                            hyperparameters JSONB,
                            cpu_usage_percent DECIMAL(5,2),
                            memory_usage_mb INTEGER,
                            gpu_usage_percent DECIMAL(5,2)
                        )
                    """)
                    
                    # Embeddings table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS embeddings (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            entity_type VARCHAR(50) NOT NULL,
                            entity_id UUID NOT NULL,
                            embeddings BYTEA NOT NULL,
                            dimension INTEGER NOT NULL,
                            model_version VARCHAR(50),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            norm DECIMAL(10,6)
                        )
                    """)
                    
                    # Audit log table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS audit_log (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            table_name VARCHAR(100) NOT NULL,
                            operation VARCHAR(20) NOT NULL,
                            row_id UUID NOT NULL,
                            old_values JSONB,
                            new_values JSONB,
                            changed_by UUID,
                            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            session_id UUID,
                            ip_address INET,
                            user_agent TEXT
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_location ON properties(location)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_bedrooms ON properties(bedrooms)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_status ON properties(status)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_property_id ON user_interactions(property_id)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp ON user_interactions(timestamp)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_search_queries_user_id ON search_queries(user_id)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id)")
                    
            logger.info("✅ Basic schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create basic schema: {e}")
            return False
            
    async def insert_test_data(self) -> bool:
        """Insert a small amount of test data"""
        logger.info("Inserting test data...")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Insert test user
                    user_id = await conn.fetchval("""
                        INSERT INTO users (email, min_price, max_price, min_bedrooms, max_bedrooms)
                        VALUES ('test@example.com', 1000, 3000, 1, 3)
                        RETURNING id
                    """)
                    
                    # Insert test property
                    property_id = await conn.fetchval("""
                        INSERT INTO properties (title, description, price, location, bedrooms, bathrooms, property_type)
                        VALUES ('Test Property', 'A nice test property', 2000, 'Downtown', 2, 1.5, 'apartment')
                        RETURNING id
                    """)
                    
                    # Insert test interaction
                    await conn.execute("""
                        INSERT INTO user_interactions (user_id, property_id, interaction_type)
                        VALUES ($1, $2, 'view')
                    """, user_id, property_id)
                    
            logger.info("✅ Test data inserted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert test data: {e}")
            return False
            
    async def cleanup_test_data(self) -> bool:
        """Clean up test data"""
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("DELETE FROM user_interactions WHERE user_id IN (SELECT id FROM users WHERE email = 'test@example.com')")
                    await conn.execute("DELETE FROM properties WHERE title = 'Test Property'")
                    await conn.execute("DELETE FROM users WHERE email = 'test@example.com'")
                    
            logger.info("✅ Test data cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup test data: {e}")
            return False
            
    async def get_database_stats(self) -> dict:
        """Get database statistics"""
        try:
            async with self.pool.acquire() as conn:
                # Get table sizes
                table_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        most_common_vals
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    ORDER BY tablename, attname
                """)
                
                # Get database size
                db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size(current_database()))")
                
                # Get connection info
                connection_info = await conn.fetchrow("""
                    SELECT 
                        current_database() as database,
                        current_user as user,
                        version() as version
                """)
                
                return {
                    "database_size": db_size,
                    "connection_info": dict(connection_info) if connection_info else {},
                    "table_count": len(set(row['tablename'] for row in table_stats)),
                    "stats_available": len(table_stats) > 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
            
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Connection pool closed")


async def main():
    """Main execution function"""
    print("🏠 Rental ML System - Database Setup")
    print("=" * 40)
    
    # Show configuration
    print("\n📋 Database Configuration:")
    print(f"Host: {os.getenv('DB_HOST', 'localhost')}")
    print(f"Port: {os.getenv('DB_PORT', '5432')}")
    print(f"Database: {os.getenv('DB_NAME', 'rental_ml')}")
    print(f"User: {os.getenv('DB_USER', 'rental_ml_user')}")
    print(f"Password: {'*' * len(os.getenv('DB_PASSWORD', ''))}")
    
    db_setup = DatabaseSetup()
    
    try:
        # Test connection
        if not await db_setup.test_connection():
            print("\n❌ Database connection failed. Please check your configuration.")
            print("\nMake sure to set these environment variables:")
            print("- DB_HOST (default: localhost)")
            print("- DB_PORT (default: 5432)")
            print("- DB_NAME (default: rental_ml)")
            print("- DB_USER (default: rental_ml_user)")
            print("- DB_PASSWORD (required)")
            return
            
        # Create connection pool
        if not await db_setup.create_connection_pool():
            print("\n❌ Failed to create connection pool")
            return
            
        # Check schema
        print("\n🔍 Checking database schema...")
        schema_status = await db_setup.check_schema()
        
        existing_tables = [name for name, status in schema_status.items() if status.get('exists')]
        missing_tables = [name for name, status in schema_status.items() if not status.get('exists')]
        
        if existing_tables:
            print(f"✅ Existing tables ({len(existing_tables)}):")
            for table in existing_tables:
                count = schema_status[table].get('count', 0)
                print(f"  - {table}: {count} records")
                
        if missing_tables:
            print(f"❌ Missing tables ({len(missing_tables)}):")
            for table in missing_tables:
                print(f"  - {table}")
                
            # Run migrations or create schema
            print("\n🔧 Setting up database schema...")
            if await db_setup.run_migrations():
                print("✅ Database schema setup completed")
            else:
                print("❌ Database schema setup failed")
                return
        else:
            print("✅ All required tables exist")
            
        # Insert and test with sample data
        print("\n🧪 Testing database operations...")
        if await db_setup.insert_test_data():
            print("✅ Test data operations successful")
            await db_setup.cleanup_test_data()
        else:
            print("❌ Test data operations failed")
            
        # Get database statistics
        print("\n📊 Database Statistics:")
        stats = await db_setup.get_database_stats()
        if "error" not in stats:
            print(f"Database size: {stats.get('database_size', 'Unknown')}")
            print(f"Tables: {stats.get('table_count', 0)}")
            print(f"PostgreSQL version: {stats.get('connection_info', {}).get('version', 'Unknown')[:50]}...")
        else:
            print(f"Could not retrieve stats: {stats['error']}")
            
        print("\n🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the data pipeline test: python test_data_pipeline.py")
        print("2. Set up data ingestion: python setup_data_ingestion.py")
        print("3. Start the application")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        
    finally:
        await db_setup.close()


if __name__ == "__main__":
    # Check if password is set
    if not os.getenv('DB_PASSWORD'):
        print("❌ DB_PASSWORD environment variable is required")
        print("\nExample usage:")
        print("export DB_PASSWORD='your_password'")
        print("python setup_database.py")
        sys.exit(1)
        
    asyncio.run(main())