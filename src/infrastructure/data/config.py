import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncpg
import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    @property
    def url(self) -> str:
        """Get database URL for SQLAlchemy"""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def asyncpg_url(self) -> str:
        """Get database URL for asyncpg"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    host: str
    port: int
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    health_check_interval: int = 30
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        else:
            return f"redis://{self.host}:{self.port}/{self.db}"


class DataConfig:
    """Main data configuration class"""
    
    def __init__(self):
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables"""
        return DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "rental_ml"),
            username=os.getenv("DB_USERNAME", "postgres"),
            password=os.getenv("DB_PASSWORD", "password"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600"))
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment variables"""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            socket_connect_timeout=int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")),
            health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
        )


class DatabaseManager:
    """Database connection and initialization manager"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self) -> asyncpg.Pool:
        """Initialize database connection pool"""
        try:
            self._pool = await asyncpg.create_pool(
                self.config.asyncpg_url,
                min_size=5,
                max_size=self.config.pool_size,
                command_timeout=self.config.pool_timeout
            )
            
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            logger.info("Database connection pool initialized successfully")
            return self._pool
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            # Connect to default postgres database
            default_url = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/postgres"
            
            conn = await asyncpg.connect(default_url)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self.config.database
            )
            
            if not exists:
                # Create database
                await conn.execute(f'CREATE DATABASE "{self.config.database}"')
                logger.info(f"Created database: {self.config.database}")
            else:
                logger.info(f"Database already exists: {self.config.database}")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    async def create_tables(self):
        """Create all required tables"""
        try:
            # Import all models to ensure they're registered
            from .repositories.postgres_property_repository import Base as PropertyBase
            from .repositories.postgres_model_repository import Base as ModelBase
            
            # You might need to create a combined Base or use alembic migrations
            # For now, we'll create tables manually
            
            async with self._pool.acquire() as conn:
                # Create users table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        email VARCHAR(255) NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE,
                        min_price DECIMAL,
                        max_price DECIMAL,
                        min_bedrooms INTEGER,
                        max_bedrooms INTEGER,
                        min_bathrooms DECIMAL,
                        max_bathrooms DECIMAL,
                        preferred_locations TEXT[],
                        required_amenities TEXT[],
                        property_types TEXT[]
                    )
                """)
                
                # Create user_interactions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_interactions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID NOT NULL REFERENCES users(id),
                        property_id UUID NOT NULL,
                        interaction_type VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        duration_seconds INTEGER
                    )
                """)
                
                # Create properties table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS properties (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        title VARCHAR(255) NOT NULL,
                        description TEXT NOT NULL,
                        price DECIMAL NOT NULL,
                        location VARCHAR(255) NOT NULL,
                        bedrooms INTEGER NOT NULL,
                        bathrooms DECIMAL NOT NULL,
                        square_feet INTEGER,
                        amenities TEXT[],
                        contact_info JSONB,
                        images TEXT[],
                        scraped_at TIMESTAMP NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        property_type VARCHAR(50) DEFAULT 'apartment',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        full_text_search TEXT,
                        price_per_sqft DECIMAL
                    )
                """)
                
                # Create ml_models table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        model_name VARCHAR(255) NOT NULL,
                        version VARCHAR(50) NOT NULL,
                        model_data BYTEA NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Create embeddings table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        entity_type VARCHAR(50) NOT NULL,
                        entity_id VARCHAR(255) NOT NULL,
                        embeddings BYTEA NOT NULL,
                        dimension INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create training_metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_metrics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        model_name VARCHAR(255) NOT NULL,
                        version VARCHAR(50) NOT NULL,
                        metrics JSONB NOT NULL,
                        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_property_id ON user_interactions(property_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_active ON properties(is_active)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_location ON properties(location)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(price)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_bedrooms ON properties(bedrooms)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_type ON properties(property_type)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_ml_models_name_version ON ml_models(model_name, version)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_metrics_model ON training_metrics(model_name, version)")
                
                # Create GIN indexes for array and full-text search
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_amenities_gin ON properties USING GIN(amenities)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_properties_fulltext_gin ON properties USING GIN(to_tsvector('english', full_text_search))")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_locations_gin ON users USING GIN(preferred_locations)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_amenities_gin ON users USING GIN(required_amenities)")
                
            logger.info("Database tables and indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed")
    
    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        """Get the database pool"""
        return self._pool


class RedisManager:
    """Redis connection manager"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._client: Optional[Redis] = None
    
    async def initialize(self) -> Redis:
        """Initialize Redis client"""
        try:
            self._client = redis.from_url(
                self.config.url,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            # Test connection
            await self._client.ping()
            
            logger.info("Redis client initialized successfully")
            return self._client
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()
            logger.info("Redis client closed")
    
    @property
    def client(self) -> Optional[Redis]:
        """Get the Redis client"""
        return self._client


class DataManagerFactory:
    """Factory for creating data managers"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.db_manager: Optional[DatabaseManager] = None
        self.redis_manager: Optional[RedisManager] = None
    
    async def initialize_all(self):
        """Initialize all data managers"""
        # Initialize database
        self.db_manager = DatabaseManager(self.config.database)
        await self.db_manager.create_database_if_not_exists()
        await self.db_manager.initialize()
        await self.db_manager.create_tables()
        
        # Initialize Redis
        self.redis_manager = RedisManager(self.config.redis)
        await self.redis_manager.initialize()
        
        logger.info("All data managers initialized successfully")
    
    async def close_all(self):
        """Close all connections"""
        if self.db_manager:
            await self.db_manager.close()
        
        if self.redis_manager:
            await self.redis_manager.close()
        
        logger.info("All data managers closed")
    
    def get_database_pool(self) -> asyncpg.Pool:
        """Get database connection pool"""
        if not self.db_manager or not self.db_manager.pool:
            raise RuntimeError("Database manager not initialized")
        return self.db_manager.pool
    
    def get_redis_client(self) -> Redis:
        """Get Redis client"""
        if not self.redis_manager or not self.redis_manager.client:
            raise RuntimeError("Redis manager not initialized")
        return self.redis_manager.client