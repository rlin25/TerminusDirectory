import logging
from typing import Optional
from redis.asyncio import Redis
import asyncpg

from .config import DataConfig, DataManagerFactory
from .repositories.postgres_user_repository import PostgreSQLUserRepository
from .repositories.postgres_property_repository import PostgresPropertyRepository
from .repositories.postgres_model_repository import PostgresModelRepository
from .repositories.redis_cache_repository import RedisCacheRepository

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """Factory for creating and managing repository instances"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.data_manager: Optional[DataManagerFactory] = None
        
        # Repository instances
        self._user_repository: Optional[PostgreSQLUserRepository] = None
        self._property_repository: Optional[PostgresPropertyRepository] = None
        self._model_repository: Optional[PostgresModelRepository] = None
        self._cache_repository: Optional[RedisCacheRepository] = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all data connections and repositories"""
        if self._initialized:
            logger.warning("Repository factory already initialized")
            return
        
        try:
            # Initialize data managers
            self.data_manager = DataManagerFactory(self.config)
            await self.data_manager.initialize_all()
            
            # Create repository instances
            await self._create_repositories()
            
            self._initialized = True
            logger.info("Repository factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize repository factory: {e}")
            raise
    
    async def _create_repositories(self):
        """Create repository instances"""
        if not self.data_manager:
            raise RuntimeError("Data manager not initialized")
        
        # Get database pool and Redis client
        db_pool = self.data_manager.get_database_pool()
        redis_client = self.data_manager.get_redis_client()
        
        # Create repositories
        self._user_repository = PostgreSQLUserRepository(db_pool)
        self._property_repository = PostgresPropertyRepository(
            self.config.database.url,
            pool_size=self.config.database.pool_size,
            max_overflow=self.config.database.max_overflow
        )
        self._model_repository = PostgresModelRepository(
            self.config.database.url,
            pool_size=self.config.database.pool_size,
            max_overflow=self.config.database.max_overflow
        )
        self._cache_repository = RedisCacheRepository(
            redis_client,
            default_ttl=3600  # 1 hour default TTL
        )
        
        logger.info("All repositories created successfully")
    
    async def close(self):
        """Close all connections and cleanup"""
        try:
            # Close individual repositories if they have cleanup methods
            if self._property_repository:
                await self._property_repository.close()
            
            if self._model_repository:
                await self._model_repository.close()
            
            # Close data managers
            if self.data_manager:
                await self.data_manager.close_all()
            
            self._initialized = False
            logger.info("Repository factory closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing repository factory: {e}")
            raise
    
    def get_user_repository(self) -> PostgreSQLUserRepository:
        """Get user repository instance"""
        if not self._initialized or not self._user_repository:
            raise RuntimeError("Repository factory not initialized or user repository not available")
        return self._user_repository
    
    def get_property_repository(self) -> PostgresPropertyRepository:
        """Get property repository instance"""
        if not self._initialized or not self._property_repository:
            raise RuntimeError("Repository factory not initialized or property repository not available")
        return self._property_repository
    
    def get_model_repository(self) -> PostgresModelRepository:
        """Get model repository instance"""
        if not self._initialized or not self._model_repository:
            raise RuntimeError("Repository factory not initialized or model repository not available")
        return self._model_repository
    
    def get_cache_repository(self) -> RedisCacheRepository:
        """Get cache repository instance"""
        if not self._initialized or not self._cache_repository:
            raise RuntimeError("Repository factory not initialized or cache repository not available")
        return self._cache_repository
    
    async def health_check(self) -> dict:
        """Perform health check on all repositories"""
        health_status = {
            "database": False,
            "redis": False,
            "repositories": False,
            "overall": False
        }
        
        try:
            # Check database health
            if self.data_manager and self.data_manager.db_manager:
                db_pool = self.data_manager.db_manager.pool
                if db_pool:
                    async with db_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    health_status["database"] = True
            
            # Check Redis health
            if self._cache_repository:
                health_status["redis"] = await self._cache_repository.health_check()
            
            # Check if all repositories are available
            health_status["repositories"] = all([
                self._user_repository is not None,
                self._property_repository is not None,
                self._model_repository is not None,
                self._cache_repository is not None
            ])
            
            # Overall health
            health_status["overall"] = all([
                health_status["database"],
                health_status["redis"],
                health_status["repositories"]
            ])
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)
        
        return health_status
    
    def is_initialized(self) -> bool:
        """Check if factory is initialized"""
        return self._initialized


# Global repository factory instance
_repository_factory: Optional[RepositoryFactory] = None


async def get_repository_factory(config: Optional[DataConfig] = None) -> RepositoryFactory:
    """Get or create the global repository factory instance"""
    global _repository_factory
    
    if _repository_factory is None:
        _repository_factory = RepositoryFactory(config)
        await _repository_factory.initialize()
    
    return _repository_factory


async def close_repository_factory():
    """Close the global repository factory instance"""
    global _repository_factory
    
    if _repository_factory:
        await _repository_factory.close()
        _repository_factory = None


class RepositoryManager:
    """Context manager for repository lifecycle"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config
        self.factory: Optional[RepositoryFactory] = None
    
    async def __aenter__(self) -> RepositoryFactory:
        """Initialize repositories when entering context"""
        self.factory = RepositoryFactory(self.config)
        await self.factory.initialize()
        return self.factory
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close repositories when exiting context"""
        if self.factory:
            await self.factory.close()


# Convenience functions for getting repositories
async def get_user_repository(config: Optional[DataConfig] = None) -> PostgreSQLUserRepository:
    """Get user repository instance"""
    factory = await get_repository_factory(config)
    return factory.get_user_repository()


async def get_property_repository(config: Optional[DataConfig] = None) -> PostgresPropertyRepository:
    """Get property repository instance"""
    factory = await get_repository_factory(config)
    return factory.get_property_repository()


async def get_model_repository(config: Optional[DataConfig] = None) -> PostgresModelRepository:
    """Get model repository instance"""
    factory = await get_repository_factory(config)
    return factory.get_model_repository()


async def get_cache_repository(config: Optional[DataConfig] = None) -> RedisCacheRepository:
    """Get cache repository instance"""
    factory = await get_repository_factory(config)
    return factory.get_cache_repository()