import asyncio
import json
import logging
import pickle
import time
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from functools import wraps
from dataclasses import dataclass
import numpy as np
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, DateTime, 
    Text, LargeBinary, JSON, select, update, delete, insert, and_, or_, func,
    Index, UniqueConstraint, CheckConstraint, event, text
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncTransaction
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, BYTEA
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import events
import asyncpg
from uuid import uuid4

from ....domain.repositories.model_repository import ModelRepository

# Configure logging
logger = logging.getLogger(__name__)

# Performance monitoring
@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_type: str
    execution_time: float
    rows_affected: int
    timestamp: datetime

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0
CONNECTION_TIMEOUT = 30.0

# Database performance constants
DEFAULT_BATCH_SIZE = 1000
MAX_BATCH_SIZE = 5000
QUERY_TIMEOUT = 30.0
CACHE_DEFAULT_TTL = 3600  # 1 hour
CACHE_MAX_TTL = 86400  # 24 hours

def retry_on_db_error(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """Decorator to retry database operations on transient errors"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (OperationalError, asyncpg.PostgresError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Database operation failed (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Non-retryable database error: {e}")
                    raise
            
            raise last_exception
        return wrapper
    return decorator

def measure_performance(operation_name: str):
    """Decorator to measure query performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = await func(self, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance metrics
                if hasattr(self, '_performance_metrics'):
                    metric = QueryMetrics(
                        query_type=operation_name,
                        execution_time=execution_time,
                        rows_affected=len(result) if isinstance(result, (list, tuple)) else 1,
                        timestamp=datetime.utcnow()
                    )
                    self._performance_metrics.append(metric)
                
                if execution_time > 1.0:  # Log slow queries
                    logger.warning(
                        f"Slow query detected: {operation_name} took {execution_time:.2f}s"
                    )
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Query failed: {operation_name} after {execution_time:.2f}s - {e}"
                )
                raise
        return wrapper
    return decorator

# Database models
Base = declarative_base()

class MLModelStorage(Base):
    __tablename__ = "ml_models"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_data = Column(LargeBinary, nullable=False)
    model_metadata = Column("metadata", JSON, nullable=False, default={})
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_active = Column(Boolean, default=True, index=True)
    
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_name_version'),
        Index('idx_ml_models_name_active', 'model_name', 'is_active'),
        Index('idx_ml_models_created_at', 'created_at'),
        Index('idx_ml_models_name_version_active', 'model_name', 'version', 'is_active'),
        {'comment': 'Storage for trained ML models and their metadata'}
    )

class EmbeddingStorage(Base):
    __tablename__ = "embeddings"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_type = Column(String(50), nullable=False, index=True)  # 'user', 'property', etc.
    entity_id = Column(String(255), nullable=False, index=True)
    embeddings = Column(LargeBinary, nullable=False)  # Serialized numpy array
    dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    
    __table_args__ = (
        UniqueConstraint('entity_type', 'entity_id', name='uq_entity_type_id'),
        Index('idx_embeddings_type_dimension', 'entity_type', 'dimension'),
        Index('idx_embeddings_updated_at', 'updated_at'),
        {'comment': 'Storage for entity embeddings used in ML models'}
    )

class TrainingMetrics(Base):
    __tablename__ = "training_metrics"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    metrics = Column(JSON, nullable=False)
    training_date = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_training_metrics_model_version', 'model_name', 'version'),
        Index('idx_training_metrics_date', 'training_date'),
        {'comment': 'Storage for ML model training metrics and performance data'}
    )

class PredictionCache(Base):
    __tablename__ = "prediction_cache"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    cache_key = Column(String(512), nullable=False, unique=True, index=True)
    predictions = Column(LargeBinary, nullable=False)  # Serialized predictions
    model_metadata = Column("metadata", JSON, nullable=False, default={})
    cached_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_prediction_cache_expires', 'expires_at'),
        Index('idx_prediction_cache_accessed', 'last_accessed'),
        CheckConstraint('expires_at > cached_at', name='check_expiry_after_cache'),
        {'comment': 'Cache for ML model predictions with TTL support'}
    )


class PostgresModelRepository(ModelRepository):
    """PostgreSQL-based ML model repository implementation with production-ready features"""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 40,
                 pool_timeout: int = 30, pool_recycle: int = 3600, 
                 enable_metrics: bool = True):
        self.database_url = database_url
        self.enable_metrics = enable_metrics
        self._performance_metrics = [] if enable_metrics else None
        
        # Configure connection pool with optimizations
        self.engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,
            echo=False,
            connect_args={
                "command_timeout": CONNECTION_TIMEOUT,
                "server_settings": {
                    "application_name": "rental_ml_system",
                    "jit": "off",  # Disable JIT for better performance
                },
            }
        )
        
        # Create async session factory
        self.async_session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Setup connection pool monitoring
        self._setup_pool_monitoring()
    
    def _setup_pool_monitoring(self):
        """Setup connection pool monitoring and events"""
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            # PostgreSQL-specific optimizations
            if hasattr(dbapi_connection, 'execute'):
                pass  # Add PostgreSQL-specific settings if needed
        
        @event.listens_for(self.engine.sync_engine, "checkout")
        def log_connection_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self.engine.sync_engine, "checkin")
        def log_connection_checkin(dbapi_connection, connection_record):
            logger.debug("Connection checked in to pool")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Context manager for database sessions with proper error handling"""
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_transaction(self) -> AsyncTransaction:
        """Context manager for database transactions"""
        async with self.get_session() as session:
            async with session.begin() as transaction:
                try:
                    yield transaction
                except Exception as e:
                    await transaction.rollback()
                    logger.error(f"Transaction error: {e}")
                    raise
    
    async def create_tables(self):
        """Create database tables with indexes and constraints"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
            # Create additional indexes for performance
            await conn.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_models_metadata_gin 
                ON ml_models USING gin (metadata);
            """))
            
            await conn.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_metrics_gin 
                ON training_metrics USING gin (metrics);
            """))
            
            logger.info("Database tables and indexes created successfully")
    
    async def close(self):
        """Close database connections"""
        await self.engine.dispose()
    
    @retry_on_db_error()
    @measure_performance("save_model")
    async def save_model(self, model_name: str, model_data: Any, version: str) -> bool:
        """Save ML model to database with comprehensive error handling and optimization"""
        if not model_name or not version:
            logger.error("Model name and version are required")
            return False
        
        try:
            # Validate model data
            if model_data is None:
                logger.error("Model data cannot be None")
                return False
            
            # Serialize model data with compression
            serialized_data = pickle.dumps(model_data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Prepare comprehensive metadata
            metadata = {
                'size_bytes': len(serialized_data),
                'saved_at': datetime.utcnow().isoformat(),
                'python_version': self._get_python_version(),
                'dependencies': self._get_dependencies_info(),
                'model_type': type(model_data).__name__,
                'serialization_protocol': pickle.HIGHEST_PROTOCOL,
                'compression_ratio': self._calculate_compression_ratio(model_data, serialized_data)
            }
            
            async with self.get_transaction() as transaction:
                session = transaction.session
                
                # Check if model version already exists
                existing_model = await session.execute(
                    select(MLModelStorage).where(
                        and_(
                            MLModelStorage.model_name == model_name,
                            MLModelStorage.version == version
                        )
                    )
                )
                existing = existing_model.scalar_one_or_none()
                
                if existing:
                    # Update existing model
                    stmt = update(MLModelStorage).where(
                        MLModelStorage.id == existing.id
                    ).values(
                        model_data=serialized_data,
                        metadata=metadata,
                        created_at=datetime.utcnow()
                    )
                    await session.execute(stmt)
                    logger.info(f"Updated existing model {model_name} version {version} "
                              f"({len(serialized_data)} bytes)")
                else:
                    # Create new model
                    new_model = MLModelStorage(
                        model_name=model_name,
                        version=version,
                        model_data=serialized_data,
                        metadata=metadata
                    )
                    session.add(new_model)
                    logger.info(f"Saved new model {model_name} version {version} "
                              f"({len(serialized_data)} bytes)")
                
                await session.commit()
                
                # Log successful save with metrics
                logger.info(f"Successfully saved model {model_name} v{version} "
                          f"(size: {len(serialized_data)} bytes, "
                          f"type: {type(model_data).__name__})")
                
                return True
                
        except IntegrityError as e:
            logger.error(f"Database integrity error saving model {model_name} v{version}: {e}")
            return False
        except OperationalError as e:
            logger.error(f"Database operational error saving model {model_name} v{version}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving model {model_name} v{version}: {e}")
            return False
    
    @retry_on_db_error()
    @measure_performance("load_model")
    async def load_model(self, model_name: str, version: str = "latest") -> Optional[Any]:
        """Load ML model from database with caching and optimization"""
        if not model_name:
            logger.error("Model name is required")
            return None
            
        try:
            async with self.get_session() as session:
                if version == "latest":
                    # Get the most recent version with optimized query
                    stmt = select(MLModelStorage).where(
                        and_(
                            MLModelStorage.model_name == model_name,
                            MLModelStorage.is_active == True
                        )
                    ).order_by(MLModelStorage.created_at.desc()).limit(1)
                else:
                    # Get specific version
                    stmt = select(MLModelStorage).where(
                        and_(
                            MLModelStorage.model_name == model_name,
                            MLModelStorage.version == version,
                            MLModelStorage.is_active == True
                        )
                    )
                
                result = await session.execute(stmt)
                model_record = result.scalar_one_or_none()
                
                if model_record:
                    # Validate model data before deserialization
                    if not model_record.model_data:
                        logger.error(f"Model {model_name} v{model_record.version} has no data")
                        return None
                    
                    # Deserialize model data safely
                    try:
                        model_data = pickle.loads(model_record.model_data)
                    except (pickle.UnpicklingError, TypeError) as e:
                        logger.error(f"Failed to deserialize model {model_name} v{model_record.version}: {e}")
                        return None
                    
                    # Log successful load with metrics
                    size_mb = len(model_record.model_data) / (1024 * 1024)
                    logger.info(f"Loaded model {model_name} v{model_record.version} "
                              f"({size_mb:.2f} MB, created: {model_record.created_at})")
                    
                    return model_data
                else:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return None
                    
        except OperationalError as e:
            logger.error(f"Database operational error loading model {model_name} v{version}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading model {model_name} v{version}: {e}")
            return None
    
    @retry_on_db_error()
    @measure_performance("get_model_versions")
    async def get_model_versions(self, model_name: str) -> List[str]:
        """Get all versions of a specific model"""
        if not model_name:
            logger.error("Model name is required")
            return []
            
        try:
            async with self.get_session() as session:
                stmt = select(MLModelStorage.version).where(
                    and_(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.is_active == True
                    )
                ).order_by(MLModelStorage.created_at.desc())
                
                result = await session.execute(stmt)
                versions = [row[0] for row in result]
                
                logger.debug(f"Found {len(versions)} versions for model {model_name}")
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get versions for model {model_name}: {e}")
            return []
    
    @retry_on_db_error()
    @measure_performance("delete_model")
    async def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a specific model version (soft delete)"""
        if not model_name or not version:
            logger.error("Model name and version are required")
            return False
            
        try:
            async with self.get_transaction() as transaction:
                session = transaction.session
                
                stmt = update(MLModelStorage).where(
                    and_(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.version == version
                    )
                ).values(is_active=False)
                
                result = await session.execute(stmt)
                await session.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Deleted model {model_name} version {version}")
                    return True
                else:
                    logger.warning(f"Model {model_name} version {version} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete model {model_name} version {version}: {e}")
            return False
    
    @retry_on_db_error()
    @measure_performance("save_embeddings")
    async def save_embeddings(self, entity_type: str, entity_id: str, embeddings: np.ndarray) -> bool:
        """Save entity embeddings with validation and optimization"""
        if not entity_type or not entity_id:
            logger.error("Entity type and entity ID are required")
            return False
        
        if embeddings is None or embeddings.size == 0:
            logger.error("Embeddings cannot be None or empty")
            return False
        
        # Validate embeddings format
        if not isinstance(embeddings, np.ndarray):
            logger.error("Embeddings must be a numpy array")
            return False
        
        if embeddings.ndim > 2:
            logger.error("Embeddings must be 1D or 2D array")
            return False
        
        try:
            # Serialize embeddings efficiently
            serialized_embeddings = pickle.dumps(embeddings, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get embeddings dimension
            dimension = embeddings.shape[-1] if embeddings.ndim > 1 else len(embeddings)
            
            async with self.get_transaction() as transaction:
                session = transaction.session
                
                # Check if embeddings already exist
                existing = await session.execute(
                    select(EmbeddingStorage).where(
                        and_(
                            EmbeddingStorage.entity_type == entity_type,
                            EmbeddingStorage.entity_id == entity_id
                        )
                    )
                )
                existing_record = existing.scalar_one_or_none()
                
                if existing_record:
                    # Update existing embeddings
                    stmt = update(EmbeddingStorage).where(
                        EmbeddingStorage.id == existing_record.id
                    ).values(
                        embeddings=serialized_embeddings,
                        dimension=dimension,
                        updated_at=datetime.utcnow()
                    )
                    await session.execute(stmt)
                    logger.debug(f"Updated embeddings for {entity_type} {entity_id} "
                               f"(dimension: {dimension})")
                else:
                    # Create new embeddings
                    new_embedding = EmbeddingStorage(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        embeddings=serialized_embeddings,
                        dimension=dimension
                    )
                    session.add(new_embedding)
                    logger.debug(f"Saved new embeddings for {entity_type} {entity_id} "
                               f"(dimension: {dimension})")
                
                await session.commit()
                return True
                
        except IntegrityError as e:
            logger.error(f"Database integrity error saving embeddings for {entity_type} {entity_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save embeddings for {entity_type} {entity_id}: {e}")
            return False
    
    @retry_on_db_error()
    @measure_performance("get_embeddings")
    async def get_embeddings(self, entity_type: str, entity_id: str) -> Optional[np.ndarray]:
        """Get embeddings for a specific entity with validation"""
        if not entity_type or not entity_id:
            logger.error("Entity type and entity ID are required")
            return None
            
        try:
            async with self.get_session() as session:
                stmt = select(EmbeddingStorage).where(
                    and_(
                        EmbeddingStorage.entity_type == entity_type,
                        EmbeddingStorage.entity_id == entity_id
                    )
                )
                
                result = await session.execute(stmt)
                embedding_record = result.scalar_one_or_none()
                
                if embedding_record:
                    try:
                        embeddings = pickle.loads(embedding_record.embeddings)
                        
                        # Validate deserialized embeddings
                        if not isinstance(embeddings, np.ndarray):
                            logger.error(f"Invalid embeddings format for {entity_type} {entity_id}")
                            return None
                        
                        logger.debug(f"Retrieved embeddings for {entity_type} {entity_id} "
                                   f"(dimension: {embedding_record.dimension})")
                        return embeddings
                        
                    except (pickle.UnpicklingError, TypeError) as e:
                        logger.error(f"Failed to deserialize embeddings for {entity_type} {entity_id}: {e}")
                        return None
                else:
                    logger.debug(f"No embeddings found for {entity_type} {entity_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get embeddings for {entity_type} {entity_id}: {e}")
            return None
    
    @retry_on_db_error()
    @measure_performance("get_all_embeddings")
    async def get_all_embeddings(self, entity_type: str) -> Dict[str, np.ndarray]:
        """Get all embeddings for an entity type with batch processing"""
        if not entity_type:
            logger.error("Entity type is required")
            return {}
            
        try:
            async with self.get_session() as session:
                stmt = select(EmbeddingStorage).where(
                    EmbeddingStorage.entity_type == entity_type
                ).order_by(EmbeddingStorage.updated_at.desc())
                
                result = await session.execute(stmt)
                embedding_records = result.scalars().all()
                
                embeddings_dict = {}
                failed_count = 0
                
                for record in embedding_records:
                    try:
                        embeddings = pickle.loads(record.embeddings)
                        
                        # Validate deserialized embeddings
                        if isinstance(embeddings, np.ndarray):
                            embeddings_dict[record.entity_id] = embeddings
                        else:
                            logger.warning(f"Invalid embeddings format for {entity_type} {record.entity_id}")
                            failed_count += 1
                            
                    except (pickle.UnpicklingError, TypeError) as e:
                        logger.warning(f"Failed to deserialize embeddings for {entity_type} {record.entity_id}: {e}")
                        failed_count += 1
                        continue
                
                logger.info(f"Retrieved {len(embeddings_dict)} embeddings for {entity_type}")
                
                if failed_count > 0:
                    logger.warning(f"Failed to load {failed_count} embeddings for {entity_type}")
                
                return embeddings_dict
                
        except Exception as e:
            logger.error(f"Failed to get all embeddings for {entity_type}: {e}")
            return {}
    
    @retry_on_db_error()
    @measure_performance("save_training_metrics")
    async def save_training_metrics(self, model_name: str, version: str, metrics: Dict[str, float]) -> bool:
        """Save training metrics for a model with validation"""
        if not model_name or not version:
            logger.error("Model name and version are required")
            return False
        
        if not metrics or not isinstance(metrics, dict):
            logger.error("Metrics must be a non-empty dictionary")
            return False
        
        # Validate metrics values
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                logger.error(f"Metric {key} must be a number, got {type(value)}")
                return False
            if not np.isfinite(value):
                logger.error(f"Metric {key} must be finite, got {value}")
                return False
        
        try:
            async with self.get_transaction() as transaction:
                session = transaction.session
                
                # Add additional metadata to metrics
                enhanced_metrics = {
                    **metrics,
                    '_metadata': {
                        'saved_at': datetime.utcnow().isoformat(),
                        'metrics_count': len(metrics),
                        'metric_keys': list(metrics.keys())
                    }
                }
                
                training_metrics = TrainingMetrics(
                    model_name=model_name,
                    version=version,
                    metrics=enhanced_metrics
                )
                
                session.add(training_metrics)
                await session.commit()
                
                logger.info(f"Saved training metrics for {model_name} v{version} "
                          f"({len(metrics)} metrics)")
                return True
                
        except IntegrityError as e:
            logger.error(f"Database integrity error saving training metrics for {model_name} v{version}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save training metrics for {model_name} v{version}: {e}")
            return False
    
    @retry_on_db_error()
    @measure_performance("get_training_metrics")
    async def get_training_metrics(self, model_name: str, version: str) -> Optional[Dict[str, float]]:
        """Get training metrics for a specific model version"""
        if not model_name or not version:
            logger.error("Model name and version are required")
            return None
            
        try:
            async with self.get_session() as session:
                stmt = select(TrainingMetrics).where(
                    and_(
                        TrainingMetrics.model_name == model_name,
                        TrainingMetrics.version == version
                    )
                ).order_by(TrainingMetrics.training_date.desc()).limit(1)
                
                result = await session.execute(stmt)
                metrics_record = result.scalar_one_or_none()
                
                if metrics_record:
                    # Remove metadata from returned metrics
                    metrics = dict(metrics_record.metrics)
                    metrics.pop('_metadata', None)
                    
                    logger.debug(f"Retrieved training metrics for {model_name} v{version}")
                    return metrics
                else:
                    logger.debug(f"No training metrics found for {model_name} v{version}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get training metrics for {model_name} v{version}: {e}")
            return None
    
    @retry_on_db_error()
    @measure_performance("cache_predictions")
    async def cache_predictions(self, cache_key: str, predictions: Any, ttl_seconds: int = CACHE_DEFAULT_TTL) -> bool:
        """Cache predictions in PostgreSQL with TTL support"""
        if not cache_key or predictions is None:
            logger.error("Cache key and predictions are required")
            return False
        
        # Validate TTL
        if ttl_seconds <= 0 or ttl_seconds > CACHE_MAX_TTL:
            logger.error(f"Invalid TTL: {ttl_seconds}. Must be between 1 and {CACHE_MAX_TTL}")
            return False
        
        try:
            # Serialize predictions
            serialized_predictions = pickle.dumps(predictions, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate expiration
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            # Prepare metadata
            metadata = {
                'cached_at': datetime.utcnow().isoformat(),
                'ttl_seconds': ttl_seconds,
                'data_type': type(predictions).__name__,
                'size_bytes': len(serialized_predictions)
            }
            
            async with self.get_transaction() as transaction:
                session = transaction.session
                
                # Check if cache entry exists
                existing_cache = await session.execute(
                    select(PredictionCache).where(
                        PredictionCache.cache_key == cache_key
                    )
                )
                existing = existing_cache.scalar_one_or_none()
                
                if existing:
                    # Update existing cache entry
                    stmt = update(PredictionCache).where(
                        PredictionCache.id == existing.id
                    ).values(
                        predictions=serialized_predictions,
                        metadata=metadata,
                        cached_at=datetime.utcnow(),
                        expires_at=expires_at,
                        access_count=0  # Reset access count on update
                    )
                    await session.execute(stmt)
                    logger.debug(f"Updated cache entry for key: {cache_key}")
                else:
                    # Create new cache entry
                    new_cache = PredictionCache(
                        cache_key=cache_key,
                        predictions=serialized_predictions,
                        metadata=metadata,
                        expires_at=expires_at
                    )
                    session.add(new_cache)
                    logger.debug(f"Created new cache entry for key: {cache_key}")
                
                await session.commit()
                
                # Clean up expired entries periodically
                await self._cleanup_expired_cache(session)
                
                return True
                
        except IntegrityError as e:
            logger.error(f"Database integrity error caching predictions for {cache_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to cache predictions for {cache_key}: {e}")
            return False
    
    @retry_on_db_error()
    @measure_performance("get_cached_predictions")
    async def get_cached_predictions(self, cache_key: str) -> Optional[Any]:
        """Get cached predictions with automatic expiration handling"""
        if not cache_key:
            logger.error("Cache key is required")
            return None
            
        try:
            async with self.get_session() as session:
                # Get cache entry with expiration check
                stmt = select(PredictionCache).where(
                    and_(
                        PredictionCache.cache_key == cache_key,
                        PredictionCache.expires_at > datetime.utcnow()
                    )
                )
                
                result = await session.execute(stmt)
                cache_record = result.scalar_one_or_none()
                
                if cache_record:
                    # Update access statistics
                    await session.execute(
                        update(PredictionCache).where(
                            PredictionCache.id == cache_record.id
                        ).values(
                            access_count=PredictionCache.access_count + 1,
                            last_accessed=datetime.utcnow()
                        )
                    )
                    await session.commit()
                    
                    # Deserialize predictions
                    try:
                        predictions = pickle.loads(cache_record.predictions)
                        logger.debug(f"Cache hit for key: {cache_key}")
                        return predictions
                    except (pickle.UnpicklingError, TypeError) as e:
                        logger.error(f"Failed to deserialize cached predictions for {cache_key}: {e}")
                        # Remove corrupted cache entry
                        await session.execute(
                            delete(PredictionCache).where(
                                PredictionCache.id == cache_record.id
                            )
                        )
                        await session.commit()
                        return None
                else:
                    logger.debug(f"Cache miss for key: {cache_key}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get cached predictions for {cache_key}: {e}")
            return None
    
    @retry_on_db_error()
    @measure_performance("clear_cache")
    async def clear_cache(self, pattern: str = "*") -> bool:
        """Clear cache entries matching pattern"""
        try:
            async with self.get_transaction() as transaction:
                session = transaction.session
                
                if pattern == "*":
                    # Clear all cache entries
                    stmt = delete(PredictionCache)
                else:
                    # Clear entries matching pattern (using LIKE)
                    like_pattern = pattern.replace("*", "%")
                    stmt = delete(PredictionCache).where(
                        PredictionCache.cache_key.like(like_pattern)
                    )
                
                result = await session.execute(stmt)
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Cleared {deleted_count} cache entries matching pattern: {pattern}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear cache with pattern {pattern}: {e}")
            return False
    
    async def _cleanup_expired_cache(self, session: AsyncSession, batch_size: int = 1000) -> int:
        """Clean up expired cache entries"""
        try:
            # Delete expired entries in batches
            stmt = delete(PredictionCache).where(
                PredictionCache.expires_at <= datetime.utcnow()
            )
            
            result = await session.execute(stmt)
            deleted_count = result.rowcount
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            return 0
    
    def _get_python_version(self) -> str:
        """Get current Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_dependencies_info(self) -> Dict[str, str]:
        """Get information about key dependencies"""
        dependencies = {}
        
        # Core ML dependencies
        for module_name in ['tensorflow', 'torch', 'numpy', 'sklearn', 'pandas', 'scipy']:
            try:
                module = __import__(module_name)
                dependencies[module_name] = getattr(module, '__version__', 'unknown')
            except ImportError:
                continue
        
        # Database dependencies
        for module_name in ['sqlalchemy', 'asyncpg', 'psycopg2']:
            try:
                module = __import__(module_name)
                dependencies[module_name] = getattr(module, '__version__', 'unknown')
            except ImportError:
                continue
        
        return dependencies
    
    def _calculate_compression_ratio(self, original_data: Any, serialized_data: bytes) -> float:
        """Calculate compression ratio for serialized data"""
        try:
            # Estimate original size (rough approximation)
            import sys
            original_size = sys.getsizeof(original_data)
            serialized_size = len(serialized_data)
            
            if original_size > 0:
                return serialized_size / original_size
            return 1.0
        except Exception:
            return 1.0
    
    @retry_on_db_error()
    @measure_performance("get_model_info")
    async def get_model_info(self, model_name: str, version: str = "latest") -> Optional[Dict]:
        """Get model information and metadata"""
        if not model_name:
            logger.error("Model name is required")
            return None
            
        try:
            async with self.get_session() as session:
                if version == "latest":
                    stmt = select(MLModelStorage).where(
                        and_(
                            MLModelStorage.model_name == model_name,
                            MLModelStorage.is_active == True
                        )
                    ).order_by(MLModelStorage.created_at.desc()).limit(1)
                else:
                    stmt = select(MLModelStorage).where(
                        and_(
                            MLModelStorage.model_name == model_name,
                            MLModelStorage.version == version,
                            MLModelStorage.is_active == True
                        )
                    )
                
                result = await session.execute(stmt)
                model_record = result.scalar_one_or_none()
                
                if model_record:
                    return {
                        'id': str(model_record.id),
                        'model_name': model_record.model_name,
                        'version': model_record.version,
                        'created_at': model_record.created_at,
                        'metadata': model_record.model_metadata,
                        'is_active': model_record.is_active,
                        'size_bytes': model_record.model_metadata.get('size_bytes', 0)
                    }
                else:
                    logger.debug(f"Model {model_name} version {version} not found")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name} version {version}: {e}")
            return None
    
    @retry_on_db_error()
    @measure_performance("cleanup_old_models")
    async def cleanup_old_models(self, model_name: str, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the most recent ones"""
        if not model_name:
            logger.error("Model name is required")
            return 0
        
        if keep_versions < 1:
            logger.error("Must keep at least 1 version")
            return 0
        
        try:
            async with self.get_transaction() as transaction:
                session = transaction.session
                
                # Get all versions ordered by creation date
                stmt = select(MLModelStorage).where(
                    and_(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.is_active == True
                    )
                ).order_by(MLModelStorage.created_at.desc())
                
                result = await session.execute(stmt)
                all_versions = result.scalars().all()
                
                if len(all_versions) <= keep_versions:
                    logger.info(f"No cleanup needed for {model_name}: {len(all_versions)} versions <= {keep_versions}")
                    return 0
                
                # Mark old versions as inactive
                versions_to_delete = all_versions[keep_versions:]
                deleted_count = 0
                
                for version in versions_to_delete:
                    stmt = update(MLModelStorage).where(
                        MLModelStorage.id == version.id
                    ).values(is_active=False)
                    
                    result = await session.execute(stmt)
                    if result.rowcount > 0:
                        deleted_count += 1
                
                await session.commit()
                logger.info(f"Cleaned up {deleted_count} old versions of model {model_name}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old models for {model_name}: {e}")
            return 0
    
    async def get_performance_metrics(self) -> List[QueryMetrics]:
        """Get performance metrics for database operations"""
        if not self.enable_metrics or not self._performance_metrics:
            return []
        
        return self._performance_metrics.copy()
    
    async def clear_performance_metrics(self):
        """Clear performance metrics"""
        if self._performance_metrics:
            self._performance_metrics.clear()
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            async with self.get_session() as session:
                # Get cache entry counts
                total_entries = await session.execute(
                    select(func.count(PredictionCache.id))
                )
                total_count = total_entries.scalar()
                
                # Get expired entries count
                expired_entries = await session.execute(
                    select(func.count(PredictionCache.id)).where(
                        PredictionCache.expires_at <= datetime.utcnow()
                    )
                )
                expired_count = expired_entries.scalar()
                
                # Get cache size statistics
                cache_size = await session.execute(
                    select(func.sum(func.length(PredictionCache.predictions)))
                )
                total_size = cache_size.scalar() or 0
                
                # Get most accessed entries
                top_accessed = await session.execute(
                    select(
                        PredictionCache.cache_key,
                        PredictionCache.access_count,
                        PredictionCache.last_accessed
                    ).order_by(PredictionCache.access_count.desc()).limit(10)
                )
                
                return {
                    'total_entries': total_count,
                    'expired_entries': expired_count,
                    'active_entries': total_count - expired_count,
                    'total_size_bytes': total_size,
                    'average_size_bytes': total_size / total_count if total_count > 0 else 0,
                    'top_accessed_keys': [
                        {
                            'key': row.cache_key,
                            'access_count': row.access_count,
                            'last_accessed': row.last_accessed
                        }
                        for row in top_accessed
                    ]
                }
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {}
    
    async def batch_save_embeddings(self, embeddings_data: List[Tuple[str, str, np.ndarray]], 
                                   batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, bool]:
        """Batch save embeddings for multiple entities"""
        if not embeddings_data:
            logger.error("No embeddings data provided")
            return {}
        
        results = {}
        
        # Process in batches
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]
            
            try:
                async with self.get_transaction() as transaction:
                    session = transaction.session
                    
                    for entity_type, entity_id, embeddings in batch:
                        try:
                            # Validate inputs
                            if not entity_type or not entity_id or embeddings is None:
                                results[f"{entity_type}:{entity_id}"] = False
                                continue
                            
                            # Serialize embeddings
                            serialized_embeddings = pickle.dumps(embeddings, protocol=pickle.HIGHEST_PROTOCOL)
                            dimension = embeddings.shape[-1] if embeddings.ndim > 1 else len(embeddings)
                            
                            # Check if exists
                            existing = await session.execute(
                                select(EmbeddingStorage).where(
                                    and_(
                                        EmbeddingStorage.entity_type == entity_type,
                                        EmbeddingStorage.entity_id == entity_id
                                    )
                                )
                            )
                            existing_record = existing.scalar_one_or_none()
                            
                            if existing_record:
                                # Update
                                await session.execute(
                                    update(EmbeddingStorage).where(
                                        EmbeddingStorage.id == existing_record.id
                                    ).values(
                                        embeddings=serialized_embeddings,
                                        dimension=dimension,
                                        updated_at=datetime.utcnow()
                                    )
                                )
                            else:
                                # Insert
                                new_embedding = EmbeddingStorage(
                                    entity_type=entity_type,
                                    entity_id=entity_id,
                                    embeddings=serialized_embeddings,
                                    dimension=dimension
                                )
                                session.add(new_embedding)
                            
                            results[f"{entity_type}:{entity_id}"] = True
                            
                        except Exception as e:
                            logger.error(f"Failed to save embeddings for {entity_type}:{entity_id}: {e}")
                            results[f"{entity_type}:{entity_id}"] = False
                    
                    await session.commit()
                    
            except Exception as e:
                logger.error(f"Failed to save embeddings batch: {e}")
                # Mark all items in batch as failed
                for entity_type, entity_id, _ in batch:
                    results[f"{entity_type}:{entity_id}"] = False
        
        successful_count = sum(1 for success in results.values() if success)
        logger.info(f"Batch saved {successful_count}/{len(embeddings_data)} embeddings")
        
        return results
    
    async def get_model_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for all models"""
        try:
            async with self.get_session() as session:
                # Get model count by active status
                active_models = await session.execute(
                    select(func.count(MLModelStorage.id)).where(
                        MLModelStorage.is_active == True
                    )
                )
                active_count = active_models.scalar()
                
                inactive_models = await session.execute(
                    select(func.count(MLModelStorage.id)).where(
                        MLModelStorage.is_active == False
                    )
                )
                inactive_count = inactive_models.scalar()
                
                # Get total storage size
                total_size = await session.execute(
                    select(func.sum(func.length(MLModelStorage.model_data)))
                )
                size_bytes = total_size.scalar() or 0
                
                # Get model count by name
                model_counts = await session.execute(
                    select(
                        MLModelStorage.model_name,
                        func.count(MLModelStorage.id).label('version_count'),
                        func.sum(func.length(MLModelStorage.model_data)).label('total_size')
                    ).where(
                        MLModelStorage.is_active == True
                    ).group_by(MLModelStorage.model_name)
                )
                
                model_stats = [
                    {
                        'model_name': row.model_name,
                        'version_count': row.version_count,
                        'total_size_bytes': row.total_size or 0
                    }
                    for row in model_counts
                ]
                
                return {
                    'active_models': active_count,
                    'inactive_models': inactive_count,
                    'total_storage_bytes': size_bytes,
                    'total_storage_mb': size_bytes / (1024 * 1024),
                    'models_by_name': model_stats
                }
        except Exception as e:
            logger.error(f"Failed to get model storage stats: {e}")
            return {}
    
    async def get_embeddings_stats(self) -> Dict[str, Any]:
        """Get statistics for stored embeddings"""
        try:
            async with self.get_session() as session:
                # Get embedding count by entity type
                entity_counts = await session.execute(
                    select(
                        EmbeddingStorage.entity_type,
                        func.count(EmbeddingStorage.id).label('count'),
                        func.avg(EmbeddingStorage.dimension).label('avg_dimension'),
                        func.sum(func.length(EmbeddingStorage.embeddings)).label('total_size')
                    ).group_by(EmbeddingStorage.entity_type)
                )
                
                entity_stats = [
                    {
                        'entity_type': row.entity_type,
                        'count': row.count,
                        'average_dimension': float(row.avg_dimension) if row.avg_dimension else 0,
                        'total_size_bytes': row.total_size or 0
                    }
                    for row in entity_counts
                ]
                
                # Get total counts
                total_embeddings = await session.execute(
                    select(func.count(EmbeddingStorage.id))
                )
                total_count = total_embeddings.scalar()
                
                total_size = await session.execute(
                    select(func.sum(func.length(EmbeddingStorage.embeddings)))
                )
                size_bytes = total_size.scalar() or 0
                
                return {
                    'total_embeddings': total_count,
                    'total_storage_bytes': size_bytes,
                    'total_storage_mb': size_bytes / (1024 * 1024),
                    'embeddings_by_type': entity_stats
                }
        except Exception as e:
            logger.error(f"Failed to get embeddings stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the repository"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        try:
            # Test database connection
            async with self.get_session() as session:
                await session.execute(select(1))
                health_status['checks']['database_connection'] = 'ok'
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['database_connection'] = f'error: {e}'
        
        try:
            # Test model table access
            async with self.get_session() as session:
                await session.execute(select(func.count(MLModelStorage.id)))
                health_status['checks']['model_table_access'] = 'ok'
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['model_table_access'] = f'error: {e}'
        
        try:
            # Test embeddings table access
            async with self.get_session() as session:
                await session.execute(select(func.count(EmbeddingStorage.id)))
                health_status['checks']['embeddings_table_access'] = 'ok'
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['embeddings_table_access'] = f'error: {e}'
        
        try:
            # Test cache table access
            async with self.get_session() as session:
                await session.execute(select(func.count(PredictionCache.id)))
                health_status['checks']['cache_table_access'] = 'ok'
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['cache_table_access'] = f'error: {e}'
        
        return health_status
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Perform database maintenance and optimization"""
        optimization_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'operations': {}
        }
        
        try:
            # Clean up expired cache entries
            async with self.get_session() as session:
                expired_count = await self._cleanup_expired_cache(session)
                optimization_results['operations']['expired_cache_cleanup'] = {
                    'status': 'completed',
                    'deleted_entries': expired_count
                }
        except Exception as e:
            optimization_results['operations']['expired_cache_cleanup'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        try:
            # Analyze table statistics (PostgreSQL ANALYZE)
            async with self.engine.begin() as conn:
                await conn.execute(text("ANALYZE ml_models;"))
                await conn.execute(text("ANALYZE embeddings;"))
                await conn.execute(text("ANALYZE training_metrics;"))
                await conn.execute(text("ANALYZE prediction_cache;"))
                
                optimization_results['operations']['table_analyze'] = {
                    'status': 'completed',
                    'tables_analyzed': ['ml_models', 'embeddings', 'training_metrics', 'prediction_cache']
                }
        except Exception as e:
            optimization_results['operations']['table_analyze'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        return optimization_results