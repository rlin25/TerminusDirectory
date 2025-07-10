import asyncio
import json
import logging
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, DateTime, 
    Text, LargeBinary, JSON, select, update, delete, insert
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.exc import SQLAlchemyError
from uuid import uuid4

from ....domain.repositories.model_repository import ModelRepository

# Configure logging
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class MLModelStorage(Base):
    __tablename__ = "ml_models"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_data = Column(LargeBinary, nullable=False)
    metadata = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        {'comment': 'Storage for trained ML models and their metadata'}
    )

class EmbeddingStorage(Base):
    __tablename__ = "embeddings"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_type = Column(String(50), nullable=False, index=True)  # 'user', 'property', etc.
    entity_id = Column(String(255), nullable=False, index=True)
    embeddings = Column(LargeBinary, nullable=False)  # Serialized numpy array
    dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
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
        {'comment': 'Storage for ML model training metrics and performance data'}
    )


class PostgresModelRepository(ModelRepository):
    """PostgreSQL-based ML model repository implementation"""
    
    def __init__(self, database_url: str, pool_size: int = 10, max_overflow: int = 20):
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=False
        )
        
        # Create async session factory
        self.async_session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self):
        """Create database tables (for testing/development)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """Close database connections"""
        await self.engine.dispose()
    
    async def save_model(self, model_name: str, model_data: Any, version: str) -> bool:
        """Save ML model to database"""
        try:
            async with self.async_session_factory() as session:
                # Serialize model data
                serialized_data = pickle.dumps(model_data)
                
                # Prepare metadata
                metadata = {
                    'size_bytes': len(serialized_data),
                    'saved_at': datetime.utcnow().isoformat(),
                    'python_version': self._get_python_version(),
                    'dependencies': self._get_dependencies_info()
                }
                
                # Check if model version already exists
                existing_model = await session.execute(
                    select(MLModelStorage).where(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.version == version
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
                    logger.info(f"Updated existing model {model_name} version {version}")
                else:
                    # Create new model
                    new_model = MLModelStorage(
                        model_name=model_name,
                        version=version,
                        model_data=serialized_data,
                        metadata=metadata
                    )
                    session.add(new_model)
                    logger.info(f"Saved new model {model_name} version {version}")
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save model {model_name} version {version}: {e}")
            return False
    
    async def load_model(self, model_name: str, version: str = "latest") -> Optional[Any]:
        """Load ML model from database"""
        try:
            async with self.async_session_factory() as session:
                if version == "latest":
                    # Get the most recent version
                    stmt = select(MLModelStorage).where(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.is_active == True
                    ).order_by(MLModelStorage.created_at.desc()).limit(1)
                else:
                    # Get specific version
                    stmt = select(MLModelStorage).where(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.version == version,
                        MLModelStorage.is_active == True
                    )
                
                result = await session.execute(stmt)
                model_record = result.scalar_one_or_none()
                
                if model_record:
                    # Deserialize model data
                    model_data = pickle.loads(model_record.model_data)
                    logger.info(f"Loaded model {model_name} version {model_record.version}")
                    return model_data
                else:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_name} version {version}: {e}")
            return None
    
    async def get_model_versions(self, model_name: str) -> List[str]:
        """Get all versions of a specific model"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(MLModelStorage.version).where(
                    MLModelStorage.model_name == model_name,
                    MLModelStorage.is_active == True
                ).order_by(MLModelStorage.created_at.desc())
                
                result = await session.execute(stmt)
                versions = [row[0] for row in result]
                
                logger.debug(f"Found {len(versions)} versions for model {model_name}")
                return versions
                
        except Exception as e:
            logger.error(f"Failed to get versions for model {model_name}: {e}")
            return []
    
    async def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a specific model version (soft delete)"""
        try:
            async with self.async_session_factory() as session:
                stmt = update(MLModelStorage).where(
                    MLModelStorage.model_name == model_name,
                    MLModelStorage.version == version
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
    
    async def save_embeddings(self, entity_type: str, entity_id: str, embeddings: np.ndarray) -> bool:
        """Save entity embeddings"""
        try:
            async with self.async_session_factory() as session:
                # Serialize embeddings
                serialized_embeddings = pickle.dumps(embeddings)
                
                # Check if embeddings already exist
                existing = await session.execute(
                    select(EmbeddingStorage).where(
                        EmbeddingStorage.entity_type == entity_type,
                        EmbeddingStorage.entity_id == entity_id
                    )
                )
                existing_record = existing.scalar_one_or_none()
                
                if existing_record:
                    # Update existing embeddings
                    stmt = update(EmbeddingStorage).where(
                        EmbeddingStorage.id == existing_record.id
                    ).values(
                        embeddings=serialized_embeddings,
                        dimension=len(embeddings),
                        updated_at=datetime.utcnow()
                    )
                    await session.execute(stmt)
                else:
                    # Create new embeddings
                    new_embedding = EmbeddingStorage(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        embeddings=serialized_embeddings,
                        dimension=len(embeddings)
                    )
                    session.add(new_embedding)
                
                await session.commit()
                logger.debug(f"Saved embeddings for {entity_type} {entity_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save embeddings for {entity_type} {entity_id}: {e}")
            return False
    
    async def get_embeddings(self, entity_type: str, entity_id: str) -> Optional[np.ndarray]:
        """Get embeddings for a specific entity"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(EmbeddingStorage).where(
                    EmbeddingStorage.entity_type == entity_type,
                    EmbeddingStorage.entity_id == entity_id
                )
                
                result = await session.execute(stmt)
                embedding_record = result.scalar_one_or_none()
                
                if embedding_record:
                    embeddings = pickle.loads(embedding_record.embeddings)
                    return embeddings
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get embeddings for {entity_type} {entity_id}: {e}")
            return None
    
    async def get_all_embeddings(self, entity_type: str) -> Dict[str, np.ndarray]:
        """Get all embeddings for an entity type"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(EmbeddingStorage).where(
                    EmbeddingStorage.entity_type == entity_type
                )
                
                result = await session.execute(stmt)
                embedding_records = result.scalars().all()
                
                embeddings_dict = {}
                for record in embedding_records:
                    embeddings = pickle.loads(record.embeddings)
                    embeddings_dict[record.entity_id] = embeddings
                
                logger.debug(f"Retrieved {len(embeddings_dict)} embeddings for {entity_type}")
                return embeddings_dict
                
        except Exception as e:
            logger.error(f"Failed to get all embeddings for {entity_type}: {e}")
            return {}
    
    async def save_training_metrics(self, model_name: str, version: str, metrics: Dict[str, float]) -> bool:
        """Save training metrics for a model"""
        try:
            async with self.async_session_factory() as session:
                training_metrics = TrainingMetrics(
                    model_name=model_name,
                    version=version,
                    metrics=metrics
                )
                
                session.add(training_metrics)
                await session.commit()
                
                logger.info(f"Saved training metrics for {model_name} version {version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save training metrics for {model_name} version {version}: {e}")
            return False
    
    async def get_training_metrics(self, model_name: str, version: str) -> Optional[Dict[str, float]]:
        """Get training metrics for a specific model version"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(TrainingMetrics).where(
                    TrainingMetrics.model_name == model_name,
                    TrainingMetrics.version == version
                ).order_by(TrainingMetrics.training_date.desc()).limit(1)
                
                result = await session.execute(stmt)
                metrics_record = result.scalar_one_or_none()
                
                if metrics_record:
                    return metrics_record.metrics
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get training metrics for {model_name} version {version}: {e}")
            return None
    
    async def cache_predictions(self, cache_key: str, predictions: Any, ttl_seconds: int = 3600) -> bool:
        """Cache predictions (delegated to Redis cache)"""
        # This method would typically delegate to a Redis cache
        # For now, we'll implement a simple in-memory cache or return False
        logger.warning("cache_predictions not implemented - use Redis cache repository")
        return False
    
    async def get_cached_predictions(self, cache_key: str) -> Optional[Any]:
        """Get cached predictions (delegated to Redis cache)"""
        logger.warning("get_cached_predictions not implemented - use Redis cache repository")
        return None
    
    async def clear_cache(self, pattern: str = "*") -> bool:
        """Clear cache (delegated to Redis cache)"""
        logger.warning("clear_cache not implemented - use Redis cache repository")
        return False
    
    def _get_python_version(self) -> str:
        """Get current Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_dependencies_info(self) -> Dict[str, str]:
        """Get information about key dependencies"""
        try:
            import tensorflow as tf
            import numpy as np
            import sklearn
            
            return {
                'tensorflow': tf.__version__,
                'numpy': np.__version__,
                'sklearn': sklearn.__version__
            }
        except ImportError:
            return {}
    
    async def get_model_info(self, model_name: str, version: str = "latest") -> Optional[Dict]:
        """Get model information and metadata"""
        try:
            async with self.async_session_factory() as session:
                if version == "latest":
                    stmt = select(MLModelStorage).where(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.is_active == True
                    ).order_by(MLModelStorage.created_at.desc()).limit(1)
                else:
                    stmt = select(MLModelStorage).where(
                        MLModelStorage.model_name == model_name,
                        MLModelStorage.version == version,
                        MLModelStorage.is_active == True
                    )
                
                result = await session.execute(stmt)
                model_record = result.scalar_one_or_none()
                
                if model_record:
                    return {
                        'model_name': model_record.model_name,
                        'version': model_record.version,
                        'created_at': model_record.created_at,
                        'metadata': model_record.metadata,
                        'is_active': model_record.is_active
                    }
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name} version {version}: {e}")
            return None
    
    async def cleanup_old_models(self, model_name: str, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            async with self.async_session_factory() as session:
                # Get all versions ordered by creation date
                stmt = select(MLModelStorage).where(
                    MLModelStorage.model_name == model_name,
                    MLModelStorage.is_active == True
                ).order_by(MLModelStorage.created_at.desc())
                
                result = await session.execute(stmt)
                all_versions = result.scalars().all()
                
                if len(all_versions) <= keep_versions:
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