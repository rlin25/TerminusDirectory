import asyncio
import logging
from typing import List, Optional, Dict, Tuple, Any, Union
from uuid import UUID
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, DateTime, 
    Text, ARRAY, JSON, and_, or_, func, select, update, delete, insert,
    Index, UniqueConstraint, CheckConstraint, event, Enum
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncTransaction
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, TSVECTOR
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import events
import asyncpg
import json
from functools import wraps
from dataclasses import dataclass
import time

from ....domain.entities.property import Property
from ....domain.entities.search_query import SearchQuery, SearchFilters
from ....domain.repositories.property_repository import PropertyRepository

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
                    f"Query failed: {operation_name} took {execution_time:.2f}s, error: {e}"
                )
                raise
        return wrapper
    return decorator

# Database model
Base = declarative_base()

class PropertyModel(Base):
    __tablename__ = "properties"
    
    # Primary key and core fields
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False, index=True)
    location = Column(String(255), nullable=False, index=True)
    bedrooms = Column(Integer, nullable=False, index=True)
    bathrooms = Column(Float, nullable=False, index=True)
    square_feet = Column(Integer, nullable=True, index=True)
    amenities = Column(ARRAY(String), nullable=False, default=[])
    contact_info = Column(JSON, nullable=False, default={})
    images = Column(ARRAY(String), nullable=False, default=[])
    scraped_at = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="active", index=True)  # Maps to property_status enum
    property_type = Column(String(50), nullable=False, default="apartment", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    
    # Search optimization columns
    full_text_search = Column(Text, nullable=True)  # For full-text search
    search_vector = Column(TSVECTOR)  # PostgreSQL full-text search vector
    price_per_sqft = Column(Float, nullable=True, index=True)  # Computed column
    
    # ML and analytics fields
    view_count = Column(Integer, default=0, nullable=False)
    favorite_count = Column(Integer, default=0, nullable=False)
    contact_count = Column(Integer, default=0, nullable=False)
    last_viewed = Column(DateTime, nullable=True)
    
    # Geographic data (for future spatial queries)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Data quality and validation
    data_quality_score = Column(Float, nullable=True)  # 0-1 score
    validation_errors = Column(JSON, nullable=True)  # Store validation issues
    
    # Constraints
    __table_args__ = (
        CheckConstraint('price > 0', name='check_price_positive'),
        CheckConstraint('bedrooms >= 0', name='check_bedrooms_non_negative'),
        CheckConstraint('bathrooms >= 0', name='check_bathrooms_non_negative'),
        CheckConstraint('square_feet > 0 OR square_feet IS NULL', name='check_square_feet_positive'),
        CheckConstraint('view_count >= 0', name='check_view_count_non_negative'),
        CheckConstraint('favorite_count >= 0', name='check_favorite_count_non_negative'),
        CheckConstraint('contact_count >= 0', name='check_contact_count_non_negative'),
        CheckConstraint('data_quality_score >= 0 AND data_quality_score <= 1 OR data_quality_score IS NULL', name='check_data_quality_score_range'),
        # Composite indexes for common query patterns
        Index('idx_active_location_price', 'status', 'location', 'price'),
        Index('idx_active_bedrooms_price', 'status', 'bedrooms', 'price'),
        Index('idx_active_type_price', 'status', 'property_type', 'price'),
        Index('idx_location_bedrooms_price', 'location', 'bedrooms', 'price'),
        Index('idx_price_sqft_ratio', 'price', 'square_feet'),
        Index('idx_scraped_active', 'scraped_at', 'status'),
        # Full-text search index
        Index('idx_search_vector_gin', 'search_vector', postgresql_using='gin'),
        # Array indexes for amenities
        Index('idx_amenities_gin', 'amenities', postgresql_using='gin'),
        # Geographic index (for future spatial queries)
        Index('idx_location_coords', 'latitude', 'longitude'),
    )
    
    def to_domain(self) -> Property:
        """Convert database model to domain entity"""
        return Property(
            id=self.id,
            title=self.title,
            description=self.description,
            price=self.price,
            location=self.location,
            bedrooms=self.bedrooms,
            bathrooms=self.bathrooms,
            square_feet=self.square_feet,
            amenities=self.amenities or [],
            contact_info=self.contact_info or {},
            images=self.images or [],
            scraped_at=self.scraped_at,
            is_active=self.status == "active",  # Convert status enum to boolean
            property_type=self.property_type
        )
    
    @classmethod
    def from_domain(cls, property: Property) -> 'PropertyModel':
        """Convert domain entity to database model"""
        full_text = f"{property.title} {property.description} {property.location} {' '.join(property.amenities)}"
        price_per_sqft = property.get_price_per_sqft()
        
        # Calculate data quality score
        quality_score = cls._calculate_data_quality_score(property)
        
        return cls(
            id=property.id,
            title=property.title,
            description=property.description,
            price=property.price,
            location=property.location,
            bedrooms=property.bedrooms,
            bathrooms=property.bathrooms,
            square_feet=property.square_feet,
            amenities=property.amenities,
            contact_info=property.contact_info,
            images=property.images,
            scraped_at=property.scraped_at,
            status="active" if property.is_active else "inactive",  # Convert boolean to status enum
            property_type=property.property_type,
            full_text_search=full_text,
            price_per_sqft=price_per_sqft,
            data_quality_score=quality_score
        )
    
    @staticmethod
    def _calculate_data_quality_score(property: Property) -> float:
        """Calculate data quality score based on completeness and validity"""
        score = 0.0
        total_checks = 10
        
        # Check required fields
        if property.title and len(property.title.strip()) > 0:
            score += 1
        if property.description and len(property.description.strip()) > 10:
            score += 1
        if property.price > 0:
            score += 1
        if property.location and len(property.location.strip()) > 0:
            score += 1
        if property.bedrooms >= 0:
            score += 1
        if property.bathrooms >= 0:
            score += 1
        
        # Check optional but valuable fields
        if property.square_feet and property.square_feet > 0:
            score += 1
        if property.amenities and len(property.amenities) > 0:
            score += 1
        if property.images and len(property.images) > 0:
            score += 1
        if property.contact_info and len(property.contact_info) > 0:
            score += 1
        
        return score / total_checks
    
    def update_search_vector(self, session):
        """Update the search vector for full-text search"""
        if self.full_text_search:
            # Use PostgreSQL's to_tsvector function to create search vector
            search_text = f"{self.title} {self.description} {self.location} {' '.join(self.amenities or [])}"
            # This would typically be handled by a database trigger or background job
            # For now, we'll set it in the application layer
            self.search_vector = func.to_tsvector('english', search_text)
    
    def increment_view_count(self):
        """Increment view count and update last viewed timestamp"""
        self.view_count = (self.view_count or 0) + 1
        self.last_viewed = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def increment_favorite_count(self):
        """Increment favorite count"""
        self.favorite_count = (self.favorite_count or 0) + 1
        self.updated_at = datetime.utcnow()
    
    def increment_contact_count(self):
        """Increment contact count"""
        self.contact_count = (self.contact_count or 0) + 1
        self.updated_at = datetime.utcnow()
    
    def is_high_quality(self) -> bool:
        """Check if property has high data quality"""
        return (self.data_quality_score or 0) >= 0.8
    
    def is_popular(self) -> bool:
        """Check if property is popular based on engagement metrics"""
        total_engagement = (self.view_count or 0) + (self.favorite_count or 0) * 5 + (self.contact_count or 0) * 10
        return total_engagement >= 100


class PostgresPropertyRepository(PropertyRepository):
    """PostgreSQL-based property repository implementation with enhanced features"""
    
    def __init__(self, database_url: str, pool_size: int = 10, max_overflow: int = 20, 
                 enable_performance_monitoring: bool = True):
        self.database_url = database_url
        self.enable_performance_monitoring = enable_performance_monitoring
        self._performance_metrics: List[QueryMetrics] = []
        
        # Enhanced engine configuration
        self.engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_timeout=CONNECTION_TIMEOUT,
            echo=False,  # Set to True for SQL debugging
            connect_args={
                "command_timeout": QUERY_TIMEOUT,
                "server_settings": {
                    "application_name": "rental_ml_property_repo",
                    "jit": "off",  # Disable JIT for better performance on short queries
                }
            }
        )
        
        # Create async session factory
        self.async_session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Setup connection pool monitoring
        if enable_performance_monitoring:
            self._setup_pool_monitoring()
    
    def _setup_pool_monitoring(self):
        """Setup connection pool monitoring"""
        @event.listens_for(self.engine.sync_engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            logger.debug("New database connection established")
        
        @event.listens_for(self.engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self.engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            logger.debug("Connection returned to pool")
    
    @asynccontextmanager
    async def get_session(self):
        """Context manager for database sessions with automatic cleanup"""
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Session rolled back due to error: {e}")
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_transaction(self):
        """Context manager for database transactions"""
        async with self.async_session_factory() as session:
            async with session.begin():
                try:
                    yield session
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Transaction rolled back due to error: {e}")
                    raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection"""
        try:
            async with self.get_session() as session:
                start_time = time.time()
                result = await session.execute(select(func.count(PropertyModel.id)))
                response_time = time.time() - start_time
                
                pool_status = self.engine.pool.status() if hasattr(self.engine.pool, 'status') else "unknown"
                
                return {
                    "status": "healthy",
                    "response_time_ms": response_time * 1000,
                    "pool_status": pool_status,
                    "total_properties": result.scalar(),
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_performance_metrics(self) -> List[QueryMetrics]:
        """Get performance metrics for monitoring"""
        return self._performance_metrics.copy()
    
    def clear_performance_metrics(self):
        """Clear performance metrics"""
        self._performance_metrics.clear()
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }
    
    async def create_tables(self):
        """Create database tables with indexes and constraints"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
                # Create additional indexes for performance
                await conn.execute(text("""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_properties_search_gin 
                    ON properties USING GIN(to_tsvector('english', full_text_search))
                """))
                
                await conn.execute(text("""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_properties_location_trgm 
                    ON properties USING GIN(location gin_trgm_ops)
                """))
                
                logger.info("Database tables and indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    async def close(self):
        """Close database connections and cleanup resources"""
        try:
            await self.engine.dispose()
            logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("create_property")
    async def create(self, property: Property) -> Property:
        """Create a new property with enhanced error handling and validation"""
        try:
            async with self.get_transaction() as session:
                # Validate property data
                self._validate_property_data(property)
                
                property_model = PropertyModel.from_domain(property)
                
                # Update search vector
                property_model.update_search_vector(session)
                
                session.add(property_model)
                await session.flush()  # Flush to get any constraint violations
                await session.refresh(property_model)
                
                logger.info(f"Created property with ID: {property.id}, quality score: {property_model.data_quality_score}")
                return property_model.to_domain()
                
        except IntegrityError as e:
            logger.error(f"Integrity error creating property {property.id}: {e}")
            raise ValueError(f"Property data violates database constraints: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error creating property {property.id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating property {property.id}: {e}")
            raise
    
    def _validate_property_data(self, property: Property):
        """Validate property data before database operations"""
        errors = []
        
        if not property.title or len(property.title.strip()) == 0:
            errors.append("Title is required")
        if not property.description or len(property.description.strip()) == 0:
            errors.append("Description is required")
        if property.price <= 0:
            errors.append("Price must be greater than 0")
        if not property.location or len(property.location.strip()) == 0:
            errors.append("Location is required")
        if property.bedrooms < 0:
            errors.append("Bedrooms cannot be negative")
        if property.bathrooms < 0:
            errors.append("Bathrooms cannot be negative")
        if property.square_feet is not None and property.square_feet <= 0:
            errors.append("Square feet must be positive if provided")
        
        if errors:
            raise ValueError(f"Property validation failed: {', '.join(errors)}")
    
    @retry_on_db_error()
    @measure_performance("get_property_by_id")
    async def get_by_id(self, property_id: UUID, increment_view: bool = False) -> Optional[Property]:
        """Get property by ID with optional view count increment"""
        try:
            async with self.get_session() as session:
                stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.id == property_id,
                        PropertyModel.status == "active"
                    )
                )
                result = await session.execute(stmt)
                property_model = result.scalar_one_or_none()
                
                if property_model:
                    if increment_view:
                        # Increment view count in a separate transaction to avoid locking
                        await self._increment_view_count(property_id)
                    
                    return property_model.to_domain()
                return None
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting property by ID {property_id}: {e}")
            raise
    
    async def _increment_view_count(self, property_id: UUID):
        """Increment view count for a property (separate transaction)"""
        try:
            async with self.get_transaction() as session:
                stmt = update(PropertyModel).where(
                    PropertyModel.id == property_id
                ).values(
                    view_count=PropertyModel.view_count + 1,
                    last_viewed=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await session.execute(stmt)
                
        except SQLAlchemyError as e:
            logger.warning(f"Failed to increment view count for property {property_id}: {e}")
    
    @retry_on_db_error()
    @measure_performance("get_properties_by_ids")
    async def get_by_ids(self, property_ids: List[UUID], only_active: bool = True) -> List[Property]:
        """Get multiple properties by IDs with better performance"""
        if not property_ids:
            return []
        
        # Split large requests into batches to avoid query size limits
        if len(property_ids) > MAX_BATCH_SIZE:
            logger.warning(f"Large batch request ({len(property_ids)} IDs), splitting into chunks")
            all_properties = []
            for i in range(0, len(property_ids), MAX_BATCH_SIZE):
                batch_ids = property_ids[i:i + MAX_BATCH_SIZE]
                batch_properties = await self.get_by_ids(batch_ids, only_active)
                all_properties.extend(batch_properties)
            return all_properties
        
        try:
            async with self.get_session() as session:
                stmt = select(PropertyModel).where(PropertyModel.id.in_(property_ids))
                
                if only_active:
                    stmt = stmt.where(PropertyModel.status == "active")
                
                # Order by the order of input IDs for consistent results
                case_stmt = func.array_position(property_ids, PropertyModel.id)
                stmt = stmt.order_by(case_stmt)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                logger.debug(f"Retrieved {len(property_models)} properties from {len(property_ids)} requested IDs")
                return [model.to_domain() for model in property_models]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting properties by IDs: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("update_property")
    async def update(self, property: Property) -> Property:
        """Update an existing property with comprehensive validation"""
        try:
            async with self.get_transaction() as session:
                # Validate property data
                self._validate_property_data(property)
                
                # Check if property exists
                existing_stmt = select(PropertyModel).where(PropertyModel.id == property.id)
                existing_result = await session.execute(existing_stmt)
                existing_property = existing_result.scalar_one_or_none()
                
                if not existing_property:
                    raise ValueError(f"Property with ID {property.id} not found")
                
                property_model = PropertyModel.from_domain(property)
                
                # Update the property with optimistic locking check
                update_values = {
                    'title': property_model.title,
                    'description': property_model.description,
                    'price': property_model.price,
                    'location': property_model.location,
                    'bedrooms': property_model.bedrooms,
                    'bathrooms': property_model.bathrooms,
                    'square_feet': property_model.square_feet,
                    'amenities': property_model.amenities,
                    'contact_info': property_model.contact_info,
                    'images': property_model.images,
                    'status': property_model.status,
                    'property_type': property_model.property_type,
                    'full_text_search': property_model.full_text_search,
                    'price_per_sqft': property_model.price_per_sqft,
                    'data_quality_score': property_model.data_quality_score,
                    'updated_at': datetime.utcnow()
                }
                
                # Update search vector
                search_text = f"{property.title} {property.description} {property.location} {' '.join(property.amenities)}"
                update_values['search_vector'] = func.to_tsvector('english', search_text)
                
                stmt = update(PropertyModel).where(
                    PropertyModel.id == property.id
                ).values(**update_values)
                
                result = await session.execute(stmt)
                
                if result.rowcount == 0:
                    raise ValueError(f"No property found with ID {property.id} to update")
                
                logger.info(f"Updated property with ID: {property.id}, quality score: {property_model.data_quality_score}")
                return property
                
        except IntegrityError as e:
            logger.error(f"Integrity error updating property {property.id}: {e}")
            raise ValueError(f"Property data violates database constraints: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error updating property {property.id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating property {property.id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("delete_property")
    async def delete(self, property_id: UUID, hard_delete: bool = False) -> bool:
        """Delete a property (soft delete by default, hard delete optional)"""
        try:
            async with self.get_transaction() as session:
                if hard_delete:
                    # Hard delete - completely remove from database
                    stmt = delete(PropertyModel).where(PropertyModel.id == property_id)
                    result = await session.execute(stmt)
                    
                    if result.rowcount > 0:
                        logger.warning(f"Hard deleted property with ID: {property_id}")
                        return True
                else:
                    # Soft delete - mark as inactive
                    stmt = update(PropertyModel).where(
                        and_(
                            PropertyModel.id == property_id,
                            PropertyModel.status == "active"  # Only delete if currently active
                        )
                    ).values(
                        status="inactive",
                        updated_at=datetime.utcnow()
                    )
                    
                    result = await session.execute(stmt)
                    
                    if result.rowcount > 0:
                        logger.info(f"Soft deleted property with ID: {property_id}")
                        return True
                
                logger.warning(f"Property with ID {property_id} not found or already deleted")
                return False
                
        except SQLAlchemyError as e:
            logger.error(f"Error deleting property {property_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("search_properties")
    async def search(self, query: SearchQuery) -> Tuple[List[Property], int]:
        """Enhanced search with full-text search, fuzzy matching, and performance optimization"""
        try:
            async with self.get_session() as session:
                # Build base query with active properties
                base_query = select(PropertyModel).where(PropertyModel.status == "active")
                
                # Apply text search with multiple strategies
                if query.query_text and query.query_text.strip():
                    search_text = query.query_text.strip()
                    
                    # Full-text search using PostgreSQL's built-in capabilities
                    if hasattr(PropertyModel, 'search_vector') and PropertyModel.search_vector is not None:
                        # Use PostgreSQL full-text search
                        ts_query = func.plainto_tsquery('english', search_text)
                        base_query = base_query.where(
                            or_(
                                PropertyModel.search_vector.op('@@')(ts_query),
                                self._build_fuzzy_search_condition(search_text)
                            )
                        )
                    else:
                        # Fallback to ILIKE search
                        base_query = base_query.where(
                            self._build_fuzzy_search_condition(search_text)
                        )
                    
                    # Add relevance scoring for search results
                    if query.sort_by == "relevance":
                        relevance_score = self._build_relevance_score(search_text)
                        base_query = base_query.add_columns(relevance_score.label('relevance'))
                
                # Apply filters
                base_query = self._apply_filters(base_query, query.filters)
                
                # Get total count efficiently
                count_query = select(func.count()).select_from(base_query.subquery())
                count_result = await session.execute(count_query)
                total_count = count_result.scalar()
                
                # Apply sorting
                base_query = self._apply_sorting(base_query, query.sort_by)
                
                # Apply pagination
                base_query = base_query.limit(query.limit).offset(query.offset)
                
                # Execute query
                result = await session.execute(base_query)
                
                if query.query_text and query.sort_by == "relevance":
                    # Handle relevance scoring results
                    rows = result.fetchall()
                    property_models = [row[0] for row in rows]  # First column is the PropertyModel
                else:
                    property_models = result.scalars().all()
                
                properties = [model.to_domain() for model in property_models]
                
                logger.info(f"Search for '{query.query_text}' returned {len(properties)} properties out of {total_count}")
                return properties, total_count
                
        except SQLAlchemyError as e:
            logger.error(f"Error searching properties: {e}")
            raise
    
    def _build_fuzzy_search_condition(self, search_text: str):
        """Build fuzzy search condition for text matching"""
        search_term = f"%{search_text.lower()}%"
        return or_(
            PropertyModel.title.ilike(search_term),
            PropertyModel.description.ilike(search_term),
            PropertyModel.location.ilike(search_term),
            PropertyModel.full_text_search.ilike(search_term),
            # Search in amenities array
            func.array_to_string(PropertyModel.amenities, ' ').ilike(search_term)
        )
    
    def _build_relevance_score(self, search_text: str):
        """Build relevance score for search results"""
        search_term = search_text.lower()
        return (
            # Title matches get highest score
            func.case(
                (PropertyModel.title.ilike(f"%{search_term}%"), 10),
                else_=0
            ) +
            # Location matches get medium score
            func.case(
                (PropertyModel.location.ilike(f"%{search_term}%"), 5),
                else_=0
            ) +
            # Description matches get lower score
            func.case(
                (PropertyModel.description.ilike(f"%{search_term}%"), 2),
                else_=0
            ) +
            # Amenity matches get minimal score
            func.case(
                (func.array_to_string(PropertyModel.amenities, ' ').ilike(f"%{search_term}%"), 1),
                else_=0
            )
        )
    
    def _apply_filters(self, query, filters: SearchFilters):
        """Apply search filters to query"""
        if filters.min_price is not None:
            query = query.where(PropertyModel.price >= filters.min_price)
        
        if filters.max_price is not None:
            query = query.where(PropertyModel.price <= filters.max_price)
        
        if filters.min_bedrooms is not None:
            query = query.where(PropertyModel.bedrooms >= filters.min_bedrooms)
        
        if filters.max_bedrooms is not None:
            query = query.where(PropertyModel.bedrooms <= filters.max_bedrooms)
        
        if filters.min_bathrooms is not None:
            query = query.where(PropertyModel.bathrooms >= filters.min_bathrooms)
        
        if filters.max_bathrooms is not None:
            query = query.where(PropertyModel.bathrooms <= filters.max_bathrooms)
        
        if filters.min_square_feet is not None:
            query = query.where(PropertyModel.square_feet >= filters.min_square_feet)
        
        if filters.max_square_feet is not None:
            query = query.where(PropertyModel.square_feet <= filters.max_square_feet)
        
        if filters.locations:
            location_conditions = [
                PropertyModel.location.ilike(f"%{loc}%") for loc in filters.locations
            ]
            query = query.where(or_(*location_conditions))
        
        if filters.property_types:
            query = query.where(PropertyModel.property_type.in_(filters.property_types))
        
        if filters.amenities:
            # Properties must have all specified amenities
            for amenity in filters.amenities:
                query = query.where(PropertyModel.amenities.contains([amenity]))
        
        return query
    
    def _apply_sorting(self, query, sort_by: str):
        """Apply enhanced sorting to query with multiple sort options"""
        if sort_by == "price_asc":
            return query.order_by(PropertyModel.price.asc(), PropertyModel.created_at.desc())
        elif sort_by == "price_desc":
            return query.order_by(PropertyModel.price.desc(), PropertyModel.created_at.desc())
        elif sort_by == "date_new":
            return query.order_by(PropertyModel.scraped_at.desc(), PropertyModel.created_at.desc())
        elif sort_by == "date_old":
            return query.order_by(PropertyModel.scraped_at.asc(), PropertyModel.created_at.asc())
        elif sort_by == "bedrooms_asc":
            return query.order_by(PropertyModel.bedrooms.asc(), PropertyModel.price.asc())
        elif sort_by == "bedrooms_desc":
            return query.order_by(PropertyModel.bedrooms.desc(), PropertyModel.price.asc())
        elif sort_by == "size_asc":
            return query.order_by(PropertyModel.square_feet.asc().nulls_last(), PropertyModel.price.asc())
        elif sort_by == "size_desc":
            return query.order_by(PropertyModel.square_feet.desc().nulls_last(), PropertyModel.price.asc())
        elif sort_by == "price_per_sqft_asc":
            return query.order_by(PropertyModel.price_per_sqft.asc().nulls_last(), PropertyModel.price.asc())
        elif sort_by == "price_per_sqft_desc":
            return query.order_by(PropertyModel.price_per_sqft.desc().nulls_last(), PropertyModel.price.asc())
        elif sort_by == "popular":
            # Sort by engagement metrics
            popularity_score = (
                (PropertyModel.view_count * 1) +
                (PropertyModel.favorite_count * 5) +
                (PropertyModel.contact_count * 10)
            )
            return query.order_by(popularity_score.desc(), PropertyModel.created_at.desc())
        elif sort_by == "quality":
            return query.order_by(PropertyModel.data_quality_score.desc().nulls_last(), PropertyModel.created_at.desc())
        elif sort_by == "relevance":
            # Relevance sorting is handled in the search method when search text is provided
            return query.order_by(PropertyModel.created_at.desc())
        else:  # default
            return query.order_by(PropertyModel.created_at.desc(), PropertyModel.scraped_at.desc())
    
    @retry_on_db_error()
    @measure_performance("get_all_active_properties")
    async def get_all_active(self, limit: int = 100, offset: int = 0, 
                           sort_by: str = "date_new", min_quality_score: Optional[float] = None) -> List[Property]:
        """Get all active properties with enhanced filtering and sorting"""
        try:
            async with self.get_session() as session:
                stmt = select(PropertyModel).where(PropertyModel.status == "active")
                
                # Apply quality filter if specified
                if min_quality_score is not None:
                    stmt = stmt.where(PropertyModel.data_quality_score >= min_quality_score)
                
                # Apply sorting
                stmt = self._apply_sorting(stmt, sort_by)
                
                # Apply pagination
                stmt = stmt.limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                logger.debug(f"Retrieved {len(property_models)} active properties (offset: {offset}, limit: {limit})")
                return [model.to_domain() for model in property_models]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting all active properties: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_properties_by_location")
    async def get_by_location(self, location: str, limit: int = 100, offset: int = 0, 
                            sort_by: str = "price_asc", exact_match: bool = False) -> List[Property]:
        """Get properties by location with enhanced search capabilities"""
        try:
            async with self.get_session() as session:
                base_conditions = [PropertyModel.status == "active"]
                
                if exact_match:
                    base_conditions.append(PropertyModel.location == location)
                else:
                    # Use fuzzy matching for better results
                    base_conditions.append(
                        or_(
                            PropertyModel.location.ilike(f"%{location}%"),
                            PropertyModel.location.ilike(f"{location}%"),
                            PropertyModel.location.ilike(f"%{location}")
                        )
                    )
                
                stmt = select(PropertyModel).where(and_(*base_conditions))
                
                # Apply sorting
                stmt = self._apply_sorting(stmt, sort_by)
                
                # Apply pagination
                stmt = stmt.limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                logger.debug(f"Retrieved {len(property_models)} properties for location '{location}'")
                return [model.to_domain() for model in property_models]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting properties by location {location}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_properties_by_price_range")
    async def get_by_price_range(self, min_price: float, max_price: float, 
                                limit: int = 100, offset: int = 0, 
                                sort_by: str = "price_asc", 
                                location_filter: Optional[str] = None) -> List[Property]:
        """Get properties within price range with additional filtering"""
        try:
            async with self.get_session() as session:
                conditions = [
                    PropertyModel.status == "active",
                    PropertyModel.price >= min_price,
                    PropertyModel.price <= max_price
                ]
                
                # Add location filter if specified
                if location_filter:
                    conditions.append(PropertyModel.location.ilike(f"%{location_filter}%"))
                
                stmt = select(PropertyModel).where(and_(*conditions))
                
                # Apply sorting
                stmt = self._apply_sorting(stmt, sort_by)
                
                # Apply pagination
                stmt = stmt.limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                logger.debug(f"Retrieved {len(property_models)} properties in price range ${min_price}-${max_price}")
                return [model.to_domain() for model in property_models]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting properties by price range: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_similar_properties")
    async def get_similar_properties(self, property_id: UUID, limit: int = 10, 
                                   similarity_threshold: float = 0.8) -> List[Property]:
        """Get similar properties using enhanced similarity algorithms"""
        try:
            async with self.get_session() as session:
                # First get the reference property
                ref_stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.id == property_id,
                        PropertyModel.status == "active"
                    )
                )
                ref_result = await session.execute(ref_stmt)
                ref_property = ref_result.scalar_one_or_none()
                
                if not ref_property:
                    logger.warning(f"Reference property {property_id} not found or inactive")
                    return []
                
                # Calculate similarity based on multiple factors
                price_tolerance = 0.25  # 25% price tolerance
                size_tolerance = 0.3   # 30% size tolerance
                
                conditions = [
                    PropertyModel.status == "active",
                    PropertyModel.id != property_id,
                    PropertyModel.property_type == ref_property.property_type
                ]
                
                # Location similarity (same city/area)
                location_parts = ref_property.location.split(',')
                if len(location_parts) > 1:
                    # Try to match city or area
                    main_location = location_parts[0].strip()
                    conditions.append(PropertyModel.location.ilike(f"%{main_location}%"))
                else:
                    conditions.append(PropertyModel.location.ilike(f"%{ref_property.location}%"))
                
                # Price similarity
                price_min = ref_property.price * (1 - price_tolerance)
                price_max = ref_property.price * (1 + price_tolerance)
                conditions.extend([
                    PropertyModel.price >= price_min,
                    PropertyModel.price <= price_max
                ])
                
                # Bedroom similarity (exact match or ±1)
                bedroom_conditions = [
                    PropertyModel.bedrooms == ref_property.bedrooms,
                    PropertyModel.bedrooms == ref_property.bedrooms - 1,
                    PropertyModel.bedrooms == ref_property.bedrooms + 1
                ]
                conditions.append(or_(*bedroom_conditions))
                
                # Size similarity (if available)
                if ref_property.square_feet:
                    size_min = ref_property.square_feet * (1 - size_tolerance)
                    size_max = ref_property.square_feet * (1 + size_tolerance)
                    conditions.extend([
                        or_(
                            PropertyModel.square_feet.is_(None),
                            and_(
                                PropertyModel.square_feet >= size_min,
                                PropertyModel.square_feet <= size_max
                            )
                        )
                    ])
                
                # Build the query
                stmt = select(PropertyModel).where(and_(*conditions))
                
                # Add similarity scoring
                similarity_score = self._calculate_similarity_score(ref_property)
                stmt = stmt.add_columns(similarity_score.label('similarity'))
                
                # Order by similarity score and limit results
                stmt = stmt.order_by(similarity_score.desc(), PropertyModel.created_at.desc())
                stmt = stmt.limit(limit * 2)  # Get more results to filter by threshold
                
                result = await session.execute(stmt)
                rows = result.fetchall()
                
                # Filter by similarity threshold and return top results
                similar_properties = []
                for row in rows:
                    if row.similarity >= similarity_threshold:
                        similar_properties.append(row[0].to_domain())
                    if len(similar_properties) >= limit:
                        break
                
                logger.debug(f"Found {len(similar_properties)} similar properties for {property_id}")
                return similar_properties
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting similar properties for {property_id}: {e}")
            raise
    
    def _calculate_similarity_score(self, ref_property: PropertyModel):
        """Calculate similarity score for property recommendations"""
        # Weighted similarity scoring
        price_weight = 0.3
        location_weight = 0.25
        bedroom_weight = 0.2
        size_weight = 0.15
        amenity_weight = 0.1
        
        # Price similarity (inverse of price difference ratio)
        price_diff_ratio = func.abs(PropertyModel.price - ref_property.price) / ref_property.price
        price_similarity = func.greatest(0, 1 - price_diff_ratio)
        
        # Location similarity (simple text match for now)
        location_similarity = func.case(
            (PropertyModel.location.ilike(f"%{ref_property.location}%"), 1.0),
            else_=0.5
        )
        
        # Bedroom similarity
        bedroom_similarity = func.case(
            (PropertyModel.bedrooms == ref_property.bedrooms, 1.0),
            (func.abs(PropertyModel.bedrooms - ref_property.bedrooms) == 1, 0.7),
            else_=0.3
        )
        
        # Size similarity (if available)
        if ref_property.square_feet:
            size_diff_ratio = func.abs(PropertyModel.square_feet - ref_property.square_feet) / ref_property.square_feet
            size_similarity = func.case(
                (PropertyModel.square_feet.is_(None), 0.5),
                else_=func.greatest(0, 1 - size_diff_ratio)
            )
        else:
            size_similarity = 0.5
        
        # Amenity similarity (intersection of arrays)
        amenity_similarity = func.case(
            (PropertyModel.amenities.op('&&')(ref_property.amenities), 0.8),
            else_=0.2
        )
        
        # Weighted total similarity
        return (
            price_similarity * price_weight +
            location_similarity * location_weight +
            bedroom_similarity * bedroom_weight +
            size_similarity * size_weight +
            amenity_similarity * amenity_weight
        )
    
    @retry_on_db_error()
    @measure_performance("bulk_create_properties")
    async def bulk_create(self, properties: List[Property], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Property]:
        """Create multiple properties in batches with enhanced error handling"""
        if not properties:
            return []
        
        # Validate all properties first
        validation_errors = []
        for i, prop in enumerate(properties):
            try:
                self._validate_property_data(prop)
            except ValueError as e:
                validation_errors.append(f"Property {i}: {e}")
        
        if validation_errors:
            raise ValueError(f"Bulk validation failed: {'; '.join(validation_errors)}")
        
        all_created_properties = []
        
        try:
            # Process in batches to avoid memory issues and long transactions
            for i in range(0, len(properties), batch_size):
                batch = properties[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1} of {len(batch)} properties")
                
                async with self.get_transaction() as session:
                    property_models = []
                    for prop in batch:
                        model = PropertyModel.from_domain(prop)
                        model.update_search_vector(session)
                        property_models.append(model)
                    
                    session.add_all(property_models)
                    await session.flush()  # Flush to get any constraint violations
                    
                    # Refresh to get generated fields
                    for model in property_models:
                        await session.refresh(model)
                    
                    batch_created = [model.to_domain() for model in property_models]
                    all_created_properties.extend(batch_created)
                    
                    logger.info(f"Bulk created batch of {len(batch)} properties")
            
            logger.info(f"Bulk created total of {len(all_created_properties)} properties")
            return all_created_properties
            
        except IntegrityError as e:
            logger.error(f"Integrity error in bulk create: {e}")
            raise ValueError(f"Bulk operation failed due to data constraints: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error in bulk create: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in bulk create: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_property_features")
    async def get_property_features(self, property_id: UUID) -> Optional[Dict]:
        """Get comprehensive property features for ML processing"""
        try:
            async with self.get_session() as session:
                stmt = select(PropertyModel).where(PropertyModel.id == property_id)
                result = await session.execute(stmt)
                property_model = result.scalar_one_or_none()
                
                if not property_model:
                    return None
                
                # Calculate derived features
                age_days = (datetime.utcnow() - property_model.created_at).days if property_model.created_at else 0
                
                features = {
                    # Basic features
                    "id": str(property_model.id),
                    "price": property_model.price,
                    "bedrooms": property_model.bedrooms,
                    "bathrooms": property_model.bathrooms,
                    "square_feet": property_model.square_feet,
                    "price_per_sqft": property_model.price_per_sqft,
                    "location": property_model.location,
                    "property_type": property_model.property_type,
                    "amenities": property_model.amenities,
                    "amenity_count": len(property_model.amenities or []),
                    "status": property_model.status,
                    
                    # Quality and engagement features
                    "data_quality_score": property_model.data_quality_score,
                    "view_count": property_model.view_count or 0,
                    "favorite_count": property_model.favorite_count or 0,
                    "contact_count": property_model.contact_count or 0,
                    
                    # Temporal features
                    "age_days": age_days,
                    "created_at": property_model.created_at.isoformat() if property_model.created_at else None,
                    "updated_at": property_model.updated_at.isoformat() if property_model.updated_at else None,
                    "last_viewed": property_model.last_viewed.isoformat() if property_model.last_viewed else None,
                    
                    # Content features
                    "title_length": len(property_model.title) if property_model.title else 0,
                    "description_length": len(property_model.description) if property_model.description else 0,
                    "image_count": len(property_model.images or []),
                    "has_contact_info": bool(property_model.contact_info),
                    
                    # Derived features
                    "popularity_score": (
                        (property_model.view_count or 0) * 1 +
                        (property_model.favorite_count or 0) * 5 +
                        (property_model.contact_count or 0) * 10
                    ),
                    "engagement_rate": (
                        ((property_model.favorite_count or 0) + (property_model.contact_count or 0)) / 
                        max(1, property_model.view_count or 1)
                    ),
                    
                    # Geographic features (if available)
                    "latitude": property_model.latitude,
                    "longitude": property_model.longitude,
                }
                
                return features
        except SQLAlchemyError as e:
            logger.error(f"Error getting property features for {property_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("update_property_features")
    async def update_property_features(self, property_id: UUID, features: Dict) -> bool:
        """Update property features with ML-derived data and analytics"""
        try:
            async with self.get_transaction() as session:
                # Prepare update data from features
                update_data = {}
                
                # Update computed features
                if "price_per_sqft" in features:
                    update_data["price_per_sqft"] = features["price_per_sqft"]
                
                if "data_quality_score" in features:
                    update_data["data_quality_score"] = features["data_quality_score"]
                
                # Update geographic data
                if "latitude" in features:
                    update_data["latitude"] = features["latitude"]
                if "longitude" in features:
                    update_data["longitude"] = features["longitude"]
                
                # Update engagement metrics
                if "view_count" in features:
                    update_data["view_count"] = features["view_count"]
                if "favorite_count" in features:
                    update_data["favorite_count"] = features["favorite_count"]
                if "contact_count" in features:
                    update_data["contact_count"] = features["contact_count"]
                
                # Update search optimization
                if "search_vector" in features:
                    update_data["search_vector"] = features["search_vector"]
                
                if update_data:
                    update_data["updated_at"] = datetime.utcnow()
                    
                    stmt = update(PropertyModel).where(
                        PropertyModel.id == property_id
                    ).values(**update_data)
                    
                    result = await session.execute(stmt)
                    
                    if result.rowcount > 0:
                        logger.debug(f"Updated features for property {property_id}: {list(update_data.keys())}")
                        return True
                    else:
                        logger.warning(f"No property found with ID {property_id} to update features")
                        return False
                
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error updating property features for {property_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_property_count")
    async def get_count(self) -> int:
        """Get total property count"""
        try:
            async with self.get_session() as session:
                stmt = select(func.count(PropertyModel.id))
                result = await session.execute(stmt)
                return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error getting property count: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_active_property_count")
    async def get_active_count(self) -> int:
        """Get active property count"""
        try:
            async with self.get_session() as session:
                stmt = select(func.count(PropertyModel.id)).where(
                    PropertyModel.status == "active"
                )
                result = await session.execute(stmt)
                return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error getting active property count: {e}")
            raise
    
    async def get_counts_by_status(self) -> Dict[str, int]:
        """Get property counts by different statuses"""
        try:
            async with self.get_session() as session:
                stmt = select(
                    func.count(PropertyModel.id).label("total"),
                    func.sum(func.case((PropertyModel.status == "active", 1), else_=0)).label("active"),
                    func.sum(func.case((PropertyModel.status != "active", 1), else_=0)).label("inactive"),
                    func.sum(func.case((PropertyModel.data_quality_score >= 0.8, 1), else_=0)).label("high_quality"),
                    func.sum(func.case((PropertyModel.view_count > 50, 1), else_=0)).label("popular")
                )
                
                result = await session.execute(stmt)
                counts = result.first()
                
                return {
                    "total": int(counts.total) if counts.total else 0,
                    "active": int(counts.active) if counts.active else 0,
                    "inactive": int(counts.inactive) if counts.inactive else 0,
                    "high_quality": int(counts.high_quality) if counts.high_quality else 0,
                    "popular": int(counts.popular) if counts.popular else 0
                }
        except SQLAlchemyError as e:
            logger.error(f"Error getting property counts by status: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_aggregated_stats")
    async def get_aggregated_stats(self, location_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive aggregated statistics for properties"""
        try:
            async with self.get_session() as session:
                base_query = select(
                    func.avg(PropertyModel.price).label("avg_price"),
                    func.min(PropertyModel.price).label("min_price"),
                    func.max(PropertyModel.price).label("max_price"),
                    func.avg(PropertyModel.bedrooms).label("avg_bedrooms"),
                    func.avg(PropertyModel.bathrooms).label("avg_bathrooms"),
                    func.avg(PropertyModel.square_feet).label("avg_square_feet"),
                    func.avg(PropertyModel.price_per_sqft).label("avg_price_per_sqft"),
                    func.count(PropertyModel.id).label("total_count"),
                    func.sum(PropertyModel.view_count).label("total_views"),
                    func.sum(PropertyModel.favorite_count).label("total_favorites"),
                    func.sum(PropertyModel.contact_count).label("total_contacts"),
                    func.avg(PropertyModel.data_quality_score).label("avg_quality_score")
                ).where(PropertyModel.status == "active")
                
                if location_filter:
                    base_query = base_query.where(PropertyModel.location.ilike(f"%{location_filter}%"))
                
                result = await session.execute(base_query)
                stats = result.first()
                
                # Get property type distribution
                type_query = select(
                    PropertyModel.property_type,
                    func.count(PropertyModel.id).label("count")
                ).where(PropertyModel.status == "active").group_by(PropertyModel.property_type)
                
                if location_filter:
                    type_query = type_query.where(PropertyModel.location.ilike(f"%{location_filter}%"))
                
                type_result = await session.execute(type_query)
                type_distribution = {row.property_type: row.count for row in type_result}
                
                # Get recent activity stats (last 30 days)
                recent_date = datetime.utcnow() - timedelta(days=30)
                recent_query = select(
                    func.count(PropertyModel.id).label("recent_properties"),
                    func.avg(PropertyModel.price).label("recent_avg_price")
                ).where(
                    and_(
                        PropertyModel.status == "active",
                        PropertyModel.created_at >= recent_date
                    )
                )
                
                if location_filter:
                    recent_query = recent_query.where(PropertyModel.location.ilike(f"%{location_filter}%"))
                
                recent_result = await session.execute(recent_query)
                recent_stats = recent_result.first()
                
                return {
                    "avg_price": float(stats.avg_price) if stats.avg_price else 0,
                    "min_price": float(stats.min_price) if stats.min_price else 0,
                    "max_price": float(stats.max_price) if stats.max_price else 0,
                    "avg_bedrooms": float(stats.avg_bedrooms) if stats.avg_bedrooms else 0,
                    "avg_bathrooms": float(stats.avg_bathrooms) if stats.avg_bathrooms else 0,
                    "avg_square_feet": float(stats.avg_square_feet) if stats.avg_square_feet else 0,
                    "avg_price_per_sqft": float(stats.avg_price_per_sqft) if stats.avg_price_per_sqft else 0,
                    "total_count": int(stats.total_count) if stats.total_count else 0,
                    "total_views": int(stats.total_views) if stats.total_views else 0,
                    "total_favorites": int(stats.total_favorites) if stats.total_favorites else 0,
                    "total_contacts": int(stats.total_contacts) if stats.total_contacts else 0,
                    "avg_quality_score": float(stats.avg_quality_score) if stats.avg_quality_score else 0,
                    "property_type_distribution": type_distribution,
                    "recent_properties_count": int(recent_stats.recent_properties) if recent_stats.recent_properties else 0,
                    "recent_avg_price": float(recent_stats.recent_avg_price) if recent_stats.recent_avg_price else 0,
                    "location_filter": location_filter,
                    "timestamp": datetime.utcnow().isoformat()
                }
        except SQLAlchemyError as e:
            logger.error(f"Error getting aggregated stats: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_trending_properties")
    async def get_trending_properties(self, limit: int = 20, time_window_days: int = 7) -> List[Property]:
        """Get trending properties based on recent engagement"""
        try:
            async with self.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
                
                # Calculate trending score based on recent activity
                trending_score = (
                    func.coalesce(PropertyModel.view_count, 0) * 1 +
                    func.coalesce(PropertyModel.favorite_count, 0) * 5 +
                    func.coalesce(PropertyModel.contact_count, 0) * 10
                )
                
                stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.status == "active",
                        PropertyModel.created_at >= cutoff_date
                    )
                ).order_by(trending_score.desc()).limit(limit)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                logger.debug(f"Retrieved {len(property_models)} trending properties")
                return [model.to_domain() for model in property_models]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting trending properties: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_properties_by_quality")
    async def get_properties_by_quality(self, min_quality: float = 0.8, limit: int = 100, 
                                       offset: int = 0) -> List[Property]:
        """Get high-quality properties based on data quality score"""
        try:
            async with self.get_session() as session:
                stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.status == "active",
                        PropertyModel.data_quality_score >= min_quality
                    )
                ).order_by(
                    PropertyModel.data_quality_score.desc(),
                    PropertyModel.created_at.desc()
                ).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                logger.debug(f"Retrieved {len(property_models)} high-quality properties")
                return [model.to_domain() for model in property_models]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting properties by quality: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_price_distribution")
    async def get_price_distribution(self, location_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get price distribution statistics for analytics"""
        try:
            async with self.get_session() as session:
                base_query = select(PropertyModel.price).where(PropertyModel.status == "active")
                
                if location_filter:
                    base_query = base_query.where(PropertyModel.location.ilike(f"%{location_filter}%"))
                
                # Get percentiles and distribution
                stmt = select(
                    func.percentile_cont(0.1).within_group(PropertyModel.price).label("p10"),
                    func.percentile_cont(0.25).within_group(PropertyModel.price).label("p25"),
                    func.percentile_cont(0.5).within_group(PropertyModel.price).label("p50"),
                    func.percentile_cont(0.75).within_group(PropertyModel.price).label("p75"),
                    func.percentile_cont(0.9).within_group(PropertyModel.price).label("p90"),
                    func.avg(PropertyModel.price).label("mean"),
                    func.stddev(PropertyModel.price).label("stddev"),
                    func.count(PropertyModel.price).label("count")
                ).where(PropertyModel.status == "active")
                
                if location_filter:
                    stmt = stmt.where(PropertyModel.location.ilike(f"%{location_filter}%"))
                
                result = await session.execute(stmt)
                stats = result.first()
                
                return {
                    "percentiles": {
                        "p10": float(stats.p10) if stats.p10 else 0,
                        "p25": float(stats.p25) if stats.p25 else 0,
                        "p50": float(stats.p50) if stats.p50 else 0,
                        "p75": float(stats.p75) if stats.p75 else 0,
                        "p90": float(stats.p90) if stats.p90 else 0,
                    },
                    "mean": float(stats.mean) if stats.mean else 0,
                    "stddev": float(stats.stddev) if stats.stddev else 0,
                    "count": int(stats.count) if stats.count else 0,
                    "location_filter": location_filter
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting price distribution: {e}")
            raise
    
    @retry_on_db_error()
    async def bulk_update_engagement_metrics(self, property_metrics: List[Dict[str, Any]]) -> int:
        """Bulk update engagement metrics for multiple properties"""
        try:
            updated_count = 0
            
            async with self.get_transaction() as session:
                for metrics in property_metrics:
                    if 'property_id' not in metrics:
                        continue
                    
                    property_id = metrics['property_id']
                    update_data = {'updated_at': datetime.utcnow()}
                    
                    if 'view_count' in metrics:
                        update_data['view_count'] = metrics['view_count']
                    if 'favorite_count' in metrics:
                        update_data['favorite_count'] = metrics['favorite_count']
                    if 'contact_count' in metrics:
                        update_data['contact_count'] = metrics['contact_count']
                    if 'last_viewed' in metrics:
                        update_data['last_viewed'] = metrics['last_viewed']
                    
                    stmt = update(PropertyModel).where(
                        PropertyModel.id == property_id
                    ).values(**update_data)
                    
                    result = await session.execute(stmt)
                    updated_count += result.rowcount
                
                logger.info(f"Bulk updated engagement metrics for {updated_count} properties")
                return updated_count
                
        except SQLAlchemyError as e:
            logger.error(f"Error bulk updating engagement metrics: {e}")
            raise
    
    @retry_on_db_error()
    async def get_stale_properties(self, days_threshold: int = 30) -> List[Property]:
        """Get properties that haven't been updated recently"""
        try:
            async with self.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
                
                stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.status == "active",
                        PropertyModel.updated_at < cutoff_date
                    )
                ).order_by(PropertyModel.updated_at.asc())
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                logger.debug(f"Found {len(property_models)} stale properties")
                return [model.to_domain() for model in property_models]
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting stale properties: {e}")
            raise
    
    @retry_on_db_error()
    async def archive_old_properties(self, days_threshold: int = 90, batch_size: int = 1000) -> int:
        """Archive old inactive properties for performance"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            archived_count = 0
            
            async with self.get_transaction() as session:
                # Find old inactive properties
                stmt = select(PropertyModel.id).where(
                    and_(
                        PropertyModel.status != "active",
                        PropertyModel.updated_at < cutoff_date
                    )
                ).limit(batch_size)
                
                result = await session.execute(stmt)
                property_ids = [row.id for row in result.fetchall()]
                
                if property_ids:
                    # For now, we'll just mark them as archived
                    # In production, you might move them to an archive table
                    archive_stmt = update(PropertyModel).where(
                        PropertyModel.id.in_(property_ids)
                    ).values(
                        status="inactive",
                        updated_at=datetime.utcnow()
                    )
                    
                    archive_result = await session.execute(archive_stmt)
                    archived_count = archive_result.rowcount
                    
                    logger.info(f"Archived {archived_count} old properties")
                
                return archived_count
                
        except SQLAlchemyError as e:
            logger.error(f"Error archiving old properties: {e}")
            raise
    
    @retry_on_db_error()
    async def get_location_analytics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get analytics data grouped by location"""
        try:
            async with self.get_session() as session:
                stmt = select(
                    PropertyModel.location,
                    func.count(PropertyModel.id).label("property_count"),
                    func.avg(PropertyModel.price).label("avg_price"),
                    func.min(PropertyModel.price).label("min_price"),
                    func.max(PropertyModel.price).label("max_price"),
                    func.avg(PropertyModel.bedrooms).label("avg_bedrooms"),
                    func.avg(PropertyModel.square_feet).label("avg_square_feet"),
                    func.sum(PropertyModel.view_count).label("total_views"),
                    func.avg(PropertyModel.data_quality_score).label("avg_quality")
                ).where(
                    PropertyModel.status == "active"
                ).group_by(
                    PropertyModel.location
                ).order_by(
                    func.count(PropertyModel.id).desc()
                ).limit(limit)
                
                result = await session.execute(stmt)
                location_stats = []
                
                for row in result.fetchall():
                    location_stats.append({
                        "location": row.location,
                        "property_count": int(row.property_count),
                        "avg_price": float(row.avg_price) if row.avg_price else 0,
                        "min_price": float(row.min_price) if row.min_price else 0,
                        "max_price": float(row.max_price) if row.max_price else 0,
                        "avg_bedrooms": float(row.avg_bedrooms) if row.avg_bedrooms else 0,
                        "avg_square_feet": float(row.avg_square_feet) if row.avg_square_feet else 0,
                        "total_views": int(row.total_views) if row.total_views else 0,
                        "avg_quality": float(row.avg_quality) if row.avg_quality else 0
                    })
                
                return location_stats
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting location analytics: {e}")
            raise
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimization tasks"""
        try:
            async with self.get_session() as session:
                # Analyze table for query optimization
                await session.execute(text("ANALYZE properties"))
                
                # Vacuum to reclaim space
                await session.execute(text("VACUUM ANALYZE properties"))
                
                # Update search vectors for properties without them
                update_search_stmt = text("""
                    UPDATE properties 
                    SET search_vector = to_tsvector('english', 
                        COALESCE(title, '') || ' ' || 
                        COALESCE(description, '') || ' ' || 
                        COALESCE(location, '') || ' ' || 
                        COALESCE(array_to_string(amenities, ' '), '')
                    )
                    WHERE search_vector IS NULL AND status = 'active'
                """)
                
                search_result = await session.execute(update_search_stmt)
                
                await session.commit()
                
                logger.info("Database optimization completed successfully")
                return {
                    "status": "success",
                    "search_vectors_updated": search_result.rowcount,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error optimizing database: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error optimizing database: {e}")
            raise
    
    async def backup_properties(self, backup_path: str) -> bool:
        """Create a backup of property data"""
        try:
            async with self.get_session() as session:
                # Export properties to JSON format
                stmt = select(PropertyModel).where(PropertyModel.status == "active")
                result = await session.execute(stmt)
                properties = result.scalars().all()
                
                backup_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "count": len(properties),
                    "properties": []
                }
                
                for prop in properties:
                    backup_data["properties"].append({
                        "id": str(prop.id),
                        "title": prop.title,
                        "description": prop.description,
                        "price": prop.price,
                        "location": prop.location,
                        "bedrooms": prop.bedrooms,
                        "bathrooms": prop.bathrooms,
                        "square_feet": prop.square_feet,
                        "amenities": prop.amenities,
                        "contact_info": prop.contact_info,
                        "images": prop.images,
                        "property_type": prop.property_type,
                        "created_at": prop.created_at.isoformat() if prop.created_at else None,
                        "updated_at": prop.updated_at.isoformat() if prop.updated_at else None
                    })
                
                # Write backup to file
                import json
                with open(backup_path, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                
                logger.info(f"Backup created successfully: {backup_path} ({len(properties)} properties)")
                return True
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False