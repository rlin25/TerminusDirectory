import asyncio
import logging
from typing import List, Optional, Dict, Tuple, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, DateTime, 
    Text, ARRAY, JSON, and_, or_, func, select, update, delete, insert
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.exc import SQLAlchemyError

from ....domain.entities.property import Property
from ....domain.entities.search_query import SearchQuery, SearchFilters
from ....domain.repositories.property_repository import PropertyRepository

# Configure logging
logger = logging.getLogger(__name__)

# Database model
Base = declarative_base()

class PropertyModel(Base):
    __tablename__ = "properties"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)
    location = Column(String(255), nullable=False)
    bedrooms = Column(Integer, nullable=False)
    bathrooms = Column(Float, nullable=False)
    square_feet = Column(Integer, nullable=True)
    amenities = Column(ARRAY(String), nullable=False, default=[])
    contact_info = Column(JSON, nullable=False, default={})
    images = Column(ARRAY(String), nullable=False, default=[])
    scraped_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    property_type = Column(String(50), nullable=False, default="apartment")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Search optimization columns
    full_text_search = Column(Text, nullable=True)  # For full-text search
    price_per_sqft = Column(Float, nullable=True)  # Computed column
    
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
            is_active=self.is_active,
            property_type=self.property_type
        )
    
    @classmethod
    def from_domain(cls, property: Property) -> 'PropertyModel':
        """Convert domain entity to database model"""
        full_text = f"{property.title} {property.description} {property.location} {' '.join(property.amenities)}"
        price_per_sqft = property.get_price_per_sqft()
        
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
            is_active=property.is_active,
            property_type=property.property_type,
            full_text_search=full_text,
            price_per_sqft=price_per_sqft
        )


class PostgresPropertyRepository(PropertyRepository):
    """PostgreSQL-based property repository implementation"""
    
    def __init__(self, database_url: str, pool_size: int = 10, max_overflow: int = 20):
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
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
    
    async def create(self, property: Property) -> Property:
        """Create a new property"""
        try:
            async with self.async_session_factory() as session:
                property_model = PropertyModel.from_domain(property)
                session.add(property_model)
                await session.commit()
                await session.refresh(property_model)
                
                logger.info(f"Created property with ID: {property.id}")
                return property_model.to_domain()
        except SQLAlchemyError as e:
            logger.error(f"Error creating property: {e}")
            raise
    
    async def get_by_id(self, property_id: UUID) -> Optional[Property]:
        """Get property by ID"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(PropertyModel).where(PropertyModel.id == property_id)
                result = await session.execute(stmt)
                property_model = result.scalar_one_or_none()
                
                if property_model:
                    return property_model.to_domain()
                return None
        except SQLAlchemyError as e:
            logger.error(f"Error getting property by ID {property_id}: {e}")
            raise
    
    async def get_by_ids(self, property_ids: List[UUID]) -> List[Property]:
        """Get multiple properties by IDs"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(PropertyModel).where(PropertyModel.id.in_(property_ids))
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                return [model.to_domain() for model in property_models]
        except SQLAlchemyError as e:
            logger.error(f"Error getting properties by IDs: {e}")
            raise
    
    async def update(self, property: Property) -> Property:
        """Update an existing property"""
        try:
            async with self.async_session_factory() as session:
                property_model = PropertyModel.from_domain(property)
                
                # Update the property
                stmt = update(PropertyModel).where(
                    PropertyModel.id == property.id
                ).values(
                    title=property_model.title,
                    description=property_model.description,
                    price=property_model.price,
                    location=property_model.location,
                    bedrooms=property_model.bedrooms,
                    bathrooms=property_model.bathrooms,
                    square_feet=property_model.square_feet,
                    amenities=property_model.amenities,
                    contact_info=property_model.contact_info,
                    images=property_model.images,
                    is_active=property_model.is_active,
                    property_type=property_model.property_type,
                    full_text_search=property_model.full_text_search,
                    price_per_sqft=property_model.price_per_sqft,
                    updated_at=datetime.utcnow()
                )
                
                await session.execute(stmt)
                await session.commit()
                
                logger.info(f"Updated property with ID: {property.id}")
                return property
        except SQLAlchemyError as e:
            logger.error(f"Error updating property {property.id}: {e}")
            raise
    
    async def delete(self, property_id: UUID) -> bool:
        """Delete a property (soft delete by setting is_active to False)"""
        try:
            async with self.async_session_factory() as session:
                stmt = update(PropertyModel).where(
                    PropertyModel.id == property_id
                ).values(
                    is_active=False,
                    updated_at=datetime.utcnow()
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Soft deleted property with ID: {property_id}")
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"Error deleting property {property_id}: {e}")
            raise
    
    async def search(self, query: SearchQuery) -> Tuple[List[Property], int]:
        """Search properties with complex filtering, sorting, and pagination"""
        try:
            async with self.async_session_factory() as session:
                # Build base query
                base_query = select(PropertyModel).where(PropertyModel.is_active == True)
                
                # Apply text search
                if query.query_text and query.query_text.strip():
                    search_term = f"%{query.query_text.lower()}%"
                    base_query = base_query.where(
                        or_(
                            PropertyModel.full_text_search.ilike(search_term),
                            PropertyModel.title.ilike(search_term),
                            PropertyModel.description.ilike(search_term),
                            PropertyModel.location.ilike(search_term)
                        )
                    )
                
                # Apply filters
                base_query = self._apply_filters(base_query, query.filters)
                
                # Get total count
                count_query = select(func.count()).select_from(base_query.subquery())
                count_result = await session.execute(count_query)
                total_count = count_result.scalar()
                
                # Apply sorting
                base_query = self._apply_sorting(base_query, query.sort_by)
                
                # Apply pagination
                base_query = base_query.limit(query.limit).offset(query.offset)
                
                # Execute query
                result = await session.execute(base_query)
                property_models = result.scalars().all()
                
                properties = [model.to_domain() for model in property_models]
                
                logger.info(f"Search returned {len(properties)} properties out of {total_count}")
                return properties, total_count
        except SQLAlchemyError as e:
            logger.error(f"Error searching properties: {e}")
            raise
    
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
        """Apply sorting to query"""
        if sort_by == "price_asc":
            return query.order_by(PropertyModel.price.asc())
        elif sort_by == "price_desc":
            return query.order_by(PropertyModel.price.desc())
        elif sort_by == "date_new":
            return query.order_by(PropertyModel.scraped_at.desc())
        elif sort_by == "date_old":
            return query.order_by(PropertyModel.scraped_at.asc())
        elif sort_by == "bedrooms_asc":
            return query.order_by(PropertyModel.bedrooms.asc())
        elif sort_by == "bedrooms_desc":
            return query.order_by(PropertyModel.bedrooms.desc())
        elif sort_by == "size_asc":
            return query.order_by(PropertyModel.square_feet.asc().nulls_last())
        elif sort_by == "size_desc":
            return query.order_by(PropertyModel.square_feet.desc().nulls_last())
        else:  # relevance or default
            return query.order_by(PropertyModel.scraped_at.desc())
    
    async def get_all_active(self, limit: int = 100, offset: int = 0) -> List[Property]:
        """Get all active properties with pagination"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(PropertyModel).where(
                    PropertyModel.is_active == True
                ).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                return [model.to_domain() for model in property_models]
        except SQLAlchemyError as e:
            logger.error(f"Error getting all active properties: {e}")
            raise
    
    async def get_by_location(self, location: str, limit: int = 100, offset: int = 0) -> List[Property]:
        """Get properties by location"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.is_active == True,
                        PropertyModel.location.ilike(f"%{location}%")
                    )
                ).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                return [model.to_domain() for model in property_models]
        except SQLAlchemyError as e:
            logger.error(f"Error getting properties by location {location}: {e}")
            raise
    
    async def get_by_price_range(self, min_price: float, max_price: float, 
                                limit: int = 100, offset: int = 0) -> List[Property]:
        """Get properties within price range"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.is_active == True,
                        PropertyModel.price >= min_price,
                        PropertyModel.price <= max_price
                    )
                ).limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                return [model.to_domain() for model in property_models]
        except SQLAlchemyError as e:
            logger.error(f"Error getting properties by price range: {e}")
            raise
    
    async def get_similar_properties(self, property_id: UUID, limit: int = 10) -> List[Property]:
        """Get similar properties based on features"""
        try:
            async with self.async_session_factory() as session:
                # First get the reference property
                ref_stmt = select(PropertyModel).where(PropertyModel.id == property_id)
                ref_result = await session.execute(ref_stmt)
                ref_property = ref_result.scalar_one_or_none()
                
                if not ref_property:
                    return []
                
                # Find similar properties based on location, price range, and size
                price_range = ref_property.price * 0.2  # 20% price tolerance
                
                stmt = select(PropertyModel).where(
                    and_(
                        PropertyModel.is_active == True,
                        PropertyModel.id != property_id,
                        PropertyModel.location.ilike(f"%{ref_property.location}%"),
                        PropertyModel.price >= (ref_property.price - price_range),
                        PropertyModel.price <= (ref_property.price + price_range),
                        PropertyModel.bedrooms == ref_property.bedrooms,
                        PropertyModel.property_type == ref_property.property_type
                    )
                ).limit(limit)
                
                result = await session.execute(stmt)
                property_models = result.scalars().all()
                
                return [model.to_domain() for model in property_models]
        except SQLAlchemyError as e:
            logger.error(f"Error getting similar properties for {property_id}: {e}")
            raise
    
    async def bulk_create(self, properties: List[Property]) -> List[Property]:
        """Create multiple properties in a single transaction"""
        try:
            async with self.async_session_factory() as session:
                property_models = [PropertyModel.from_domain(prop) for prop in properties]
                
                session.add_all(property_models)
                await session.commit()
                
                # Refresh all models to get generated IDs
                for model in property_models:
                    await session.refresh(model)
                
                logger.info(f"Bulk created {len(properties)} properties")
                return [model.to_domain() for model in property_models]
        except SQLAlchemyError as e:
            logger.error(f"Error bulk creating properties: {e}")
            raise
    
    async def get_property_features(self, property_id: UUID) -> Optional[Dict]:
        """Get property features for ML processing"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(PropertyModel).where(PropertyModel.id == property_id)
                result = await session.execute(stmt)
                property_model = result.scalar_one_or_none()
                
                if not property_model:
                    return None
                
                features = {
                    "id": str(property_model.id),
                    "price": property_model.price,
                    "bedrooms": property_model.bedrooms,
                    "bathrooms": property_model.bathrooms,
                    "square_feet": property_model.square_feet,
                    "price_per_sqft": property_model.price_per_sqft,
                    "location": property_model.location,
                    "property_type": property_model.property_type,
                    "amenities": property_model.amenities,
                    "amenity_count": len(property_model.amenities),
                    "is_active": property_model.is_active
                }
                
                return features
        except SQLAlchemyError as e:
            logger.error(f"Error getting property features for {property_id}: {e}")
            raise
    
    async def update_property_features(self, property_id: UUID, features: Dict) -> bool:
        """Update property features (for ML-derived data)"""
        try:
            async with self.async_session_factory() as session:
                # Store ML-derived features in a separate JSON column if needed
                # For now, we'll update the existing fields
                update_data = {}
                
                if "price_per_sqft" in features:
                    update_data["price_per_sqft"] = features["price_per_sqft"]
                
                if update_data:
                    stmt = update(PropertyModel).where(
                        PropertyModel.id == property_id
                    ).values(**update_data, updated_at=datetime.utcnow())
                    
                    result = await session.execute(stmt)
                    await session.commit()
                    
                    return result.rowcount > 0
                
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error updating property features for {property_id}: {e}")
            raise
    
    async def get_count(self) -> int:
        """Get total property count"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(func.count(PropertyModel.id))
                result = await session.execute(stmt)
                return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error getting property count: {e}")
            raise
    
    async def get_active_count(self) -> int:
        """Get active property count"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(func.count(PropertyModel.id)).where(
                    PropertyModel.is_active == True
                )
                result = await session.execute(stmt)
                return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error getting active property count: {e}")
            raise
    
    async def get_aggregated_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for properties"""
        try:
            async with self.async_session_factory() as session:
                stmt = select(
                    func.avg(PropertyModel.price).label("avg_price"),
                    func.min(PropertyModel.price).label("min_price"),
                    func.max(PropertyModel.price).label("max_price"),
                    func.avg(PropertyModel.bedrooms).label("avg_bedrooms"),
                    func.avg(PropertyModel.bathrooms).label("avg_bathrooms"),
                    func.avg(PropertyModel.square_feet).label("avg_square_feet"),
                    func.count(PropertyModel.id).label("total_count")
                ).where(PropertyModel.is_active == True)
                
                result = await session.execute(stmt)
                stats = result.first()
                
                return {
                    "avg_price": float(stats.avg_price) if stats.avg_price else 0,
                    "min_price": float(stats.min_price) if stats.min_price else 0,
                    "max_price": float(stats.max_price) if stats.max_price else 0,
                    "avg_bedrooms": float(stats.avg_bedrooms) if stats.avg_bedrooms else 0,
                    "avg_bathrooms": float(stats.avg_bathrooms) if stats.avg_bathrooms else 0,
                    "avg_square_feet": float(stats.avg_square_feet) if stats.avg_square_feet else 0,
                    "total_count": int(stats.total_count) if stats.total_count else 0
                }
        except SQLAlchemyError as e:
            logger.error(f"Error getting aggregated stats: {e}")
            raise