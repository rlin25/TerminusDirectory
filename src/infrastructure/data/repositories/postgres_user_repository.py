import asyncio
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import asyncpg
from asyncpg import Pool
from contextlib import asynccontextmanager
from functools import wraps
import time

from ....domain.entities.user import User, UserPreferences, UserInteraction
from ....domain.repositories.user_repository import UserRepository


# Performance monitoring decorator
def measure_performance(operation_name: str):
    """Decorator to measure query performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = await func(self, *args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > 1.0:  # Log slow queries
                    self.logger.warning(
                        f"Slow query detected: {operation_name} took {execution_time:.2f}s"
                    )
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"Query failed: {operation_name} took {execution_time:.2f}s, error: {e}"
                )
                raise
        return wrapper
    return decorator


# Retry decorator for database operations
def retry_on_db_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry database operations on transient errors"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (asyncpg.PostgresError, asyncpg.InterfaceError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                except Exception as e:
                    # Non-retryable error
                    raise
            
            raise last_exception
        return wrapper
    return decorator


class PostgreSQLUserRepository(UserRepository):
    """PostgreSQL implementation of UserRepository with enhanced features"""
    
    def __init__(self, connection_pool: Pool):
        self.pool = connection_pool
        self.logger = logging.getLogger(__name__)
        self._connection_timeout = 30.0
        self._query_timeout = 30.0
    
    @asynccontextmanager
    async def get_connection(self):
        """Context manager for database connections with proper error handling"""
        connection = None
        try:
            connection = await asyncio.wait_for(
                self.pool.acquire(),
                timeout=self._connection_timeout
            )
            yield connection
        except asyncio.TimeoutError:
            self.logger.error("Database connection timeout")
            raise
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                await self.pool.release(connection)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection"""
        try:
            async with self.get_connection() as conn:
                start_time = time.time()
                await conn.fetchval("SELECT 1")
                response_time = time.time() - start_time
                
                return {
                    "status": "healthy",
                    "response_time_ms": response_time * 1000,
                    "pool_size": self.pool.get_size(),
                    "pool_free": self.pool.get_idle_size(),
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _validate_user_data(self, user: User):
        """Validate user data before database operations"""
        errors = []
        
        if not user.email or not user.email.strip():
            errors.append("Email is required")
        if "@" not in user.email:
            errors.append("Invalid email format")
        if user.preferences:
            if user.preferences.min_price is not None and user.preferences.min_price < 0:
                errors.append("Minimum price cannot be negative")
            if user.preferences.max_price is not None and user.preferences.max_price < 0:
                errors.append("Maximum price cannot be negative")
            if (user.preferences.min_price is not None and user.preferences.max_price is not None 
                and user.preferences.min_price > user.preferences.max_price):
                errors.append("Minimum price cannot be greater than maximum price")
            if user.preferences.min_bedrooms is not None and user.preferences.min_bedrooms < 0:
                errors.append("Minimum bedrooms cannot be negative")
            if user.preferences.max_bedrooms is not None and user.preferences.max_bedrooms < 0:
                errors.append("Maximum bedrooms cannot be negative")
            if (user.preferences.min_bedrooms is not None and user.preferences.max_bedrooms is not None 
                and user.preferences.min_bedrooms > user.preferences.max_bedrooms):
                errors.append("Minimum bedrooms cannot be greater than maximum bedrooms")
            if user.preferences.min_bathrooms is not None and user.preferences.min_bathrooms < 0:
                errors.append("Minimum bathrooms cannot be negative")
            if user.preferences.max_bathrooms is not None and user.preferences.max_bathrooms < 0:
                errors.append("Maximum bathrooms cannot be negative")
            if (user.preferences.min_bathrooms is not None and user.preferences.max_bathrooms is not None 
                and user.preferences.min_bathrooms > user.preferences.max_bathrooms):
                errors.append("Minimum bathrooms cannot be greater than maximum bathrooms")
            # Validate property types against enum values
            valid_property_types = ['apartment', 'house', 'condo', 'townhouse', 'studio', 'loft']
            if user.preferences.property_types:
                invalid_types = [pt for pt in user.preferences.property_types if pt not in valid_property_types]
                if invalid_types:
                    errors.append(f"Invalid property types: {', '.join(invalid_types)}")
        
        if errors:
            raise ValueError(f"User validation failed: {', '.join(errors)}")
    
    @retry_on_db_error()
    @measure_performance("create_user")
    async def create(self, user: User) -> User:
        """Create a new user with validation and error handling"""
        try:
            self._validate_user_data(user)
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Check if user already exists
                    existing_user = await conn.fetchrow(
                        "SELECT id FROM users WHERE LOWER(email) = LOWER($1)",
                        user.email
                    )
                    
                    if existing_user:
                        raise ValueError(f"User with email {user.email} already exists")
                    
                    # Insert user
                    await conn.execute(
                        """
                        INSERT INTO users (
                            id, email, created_at, status,
                            min_price, max_price, min_bedrooms, max_bedrooms,
                            min_bathrooms, max_bathrooms, preferred_locations,
                            required_amenities, property_types, updated_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                        )
                        """,
                        user.id,
                        user.email,
                        user.created_at,
                        'active' if user.is_active else 'inactive',
                        user.preferences.min_price,
                        user.preferences.max_price,
                        user.preferences.min_bedrooms,
                        user.preferences.max_bedrooms,
                        user.preferences.min_bathrooms,
                        user.preferences.max_bathrooms,
                        user.preferences.preferred_locations,
                        user.preferences.required_amenities,
                        user.preferences.property_types,
                        datetime.utcnow()
                    )
                    
                    # Insert interactions if any
                    if user.interactions:
                        await self._save_user_interactions(conn, user.id, user.interactions)
                    
                    self.logger.info(f"Created user with ID: {user.id}, email: {user.email}")
                    return user
                    
        except ValueError as e:
            self.logger.error(f"Validation error creating user: {e}")
            raise
        except asyncpg.UniqueViolationError as e:
            self.logger.error(f"User already exists: {e}")
            raise ValueError(f"User with email {user.email} already exists")
        except Exception as e:
            self.logger.error(f"Failed to create user {user.id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_by_id")
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID with comprehensive error handling"""
        try:
            async with self.get_connection() as conn:
                # Get user basic info
                user_row = await conn.fetchrow(
                    """
                    SELECT id, email, created_at, status,
                           min_price, max_price, min_bedrooms, max_bedrooms,
                           min_bathrooms, max_bathrooms, preferred_locations,
                           required_amenities, property_types, updated_at
                    FROM users 
                    WHERE id = $1 AND status = 'active'
                    """,
                    user_id
                )
                
                if not user_row:
                    return None
                
                # Get user interactions
                interactions = await self._get_user_interactions(conn, user_id)
                
                # Build user preferences
                preferences = UserPreferences(
                    min_price=user_row['min_price'],
                    max_price=user_row['max_price'],
                    min_bedrooms=user_row['min_bedrooms'],
                    max_bedrooms=user_row['max_bedrooms'],
                    min_bathrooms=user_row['min_bathrooms'],
                    max_bathrooms=user_row['max_bathrooms'],
                    preferred_locations=user_row['preferred_locations'] or [],
                    required_amenities=user_row['required_amenities'] or [],
                    property_types=user_row['property_types'] or ["apartment"]
                )
                
                # Build user object
                user = User(
                    id=user_row['id'],
                    email=user_row['email'],
                    preferences=preferences,
                    interactions=interactions,
                    created_at=user_row['created_at'],
                    is_active=(user_row['status'] == 'active')
                )
                
                return user
                
        except Exception as e:
            self.logger.error(f"Failed to get user {user_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_by_email")
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address with validation"""
        if not email or not email.strip():
            raise ValueError("Email is required")
        
        try:
            async with self.get_connection() as conn:
                user_row = await conn.fetchrow(
                    "SELECT id FROM users WHERE LOWER(email) = LOWER($1) AND status = 'active'",
                    email.strip()
                )
                
                if user_row:
                    return await self.get_by_id(user_row['id'])
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get user by email {email}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("update_user")
    async def update(self, user: User) -> User:
        """Update an existing user with validation and error handling"""
        try:
            self._validate_user_data(user)
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Check if user exists
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)",
                        user.id
                    )
                    
                    if not exists:
                        raise ValueError(f"User with ID {user.id} not found")
                    
                    # Update existing user
                    result = await conn.execute(
                        """
                        UPDATE users SET
                            email = $2,
                            status = $3,
                            min_price = $4,
                            max_price = $5,
                            min_bedrooms = $6,
                            max_bedrooms = $7,
                            min_bathrooms = $8,
                            max_bathrooms = $9,
                            preferred_locations = $10,
                            required_amenities = $11,
                            property_types = $12,
                            updated_at = $13
                        WHERE id = $1
                        """,
                        user.id,
                        user.email,
                        'active' if user.is_active else 'inactive',
                        user.preferences.min_price,
                        user.preferences.max_price,
                        user.preferences.min_bedrooms,
                        user.preferences.max_bedrooms,
                        user.preferences.min_bathrooms,
                        user.preferences.max_bathrooms,
                        user.preferences.preferred_locations,
                        user.preferences.required_amenities,
                        user.preferences.property_types,
                        datetime.utcnow()
                    )
                    
                    if result == "UPDATE 0":
                        raise ValueError(f"Failed to update user {user.id}")
                    
                    # Update interactions
                    await self._save_user_interactions(conn, user.id, user.interactions)
                    
                    self.logger.info(f"Updated user with ID: {user.id}")
                    return user
                    
        except ValueError as e:
            self.logger.error(f"Validation error updating user: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to update user {user.id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("delete_user")
    async def delete(self, user_id: UUID) -> bool:
        """Delete user (soft delete with security considerations)"""
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Check if user exists and is active
                    user_exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1 AND status = 'active')",
                        user_id
                    )
                    
                    if not user_exists:
                        self.logger.warning(f"User {user_id} not found or already deleted")
                        return False
                    
                    # Soft delete the user
                    result = await conn.execute(
                        """
                        UPDATE users SET 
                            status = 'inactive',
                            updated_at = $2
                        WHERE id = $1 AND status = 'active'
                        """,
                        user_id,
                        datetime.utcnow()
                    )
                    
                    if result == "UPDATE 1":
                        self.logger.info(f"Soft deleted user with ID: {user_id}")
                        return True
                    else:
                        self.logger.warning(f"Failed to delete user {user_id}")
                        return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_all_active_users")
    async def get_all_active(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get all active users with pagination"""
        try:
            async with self.get_connection() as conn:
                user_rows = await conn.fetch(
                    """
                    SELECT id, email, created_at, status,
                           min_price, max_price, min_bedrooms, max_bedrooms,
                           min_bathrooms, max_bathrooms, preferred_locations,
                           required_amenities, property_types, updated_at
                    FROM users 
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset
                )
                
                # Use efficient method to get users with interactions
                user_id_list = [row['id'] for row in user_rows]
                users = await self._get_users_by_ids(conn, user_id_list)
                
                return users
                
        except Exception as e:
            self.logger.error(f"Failed to get all active users: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("add_user_interaction")
    async def add_interaction(self, user_id: UUID, interaction: UserInteraction) -> bool:
        """Add user interaction with validation"""
        try:
            # Validate interaction data
            if not interaction.property_id:
                raise ValueError("Property ID is required")
            if not interaction.interaction_type:
                raise ValueError("Interaction type is required")
            valid_interaction_types = ["view", "like", "inquiry", "save", "contact", "favorite"]
            if interaction.interaction_type not in valid_interaction_types:
                raise ValueError(f"Invalid interaction type: {interaction.interaction_type}. Valid types: {', '.join(valid_interaction_types)}")
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Check if user exists
                    user_exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1 AND status = 'active')",
                        user_id
                    )
                    
                    if not user_exists:
                        raise ValueError(f"User {user_id} not found or inactive")
                    
                    # Insert interaction
                    await conn.execute(
                        """
                        INSERT INTO user_interactions (
                            user_id, property_id, interaction_type, 
                            timestamp, duration_seconds
                        ) VALUES ($1, $2, $3, $4, $5)
                        """,
                        user_id,
                        interaction.property_id,
                        interaction.interaction_type,
                        interaction.timestamp,
                        interaction.duration_seconds
                    )
                    
                    self.logger.debug(f"Added {interaction.interaction_type} interaction for user {user_id}")
                    return True
                
        except ValueError as e:
            self.logger.error(f"Validation error adding interaction: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to add interaction for user {user_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_interactions")
    async def get_interactions(self, user_id: UUID, 
                             interaction_type: Optional[str] = None,
                             limit: int = 100, offset: int = 0) -> List[UserInteraction]:
        """Get user interactions with optional filtering and pagination"""
        try:
            async with self.get_connection() as conn:
                return await self._get_user_interactions(
                    conn, user_id, interaction_type, limit, offset
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get interactions for user {user_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_interaction_matrix")
    async def get_user_interaction_matrix(self) -> dict:
        """Get user-property interaction matrix for recommendation algorithms"""
        try:
            async with self.get_connection() as conn:
                # Get all interactions with aggregated data
                interactions = await conn.fetch(
                    """
                    SELECT 
                        user_id,
                        property_id,
                        interaction_type,
                        COUNT(*) as interaction_count,
                        MAX(timestamp) as last_interaction,
                        AVG(duration_seconds) as avg_duration
                    FROM user_interactions ui
                    JOIN users u ON ui.user_id = u.id
                    WHERE u.status = 'active'
                    GROUP BY user_id, property_id, interaction_type
                    ORDER BY user_id, property_id, interaction_count DESC
                    """
                )
                
                # Build interaction matrix
                matrix = {}
                for row in interactions:
                    user_id = str(row['user_id'])
                    property_id = str(row['property_id'])
                    
                    if user_id not in matrix:
                        matrix[user_id] = {}
                    
                    if property_id not in matrix[user_id]:
                        matrix[user_id][property_id] = {}
                    
                    # Weight interactions by type
                    weights = {"view": 1, "like": 3, "save": 2, "inquiry": 5}
                    weight = weights.get(row['interaction_type'], 1)
                    
                    matrix[user_id][property_id][row['interaction_type']] = {
                        'count': row['interaction_count'],
                        'weight': weight * row['interaction_count'],
                        'last_interaction': row['last_interaction'].isoformat() if row['last_interaction'] else None,
                        'avg_duration': float(row['avg_duration']) if row['avg_duration'] else None
                    }
                
                return matrix
                
        except Exception as e:
            self.logger.error(f"Failed to get user interaction matrix: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_users_who_liked_property")
    async def get_users_who_liked_property(self, property_id: UUID) -> List[User]:
        """Get users who liked a specific property"""
        try:
            async with self.get_connection() as conn:
                user_ids = await conn.fetch(
                    """
                    SELECT DISTINCT ui.user_id
                    FROM user_interactions ui
                    JOIN users u ON ui.user_id = u.id
                    WHERE ui.property_id = $1 
                      AND ui.interaction_type = 'like'
                      AND u.status = 'active'
                    ORDER BY ui.user_id
                    """,
                    property_id
                )
                
                # Get user IDs and fetch them efficiently
                user_id_list = [row['user_id'] for row in user_ids]
                users = await self._get_users_by_ids(conn, user_id_list)
                
                return users
                
        except Exception as e:
            self.logger.error(f"Failed to get users who liked property {property_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_similar_users")
    async def get_similar_users(self, user_id: UUID, limit: int = 10) -> List[User]:
        """Find users with similar preferences and interactions using advanced similarity scoring"""
        try:
            async with self.get_connection() as conn:
                # Get target user preferences
                target_user = await self.get_by_id(user_id)
                if not target_user:
                    return []
                
                # Enhanced similarity algorithm with behavioral data
                similar_user_rows = await conn.fetch(
                    """
                    WITH target_prefs AS (
                        SELECT $1 as target_user_id,
                               $2::text[] as target_locations,
                               $3::text[] as target_amenities,
                               $4 as target_min_price,
                               $5 as target_max_price,
                               $6 as target_min_bedrooms,
                               $7 as target_max_bedrooms
                    ),
                    target_interactions AS (
                        SELECT DISTINCT property_id
                        FROM user_interactions
                        WHERE user_id = $1 AND interaction_type IN ('like', 'save')
                    ),
                    user_similarity AS (
                        SELECT 
                            u.id,
                            u.email,
                            -- Preference similarity (weighted)
                            CASE 
                                WHEN u.preferred_locations && tp.target_locations THEN 3
                                ELSE 0
                            END +
                            CASE
                                WHEN u.required_amenities && tp.target_amenities THEN 2
                                ELSE 0
                            END +
                            -- Price range overlap
                            CASE
                                WHEN (u.min_price IS NULL OR tp.target_max_price IS NULL OR u.min_price <= tp.target_max_price)
                                 AND (u.max_price IS NULL OR tp.target_min_price IS NULL OR u.max_price >= tp.target_min_price)
                                THEN 2
                                ELSE 0
                            END +
                            -- Bedroom similarity
                            CASE
                                WHEN (u.min_bedrooms IS NULL OR tp.target_max_bedrooms IS NULL OR u.min_bedrooms <= tp.target_max_bedrooms)
                                 AND (u.max_bedrooms IS NULL OR tp.target_min_bedrooms IS NULL OR u.max_bedrooms >= tp.target_min_bedrooms)
                                THEN 1
                                ELSE 0
                            END +
                            -- Behavioral similarity (liked same properties)
                            COALESCE((
                                SELECT COUNT(*) * 2
                                FROM user_interactions ui
                                JOIN target_interactions ti ON ui.property_id = ti.property_id
                                WHERE ui.user_id = u.id 
                                  AND ui.interaction_type IN ('like', 'save')
                            ), 0) as similarity_score
                        FROM users u
                        CROSS JOIN target_prefs tp
                        WHERE u.id != tp.target_user_id 
                          AND u.status = 'active'
                    )
                    SELECT id, similarity_score
                    FROM user_similarity
                    WHERE similarity_score > 0
                    ORDER BY similarity_score DESC, id
                    LIMIT $8
                    """,
                    user_id,
                    target_user.preferences.preferred_locations,
                    target_user.preferences.required_amenities,
                    target_user.preferences.min_price,
                    target_user.preferences.max_price,
                    target_user.preferences.min_bedrooms,
                    target_user.preferences.max_bedrooms,
                    limit
                )
                
                # Fetch full user objects efficiently
                user_id_list = [row['id'] for row in similar_user_rows]
                similar_users = await self._get_users_by_ids(conn, user_id_list)
                
                return similar_users
                
        except Exception as e:
            self.logger.error(f"Failed to get similar users for {user_id}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_count")
    async def get_count(self) -> int:
        """Get total user count"""
        try:
            async with self.get_connection() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM users"
                )
                return count
        except Exception as e:
            self.logger.error(f"Failed to get user count: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_active_user_count")
    async def get_active_count(self) -> int:
        """Get active user count"""
        try:
            async with self.get_connection() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM users WHERE status = 'active'"
                )
                return count
        except Exception as e:
            self.logger.error(f"Failed to get active user count: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_statistics")
    async def get_user_statistics(self, user_id: UUID) -> Dict[str, Any]:
        """Get comprehensive user activity statistics"""
        try:
            async with self.get_connection() as conn:
                # Get basic interaction statistics
                stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT property_id) as unique_properties,
                        COUNT(*) FILTER (WHERE interaction_type = 'view') as views,
                        COUNT(*) FILTER (WHERE interaction_type = 'like') as likes,
                        COUNT(*) FILTER (WHERE interaction_type = 'inquiry') as inquiries,
                        COUNT(*) FILTER (WHERE interaction_type = 'save') as saves,
                        MAX(timestamp) as last_activity,
                        MIN(timestamp) as first_activity,
                        AVG(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL) as avg_duration
                    FROM user_interactions
                    WHERE user_id = $1
                    """,
                    user_id
                )
                
                # Get activity by time period
                activity_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '7 days') as interactions_last_week,
                        COUNT(*) FILTER (WHERE timestamp >= NOW() - INTERVAL '30 days') as interactions_last_month,
                        COUNT(DISTINCT property_id) FILTER (WHERE timestamp >= NOW() - INTERVAL '7 days') as properties_last_week
                    FROM user_interactions
                    WHERE user_id = $1
                    """,
                    user_id
                )
                
                # Calculate engagement metrics
                engagement_rate = 0
                if stats['views'] > 0:
                    engagement_rate = (stats['likes'] + stats['inquiries'] + stats['saves']) / stats['views']
                
                return {
                    'total_interactions': stats['total_interactions'] or 0,
                    'unique_properties': stats['unique_properties'] or 0,
                    'views': stats['views'] or 0,
                    'likes': stats['likes'] or 0,
                    'inquiries': stats['inquiries'] or 0,
                    'saves': stats['saves'] or 0,
                    'last_activity': stats['last_activity'].isoformat() if stats['last_activity'] else None,
                    'first_activity': stats['first_activity'].isoformat() if stats['first_activity'] else None,
                    'avg_duration_seconds': float(stats['avg_duration']) if stats['avg_duration'] else None,
                    'interactions_last_week': activity_stats['interactions_last_week'] or 0,
                    'interactions_last_month': activity_stats['interactions_last_month'] or 0,
                    'properties_last_week': activity_stats['properties_last_week'] or 0,
                    'engagement_rate': engagement_rate,
                    'activity_level': self._calculate_activity_level(stats['total_interactions'] or 0)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics for user {user_id}: {e}")
            raise
    
    def _calculate_activity_level(self, total_interactions: int) -> str:
        """Calculate user activity level based on interaction count"""
        if total_interactions >= 100:
            return "high"
        elif total_interactions >= 20:
            return "medium"
        elif total_interactions >= 5:
            return "low"
        else:
            return "very_low"
    
    async def _get_user_interactions(self, conn, user_id: UUID, 
                                   interaction_type: Optional[str] = None,
                                   limit: int = 100, offset: int = 0) -> List[UserInteraction]:
        """Helper to get user interactions from database connection with pagination"""
        query = """
            SELECT property_id, interaction_type, timestamp, duration_seconds
            FROM user_interactions
            WHERE user_id = $1
        """
        params = [user_id]
        
        if interaction_type:
            query += " AND interaction_type = $2"
            params.append(interaction_type)
        
        query += f" ORDER BY timestamp DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.extend([limit, offset])
        
        rows = await conn.fetch(query, *params)
        
        interactions = []
        for row in rows:
            interaction = UserInteraction(
                property_id=row['property_id'],
                interaction_type=row['interaction_type'],
                timestamp=row['timestamp'],
                duration_seconds=row['duration_seconds']
            )
            interactions.append(interaction)
        
        return interactions
    
    async def _save_user_interactions(self, conn, user_id: UUID, 
                                    interactions: List[UserInteraction]):
        """Helper to save user interactions with deduplication"""
        if not interactions:
            return
        
        # Use INSERT ... ON CONFLICT to handle duplicates gracefully
        interaction_data = [
            (user_id, interaction.property_id, interaction.interaction_type,
             interaction.timestamp, interaction.duration_seconds)
            for interaction in interactions
        ]
        
        await conn.executemany(
            """
            INSERT INTO user_interactions (
                user_id, property_id, interaction_type, timestamp, duration_seconds
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id, property_id, interaction_type, timestamp)
            DO UPDATE SET duration_seconds = EXCLUDED.duration_seconds
            """,
            interaction_data
        )
    
    async def _get_users_by_ids(self, conn, user_ids: List[UUID]) -> List[User]:
        """Helper to efficiently fetch multiple users by their IDs"""
        if not user_ids:
            return []
        
        # Get user basic info for all users in one query
        user_rows = await conn.fetch(
            """
            SELECT id, email, created_at, status,
                   min_price, max_price, min_bedrooms, max_bedrooms,
                   min_bathrooms, max_bathrooms, preferred_locations,
                   required_amenities, property_types, updated_at
            FROM users 
            WHERE id = ANY($1) AND status = 'active'
            ORDER BY created_at DESC
            """,
            user_ids
        )
        
        if not user_rows:
            return []
        
        # Get all interactions for these users in one query
        user_interactions_map = {}
        if user_rows:
            all_interactions = await conn.fetch(
                """
                SELECT user_id, property_id, interaction_type, timestamp, duration_seconds
                FROM user_interactions
                WHERE user_id = ANY($1)
                ORDER BY user_id, timestamp DESC
                """,
                [row['id'] for row in user_rows]
            )
            
            # Group interactions by user_id
            for interaction_row in all_interactions:
                user_id = interaction_row['user_id']
                if user_id not in user_interactions_map:
                    user_interactions_map[user_id] = []
                
                interaction = UserInteraction(
                    property_id=interaction_row['property_id'],
                    interaction_type=interaction_row['interaction_type'],
                    timestamp=interaction_row['timestamp'],
                    duration_seconds=interaction_row['duration_seconds']
                )
                user_interactions_map[user_id].append(interaction)
        
        # Build user objects
        users = []
        for row in user_rows:
            preferences = UserPreferences(
                min_price=row['min_price'],
                max_price=row['max_price'],
                min_bedrooms=row['min_bedrooms'],
                max_bedrooms=row['max_bedrooms'],
                min_bathrooms=row['min_bathrooms'],
                max_bathrooms=row['max_bathrooms'],
                preferred_locations=row['preferred_locations'] or [],
                required_amenities=row['required_amenities'] or [],
                property_types=row['property_types'] or ["apartment"]
            )
            
            user = User(
                id=row['id'],
                email=row['email'],
                preferences=preferences,
                interactions=user_interactions_map.get(row['id'], []),
                created_at=row['created_at'],
                is_active=(row['status'] == 'active')
            )
            users.append(user)
        
        return users
    
    @retry_on_db_error()
    @measure_performance("bulk_update_user_preferences")
    async def bulk_update_user_preferences(self, user_updates: List[Dict[str, Any]]) -> int:
        """Bulk update user preferences for multiple users"""
        try:
            updated_count = 0
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for update in user_updates:
                        if 'user_id' not in update:
                            continue
                        
                        user_id = update['user_id']
                        update_data = {}
                        
                        # Build update data from provided fields
                        if 'min_price' in update:
                            update_data['min_price'] = update['min_price']
                        if 'max_price' in update:
                            update_data['max_price'] = update['max_price']
                        if 'min_bedrooms' in update:
                            update_data['min_bedrooms'] = update['min_bedrooms']
                        if 'max_bedrooms' in update:
                            update_data['max_bedrooms'] = update['max_bedrooms']
                        if 'preferred_locations' in update:
                            update_data['preferred_locations'] = update['preferred_locations']
                        if 'required_amenities' in update:
                            update_data['required_amenities'] = update['required_amenities']
                        
                        if update_data:
                            update_data['updated_at'] = datetime.utcnow()
                            
                            # Build dynamic query
                            set_clauses = []
                            params = [user_id]
                            param_counter = 2
                            
                            for field, value in update_data.items():
                                set_clauses.append(f"{field} = ${param_counter}")
                                params.append(value)
                                param_counter += 1
                            
                            query = f"""
                                UPDATE users 
                                SET {', '.join(set_clauses)}
                                WHERE id = $1 AND status = 'active'
                            """
                            
                            result = await conn.execute(query, *params)
                            if result == "UPDATE 1":
                                updated_count += 1
                    
                    self.logger.info(f"Bulk updated preferences for {updated_count} users")
                    return updated_count
                    
        except Exception as e:
            self.logger.error(f"Error bulk updating user preferences: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_behavior_analytics")
    async def get_user_behavior_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get aggregated user behavior analytics"""
        try:
            async with self.get_connection() as conn:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get overall user behavior metrics
                behavior_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(DISTINCT u.id) as active_users,
                        COUNT(DISTINCT ui.user_id) as interacting_users,
                        COUNT(*) as total_interactions,
                        COUNT(*) FILTER (WHERE ui.interaction_type = 'view') as total_views,
                        COUNT(*) FILTER (WHERE ui.interaction_type = 'like') as total_likes,
                        COUNT(*) FILTER (WHERE ui.interaction_type = 'inquiry') as total_inquiries,
                        COUNT(*) FILTER (WHERE ui.interaction_type = 'save') as total_saves,
                        AVG(ui.duration_seconds) FILTER (WHERE ui.duration_seconds IS NOT NULL) as avg_session_duration
                    FROM users u
                    LEFT JOIN user_interactions ui ON u.id = ui.user_id AND ui.timestamp >= $1
                    WHERE u.status = 'active'
                    """,
                    cutoff_date
                )
                
                # Get top interacting users
                top_users = await conn.fetch(
                    """
                    SELECT 
                        u.id,
                        u.email,
                        COUNT(*) as interaction_count,
                        COUNT(DISTINCT ui.property_id) as unique_properties
                    FROM users u
                    JOIN user_interactions ui ON u.id = ui.user_id
                    WHERE ui.timestamp >= $1 AND u.status = 'active'
                    GROUP BY u.id, u.email
                    ORDER BY interaction_count DESC
                    LIMIT 10
                    """,
                    cutoff_date
                )
                
                # Get interaction patterns by hour
                hourly_patterns = await conn.fetch(
                    """
                    SELECT 
                        EXTRACT(HOUR FROM timestamp) as hour,
                        COUNT(*) as interaction_count
                    FROM user_interactions
                    WHERE timestamp >= $1
                    GROUP BY EXTRACT(HOUR FROM timestamp)
                    ORDER BY hour
                    """,
                    cutoff_date
                )
                
                return {
                    'period_days': days,
                    'active_users': behavior_stats['active_users'] or 0,
                    'interacting_users': behavior_stats['interacting_users'] or 0,
                    'total_interactions': behavior_stats['total_interactions'] or 0,
                    'total_views': behavior_stats['total_views'] or 0,
                    'total_likes': behavior_stats['total_likes'] or 0,
                    'total_inquiries': behavior_stats['total_inquiries'] or 0,
                    'total_saves': behavior_stats['total_saves'] or 0,
                    'avg_session_duration': float(behavior_stats['avg_session_duration']) if behavior_stats['avg_session_duration'] else None,
                    'engagement_rate': (behavior_stats['interacting_users'] / max(1, behavior_stats['active_users'])) if behavior_stats['active_users'] else 0,
                    'top_users': [{
                        'user_id': str(row['id']),
                        'email': row['email'],
                        'interaction_count': row['interaction_count'],
                        'unique_properties': row['unique_properties']
                    } for row in top_users],
                    'hourly_patterns': [{
                        'hour': int(row['hour']),
                        'interaction_count': row['interaction_count']
                    } for row in hourly_patterns],
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get user behavior analytics: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_users_by_preference_location")
    async def get_users_by_preference_location(self, location: str, 
                                             limit: int = 50) -> List[User]:
        """Get users who prefer a specific location"""
        try:
            async with self.get_connection() as conn:
                user_ids = await conn.fetch(
                    """
                    SELECT id FROM users 
                    WHERE status = 'active' 
                      AND $1 = ANY(preferred_locations)
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    location,
                    limit
                )
                
                # Get user IDs and fetch them efficiently
                user_id_list = [row['id'] for row in user_ids]
                users = await self._get_users_by_ids(conn, user_id_list)
                
                return users
                
        except Exception as e:
            self.logger.error(f"Failed to get users by location {location}: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("cleanup_old_interactions")
    async def cleanup_old_interactions(self, days_threshold: int = 365) -> int:
        """Clean up old user interactions for performance"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Delete old interactions
                    result = await conn.execute(
                        """
                        DELETE FROM user_interactions 
                        WHERE timestamp < $1
                        """,
                        cutoff_date
                    )
                    
                    # Extract count from result
                    deleted_count = int(result.split()[1]) if result.startswith('DELETE') else 0
                    
                    self.logger.info(f"Cleaned up {deleted_count} old interactions")
                    return deleted_count
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old interactions: {e}")
            raise
    
    @retry_on_db_error()
    @measure_performance("get_user_segmentation")
    async def get_user_segmentation(self) -> Dict[str, Any]:
        """Get user segmentation data for analytics"""
        try:
            async with self.get_connection() as conn:
                # Get users by activity level
                activity_segments = await conn.fetch(
                    """
                    SELECT 
                        CASE 
                            WHEN interaction_count >= 100 THEN 'high_activity'
                            WHEN interaction_count >= 20 THEN 'medium_activity'
                            WHEN interaction_count >= 5 THEN 'low_activity'
                            ELSE 'minimal_activity'
                        END as activity_level,
                        COUNT(*) as user_count
                    FROM (
                        SELECT 
                            u.id,
                            COUNT(ui.id) as interaction_count
                        FROM users u
                        LEFT JOIN user_interactions ui ON u.id = ui.user_id
                        WHERE u.status = 'active'
                        GROUP BY u.id
                    ) user_activity
                    GROUP BY activity_level
                    ORDER BY user_count DESC
                    """
                )
                
                # Get users by preference completeness
                preference_segments = await conn.fetch(
                    """
                    SELECT 
                        CASE 
                            WHEN (min_price IS NOT NULL AND max_price IS NOT NULL AND 
                                  min_bedrooms IS NOT NULL AND max_bedrooms IS NOT NULL AND
                                  array_length(preferred_locations, 1) > 0) THEN 'complete_preferences'
                            WHEN (min_price IS NOT NULL OR max_price IS NOT NULL OR 
                                  min_bedrooms IS NOT NULL OR max_bedrooms IS NOT NULL) THEN 'partial_preferences'
                            ELSE 'minimal_preferences'
                        END as preference_level,
                        COUNT(*) as user_count
                    FROM users
                    WHERE status = 'active'
                    GROUP BY preference_level
                    ORDER BY user_count DESC
                    """
                )
                
                # Get users by registration time
                registration_segments = await conn.fetch(
                    """
                    SELECT 
                        CASE 
                            WHEN created_at >= NOW() - INTERVAL '7 days' THEN 'new_users'
                            WHEN created_at >= NOW() - INTERVAL '30 days' THEN 'recent_users'
                            WHEN created_at >= NOW() - INTERVAL '90 days' THEN 'regular_users'
                            ELSE 'veteran_users'
                        END as user_age_group,
                        COUNT(*) as user_count
                    FROM users
                    WHERE status = 'active'
                    GROUP BY user_age_group
                    ORDER BY user_count DESC
                    """
                )
                
                return {
                    'activity_segments': [{
                        'level': row['activity_level'],
                        'count': row['user_count']
                    } for row in activity_segments],
                    'preference_segments': [{
                        'level': row['preference_level'],
                        'count': row['user_count']
                    } for row in preference_segments],
                    'registration_segments': [{
                        'age_group': row['user_age_group'],
                        'count': row['user_count']
                    } for row in registration_segments],
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get user segmentation: {e}")
            raise
    
    async def close(self):
        """Close database connections and cleanup resources"""
        try:
            await self.pool.close()
            self.logger.info("Database connections closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
            raise